use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use fast_hnsw::{Builder, Hnsw};
use image::GenericImageView;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sift_rs::{KeyPoint, Normalization, SiftDescriptor, SiftDetector, SiftParams};

use crate::imgproc::feature_dimensions;
use crate::sfm::{ransac_fundamental, RANSAC_ITERATIONS, RANSAC_THRESHOLD_PX};
use crate::types::{ImageFeatures, DISTANCE_THRESHOLD, GUIDED_EPIPOLAR_THRESHOLD_PX, RANSAC_SEED};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedKeyPoint {
    x: f32,
    y: f32,
    scale: f32,
    octave: usize,
    first_octave: i32,
    orientation: f32,
    descriptor: Vec<f32>,
}

impl From<&KeyPoint> for CachedKeyPoint {
    fn from(kp: &KeyPoint) -> Self {
        CachedKeyPoint {
            x: kp.x,
            y: kp.y,
            scale: kp.scale,
            octave: kp.octave,
            first_octave: kp.first_octave,
            orientation: kp.orientation,
            descriptor: kp.descriptor.clone(),
        }
    }
}

impl From<CachedKeyPoint> for KeyPoint {
    fn from(ckp: CachedKeyPoint) -> Self {
        KeyPoint {
            x: ckp.x,
            y: ckp.y,
            scale: ckp.scale,
            octave: ckp.octave,
            first_octave: ckp.first_octave,
            orientation: ckp.orientation,
            descriptor: ckp.descriptor,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedImageFeatures {
    path: String,
    keypoints: Vec<CachedKeyPoint>,
}

impl From<&ImageFeatures> for CachedImageFeatures {
    fn from(f: &ImageFeatures) -> Self {
        CachedImageFeatures {
            path: f.path.clone(),
            keypoints: f.keypoints.iter().map(CachedKeyPoint::from).collect(),
        }
    }
}

impl From<CachedImageFeatures> for ImageFeatures {
    fn from(cf: CachedImageFeatures) -> Self {
        ImageFeatures {
            path: cf.path,
            keypoints: cf.keypoints.into_iter().map(KeyPoint::from).collect(),
        }
    }
}

/// 并行提取所有图像的 SIFT 特征点，带缓存机制。
///
/// 先检查 `features_cache/` 目录下是否已有缓存 JSON 文件，命中则直接读取；
/// 否则计算 SIFT 特征并写回缓存，供后续重复运行加速。
///
/// # Arguments
///
/// * `paths` - 图像文件路径列表
/// * `first_octave` - SIFT 首层 octave 参数
///
/// # Examples
///
/// ```rust,ignore
/// let paths = load_image_paths("./images");
/// let features = extract_features(&paths, 0);
/// ```
pub(crate) fn extract_features(paths: &[String], first_octave: i32) -> Vec<ImageFeatures> {
    let cache_dir = "features_cache";
    fs::create_dir_all(cache_dir).expect("Failed to create features_cache directory");

    println!(
        "Extracting SIFT features from {} images (parallel)...",
        paths.len()
    );
    let start = Instant::now();

    let atomic_cached = std::sync::atomic::AtomicU32::new(0);
    let atomic_computed = std::sync::atomic::AtomicU32::new(0);

    let mut results: Vec<ImageFeatures> = paths
        .par_iter()
        .map(|path| {
            let stem = std::path::Path::new(path)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();
            let cache_path = format!("{}/{}.json", cache_dir, stem);

            if let Ok(data) = fs::read_to_string(&cache_path) {
                if let Ok(cached) = serde_json::from_str::<CachedImageFeatures>(&data) {
                    if cached.path == *path {
                        atomic_cached.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return ImageFeatures::from(cached);
                    }
                }
            }

            let img = image::open(path).unwrap_or_else(|_| panic!("Failed to open {}", path));
            let mut gray = img.grayscale();
            let (w, h) = gray.dimensions();
            let (nw, nh) = feature_dimensions(w, h);
            if nw != w || nh != h {
                gray = gray.resize(nw, nh, image::imageops::FilterType::Triangle);
            }
            let mut detector = SiftDetector::new();
            let mut params = SiftParams {
                max_num_features: 8192,
                first_octave,
                num_octaves: 4,
                edge_threshold: 10.0,
                ..SiftParams::default()
            };
            params.peak_threshold = 0.02 / params.octave_resolution as f32;
            let mut keypoints = detector.detect(gray, &params);
            let descriptor = SiftDescriptor::new();
            descriptor.compute_descriptors(
                &mut keypoints,
                detector.scale_spaces,
                Normalization::L1,
            );

            let kps = keypoints;

            let result = ImageFeatures {
                path: path.clone(),
                keypoints: kps,
            };

            let cached: CachedImageFeatures = (&result).into();
            let json = serde_json::to_string(&cached).expect("Failed to serialize features");
            fs::write(&cache_path, json).expect("Failed to write cache file");

            atomic_computed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            result
        })
        .collect();

    let cached_count = atomic_cached.load(std::sync::atomic::Ordering::Relaxed);
    let computed_count = atomic_computed.load(std::sync::atomic::Ordering::Relaxed);

    results.sort_by(|a, b| a.path.cmp(&b.path));

    let elapsed = start.elapsed();
    println!(
        "Feature extraction completed in {:.2}s ({} cached, {} computed)",
        elapsed.as_secs_f32(),
        cached_count,
        computed_count
    );
    let total: usize = results.iter().map(|f| f.keypoints.len()).sum();
    println!("  Total keypoints: {}\n", total);

    for f in &results {
        println!(
            "  {}: {} keypoints",
            std::path::Path::new(&f.path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            f.keypoints.len()
        );
    }

    results
}

pub(crate) struct Match {
    pub(crate) img1: usize,
    pub(crate) kp1: usize,
    pub(crate) img2: usize,
    pub(crate) kp2: usize,
}

#[derive(Serialize, Deserialize)]
struct CachedPairMatches {
    img1: usize,
    img2: usize,
    matches: Vec<(usize, usize)>,
}

/// 计算两个 128 维 SIFT 描述子向量之间的欧氏距离。
///
/// 描述子距离越小表示两个特征点越相似，是特征匹配中
/// 判断相似度的基础度量。
///
/// # Arguments
///
/// * `a` - 第一个描述子向量
/// * `b` - 第二个描述子向量
///
/// # Examples
///
/// ```rust,ignore
/// let a = vec![0.0f32; 128];
/// let b = vec![1.0f32; 128];
/// let d = euclidean_distance(&a, &b);
/// assert!(d > 0.0);
/// ```
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// SIMD 友好的欧氏距离度量（零大小标记类型）。
///
/// 通过 `fast_hnsw::distance::Distance` 特征接入 HNSW 索引，
/// 其 [`distance`](SimdEuclidean::distance) 实现采用 4 路展开
/// 累加，配合 `target-cpu=native` 时可被编译器自动向量化。
struct SimdEuclidean;

impl fast_hnsw::distance::Distance for SimdEuclidean {
    /// 计算两个浮点切片之间的欧氏距离。
    ///
    /// 将累加过程拆分为 4 路独立求和以打破串行依赖，
    /// 相比朴素循环具有更好的指令级并行与向量化潜力。
    ///
    /// # Arguments
    ///
    /// * `a` - 第一个向量切片
    /// * `b` - 第二个向量切片
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let metric = SimdEuclidean;
    /// let a = [0.0f32; 128];
    /// let b = [1.0f32; 128];
    /// let d = metric.distance(&a, &b);
    /// assert!(d > 0.0);
    /// ```
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut i = 0;
        while i + 4 <= n {
            let d0 = a[i] - b[i];
            let d1 = a[i + 1] - b[i + 1];
            let d2 = a[i + 2] - b[i + 2];
            let d3 = a[i + 3] - b[i + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
            i += 4;
        }
        while i < n {
            let d = a[i] - b[i];
            s0 += d * d;
            i += 1;
        }
        (s0 + s1 + s2 + s3).sqrt()
    }
}

/// 为一组关键点的描述子构建 HNSW 近似最近邻索引。
///
/// 基于 `fast_hnsw` 构建索引，等价于 FLANN 的近似最近邻搜索，
/// 供后续 2-NN 匹配时快速查询每个描述子的近邻。每个关键点的
/// `descriptor` 字段被作为索引向量插入。
///
/// # Arguments
///
/// * `kp` - 关键点切片，取其描述子作为索引向量
///
/// # Examples
///
/// ```rust,ignore
/// let kp: Vec<KeyPoint> = detect_features(&image);
/// let index = build_hnsw_index(&kp);
/// ```
fn build_hnsw_index(kp: &[KeyPoint]) -> Hnsw<SimdEuclidean> {
    let mut index: Hnsw<SimdEuclidean> = Builder::new()
        .m(16)
        .ef_construction(200)
        .capacity(kp.len())
        .seed(RANSAC_SEED)
        .build(SimdEuclidean);
    for k in kp {
        index.insert(k.descriptor.clone());
    }
    index
}

/// 对 HNSW 索引执行 kNN=2 查询，按 Lowe 比率检验 + 绝对距离阈值筛选匹配。
///
/// 对每个查询关键点取最近与次近两个邻居，仅当最近距离满足
/// `best < distance_threshold` 且 `best < ratio_threshold * second`
/// 时保留该匹配。
///
/// # Arguments
///
/// * `index` - 目标图像的 HNSW 索引
/// * `kp_a` - 查询侧关键点
/// * `ratio_threshold` - Lowe 比率阈值
/// * `distance_threshold` - 描述子绝对距离阈值
///
/// # Examples
///
/// ```rust,ignore
/// let index = build_hnsw_index(&kp_b);
/// let matches = query_2nn(&index, &kp_a, 0.8, 0.7);
/// ```
fn query_2nn(
    index: &Hnsw<SimdEuclidean>,
    kp_a: &[KeyPoint],
    ratio_threshold: f32,
    distance_threshold: f32,
) -> Vec<(usize, usize)> {
    kp_a.par_iter()
        .enumerate()
        .filter_map(|(i, k1)| {
            let results = index.search(&k1.descriptor, 2, 32);
            if results.len() < 2 {
                return None;
            }
            let best = results[0].distance;
            let second = results[1].distance;
            if best < distance_threshold && best < ratio_threshold * second {
                Some((i, results[0].id))
            } else {
                None
            }
        })
        .collect()
}

/// 引导匹配：利用基础矩阵 F 在未匹配关键点中按极线距离搜索几何一致的新对应点。
///
/// 对每个未使用的查询关键点沿其极线带搜索候选，再以描述子
/// 2-NN 比率检验 + 距离阈值筛选，用于在已有内点基础上增补匹配。
/// 使用空间网格分桶沿极线带采样，避免 O(n²) 全扫描。
///
/// # Arguments
///
/// * `f` - 已估计的基础矩阵
/// * `kp_a` - 图像 A 的关键点
/// * `kp_b` - 图像 B 的关键点
/// * `used_a` - 图像 A 中已匹配的关键点索引
/// * `used_b` - 图像 B 中已匹配的关键点索引
/// * `epi_threshold_px` - 极线距离阈值（像素）
/// * `desc_threshold` - 描述子距离阈值
/// * `ratio_threshold` - Lowe 比率阈值
///
/// # Examples
///
/// ```rust,ignore
/// let extra = guided_matching(&f, &kp_a, &kp_b, &used_a, &used_b, 4.0, 0.7, 0.8);
/// ```
fn guided_matching(
    f: &Matrix3<f64>,
    kp_a: &[KeyPoint],
    kp_b: &[KeyPoint],
    used_a: &std::collections::HashSet<usize>,
    used_b: &std::collections::HashSet<usize>,
    epi_threshold_px: f64,
    desc_threshold: f32,
    ratio_threshold: f32,
) -> Vec<(usize, usize)> {
    if kp_b.is_empty() {
        return Vec::new();
    }

    let cell = epi_threshold_px.max(8.0);

    // kp_b 的空间包围盒
    let mut xmin = f64::MAX;
    let mut xmax = f64::MIN;
    let mut ymin = f64::MAX;
    let mut ymax = f64::MIN;
    for k2 in kp_b {
        xmin = xmin.min(k2.x as f64);
        xmax = xmax.max(k2.x as f64);
        ymin = ymin.min(k2.y as f64);
        ymax = ymax.max(k2.y as f64);
    }

    // 未被匹配的 B 关键点按网格分桶
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (j, k2) in kp_b.iter().enumerate() {
        if used_b.contains(&j) {
            continue;
        }
        grid.entry(((k2.x as f64 / cell) as i64, (k2.y as f64 / cell) as i64))
            .or_default()
            .push(j);
    }

    let candidates: Vec<(usize, usize, f32)> = kp_a
        .par_iter()
        .enumerate()
        .filter(|(i, _)| !used_a.contains(i))
        .flat_map(|(i, k1)| {
            let p1 = Vector3::new(k1.x as f64, k1.y as f64, 1.0);
            let l = f * p1;
            let (lx, ly, lz) = (l.x, l.y, l.z);
            let denom = (lx * lx + ly * ly).sqrt();
            if denom < 1e-12 {
                return Vec::new();
            }

            let mut visited: std::collections::HashSet<(i64, i64)> =
                std::collections::HashSet::new();
            let mut best: Option<(f32, usize)> = None;
            let mut second: Option<f32> = None;

            // 沿极线带逐步采样网格单元，记录最近/次近描述子距离
            let consider = |j: usize, best: &mut Option<(f32, usize)>, second: &mut Option<f32>| {
                let k2 = &kp_b[j];
                let d = (lx * k2.x as f64 + ly * k2.y as f64 + lz).abs() / denom;
                if d <= epi_threshold_px {
                    let dist = euclidean_distance(&k1.descriptor, &k2.descriptor);
                    if dist < desc_threshold {
                        match *best {
                            None => *best = Some((dist, j)),
                            Some((bd, _)) if dist < bd => {
                                *second = Some(bd);
                                *best = Some((dist, j));
                            }
                            Some(_) => {
                                let update = match *second {
                                    Some(s) => dist < s,
                                    None => true,
                                };
                                if update {
                                    *second = Some(dist);
                                }
                            }
                        }
                    }
                }
            };

            let probe = |cx: i64,
                         cy: i64,
                         visited: &mut std::collections::HashSet<(i64, i64)>,
                         best: &mut Option<(f32, usize)>,
                         second: &mut Option<f32>| {
                for dcx in -1i64..=1 {
                    for dcy in -1i64..=1 {
                        let key = (cx + dcx, cy + dcy);
                        if visited.insert(key) {
                            if let Some(pts) = grid.get(&key) {
                                for &j in pts {
                                    consider(j, best, second);
                                }
                            }
                        }
                    }
                }
            };

            if ly.abs() >= lx.abs() {
                let mut x = xmin;
                while x <= xmax {
                    let y = -(lx * x + lz) / ly;
                    let cx = (x / cell) as i64;
                    let cy = (y / cell) as i64;
                    probe(cx, cy, &mut visited, &mut best, &mut second);
                    x += cell;
                }
            } else {
                let mut y = ymin;
                while y <= ymax {
                    let x = -(ly * y + lz) / lx;
                    let cx = (x / cell) as i64;
                    let cy = (y / cell) as i64;
                    probe(cx, cy, &mut visited, &mut best, &mut second);
                    y += cell;
                }
            }

            match (best, second) {
                (Some((dist, j)), Some(s)) if dist < ratio_threshold * s => vec![(i, j, dist)],
                (Some((dist, j)), None) => vec![(i, j, dist)],
                _ => Vec::new(),
            }
        })
        .collect();

    let mut sorted = candidates;
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut used_a2 = used_a.clone();
    let mut used_b2 = used_b.clone();
    let mut out = Vec::new();
    for (a, b, _) in sorted {
        if !used_a2.contains(&a) && !used_b2.contains(&b) {
            used_a2.insert(a);
            used_b2.insert(b);
            out.push((a, b));
        }
    }
    out
}

/// 匹配所有图像对：交叉一致性检查 + RANSAC 基础矩阵极线约束过滤。
///
/// 对每对图像执行双向 2-NN 匹配并取交集（交叉一致性），再用
/// RANSAC 估计基础矩阵过滤几何不一致的匹配，对可靠图像对进一步
/// 执行引导匹配增补内点。结果带缓存（`matches_cache/`）。
///
/// # Arguments
///
/// * `features` - 所有图像的特征
/// * `ratio_threshold` - Lowe 比例阈值
///
/// # Examples
///
/// ```rust,ignore
/// let matches = match_all_pairs(&features, 0.8);
/// ```
pub(crate) fn match_all_pairs(features: &[ImageFeatures], ratio_threshold: f32) -> Vec<Match> {
    let num_images = features.len();

    let pairs: Vec<(usize, usize)> = (0..num_images)
        .flat_map(|i| ((i + 1)..num_images).map(move |j| (i, j)))
        .collect();

    let cache_dir = "matches_cache";
    fs::create_dir_all(cache_dir).expect("Failed to create matches_cache directory");

    println!(
        "Matching all image pairs ({} pairs, cross-consistency + RANSAC F-matrix)...",
        pairs.len()
    );
    let start = Instant::now();
    let ransac_total = std::sync::atomic::AtomicU32::new(0);
    let cache_hit = std::sync::atomic::AtomicU32::new(0);

    // 每张图像只构建一次 HNSW 索引，跨所有图像对复用
    let indices: Vec<Hnsw<SimdEuclidean>> = features
        .par_iter()
        .map(|f| build_hnsw_index(&f.keypoints))
        .collect();

    let all_matches: Vec<Match> = pairs
        .par_iter()
        .flat_map(|&(i, j)| {
            let cache_path = format!("{}/{}_{}.json", cache_dir, i, j);

            if let Ok(data) = fs::read_to_string(&cache_path) {
                if let Ok(cached) = serde_json::from_str::<CachedPairMatches>(&data) {
                    if cached.img1 == i && cached.img2 == j {
                        cache_hit.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return cached
                            .matches
                            .into_iter()
                            .map(|(kp1, kp2)| Match {
                                img1: i,
                                kp1,
                                img2: j,
                                kp2,
                            })
                            .collect::<Vec<_>>();
                    }
                }
            }

            let matches_ij = query_2nn(
                &indices[j],
                &features[i].keypoints,
                ratio_threshold,
                DISTANCE_THRESHOLD,
            );
            let matches_ji = query_2nn(
                &indices[i],
                &features[j].keypoints,
                ratio_threshold,
                DISTANCE_THRESHOLD,
            );

            let mut reverse_map: HashMap<usize, usize> = HashMap::with_capacity(matches_ji.len());
            for &(kp_j_idx, kp_i_idx) in &matches_ji {
                reverse_map.insert(kp_j_idx, kp_i_idx);
            }

            let cross_matches: Vec<(usize, usize)> = matches_ij
                .into_iter()
                .filter(|(kp_i, kp_j)| reverse_map.get(kp_j) == Some(kp_i))
                .collect();

            if cross_matches.len() < 8 {
                let empty = CachedPairMatches {
                    img1: i,
                    img2: j,
                    matches: Vec::new(),
                };
                if let Ok(json) = serde_json::to_string(&empty) {
                    let _ = fs::write(&cache_path, json);
                }
                return Vec::new();
            }

            let pts_i: Vec<Vector3<f64>> = cross_matches
                .iter()
                .map(|&(kpi, _)| {
                    let kp = &features[i].keypoints[kpi];
                    Vector3::new(kp.x as f64, kp.y as f64, 1.0)
                })
                .collect();
            let pts_j: Vec<Vector3<f64>> = cross_matches
                .iter()
                .map(|&(_, kpj)| {
                    let kp = &features[j].keypoints[kpj];
                    Vector3::new(kp.x as f64, kp.y as f64, 1.0)
                })
                .collect();

            let sampson_threshold = RANSAC_THRESHOLD_PX * RANSAC_THRESHOLD_PX;
            let (f_mat, inliers) =
                ransac_fundamental(&pts_i, &pts_j, RANSAC_ITERATIONS, sampson_threshold);

            ransac_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let inlier_set: std::collections::HashSet<usize> = inliers.into_iter().collect();

            let mut pair_matches: Vec<Match> = cross_matches
                .iter()
                .enumerate()
                .filter_map(|(idx, &(kp_i, kp_j))| {
                    if inlier_set.contains(&idx) {
                        Some(Match {
                            img1: i,
                            kp1: kp_i,
                            img2: j,
                            kp2: kp_j,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            // 只有当基础矩阵可靠（足够多内点）时才做引导匹配，避免在
            // 无重叠图像对上用退化 F 增补大量虚假匹配
            if pair_matches.len() >= 30 {
                let used_a: std::collections::HashSet<usize> =
                    pair_matches.iter().map(|m| m.kp1).collect();
                let used_b: std::collections::HashSet<usize> =
                    pair_matches.iter().map(|m| m.kp2).collect();
                let extra = guided_matching(
                    &f_mat,
                    &features[i].keypoints,
                    &features[j].keypoints,
                    &used_a,
                    &used_b,
                    GUIDED_EPIPOLAR_THRESHOLD_PX,
                    DISTANCE_THRESHOLD,
                    ratio_threshold,
                );
                for (a, b) in extra {
                    pair_matches.push(Match {
                        img1: i,
                        kp1: a,
                        img2: j,
                        kp2: b,
                    });
                }
            }

            let cached = CachedPairMatches {
                img1: i,
                img2: j,
                matches: pair_matches.iter().map(|m| (m.kp1, m.kp2)).collect(),
            };
            let json = serde_json::to_string(&cached).expect("Failed to serialize pair matches");
            fs::write(&cache_path, json).expect("Failed to write match cache file");

            pair_matches
        })
        .collect();

    let hits = cache_hit.load(std::sync::atomic::Ordering::Relaxed);
    let elapsed = start.elapsed();
    println!(
        "Matching completed in {:.2}s ({} cached, {} computed via RANSAC)",
        elapsed.as_secs_f32(),
        hits,
        ransac_total.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "Found {} cross-consistent + epipolar matches\n",
        all_matches.len()
    );

    all_matches
}
