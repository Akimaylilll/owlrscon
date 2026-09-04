use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

use fast_hnsw::{Builder, Hnsw};
use image::{GenericImage, Rgb, RgbImage};
use imageproc::drawing;
use nalgebra::{self as na, Matrix3, Matrix3x4, Vector3, SVD};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sift_rs::{KeyPoint, Normalization, SiftDescriptor, SiftDetector, SiftParams};

const RATIO_THRESHOLD: f32 = 0.8;
const DISTANCE_THRESHOLD: f32 = 0.7;
const GUIDED_EPIPOLAR_THRESHOLD_PX: f64 = 4.0;
const MAX_KEYPOINTS_PER_IMAGE: usize = 5000;
const SLIDING_WINDOW: usize = 10;
const MIN_TRACK_LENGTH: usize = 2;
const RANSAC_SEED: u64 = 42;

struct ImageFeatures {
    path: String,
    keypoints: Vec<KeyPoint>,
}

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

struct Match {
    img1: usize,
    kp1: usize,
    img2: usize,
    kp2: usize,
}

#[derive(Serialize, Deserialize)]
struct CachedPairMatches {
    img1: usize,
    img2: usize,
    matches: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize)]
struct ObservationDetail {
    image_idx: usize,
    image_name: String,
    kp_idx: usize,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
}

#[derive(Serialize)]
struct TrackExportEntry {
    track_id: usize,
    length: usize,
    observations: Vec<ObservationDetail>,
}

#[derive(Serialize)]
struct TrackExport {
    num_images: usize,
    num_tracks: usize,
    num_raw_matches: usize,
    parameters: TrackParameters,
    image_paths: Vec<String>,
    keypoints_per_image: Vec<usize>,
    tracks: Vec<TrackExportEntry>,
}

#[derive(Serialize)]
struct TrackParameters {
    ratio_threshold: f32,
    max_keypoints_per_image: usize,
    sliding_window: usize,
    min_track_length: usize,
}

struct Track {
    observations: Vec<(usize, usize)>,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            self.parent[root] = self.parent[self.parent[root]];
            root = self.parent[root];
        }
        root
    }

    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px != py {
            match self.rank[px].cmp(&self.rank[py]) {
                std::cmp::Ordering::Less => self.parent[px] = py,
                std::cmp::Ordering::Greater => self.parent[py] = px,
                std::cmp::Ordering::Equal => {
                    self.parent[py] = px;
                    self.rank[px] += 1;
                }
            }
        }
    }
}

/// 从目录加载所有支持的图像文件路径，按文件名排序
/// @param dir  图像目录路径，如"./images"
/// @return     排序后的图像文件绝对路径列表
fn load_image_paths(dir: &str) -> Vec<String> {
    let mut paths: Vec<String> = fs::read_dir(dir)
        .expect("Failed to read images directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let ext = path.extension()?.to_str()?.to_lowercase();
            if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
                Some(path.to_str()?.to_owned())
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths
}

/// 并行提取所有图像的SIFT特征点，带缓存机制
/// 先检查 features_cache/ 目录下是否已有缓存的JSON文件，若有则直接读取；
/// 若无则计算特征并保存到缓存文件中。
/// @param paths  图像文件路径列表
/// @return      每幅图的SIFT关键点（含128维描述子）
fn extract_features(paths: &[String]) -> Vec<ImageFeatures> {
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
            let gray = img.grayscale();
            let mut detector = SiftDetector::new();
            let mut params = SiftParams {
                max_num_features: 8192,
                first_octave: -1,
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

/// 计算两个128维SIFT描述子向量之间的欧氏距离
/// 计算两个128维SIFT描述子向量之间的欧氏距离
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// SIMD 友好的欧氏距离（4 路展开打破串行依赖，配合 target-cpu=native 自动向量化）
struct SimdEuclidean;

impl fast_hnsw::distance::Distance for SimdEuclidean {
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

/// 为一组关键点描述子构建 HNSW 索引（FLANN 等价近似最近邻）
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

/// 2-NN 匹配：对已构建的 HNSW 索引查询 kNN=2，
/// Lowe 比率检验(0.8) + 绝对距离阈值(0.7)
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

/// 引导匹配：利用已估计的基础矩阵 F，在未被匹配的关键点中按极线距离
/// 搜索几何一致的新对应点，再以描述子 2-NN 比率检验 + 距离阈值筛选，
/// 用于增补内点。用空间网格沿极线带搜索，避免 O(n²) 全扫描。
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

/// 匹配所有图像对：交叉一致性检查 + RANSAC基础矩阵极线约束过滤
/// @param features        所有图像特征
/// @param ratio_threshold  Lowe比例阈值
fn match_all_pairs(features: &[ImageFeatures], ratio_threshold: f32) -> Vec<Match> {
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

/// 并查集Union-Find将匹配对连成Track：全局ID→连通分量→去重
fn build_tracks(matches: &[Match], features: &[ImageFeatures]) -> Vec<Track> {
    println!("Building feature tracks...");
    let start = Instant::now();

    let total_kps: usize = features.iter().map(|f| f.keypoints.len()).sum();
    let mut offsets = vec![0usize; features.len() + 1];
    for i in 0..features.len() {
        offsets[i + 1] = offsets[i] + features[i].keypoints.len();
    }

    let to_global = |img: usize, kp: usize| -> usize { offsets[img] + kp };

    let mut uf = UnionFind::new(total_kps);
    for m in matches {
        uf.union(to_global(m.img1, m.kp1), to_global(m.img2, m.kp2));
    }

    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for gid in 0..total_kps {
        let root = uf.find(gid);
        components.entry(root).or_default().push(gid);
    }

    let mut tracks = Vec::new();
    let mut rejected_duplicates = 0usize;
    let mut rejected_singles = 0usize;

    for (_root, comp) in components {
        if comp.len() < MIN_TRACK_LENGTH {
            rejected_singles += 1;
            continue;
        }

        let mut observations: Vec<(usize, usize)> = comp
            .iter()
            .map(|&gid| {
                let img = offsets.partition_point(|&o| o <= gid) - 1;
                let kp = gid - offsets[img];
                (img, kp)
            })
            .collect();
        observations.sort_unstable_by_key(|&(img, _)| img);

        let has_duplicate = observations.windows(2).any(|w| w[0].0 == w[1].0);
        if has_duplicate {
            rejected_duplicates += 1;
            continue;
        }

        tracks.push(Track { observations });
    }

    let elapsed = start.elapsed();
    println!("Track building completed in {:.2}s", elapsed.as_secs_f32());
    println!(
        "  Rejected: {} single-feature components, {} tracks with duplicate image observations",
        rejected_singles, rejected_duplicates
    );

    tracks
}

/// 打印轨迹统计：长度分布和最长15条逐点坐标
fn print_track_details(tracks: &[Track], features: &[ImageFeatures], paths: &[String]) {
    let track_lengths: Vec<usize> = tracks.iter().map(|t| t.observations.len()).collect();
    if track_lengths.is_empty() {
        println!("No valid tracks found.");
        return;
    }

    let max_len = track_lengths.iter().max().unwrap();
    let min_len = track_lengths.iter().min().unwrap();
    let avg_len = track_lengths.iter().sum::<usize>() as f32 / track_lengths.len() as f32;

    println!("\n============ Track Statistics ============");
    println!("  Number of tracks:      {}", tracks.len());
    println!(
        "  Total observations:    {}",
        track_lengths.iter().sum::<usize>()
    );
    println!("  Min track length:      {}", min_len);
    println!("  Max track length:      {}", max_len);
    println!("  Average track length:  {:.2}", avg_len);

    let mut dist: HashMap<usize, usize> = HashMap::new();
    for len in &track_lengths {
        *dist.entry(*len).or_default() += 1;
    }
    println!("\n  Track length distribution:");
    let mut keys: Vec<_> = dist.keys().copied().collect();
    keys.sort_unstable();
    for len in keys {
        let count = dist[&len];
        let bar = "█".repeat(((count as f32 / track_lengths.len() as f32) * 50.0).ceil() as usize);
        println!("    {:2} images: {:4} tracks  {}", len, count, bar);
    }

    println!("\n══════════════════════════════════════════");
    println!("  Detailed Track Content (top 15 by length)");
    println!("══════════════════════════════════════════");

    let mut indexed_tracks: Vec<(usize, &Track)> = tracks.iter().enumerate().collect();
    indexed_tracks.sort_by_key(|(_, t)| std::cmp::Reverse(t.observations.len()));

    let file_stem = |path: &str| -> String {
        std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_owned())
            .unwrap_or_else(|| path.to_owned())
    };

    for (_rank, (track_idx, track)) in indexed_tracks.iter().take(15).enumerate() {
        let obs = &track.observations;
        println!(
            "\n  ┌─ Track #{:<4}  length: {}  images: [{}, {}]",
            track_idx,
            obs.len(),
            obs.first().map(|o| o.0).unwrap_or(0),
            obs.last().map(|o| o.0).unwrap_or(0),
        );
        println!(
            "  │ {:>6}  {:>12} {:>8} {:>8} {:>8} {:>8}",
            "Image", "Name", "kp_idx", "x", "y", "scale"
        );
        println!("  ├{}", "─".repeat(55));

        for (img_idx, kp_idx) in obs {
            let kp = &features[*img_idx].keypoints[*kp_idx];
            println!(
                "  │ {:>5}  {:>12} {:>8} {:>8.1} {:>8.1} {:>8.2}",
                img_idx,
                file_stem(&paths[*img_idx]),
                kp_idx,
                kp.x,
                kp.y,
                kp.scale,
            );
        }
        println!("  └{}", "─".repeat(55));
    }

    println!(
        "\n  ... and {} more tracks (use `--full` or check tracks.json for all)\n",
        tracks.len().saturating_sub(15)
    );
}

/// 将轨迹序列化为JSON文件
fn save_tracks_to_json(
    tracks: &[Track],
    features: &[ImageFeatures],
    paths: &[String],
    num_matches: usize,
    output_path: &str,
) {
    println!("Saving track data to {} ...", output_path);

    let file_stem = |path: &str| -> String {
        std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_owned())
            .unwrap_or_else(|| path.to_owned())
    };

    let export_entries: Vec<TrackExportEntry> = tracks
        .iter()
        .enumerate()
        .map(|(track_id, track)| {
            let observations: Vec<ObservationDetail> = track
                .observations
                .iter()
                .map(|&(img_idx, kp_idx)| {
                    let kp = &features[img_idx].keypoints[kp_idx];
                    ObservationDetail {
                        image_idx: img_idx,
                        image_name: file_stem(&paths[img_idx]),
                        kp_idx,
                        x: kp.x,
                        y: kp.y,
                        scale: kp.scale,
                        orientation: kp.orientation,
                    }
                })
                .collect();
            TrackExportEntry {
                track_id,
                length: track.observations.len(),
                observations,
            }
        })
        .collect();

    let export = TrackExport {
        num_images: paths.len(),
        num_tracks: tracks.len(),
        num_raw_matches: num_matches,
        parameters: TrackParameters {
            ratio_threshold: RATIO_THRESHOLD,
            max_keypoints_per_image: MAX_KEYPOINTS_PER_IMAGE,
            sliding_window: SLIDING_WINDOW,
            min_track_length: MIN_TRACK_LENGTH,
        },
        image_paths: paths.iter().map(|p| file_stem(p)).collect(),
        keypoints_per_image: features.iter().map(|f| f.keypoints.len()).collect(),
        tracks: export_entries,
    };

    let file = File::create(output_path).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &export).expect("Failed to serialize tracks");
    writer.flush().expect("Failed to flush output");

    let metadata = fs::metadata(output_path).expect("Failed to read output file metadata");
    println!(
        "Saved {} tracks to {} ({:.1} KB)",
        tracks.len(),
        output_path,
        metadata.len() as f64 / 1024.0
    );
}

fn generate_colors(n: usize) -> Vec<Rgb<u8>> {
    let golden_ratio = 0.618033988749895;
    (0..n)
        .map(|i| {
            let hue = (i as f32 * golden_ratio) % 1.0;
            /// 黄金比例法生成N个色环均匀分布的RGB颜色
            let s = 0.7;
            let v = 0.9;
            let c = v * s;
            let x = c * (1.0 - ((hue * 6.0) % 2.0 - 1.0).abs());
            let m = v - c;
            let (r, g, b) = match (hue * 6.0) as u32 % 6 {
                0 => (c, x, 0.0),
                1 => (x, c, 0.0),
                2 => (0.0, c, x),
                3 => (0.0, x, c),
                4 => (x, 0.0, c),
                _ => (c, 0.0, x),
            };
            Rgb([
                ((r + m) * 255.0) as u8,
                ((g + m) * 255.0) as u8,
                ((b + m) * 255.0) as u8,
            ])
        })
        .collect()
}

/// 热力图颜色映射
fn heat_color(t: f32) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let r = if t < 0.5 {
        (t * 2.0 * 255.0) as u8
    } else {
        255
    };
    let g = if t < 0.5 {
        100
    } else {
        ((1.0 - (t - 0.5) * 2.0) * 155.0 + 100.0) as u8
    };
    let b = ((1.0 - t) * 255.0) as u8;
    Rgb([r, g, b])
}

/// SfM轨道覆盖图：图像环形排列为节点
fn visualize_track_coverage_graph(
    tracks: &[Track],
    features: &[ImageFeatures],
    paths: &[String],
    output_path: &str,
) {
    println!("Generating track coverage graph...");
    let start = Instant::now();

    let num_images = paths.len();
    let canvas_size: u32 = 2400;
    let margin: u32 = 280;
    let inner_radius = (canvas_size as f32 / 2.0 - margin as f32 * 0.2) as f32;
    let center = canvas_size as f32 / 2.0;

    let start_angle = -std::f32::consts::PI / 2.0;
    let total_arc = 2.0 * std::f32::consts::PI * 0.92;

    let positions: Vec<(f32, f32)> = (0..num_images)
        .map(|i| {
            let t = i as f32 / (num_images - 1).max(1) as f32;
            let angle = start_angle + t * total_arc;
            (
                center + inner_radius * angle.cos(),
                center + inner_radius * angle.sin(),
            )
        })
        .collect();

    let _node_radius = 16i32;
    let line_thickness = 1.5f32;

    let mut pair_counts = vec![vec![0usize; num_images]; num_images];

    for track in tracks {
        for w in track.observations.windows(2) {
            let (a, _) = w[0];
            let (b, _) = w[1];
            pair_counts[a][b] += 1;
            pair_counts[b][a] += 1;
        }
    }

    let mut canvas = RgbImage::from_pixel(canvas_size, canvas_size, Rgb([10, 10, 16]));
    let max_count = pair_counts
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as f32;

    for i in 0..num_images {
        for j in (i + 1)..num_images {
            let count = pair_counts[i][j];
            if count == 0 {
                continue;
            }
            let intensity = count as f32 / max_count;
            let color = heat_color(intensity * 0.8);

            let p1 = positions[i];
            let p2 = positions[j];
            let thick = (1.0 + intensity * 5.0 * line_thickness) as f32;
            draw_thick_line(&mut canvas, p1, p2, color, thick);
        }
    }

    for (i, &(x, y)) in positions.iter().enumerate() {
        let ix = x.round() as i32;
        let iy = y.round() as i32;

        let total_kp = features[i].keypoints.len();
        let node_r = (10.0 + (total_kp as f32 / 5000.0) * 16.0) as i32;

        drawing::draw_filled_circle_mut(&mut canvas, (ix, iy), node_r, Rgb([40, 80, 180]));

        drawing::draw_filled_circle_mut(
            &mut canvas,
            (ix, iy),
            node_r.max(2) - 2,
            Rgb([60, 140, 240]),
        );

        let label_angle = (y - center).atan2(x - center);
        let lx = ix + ((label_angle.cos() * (node_r + 22) as f32) as i32);
        let ly = iy + ((label_angle.sin() * (node_r + 22) as f32) as i32);

        let name = std::path::Path::new(&paths[i])
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .replace("IMG_", "");

        draw_text_simple(&mut canvas, lx, ly, &name, Rgb([200, 210, 240]));
    }

    {
        // Legend
        let lx = 40;
        let mut ly = canvas_size as i32 - 200;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "TRACK COVERAGE GRAPH",
            Rgb([180, 190, 220]),
        );
        ly += 24;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            &format!(
                "{} images  ·  {} tracks  ·  window={}",
                num_images,
                tracks.len(),
                SLIDING_WINDOW,
            ),
            Rgb([120, 130, 160]),
        );
        ly += 28;

        for (t, label) in [(0.05, "few"), (0.4, "medium"), (0.85, "many")] {
            let color = heat_color(t);
            drawing::draw_filled_circle_mut(&mut canvas, (lx + 6, ly + 3), 5, color);
            draw_text_simple(&mut canvas, lx + 20, ly, label, Rgb([160, 170, 200]));
            ly += 18;
        }

        ly += 8;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "Line color = shared track count  ·  Node size = keypoint count",
            Rgb([100, 110, 140]),
        );
        ly += 20;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            &format!(
                "Max shared tracks on one edge: {}  ·  Parameters: ratio={} kp_max={}",
                max_count as usize, RATIO_THRESHOLD, MAX_KEYPOINTS_PER_IMAGE
            ),
            Rgb([100, 110, 140]),
        );
    }

    canvas
        .save(output_path)
        .expect("Failed to save coverage graph");

    let elapsed = start.elapsed();
    println!(
        "Coverage graph saved to {} ({:.1} KB) in {:.2}s\n",
        output_path,
        fs::metadata(output_path)
            .map(|m| m.len() as f64 / 1024.0)
            .unwrap_or(0.0),
        elapsed.as_secs_f32(),
    );
}

fn draw_thick_line(
    canvas: &mut RgbImage,
    p1: (f32, f32),
    p2: (f32, f32),
    color: Rgb<u8>,
    thickness: f32,
) {
    let dx = p2.0 - p1.0;
    /// 带厚度的反锯齿直线
    let dy = p2.1 - p1.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        return;
    }
    let nx = -dy / len * thickness * 0.5;
    let ny = dx / len * thickness * 0.5;

    let (w, h) = canvas.dimensions();
    let (_wf, _hf) = (w as f32, h as f32);

    let steps = len.ceil() as i32;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = p1.0 + dx * t;
        let cy = p1.1 + dy * t;

        for o in -1i32..=1 {
            let off = o as f32;
            let px = (cx + nx * off) as i32;
            let py = (cy + ny * off) as i32;
            if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                let pixel = canvas.get_pixel_mut(px as u32, py as u32);
                let blend = 0.4f32;
                pixel[0] = (pixel[0] as f32 * (1.0 - blend) + color[0] as f32 * blend) as u8;
                pixel[1] = (pixel[1] as f32 * (1.0 - blend) + color[1] as f32 * blend) as u8;
                pixel[2] = (pixel[2] as f32 * (1.0 - blend) + color[2] as f32 * blend) as u8;
            }
        }
    }
}

fn draw_text_simple(canvas: &mut RgbImage, x: i32, y: i32, text: &str, color: Rgb<u8>) {
    let (w, h) = canvas.dimensions();
    let scale = 1;
    let char_w = 6 * scale;
    let _char_h = 10 * scale;

    let font_data: [(u8, [bool; 35]); 64] = build_font();

    for (ci, ch) in text.chars().enumerate() {
        /// 5x7位图字体在RGB图像上绘制ASCII文本
        let cx = x + ci as i32 * char_w;
        let cy = y;

        let idx = (ch as usize).min(63);
        let (w_bits, bits) = &font_data[idx];

        for row in 0..7 {
            for col in 0..*w_bits as usize {
                if bits[row * 5 + col] {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + col as i32 * scale + sx;
                            let py = cy + row as i32 * scale + sy;
                            if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                                canvas.put_pixel(px as u32, py as u32, color);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn visualize_track_pairs(
    tracks: &[Track],
    features: &[ImageFeatures],
    paths: &[String],
    output_dir: &str,
) {
    println!("\nGenerating track visualization images...");
    let start = Instant::now();

    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    /// 为每对相邻图像生成并排匹配可视化图
    let num_images = paths.len();
    let colors = generate_colors(tracks.len());

    let pairs: Vec<(usize, usize)> = (0..num_images.saturating_sub(1))
        .map(|i| (i, i + 1))
        .collect();

    pairs.par_iter().for_each(|&(i, j)| {
        let img_a = image::open(&paths[i])
            .unwrap_or_else(|_| panic!("Failed to open {}", paths[i]))
            .to_rgb8();
        let img_b = image::open(&paths[j])
            .unwrap_or_else(|_| panic!("Failed to open {}", paths[j]))
            .to_rgb8();

        let (w1, h1) = img_a.dimensions();
        let (w2, h2) = img_b.dimensions();
        let combined_w = w1 + w2 + 3;
        let combined_h = h1.max(h2);

        let mut canvas = RgbImage::from_pixel(combined_w, combined_h, Rgb([30, 30, 30]));

        canvas.copy_from(&img_a, 0, 0).unwrap();
        canvas.copy_from(&img_b, w1 + 3, 0).unwrap();

        let mut pair_track_count = 0usize;

        for (track_id, track) in tracks.iter().enumerate() {
            let has_img_i = track
                .observations
                .binary_search_by_key(&i, |&(img, _)| img)
                .ok();
            let has_img_j = track
                .observations
                .binary_search_by_key(&j, |&(img, _)| img)
                .ok();

            if let (Some(pos_i), Some(pos_j)) = (has_img_i, has_img_j) {
                pair_track_count += 1;
                let (_, kp_i) = track.observations[pos_i];
                let (_, kp_j) = track.observations[pos_j];

                let pt_i = &features[i].keypoints[kp_i];
                let pt_j = &features[j].keypoints[kp_j];

                let x1 = pt_i.x.round() as i32;
                let y1 = pt_i.y.round() as i32;
                let x2 = pt_j.x.round() as i32 + w1 as i32 + 3;
                let y2 = pt_j.y.round() as i32;

                let color = colors[track_id % colors.len()];

                drawing::draw_cross_mut(&mut canvas, color, x1, y1);
                drawing::draw_cross_mut(&mut canvas, color, x2, y2);
                drawing::draw_line_segment_mut(
                    &mut canvas,
                    (x1 as f32, y1 as f32),
                    (x2 as f32, y2 as f32),
                    color,
                );
            }
        }

        let name_a = std::path::Path::new(&paths[i])
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("?");
        let name_b = std::path::Path::new(&paths[j])
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("?");

        let out_path = format!(
            "{}/pair_{:02}_{:02}_{}_{}.png",
            output_dir, i, j, name_a, name_b
        );

        canvas.save(&out_path).expect("Failed to save viz image");
        println!(
            "  pair ({:2},{:2}) {:>12} ↔ {:<12}  {} tracks drawn → {}",
            i,
            j,
            name_a,
            name_b,
            pair_track_count,
            std::path::Path::new(&out_path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        );
    });

    let elapsed = start.elapsed();
    println!(
        "Generated {} pair visualizations in {:.2}s → {}/\n",
        pairs.len(),
        elapsed.as_secs_f32(),
        output_dir
    );
}

fn build_font() -> [(u8, [bool; 35]); 64] {
    let blank = [false; 35];
    let mut font = [(0u8, blank); 64];

    let chars: [(char, &[&str]); 16] = [
        ('0', &[" ### ", "#  ##", "# # #", "##  #", " ### "]),
        ('1', &["  ## ", " # # ", "   # ", "   # ", " #####"]),
        ('2', &[" ### ", "#   #", "  ## ", " #   ", "#####"]),
        ('3', &[" ### ", "#   #", "  ## ", "#   #", " ### "]),
        ('4', &["#   #", "#   #", "#####", "    #", "    #"]),
        ('5', &["#####", "#    ", "#### ", "    #", "#### "]),
        /// 构建5x7像素位图字体查找表
        ('6', &[" ### ", "#    ", "#### ", "#   #", " ### "]),
        ('7', &["#####", "    #", "   # ", "  #  ", " #   "]),
        ('8', &[" ### ", "#   #", " ### ", "#   #", " ### "]),
        ('9', &[" ### ", "#   #", " ####", "    #", " ### "]),
        ('.', &["     ", "     ", "     ", "     ", "  #  "]),
        ('-', &["     ", "     ", " ### ", "     ", "     "]),
        (':', &["     ", "  #  ", "     ", "  #  ", "     "]),
        ('=', &["     ", " ### ", "     ", " ### ", "     "]),
        ('·', &["     ", "     ", "  #  ", "     ", "     "]),
        ('/', &["   # ", "   # ", "  #  ", " #   ", " #   "]),
    ];

    for (ch, rows) in chars {
        let mut bits = [false; 35];
        for (r, row) in rows.iter().enumerate() {
            let bytes = row.as_bytes();
            for (c, &b) in bytes.iter().enumerate() {
                if c < 5 && b == b'#' {
                    bits[r * 5 + c] = true;
                }
            }
        }
        let idx = ch as usize;
        if idx < 64 {
            font[idx] = (5, bits);
        }
    }

    let alpha: [(char, &[&str]); 26] = [
        ('A', &[" ### ", "#   #", "#####", "#   #", "#   #"]),
        ('B', &["#### ", "#   #", "#### ", "#   #", "#### "]),
        ('C', &[" ####", "#    ", "#    ", "#    ", " ####"]),
        ('D', &["###  ", "#  # ", "#   #", "#  # ", "###  "]),
        ('E', &["#####", "#    ", "#### ", "#    ", "#####"]),
        ('F', &["#####", "#    ", "#### ", "#    ", "#    "]),
        ('G', &[" ####", "#    ", "#  ##", "#   #", " ####"]),
        ('H', &["#   #", "#   #", "#####", "#   #", "#   #"]),
        ('I', &[" ### ", "  #  ", "  #  ", "  #  ", " ### "]),
        ('J', &[" ####", "   # ", "   # ", "#  # ", " ##  "]),
        ('K', &["#   #", "#  # ", "###  ", "#  # ", "#   #"]),
        ('L', &["#    ", "#    ", "#    ", "#    ", "#####"]),
        ('M', &["#   #", "## ##", "# # #", "#   #", "#   #"]),
        ('N', &["#   #", "##  #", "# # #", "#  ##", "#   #"]),
        ('O', &[" ### ", "#   #", "#   #", "#   #", " ### "]),
        ('P', &["#### ", "#   #", "#### ", "#    ", "#    "]),
        ('Q', &[" ### ", "#   #", "# # #", "#  ##", " ####"]),
        ('R', &["#### ", "#   #", "#### ", "#  # ", "#   #"]),
        ('S', &[" ####", "#    ", " ### ", "    #", "#### "]),
        ('T', &["#####", "  #  ", "  #  ", "  #  ", "  #  "]),
        ('U', &["#   #", "#   #", "#   #", "#   #", " ### "]),
        ('V', &["#   #", "#   #", " # # ", " # # ", "  #  "]),
        ('W', &["#   #", "#   #", "# # #", "## ##", "#   #"]),
        ('X', &["#   #", " # # ", "  #  ", " # # ", "#   #"]),
        ('Y', &["#   #", " # # ", "  #  ", "  #  ", "  #  "]),
        ('Z', &["#####", "   # ", "  #  ", " #   ", "#####"]),
    ];

    for (ch, rows) in alpha {
        let mut bits = [false; 35];
        for (r, row) in rows.iter().enumerate() {
            let bytes = row.as_bytes();
            for (c, &b) in bytes.iter().enumerate() {
                if c < 5 && b == b'#' {
                    bits[r * 5 + c] = true;
                }
            }
        }
        let idx = ch as usize;
        if idx < 64 {
            font[idx] = (5, bits);
        }
    }

    let lower: [(char, &[&str]); 26] = [
        ('a', &["     ", " ### ", "   #", " ####", "#  ##"]),
        ('b', &["#    ", "#    ", "#### ", "#   #", "#### "]),
        ('c', &["     ", " ### ", "#    ", "#    ", " ### "]),
        ('d', &["   # ", "   # ", " ####", "#   #", " ####"]),
        ('e', &["     ", " ### ", "#####", "#    ", " ####"]),
        ('f', &["  ## ", " #   ", "#### ", " #   ", " #   "]),
        ('g', &["     ", " ####", "#   #", " ####", "   # "]),
        ('h', &["#    ", "#    ", "#### ", "#   #", "#   #"]),
        ('i', &["  #  ", "     ", " ##  ", "  #  ", " ####"]),
        ('j', &["   # ", "     ", "  ## ", "   # ", " ##  "]),
        ('k', &["#    ", "#  # ", "###  ", "#  # ", "#   #"]),
        ('l', &[" ##  ", "  #  ", "  #  ", "  #  ", " ####"]),
        ('m', &["     ", "## # ", "# # #", "#   #", "#   #"]),
        ('n', &["     ", "#### ", "#   #", "#   #", "#   #"]),
        ('o', &["     ", " ### ", "#   #", "#   #", " ### "]),
        ('p', &["     ", "#### ", "#   #", "#### ", "#    "]),
        ('q', &["     ", " ####", "#   #", " ####", "    #"]),
        ('r', &["     ", "# ## ", "##   ", "#    ", "#    "]),
        ('s', &["     ", " ####", "#    ", "    #", "#### "]),
        ('t', &[" #   ", "#### ", " #   ", " #  #", "  ## "]),
        ('u', &["     ", "#   #", "#   #", "#   #", " ####"]),
        ('v', &["     ", "#   #", " # # ", " # # ", "  #  "]),
        ('w', &["     ", "#   #", "# # #", "## ##", " # # "]),
        ('x', &["     ", "#   #", " # # ", " # # ", "#   #"]),
        ('y', &["     ", "#   #", " # # ", "  #  ", "##   "]),
        ('z', &["     ", "#####", "  #  ", " #   ", "#####"]),
    ];

    for (ch, rows) in lower {
        let mut bits = [false; 35];
        for (r, row) in rows.iter().enumerate() {
            let bytes = row.as_bytes();
            for (c, &b) in bytes.iter().enumerate() {
                if c < 5 && b == b'#' {
                    bits[r * 5 + c] = true;
                }
            }
        }
        let idx = ch as usize;
        if idx < 64 {
            font[idx] = (5, bits);
        }
    }

    let extra: [(char, &[&str]); 3] = [
        ('_', &["     ", "     ", "     ", "     ", "#####"]),
        (' ', &["     ", "     ", "     ", "     ", "     "]),
        ('*', &["# # #", " ### ", "#####", " ### ", "# # #"]),
    ];
    for (ch, rows) in extra {
        let mut bits = [false; 35];
        for (r, row) in rows.iter().enumerate() {
            let bytes = row.as_bytes();
            for (c, &b) in bytes.iter().enumerate() {
                if c < 5 && b == b'#' {
                    bits[r * 5 + c] = true;
                }
            }
        }
        let idx = ch as usize;
        if idx < 64 {
            font[idx] = (5, bits);
        }
    }

    font
}

// =====================================================================
//  Sparse Reconstruction (COLMAP-style incremental SfM)
// =====================================================================

// ---------------------------------------------------------------------
//  Options (COLMAP incremental mapper defaults)
// ---------------------------------------------------------------------
const INIT_MIN_NUM_INLIERS: usize = 100;
const INIT_MIN_TRI_ANGLE_DEG: f64 = 4.0;
const INIT_MAX_FORWARD_MOTION: f64 = 0.95;
const INIT_MAX_REG_TRIALS: usize = 3;
const INIT_NUM_TRIALS: usize = 3;

const MAX_REG_TRIALS: usize = 3;
const MAX_MODEL_OVERLAP: usize = 20;

const ABS_POSE_MIN_NUM_INLIERS: usize = 30;
const ABS_POSE_MAX_ERROR_PX: f64 = 12.0;
const ABS_POSE_MIN_INLIER_RATIO: f64 = 0.25;

const CREATE_MAX_ANGLE_ERROR_DEG: f64 = 2.0;
#[allow(dead_code)]
const CONTINUE_MAX_ANGLE_ERROR_DEG: f64 = 2.0;
const MERGE_MAX_REPROJ_ERROR_PX: f64 = 4.0;
const COMPLETE_MAX_REPROJ_ERROR_PX: f64 = 4.0;
const FILTER_MAX_REPROJ_ERROR_PX: f64 = 4.0;
const FILTER_MIN_TRI_ANGLE_DEG: f64 = 1.5;

const MAX_REFINEMENTS: usize = 5;
const MAX_REFINEMENT_CHANGE: f64 = 0.01;

const BA_LOCAL_NUM_IMAGES: usize = 6;
const BA_LOCAL_MAX_ITERATIONS: usize = 25;
const BA_GLOBAL_MAX_ITERATIONS: usize = 50;

const MIN_FOCAL_LENGTH_RATIO: f64 = 0.1;
const MAX_FOCAL_LENGTH_RATIO: f64 = 10.0;

const RANSAC_ITERATIONS: usize = 2000;
const RANSAC_THRESHOLD_PX: f64 = 4.0;

// ---------------------------------------------------------------------
//  Core types
// ---------------------------------------------------------------------
#[derive(Clone)]
struct CameraPose {
    r: Matrix3<f64>,
    t: Vector3<f64>,
}

#[derive(Clone)]
struct Point3D {
    xyz: Vector3<f64>,
    track_id: usize,
    error: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum LossType {
    Trivial,
    SoftL1,
    Cauchy,
}

impl LossType {
    fn weight(&self, r2: f64) -> f64 {
        match self {
            LossType::Trivial => 1.0,
            LossType::SoftL1 => (1.0 + r2).powf(-0.5),
            LossType::Cauchy => 1.0 / (1.0 + r2),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum ImageSelectionMethod {
    MaxVisiblePointsNum,
    MaxVisiblePointsRatio,
    MinUncertainty,
}

const IMAGE_SELECTION_METHOD: ImageSelectionMethod = ImageSelectionMethod::MaxVisiblePointsNum;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SfmStatus {
    Success,
    BadInitialPair,
    NoInitialPair,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum TwoViewConfig {
    Undefined,
    Calibrated,
    Uncalibrated,
    Degenerate,
    PlanarOrPanoramic,
}

#[allow(dead_code)]
struct TwoViewGeometry {
    config: TwoViewConfig,
    f: Matrix3<f64>,
    e: Matrix3<f64>,
    h: Matrix3<f64>,
    cam2_from_cam1: CameraPose,
    inliers: Vec<usize>,
    tri_angle: f64,
}

// ---------------------------------------------------------------------
//  Calibration / EXIF
// ---------------------------------------------------------------------
struct CameraIntrinsics {
    k: Matrix3<f64>,
    has_prior_focal: bool,
}

impl CameraIntrinsics {
    fn from_image_dimensions(w: u32, h: u32) -> Self {
        let wf = w as f64;
        let hf = h as f64;
        let f = wf.max(hf);
        let cx = wf / 2.0;
        let cy = hf / 2.0;
        let k = Matrix3::new(f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0);
        CameraIntrinsics {
            k,
            has_prior_focal: false,
        }
    }

    fn from_image_with_exif(path: &str, w: u32, h: u32) -> Self {
        let wf = w as f64;
        let hf = h as f64;
        let mut f_px = wf.max(hf);
        let mut has_prior = false;

        if let Ok(data) = fs::read(path) {
            if let Some(fl_mm) = read_exif_focal_length(&data) {
                let sensor_w_mm = 6.2;
                let f_from_exif = wf * fl_mm / sensor_w_mm;
                if f_from_exif > 500.0 && f_from_exif < 100000.0 && f_from_exif > wf * 0.1 {
                    f_px = f_from_exif;
                    has_prior = true;
                }
            }
        }

        let cx = wf / 2.0;
        let cy = hf / 2.0;
        let k = Matrix3::new(f_px, 0.0, cx, 0.0, f_px, cy, 0.0, 0.0, 1.0);
        CameraIntrinsics {
            k,
            has_prior_focal: has_prior,
        }
    }
}

fn read_exif_focal_length(data: &[u8]) -> Option<f64> {
    let mut pos = 0;
    while pos + 4 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xE1 {
            break;
        }
        pos += 1;
    }
    if pos + 10 > data.len() {
        return None;
    }
    let app1_len = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
    if pos + 2 + app1_len > data.len() {
        return None;
    }
    let exif_start = pos + 4;
    if &data[exif_start..exif_start + 6] != b"Exif\x00\x00" {
        return None;
    }
    let tiff_start = exif_start + 6;
    if tiff_start + 8 > data.len() {
        return None;
    }
    let le = &data[tiff_start..tiff_start + 2] == b"\x49\x49";

    let read_u16 = |p: usize| -> u16 {
        if le {
            (data[p] as u16) | ((data[p + 1] as u16) << 8)
        } else {
            ((data[p] as u16) << 8) | (data[p + 1] as u16)
        }
    };
    let read_u32 = |p: usize| -> u32 {
        if le {
            (data[p] as u32)
                | ((data[p + 1] as u32) << 8)
                | ((data[p + 2] as u32) << 16)
                | ((data[p + 3] as u32) << 24)
        } else {
            ((data[p] as u32) << 24)
                | ((data[p + 1] as u32) << 16)
                | ((data[p + 2] as u32) << 8)
                | (data[p + 3] as u32)
        }
    };

    let tiff_off = read_u32(tiff_start + 4) as usize;
    let parse_ifd = |ifd_base: usize| -> Option<f64> {
        if ifd_base + 2 > data.len() {
            return None;
        }
        let n = read_u16(ifd_base) as usize;
        let mut ep = ifd_base + 2;
        for _ in 0..n.min(200) {
            if ep + 12 > data.len() {
                break;
            }
            let tag = read_u16(ep);
            if tag == 0x920A || tag == 0xA405 {
                let val_off = read_u32(ep + 8) as usize;
                let vp = tiff_start + val_off;
                if vp + 8 <= data.len() {
                    let num = read_u32(vp) as f64;
                    let den = read_u32(vp + 4) as f64;
                    if den > 0.0 {
                        return Some(num / den);
                    }
                }
            }
            ep += 12;
        }
        None
    };

    if let Some(fl) = parse_ifd(tiff_start + tiff_off) {
        return Some(fl);
    }
    let ifd0_pos = tiff_start + tiff_off;
    let n_entries = read_u16(ifd0_pos) as usize;
    let mut ep = ifd0_pos + 2;
    for _ in 0..n_entries.min(200) {
        if ep + 12 > data.len() {
            break;
        }
        if read_u16(ep) == 0x8769 {
            let sub_off = read_u32(ep + 8) as usize;
            if let Some(fl) = parse_ifd(tiff_start + sub_off) {
                return Some(fl);
            }
        }
        ep += 12;
    }
    None
}

fn img_file_stem(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_owned()
}

// ---------------------------------------------------------------------
//  SO(3) utilities
// ---------------------------------------------------------------------
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn so3_exp(omega: &Vector3<f64>) -> Matrix3<f64> {
    let theta = omega.norm();
    if theta < 1e-12 {
        return Matrix3::identity() + skew_symmetric(omega);
    }
    let a = omega / theta;
    let ax = skew_symmetric(&a);
    Matrix3::identity() + theta.sin() * ax + (1.0 - theta.cos()) * (ax * ax)
}

fn so3_log(r: &Matrix3<f64>) -> Vector3<f64> {
    let theta = ((r.trace() - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    if theta.abs() < 1e-8 {
        return Vector3::new(
            (r[(2, 1)] - r[(1, 2)]) * 0.5,
            (r[(0, 2)] - r[(2, 0)]) * 0.5,
            (r[(1, 0)] - r[(0, 1)]) * 0.5,
        );
    }
    let c = theta / (2.0 * theta.sin());
    Vector3::new(
        c * (r[(2, 1)] - r[(1, 2)]),
        c * (r[(0, 2)] - r[(2, 0)]),
        c * (r[(1, 0)] - r[(0, 1)]),
    )
}

fn camera_center(r: &Matrix3<f64>, t: &Vector3<f64>) -> Vector3<f64> {
    r.transpose() * (-t)
}

// ---------------------------------------------------------------------
//  Projection
// ---------------------------------------------------------------------
#[inline]
fn project_point(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    pt: &Vector3<f64>,
    f: f64,
    cx: f64,
    cy: f64,
) -> Option<(f64, f64, f64)> {
    let xc = r * pt + t;
    if xc.z <= 1e-8 {
        return None;
    }
    Some((f * xc.x / xc.z + cx, f * xc.y / xc.z + cy, xc.z))
}

fn proj_matrix(r: &Matrix3<f64>, t: &Vector3<f64>, k: &Matrix3<f64>) -> Matrix3x4<f64> {
    let mut p = Matrix3x4::zeros();
    for c in 0..3 {
        p.column_mut(c).copy_from(&r.column(c));
    }
    p.column_mut(3).copy_from(t);
    k * p
}

// ---------------------------------------------------------------------
//  Normalization + DLT solvers
// ---------------------------------------------------------------------
fn normalize_points(points: &[Vector3<f64>]) -> (Vec<Vector3<f64>>, Matrix3<f64>) {
    let n = points.len() as f64;
    let cx: f64 = points.iter().map(|p| p.x).sum::<f64>() / n;
    let cy: f64 = points.iter().map(|p| p.y).sum::<f64>() / n;
    let mean_dist: f64 = points
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if mean_dist > 1e-10 {
        (2.0_f64).sqrt() / mean_dist
    } else {
        1.0
    };
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let normalized: Vec<Vector3<f64>> = points
        .iter()
        .map(|p| Vector3::new(s * p.x - s * cx, s * p.y - s * cy, 1.0))
        .collect();
    (normalized, t)
}

fn compute_fundamental_8point(pts1: &[Vector3<f64>], pts2: &[Vector3<f64>]) -> Matrix3<f64> {
    let n = pts1.len();
    let (norms1, t1) = normalize_points(pts1);
    let (norms2, t2) = normalize_points(pts2);

    let mut a = na::DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let x1 = norms1[i].x;
        let y1 = norms1[i].y;
        let x2 = norms2[i].x;
        let y2 = norms2[i].y;
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.unwrap();
    let f_vec = v_t.row(v_t.nrows() - 1);

    let mut f_norm = Matrix3::new(
        f_vec[0], f_vec[1], f_vec[2], f_vec[3], f_vec[4], f_vec[5], f_vec[6], f_vec[7], f_vec[8],
    );

    let svd_f = SVD::new(f_norm, true, true);
    let mut s = svd_f.singular_values;
    s[2] = 0.0;
    let s_mat = Matrix3::from_diagonal(&Vector3::new(s[0], s[1], s[2]));
    f_norm = svd_f.u.unwrap() * s_mat * svd_f.v_t.unwrap();

    t2.transpose() * f_norm * t1
}

fn compute_homography_4point(pts1: &[Vector3<f64>], pts2: &[Vector3<f64>]) -> Matrix3<f64> {
    let n = pts1.len();
    let (norms1, t1) = normalize_points(pts1);
    let (norms2, t2) = normalize_points(pts2);

    let mut a = na::DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let x1 = norms1[i].x;
        let y1 = norms1[i].y;
        let x2 = norms2[i].x;
        let y2 = norms2[i].y;
        a[(2 * i, 0)] = -x1;
        a[(2 * i, 1)] = -y1;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = x2 * x1;
        a[(2 * i, 7)] = x2 * y1;
        a[(2 * i, 8)] = x2;
        a[(2 * i + 1, 3)] = -x1;
        a[(2 * i + 1, 4)] = -y1;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = y2 * x1;
        a[(2 * i + 1, 7)] = y2 * y1;
        a[(2 * i + 1, 8)] = y2;
    }
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.unwrap();
    let h_vec = v_t.row(v_t.nrows() - 1);
    let h_norm = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4], h_vec[5], h_vec[6], h_vec[7], h_vec[8],
    );
    let h_unnorm = t2.try_inverse().unwrap() * h_norm * t1;
    h_unnorm / h_unnorm[(2, 2)]
}

fn sampson_distance(f: &Matrix3<f64>, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
    let fp1 = f * p1;
    let ftp2 = f.transpose() * p2;
    let numerator = (p2.dot(&fp1)).powi(2);
    let denominator = fp1.x.powi(2) + fp1.y.powi(2) + ftp2.x.powi(2) + ftp2.y.powi(2);
    if denominator < 1e-12 {
        return 1e9;
    }
    numerator / denominator
}

fn ransac_fundamental(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    max_iterations: usize,
    threshold: f64,
) -> (Matrix3<f64>, Vec<usize>) {
    let n = pts1.len();
    if n < 8 {
        return (Matrix3::zeros(), Vec::new());
    }
    let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
    let mut best_f = Matrix3::zeros();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut indices: Vec<usize> = (0..n).collect();

    for _ in 0..max_iterations {
        indices.shuffle(&mut rng);
        let sample = &indices[..8];
        let s1: Vec<Vector3<f64>> = sample.iter().map(|&i| pts1[i]).collect();
        let s2: Vec<Vector3<f64>> = sample.iter().map(|&i| pts2[i]).collect();
        let f = compute_fundamental_8point(&s1, &s2);
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| sampson_distance(&f, &pts1[i], &pts2[i]) < threshold)
            .collect();
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_f = f;
        }
    }

    if best_inliers.len() >= 8 {
        let i1: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| pts1[i]).collect();
        let i2: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| pts2[i]).collect();
        best_f = compute_fundamental_8point(&i1, &i2);
        best_inliers = (0..n)
            .filter(|&i| sampson_distance(&best_f, &pts1[i], &pts2[i]) < threshold)
            .collect();
    }

    (best_f, best_inliers)
}

fn ransac_homography(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    max_iterations: usize,
    threshold: f64,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let n = pts1.len();
    if n < 4 {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
    let mut best_h = Matrix3::zeros();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut indices: Vec<usize> = (0..n).collect();

    for _ in 0..max_iterations {
        indices.shuffle(&mut rng);
        let sample = &indices[..4];
        let s1: Vec<Vector3<f64>> = sample.iter().map(|&i| pts1[i]).collect();
        let s2: Vec<Vector3<f64>> = sample.iter().map(|&i| pts2[i]).collect();
        let h = compute_homography_4point(&s1, &s2);
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| {
                let p2p = h * pts1[i];
                let w = p2p.z;
                if w.abs() < 1e-12 {
                    return false;
                }
                let dx = p2p.x / w - pts2[i].x;
                let dy = p2p.y / w - pts2[i].y;
                dx * dx + dy * dy < threshold
            })
            .collect();
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_h = h;
        }
    }

    if best_inliers.len() >= 4 {
        let i1: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| pts1[i]).collect();
        let i2: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| pts2[i]).collect();
        best_h = compute_homography_4point(&i1, &i2);
        best_inliers = (0..n)
            .filter(|&i| {
                let p2p = best_h * pts1[i];
                let w = p2p.z;
                if w.abs() < 1e-12 {
                    return false;
                }
                let dx = p2p.x / w - pts2[i].x;
                let dy = p2p.y / w - pts2[i].y;
                dx * dx + dy * dy < threshold
            })
            .collect();
    }

    Some((best_h, best_inliers))
}

fn essential_from_fundamental(f: &Matrix3<f64>, k: &CameraIntrinsics) -> Matrix3<f64> {
    let e = k.k.transpose() * f * k.k;
    let svd = SVD::new(e, true, true);
    let s = svd.singular_values;
    let avg = (s[0] + s[1]) / 2.0;
    let s_vec = Vector3::new(avg, avg, 0.0);
    svd.u.unwrap() * Matrix3::from_diagonal(&s_vec) * svd.v_t.unwrap()
}

fn decompose_essential(e: &Matrix3<f64>) -> (Matrix3<f64>, Matrix3<f64>, Vector3<f64>) {
    let svd = SVD::new(*e, true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let v = v_t.transpose();

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let mut r1 = u * w * v.transpose();
    let mut r2 = u * w.transpose() * v.transpose();
    if r1.determinant() < 0.0 {
        r1 = -r1;
        r2 = -r2;
    }
    let t = u.column(2).into_owned();
    (r1, r2, t)
}

fn triangulate_dlt_multi(projs: &[Matrix3x4<f64>], pts: &[Vector3<f64>]) -> Option<Vector3<f64>> {
    let n = pts.len();
    if n < 2 {
        return None;
    }
    let mut a = na::DMatrix::<f64>::zeros(2 * n, 4);
    for i in 0..n {
        let p = &projs[i];
        let x = pts[i].x;
        let y = pts[i].y;
        a[(2 * i, 0)] = x * p[(2, 0)] - p[(0, 0)];
        a[(2 * i, 1)] = x * p[(2, 1)] - p[(0, 1)];
        a[(2 * i, 2)] = x * p[(2, 2)] - p[(0, 2)];
        a[(2 * i, 3)] = x * p[(2, 3)] - p[(0, 3)];
        a[(2 * i + 1, 0)] = y * p[(2, 0)] - p[(1, 0)];
        a[(2 * i + 1, 1)] = y * p[(2, 1)] - p[(1, 1)];
        a[(2 * i + 1, 2)] = y * p[(2, 2)] - p[(1, 2)];
        a[(2 * i + 1, 3)] = y * p[(2, 3)] - p[(1, 3)];
    }
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.unwrap();
    let x = v_t.row(3);
    let w = x[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Vector3::new(x[0] / w, x[1] / w, x[2] / w))
}

fn calc_triangulation_angle(
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
    pt: &Vector3<f64>,
) -> f64 {
    let c1 = camera_center(r1, t1);
    let c2 = camera_center(r2, t2);
    let v1 = (pt - c1).normalize();
    let v2 = (pt - c2).normalize();
    v1.dot(&v2).clamp(-1.0, 1.0).acos()
}

// ---------------------------------------------------------------------
//  Generic dense Levenberg-Marquardt (numeric Jacobian)
// ---------------------------------------------------------------------
fn lm_refine<F>(mut params: na::DVector<f64>, residual_fn: &F, max_iter: usize) -> na::DVector<f64>
where
    F: Fn(&na::DVector<f64>) -> na::DVector<f64>,
{
    let mut lambda = 1e-3f64;
    for _ in 0..max_iter {
        let r = residual_fn(&params);
        let cost = r.dot(&r);
        let np = params.len();
        let nr = r.len();

        let mut j = na::DMatrix::<f64>::zeros(nr, np);
        let eps = 1e-7;
        for c in 0..np {
            let orig = params[c];
            let h = (orig.abs() * eps).max(1e-8);
            params[c] = orig + h;
            let rp = residual_fn(&params);
            params[c] = orig - h;
            let rm = residual_fn(&params);
            params[c] = orig;
            for row in 0..nr {
                j[(row, c)] = (rp[row] - rm[row]) / (2.0 * h);
            }
        }

        let jt = j.transpose();
        let mut hmat = &jt * &j;
        for d in 0..np {
            hmat[(d, d)] += lambda * hmat[(d, d)].abs().max(1e-6);
        }
        let g = &jt * &r;
        let delta = match hmat.clone().lu().solve(&(-g)) {
            Some(d) => d,
            None => {
                lambda *= 10.0;
                if lambda > 1e12 {
                    break;
                }
                continue;
            }
        };

        let new_params = &params + &delta;
        let new_r = residual_fn(&new_params);
        let new_cost = new_r.dot(&new_r);

        if new_cost < cost {
            params = new_params;
            if (cost - new_cost).abs() < 1e-9 * cost.max(1e-12) {
                break;
            }
            lambda = (lambda * 0.5).max(1e-12);
        } else {
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
    }
    params
}

// ---------------------------------------------------------------------
//  Absolute pose estimation (PnP)
// ---------------------------------------------------------------------
fn solve_pnp_dlt(
    points_3d: &[Vector3<f64>],
    points_2d: &[(f64, f64)],
    k_inv: &Matrix3<f64>,
) -> (Matrix3<f64>, Vector3<f64>) {
    let n = points_3d.len();
    let mut a = na::DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        let x = points_3d[i].x;
        let y = points_3d[i].y;
        let z = points_3d[i].z;
        let u = k_inv[(0, 0)] * points_2d[i].0 + k_inv[(0, 2)];
        let v = k_inv[(1, 1)] * points_2d[i].1 + k_inv[(1, 2)];
        let r1 = 2 * i;
        let r2 = 2 * i + 1;
        a[(r1, 0)] = x;
        a[(r1, 1)] = y;
        a[(r1, 2)] = z;
        a[(r1, 3)] = 1.0;
        a[(r1, 8)] = -u * x;
        a[(r1, 9)] = -u * y;
        a[(r1, 10)] = -u * z;
        a[(r1, 11)] = -u;
        a[(r2, 4)] = x;
        a[(r2, 5)] = y;
        a[(r2, 6)] = z;
        a[(r2, 7)] = 1.0;
        a[(r2, 8)] = -v * x;
        a[(r2, 9)] = -v * y;
        a[(r2, 10)] = -v * z;
        a[(r2, 11)] = -v;
    }
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.unwrap();
    let p_vec = v_t.row(v_t.nrows() - 1);
    let mut p = Matrix3x4::zeros();
    for c in 0..4 {
        p[(0, c)] = p_vec[c];
        p[(1, c)] = p_vec[4 + c];
        p[(2, c)] = p_vec[8 + c];
    }
    let r_approx = p.fixed_view::<3, 3>(0, 0).into_owned();
    let t_approx = p.column(3).into_owned();
    let svd_r = SVD::new(r_approx, true, true);
    let scale =
        (svd_r.singular_values[0] + svd_r.singular_values[1] + svd_r.singular_values[2]) / 3.0;
    let mut s = Vector3::new(1.0, 1.0, 1.0);
    if svd_r.u.as_ref().map(|u| u.determinant()).unwrap_or(1.0)
        * svd_r.v_t.as_ref().map(|v| v.determinant()).unwrap_or(1.0)
        < 0.0
    {
        s[2] = -1.0;
    }
    let r = svd_r.u.unwrap() * Matrix3::from_diagonal(&s) * svd_r.v_t.unwrap();
    let mut t = t_approx / scale.max(1e-12);
    let neg = points_3d.iter().filter(|&p3| (r * p3 + t).z <= 0.0).count();
    if neg > points_3d.len() / 2 {
        t = -t;
    }
    (r, t)
}

fn refine_absolute_pose(
    pts3d: &[Vector3<f64>],
    pts2d: &[(f64, f64)],
    init: &CameraPose,
    f0: f64,
    cx: f64,
    cy: f64,
    refine_focal: bool,
) -> (CameraPose, f64) {
    let nparam = if refine_focal { 7 } else { 6 };
    let mut params = na::DVector::<f64>::zeros(nparam);
    let w0 = so3_log(&init.r);
    params[0] = w0.x;
    params[1] = w0.y;
    params[2] = w0.z;
    params[3] = init.t.x;
    params[4] = init.t.y;
    params[5] = init.t.z;
    if refine_focal {
        params[6] = f0;
    }

    let residual = |p: &na::DVector<f64>| -> na::DVector<f64> {
        let omega = Vector3::new(p[0], p[1], p[2]);
        let r = so3_exp(&omega);
        let t = Vector3::new(p[3], p[4], p[5]);
        let f = if refine_focal {
            p[6].abs().max(1.0)
        } else {
            f0
        };
        let mut res = na::DVector::<f64>::zeros(pts3d.len() * 2);
        for (i, (p3, p2)) in pts3d.iter().zip(pts2d.iter()).enumerate() {
            match project_point(&r, &t, p3, f, cx, cy) {
                Some((u, v, _)) => {
                    res[2 * i] = u - p2.0;
                    res[2 * i + 1] = v - p2.1;
                }
                None => {
                    res[2 * i] = 1000.0;
                    res[2 * i + 1] = 1000.0;
                }
            }
        }
        res
    };

    let out = lm_refine(params, &residual, 30);
    let omega = Vector3::new(out[0], out[1], out[2]);
    let r = so3_exp(&omega);
    let t = Vector3::new(out[3], out[4], out[5]);
    let f = if refine_focal {
        out[6].abs().max(1.0)
    } else {
        f0
    };
    let fin = residual(&out);
    let _rms = (fin.dot(&fin) / (fin.len() as f64).max(1.0)).sqrt();
    (CameraPose { r, t }, f)
}

fn estimate_absolute_pose(
    pts3d: &[Vector3<f64>],
    pts2d: &[(f64, f64)],
    k_inv: &Matrix3<f64>,
    f0: f64,
    cx: f64,
    cy: f64,
    max_error_px: f64,
    refine_focal: bool,
) -> Option<(CameraPose, f64, Vec<bool>)> {
    let n = pts3d.len();
    if n < 6 {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
    let mut best_inliers = 0usize;
    let mut best_pose = CameraPose {
        r: Matrix3::identity(),
        t: Vector3::zeros(),
    };

    for _ in 0..RANSAC_ITERATIONS.min(1000) {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample: Vec<usize> = indices[..6].to_vec();
        let s3d: Vec<Vector3<f64>> = sample.iter().map(|&i| pts3d[i]).collect();
        let s2d: Vec<(f64, f64)> = sample.iter().map(|&i| pts2d[i]).collect();
        let (r, t) = solve_pnp_dlt(&s3d, &s2d, k_inv);
        let mut cnt = 0usize;
        for i in 0..n {
            if let Some((u, v, _)) = project_point(&r, &t, &pts3d[i], f0, cx, cy) {
                let e2 = (u - pts2d[i].0).powi(2) + (v - pts2d[i].1).powi(2);
                if e2 < max_error_px * max_error_px {
                    cnt += 1;
                }
            }
        }
        if cnt > best_inliers {
            best_inliers = cnt;
            best_pose = CameraPose { r, t };
        }
    }

    if best_inliers < ABS_POSE_MIN_NUM_INLIERS {
        return None;
    }

    let mut in3d = Vec::new();
    let mut in2d = Vec::new();
    let mut mask = vec![false; n];
    for i in 0..n {
        if let Some((u, v, _)) = project_point(&best_pose.r, &best_pose.t, &pts3d[i], f0, cx, cy) {
            let e2 = (u - pts2d[i].0).powi(2) + (v - pts2d[i].1).powi(2);
            if e2 < max_error_px * max_error_px {
                in3d.push(pts3d[i]);
                in2d.push(pts2d[i]);
                mask[i] = true;
            }
        }
    }

    let (refined, f_refined) =
        refine_absolute_pose(&in3d, &in2d, &best_pose, f0, cx, cy, refine_focal);
    Some((refined, f_refined, mask))
}

// ---------------------------------------------------------------------
//  Two-view geometry estimation
// ---------------------------------------------------------------------
fn recover_pose_from_essential(
    e: &Matrix3<f64>,
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    k: &CameraIntrinsics,
    min_num_inliers: usize,
) -> Option<(CameraPose, Vec<Vector3<f64>>, Vec<usize>)> {
    let (r1, r2, t_unit) = decompose_essential(e);
    let proj0 = proj_matrix(&Matrix3::identity(), &Vector3::zeros(), &k.k);

    let poses = [(r1, t_unit), (r1, -t_unit), (r2, t_unit), (r2, -t_unit)];
    let mut best_count = 0usize;
    let mut best_pose = (Matrix3::identity(), Vector3::zeros());
    let mut best_points: Vec<Vector3<f64>> = Vec::new();
    let mut best_valid: Vec<usize> = Vec::new();

    for &(r, t_vec) in &poses {
        let proj1 = proj_matrix(&r, &t_vec, &k.k);
        let mut count = 0usize;
        let mut valid: Vec<usize> = Vec::new();
        let mut points: Vec<Vector3<f64>> = Vec::new();
        for (i, (p1, p2)) in pts1.iter().zip(pts2.iter()).enumerate() {
            if let Some(pt) = triangulate_dlt_multi(&[proj0.clone(), proj1.clone()], &[*p1, *p2]) {
                let xc0 = pt;
                let xc1 = r * pt + t_vec;
                if xc0.z > 0.0 && xc1.z > 0.0 {
                    count += 1;
                    valid.push(i);
                }
                points.push(pt);
            } else {
                points.push(Vector3::zeros());
            }
        }
        if count > best_count {
            best_count = count;
            best_pose = (r, t_vec);
            best_points = points;
            best_valid = valid;
        }
    }

    if best_count < min_num_inliers {
        return None;
    }
    Some((
        CameraPose {
            r: best_pose.0,
            t: best_pose.1,
        },
        best_points,
        best_valid,
    ))
}

fn estimate_two_view_geometry(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    intrinsics: &CameraIntrinsics,
    min_num_inliers: usize,
) -> Option<TwoViewGeometry> {
    let (f_mat, f_inliers) =
        ransac_fundamental(pts1, pts2, RANSAC_ITERATIONS, RANSAC_THRESHOLD_PX.powi(2));
    if f_inliers.len() < 8 {
        return None;
    }

    let e_mat = essential_from_fundamental(&f_mat, intrinsics);

    let (h_mat, h_inliers) =
        ransac_homography(pts1, pts2, RANSAC_ITERATIONS, RANSAC_THRESHOLD_PX.powi(2))
            .unwrap_or((Matrix3::identity(), Vec::new()));

    let (pose, points, _valid) =
        recover_pose_from_essential(&e_mat, pts1, pts2, intrinsics, min_num_inliers.min(30))?;

    let mut angles: Vec<f64> = Vec::new();
    for pt in points.iter() {
        if pt.z > 0.0 {
            let a = calc_triangulation_angle(
                &Matrix3::identity(),
                &Vector3::zeros(),
                &pose.r,
                &pose.t,
                pt,
            );
            angles.push(a);
        }
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tri_angle = if angles.is_empty() {
        0.0
    } else {
        angles[angles.len() / 2]
    };

    let config = if f_inliers.len() >= h_inliers.len() {
        TwoViewConfig::Calibrated
    } else {
        TwoViewConfig::PlanarOrPanoramic
    };

    Some(TwoViewGeometry {
        config,
        f: f_mat,
        e: e_mat,
        h: h_mat,
        cam2_from_cam1: pose,
        inliers: f_inliers,
        tri_angle,
    })
}

// ---------------------------------------------------------------------
//  Reconstruction state (adapted to existing flat arrays)
// ---------------------------------------------------------------------
struct Sfm {
    num_images: usize,
    intrinsics: CameraIntrinsics,
    focals: Vec<f64>,
    has_prior_focal: Vec<bool>,
    cameras: Vec<Option<CameraPose>>,
    points: Vec<Point3D>,
    track_to_point: BTreeMap<usize, usize>,
    reg_trials: Vec<usize>,
    filtered: Vec<bool>,
}

impl Sfm {
    fn new(num_images: usize, intrinsics: CameraIntrinsics) -> Self {
        let f = intrinsics.k[(0, 0)];
        let has_prior = intrinsics.has_prior_focal;
        Sfm {
            num_images,
            intrinsics,
            focals: vec![f; num_images],
            has_prior_focal: vec![has_prior; num_images],
            cameras: vec![None; num_images],
            points: Vec::new(),
            track_to_point: BTreeMap::new(),
            reg_trials: vec![0; num_images],
            filtered: vec![false; num_images],
        }
    }

    fn cx(&self) -> f64 {
        self.intrinsics.k[(0, 2)]
    }
    fn cy(&self) -> f64 {
        self.intrinsics.k[(1, 2)]
    }

    fn rebuild_track_to_point(&mut self) {
        self.track_to_point.clear();
        for (pidx, pt) in self.points.iter().enumerate() {
            if pt.track_id != usize::MAX {
                self.track_to_point.insert(pt.track_id, pidx);
            }
        }
    }

    fn invalidate_point(&mut self, pidx: usize) {
        self.points[pidx].track_id = usize::MAX;
        self.points[pidx].xyz = Vector3::zeros();
        self.points[pidx].error = f64::MAX;
    }

    fn num_registered(&self) -> usize {
        self.cameras.iter().filter(|c| c.is_some()).count()
    }

    fn registered_images(&self) -> Vec<usize> {
        self.cameras
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_some())
            .map(|(i, _)| i)
            .collect()
    }

    fn visible_points_count(&self, tracks: &[Track], img: usize) -> usize {
        let mut cnt = 0usize;
        for (tid, &pidx) in &self.track_to_point {
            let track = &tracks[*tid];
            if track
                .observations
                .binary_search_by_key(&img, |&(im, _)| im)
                .is_ok()
                && self.points[pidx].track_id != usize::MAX
                && self.points[pidx].xyz.x.is_finite()
            {
                cnt += 1;
            }
        }
        cnt
    }

    fn point_is_valid(&self, pidx: usize) -> bool {
        let pt = &self.points[pidx];
        pt.track_id != usize::MAX
            && pt.xyz.x.is_finite()
            && pt.xyz.y.is_finite()
            && pt.xyz.z.is_finite()
            && pt.xyz.norm_squared() > 1e-12
    }

    fn normalize(&mut self) {
        let mut coords: Vec<Vector3<f64>> = Vec::new();
        for pt in &self.points {
            if pt.track_id != usize::MAX && pt.xyz.x.is_finite() {
                coords.push(pt.xyz);
            }
        }
        for cam in self.cameras.iter().flatten() {
            coords.push(camera_center(&cam.r, &cam.t));
        }
        if coords.is_empty() {
            return;
        }
        let n = coords.len() as f64;
        let mean = coords.iter().fold(Vector3::zeros(), |a, b| a + *b) / n;

        for pt in &mut self.points {
            if pt.track_id != usize::MAX {
                pt.xyz -= mean;
            }
        }
        for cam in self.cameras.iter_mut().flatten() {
            cam.t += cam.r * mean;
        }

        let scale = coords
            .iter()
            .map(|c| {
                let c = c - mean;
                c.x.abs() + c.y.abs() + c.z.abs()
            })
            .sum::<f64>()
            / (3.0 * n);
        if scale > 1e-8 {
            for pt in &mut self.points {
                if pt.track_id != usize::MAX {
                    pt.xyz /= scale;
                }
            }
            for cam in self.cameras.iter_mut().flatten() {
                cam.t /= scale;
            }
        }
    }

    // -----------------------------------------------------------------
    //  Initialization
    // -----------------------------------------------------------------
    fn find_first_initial_image(
        &self,
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) -> Option<usize> {
        let mut best: Option<(usize, bool, usize)> = None; // (num_corrs, has_prior, img)
        for img in 0..self.num_images {
            if self.filtered[img] || self.reg_trials[img] >= INIT_MAX_REG_TRIALS {
                continue;
            }
            let mut num_corrs = 0usize;
            for (key, v) in index {
                if key.0 == img || key.1 == img {
                    num_corrs += v.len();
                }
            }
            if num_corrs == 0 {
                continue;
            }
            let cand = (num_corrs, self.has_prior_focal[img], img);
            let replace = match &best {
                None => true,
                Some(b) => (cand.1, cand.0) > (b.1, b.0),
            };
            if replace {
                best = Some(cand);
            }
        }
        best.map(|(_, _, img)| img)
    }

    fn find_second_initial_image(
        &self,
        first: usize,
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        min_num_inliers: usize,
    ) -> Vec<usize> {
        let mut candidates: Vec<(usize, usize, bool)> = Vec::new(); // (num_corrs, img, has_prior)
        for img in 0..self.num_images {
            if img == first || self.filtered[img] || self.reg_trials[img] >= INIT_MAX_REG_TRIALS {
                continue;
            }
            let key = if first < img {
                (first, img)
            } else {
                (img, first)
            };
            let n = index.get(&key).map(|v| v.len()).unwrap_or(0);
            if n >= min_num_inliers {
                candidates.push((n, img, self.has_prior_focal[img]));
            }
        }
        candidates.sort_by(|a, b| (b.2 as usize, b.0).cmp(&(a.2 as usize, a.0)));
        candidates.into_iter().map(|(_, img, _)| img).collect()
    }

    fn estimate_initial_two_view_geometry(
        &self,
        first: usize,
        second: usize,
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        features: &[ImageFeatures],
        min_num_inliers: usize,
        min_tri_angle: f64,
    ) -> Option<TwoViewGeometry> {
        let key = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        let corrs = index.get(&key)?;
        if corrs.len() < min_num_inliers {
            return None;
        }

        let mut pts1 = Vec::with_capacity(corrs.len());
        let mut pts2 = Vec::with_capacity(corrs.len());
        for &(ka, kb) in corrs {
            let a = if first < second { ka } else { kb };
            let b = if first < second { kb } else { ka };
            let kpa = &features[first].keypoints[a];
            let kpb = &features[second].keypoints[b];
            pts1.push(Vector3::new(kpa.x as f64, kpa.y as f64, 1.0));
            pts2.push(Vector3::new(kpb.x as f64, kpb.y as f64, 1.0));
        }

        let g = estimate_two_view_geometry(&pts1, &pts2, &self.intrinsics, min_num_inliers)?;
        if g.inliers.len() < min_num_inliers {
            return None;
        }
        if g.cam2_from_cam1.t.z.abs() / g.cam2_from_cam1.t.norm().max(1e-12)
            > INIT_MAX_FORWARD_MOTION
        {
            return None;
        }
        if g.tri_angle < min_tri_angle {
            return None;
        }
        Some(g)
    }

    fn find_initial_image_pair(
        &self,
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        features: &[ImageFeatures],
        min_num_inliers: usize,
        min_tri_angle: f64,
    ) -> Option<(usize, usize, TwoViewGeometry)> {
        let first = self.find_first_initial_image(index)?;
        let seconds = self.find_second_initial_image(first, index, min_num_inliers);
        for second in seconds {
            if let Some(g) = self.estimate_initial_two_view_geometry(
                first,
                second,
                index,
                features,
                min_num_inliers,
                min_tri_angle,
            ) {
                return Some((first, second, g));
            }
        }
        None
    }

    fn initialize(
        &mut self,
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        features: &[ImageFeatures],
        tracks: &[Track],
        min_num_inliers: usize,
        min_tri_angle: f64,
    ) -> SfmStatus {
        let (first, second, geometry) =
            match self.find_initial_image_pair(index, features, min_num_inliers, min_tri_angle) {
                Some(v) => v,
                None => return SfmStatus::NoInitialPair,
            };

        self.cameras[first] = Some(CameraPose {
            r: Matrix3::identity(),
            t: Vector3::zeros(),
        });
        self.cameras[second] = Some(geometry.cam2_from_cam1.clone());
        self.reg_trials[first] += 1;
        self.reg_trials[second] += 1;

        self.triangulate_image(first, tracks, features, index);
        self.triangulate_image(second, tracks, features, index);

        self.adjust_global_bundle(
            tracks,
            features,
            LossType::Trivial,
            BA_GLOBAL_MAX_ITERATIONS,
        );
        self.normalize();
        self.filter_points3d(tracks, features);
        self.filter_frames(tracks);

        if self.num_registered() > 0
            && self
                .points
                .iter()
                .filter(|p| p.track_id != usize::MAX)
                .count()
                >= ABS_POSE_MIN_NUM_INLIERS
        {
            println!(
                "  Initialized with pair ({},{}) -> {} points",
                first,
                second,
                self.points
                    .iter()
                    .filter(|p| p.track_id != usize::MAX)
                    .count()
            );
            SfmStatus::Success
        } else {
            self.cameras[first] = None;
            self.cameras[second] = None;
            self.points.clear();
            self.rebuild_track_to_point();
            SfmStatus::BadInitialPair
        }
    }

    // -----------------------------------------------------------------
    //  Next image selection
    // -----------------------------------------------------------------
    fn find_next_images(&self, tracks: &[Track], features: &[ImageFeatures]) -> Vec<usize> {
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for img in 0..self.num_images {
            if self.cameras[img].is_some()
                || self.filtered[img]
                || self.reg_trials[img] >= MAX_REG_TRIALS
            {
                continue;
            }
            let visible = self.visible_points_count(tracks, img);
            if visible < ABS_POSE_MIN_NUM_INLIERS {
                continue;
            }
            let score = match IMAGE_SELECTION_METHOD {
                ImageSelectionMethod::MaxVisiblePointsNum => visible as f64,
                ImageSelectionMethod::MaxVisiblePointsRatio => {
                    let total = features[img].keypoints.len().max(1);
                    visible as f64 / total as f64
                }
                ImageSelectionMethod::MinUncertainty => {
                    let mut s = 0.0;
                    for (tid, &pidx) in &self.track_to_point {
                        let track = &tracks[*tid];
                        if track
                            .observations
                            .binary_search_by_key(&img, |&(im, _)| im)
                            .is_ok()
                        {
                            s += 1.0 / (1.0 + self.points[pidx].error);
                        }
                    }
                    s
                }
            };
            let penalty = if self.reg_trials[img] > 0 { 1e6 } else { 0.0 };
            scored.push((score - penalty, img));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, img)| img).collect()
    }

    // -----------------------------------------------------------------
    //  Image registration (structure-based PnP)
    // -----------------------------------------------------------------
    fn collect_2d3d(
        &self,
        tracks: &[Track],
        features: &[ImageFeatures],
        img: usize,
    ) -> Vec<(usize, Vector3<f64>, f64, f64)> {
        let mut corrs = Vec::new();
        for (tid, &pidx) in &self.track_to_point {
            if !self.point_is_valid(pidx) {
                continue;
            }
            let track = &tracks[*tid];
            if let Ok(pos) = track.observations.binary_search_by_key(&img, |&(im, _)| im) {
                let kp_idx = track.observations[pos].1;
                let kp = &features[img].keypoints[kp_idx];
                corrs.push((pidx, self.points[pidx].xyz, kp.x as f64, kp.y as f64));
            }
        }
        corrs
    }

    fn register_next_image(
        &mut self,
        img: usize,
        tracks: &[Track],
        features: &[ImageFeatures],
    ) -> bool {
        self.reg_trials[img] += 1;

        let corrs = self.collect_2d3d(tracks, features, img);
        if corrs.len() < ABS_POSE_MIN_NUM_INLIERS {
            return false;
        }

        let pts3d: Vec<Vector3<f64>> = corrs.iter().map(|c| c.1).collect();
        let pts2d: Vec<(f64, f64)> = corrs.iter().map(|c| (c.2, c.3)).collect();

        let f0 = self.focals[img];
        let k_inv = Matrix3::new(
            1.0 / f0,
            0.0,
            -self.cx() / f0,
            0.0,
            1.0 / f0,
            -self.cy() / f0,
            0.0,
            0.0,
            1.0,
        );

        let refine_focal = !self.has_prior_focal[img];
        let (pose, f_refined, _mask) = match estimate_absolute_pose(
            &pts3d,
            &pts2d,
            &k_inv,
            f0,
            self.cx(),
            self.cy(),
            ABS_POSE_MAX_ERROR_PX,
            refine_focal,
        ) {
            Some(v) => v,
            None => return false,
        };

        let inlier_ratio = {
            let mut inl = 0usize;
            for c in corrs.iter() {
                if let Some((u, v, _)) =
                    project_point(&pose.r, &pose.t, &c.1, f_refined, self.cx(), self.cy())
                {
                    let e2 = (u - c.2).powi(2) + (v - c.3).powi(2);
                    if e2 < ABS_POSE_MAX_ERROR_PX * ABS_POSE_MAX_ERROR_PX {
                        inl += 1;
                    }
                }
            }
            inl as f64 / corrs.len().max(1) as f64
        };

        if inlier_ratio < ABS_POSE_MIN_INLIER_RATIO {
            return false;
        }

        self.cameras[img] = Some(pose);
        self.focals[img] = f_refined;
        true
    }

    // -----------------------------------------------------------------
    //  Image registration (structure-less 2D-2D essential matrix)
    // -----------------------------------------------------------------
    fn register_structure_less_image(
        &mut self,
        img: usize,
        _tracks: &[Track],
        features: &[ImageFeatures],
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) -> bool {
        self.reg_trials[img] += 1;

        let mut best_ref: Option<(usize, usize)> = None; // (num_corrs, ref_img)
        for ref_img in self.registered_images() {
            let key = if img < ref_img {
                (img, ref_img)
            } else {
                (ref_img, img)
            };
            let n = index.get(&key).map(|v| v.len()).unwrap_or(0);
            if n >= 2 * ABS_POSE_MIN_NUM_INLIERS {
                if best_ref.map(|(bn, _)| n > bn).unwrap_or(true) {
                    best_ref = Some((n, ref_img));
                }
            }
        }

        let ref_img = match best_ref {
            Some((_, r)) => r,
            None => return false,
        };

        let key = if img < ref_img {
            (img, ref_img)
        } else {
            (ref_img, img)
        };
        let corrs = index.get(&key).cloned().unwrap_or_default();

        let mut pts_new = Vec::with_capacity(corrs.len());
        let mut pts_ref = Vec::with_capacity(corrs.len());
        for &(ka, kb) in &corrs {
            let (kn, kr) = if img < ref_img { (ka, kb) } else { (kb, ka) };
            let kpn = &features[img].keypoints[kn];
            let kpr = &features[ref_img].keypoints[kr];
            pts_new.push(Vector3::new(kpn.x as f64, kpn.y as f64, 1.0));
            pts_ref.push(Vector3::new(kpr.x as f64, kpr.y as f64, 1.0));
        }

        let (f_mat, _) = ransac_fundamental(
            &pts_new,
            &pts_ref,
            RANSAC_ITERATIONS,
            RANSAC_THRESHOLD_PX.powi(2),
        );
        let e_mat = essential_from_fundamental(&f_mat, &self.intrinsics);
        let (r1, r2, t_rel) = decompose_essential(&e_mat);

        let ref_cam = self.cameras[ref_img].clone().unwrap();
        let mut best_count = 0usize;
        let mut best_pose = (ref_cam.r, ref_cam.t);
        for &(r_rel_c, t_rel_c) in &[(r1, t_rel), (r1, -t_rel), (r2, t_rel), (r2, -t_rel)] {
            let r_w = r_rel_c * ref_cam.r;
            let t_w = r_rel_c * ref_cam.t + t_rel_c;
            let mut cnt = 0usize;
            for (pn, pr) in pts_new.iter().zip(pts_ref.iter()) {
                let proj_new = proj_matrix(&r_w, &t_w, &self.intrinsics.k);
                let proj_ref = proj_matrix(&ref_cam.r, &ref_cam.t, &self.intrinsics.k);
                if let Some(pt) =
                    triangulate_dlt_multi(&[proj_new.clone(), proj_ref.clone()], &[*pn, *pr])
                {
                    if (r_w * pt + t_w).z > 0.0 && (ref_cam.r * pt + ref_cam.t).z > 0.0 {
                        cnt += 1;
                    }
                }
            }
            if cnt > best_count {
                best_count = cnt;
                best_pose = (r_w, t_w);
            }
        }

        if best_count < pts_new.len() / 4 {
            return false;
        }

        let mut pose = CameraPose {
            r: best_pose.0,
            t: best_pose.1,
        };

        // Refine with triangulated 3D points (CAUCHY-style robust via inliers)
        let mut pts3d: Vec<Vector3<f64>> = Vec::new();
        let mut pts2d: Vec<(f64, f64)> = Vec::new();
        for (pn, pr) in pts_new.iter().zip(pts_ref.iter()) {
            let proj_new = proj_matrix(&pose.r, &pose.t, &self.intrinsics.k);
            let proj_ref = proj_matrix(&ref_cam.r, &ref_cam.t, &self.intrinsics.k);
            if let Some(pt) =
                triangulate_dlt_multi(&[proj_new.clone(), proj_ref.clone()], &[*pn, *pr])
            {
                if (pose.r * pt + pose.t).z > 0.0 {
                    pts3d.push(pt);
                    pts2d.push((pn.x, pn.y));
                }
            }
        }
        if pts3d.len() >= ABS_POSE_MIN_NUM_INLIERS {
            let f0 = self.focals[img];
            let (refined, _) =
                refine_absolute_pose(&pts3d, &pts2d, &pose, f0, self.cx(), self.cy(), false);
            pose = refined;
        }

        self.cameras[img] = Some(pose);
        true
    }

    // -----------------------------------------------------------------
    //  Triangulation
    // -----------------------------------------------------------------
    fn triangulate_image(
        &mut self,
        img: usize,
        tracks: &[Track],
        features: &[ImageFeatures],
        _index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) {
        for (track_id, track) in tracks.iter().enumerate() {
            if self.track_to_point.contains_key(&track_id) {
                continue;
            }
            if track
                .observations
                .binary_search_by_key(&img, |&(im, _)| im)
                .is_err()
            {
                continue;
            }
            let mut obs: Vec<(usize, usize)> = Vec::new();
            for &(im, kp) in &track.observations {
                if self.cameras[im].is_some() {
                    obs.push((im, kp));
                }
            }
            if obs.len() < 2 {
                continue;
            }
            if let Some(pt) = self.triangulate_track(track_id, &obs, features) {
                let pidx = self.points.len();
                self.points.push(Point3D {
                    xyz: pt,
                    track_id,
                    error: 0.0,
                });
                self.track_to_point.insert(track_id, pidx);
            }
        }
    }

    fn triangulate_track(
        &self,
        _track_id: usize,
        obs: &[(usize, usize)],
        features: &[ImageFeatures],
    ) -> Option<Vector3<f64>> {
        if obs.len() < 2 {
            return None;
        }
        let mut projs = Vec::with_capacity(obs.len());
        let mut pts = Vec::with_capacity(obs.len());
        for &(im, kp) in obs {
            let cam = self.cameras[im].as_ref()?;
            let f = self.focals[im];
            let k = Matrix3::new(f, 0.0, self.cx(), 0.0, f, self.cy(), 0.0, 0.0, 1.0);
            let kp_pt = &features[im].keypoints[kp];
            projs.push(proj_matrix(&cam.r, &cam.t, &k));
            pts.push(Vector3::new(kp_pt.x as f64, kp_pt.y as f64, 1.0));
        }
        let pt = triangulate_dlt_multi(&projs, &pts)?;
        if !pt.x.is_finite() || !pt.y.is_finite() || !pt.z.is_finite() {
            return None;
        }
        for &(im, _) in obs.iter() {
            let cam = self.cameras[im].as_ref()?;
            if (cam.r * pt + cam.t).z <= 1e-8 {
                return None;
            }
        }
        if obs.len() >= 2 {
            let cam0 = self.cameras[obs[0].0].as_ref()?;
            let cam1 = self.cameras[obs[1].0].as_ref()?;
            let angle = calc_triangulation_angle(&cam0.r, &cam0.t, &cam1.r, &cam1.t, &pt);
            if angle < CREATE_MAX_ANGLE_ERROR_DEG.to_radians() {
                return None;
            }
        }
        Some(pt)
    }

    fn complete_tracks(&mut self, tracks: &[Track], features: &[ImageFeatures]) -> usize {
        let mut changed = 0usize;
        for pidx in 0..self.points.len() {
            if !self.point_is_valid(pidx) {
                continue;
            }
            let track_id = self.points[pidx].track_id;
            let track = &tracks[track_id];
            let obs: Vec<(usize, usize)> = track
                .observations
                .iter()
                .filter(|(im, _)| self.cameras[*im].is_some())
                .copied()
                .collect();
            if obs.len() < 3 {
                continue;
            }
            if let Some(pt) = self.triangulate_track(track_id, &obs, features) {
                let mut errs: Vec<f64> = Vec::new();
                for &(im, kp) in &obs {
                    let cam = self.cameras[im].as_ref().unwrap();
                    let f = self.focals[im];
                    if let Some((u, v, _)) =
                        project_point(&cam.r, &cam.t, &pt, f, self.cx(), self.cy())
                    {
                        let kd = &features[im].keypoints[kp];
                        errs.push(((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt());
                    }
                }
                if errs.is_empty() {
                    continue;
                }
                errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_err = errs[errs.len() / 2];
                if median_err > COMPLETE_MAX_REPROJ_ERROR_PX {
                    continue;
                }
                if (pt - self.points[pidx].xyz).norm() > 1e-6 {
                    self.points[pidx].xyz = pt;
                    changed += 1;
                }
            }
        }
        changed
    }

    fn merge_tracks(&mut self, tracks: &[Track], features: &[ImageFeatures]) -> usize {
        let mut order: Vec<usize> = (0..self.points.len())
            .filter(|&p| self.point_is_valid(p))
            .collect();
        if order.len() < 2 {
            return 0;
        }
        order.sort_by(|&a, &b| {
            self.points[a]
                .xyz
                .x
                .partial_cmp(&self.points[b].xyz.x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut norm: Vec<f64> = order.iter().map(|&p| self.points[p].xyz.norm()).collect();
        norm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = norm[norm.len() / 2];
        let spatial_threshold = (med * 0.05).max(1e-3);

        let mut merged = 0usize;
        for i in 0..order.len() {
            let a = order[i];
            if !self.point_is_valid(a) {
                continue;
            }
            for j in (i + 1)..order.len() {
                let b = order[j];
                if !self.point_is_valid(b) {
                    continue;
                }
                if self.points[b].xyz.x - self.points[a].xyz.x > spatial_threshold {
                    break;
                }
                let pa = self.points[a].xyz;
                let pb = self.points[b].xyz;
                if (pa - pb).norm() > spatial_threshold {
                    continue;
                }
                let ta = self.points[a].track_id;
                let tb = self.points[b].track_id;
                let shared = tracks[ta].observations.iter().any(|(im, _)| {
                    tracks[tb]
                        .observations
                        .binary_search_by_key(im, |&(x, _)| x)
                        .is_ok()
                });
                if shared {
                    continue;
                }

                let la = tracks[ta].observations.len() as f64;
                let lb = tracks[tb].observations.len() as f64;
                let pm = (pa * la + pb * lb) / (la + lb);

                let mut errs: Vec<f64> = Vec::new();
                for &(im, kp) in tracks[ta]
                    .observations
                    .iter()
                    .chain(tracks[tb].observations.iter())
                {
                    if let Some(cam) = &self.cameras[im] {
                        let f = self.focals[im];
                        if let Some((u, v, _)) =
                            project_point(&cam.r, &cam.t, &pm, f, self.cx(), self.cy())
                        {
                            let kd = &features[im].keypoints[kp];
                            errs.push(
                                ((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt(),
                            );
                        }
                    }
                }
                if errs.is_empty() {
                    continue;
                }
                errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if errs[errs.len() / 2] > MERGE_MAX_REPROJ_ERROR_PX {
                    continue;
                }

                // always keep point `a`; drop `b` and remap its track to `a`
                let drop_track = self.points[b].track_id;
                self.points[a].xyz = pm;
                self.track_to_point.insert(drop_track, a);
                self.invalidate_point(b);
                merged += 1;
            }
        }
        merged
    }

    // -----------------------------------------------------------------
    //  Filtering
    // -----------------------------------------------------------------
    fn filter_points3d(&mut self, tracks: &[Track], features: &[ImageFeatures]) -> usize {
        let mut removed = 0usize;
        for pidx in 0..self.points.len() {
            if self.points[pidx].track_id == usize::MAX {
                continue;
            }
            let track_id = self.points[pidx].track_id;
            let track = &tracks[track_id];
            let pt = self.points[pidx].xyz;

            let mut errs: Vec<f64> = Vec::new();
            let mut behind = 0usize;
            let mut reg_ims: Vec<usize> = Vec::new();
            for &(im, kp) in &track.observations {
                if self.cameras[im].is_none() {
                    continue;
                }
                reg_ims.push(im);
                let cam = self.cameras[im].as_ref().unwrap();
                let f = self.focals[im];
                match project_point(&cam.r, &cam.t, &pt, f, self.cx(), self.cy()) {
                    Some((u, v, _)) => {
                        let kd = &features[im].keypoints[kp];
                        errs.push(((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt());
                    }
                    None => behind += 1,
                }
            }

            let valid_obs = errs.len();
            if valid_obs < 2 {
                self.invalidate_point(pidx);
                removed += 1;
                continue;
            }

            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_err = errs[errs.len() / 2];
            self.points[pidx].error = median_err;

            if behind > valid_obs {
                self.invalidate_point(pidx);
                removed += 1;
                continue;
            }

            if reg_ims.len() >= 2 {
                let c0 = self.cameras[reg_ims[0]].as_ref().unwrap();
                let c1 = self.cameras[reg_ims[1]].as_ref().unwrap();
                let angle = calc_triangulation_angle(&c0.r, &c0.t, &c1.r, &c1.t, &pt);
                if angle < FILTER_MIN_TRI_ANGLE_DEG.to_radians() {
                    self.invalidate_point(pidx);
                    removed += 1;
                    continue;
                }
            }

            if median_err > FILTER_MAX_REPROJ_ERROR_PX * 2.0 {
                self.invalidate_point(pidx);
                removed += 1;
            }
        }
        if removed > 0 {
            self.rebuild_track_to_point();
        }
        removed
    }

    fn filter_negative_depth(&mut self, tracks: &[Track]) -> usize {
        let mut removed = 0usize;
        for pidx in 0..self.points.len() {
            if self.points[pidx].track_id == usize::MAX {
                continue;
            }
            let track_id = self.points[pidx].track_id;
            let track = &tracks[track_id];
            let pt = self.points[pidx].xyz;
            let mut negative = 0usize;
            let mut total = 0usize;
            for &(im, _) in &track.observations {
                if let Some(cam) = &self.cameras[im] {
                    total += 1;
                    if (cam.r * pt + cam.t).z <= 1e-8 {
                        negative += 1;
                    }
                }
            }
            if negative > 0 && negative > total / 2 {
                self.invalidate_point(pidx);
                removed += 1;
            }
        }
        if removed > 0 {
            self.rebuild_track_to_point();
        }
        removed
    }

    fn filter_frames(&mut self, _tracks: &[Track]) -> usize {
        let mut removed = 0usize;
        if self.num_registered() < 20 {
            return 0;
        }
        let base_f = self.intrinsics.k[(0, 0)];
        for img in 0..self.num_images {
            if self.cameras[img].is_none() {
                continue;
            }
            let ratio = self.focals[img] / base_f.max(1e-6);
            if ratio < MIN_FOCAL_LENGTH_RATIO || ratio > MAX_FOCAL_LENGTH_RATIO {
                self.cameras[img] = None;
                self.filtered[img] = true;
                removed += 1;
            }
        }
        removed
    }

    fn update_point_errors(&mut self, tracks: &[Track], features: &[ImageFeatures]) {
        for pidx in 0..self.points.len() {
            if self.points[pidx].track_id == usize::MAX {
                continue;
            }
            let track_id = self.points[pidx].track_id;
            let track = &tracks[track_id];
            let pt = self.points[pidx].xyz;
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for &(im, kp) in &track.observations {
                if let Some(cam) = &self.cameras[im] {
                    let f = self.focals[im];
                    if let Some((u, v, _)) =
                        project_point(&cam.r, &cam.t, &pt, f, self.cx(), self.cy())
                    {
                        let kd = &features[im].keypoints[kp];
                        sum += ((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt();
                        cnt += 1;
                    }
                }
            }
            self.points[pidx].error = if cnt > 0 { sum / cnt as f64 } else { f64::MAX };
        }
    }

    // -----------------------------------------------------------------
    //  Bundle adjustment (local / global)
    // -----------------------------------------------------------------
    fn adjust_global_bundle(
        &mut self,
        tracks: &[Track],
        features: &[ImageFeatures],
        loss: LossType,
        max_iterations: usize,
    ) {
        self.filter_negative_depth(tracks);
        let active = self.registered_images();
        if active.is_empty() {
            return;
        }
        let fixed = vec![active[0]];
        bundle_adjust(
            self,
            tracks,
            features,
            &active,
            &fixed,
            loss,
            max_iterations,
        );
    }

    fn find_local_bundle(&self, tracks: &[Track], img: usize, num: usize) -> Vec<usize> {
        let mut scores: Vec<(usize, usize)> = Vec::new();
        for other in self.registered_images() {
            if other == img {
                continue;
            }
            let mut shared = 0usize;
            for (tid, _pidx) in &self.track_to_point {
                let track = &tracks[*tid];
                let in_a = track
                    .observations
                    .binary_search_by_key(&img, |&(im, _)| im)
                    .is_ok();
                let in_b = track
                    .observations
                    .binary_search_by_key(&other, |&(im, _)| im)
                    .is_ok();
                if in_a && in_b {
                    shared += 1;
                }
            }
            scores.push((shared, other));
        }
        scores.sort_by(|a, b| b.0.cmp(&a.0));
        let mut local = vec![img];
        for (_, o) in scores.into_iter().take(num - 1) {
            local.push(o);
        }
        local
    }

    fn adjust_local_bundle(&mut self, tracks: &[Track], features: &[ImageFeatures], img: usize) {
        let local = self.find_local_bundle(tracks, img, BA_LOCAL_NUM_IMAGES);
        if local.len() < 3 {
            return;
        }
        let anchor = *local.iter().min().unwrap();
        let fixed = vec![anchor];
        bundle_adjust(
            self,
            tracks,
            features,
            &local,
            &fixed,
            LossType::SoftL1,
            BA_LOCAL_MAX_ITERATIONS,
        );
    }

    fn iterative_refinement(
        &mut self,
        tracks: &[Track],
        features: &[ImageFeatures],
        global: bool,
        img: Option<usize>,
    ) {
        for _ in 0..MAX_REFINEMENTS {
            let before = self
                .points
                .iter()
                .filter(|p| p.track_id != usize::MAX)
                .count();
            if global {
                self.adjust_global_bundle(
                    tracks,
                    features,
                    LossType::Trivial,
                    BA_GLOBAL_MAX_ITERATIONS,
                );
            } else if let Some(i) = img {
                self.adjust_local_bundle(tracks, features, i);
            }
            self.complete_tracks(tracks, features);
            self.merge_tracks(tracks, features);
            let removed = self.filter_points3d(tracks, features);
            let after = self
                .points
                .iter()
                .filter(|p| p.track_id != usize::MAX)
                .count();
            let change = (before as f64 - after as f64).abs() / before.max(1) as f64;
            if removed == 0 && change < MAX_REFINEMENT_CHANGE {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------
//  Bundle adjustment core (Schur-complement Levenberg-Marquardt)
// ---------------------------------------------------------------------
fn bundle_adjust(
    sfm: &mut Sfm,
    tracks: &[Track],
    features: &[ImageFeatures],
    active_cams: &[usize],
    fixed_cams: &[usize],
    loss: LossType,
    max_iterations: usize,
) {
    let nc = active_cams.len();
    if nc < 2 {
        return;
    }

    let mut cam_block: Vec<Option<usize>> = vec![None; sfm.num_images];
    for (ci, &im) in active_cams.iter().enumerate() {
        cam_block[im] = Some(ci);
    }
    let mut fixed = vec![false; nc];
    for &fim in fixed_cams {
        if let Some(ci) = cam_block[fim] {
            fixed[ci] = true;
        }
    }

    let mut pts: Vec<usize> = Vec::new();
    for (pidx, pt) in sfm.points.iter().enumerate() {
        if pt.track_id == usize::MAX {
            continue;
        }
        if !pt.xyz.x.is_finite() || !pt.xyz.y.is_finite() || !pt.xyz.z.is_finite() {
            continue;
        }
        let track = &tracks[pt.track_id];
        let cnt = track
            .observations
            .iter()
            .filter(|(im, _)| cam_block[*im].is_some())
            .count();
        if cnt >= 2 {
            pts.push(pidx);
        }
    }
    let np = pts.len();
    if np == 0 {
        return;
    }
    let ncp = nc * 6;
    let cx = sfm.cx();
    let cy = sfm.cy();

    let mut lambda = 1e-6f64;
    let mut best_cost = f64::MAX;

    for _it in 0..max_iterations {
        let mut u = vec![na::SMatrix::<f64, 6, 6>::zeros(); nc];
        let mut ec = vec![na::SVector::<f64, 6>::zeros(); nc];
        let mut wpairs: Vec<Vec<(usize, na::SMatrix<f64, 6, 3>)>> = vec![Vec::new(); nc];
        let mut vpinv = vec![na::SMatrix::<f64, 3, 3>::zeros(); np];
        let mut ep = vec![na::SVector::<f64, 3>::zeros(); np];
        let mut total_cost = 0.0f64;
        let mut nobs = 0usize;

        for (pi, &pidx) in pts.iter().enumerate() {
            let pt = sfm.points[pidx].xyz;
            let track = &tracks[sfm.points[pidx].track_id];
            for &(im, kp) in &track.observations {
                let ci = match cam_block[im] {
                    Some(ci) => ci,
                    None => continue,
                };
                let cam = sfm.cameras[im].as_ref().unwrap();
                let f = sfm.focals[im];
                let xc = cam.r * pt + cam.t;
                if xc.z < 1e-8 {
                    continue;
                }
                let zi = 1.0 / xc.z;
                let up = f * xc.x * zi + cx;
                let vp = f * xc.y * zi + cy;
                let kd = &features[im].keypoints[kp];
                let ex = up - kd.x as f64;
                let ey = vp - kd.y as f64;
                let e2 = ex * ex + ey * ey;
                let w = loss.weight(e2);
                total_cost += 0.5 * w * e2;
                nobs += 1;

                let du = (f * zi, 0.0, -f * xc.x * zi * zi);
                let dv = (0.0, f * zi, -f * xc.y * zi * zi);
                let sk = skew_symmetric(&(cam.r * pt));
                let mut jc = na::SMatrix::<f64, 2, 6>::zeros();
                for k in 0..3 {
                    let col = sk.column(k);
                    jc[(0, k)] = -(du.0 * col.x + du.1 * col.y + du.2 * col.z);
                    jc[(1, k)] = -(dv.0 * col.x + dv.1 * col.y + dv.2 * col.z);
                }
                jc[(0, 3)] = du.0;
                jc[(0, 4)] = du.1;
                jc[(0, 5)] = du.2;
                jc[(1, 3)] = dv.0;
                jc[(1, 4)] = dv.1;
                jc[(1, 5)] = dv.2;

                let mut jp = na::SMatrix::<f64, 2, 3>::zeros();
                let r = &cam.r;
                jp[(0, 0)] = du.0 * r[(0, 0)] + du.1 * r[(1, 0)] + du.2 * r[(2, 0)];
                jp[(0, 1)] = du.0 * r[(0, 1)] + du.1 * r[(1, 1)] + du.2 * r[(2, 1)];
                jp[(0, 2)] = du.0 * r[(0, 2)] + du.1 * r[(1, 2)] + du.2 * r[(2, 2)];
                jp[(1, 0)] = dv.0 * r[(0, 0)] + dv.1 * r[(1, 0)] + dv.2 * r[(2, 0)];
                jp[(1, 1)] = dv.0 * r[(0, 1)] + dv.1 * r[(1, 1)] + dv.2 * r[(2, 1)];
                jp[(1, 2)] = dv.0 * r[(0, 2)] + dv.1 * r[(1, 2)] + dv.2 * r[(2, 2)];

                let res = na::SVector::<f64, 2>::new(w * ex, w * ey);
                u[ci] += jc.transpose() * jc * w;
                ec[ci] += jc.transpose() * res;
                wpairs[ci].push((pi, jc.transpose() * jp * w));
                vpinv[pi] += jp.transpose() * jp * w;
                ep[pi] += jp.transpose() * res;
            }
        }

        let cost = total_cost / nobs.max(1) as f64;
        if _it == 0 {
            best_cost = cost;
        }

        for pi in 0..np {
            vpinv[pi] = vpinv[pi].try_inverse().unwrap_or(na::SMatrix::zeros());
        }

        let mut s: na::DMatrix<f64> = na::DMatrix::zeros(ncp, ncp);
        let mut es = na::DVector::<f64>::zeros(ncp);
        for ci in 0..nc {
            let cr = ci * 6;
            for r in 0..6 {
                for c in 0..6 {
                    s[(cr + r, cr + c)] += u[ci][(r, c)];
                }
            }
            for r in 0..6 {
                es[cr + r] += ec[ci][r];
            }
            for (pi, w) in &wpairs[ci] {
                for wr in 0..6 {
                    for wc in 0..6 {
                        let mut sum = 0.0;
                        for k in 0..3 {
                            for l in 0..3 {
                                sum += w[(wr, k)] * vpinv[*pi][(k, l)] * w[(wc, l)];
                            }
                        }
                        s[(cr + wr, cr + wc)] -= sum;
                    }
                    let mut sum = 0.0;
                    for k in 0..3 {
                        for l in 0..3 {
                            sum += w[(wr, k)] * vpinv[*pi][(k, l)] * ep[*pi][l];
                        }
                    }
                    es[cr + wr] -= sum;
                }
            }
        }
        for ci in 0..nc {
            if fixed[ci] {
                let cr = ci * 6;
                for d in 0..6 {
                    s[(cr + d, cr + d)] = 1e10;
                    es[cr + d] = 0.0;
                }
            }
        }
        for d in 0..ncp {
            s[(d, d)] += lambda * s[(d, d)].abs().max(1e-6);
        }
        let delta_c = match s.clone().lu().solve(&(-es)) {
            Some(d) => d,
            None => {
                lambda *= 3.0;
                continue;
            }
        };

        let backup_cams: Vec<CameraPose> = active_cams
            .iter()
            .map(|&im| sfm.cameras[im].clone().unwrap())
            .collect();
        let backup_pts: Vec<Vector3<f64>> = pts.iter().map(|&pidx| sfm.points[pidx].xyz).collect();

        for ci in 0..nc {
            if fixed[ci] {
                continue;
            }
            let cam = sfm.cameras[active_cams[ci]].as_mut().unwrap();
            let cr = ci * 6;
            let dw = Vector3::new(delta_c[cr], delta_c[cr + 1], delta_c[cr + 2]);
            let dt = Vector3::new(delta_c[cr + 3], delta_c[cr + 4], delta_c[cr + 5]);
            cam.r = so3_exp(&dw) * cam.r;
            cam.t += dt;
        }
        for pi in 0..np {
            let pidx = pts[pi];
            let mut dp = na::SVector::<f64, 3>::zeros();
            for ci in 0..nc {
                for (pj, w) in &wpairs[ci] {
                    if *pj == pi {
                        for rc in 0..3 {
                            for row in 0..6 {
                                dp[rc] += w[(row, rc)] * delta_c[ci * 6 + row];
                            }
                        }
                    }
                }
            }
            let mut corr = na::SVector::<f64, 3>::zeros();
            for r in 0..3 {
                for c in 0..3 {
                    corr[r] += vpinv[pi][(r, c)] * (-ep[pi][c] - dp[c]);
                }
            }
            sfm.points[pidx].xyz += Vector3::new(corr[0], corr[1], corr[2]);
        }

        let mut new_cost = 0.0;
        let mut nn = 0usize;
        for &pidx in &pts {
            let pt = sfm.points[pidx].xyz;
            let track = &tracks[sfm.points[pidx].track_id];
            for &(im, kp) in &track.observations {
                if cam_block[im].is_none() {
                    continue;
                }
                let cam = sfm.cameras[im].as_ref().unwrap();
                let f = sfm.focals[im];
                let xc = cam.r * pt + cam.t;
                if xc.z > 1e-8 {
                    let up = f * xc.x / xc.z + cx;
                    let vp = f * xc.y / xc.z + cy;
                    let kd = &features[im].keypoints[kp];
                    let e2 = (up - kd.x as f64).powi(2) + (vp - kd.y as f64).powi(2);
                    new_cost += 0.5 * loss.weight(e2) * e2;
                    nn += 1;
                }
            }
        }
        let new_cost = new_cost / nn.max(1) as f64;

        if new_cost < best_cost {
            best_cost = new_cost;
            lambda = (lambda * 0.5).max(1e-12);
        } else {
            for (ci, &im) in active_cams.iter().enumerate() {
                sfm.cameras[im] = Some(backup_cams[ci].clone());
            }
            for (pi, &pidx) in pts.iter().enumerate() {
                sfm.points[pidx].xyz = backup_pts[pi];
            }
            lambda *= 3.0;
            if lambda > 1e12 {
                break;
            }
        }
        if lambda < 1e-12 {
            break;
        }
    }
}

// ---------------------------------------------------------------------
//  Reconstruct (top-level control flow)
// ---------------------------------------------------------------------
fn reconstruct_sub_model(
    sfm: &mut Sfm,
    index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
    min_num_inliers: usize,
    min_tri_angle: f64,
) -> SfmStatus {
    let status = sfm.initialize(index, features, tracks, min_num_inliers, min_tri_angle);
    if status != SfmStatus::Success {
        return status;
    }

    let mut prev_success = true;
    loop {
        let mut success = false;
        let mut last_img = None;

        for structure_less in [false, true] {
            let next = sfm.find_next_images(tracks, features);
            for img in next {
                let ok = if structure_less {
                    sfm.register_structure_less_image(img, tracks, features, index)
                } else {
                    sfm.register_next_image(img, tracks, features)
                };
                if ok {
                    success = true;
                    last_img = Some(img);
                    break;
                }
            }
            if success {
                break;
            }
        }

        if success {
            let img = last_img.unwrap();
            sfm.triangulate_image(img, tracks, features, index);
            sfm.iterative_refinement(tracks, features, false, Some(img));
            sfm.iterative_refinement(tracks, features, true, None);
            let registered = sfm.num_registered();
            println!(
                "  [{:3}/{:3}] registered {} ({})  {} points",
                registered,
                sfm.num_images,
                img,
                img_file_stem(&paths[img]),
                sfm.points
                    .iter()
                    .filter(|p| p.track_id != usize::MAX)
                    .count(),
            );
        }

        let shared = 0usize;
        if shared >= MAX_MODEL_OVERLAP {
            break;
        }

        if !success && prev_success {
            let mut diag: Vec<(usize, usize)> = (0..sfm.num_images)
                .filter(|&u| sfm.cameras[u].is_none())
                .map(|u| (u, sfm.visible_points_count(tracks, u)))
                .collect();
            diag.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = diag
                .iter()
                .take(8)
                .map(|(u, c)| format!("{}:{}", u, c))
                .collect();
            println!(
                "  Stalled; best unregistered images (idx:visible) {}",
                top.join(" ")
            );
            sfm.iterative_refinement(tracks, features, true, None);
            break;
        }

        if !success && !prev_success {
            break;
        }

        prev_success = success;
    }

    sfm.update_point_errors(tracks, features);
    SfmStatus::Success
}

fn reconstruct(
    sfm: &mut Sfm,
    index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
) {
    let mut min_num_inliers = INIT_MIN_NUM_INLIERS;
    let mut min_tri_angle = INIT_MIN_TRI_ANGLE_DEG.to_radians();

    for _round in 0..2 {
        for _trial in 0..INIT_NUM_TRIALS {
            let status = reconstruct_sub_model(
                sfm,
                index,
                features,
                tracks,
                paths,
                min_num_inliers,
                min_tri_angle,
            );
            match status {
                SfmStatus::Success => {
                    sfm.iterative_refinement(tracks, features, true, None);
                    sfm.update_point_errors(tracks, features);
                    return;
                }
                SfmStatus::BadInitialPair => {
                    sfm.cameras = vec![None; sfm.num_images];
                    sfm.points.clear();
                    sfm.rebuild_track_to_point();
                }
                SfmStatus::NoInitialPair => break,
            }
        }
        min_num_inliers /= 2;
        min_tri_angle /= 2.0;
    }
}

fn build_correspondence_index(matches: &[Match]) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
    let mut map: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for m in matches {
        let (a, kpa, b, kpb) = if m.img1 < m.img2 {
            (m.img1, m.kp1, m.img2, m.kp2)
        } else {
            (m.img2, m.kp2, m.img1, m.kp1)
        };
        map.entry((a, b)).or_default().push((kpa, kpb));
    }
    map
}

fn run_incremental_sfm(
    matches: &[Match],
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
) {
    let first_img = image::open(&paths[0]).expect("Failed to open first image");
    let (w, h) = (first_img.width(), first_img.height());
    let intrinsics = CameraIntrinsics::from_image_with_exif(&paths[0], w, h);
    println!(
        "Camera intrinsics: f={:.0}  cx={:.0}  cy={:.0}  prior={}",
        intrinsics.k[(0, 0)],
        intrinsics.k[(0, 2)],
        intrinsics.k[(1, 2)],
        intrinsics.has_prior_focal,
    );

    let index = build_correspondence_index(matches);
    println!("Correspondence graph: {} image pairs", index.len(),);

    let mut sfm = Sfm::new(paths.len(), intrinsics);
    reconstruct(&mut sfm, &index, features, tracks, paths);

    let registered = sfm.num_registered();
    println!(
        "Final: {}/{} images registered, {} points",
        registered,
        sfm.num_images,
        sfm.points
            .iter()
            .filter(|p| p.track_id != usize::MAX)
            .count(),
    );

    let fixed = sanitize_camera_centers(&mut sfm);
    if fixed > 0 {
        println!("  Sanitized {} wild camera(s)", fixed);
    }

    let (e0, e1, e2, ratio) = pca_check_cameras(&sfm);
    println!(
        "  Camera PCA: λ=({:.1},{:.1},{:.1}) ratio={:.3} {}",
        e0,
        e1,
        e2,
        ratio,
        if ratio < 0.1 {
            "FLAT DISTRIBUTION"
        } else {
            "3D spread"
        }
    );

    save_sparse_reconstruction(&sfm, features, tracks, paths);
}

// ---------------------------------------------------------------------
//  Post-processing / export
// ---------------------------------------------------------------------
fn sanitize_camera_centers(sfm: &mut Sfm) -> usize {
    let centers: Vec<(usize, Vector3<f64>)> = sfm
        .cameras
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|c| (i, camera_center(&c.r, &c.t))))
        .collect();
    if centers.len() < 3 {
        return 0;
    }
    let n = centers.len();
    let mut xs: Vec<f64> = centers.iter().map(|(_, c)| c.x).collect();
    let mut ys: Vec<f64> = centers.iter().map(|(_, c)| c.y).collect();
    let mut zs: Vec<f64> = centers.iter().map(|(_, c)| c.z).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = Vector3::new(xs[n / 2], ys[n / 2], zs[n / 2]);

    let mut dists: Vec<f64> = centers.iter().map(|(_, c)| (*c - med).norm()).collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_dist = dists[n / 2].max(1e-3);
    let threshold = (med_dist * 10.0 + 1.0).max(10.0);

    let valid_poses: Vec<CameraPose> = centers
        .iter()
        .filter(|(_, c)| (*c - med).norm() <= threshold)
        .map(|(i, _)| sfm.cameras[*i].clone().unwrap())
        .collect();
    if valid_poses.is_empty() {
        return 0;
    }

    let mut fixed = 0usize;
    for (i, c) in &centers {
        if (*c - med).norm() > threshold {
            let nearest = valid_poses
                .iter()
                .min_by(|a, b| {
                    let da = (camera_center(&a.r, &a.t) - *c).norm();
                    let db = (camera_center(&b.r, &b.t) - *c).norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            sfm.cameras[*i] = Some(nearest.clone());
            fixed += 1;
        }
    }
    fixed
}

fn pca_check_cameras(sfm: &Sfm) -> (f64, f64, f64, f64) {
    let mut centers = Vec::new();
    for cam in sfm.cameras.iter().flatten() {
        centers.push(cam.r.transpose() * (-cam.t));
    }
    if centers.len() < 3 {
        return (1.0, 1.0, 1.0, 1.0);
    }
    let n = centers.len() as f64;
    let mean = centers.iter().fold(Vector3::zeros(), |a, b| a + *b) / n;
    let mut cov = Matrix3::zeros();
    for c in &centers {
        let d = c - mean;
        cov += d * d.transpose();
    }
    cov /= n;
    let eig = cov.symmetric_eigen();
    let mut ev: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let ratio = if ev[0] > 1e-10 { ev[2] / ev[0] } else { 1.0 };
    (ev[0], ev[1], ev[2], ratio)
}

fn ransac_remove_planes(points: &mut Vec<(Vector3<f64>, u8, u8, u8)>) -> usize {
    if points.len() < 100 {
        return 0;
    }
    let mut zs: Vec<f64> = points.iter().map(|(p, _, _, _)| p.z).collect();
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let z_lo = zs[zs.len() / 10];
    let z_hi = zs[9 * zs.len() / 10];
    let z_spread = z_hi - z_lo;
    let table_z = z_lo + z_spread * 0.25;

    let before = points.len();
    points.retain(|(pt, _, _, _)| pt.z > table_z);
    let removed = before - points.len();
    if removed > 0 {
        println!("  Table filter: {} pts removed (z<{:.1})", removed, table_z);
    }
    removed
}

fn filter_spatial_outliers(points: &mut Vec<(Vector3<f64>, u8, u8, u8)>) -> usize {
    let n = points.len();
    if n < 20 {
        return 0;
    }
    let mut xs: Vec<f64> = points.iter().map(|(p, _, _, _)| p.x).collect();
    let mut ys: Vec<f64> = points.iter().map(|(p, _, _, _)| p.y).collect();
    let mut zs: Vec<f64> = points.iter().map(|(p, _, _, _)| p.z).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = Vector3::new(xs[n / 2], ys[n / 2], zs[n / 2]);

    let mut dists: Vec<f64> = points.iter().map(|(p, _, _, _)| (p - med).norm()).collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_dist = dists[n / 2];

    let mut devs: Vec<f64> = dists.iter().map(|&d| (d - med_dist).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = devs[n / 2].max(1e-3);

    let threshold = med_dist + 10.0 * mad;
    let before = points.len();
    points.retain(|(p, _, _, _)| (p - med).norm() <= threshold);
    let removed = before - points.len();
    if removed > 0 {
        println!(
            "  Outlier filter: {} pts removed (dist > {:.1})",
            removed, threshold
        );
    }
    removed
}

fn save_sparse_reconstruction(
    sfm: &Sfm,
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
) {
    let num_registered = sfm.num_registered();
    println!("\n============ Final Reconstruction ============");
    println!("Cameras: {}/{} registered", num_registered, sfm.num_images);

    let mut valid_pts: Vec<(Vector3<f64>, u8, u8, u8)> = Vec::new();
    let mut image_cache: HashMap<usize, RgbImage> = HashMap::new();

    for pt in sfm.points.iter() {
        if pt.track_id == usize::MAX || !pt.xyz.x.is_finite() || pt.xyz.norm() < 0.01 {
            continue;
        }
        let track = &tracks[pt.track_id];
        let mut r = 128u8;
        let mut g = 128u8;
        let mut b = 128u8;
        for &(im, kp) in &track.observations {
            if sfm.cameras[im].is_some() {
                if !image_cache.contains_key(&im) {
                    if let Ok(img) = image::open(&paths[im]) {
                        image_cache.insert(im, img.to_rgb8());
                    }
                }
                if let Some(rgb) = image_cache.get(&im) {
                    let kp_data = &features[im].keypoints[kp];
                    let px = rgb.get_pixel(kp_data.x.round() as u32, kp_data.y.round() as u32);
                    r = px[0];
                    g = px[1];
                    b = px[2];
                }
                break;
            }
        }
        valid_pts.push((pt.xyz, r, g, b));
    }

    let _ = filter_spatial_outliers(&mut valid_pts);
    ransac_remove_planes(&mut valid_pts);

    if valid_pts.is_empty() {
        println!("No valid 3D points");
        return;
    }

    let mut ply = String::new();
    ply.push_str("ply\nformat ascii 1.0\n");
    ply.push_str(&format!(
        "element vertex {}\n",
        valid_pts.len() + num_registered
    ));
    ply.push_str("property float x\nproperty float y\nproperty float z\n");
    ply.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    ply.push_str("end_header\n");
    for (pt, r, g, b) in &valid_pts {
        ply.push_str(&format!("{} {} {} {} {} {}\n", pt.x, pt.y, pt.z, r, g, b));
    }
    for cam in sfm.cameras.iter().flatten() {
        let center = cam.r.transpose() * (-cam.t);
        ply.push_str(&format!("{} {} {} 255 0 0\n", center.x, center.y, center.z));
    }

    let mut mn = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut mx = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
    let mut sm = Vector3::zeros();
    for (pt, _, _, _) in &valid_pts {
        mn = mn.inf(pt);
        mx = mx.sup(pt);
        sm += pt;
    }
    let centroid = sm / valid_pts.len() as f64;

    fs::write("sparse.ply", &ply).unwrap();

    let mut jposes: Vec<serde_json::Value> = Vec::new();
    for (i, cam) in sfm.cameras.iter().enumerate() {
        if let Some(c) = cam {
            jposes.push(serde_json::json!({
                "image": std::path::Path::new(&paths[i]).file_stem().unwrap().to_str().unwrap(),
                "f": sfm.focals[i],
                "cx": sfm.cx(),
                "cy": sfm.cy(),
                "R": [[c.r[(0,0)], c.r[(0,1)], c.r[(0,2)]],
                      [c.r[(1,0)], c.r[(1,1)], c.r[(1,2)]],
                      [c.r[(2,0)], c.r[(2,1)], c.r[(2,2)]]],
                "t": [c.t.x, c.t.y, c.t.z],
            }));
        }
    }
    let jdata = serde_json::json!({
        "num_images": sfm.num_images,
        "num_registered": num_registered,
        "num_points": valid_pts.len(),
        "poses": jposes,
    });
    fs::write(
        "camera_poses.json",
        serde_json::to_string_pretty(&jdata).unwrap(),
    )
    .unwrap();

    println!("Points:  {} with RGB color", valid_pts.len());
    println!(
        "3D bbox: [{:.2},{:.2},{:.2}] → [{:.2},{:.2},{:.2}]",
        mn.x, mn.y, mn.z, mx.x, mx.y, mx.z
    );
    println!(
        "centroid: [{:.2}, {:.2}, {:.2}]",
        centroid.x, centroid.y, centroid.z
    );
    println!(
        "Saved sparse.ply ({} pts + {} cameras, {:.1} KB)",
        valid_pts.len(),
        num_registered,
        ply.len() as f64 / 1024.0
    );
}

/// 平面扫描单应性：H = K_src·(R_sr + t_sr·nᵀ/d)·K_ref⁻¹，把参考图像素映射到源图像素
fn homography_for_depth(
    k_src: &Matrix3<f64>,
    k_ref: &Matrix3<f64>,
    r_sr: &Matrix3<f64>,
    t_sr: &Vector3<f64>,
    d: f64,
    n: &Vector3<f64>,
) -> Matrix3<f64> {
    let m = r_sr + t_sr * n.transpose() / d;
    let k_ref_inv = k_ref.try_inverse().expect("K not invertible");
    k_src * m * k_ref_inv
}

/// 用平面单应性把参考 patch 映射到源图，计算 NCC（PatchMatch 光度一致性代价）
fn ncc_patch_h(
    ref_gray: &[f32],
    src_gray: &[f32],
    iw: usize,
    ih: usize,
    h: &Matrix3<f64>,
    x: i64,
    y: i64,
    half: i64,
) -> Option<f32> {
    let mut sr = 0.0f32;
    let mut ss = 0.0f32;
    let mut sr2 = 0.0f32;
    let mut ss2 = 0.0f32;
    let mut srs = 0.0f32;
    let mut n = 0usize;
    for dy in -half..=half {
        let ry = y + dy;
        if ry < 0 || ry >= ih as i64 {
            continue;
        }
        for dx in -half..=half {
            let rx = x + dx;
            if rx < 0 || rx >= iw as i64 {
                continue;
            }
            let ph = h * Vector3::new(rx as f64, ry as f64, 1.0);
            if ph.z.abs() < 1e-9 {
                continue;
            }
            let sx = (ph.x / ph.z).round() as i64;
            let sy = (ph.y / ph.z).round() as i64;
            if sx < 0 || sx >= iw as i64 || sy < 0 || sy >= ih as i64 {
                continue;
            }
            let rv = ref_gray[(ry as usize) * iw + rx as usize];
            let sv = src_gray[(sy as usize) * iw + sx as usize];
            sr += rv;
            ss += sv;
            sr2 += rv * rv;
            ss2 += sv * sv;
            srs += rv * sv;
            n += 1;
        }
    }
    if n < 9 {
        return None;
    }
    let nf = n as f32;
    let mean_r = sr / nf;
    let mean_s = ss / nf;
    let cov = srs - nf * mean_r * mean_s;
    let var_r = sr2 - nf * mean_r * mean_r;
    let var_s = ss2 - nf * mean_s * mean_s;
    let denom = (var_r * var_s).sqrt();
    if denom < 1e-6 {
        return None;
    }
    Some((cov / denom).clamp(-1.0, 1.0))
}

/// MVS 使用的相机（位姿 + 焦距 + 降采样后的灰度/彩色图）
struct RegCam {
    r: Matrix3<f64>,
    t: Vector3<f64>,
    f: f64,
    f_full: f64,
    img: RgbImage,
    gray: Vec<f32>,
    path: String,
}

/// PatchMatch 立体匹配（CPU 版，仿 COLMAP patch_match）：
/// 随机初始化深度/法向 → num_iterations 次 × 4 方向 sweep（传播 + 随机扰动），
/// 代价为源图像集合上的 NCC 均值。返回每像素 (深度, 法向, 代价)。
fn patch_match_image(
    si: usize,
    cams: &[RegCam],
    neighbors: &[usize],
    dmin: f64,
    dmax: f64,
    iw: usize,
    ih: usize,
    cx: f64,
    cy: f64,
    patch_half: i64,
    num_iterations: usize,
    geom_depths: Option<&[Vec<f64>]>,
    geom_weight: f64,
    geom_max_err: f64,
) -> (Vec<f64>, Vec<[f64; 3]>, Vec<f32>) {
    let n_px = iw * ih;
    let f_ref = cams[si].f;
    let k_ref = Matrix3::new(f_ref, 0.0, cx, 0.0, f_ref, cy, 0.0, 0.0, 1.0);
    let r_w = cams[si].r.transpose();
    let t_ref = cams[si].t;

    // 相对位姿 + 源图内参（预计算，避免每像素重复）
    let rel: Vec<(usize, Matrix3<f64>, Vector3<f64>, Matrix3<f64>)> = neighbors
        .iter()
        .map(|&ri| {
            let (r_sr, t_sr) =
                relative_pose_mvs(&cams[si].r, &cams[si].t, &cams[ri].r, &cams[ri].t);
            let k_src = Matrix3::new(cams[ri].f, 0.0, cx, 0.0, cams[ri].f, cy, 0.0, 0.0, 1.0);
            (ri, r_sr, t_sr, k_src)
        })
        .collect();

    // 代价 = 光度一致性(NCC) 与几何一致性(前向-后向重投影) 的加权组合（越高越好）
    let cost_at = |x: i64, y: i64, d: f64, n: [f64; 3]| -> f32 {
        let u = (x as f64 - cx) / f_ref;
        let v = (y as f64 - cy) / f_ref;
        let d_plane = d * (n[0] * u + n[1] * v + n[2]);
        if d_plane.abs() < 1e-9 {
            return -1.0;
        }
        let nvec = Vector3::new(n[0], n[1], n[2]);
        let mut total = 0.0f32;
        let mut cnt = 0usize;
        for (ri, r_sr, t_sr, k_src) in &rel {
            let h = homography_for_depth(k_src, &k_ref, r_sr, t_sr, d_plane, &nvec);
            if let Some(ncc) = ncc_patch_h(
                &cams[si].gray,
                &cams[*ri].gray,
                iw,
                ih,
                &h,
                x,
                y,
                patch_half,
            ) {
                total += ncc;
                cnt += 1;
            }
        }
        let ncc_mean = if cnt == 0 { 0.0 } else { total / cnt as f32 };

        // 几何一致性：前向-后向重投影误差（需要源图已有深度图）
        let geom_score = if let Some(gd) = geom_depths {
            let xc = Vector3::new(u * d, v * d, d);
            let xw = r_w * (xc - t_ref);
            let mut err_sum = 0.0f64;
            let mut err_cnt = 0usize;
            for &ri in neighbors {
                let xc2 = cams[ri].r * xw + cams[ri].t;
                if xc2.z <= 1e-6 {
                    continue;
                }
                let ux = (cams[ri].f * xc2.x / xc2.z + cx).round() as i64;
                let uy = (cams[ri].f * xc2.y / xc2.z + cy).round() as i64;
                if ux < 0 || ux >= iw as i64 || uy < 0 || uy >= ih as i64 {
                    continue;
                }
                let d_ri = gd[ri][(uy * iw as i64 + ux) as usize];
                if d_ri <= 0.0 {
                    continue;
                }
                let u_ri = (ux as f64 - cx) / cams[ri].f;
                let v_ri = (uy as f64 - cy) / cams[ri].f;
                let xc_ri = Vector3::new(u_ri * d_ri, v_ri * d_ri, d_ri);
                let xw_ri = cams[ri].r.transpose() * (xc_ri - cams[ri].t);
                let xc_ref = cams[si].r * xw_ri + t_ref;
                if xc_ref.z <= 1e-6 {
                    continue;
                }
                let bx = cams[si].f * xc_ref.x / xc_ref.z + cx;
                let by = cams[si].f * xc_ref.y / xc_ref.z + cy;
                let err = ((bx - x as f64).powi(2) + (by - y as f64).powi(2)).sqrt();
                err_sum += (err / geom_max_err).min(1.0);
                err_cnt += 1;
            }
            if err_cnt > 0 {
                (1.0 - err_sum / err_cnt as f64) as f32
            } else {
                0.0
            }
        } else {
            1.0
        };

        (1.0 - geom_weight as f32) * ncc_mean + geom_weight as f32 * geom_score
    };

    let mut rng = StdRng::seed_from_u64(RANSAC_SEED ^ (si as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let mut depth = vec![0.0f64; n_px];
    let mut normal = vec![[0.0f64, 0.0f64, 1.0f64]; n_px];
    let mut costs = vec![-1.0f32; n_px];

    // 随机初始化
    for y in 0..ih as i64 {
        for x in 0..iw as i64 {
            let idx = (y * iw as i64 + x) as usize;
            let d = dmin + rng.gen_range(0.0f64..1.0f64) * (dmax - dmin);
            let nx = rng.gen_range(-0.5f64..0.5f64);
            let ny = rng.gen_range(-0.5f64..0.5f64);
            let nz = (1.0 - nx * nx - ny * ny).sqrt();
            let nn = [nx, ny, nz];
            depth[idx] = d;
            normal[idx] = nn;
            costs[idx] = cost_at(x, y, d, nn);
        }
    }

    // 迭代优化：4 方向 sweep，传播 + 扰动
    for iter in 0..num_iterations {
        let d_perturb = (dmax - dmin) * 0.5f64.powi(iter as i32);
        let n_perturb = 0.5f64.powi(iter as i32 + 1);
        for dir in 0..4 {
            for yy in 0..ih as i64 {
                for xx in 0..iw as i64 {
                    let (x, y) = match dir {
                        0 => (xx, yy),
                        1 => (iw as i64 - 1 - xx, yy),
                        2 => (xx, ih as i64 - 1 - yy),
                        _ => (iw as i64 - 1 - xx, ih as i64 - 1 - yy),
                    };
                    let idx = (y * iw as i64 + x) as usize;
                    // (a) 传播：取 sweep 方向上前一个像素的深度/法向
                    let (px, py) = match dir {
                        0 => (x - 1, y),
                        1 => (x + 1, y),
                        2 => (x, y - 1),
                        _ => (x, y + 1),
                    };
                    if px >= 0 && px < iw as i64 && py >= 0 && py < ih as i64 {
                        let pidx = (py * iw as i64 + px) as usize;
                        let c = cost_at(x, y, depth[pidx], normal[pidx]);
                        if c > costs[idx] {
                            depth[idx] = depth[pidx];
                            normal[idx] = normal[pidx];
                            costs[idx] = c;
                        }
                    }
                    // (b) 随机扰动
                    let d_new =
                        (depth[idx] + rng.gen_range(-1.0f64..1.0f64) * d_perturb).clamp(dmin, dmax);
                    let nx_new = (normal[idx][0] + rng.gen_range(-1.0f64..1.0f64) * n_perturb)
                        .clamp(-0.9, 0.9);
                    let ny_new = (normal[idx][1] + rng.gen_range(-1.0f64..1.0f64) * n_perturb)
                        .clamp(-0.9, 0.9);
                    let nz_new = (1.0 - nx_new * nx_new - ny_new * ny_new).sqrt();
                    let nn_new = [nx_new, ny_new, nz_new];
                    let c = cost_at(x, y, d_new, nn_new);
                    if c > costs[idx] {
                        depth[idx] = d_new;
                        normal[idx] = nn_new;
                        costs[idx] = c;
                    }
                }
            }
        }
    }

    (depth, normal, costs)
}

/// 深度图融合：对每像素反投影为 3D 点，在邻帧间做深度一致性检查，
/// 达到 min_consistent 个一致视角才保留该点（仿 COLMAP StereoFusion 的几何一致性）。
fn fuse_depths(
    cams: &[RegCam],
    neighbors: &[Vec<usize>],
    depths: &[Vec<f64>],
    normals: &[Vec<[f64; 3]>],
    costs: &[Vec<f32>],
    iw: usize,
    ih: usize,
    cx: f64,
    cy: f64,
    cost_min: f32,
    depth_err_px: f64,
    min_consistent: usize,
    max_normal_error: f64,
) -> Vec<(f64, f64, f64, u8, u8, u8)> {
    let nc = cams.len();
    let max_normal_cos = max_normal_error.cos();
    let mut fused: Vec<(f64, f64, f64, u8, u8, u8)> = Vec::new();
    for si in 0..nc {
        let f = cams[si].f;
        let r_w = cams[si].r.transpose();
        for y in 0..ih {
            for x in 0..iw {
                let idx = y * iw + x;
                let d = depths[si][idx];
                if d <= 0.0 || costs[si][idx] < cost_min {
                    continue;
                }
                let u = (x as f64 - cx) / f;
                let v = (y as f64 - cy) / f;
                let xc = Vector3::new(u * d, v * d, d);
                let xw = r_w * (xc - cams[si].t);

                // 参考视角的法向（相机系 → 世界系）
                let nc = normals[si][idx];
                let nw = r_w * Vector3::new(nc[0], nc[1], nc[2]);

                // 几何一致性：投影到邻帧，比较深度 + 法向夹角
                let mut consistent = 0usize;
                for &ri in &neighbors[si] {
                    let xc2 = cams[ri].r * xw + cams[ri].t;
                    if xc2.z <= 1e-6 {
                        continue;
                    }
                    let ux = (cams[ri].f * xc2.x / xc2.z + cx).round();
                    let uy = (cams[ri].f * xc2.y / xc2.z + cy).round();
                    if ux < 0.0 || ux >= iw as f64 || uy < 0.0 || uy >= ih as f64 {
                        continue;
                    }
                    let j = (uy as usize) * iw + ux as usize;
                    let d2 = depths[ri][j];
                    if d2 <= 0.0 {
                        continue;
                    }
                    if (d2 - xc2.z).abs() / xc2.z < depth_err_px {
                        // 法向一致性：两视角估计的法向（世界系）夹角要小
                        let nc2 = normals[ri][j];
                        let nw2 = cams[ri].r.transpose() * Vector3::new(nc2[0], nc2[1], nc2[2]);
                        if nw.dot(&nw2) > max_normal_cos {
                            consistent += 1;
                        }
                    }
                }
                if consistent < min_consistent {
                    continue;
                }
                let pix = cams[si].img.get_pixel(x as u32, y as u32);
                fused.push((xw.x, xw.y, xw.z, pix[0], pix[1], pix[2]));
            }
        }
    }
    fused
}

/// 移动最小二乘(MLS)平面投影平滑（核心，仅处理坐标）：对每个点拟合局部平面并投影，
/// 消除表面毛刺。邻域半径 h，高斯加权。
fn mls_project(points: &[(f64, f64, f64)], h: f64) -> Vec<(f64, f64, f64)> {
    let n = points.len();
    if n < 10 || h <= 0.0 {
        return points.to_vec();
    }
    let h2 = h * h;
    let cell = h.max(1e-4);
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, (x, y, z)) in points.iter().enumerate() {
        grid.entry(((x / cell) as i64, (y / cell) as i64, (z / cell) as i64))
            .or_default()
            .push(i);
    }
    let mut out: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for (x, y, z) in points {
        let cx = (x / cell) as i64;
        let cy = (y / cell) as i64;
        let cz = (z / cell) as i64;
        let mut wsum = 0.0f64;
        let mut cent = Vector3::zeros();
        let mut cov = Matrix3::zeros();
        for dz in -1i64..=1 {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in ids {
                            let (xj, yj, zj) = (points[j].0, points[j].1, points[j].2);
                            let d2 = (x - xj).powi(2) + (y - yj).powi(2) + (z - zj).powi(2);
                            if d2 >= h2 {
                                continue;
                            }
                            let w = (-d2 / h2).exp();
                            let pj = Vector3::new(xj, yj, zj);
                            wsum += w;
                            cent += w * pj;
                            cov += w * pj * pj.transpose();
                        }
                    }
                }
            }
        }
        if wsum < 1e-9 {
            out.push((*x, *y, *z));
            continue;
        }
        cent /= wsum;
        cov /= wsum;
        cov -= cent * cent.transpose();
        let eig = cov.symmetric_eigen();
        let normal = eig.eigenvectors.column(0);
        let p = Vector3::new(*x, *y, *z);
        let dist = (p - cent).dot(&normal);
        let pp = p - dist * normal;
        out.push((pp.x, pp.y, pp.z));
    }
    out
}

/// MLS 平滑（带颜色版本）
fn mls_smooth(points: &[(f64, f64, f64, u8, u8, u8)], h: f64) -> Vec<(f64, f64, f64, u8, u8, u8)> {
    let pos: Vec<(f64, f64, f64)> = points.iter().map(|&(x, y, z, _, _, _)| (x, y, z)).collect();
    let proj = mls_project(&pos, h);
    proj.into_iter()
        .zip(points.iter())
        .map(|((x, y, z), &(_, _, _, r, g, b))| (x, y, z, r, g, b))
        .collect()
}

/// MVS稠密重建：平面扫描64层ZNCC→LRC几何一致性→深度图融合→体素去重→三角网格→ply+obj
fn run_mvs_dense(paths: &[String]) {
    println!("\n============ MVS Dense Reconstruction ============");
    let pose_str = match fs::read_to_string("camera_poses.json") {
        Ok(s) => s,
        Err(_) => {
            println!("No camera_poses.json");
            return;
        }
    };
    let pose_data: serde_json::Value = serde_json::from_str(&pose_str).expect("Invalid JSON");
    let poses = pose_data["poses"].as_array().expect("No poses array");
    if poses.len() < 2 {
        println!("Need >=2 cameras");
        return;
    }

    println!("Loading {} images...", poses.len());
    let first_img = image::open(&paths[0]).expect("Failed open");
    let (ow, oh) = (first_img.width(), first_img.height());
    let scale = 2u32;
    let (iw, ih) = (ow / scale, oh / scale);
    let intrinsics = CameraIntrinsics::from_image_dimensions(ow, oh);
    let f_default = intrinsics.k[(0, 0)];
    let cx_full = intrinsics.k[(0, 2)];
    let cy_full = intrinsics.k[(1, 2)];
    let cx = cx_full / scale as f64;
    let cy = cy_full / scale as f64;

    let mut cams: Vec<RegCam> = Vec::new();
    for pv in poses {
        let fname = pv["image"].as_str().unwrap();
        let img_idx = paths.iter().position(|p| p.contains(fname)).unwrap();
        let ra = pv["R"].as_array().unwrap();
        let ta = pv["t"].as_array().unwrap();
        // 使用稀疏重建阶段每图标定出的焦距，保证与 sparse.ply 几何一致
        let f_full = pv["f"].as_f64().unwrap_or(f_default);
        let r = Matrix3::new(
            ra[0][0].as_f64().unwrap(),
            ra[0][1].as_f64().unwrap(),
            ra[0][2].as_f64().unwrap(),
            ra[1][0].as_f64().unwrap(),
            ra[1][1].as_f64().unwrap(),
            ra[1][2].as_f64().unwrap(),
            ra[2][0].as_f64().unwrap(),
            ra[2][1].as_f64().unwrap(),
            ra[2][2].as_f64().unwrap(),
        );
        let t = Vector3::new(
            ta[0].as_f64().unwrap(),
            ta[1].as_f64().unwrap(),
            ta[2].as_f64().unwrap(),
        );
        let img = image::open(&paths[img_idx])
            .unwrap()
            .resize_exact(iw, ih, image::imageops::FilterType::Lanczos3)
            .to_rgb8();
        let gray: Vec<f32> = img
            .pixels()
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect();
        cams.push(RegCam {
            r,
            t,
            f: f_full / scale as f64,
            f_full,
            img,
            gray,
            path: paths[img_idx].clone(),
        });
    }
    if cams.len() < 2 {
        return;
    }
    let nc = cams.len();

    let n_nbr = 4usize.min(nc - 1);
    let cam_centers: Vec<Vector3<f64>> = cams.iter().map(|c| -c.r.transpose() * c.t).collect();

    // Compute per-camera depth ranges from sparse points（跳过相机中心红点，只统计真实场景点）
    let mut cam_depths: Vec<(f64, f64)> = vec![(1.0, 300.0); nc];
    let mut median_depth = 20.0f64;
    let mut sparse_bbox: Option<(Vector3<f64>, Vector3<f64>)> = None;
    let mut sparse_pts: Vec<Vector3<f64>> = Vec::new();
    if let Ok(sp) = fs::read_to_string("sparse.ply") {
        for line in sp.lines().skip(10) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                if let (Ok(x), Ok(y), Ok(z)) =
                    (parts[0].parse(), parts[1].parse(), parts[2].parse())
                {
                    // 相机中心在 sparse.ply 中以红色 (255 0 0) 表示，需剔除
                    if parts[3] == "255" && parts[4] == "0" && parts[5] == "0" {
                        continue;
                    }
                    if z > 0.1 {
                        sparse_pts.push(Vector3::new(x, y, z));
                    }
                }
            }
        }
        if sparse_pts.len() > 20 {
            // 用稀疏点云包围盒约束稠密重建，剔除飘在场景外的噪声
            let mut mn = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
            let mut mx = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
            for pt in &sparse_pts {
                mn = mn.inf(pt);
                mx = mx.sup(pt);
            }
            let extent = mx - mn;
            let margin = Vector3::new(
                (extent.x.abs() * 0.2).max(1.0),
                (extent.y.abs() * 0.2).max(1.0),
                (extent.z.abs() * 0.2).max(1.0),
            );
            sparse_bbox = Some((mn - margin, mx + margin));

            for ci in 0..nc {
                let mut ds: Vec<f64> = Vec::new();
                for pt in &sparse_pts {
                    let xc = cams[ci].r * pt + cams[ci].t;
                    if xc.z > 0.5 {
                        ds.push(xc.z);
                    }
                }
                if ds.len() > 10 {
                    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let dlo = ds[ds.len() / 4].max(0.3);
                    let dhi = ds[3 * ds.len() / 4].min(500.0);
                    cam_depths[ci] = (dlo, dhi);
                }
            }
            let mut mids: Vec<f64> = cam_depths.iter().map(|&(a, b)| (a + b) * 0.5).collect();
            mids.sort_by(|a, b| a.partial_cmp(b).unwrap());
            median_depth = mids[mids.len() / 2];
            // 视野内稀疏点过少（位姿朝外）的相机沿用默认 300 会扫出巨大深度，
            // 这里回退到全局中位深度附近的合理范围
            for ci in 0..nc {
                if cam_depths[ci].1 > median_depth * 5.0 {
                    cam_depths[ci] = (median_depth * 0.5, median_depth * 2.0);
                }
            }
            println!(
                "  Depth from sparse: cam0=[{:.1},{:.1}] median={:.1}",
                cam_depths[0].0, cam_depths[0].1, median_depth
            );
        }
    }

    // 按基线距离选邻帧：基线太小深度分辨率差（几乎等于单目），取接近 0.25*中位深度的基线，
    // 既保证足够视差又避免过远导致的遮挡/外观变化。
    let target_base = median_depth * 0.25;
    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(nc);
    for si in 0..nc {
        let mut order: Vec<(f64, usize)> = (0..nc)
            .filter(|&r| r != si)
            .map(|r| {
                let b = (cam_centers[si] - cam_centers[r]).norm();
                ((b - target_base).abs(), r)
            })
            .collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.push(order.into_iter().take(n_nbr).map(|(_, r)| r).collect());
    }

    let patch_half: i64 = 3; // 7x7 NCC patch
    let num_iterations = 5; // PatchMatch 迭代次数
    let cost_min = 0.3f32; // 最小 NCC 代价（低于此剔除）
    let min_consistent = 2usize; // 融合时最少一致视角数
    let depth_err_px = 0.01f64; // 深度一致性相对误差
    let max_normal_error = 45.0f64.to_radians(); // 法向一致性最大夹角

    println!(
        "Downsampled {}x{} f={:.0}, PatchMatch stereo (iter={}, patch={}px)...",
        iw,
        ih,
        cams[0].f,
        num_iterations,
        2 * patch_half + 1
    );

    // PatchMatch 逐参考图像估计深度/法向（并行，两趟：先光度，再叠加几何一致性正则）
    let geom_weight = 0.5f64; // 几何一致性正则项权重
    let geom_max_err = 3.0f64; // 前向-后向重投影误差上限（像素）

    let pm1: Vec<(Vec<f64>, Vec<[f64; 3]>, Vec<f32>)> = (0..nc)
        .into_par_iter()
        .map(|si| {
            let (dlo, dhi) = cam_depths[si];
            patch_match_image(
                si,
                &cams,
                &neighbors[si],
                dlo,
                dhi * 1.5,
                iw as usize,
                ih as usize,
                cx,
                cy,
                patch_half,
                num_iterations,
                None,
                geom_weight,
                geom_max_err,
            )
        })
        .collect();
    let init_depths: Vec<Vec<f64>> = pm1.into_iter().map(|(d, _, _)| d).collect();

    let pm2: Vec<(Vec<f64>, Vec<[f64; 3]>, Vec<f32>)> = (0..nc)
        .into_par_iter()
        .map(|si| {
            let (dlo, dhi) = cam_depths[si];
            patch_match_image(
                si,
                &cams,
                &neighbors[si],
                dlo,
                dhi * 1.5,
                iw as usize,
                ih as usize,
                cx,
                cy,
                patch_half,
                num_iterations,
                Some(&init_depths),
                geom_weight,
                geom_max_err,
            )
        })
        .collect();
    let mut depths: Vec<Vec<f64>> = Vec::with_capacity(nc);
    let mut normals: Vec<Vec<[f64; 3]>> = Vec::with_capacity(nc);
    let mut costs: Vec<Vec<f32>> = Vec::with_capacity(nc);
    for (d, n, c) in pm2 {
        depths.push(d);
        normals.push(n);
        costs.push(c);
    }
    println!("  depth maps computed for {} cameras", nc);

    // 深度图中值滤波：用 5x5 邻域中值替换每像素深度，平滑估计噪声（保留密度）
    for si in 0..nc {
        let mut smoothed = depths[si].clone();
        for y in 2..ih as usize - 2 {
            for x in 2..iw as usize - 2 {
                let idx = y * iw as usize + x;
                let mut nbr: Vec<f64> = Vec::new();
                for dy in -2i64..=2 {
                    for dx in -2i64..=2 {
                        let nd = depths[si]
                            [(y as i64 + dy) as usize * iw as usize + (x as i64 + dx) as usize];
                        if nd > 0.0 {
                            nbr.push(nd);
                        }
                    }
                }
                if nbr.len() >= 13 {
                    nbr.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    smoothed[idx] = nbr[nbr.len() / 2];
                }
            }
        }
        depths[si] = smoothed;
    }
    println!("  depth median filter applied (5x5)");

    // 深度图融合：反投影 + 邻帧深度一致性（仿 COLMAP StereoFusion 几何一致性）
    let mut all_pts: Vec<(f64, f64, f64, u8, u8, u8)> = fuse_depths(
        &cams,
        &neighbors,
        &depths,
        &normals,
        &costs,
        iw as usize,
        ih as usize,
        cx,
        cy,
        cost_min,
        depth_err_px,
        min_consistent,
        max_normal_error,
    );
    println!("  fused {} points", all_pts.len());
    // 用稀疏点云包围盒约束：剔除场景外漂浮噪声
    if let Some((lo, hi)) = sparse_bbox {
        all_pts.retain(|(x, y, z, _, _, _)| {
            *x > lo.x && *x < hi.x && *y > lo.y && *y < hi.y && *z > lo.z && *z < hi.z
        });
        println!(
            "  bbox constraint: {} points within sparse bounds [{:.1},{:.1},{:.1}]~[{:.1},{:.1},{:.1}]",
            all_pts.len(),
            lo.x,
            lo.y,
            lo.z,
            hi.x,
            hi.y,
            hi.z
        );
    }
    if all_pts.is_empty() {
        println!("No dense points");
        return;
    }

    // 到稀疏点云的距离约束：剔除离稀疏表面过远的稠密点（外部团状噪声）
    if !sparse_pts.is_empty() {
        let max_dist = (median_depth * 0.06).max(0.1);
        let cell = max_dist;
        let mut sgrid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in sparse_pts.iter().enumerate() {
            sgrid
                .entry((
                    (p.x / cell) as i64,
                    (p.y / cell) as i64,
                    (p.z / cell) as i64,
                ))
                .or_default()
                .push(i);
        }
        let before = all_pts.len();
        all_pts.retain(|(x, y, z, _, _, _)| {
            let cx = (*x / cell) as i64;
            let cy = (*y / cell) as i64;
            let cz = (*z / cell) as i64;
            for dz in -2i64..=2 {
                for dy in -2i64..=2 {
                    for dx in -2i64..=2 {
                        if let Some(ids) = sgrid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in ids {
                                let p = sparse_pts[j];
                                if (x - p.x).powi(2) + (y - p.y).powi(2) + (z - p.z).powi(2)
                                    < max_dist * max_dist
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            false
        });
        println!(
            "  distance-to-sparse: {} -> {} points (within {:.2})",
            before,
            all_pts.len(),
            max_dist
        );
    }

    let mut xs: Vec<f64> = all_pts.iter().map(|(x, _, _, _, _, _)| *x).collect();
    let mut ys: Vec<f64> = all_pts.iter().map(|(_, y, _, _, _, _)| *y).collect();
    let mut zs: Vec<f64> = all_pts.iter().map(|(_, _, z, _, _, _)| *z).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mx, my, mz) = (xs[xs.len() / 2], ys[ys.len() / 2], zs[zs.len() / 2]);
    let mut dists: Vec<f64> = all_pts
        .iter()
        .map(|(x, y, z, _, _, _)| ((x - mx).powi(2) + (y - my).powi(2) + (z - mz).powi(2)).sqrt())
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = dists[dists.len() / 2];
    let max_d = (med * 5.0).max(med + 1.0);
    all_pts.retain(|(x, y, z, _, _, _)| {
        ((x - mx).powi(2) + (y - my).powi(2) + (z - mz).powi(2)).sqrt() < max_d
    });

    // 体素尺寸随场景尺度自适应（约为中位半径的 1%），避免点云过稀疏
    let vs = (med * 0.01).max(0.01);
    let mut vox: BTreeMap<(i64, i64, i64), (usize, f64, f64, f64, u64, u64, u64)> = BTreeMap::new();
    for (x, y, z, r, g, b) in &all_pts {
        let e = vox
            .entry((
                (x / vs).round() as i64,
                (y / vs).round() as i64,
                (z / vs).round() as i64,
            ))
            .or_default();
        e.0 += 1;
        e.1 += x;
        e.2 += y;
        e.3 += z;
        e.4 += *r as u64;
        e.5 += *g as u64;
        e.6 += *b as u64;
    }
    // 体素去重，保留体素坐标用于邻域统计
    let mut dedup_vox: Vec<((i64, i64, i64), f64, f64, f64, u8, u8, u8)> = vox
        .into_iter()
        .map(|((vx, vy, vz), (n, sx, sy, sz, sr, sg, sb))| {
            (
                (vx, vy, vz),
                sx / n as f64,
                sy / n as f64,
                sz / n as f64,
                (sr / n as u64) as u8,
                (sg / n as u64) as u8,
                (sb / n as u64) as u8,
            )
        })
        .collect();

    // 统计离群滤波（PCL statistical outlier removal）：对每个点求 k 近邻平均距离，
    // 剔除平均距离 > 全局均值 + std_mult*标准差 的点。
    let k_nn = 8usize;
    let std_mult = 0.5f64;
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, ((vx, vy, vz), ..)) in dedup_vox.iter().enumerate() {
        grid.entry((*vx, *vy, *vz)).or_default().push(i);
    }
    let n_pts = dedup_vox.len();
    let mut mean_dists: Vec<f64> = vec![0.0; n_pts];
    for (i, ((vx, vy, vz), x, y, z, ..)) in dedup_vox.iter().enumerate() {
        let mut dists: Vec<f64> = Vec::new();
        'outer: for rad in 1i64..=4 {
            for dz in -rad..=rad {
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        if dx.abs() < rad && dy.abs() < rad && dz.abs() < rad {
                            continue;
                        }
                        if let Some(pts) = grid.get(&(vx + dx, vy + dy, vz + dz)) {
                            for &j in pts {
                                if j == i {
                                    continue;
                                }
                                let (xj, yj, zj) = (dedup_vox[j].1, dedup_vox[j].2, dedup_vox[j].3);
                                dists.push(
                                    ((x - xj).powi(2) + (y - yj).powi(2) + (z - zj).powi(2)).sqrt(),
                                );
                                if dists.len() >= k_nn {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
        if dists.is_empty() {
            mean_dists[i] = f64::MAX;
        } else {
            let m = dists.len().min(k_nn);
            let s: f64 = dists.iter().take(m).sum();
            mean_dists[i] = s / m as f64;
        }
    }
    let finite: Vec<f64> = mean_dists
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    let mean_md = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
    let var_md =
        finite.iter().map(|d| (d - mean_md).powi(2)).sum::<f64>() / finite.len().max(1) as f64;
    let threshold = mean_md + std_mult * var_md.sqrt();
    let dedup: Vec<(f64, f64, f64, u8, u8, u8)> = dedup_vox
        .into_iter()
        .zip(mean_dists)
        .filter(|(_, md)| *md < threshold)
        .map(|(((_, _, _), x, y, z, r, g, b), _)| (x, y, z, r, g, b))
        .collect();

    // MLS 平滑：把每个点投影到局部拟合平面上，消除表面毛刺
    let mls_h = (vs * 3.0).max(0.03);
    let dedup = mls_smooth(&dedup, mls_h);
    println!("  MLS smoothing applied (h={:.3})", mls_h);

    let mut ply = format!("ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", dedup.len());
    for (x, y, z, r, g, b) in &dedup {
        ply.push_str(&format!("{} {} {} {} {} {}\n", x, y, z, r, g, b));
    }
    fs::write("dense.ply", &ply).unwrap();
    println!(
        "Dense: {}→{} pts → dense.ply ({:.1} KB)",
        all_pts.len(),
        dedup.len(),
        ply.len() as f64 / 1024.0
    );

    // ===== 网格化前：用稀疏点云距离约束过滤深度图，保证 mesh.obj 与 dense.ply 一致 =====
    if !sparse_pts.is_empty() {
        let max_dist = (median_depth * 0.06).max(0.1);
        let cell = max_dist;
        let mut sgrid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in sparse_pts.iter().enumerate() {
            sgrid
                .entry((
                    (p.x / cell) as i64,
                    (p.y / cell) as i64,
                    (p.z / cell) as i64,
                ))
                .or_default()
                .push(i);
        }
        for si in 0..nc {
            let r_w = cams[si].r.transpose();
            let f = cams[si].f;
            for y in 0..ih {
                for x in 0..iw {
                    let idx = (y * iw + x) as usize;
                    let d = depths[si][idx];
                    if d <= 0.0 {
                        continue;
                    }
                    let u = (x as f64 - cx) / f;
                    let v = (y as f64 - cy) / f;
                    let xc = Vector3::new(u * d, v * d, d);
                    let xw = r_w * (xc - cams[si].t);
                    let cxg = (xw.x / cell) as i64;
                    let cyg = (xw.y / cell) as i64;
                    let czg = (xw.z / cell) as i64;
                    let mut ok = false;
                    'search: for dz in -2i64..=2 {
                        for dy in -2i64..=2 {
                            for dx in -2i64..=2 {
                                if let Some(ids) = sgrid.get(&(cxg + dx, cyg + dy, czg + dz)) {
                                    for &j in ids {
                                        let p = sparse_pts[j];
                                        if (xw.x - p.x).powi(2)
                                            + (xw.y - p.y).powi(2)
                                            + (xw.z - p.z).powi(2)
                                            < max_dist * max_dist
                                        {
                                            ok = true;
                                            break 'search;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !ok {
                        depths[si][idx] = 0.0;
                    }
                }
            }
        }
        println!("  depth maps filtered by sparse distance (for meshing)");
    }

    // ===== 贴图模型：深度图网格三角化 + 单张纹理图集烘焙 =====
    println!("\n===== Building textured mesh =====");

    struct Face {
        v: [Vector3<f64>; 3],
        uv: [(f64, f64); 3],
        cam: usize,
    }
    let mut faces: Vec<Face> = Vec::new();
    for si in 0..nc {
        let r_w = cams[si].r.transpose();
        let mut wg: Vec<Vec<Option<Vector3<f64>>>> = vec![vec![None; iw as usize]; ih as usize];
        let mut dg: Vec<Vec<f64>> = vec![vec![0.0; iw as usize]; ih as usize];
        for y in 0..ih {
            for x in 0..iw {
                let d = depths[si][(y * iw + x) as usize];
                let score = costs[si][(y * iw + x) as usize];
                if d <= 0.0 || score < cost_min {
                    continue;
                }
                let u = (x as f64 - cx) / cams[si].f;
                let v = (y as f64 - cy) / cams[si].f;
                let xc = Vector3::new(u * d, v * d, d);
                let xw = r_w * (xc - cams[si].t);
                wg[y as usize][x as usize] = Some(xw);
                dg[y as usize][x as usize] = d;
            }
        }
        for y in 0..ih as usize - 1 {
            for x in 0..iw as usize - 1 {
                let a = wg[y][x];
                let b = wg[y][x + 1];
                let c = wg[y + 1][x];
                let d = wg[y + 1][x + 1];
                if a.is_none() || b.is_none() || c.is_none() || d.is_none() {
                    continue;
                }
                let da = dg[y][x];
                let db = dg[y][x + 1];
                let dc = dg[y + 1][x];
                let dd = dg[y + 1][x + 1];
                let dmin = da.min(db).min(dc).min(dd);
                let dmax = da.max(db).max(dc).max(dd);
                if dmin <= 0.0 || (dmax - dmin) / dmin > 0.1 {
                    continue;
                }
                let (a, b, c, d) = (a.unwrap(), b.unwrap(), c.unwrap(), d.unwrap());
                // 两个三角形：a-b-c 与 b-d-c（法向朝相机）
                for tri in [&[a, b, c][..], &[b, d, c][..]] {
                    let mut uv = [(0.0f64, 0.0f64); 3];
                    let mut ok = true;
                    for (i, xw) in tri.iter().enumerate() {
                        let xc = cams[si].r * xw + cams[si].t;
                        if xc.z <= 1e-6 {
                            ok = false;
                            break;
                        }
                        uv[i] = (
                            cams[si].f_full * xc.x / xc.z + cx_full,
                            cams[si].f_full * xc.y / xc.z + cy_full,
                        );
                    }
                    if ok {
                        faces.push(Face {
                            v: [tri[0], tri[1], tri[2]],
                            uv,
                            cam: si,
                        });
                    }
                }
            }
        }
    }
    println!("  {} triangles", faces.len());
    if faces.is_empty() {
        return;
    }

    // ---- 跨相机顶点融合：体素去重（共享顶点，消除相邻相机表面重叠/z-fighting）----
    let n_pts = faces.len() * 3;
    let (mut mx, mut my, mut mz) = (0.0f64, 0.0f64, 0.0f64);
    for fa in &faces {
        for v in &fa.v {
            mx += v.x;
            my += v.y;
            mz += v.z;
        }
    }
    mx /= n_pts as f64;
    my /= n_pts as f64;
    mz /= n_pts as f64;
    let mut dists: Vec<f64> = Vec::with_capacity(n_pts);
    for fa in &faces {
        for v in &fa.v {
            dists.push(((v.x - mx).powi(2) + (v.y - my).powi(2) + (v.z - mz).powi(2)).sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_d = dists[dists.len() / 2];
    let vs = (med_d * 0.01).max(1e-3);
    println!("  voxel size = {:.4} (median radius {:.2})", vs, med_d);

    let mut vpos: Vec<(f64, f64, f64)> = Vec::new();
    let mut vmap: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut face_vi: Vec<[usize; 3]> = Vec::with_capacity(faces.len());
    for fa in &faces {
        let mut idx = [0usize; 3];
        for (i, v) in fa.v.iter().enumerate() {
            let key = (
                (v.x / vs).round() as i64,
                (v.y / vs).round() as i64,
                (v.z / vs).round() as i64,
            );
            let id = *vmap.entry(key).or_insert_with(|| {
                vpos.push((v.x, v.y, v.z));
                vpos.len() - 1
            });
            idx[i] = id;
        }
        face_vi.push(idx);
    }
    println!("  {} unique vertices (from {})", vpos.len(), n_pts);

    // MLS 平滑网格顶点（与 dense.ply 保持一致）
    let mls_h_mesh = (vs * 3.0).max(0.03);
    vpos = mls_project(&vpos, mls_h_mesh);
    println!("  mesh vertices MLS smoothed (h={:.3})", mls_h_mesh);

    // ---- 三角形去重：按 (排序顶点索引) 去重，去掉跨相机完全重叠的三角面 ----
    use std::collections::HashSet;
    let mut seen: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut kept: Vec<usize> = Vec::new();
    for (fi, idx) in face_vi.iter().enumerate() {
        let (a, b, c) = (idx[0], idx[1], idx[2]);
        if a == b || b == c || a == c {
            continue;
        }
        let mut s = [a, b, c];
        s.sort_unstable();
        if seen.insert((s[0], s[1], s[2])) {
            kept.push(fi);
        }
    }
    println!(
        "  {} unique faces (removed {} duplicate/degenerate)",
        kept.len(),
        faces.len() - kept.len()
    );

    // 每个保留三角形在源图像中的包围盒 + padding
    let pad: u32 = 2;
    struct Rect {
        fi: usize,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        min_u: f64,
        min_v: f64,
    }
    let mut rects: Vec<Rect> = Vec::with_capacity(kept.len());
    for &fi in &kept {
        let fa = &faces[fi];
        let min_u = fa.uv.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
        let max_u = fa.uv.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max);
        let min_v = fa.uv.iter().map(|q| q.1).fold(f64::INFINITY, f64::min);
        let max_v = fa.uv.iter().map(|q| q.1).fold(f64::NEG_INFINITY, f64::max);
        let w = (max_u - min_u).ceil() as u32 + 1 + 2 * pad;
        let h = (max_v - min_v).ceil() as u32 + 1 + 2 * pad;
        rects.push(Rect {
            fi,
            x: 0,
            y: 0,
            w,
            h,
            min_u,
            min_v,
        });
    }
    // 图集尺寸：按总面积开方再取 2 的幂
    let total_area: u64 = rects.iter().map(|r| (r.w as u64) * (r.h as u64)).sum();
    let mut atlas_side = (total_area as f64).sqrt().ceil() as u32;
    atlas_side = atlas_side.max(512).next_power_of_two();
    // shelf 打包（按高度降序）
    rects.sort_by(|a, b| b.h.cmp(&a.h));
    let mut cur_x = 0u32;
    let mut cur_y = 0u32;
    let mut shelf_h = 0u32;
    for r in rects.iter_mut() {
        if r.w > atlas_side || r.h > atlas_side {
            r.x = u32::MAX;
            continue;
        }
        if cur_x + r.w > atlas_side {
            cur_x = 0;
            cur_y += shelf_h;
            shelf_h = 0;
        }
        if cur_y + r.h > atlas_side {
            r.x = u32::MAX;
            continue;
        }
        r.x = cur_x;
        r.y = cur_y;
        cur_x += r.w;
        shelf_h = shelf_h.max(r.h);
    }

    // 烘焙：把每个三角形从源图像光栅化到图集
    let mut atlas_img = RgbImage::from_pixel(atlas_side, atlas_side, Rgb([0, 0, 0]));
    let mut src_cache: HashMap<usize, RgbImage> = HashMap::new();
    for r in &rects {
        if r.x == u32::MAX {
            continue;
        }
        let fa = &faces[r.fi];
        let src = match src_cache.entry(fa.cam) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let im = image::open(&cams[fa.cam].path).unwrap().to_rgb8();
                e.insert(im)
            }
        };
        let (u0, v0) = fa.uv[0];
        let (u1, v1) = fa.uv[1];
        let (u2, v2) = fa.uv[2];
        let max_u = fa.uv.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max);
        let max_v = fa.uv.iter().map(|q| q.1).fold(f64::NEG_INFINITY, f64::max);
        let area = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
        if area.abs() < 1e-9 {
            continue;
        }
        let sgn = if area > 0.0 { 1.0 } else { -1.0 };
        let bx0 = r.min_u.floor() as i64 - 1;
        let bx1 = max_u.ceil() as i64 + 1;
        let by0 = r.min_v.floor() as i64 - 1;
        let by1 = max_v.ceil() as i64 + 1;
        for py in by0..=by1 {
            for px in bx0..=bx1 {
                if px < 0 || px >= src.width() as i64 || py < 0 || py >= src.height() as i64 {
                    continue;
                }
                let fx = px as f64 + 0.5;
                let fy = py as f64 + 0.5;
                let w0 = ((u1 - u0) * (fy - v0) - (v1 - v0) * (fx - u0)) * sgn;
                let w1 = ((u2 - u1) * (fy - v1) - (v2 - v1) * (fx - u1)) * sgn;
                let w2 = ((u0 - u2) * (fy - v2) - (v0 - v2) * (fx - u2)) * sgn;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let ax = ((r.x + pad) as f64 + (fx - r.min_u)).floor() as u32;
                let ay = ((r.y + pad) as f64 + (fy - r.min_v)).floor() as u32;
                if ax >= atlas_side || ay >= atlas_side {
                    continue;
                }
                let sp = src.get_pixel(px as u32, py as u32);
                atlas_img.put_pixel(ax, ay, *sp);
            }
        }
    }

    // 写 mesh.obj：共享顶点 v + 每面角独立的 vt（UV 接缝处理）
    let mut obj = String::new();
    obj.push_str("# Textured mesh from MVS\nmtllib mesh.mtl\nusemtl mvs_tex\no MVS_Mesh\n");
    for (x, y, z) in &vpos {
        obj.push_str(&format!("v {:.6} {:.6} {:.6}\n", x, y, z));
    }
    let mut vt_count = 0usize;
    let mut n_written = 0usize;
    for r in &rects {
        if r.x == u32::MAX {
            continue;
        }
        let fa = &faces[r.fi];
        let vi = &face_vi[r.fi];
        for i in 0..3 {
            let u = ((r.x + pad) as f64 + (fa.uv[i].0 - r.min_u)) / atlas_side as f64;
            let v = 1.0 - ((r.y + pad) as f64 + (fa.uv[i].1 - r.min_v)) / atlas_side as f64;
            obj.push_str(&format!("vt {:.6} {:.6}\n", u, v));
        }
        obj.push_str(&format!(
            "f {}/{} {}/{} {}/{}\n",
            vi[0] + 1,
            vt_count + 1,
            vi[1] + 1,
            vt_count + 2,
            vi[2] + 1,
            vt_count + 3
        ));
        vt_count += 3;
        n_written += 1;
    }
    fs::write("mesh.obj", &obj).expect("Failed to write mesh.obj");
    fs::write(
        "mesh.mtl",
        "newmtl mvs_tex\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd mesh_texture.png\n",
    )
    .expect("Failed to write mesh.mtl");
    atlas_img
        .save("mesh_texture.png")
        .expect("Failed to write mesh_texture.png");
    println!(
        "Mesh: {} faces ({} dropped), {} vertices, atlas {}x{}, -> mesh.obj + mesh_texture.png",
        n_written,
        kept.len() - n_written,
        vpos.len(),
        atlas_side,
        atlas_side
    );
}

fn relative_pose_mvs(
    r_s: &Matrix3<f64>,
    t_s: &Vector3<f64>,
    r_r: &Matrix3<f64>,
    t_r: &Vector3<f64>,
) -> (Matrix3<f64>, Vector3<f64>) {
    let r_sr = r_r * r_s.transpose();
    (r_sr, t_r - r_sr * t_s)
}

fn main() {
    let total_start = Instant::now();

    let paths = load_image_paths("./images");
    if paths.is_empty() {
        eprintln!("No images found in ./images/ directory");
        return;
    }
    println!("Found {} images\n", paths.len());

    let features = extract_features(&paths);

    let matches = match_all_pairs(&features, RATIO_THRESHOLD);
    let num_matches = matches.len();
    /// 主入口：SfI初始化→增量注册→BA→MVS稠密重建
    let tracks = build_tracks(&matches, &features);

    // print_track_details(&tracks, &features, &paths);

    // save_tracks_to_json(&tracks, &features, &paths, num_matches, "tracks.json");

    // visualize_track_pairs(&tracks, &features, &paths, "tracks_viz");

    visualize_track_coverage_graph(&tracks, &features, &paths, "tracks_viz/coverage_graph.png");

    run_incremental_sfm(&matches, &features, &tracks, &paths);

    // MVS Dense Reconstruction
    /// 计算两相机间相对位姿
    run_mvs_dense(&paths);
    let total_elapsed = total_start.elapsed();
    println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
}
