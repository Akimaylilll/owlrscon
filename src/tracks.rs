use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::time::Instant;

use image::{Rgb, RgbImage};
use imageproc::drawing;
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;

use crate::matching::Match;
use crate::sfm::camera_center;
use crate::types::{
    ImageFeatures, MAX_KEYPOINTS_PER_IMAGE, MIN_TRACK_LENGTH, RATIO_THRESHOLD, SLIDING_WINDOW,
    SNAKE_GAP_THRESHOLD,
};

pub(crate) struct Track {
    pub(crate) observations: Vec<(usize, usize)>,
}

/// 并查集（Union-Find），用于将匹配对聚合为连通分量（Track）。
struct UnionFind {
    /// 每个元素的父节点索引，根节点的父节点指向自身
    parent: Vec<usize>,
    /// 按秩合并时记录子树高度（秩）
    rank: Vec<usize>,
}

impl UnionFind {
    /// 创建包含 `n` 个独立元素的并查集。
    ///
    /// # Arguments
    ///
    /// * `n` - 元素个数
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// 查找元素 `x` 所在集合的根节点，并做路径压缩。
    ///
    /// # Arguments
    ///
    /// * `x` - 待查找的元素
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            self.parent[root] = self.parent[self.parent[root]];
            root = self.parent[root];
        }
        root
    }

    /// 合并元素 `x` 与 `y` 所在的集合（按秩合并）。
    ///
    /// # Arguments
    ///
    /// * `x` - 第一个元素
    /// * `y` - 第二个元素
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

/// 并查集 Union-Find 将匹配对连成 Track：全局 ID → 连通分量 → 去重。
///
/// 为每个特征点分配全局 ID，用并查集合并匹配对，将同一连通分量的
/// 特征点归入同一条 Track，并剔除长度不足或同一图像内重复观测的
/// 分量。
///
/// # Arguments
///
/// * `matches` - 所有图像对的匹配
/// * `features` - 所有图像的特征
///
/// # Examples
///
/// ```rust,ignore
/// let tracks = build_tracks(&matches, &features);
/// ```
pub(crate) fn build_tracks(matches: &[Match], features: &[ImageFeatures]) -> Vec<Track> {
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

#[derive(Debug, Deserialize)]
struct CameraPoseEntry {
    image: String,
    #[serde(rename = "R")]
    r: [[f64; 3]; 3],
    #[serde(rename = "t")]
    t: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct CameraPosesFile {
    poses: Vec<CameraPoseEntry>,
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

/// 从文件名序列号推断飞行条带，返回蛇形（来回）布局的节点位置
fn compute_snake_positions(paths: &[String], canvas_size: u32, margin: u32) -> Vec<(f32, f32)> {
    let n = paths.len();
    if n == 0 {
        return Vec::new();
    }

    let seq: Vec<Option<i64>> = paths
        .iter()
        .map(|p| {
            std::path::Path::new(p)
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.rsplit('_').next())
                .and_then(|seg| seg.parse::<i64>().ok())
        })
        .collect();

    // 序列号跳变超过阈值时切分为新条带
    let mut strips: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in 0..n {
        if let Some(last) = cur.last() {
            let gap = seq[i]
                .zip(seq[*last])
                .map(|(a, b)| a - b)
                .unwrap_or(SNAKE_GAP_THRESHOLD + 1);
            if gap > SNAKE_GAP_THRESHOLD {
                strips.push(std::mem::take(&mut cur));
            }
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        strips.push(cur);
    }

    let rows = strips.len().max(1);
    let usable_w = (canvas_size - 2 * margin) as f32;
    let usable_h = (canvas_size - 2 * margin) as f32;
    let row_h = usable_h / rows as f32;

    let mut positions = vec![(0.0f32, 0.0f32); n];
    for (r, strip) in strips.iter().enumerate() {
        let k = strip.len();
        let col_w = usable_w / k.max(1) as f32;
        let y = margin as f32 + row_h * r as f32 + row_h / 2.0;
        for (ci, &idx) in strip.iter().enumerate() {
            let col = if r % 2 == 0 { ci } else { k - 1 - ci };
            let x = margin as f32 + col_w * col as f32 + col_w / 2.0;
            positions[idx] = (x, y);
        }
    }

    positions
}

/// 轨迹覆盖图：节点布局反映无人机飞行路径（俯视图相机位姿或蛇形条带）。
///
/// 若存在 `camera_poses.json` 则按相机中心的 XY 平面俯视投影布局，
/// 否则按文件名序列号推断飞行条带做蛇形布局；边表示两图像间共享
/// 的 Track 数量，节点大小表示关键点数量。
///
/// # Arguments
///
/// * `tracks` - 轨迹列表
/// * `features` - 所有图像的特征
/// * `paths` - 图像文件路径列表
/// * `output_path` - 输出 PNG 路径
pub(crate) fn visualize_track_coverage_graph(
    tracks: &[Track],
    features: &[ImageFeatures],
    paths: &[String],
    output_path: &str,
) {
    println!("Generating track coverage graph...");
    let start = Instant::now();

    let num_images = paths.len();
    let canvas_size: u32 = 2400;
    let margin: u32 = 200;

    // 读取已注册相机的世界坐标中心（camera_poses.json），缺失或未注册为 None
    let mut camera_centers: Vec<Option<Vector3<f64>>> = vec![None; num_images];
    if let Ok(data) = fs::read_to_string("camera_poses.json") {
        if let Ok(parsed) = serde_json::from_str::<CameraPosesFile>(&data) {
            let mut by_name: HashMap<String, Vector3<f64>> = HashMap::new();
            for entry in parsed.poses {
                let r = Matrix3::new(
                    entry.r[0][0],
                    entry.r[0][1],
                    entry.r[0][2],
                    entry.r[1][0],
                    entry.r[1][1],
                    entry.r[1][2],
                    entry.r[2][0],
                    entry.r[2][1],
                    entry.r[2][2],
                );
                let t = Vector3::new(entry.t[0], entry.t[1], entry.t[2]);
                by_name.insert(entry.image, camera_center(&r, &t));
            }
            for (i, path) in paths.iter().enumerate() {
                if let Some(stem) = std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                {
                    if let Some(c) = by_name.get(stem) {
                        camera_centers[i] = Some(*c);
                    }
                }
            }
        }
    }

    let positioned: Vec<usize> = camera_centers
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.map(|_| i))
        .collect();

    // 有相机位置 -> 俯视图 XY 投影；否则按文件名序列号蛇形布局
    let positions: Vec<Option<(f32, f32)>> = if positioned.len() >= 2 {
        let pts: Vec<(f64, f64)> = positioned
            .iter()
            .map(|&i| {
                let c = camera_centers[i].unwrap();
                (c.x, c.y)
            })
            .collect();
        let min_x = pts.iter().map(|p| p.0).fold(f64::MAX, f64::min);
        let max_x = pts.iter().map(|p| p.0).fold(f64::MIN, f64::max);
        let min_y = pts.iter().map(|p| p.1).fold(f64::MAX, f64::min);
        let max_y = pts.iter().map(|p| p.1).fold(f64::MIN, f64::max);
        let span_x = (max_x - min_x).max(1e-6);
        let span_y = (max_y - min_y).max(1e-6);
        let usable = (canvas_size - 2 * margin) as f64;
        let scale = usable / span_x.max(span_y);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        let mut pos = vec![None; num_images];
        for &i in &positioned {
            let c = camera_centers[i].unwrap();
            let px = (canvas_size as f64 / 2.0 + (c.x - center_x) * scale) as f32;
            let py = (canvas_size as f64 / 2.0 - (c.y - center_y) * scale) as f32;
            pos[i] = Some((px, py));
        }
        pos
    } else {
        compute_snake_positions(paths, canvas_size, margin)
            .into_iter()
            .map(Some)
            .collect()
    };

    let layout_mode = if positioned.len() >= 2 {
        "top-down camera path (XY)"
    } else {
        "snake strip (filename sequence)"
    };

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

            let (p1, p2) = match (positions[i], positions[j]) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let thick = (1.0 + intensity * 5.0 * line_thickness) as f32;
            draw_thick_line(&mut canvas, p1, p2, color, thick);
        }
    }

    for (i, pos) in positions.iter().enumerate() {
        let (x, y) = match pos {
            Some(p) => *p,
            None => continue,
        };
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

        let lx = ix + node_r + 6;
        let ly = iy - node_r - 6;

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
        ly += 24;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            &format!(
                "Layout: {}  ·  positioned: {}/{}",
                layout_mode,
                positioned.len(),
                num_images
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
    // 带厚度的反锯齿直线
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
        // 5x7位图字体在RGB图像上绘制ASCII文本
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
        // 构建5x7像素位图字体查找表
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
