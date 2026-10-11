use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::time::Instant;

use image::{GenericImageView, Rgb, RgbImage};

use crate::imgproc::feature_dimensions;
use crate::matching::Match;
use crate::types::{ImageFeatures, MIN_TRACK_LENGTH, SNAKE_GAP_THRESHOLD};

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

/// 匹配点对过少时视为无有效重叠。
const MIN_PAIR_MATCHES: usize = 3;
/// 重叠比例低于此阈值时该图对不绘制重叠（脚印相切）。
const MIN_OVERLAP: f32 = 0.05;

/// 合成飞行高度（单位空间，脚印高为 1）。
const Z_CAM: f32 = 1.5;
/// 中间层高度比例。
const MID_T: f32 = 0.5;
/// 中间层缩放比例。
const MID_S: f32 = 0.25;
/// 等轴测投影角度（度）。
const ISO_ALPHA_DEG: f32 = 30.0;
/// 高度夸张系数。
const ISO_K: f32 = 0.6;

/// 图像对 -> 匹配点坐标列表：`(x_lo, y_lo, x_hi, y_hi)`。
type PairPoints = HashMap<(usize, usize), Vec<(f32, f32, f32, f32)>>;

/// 从文件名序列号推断飞行条带切分，返回每个条带包含的图像索引（按飞行顺序）。
fn infer_strips(paths: &[String]) -> Vec<Vec<usize>> {
    let n = paths.len();
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
    strips
}

/// 收集所有图像对之间的匹配特征点坐标。
///
/// 键为 `(min, max)` 图像索引，值为 `(x_lo, y_lo, x_hi, y_hi)` 列表，其中
/// lo/hi 分别对应索引较小/较大的图像；坐标来自关键点在特征提取帧的像素位置。
fn collect_pair_points(tracks: &[Track], features: &[ImageFeatures]) -> PairPoints {
    let mut map: PairPoints = HashMap::new();
    for track in tracks {
        let obs = &track.observations;
        for a in 0..obs.len() {
            for b in (a + 1)..obs.len() {
                let (ia, ka) = obs[a];
                let (ib, kb) = obs[b];
                let (xa, ya) = (features[ia].keypoints[ka].x, features[ia].keypoints[ka].y);
                let (xb, yb) = (features[ib].keypoints[kb].x, features[ib].keypoints[kb].y);
                let (lo, hi, entry) = if ia <= ib {
                    (ia, ib, (xa, ya, xb, yb))
                } else {
                    (ib, ia, (xb, yb, xa, ya))
                };
                map.entry((lo, hi)).or_default().push(entry);
            }
        }
    }
    map
}

/// 浮点中位数。
fn median(vals: &[f32]) -> Option<f32> {
    if vals.is_empty() {
        return None;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) * 0.5
    } else {
        v[mid]
    })
}

/// 判断主飞行轴是否沿图像 x 轴（横向条带），否则为纵向。
fn is_horizontal_flight(pair_points: &PairPoints) -> bool {
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;
    for pts in pair_points.values() {
        for &(xl, yl, xh, yh) in pts {
            sum_dx += (xl - xh).abs() as f64;
            sum_dy += (yl - yh).abs() as f64;
        }
    }
    sum_dx >= sum_dy
}

/// 用匹配点在两图中各自覆盖范围的重合比例估算两图重叠比例。
///
/// `axis_is_x` 为 true 时沿 x 轴（前向重叠），否则沿 y 轴（侧向重叠）；`extent`
/// 为该轴的图像尺寸（关键点帧）。匹配点过少或无匹配时返回 `None`。
fn overlap_fraction(
    pair_points: &PairPoints,
    a: usize,
    b: usize,
    axis_is_x: bool,
    extent: f32,
) -> Option<f32> {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let pts = pair_points.get(&(lo, hi))?;
    if pts.len() < MIN_PAIR_MATCHES {
        return None;
    }

    let mut lo_a = f32::MAX;
    let mut hi_a = f32::MIN;
    let mut lo_b = f32::MAX;
    let mut hi_b = f32::MIN;
    for &(xl, yl, xh, yh) in pts {
        let (ca, cb) = match (axis_is_x, a == lo) {
            (true, true) => (xl, xh),
            (true, false) => (xh, xl),
            (false, true) => (yl, yh),
            (false, false) => (yh, yl),
        };
        lo_a = lo_a.min(ca);
        hi_a = hi_a.max(ca);
        lo_b = lo_b.min(cb);
        hi_b = hi_b.max(cb);
    }
    let shared = ((hi_a - lo_a) + (hi_b - lo_b)) * 0.5;
    Some((shared / extent).clamp(0.0, 1.0))
}

/// 用匹配点对拟合 2D 相似变换，估计两图相对旋转角（Procrustes）。
fn relative_yaw(pair_points: &PairPoints, a: usize, b: usize) -> f32 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let Some(pts) = pair_points.get(&(lo, hi)) else {
        return 0.0;
    };
    if pts.len() < MIN_PAIR_MATCHES {
        return 0.0;
    }
    let n = pts.len() as f32;
    let mut mx_a = 0.0f32;
    let mut my_a = 0.0f32;
    let mut mx_b = 0.0f32;
    let mut my_b = 0.0f32;
    for &(xl, yl, xh, yh) in pts {
        let (xa, ya, xb, yb) = if a == lo {
            (xl, yl, xh, yh)
        } else {
            (xh, yh, xl, yl)
        };
        mx_a += xa;
        my_a += ya;
        mx_b += xb;
        my_b += yb;
    }
    mx_a /= n;
    my_a /= n;
    mx_b /= n;
    my_b /= n;

    let mut dot = 0.0f32;
    let mut cross = 0.0f32;
    for &(xl, yl, xh, yh) in pts {
        let (xa, ya, xb, yb) = if a == lo {
            (xl, yl, xh, yh)
        } else {
            (xh, yh, xl, yl)
        };
        let dax = xa - mx_a;
        let day = ya - my_a;
        let dbx = xb - mx_b;
        let dby = yb - my_b;
        dot += dax * dbx + day * dby;
        cross += dax * dby - day * dbx;
    }
    cross.atan2(dot)
}

/// 按该对匹配点的主位移轴（|dx| vs |dy|）计算重叠比例。
fn overlap_principal(
    pair_points: &PairPoints,
    a: usize,
    b: usize,
    ext_x: f32,
    ext_y: f32,
) -> Option<f32> {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let pts = pair_points.get(&(lo, hi))?;
    if pts.len() < MIN_PAIR_MATCHES {
        return None;
    }
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;
    for &(xl, yl, xh, yh) in pts {
        sum_dx += (xl - xh).abs() as f64;
        sum_dy += (yl - yh).abs() as f64;
    }
    let axis_is_x = sum_dx >= sum_dy;
    let extent = if axis_is_x { ext_x } else { ext_y };
    overlap_fraction(pair_points, a, b, axis_is_x, extent)
}

/// 覆盖次数 → 热力色（冷色蓝→青→绿→黄绿，避免与红色描边冲突）。
fn coverage_color(c: u32) -> Rgb<u8> {
    match c {
        0 => Rgb([10, 10, 16]),
        1 => Rgb([40, 90, 170]),
        2 => Rgb([60, 140, 230]),
        3 => Rgb([70, 200, 190]),
        4 => Rgb([110, 220, 120]),
        _ => Rgb([200, 230, 90]),
    }
}

fn set_px(canvas: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    let (w, h) = canvas.dimensions();
    if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
        canvas.put_pixel(x as u32, y as u32, color);
    }
}

/// 简单直线（DDA）。
fn draw_line(canvas: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgb<u8>) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).ceil() as i32;
    if steps == 0 {
        set_px(canvas, x0.round() as i32, y0.round() as i32, color);
        return;
    }
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let px = (x0 + dx * t).round() as i32;
        let py = (y0 + dy * t).round() as i32;
        set_px(canvas, px, py, color);
    }
}

/// 虚线（按像素距离做 dash/gap）。
fn draw_dashed_line(
    canvas: &mut RgbImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: Rgb<u8>,
    dash: f32,
    gap: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        return;
    }
    let steps = len.ceil() as i32;
    let period = dash + gap;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        if (t * len) % period < dash {
            let px = (x0 + dx * t).round() as i32;
            let py = (y0 + dy * t).round() as i32;
            set_px(canvas, px, py, color);
        }
    }
}

/// 实心圆点。
fn draw_dot(canvas: &mut RgbImage, cx: f32, cy: f32, r: i32, color: Rgb<u8>) {
    let cx = cx.round() as i32;
    let cy = cy.round() as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                set_px(canvas, cx + dx, cy + dy, color);
            }
        }
    }
}

/// 带厚度的直线（按垂直方向偏移采样）。
fn draw_line_thick(
    canvas: &mut RgbImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: Rgb<u8>,
    thickness: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        let r = (thickness * 0.5).ceil() as i32;
        draw_dot(canvas, x0, y0, r, color);
        return;
    }
    let nx = -dy / len;
    let ny = dx / len;
    let half = thickness * 0.5;
    let steps = len.ceil() as i32;
    let offs = half.floor() as i32;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = x0 + dx * t;
        let cy = y0 + dy * t;
        for o in -offs..=offs {
            let off = o as f32;
            let px = (cx + nx * off).round() as i32;
            let py = (cy + ny * off).round() as i32;
            set_px(canvas, px, py, color);
        }
    }
}

/// 2.5D 视锥覆盖图：条带以网格、环绕以圆形（部分弧）布局合成相机视锥，
/// 脚印朝向由相邻图相对旋转反推，地面足迹按覆盖次数栅格化着色，
/// 等轴测投影统一绘制视锥、轨迹与覆盖。
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
    println!("Generating 2.5D frustum coverage map...");
    let start = Instant::now();

    let num_images = paths.len();
    let canvas_size: u32 = 2400;
    let margin: u32 = 200;

    // 真实图像尺寸决定脚印宽高比；关键点在降采样帧，用其尺寸做重叠归一化
    let (iw_img, ih_img) = image::open(&paths[0])
        .map(|img| img.dimensions())
        .unwrap_or((4000, 3000));
    let (nw, nh) = feature_dimensions(iw_img, ih_img);
    let aspect = iw_img as f32 / ih_img as f32;

    // 条带结构（文件名序列）与匹配点坐标
    let strips = infer_strips(paths);
    let pair_points = collect_pair_points(tracks, features);

    // 轨迹顺序（条带展平）
    let traj: Vec<usize> = strips.iter().flatten().copied().collect();

    // 相对旋转 + 累计转向（检测环绕）+ 累加朝向
    let mut yaw_rel: Vec<f32> = Vec::with_capacity(traj.len().saturating_sub(1));
    for w in traj.windows(2) {
        yaw_rel.push(relative_yaw(&pair_points, w[0], w[1]));
    }
    let total_turn: f32 = yaw_rel.iter().map(|y| y.abs()).sum();
    let orbital = total_turn > 90.0f32.to_radians();

    let mut yaw = vec![0.0f32; num_images];
    let mut acc = 0.0f32;
    for (i, &idx) in traj.iter().enumerate() {
        if i > 0 {
            acc += yaw_rel[i - 1];
        }
        yaw[idx] = acc;
    }

    // 单位脚印尺寸：沿飞行轴为图像长边（真实宽高比）
    let unit_h = 1.0f32;
    let unit_w = aspect;
    let hw = unit_w * 0.5;
    let hh = unit_h * 0.5;

    // 脚印中心（地面）与相机中心（单位空间）
    let mut ground_center = vec![(0.0f32, 0.0f32); num_images];
    let mut cam_center = vec![(0.0f32, 0.0f32); num_images];
    let mut close_loop = false;
    // 脚印「径向」半长：条带为图像高的一半，环绕为 r_g 的一半（使脚印延伸至圆心、覆盖中心）
    let mut radial_half = hh;

    if orbital {
        // 环绕：θ_fov 由内参推（默认焦距），角度步由重叠反推，半径由数据推
        let f_px = nw.max(nh) as f32;
        let theta_fov = 2.0 * (nw as f32 / (2.0 * f_px)).atan();
        let mut o_i = vec![0.3f32; traj.len().saturating_sub(1)];
        for i in 0..o_i.len() {
            if let Some(o) =
                overlap_principal(&pair_points, traj[i], traj[i + 1], nw as f32, nh as f32)
            {
                o_i[i] = o.clamp(0.0, 1.0);
            }
        }
        let mut theta = vec![0.0f32; num_images];
        let mut cum = 0.0f32;
        for (i, &idx) in traj.iter().enumerate() {
            theta[idx] = cum;
            if i + 1 < traj.len() {
                cum += (1.0 - o_i[i]) * theta_fov;
            }
        }
        let span = cum;
        close_loop = span >= 300.0f32.to_radians();
        let r_g = unit_w / theta_fov;
        let r_cam = r_g + 1.0;
        radial_half = r_g * 0.5;
        for &idx in &traj {
            let (s, c) = theta[idx].sin_cos();
            ground_center[idx] = (r_g * 0.5 * c, r_g * 0.5 * s);
            cam_center[idx] = (r_cam * c, r_cam * s);
        }
    } else {
        // 条带：网格布局，前向/侧向重叠反推
        let horizontal = is_horizontal_flight(&pair_points);
        let (fwd_extent, side_extent) = if horizontal {
            (nw as f32, nh as f32)
        } else {
            (nh as f32, nw as f32)
        };

        let mut side_overlaps: Vec<f32> = Vec::new();
        for r in 0..strips.len().saturating_sub(1) {
            let k_max = strips[r].len().min(strips[r + 1].len());
            for k in 0..k_max {
                if let Some(o) = overlap_fraction(
                    &pair_points,
                    strips[r][k],
                    strips[r + 1][k],
                    !horizontal,
                    side_extent,
                ) {
                    side_overlaps.push(o);
                }
            }
        }
        let side_overlap = median(&side_overlaps).unwrap_or(0.3).clamp(0.0, 1.0);

        let mut y_cursor = 0.0f32;
        for strip in &strips {
            let mut x_cursor = 0.0f32;
            for (ci, &idx) in strip.iter().enumerate() {
                ground_center[idx] = (x_cursor + hw, y_cursor + hh);
                if ci + 1 < strip.len() {
                    let next = strip[ci + 1];
                    let o = overlap_fraction(&pair_points, idx, next, horizontal, fwd_extent)
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0);
                    let o = if o < MIN_OVERLAP { 0.0 } else { o };
                    x_cursor += unit_w * (1.0 - o);
                }
            }
            y_cursor += unit_h * (1.0 - side_overlap);
        }
        for &idx in &traj {
            cam_center[idx] = ground_center[idx];
        }
    }

    // 等轴测投影（单位空间）
    let cos_a = ISO_ALPHA_DEG.to_radians().cos();
    let sin_a = ISO_ALPHA_DEG.to_radians().sin();
    let project =
        |x: f32, y: f32, z: f32| -> (f32, f32) { ((x - y) * cos_a, (x + y) * sin_a - z * ISO_K) };

    let z_mid = Z_CAM * MID_T;

    struct Frustum {
        depth: f32,
        ground: [(f32, f32); 4],
        mid: [(f32, f32); 4],
    }

    // 第一遍：按 yaw 旋转脚印、lerp 中间层，投影求包围盒
    let mut min_px = f32::MAX;
    let mut max_px = f32::MIN;
    let mut min_py = f32::MAX;
    let mut max_py = f32::MIN;

    let corner_offsets = [
        (-hw, -radial_half),
        (hw, -radial_half),
        (hw, radial_half),
        (-hw, radial_half),
    ];

    let mut frustums: Vec<Frustum> = Vec::with_capacity(num_images);
    for &idx in &traj {
        let (gx, gy) = ground_center[idx];
        let (cx3, cy3) = cam_center[idx];
        let (sy, cy) = yaw[idx].sin_cos();
        // 中间层中心 = lerp(地面中心, 相机中心, MID_T)
        let mx3 = gx + (cx3 - gx) * MID_T;
        let my3 = gy + (cy3 - gy) * MID_T;

        let mut g = [(0.0f32, 0.0f32); 4];
        let mut m = [(0.0f32, 0.0f32); 4];
        for k in 0..4 {
            let (ox, oy) = corner_offsets[k];
            let rx = ox * cy - oy * sy;
            let ry = ox * sy + oy * cy;
            let gx3 = gx + rx;
            let gy3 = gy + ry;
            let mxk = mx3 + (gx3 - gx) * MID_S;
            let myk = my3 + (gy3 - gy) * MID_S;
            let gp = project(gx3, gy3, 0.0);
            let mp = project(mxk, myk, z_mid);
            g[k] = gp;
            m[k] = mp;
            min_px = min_px.min(gp.0).min(mp.0);
            max_px = max_px.max(gp.0).max(mp.0);
            min_py = min_py.min(gp.1).min(mp.1);
            max_py = max_py.max(gp.1).max(mp.1);
        }
        frustums.push(Frustum {
            depth: gx + gy,
            ground: g,
            mid: m,
        });
    }

    let span_x = (max_px - min_px).max(1e-6);
    let span_y = (max_py - min_py).max(1e-6);
    let usable = (canvas_size - 2 * margin) as f32;
    let scale = (usable / span_x).min(usable / span_y);
    let offset_x = (canvas_size as f32 - span_x * scale) * 0.5;
    let offset_y = (canvas_size as f32 - span_y * scale) * 0.5;
    let to_canvas = |p: (f32, f32)| -> (f32, f32) {
        (
            offset_x + (p.0 - min_px) * scale,
            offset_y + (p.1 - min_py) * scale,
        )
    };

    // 第二遍：转换到画布坐标
    let cams: Vec<Frustum> = frustums
        .into_iter()
        .map(|f| Frustum {
            depth: f.depth,
            ground: [
                to_canvas(f.ground[0]),
                to_canvas(f.ground[1]),
                to_canvas(f.ground[2]),
                to_canvas(f.ground[3]),
            ],
            mid: [
                to_canvas(f.mid[0]),
                to_canvas(f.mid[1]),
                to_canvas(f.mid[2]),
                to_canvas(f.mid[3]),
            ],
        })
        .collect();

    // 中间层中心点（按图像索引，供轨迹使用）
    let mid_centers: Vec<(f32, f32)> = cams
        .iter()
        .map(|f| {
            (
                (f.mid[0].0 + f.mid[2].0) * 0.5,
                (f.mid[0].1 + f.mid[2].1) * 0.5,
            )
        })
        .collect();

    // 覆盖次数栅格化：把每个地面足迹（投影后平行四边形）累加计数
    let mut count = vec![0u32; (canvas_size as usize) * (canvas_size as usize)];
    for f in &cams {
        let p0 = f.ground[0];
        let e1 = (f.ground[1].0 - p0.0, f.ground[1].1 - p0.1);
        let e2 = (f.ground[3].0 - p0.0, f.ground[3].1 - p0.1);
        let cross = e1.0 * e2.1 - e1.1 * e2.0;
        if cross.abs() < 1e-6 {
            continue;
        }
        let minx = f
            .ground
            .iter()
            .map(|p| p.0)
            .fold(f32::MAX, f32::min)
            .floor() as i32;
        let maxx = f.ground.iter().map(|p| p.0).fold(f32::MIN, f32::max).ceil() as i32;
        let miny = f
            .ground
            .iter()
            .map(|p| p.1)
            .fold(f32::MAX, f32::min)
            .floor() as i32;
        let maxy = f.ground.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil() as i32;
        for py in miny.max(0)..=maxy.min(canvas_size as i32 - 1) {
            for px in minx.max(0)..=maxx.min(canvas_size as i32 - 1) {
                let dx = px as f32 - p0.0;
                let dy = py as f32 - p0.1;
                let u = (dx * e2.1 - dy * e2.0) / cross;
                let v = (e1.0 * dy - e1.1 * dx) / cross;
                if u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 {
                    count[py as usize * canvas_size as usize + px as usize] += 1;
                }
            }
        }
    }

    // 地面层：按覆盖次数着色
    let mut canvas = RgbImage::new(canvas_size, canvas_size);
    for py in 0..canvas_size {
        for px in 0..canvas_size {
            let c = count[py as usize * canvas_size as usize + px as usize];
            canvas.put_pixel(px, py, coverage_color(c));
        }
    }

    // 深度排序（远→近）
    let mut order: Vec<usize> = (0..cams.len()).collect();
    order.sort_by(|&a, &b| {
        cams[a]
            .depth
            .partial_cmp(&cams[b].depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid_color = Rgb([255, 60, 60]);
    let edge_color = Rgb([130, 200, 255]);

    // 依次绘制每个视锥：中间层矩形（红色加宽描边）+ 中间层→地面虚线
    for &i in &order {
        let f = &cams[i];
        for k in 0..4 {
            let a = f.mid[k];
            let b = f.mid[(k + 1) % 4];
            draw_line_thick(&mut canvas, a.0, a.1, b.0, b.1, mid_color, 2.0);
        }
        for k in 0..4 {
            draw_dashed_line(
                &mut canvas,
                f.mid[k].0,
                f.mid[k].1,
                f.ground[k].0,
                f.ground[k].1,
                edge_color,
                4.0,
                3.0,
            );
        }
    }

    // 飞行轨迹：按轨迹顺序连接中间层中心点；环绕时闭环
    let traj_color = Rgb([255, 210, 80]);
    for w in traj.windows(2) {
        let a = mid_centers[w[0]];
        let b = mid_centers[w[1]];
        draw_line(&mut canvas, a.0, a.1, b.0, b.1, traj_color);
    }
    if close_loop && traj.len() > 2 {
        let a = mid_centers[traj[traj.len() - 1]];
        let b = mid_centers[traj[0]];
        draw_line(&mut canvas, a.0, a.1, b.0, b.1, traj_color);
    }

    // 中间层矩形中心点（红色圆点）
    let center_dot_color = Rgb([255, 40, 40]);
    for &p in &mid_centers {
        draw_dot(&mut canvas, p.0, p.1, 4, center_dot_color);
    }

    // 图例
    {
        let lx = 40;
        let mut ly = canvas_size as i32 - 280;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "2.5D FRUSTUM COVERAGE",
            Rgb([180, 190, 220]),
        );
        ly += 24;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            &format!(
                "{} images - {} tracks - mode {}",
                num_images,
                tracks.len(),
                if orbital { "orbital" } else { "strip" }
            ),
            Rgb([120, 130, 160]),
        );
        ly += 24;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "Ground color = overlap count",
            Rgb([100, 110, 140]),
        );
        ly += 20;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "Red outline = mid layer",
            Rgb([100, 110, 140]),
        );
        ly += 20;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "Red dot = mid layer center",
            Rgb([100, 110, 140]),
        );
        ly += 20;
        draw_text_simple(
            &mut canvas,
            lx,
            ly,
            "Yellow line = flight trajectory (mid layer)",
            Rgb([100, 110, 140]),
        );
        ly += 26;
        let mut sx = lx;
        for (c, label) in [(1u32, "1"), (2, "2"), (3, "3"), (4, "4")] {
            draw_dot(
                &mut canvas,
                sx as f32 + 5.0,
                ly as f32 + 3.0,
                4,
                coverage_color(c),
            );
            draw_text_simple(&mut canvas, sx + 14, ly, label, Rgb([150, 160, 190]));
            sx += 34;
        }
    }

    if let Some(parent) = std::path::Path::new(output_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("Failed to create coverage output directory");
        }
    }
    canvas
        .save(output_path)
        .expect("Failed to save coverage graph");

    let elapsed = start.elapsed();
    println!(
        "Coverage map saved to {} ({:.1} KB) in {:.2}s\n",
        output_path,
        fs::metadata(output_path)
            .map(|m| m.len() as f64 / 1024.0)
            .unwrap_or(0.0),
        elapsed.as_secs_f32(),
    );
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
