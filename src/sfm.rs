use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::RgbImage;
use nalgebra::{self as na, Matrix3, Matrix3x4, Vector3, SVD};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::imgproc::feature_dimensions;
use crate::matching::Match;
use crate::tracks::Track;
use crate::types::{ImageFeatures, RANSAC_SEED};

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

// 选图时的序列邻近权重：grow 优先注册「离最近一张图 index 最近」的图（同航线前向延续），
// 避免被重复纹理/高纹理的跨航线图顶到队首、把航线间距拼乱。对蛇形航线，同航线图 index 相邻，
// 跨航线图 index 差 ~45-60；该权重让前向图（|Δindex|≈1）稳定压过跨航线图（|Δindex|≈60）。
const SEQ_PROXIMITY_WEIGHT: f64 = 50.0;

// 多子模型重建：图像对匹配数达到该阈值才作为子模型种子
const SUB_MODEL_MIN_MATCHES: usize = 50;

// 子模型合并：无共享相机、仅靠薄接缝共享 track 时，可接受的最小内点数。
// 跨线（180° 转向、相邻航线只在掉头处重叠）真对应点本就不多，但 3 个内点
// 会让 1~2 条假 track 就通过门槛，把航线在错误的相对位姿下拼起来（实测
// 13 条航线被拼乱、整体塌缩到 ~1/10 尺度）。10 个内点是安全下限。
const MIN_MERGE_SHARED_TRACKS: usize = 10;

// 纯 track（无共享相机）合并时，至少需要这么多共享 track 才尝试对齐。
// （当前循环式合并用 collect_merge_anchors 的 bn<4 门槛，此常量仅作参考保留。）
#[allow(dead_code)]
const MIN_TRACK_ONLY_ALIGN: usize = 4;

// Sim3 合并的尺度范围保护：子模型尺度任意（无 GPS），但不应极端。
const MERGE_SCALE_MIN: f64 = 0.1;
const MERGE_SCALE_MAX: f64 = 10.0;

// 增量注册时，每注册 N 张图才做一次全局 BA（局部 BA 每张都做），加速重建
const GLOBAL_BA_EVERY: usize = 5;
// 诊断开关：置 true 关闭 grow 阶段的全局 BA（含每 N 张与 grow 结束），只剩 local BA + PnP，
// 用来判断航线被拼乱是全局 BA 导致的还是注册本身导致的。定位后再改回 false。
const DISABLE_GLOBAL_BA_DIAG: bool = false;

// 每个子模型最多额外注册多少张图（不含种子 2 张）。航线每条约 7~8 张，设为 6 让每个
// 子模型严格只覆盖一条航线（2+6=8 张），跨航线对齐全部交给 merge（用全部跨航线匹配 +
// RANSAC，比增量 PnP 更鲁棒）。这是多子模型思路的关键：限制单个子模型的规模，避免
// 一条增量链跨多条航线累积漂移。
const MAX_GROW_IMAGES: usize = 6;

// 调平开关：合并前把每个子模型的相机视向（R^T e_z）对齐到共同参考。对陡坡地形这会
// 把山坡起伏一起转掉、压扁 z（实测 z 从 94m 压到 40m），先置 false 验证 z 是否恢复。
const ENABLE_LEVELING: bool = false;

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

const MAX_REFINEMENTS: usize = 8;
const MAX_REFINEMENT_CHANGE: f64 = 0.01;

const BA_LOCAL_NUM_IMAGES: usize = 6;
const BA_LOCAL_MAX_ITERATIONS: usize = 25;
const BA_GLOBAL_MAX_ITERATIONS: usize = 400;

const MIN_FOCAL_LENGTH_RATIO: f64 = 0.1;
const MAX_FOCAL_LENGTH_RATIO: f64 = 10.0;

pub(crate) const RANSAC_ITERATIONS: usize = 2000;
// 实验：放宽几何过滤阈值（4→8px），观察北部重复纹理区的匹配是否变多、稀疏点是否变均衡。
// 若有效，再按区域自适应细化；若无效则说明是重复纹理假匹配的硬天花板。
pub(crate) const RANSAC_THRESHOLD_PX: f64 = 8.0;

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
#[derive(Clone)]
pub(crate) struct CameraIntrinsics {
    pub(crate) k: Matrix3<f64>,
    has_prior_focal: bool,
}

impl CameraIntrinsics {
    pub(crate) fn from_image_dimensions(w: u32, h: u32) -> Self {
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
            // 优先用 35mm 等效焦距（EXIF 0xA405），与传感器尺寸无关、最可靠：
            // f_px = 图像宽 × (focal35 / 36mm 全画幅宽)。避免依赖易错的传感器宽度猜测。
            if let Some(fl35) = read_exif_focal_length_35mm(&data) {
                let f_from_exif = wf * fl35 / 36.0;
                if f_from_exif > 500.0 && f_from_exif < 100000.0 && f_from_exif > wf * 0.1 {
                    f_px = f_from_exif;
                    has_prior = true;
                }
            }
            // 兜底：物理焦距(mm) + 传感器宽度，仅在没有 35mm 等效焦距时使用
            if !has_prior {
                if let Some(fl_mm) = read_exif_focal_length(&data) {
                    let sensor_w_mm = 6.2;
                    let f_from_exif = wf * fl_mm / sensor_w_mm;
                    if f_from_exif > 500.0 && f_from_exif < 100000.0 && f_from_exif > wf * 0.1 {
                        f_px = f_from_exif;
                        has_prior = true;
                    }
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

fn read_exif_focal_tag(data: &[u8], target_tag: u16) -> Option<f64> {
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
            if tag == target_tag {
                let typ = read_u16(ep + 2);
                let count = read_u32(ep + 4);
                if typ == 3 && count >= 1 {
                    // SHORT：值内联存储在 4 字节值域的低 16 位（如 FocalLengthIn35mm）
                    return Some(read_u16(ep + 8) as f64);
                } else if typ == 5 && count >= 1 {
                    // RATIONAL：值存于偏移处，8 字节 num/den（如 FocalLength）
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

/// 物理焦距(mm)，EXIF 0x920A。
fn read_exif_focal_length(data: &[u8]) -> Option<f64> {
    read_exif_focal_tag(data, 0x920A)
}

/// 35mm 等效焦距，EXIF 0xA405。
fn read_exif_focal_length_35mm(data: &[u8]) -> Option<f64> {
    read_exif_focal_tag(data, 0xA405)
}

/// 读取 EXIF GPS（纬度/经度/海拔），返回 `(lat_deg, lon_deg, alt_m)`。无 GPS 时返回 `None`。
/// 解析 GPS IFD（EXIF 0x8825 指向），读 GPSLatitude(0x0002)/GPSLongitude(0x0004)（3 个
/// RATIONAL：度/分/秒）与 GPSAltitude(0x0006)，并处理 N/S、E/W 符号。
/// 当前未启用（用户要求不依赖 GPS），保留供可选米制对齐参考。
#[allow(dead_code)]
fn read_exif_gps(data: &[u8]) -> Option<(f64, f64, f64)> {
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
    let tiff = exif_start + 6;
    if tiff + 8 > data.len() {
        return None;
    }
    let le = &data[tiff..tiff + 2] == b"\x49\x49";
    let read_u16 = |p: usize| -> u16 {
        if le {
            data[p] as u16 | ((data[p + 1] as u16) << 8)
        } else {
            ((data[p] as u16) << 8) | data[p + 1] as u16
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

    let ifd0 = tiff + read_u32(tiff + 4) as usize;
    let n0 = read_u16(ifd0) as usize;
    let mut gps_ifd = None;
    let mut ep = ifd0 + 2;
    for _ in 0..n0.min(200) {
        if ep + 12 > data.len() {
            break;
        }
        if read_u16(ep) == 0x8825 {
            gps_ifd = Some(tiff + read_u32(ep + 8) as usize);
            break;
        }
        ep += 12;
    }
    let gps = gps_ifd?;
    if gps + 2 > data.len() {
        return None;
    }

    let read_rat = |p: usize| -> f64 {
        let num = read_u32(p) as f64;
        let den = read_u32(p + 4) as f64;
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    };

    let n = read_u16(gps) as usize;
    let mut lat: Option<f64> = None;
    let mut lon: Option<f64> = None;
    let mut alt: Option<f64> = None;
    let mut lat_ref = b'N';
    let mut lon_ref = b'E';
    let mut ep = gps + 2;
    for _ in 0..n.min(64) {
        if ep + 12 > data.len() {
            break;
        }
        let tag = read_u16(ep);
        let typ = read_u16(ep + 2);
        let count = read_u32(ep + 4);
        match tag {
            0x0001 => lat_ref = data[ep + 8],
            0x0003 => lon_ref = data[ep + 8],
            0x0002 if typ == 5 && count >= 3 => {
                let vp = tiff + read_u32(ep + 8) as usize;
                if vp + 24 <= data.len() {
                    lat = Some(read_rat(vp) + read_rat(vp + 8) / 60.0 + read_rat(vp + 16) / 3600.0);
                }
            }
            0x0004 if typ == 5 && count >= 3 => {
                let vp = tiff + read_u32(ep + 8) as usize;
                if vp + 24 <= data.len() {
                    lon = Some(read_rat(vp) + read_rat(vp + 8) / 60.0 + read_rat(vp + 16) / 3600.0);
                }
            }
            0x0006 if typ == 5 && count >= 1 => {
                let vp = tiff + read_u32(ep + 8) as usize;
                if vp + 8 <= data.len() {
                    alt = Some(read_rat(vp));
                }
            }
            _ => {}
        }
        ep += 12;
    }

    let mut lat = lat?;
    let mut lon = lon?;
    if lat_ref == b'S' {
        lat = -lat;
    }
    if lon_ref == b'W' {
        lon = -lon;
    }
    Some((lat, lon, alt.unwrap_or(0.0)))
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

pub(crate) fn camera_center(r: &Matrix3<f64>, t: &Vector3<f64>) -> Vector3<f64> {
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

pub(crate) fn ransac_fundamental(
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
        let sample: Vec<usize> = sample(&mut rng, n, 6).into_vec();
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
            println!("    [seed] ({},{}) reject: {} matches < {}", first, second, corrs.len(), min_num_inliers);
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

        let g = match estimate_two_view_geometry(&pts1, &pts2, &self.intrinsics, min_num_inliers) {
            Some(g) => g,
            None => {
                println!("    [seed] ({},{}) reject: two-view geometry failed", first, second);
                return None;
            }
        };
        if g.inliers.len() < min_num_inliers {
            println!("    [seed] ({},{}) reject: F inliers {} < {}", first, second, g.inliers.len(), min_num_inliers);
            return None;
        }
        let fwd = g.cam2_from_cam1.t.z.abs() / g.cam2_from_cam1.t.norm().max(1e-12);
        if fwd > INIT_MAX_FORWARD_MOTION {
            println!("    [seed] ({},{}) reject: forward motion {:.2} > {}", first, second, fwd, INIT_MAX_FORWARD_MOTION);
            return None;
        }
        if g.tri_angle < min_tri_angle {
            println!("    [seed] ({},{}) reject: tri angle {:.2}° < {:.2}°", first, second, g.tri_angle.to_degrees(), min_tri_angle.to_degrees());
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

        self.initialize_with_pair(first, second, &geometry, tracks, features, index)
    }

    fn initialize_with_pair(
        &mut self,
        first: usize,
        second: usize,
        geometry: &TwoViewGeometry,
        tracks: &[Track],
        features: &[ImageFeatures],
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) -> SfmStatus {
        self.cameras[first] = Some(CameraPose {
            r: Matrix3::identity(),
            t: Vector3::zeros(),
        });
        self.cameras[second] = Some(geometry.cam2_from_cam1.clone());
        self.reg_trials[first] += 1;
        self.reg_trials[second] += 1;

        self.triangulate_image(first, tracks, features, index);
        self.triangulate_image(second, tracks, features, index);
        let after_tri = self
            .points
            .iter()
            .filter(|p| p.track_id != usize::MAX)
            .count();

        // 两视图位姿是可信的（recover_pose 已过 cheirality）。下面的初始对 BA 有尺度 gauge，
        // 对前向航线的种子对（如 16,17）会塌缩到 baseline=0 → 点全错 → filter 全删。
        // 保存 BA 前的第二相机位姿，若 BA 塌缩就回退重三角化。
        let cam_second_pre = self.cameras[second].clone();

        self.adjust_global_bundle_fixed(
            tracks,
            features,
            LossType::SoftL1,
            BA_GLOBAL_MAX_ITERATIONS,
            Some(first),
        );
        let after_ba = self
            .points
            .iter()
            .filter(|p| p.track_id != usize::MAX)
            .count();

        // 注意：不再做 L1 归一化。L1 归一化会把每个子模型按「各自点分布的平均半径」缩放到
        // 单位尺度，导致不同子模型尺度不一致（实测相差 0.6~1.7 倍），刚体 merge（scale=1）
        // 因此对不上（robust_sim3 全部 "no rigid consensus"）。所有子模型用同一相机、同一
        // 焦距，本质矩阵给出的单位基线尺度本就一致，直接保留这个一致尺度即可。
        // self.normalize();
        // BA 塌缩检测：若第二相机中心与第一相机中心重合（baseline→0），说明初始对 BA
        // 掉进了尺度 gauge 的退化解，回退到两视图位姿并重三角化。
        {
            let c0 = camera_center(
                &self.cameras[first].as_ref().unwrap().r,
                &self.cameras[first].as_ref().unwrap().t,
            );
            let c1 = camera_center(
                &self.cameras[second].as_ref().unwrap().r,
                &self.cameras[second].as_ref().unwrap().t,
            );
            let baseline = (c1 - c0).norm();
            if baseline < 1e-4 {
                println!(
                    "    [seed] ({},{}) BA collapsed baseline={:.6}, reverting to two-view pose",
                    first, second, baseline
                );
                self.cameras[second] = cam_second_pre;
                self.points.clear();
                self.rebuild_track_to_point();
                self.triangulate_image(first, tracks, features, index);
                self.triangulate_image(second, tracks, features, index);
                // 保持与主路径一致：不做 L1 归一化（见上方注释），维持本质矩阵的一致尺度。
            }
        }
        // 诊断：过滤前统计初始对位姿/重投影/角度，定位 filter 全删点（16,17 被全删、37,69 只删 33）的原因
        {
            let cx = self.cx();
            let cy = self.cy();
            let c0 = camera_center(
                &self.cameras[first].as_ref().unwrap().r,
                &self.cameras[first].as_ref().unwrap().t,
            );
            let c1 = camera_center(
                &self.cameras[second].as_ref().unwrap().r,
                &self.cameras[second].as_ref().unwrap().t,
            );
            let baseline = (c1 - c0).norm();
            let mut errs: Vec<f64> = Vec::new();
            let mut angs: Vec<f64> = Vec::new();
            for p in self.points.iter().filter(|p| p.track_id != usize::MAX) {
                let track = &tracks[p.track_id];
                let mut best_e = f64::INFINITY;
                for &(im, kp) in &track.observations {
                    if let Some(cam) = self.cameras[im].as_ref() {
                        if let Some((u, v, _)) =
                            project_point(&cam.r, &cam.t, &p.xyz, self.focals[im], cx, cy)
                        {
                            let kd = &features[im].keypoints[kp];
                            let e = ((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt();
                            best_e = best_e.min(e);
                        }
                    }
                }
                if best_e.is_finite() {
                    errs.push(best_e);
                }
                let v0 = (p.xyz - c0).normalize();
                let v1 = (p.xyz - c1).normalize();
                angs.push(v0.dot(&v1).clamp(-1.0, 1.0).acos().to_degrees());
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            angs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_e = if errs.is_empty() { -1.0 } else { errs[errs.len() / 2] };
            let med_a = if angs.is_empty() { -1.0 } else { angs[angs.len() / 2] };
            println!(
                "    [seed] ({},{}) pre-filter: baseline={:.4} med_reproj={:.2}px med_angle={:.2}deg",
                first, second, baseline, med_e, med_a
            );
        }
        let removed = self.filter_points3d(tracks, features);
        self.filter_frames(tracks);
        println!(
            "    [seed] ({},{}) tri={} pts, after_ba={}, filter_removed={}",
            first, second, after_tri, after_ba, removed
        );

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
            let np = self.points.iter().filter(|p| p.track_id != usize::MAX).count();
            println!(
                "    [seed] ({},{}) reject: init produced {} points (< {}), {} registered",
                first,
                second,
                np,
                ABS_POSE_MIN_NUM_INLIERS,
                self.num_registered()
            );
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
    fn find_next_images(
        &self,
        tracks: &[Track],
        features: &[ImageFeatures],
        last_img: Option<usize>,
    ) -> Vec<usize> {
        let mut scored: Vec<(f64, usize)> = (0..self.num_images)
            .into_par_iter()
            .filter(|&img| {
                self.cameras[img].is_none()
                    && !self.filtered[img]
                    && self.reg_trials[img] < MAX_REG_TRIALS
            })
            .filter_map(|img| {
                let visible = self.visible_points_count(tracks, img);
                if visible < ABS_POSE_MIN_NUM_INLIERS {
                    return None;
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
                // 序列邻近软惩罚：优先同航线前向延续（index 相邻），其次最近的相邻航线。
                let seq_penalty = match last_img {
                    Some(last) => (img as i64 - last as i64).abs() as f64 * SEQ_PROXIMITY_WEIGHT,
                    None => 0.0,
                };
                Some((score - penalty - seq_penalty, img))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, img)| img).collect()
    }

    // -----------------------------------------------------------------
    //  Structure-less registration candidates
    // -----------------------------------------------------------------
    // Unlike find_next_images, this does NOT require `visible >=
    // ABS_POSE_MIN_NUM_INLIERS` structure points. It returns every
    // unregistered image that has strong *direct* feature matches
    // (>= 2*ABS_POSE_MIN_NUM_INLIERS) against at least one already
    // registered image, ordered by strongest match first. This lets a
    // sub-model grow across a weak seam (few triangulated points but
    // plenty of verified two-view matches) so that neighbouring
    // sub-models overlap on real cameras — which is what makes the
    // later Sim3 merge accurate.
    fn structure_less_candidates(
        &self,
        index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) -> Vec<usize> {
        let registered = self.registered_images();
        let mut scored: Vec<(usize, usize)> = (0..self.num_images)
            .into_par_iter()
            .filter(|&img| {
                self.cameras[img].is_none()
                    && !self.filtered[img]
                    && self.reg_trials[img] < MAX_REG_TRIALS
            })
            .filter_map(|img| {
                let mut best = 0usize;
                for &ref_img in &registered {
                    let key = if img < ref_img { (img, ref_img) } else { (ref_img, img) };
                    let n = index.get(&key).map(|v| v.len()).unwrap_or(0);
                    if n > best {
                        best = n;
                    }
                }
                if best >= 2 * ABS_POSE_MIN_NUM_INLIERS {
                    Some((best, img))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
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
        // 并行计算各 track 的三角化结果（只读），再串行应用 push/insert 保序
        let results: Vec<Option<Vector3<f64>>> = tracks
            .par_iter()
            .enumerate()
            .map(|(track_id, track)| {
                if self.track_to_point.contains_key(&track_id) {
                    return None;
                }
                if track
                    .observations
                    .binary_search_by_key(&img, |&(im, _)| im)
                    .is_err()
                {
                    return None;
                }
                let obs: Vec<(usize, usize)> = track
                    .observations
                    .iter()
                    .filter(|(im, _)| self.cameras[*im].is_some())
                    .copied()
                    .collect();
                if obs.len() < 2 {
                    return None;
                }
                self.triangulate_track(track_id, &obs, features)
            })
            .collect();

        for (track_id, pt) in results.into_iter().enumerate() {
            if let Some(pt) = pt {
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
        // 并行计算每点重三角化结果（只读），再串行应用与计数
        let s = &*self;
        let results: Vec<Option<Vector3<f64>>> = s
            .points
            .par_iter()
            .enumerate()
            .map(|(pidx, _pt)| {
                if !s.point_is_valid(pidx) {
                    return None;
                }
                let track_id = s.points[pidx].track_id;
                let track = &tracks[track_id];
                let obs: Vec<(usize, usize)> = track
                    .observations
                    .iter()
                    .filter(|(im, _)| s.cameras[*im].is_some())
                    .copied()
                    .collect();
                if obs.len() < 3 {
                    return None;
                }
                s.triangulate_track(track_id, &obs, features)
                    .and_then(|pt| {
                        let mut errs: Vec<f64> = Vec::new();
                        for &(im, kp) in &obs {
                            let cam = s.cameras[im].as_ref().unwrap();
                            let f = s.focals[im];
                            if let Some((u, v, _)) =
                                project_point(&cam.r, &cam.t, &pt, f, s.cx(), s.cy())
                            {
                                let kd = &features[im].keypoints[kp];
                                errs.push(
                                    ((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt(),
                                );
                            }
                        }
                        if errs.is_empty() {
                            return None;
                        }
                        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median_err = errs[errs.len() / 2];
                        if median_err > COMPLETE_MAX_REPROJ_ERROR_PX {
                            None
                        } else {
                            Some(pt)
                        }
                    })
            })
            .collect();

        let mut changed = 0usize;
        for (pidx, opt) in results.into_iter().enumerate() {
            if let Some(pt) = opt {
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
        let cameras = &self.cameras;
        let focals = &self.focals;
        let cx = self.cx();
        let cy = self.cy();
        let removed = AtomicUsize::new(0);

        self.points.par_iter_mut().for_each(|pt| {
            if pt.track_id == usize::MAX {
                return;
            }
            let track = &tracks[pt.track_id];
            let xyz = pt.xyz;

            let mut errs: Vec<f64> = Vec::new();
            let mut behind = 0usize;
            let mut reg_ims: Vec<usize> = Vec::new();
            for &(im, kp) in &track.observations {
                if cameras[im].is_none() {
                    continue;
                }
                reg_ims.push(im);
                let cam = cameras[im].as_ref().unwrap();
                let f = focals[im];
                match project_point(&cam.r, &cam.t, &xyz, f, cx, cy) {
                    Some((u, v, _)) => {
                        let kd = &features[im].keypoints[kp];
                        errs.push(((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt());
                    }
                    None => behind += 1,
                }
            }

            let invalidate = |pt: &mut Point3D| {
                pt.track_id = usize::MAX;
                pt.xyz = Vector3::zeros();
                pt.error = f64::MAX;
                removed.fetch_add(1, Ordering::Relaxed);
            };

            let valid_obs = errs.len();
            if valid_obs < 2 {
                invalidate(pt);
                return;
            }

            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_err = errs[errs.len() / 2];
            pt.error = median_err;

            if behind > valid_obs {
                invalidate(pt);
                return;
            }

            if reg_ims.len() >= 2 {
                let c0 = cameras[reg_ims[0]].as_ref().unwrap();
                let c1 = cameras[reg_ims[1]].as_ref().unwrap();
                let angle = calc_triangulation_angle(&c0.r, &c0.t, &c1.r, &c1.t, &xyz);
                if angle < FILTER_MIN_TRI_ANGLE_DEG.to_radians() {
                    invalidate(pt);
                    return;
                }
            }

            if median_err > FILTER_MAX_REPROJ_ERROR_PX * 2.0 {
                invalidate(pt);
            }
        });

        let removed = removed.into_inner();
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
        let cameras = &self.cameras;
        let focals = &self.focals;
        let cx = self.cx();
        let cy = self.cy();
        self.points.par_iter_mut().for_each(|pt| {
            if pt.track_id == usize::MAX {
                return;
            }
            let track = &tracks[pt.track_id];
            let xyz = pt.xyz;
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for &(im, kp) in &track.observations {
                if let Some(cam) = &cameras[im] {
                    let f = focals[im];
                    if let Some((u, v, _)) = project_point(&cam.r, &cam.t, &xyz, f, cx, cy) {
                        let kd = &features[im].keypoints[kp];
                        sum += ((u - kd.x as f64).powi(2) + (v - kd.y as f64).powi(2)).sqrt();
                        cnt += 1;
                    }
                }
            }
            pt.error = if cnt > 0 { sum / cnt as f64 } else { f64::MAX };
        });
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
        self.adjust_global_bundle_fixed(tracks, features, loss, max_iterations, None);
    }

    fn adjust_global_bundle_fixed(
        &mut self,
        tracks: &[Track],
        features: &[ImageFeatures],
        loss: LossType,
        max_iterations: usize,
        fixed_cam: Option<usize>,
    ) {
        self.filter_negative_depth(tracks);
        let active = self.registered_images();
        if active.is_empty() {
            return;
        }
        let fixed = match fixed_cam {
            Some(fc) => vec![fc],
            None => vec![active[0]],
        };
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
        let registered = self.registered_images();
        let mut scores: Vec<(usize, usize)> = registered
            .into_par_iter()
            .filter(|&other| other != img)
            .map(|other| {
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
                (shared, other)
            })
            .collect();
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
                    LossType::SoftL1,
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
    let ncp = nc * 7;
    let cx = sfm.cx();
    let cy = sfm.cy();

    let mut lambda = 1e-6f64;
    let mut best_cost = f64::MAX;

    for _it in 0..max_iterations {
        let mut u = vec![na::SMatrix::<f64, 7, 7>::zeros(); nc];
        let mut ec = vec![na::SVector::<f64, 7>::zeros(); nc];
        let mut wpairs: Vec<Vec<(usize, na::SMatrix<f64, 7, 3>)>> = vec![Vec::new(); nc];
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
                let z = xc.z.max(1e-2);
                let zi = 1.0 / z;
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
                let mut jc = na::SMatrix::<f64, 2, 7>::zeros();
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
                jc[(0, 6)] = xc.x * zi;
                jc[(1, 6)] = xc.y * zi;

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
            let cr = ci * 7;
            for r in 0..7 {
                for c in 0..7 {
                    s[(cr + r, cr + c)] += u[ci][(r, c)];
                }
            }
            for r in 0..7 {
                es[cr + r] += ec[ci][r];
            }
            for (pi, w) in &wpairs[ci] {
                for wr in 0..7 {
                    for wc in 0..7 {
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
                let cr = ci * 7;
                for d in 0..6 {
                    s[(cr + d, cr + d)] = 1e10;
                    es[cr + d] = 0.0;
                }
            }
            // 有 EXIF 焦距先验的相机：焦距维度固定；否则焦距参与自标定
            if sfm.has_prior_focal[active_cams[ci]] {
                s[(ci * 7 + 6, ci * 7 + 6)] = 1e10;
                es[ci * 7 + 6] = 0.0;
            }
        }
        for d in 0..ncp {
            s[(d, d)] += lambda * s[(d, d)].abs().max(1e-6);
        }
        let delta_c = match s.lu().solve(&(-es)) {
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
        let backup_focals: Vec<f64> = active_cams.iter().map(|&im| sfm.focals[im]).collect();
        let backup_pts: Vec<Vector3<f64>> = pts.iter().map(|&pidx| sfm.points[pidx].xyz).collect();

        for ci in 0..nc {
            if fixed[ci] {
                continue;
            }
            let im = active_cams[ci];
            let cam = sfm.cameras[im].as_mut().unwrap();
            let cr = ci * 7;
            let dw = Vector3::new(delta_c[cr], delta_c[cr + 1], delta_c[cr + 2]);
            let dt = Vector3::new(delta_c[cr + 3], delta_c[cr + 4], delta_c[cr + 5]);
            cam.r = so3_exp(&dw) * cam.r;
            cam.t += dt;
            // 焦距自标定：无 EXIF 先验时更新，并限制在合理范围内防止发散
            if !sfm.has_prior_focal[im] {
                let f = sfm.focals[im];
                sfm.focals[im] =
                    (f + delta_c[cr + 6]).clamp(f * MIN_FOCAL_LENGTH_RATIO, f * MAX_FOCAL_LENGTH_RATIO);
            }
        }
        // 先一趟按 wpairs 累加每个点的 dp（O(n_obs)，避免逐点重复扫描）
        let mut dp = vec![na::SVector::<f64, 3>::zeros(); np];
        for ci in 0..nc {
            for (pj, w) in &wpairs[ci] {
                for rc in 0..3 {
                    for row in 0..7 {
                        dp[*pj][rc] += w[(row, rc)] * delta_c[ci * 7 + row];
                    }
                }
            }
        }
        for pi in 0..np {
            let pidx = pts[pi];
            let mut corr = na::SVector::<f64, 3>::zeros();
            for r in 0..3 {
                for c in 0..3 {
                    corr[r] += vpinv[pi][(r, c)] * (-ep[pi][c] - dp[pi][c]);
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
                {
                    let z = xc.z.max(1e-2);
                    let up = f * xc.x / z + cx;
                    let vp = f * xc.y / z + cy;
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
                sfm.focals[im] = backup_focals[ci];
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
fn grow_sub_model(
    sfm: &mut Sfm,
    index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
) {
    let mut prev_success = true;
    let mut reg_count = 0usize;
    loop {
        let mut success = false;
        // 从已注册图里取 index 最大的作为「前沿」（种子对通常是连续两张，取第二张），
        // 这样首次选图就能用序列邻近惩罚优先同航线前向延续，而不是跳去跨航线图。
        let mut last_img = (0..sfm.num_images).rev().find(|&i| sfm.cameras[i].is_some());

        for structure_less in [false, true] {
            let next = if structure_less {
                sfm.structure_less_candidates(index)
            } else {
                sfm.find_next_images(tracks, features, last_img)
            };
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
            reg_count += 1;
            if reg_count % GLOBAL_BA_EVERY == 0 && !DISABLE_GLOBAL_BA_DIAG {
                sfm.iterative_refinement(tracks, features, true, None);
                // 诊断：打印全局 BA 后相机中心 bbox，观察尺度/结构在 grow 过程中从哪一步开始塌
                let mut lo = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut hi = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for cam in sfm.cameras.iter().flatten() {
                    let c = camera_center(&cam.r, &cam.t);
                    lo.x = lo.x.min(c.x); lo.y = lo.y.min(c.y); lo.z = lo.z.min(c.z);
                    hi.x = hi.x.max(c.x); hi.y = hi.y.max(c.y); hi.z = hi.z.max(c.z);
                }
                println!(
                    "    [grow-bbox] after {} reg: x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]",
                    reg_count, lo.x, hi.x, lo.y, hi.y, lo.z, hi.z
                );
            }
            if reg_count >= MAX_GROW_IMAGES {
                println!(
                    "  grow stopped at MAX_GROW_IMAGES={} (sub-model {} images)",
                    MAX_GROW_IMAGES,
                    sfm.num_registered()
                );
                break;
            }
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

    // 结束前做一次全局 BA 优化
    if !DISABLE_GLOBAL_BA_DIAG {
        sfm.iterative_refinement(tracks, features, true, None);
    }
    sfm.update_point_errors(tracks, features);
}

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

    grow_sub_model(sfm, index, features, tracks, paths);
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

/// Umeyama 相似变换（sim3）：求 `dst ≈ scale * rot * src + trans`。
fn umeyama_sim3(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
) -> Option<(f64, Matrix3<f64>, Vector3<f64>)> {
    let n = src.len();
    if n < 3 {
        return None;
    }
    let mx = src.iter().copied().sum::<Vector3<f64>>() / n as f64;
    let my = dst.iter().copied().sum::<Vector3<f64>>() / n as f64;
    let mut cov = Matrix3::zeros();
    let mut var = 0.0f64;
    for (x, y) in src.iter().zip(dst.iter()) {
        let xc = x - mx;
        let yc = y - my;
        cov += yc * xc.transpose();
        var += xc.norm_squared();
    }
    let svd = SVD::new(cov, true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let d = svd.singular_values;
    // 近共线采样退化保护：第二奇异值过小说明采样点近共线，Sim3 不可靠
    if d[1].abs() < 1e-6 * d[0].abs() {
        return None;
    }
    let det = (u * v_t).determinant();
    let sign = if det < 0.0 { -1.0 } else { 1.0 };
    let mut s_corr = Matrix3::identity();
    s_corr[(2, 2)] = sign;
    let rot = u * s_corr * v_t;
    let scale = (d[0] + d[1] + sign * d[2]) / var.max(1e-12);
    let trans = my - scale * (rot * mx);
    Some((scale, rot, trans))
}

/// 刚体变换（旋转 + 平移，尺度固定为 1）：`dst ≈ rot * src + trans`。
/// 用于无共享相机、尺度不可靠时的薄接缝对齐（尺度由先验单独给定）。
fn umeyama_rigid(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let n = src.len();
    if n < 3 {
        return None;
    }
    let mx = src.iter().copied().sum::<Vector3<f64>>() / n as f64;
    let my = dst.iter().copied().sum::<Vector3<f64>>() / n as f64;
    let mut cov = Matrix3::zeros();
    for (x, y) in src.iter().zip(dst.iter()) {
        cov += (y - my) * (x - mx).transpose();
    }
    let svd = SVD::new(cov, true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let d = svd.singular_values;
    if d[1].abs() < 1e-6 * d[0].abs() {
        return None;
    }
    let det = (u * v_t).determinant();
    let sign = if det < 0.0 { -1.0 } else { 1.0 };
    let mut s_corr = Matrix3::identity();
    s_corr[(2, 2)] = sign;
    let rot = u * s_corr * v_t;
    let trans = my - rot * mx;
    Some((rot, trans))
}

/// 返回把单位向量 `a` 旋转到 `b` 的最短旋转矩阵（绕 a×b 轴，Rodrigues）。
/// 用于「调平」：把各子模型的平均相机视向（上方向）对齐到共同参考。
fn rotation_between(a: &Vector3<f64>, b: &Vector3<f64>) -> Matrix3<f64> {
    let a = a.normalize();
    let b = b.normalize();
    let c = a.dot(&b).clamp(-1.0, 1.0);
    if c > 0.999_999 {
        return Matrix3::identity();
    }
    if c < -0.999_999 {
        let axis = if a.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
        let axis = a.cross(&axis).normalize();
        return so3_exp(&(axis * std::f64::consts::PI));
    }
    let v = a.cross(&b);
    let s = v.norm();
    // Rodrigues：R = I + skew(v) + skew(v)^2 * (1-c)/s^2，其中 skew(v) = sinθ·K（单位轴斜对称）。
    let vx = Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    );
    Matrix3::identity() + vx + vx * vx * ((1.0 - c) / (s * s))
}

/// 鲁棒相似变换：用 RANSAC 剔除离群共享点后求 `dst ≈ scale * rot * src + trans`。
///
/// `scale_prior` 用于无共享相机、仅靠薄接缝共享 track 的合并：薄接缝上自由拟合的
/// 尺度极易被重复纹理污染（曾出现 scale=0.392 的错值，导致子模型被缩小嵌套进主模型），
/// 因此这种情况下把尺度固定为先验值（两个子模型都做过 L1 归一化，真实尺度比应≈1），
/// 只做刚体 RANSAC（3 点）。有共享相机时传 `None`，尺度由数据自由决定（更可靠）。
/// 返回 `(scale, rot, trans, n_inliers)` 供上层诊断与可靠性判断。
fn robust_sim3(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    scale_prior: Option<f64>,
) -> Option<(f64, Matrix3<f64>, Vector3<f64>, usize)> {
    let n = src.len();
    if n < 4 {
        return None;
    }

    // 内点阈值：取锚点云包围盒对角线的 2%。
    // 注意不能用「中位半径」——子模型原点在种子图像处，共享点若集中在接缝附近
    // （如左右两半仅沿一条薄接缝重叠），中位半径会极小，阈值被压到不合理的低值。
    // 包围盒对角线不依赖原点位置，对薄接缝也保持合理尺度。
    let mut lo = src[0];
    let mut hi = src[0];
    for p in src {
        lo.x = lo.x.min(p.x);
        lo.y = lo.y.min(p.y);
        lo.z = lo.z.min(p.z);
        hi.x = hi.x.max(p.x);
        hi.y = hi.y.max(p.y);
        hi.z = hi.z.max(p.z);
    }
    let spread = (hi - lo).norm().max(1e-3);
    let threshold = (0.02 * spread).max(1e-3);
    // 宽松阈值：两个独立重建在薄接缝处的真对应点会有几%的 BA 漂移，残差可能略超
    // 紧阈值；用宽松阈值把这些点也计入，避免误判为内点不足。错误对应（重复纹理）
    // 残差远大于此，仍会被排除。
    let loose = (0.06 * spread).max(1e-3);

    // ---- 固定尺度（刚体）路径：无共享相机时用先验尺度，避免薄接缝的尺度被污染 ----
    if let Some(p) = scale_prior {
        let p = p.max(1e-6);
        // 预缩放 src，使问题变为纯刚体对齐 dst ≈ R * (p*src) + t
        let ssrc: Vec<Vector3<f64>> = src.iter().map(|x| p * x).collect();

        let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_rigid: Option<(Matrix3<f64>, Vector3<f64>)> = None;
        for _ in 0..20000 {
            let sample = sample(&mut rng, n, 3).into_vec();
            let s: Vec<Vector3<f64>> = sample.iter().map(|&i| ssrc[i]).collect();
            let d: Vec<Vector3<f64>> = sample.iter().map(|&i| dst[i]).collect();
            let (rot, trans) = match umeyama_rigid(&s, &d) {
                Some(v) => v,
                None => continue,
            };
            let mut inliers = Vec::new();
            for i in 0..n {
                if (rot * ssrc[i] + trans - dst[i]).norm() < threshold {
                    inliers.push(i);
                }
            }
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_rigid = Some((rot, trans));
            }
        }

        let mut rigid = best_rigid;
        if best_inliers.len() >= 3 {
            // 用刚性内点精化旋转与平移（尺度保持为先验值）
            let s: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| ssrc[i]).collect();
            let d: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| dst[i]).collect();
            if let Some(v) = umeyama_rigid(&s, &d) {
                rigid = Some(v);
            }
        }
        if let Some((rot, trans)) = rigid {
            let cnt = (0..n)
                .filter(|&i| (rot * ssrc[i] + trans - dst[i]).norm() < loose)
                .count();
            if cnt >= 3 {
                return Some((p, rot, trans, cnt));
            }
        }
        println!(
            "    robust_sim3: n={} spread={:.1} fixed-scale={:.3} no rigid consensus",
            n, spread, p
        );
        return None;
    }

    let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_sim: Option<(f64, Matrix3<f64>, Vector3<f64>)> = None;

    for _ in 0..1000 {
        let sample = sample(&mut rng, n, 4).into_vec();
        let s: Vec<Vector3<f64>> = sample.iter().map(|&i| src[i]).collect();
        let d: Vec<Vector3<f64>> = sample.iter().map(|&i| dst[i]).collect();
        let (scale, rot, trans) = match umeyama_sim3(&s, &d) {
            Some(v) => v,
            None => continue,
        };
        // scale 合理性保护：子模型都是同一相机的度量重建，scale 不应极端
        if !(MERGE_SCALE_MIN..=MERGE_SCALE_MAX).contains(&scale) {
            continue;
        }
        let mut inliers = Vec::new();
        for i in 0..n {
            let pred = scale * (rot * src[i]) + trans;
            if (pred - dst[i]).norm() < threshold {
                inliers.push(i);
            }
        }
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_sim = Some((scale, rot, trans));
        }
    }

    if best_inliers.len() >= 4 {
        // 用紧内点重跑 Umeyama 精化
        let s: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| src[i]).collect();
        let d: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| dst[i]).collect();
        let (mut scale, mut rot, mut trans) = umeyama_sim3(&s, &d).or(best_sim).unwrap();
        // 用宽松阈值重新统计：若更多，则再用这些点精化一次，得到更稳的 Sim3
        let loose_inliers: Vec<usize> = (0..n)
            .filter(|&i| (scale * (rot * src[i]) + trans - dst[i]).norm() < loose)
            .collect();
        if loose_inliers.len() > best_inliers.len() {
            let s2: Vec<Vector3<f64>> = loose_inliers.iter().map(|&i| src[i]).collect();
            let d2: Vec<Vector3<f64>> = loose_inliers.iter().map(|&i| dst[i]).collect();
            if let Some(v) = umeyama_sim3(&s2, &d2) {
                scale = v.0;
                rot = v.1;
                trans = v.2;
            }
        }
        let cnt = (0..n)
            .filter(|&i| (scale * (rot * src[i]) + trans - dst[i]).norm() < loose)
            .count();
        return Some((scale, rot, trans, cnt));
    }

    // RANSAC 未找到共识（薄接缝时四点最小样本命中率低）：退化为全体锚点直接最小二乘，
    // 用中位残差判断整体是否一致，避免把内点占比较高但样本难命中的情况误判为不可合并。
    let (s, r, t) = match umeyama_sim3(src, dst) {
        Some(v) => v,
        None => {
            println!(
                "    robust_sim3: n={} spread={:.1} best_inliers={} -> degenerate",
                n,
                spread,
                best_inliers.len()
            );
            return None;
        }
    };
    if !(MERGE_SCALE_MIN..=MERGE_SCALE_MAX).contains(&s) {
        println!(
            "    robust_sim3: n={} direct-LS scale={:.3} out of range",
            n, s
        );
        return None;
    }
    let mut res: Vec<f64> = (0..n)
        .map(|i| (s * (r * src[i]) + t - dst[i]).norm())
        .collect();
    res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_res = res[n / 2];
    let inliers = res.iter().filter(|&&v| v < loose).count();
    if med_res > loose {
        println!(
            "    robust_sim3: n={} direct-LS median_res={:.1} > {:.1} -> reject",
            n, med_res, loose
        );
        return None;
    }
    Some((s, r, t, inliers))
}

/// 子模型内有效点数量。
fn submodel_point_count(s: &Sfm) -> usize {
    s.points.iter().filter(|p| p.track_id != usize::MAX).count()
}

/// 子模型场景尺度：有效点到质心的均方根距离。
/// 两个子模型都在 `initialize_with_pair` 做过 L1 归一化，因此各自场景尺度应接近；
/// 两者之比是薄接缝合并时的可靠尺度先验（薄接缝自身无法可靠约束尺度）。
/// 已停用（它是覆盖范围半径不是尺度，做尺度先验会给出错误缩放），保留供参考。
#[allow(dead_code)]
fn scene_spread(s: &Sfm) -> f64 {
    let pts: Vec<Vector3<f64>> = s
        .points
        .iter()
        .filter(|p| p.track_id != usize::MAX)
        .map(|p| p.xyz)
        .collect();
    if pts.len() < 2 {
        return 1.0;
    }
    let mean = pts.iter().copied().sum::<Vector3<f64>>() / pts.len() as f64;
    let var = pts.iter().map(|p| (p - mean).norm_squared()).sum::<f64>() / pts.len() as f64;
    var.sqrt().max(1e-6)
}

/// 收集两个子模型间的共享锚点：共享 3D track + 共享相机中心。
/// 返回 `(src, dst, n_shared_cams)`，其中 `src.len()` 为锚点总数。
/// `src` 在 `sub` 坐标系，`dst` 在 `merged` 坐标系。
fn collect_merge_anchors(
    merged: &Sfm,
    sub: &Sfm,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>, usize) {
    let mut src: Vec<Vector3<f64>> = Vec::new();
    let mut dst: Vec<Vector3<f64>> = Vec::new();

    // 1) 共享 3D track
    for (tid, &pidx_sub) in &sub.track_to_point {
        if let Some(&pidx_merged) = merged.track_to_point.get(tid) {
            if sub.point_is_valid(pidx_sub) && merged.point_is_valid(pidx_merged) {
                src.push(sub.points[pidx_sub].xyz);
                dst.push(merged.points[pidx_merged].xyz);
            }
        }
    }

    // 2) 共享相机（同一张图在两个子模型里都注册了）
    let mut n_shared_cams = 0usize;
    for i in 0..merged.num_images {
        if let (Some(c_m), Some(c_s)) = (&merged.cameras[i], &sub.cameras[i]) {
            src.push(camera_center(&c_s.r, &c_s.t));
            dst.push(camera_center(&c_m.r, &c_m.t));
            n_shared_cams += 1;
        }
    }

    (src, dst, n_shared_cams)
}

/// 把 `sub` 变换到 `merged` 坐标系后并入（相机补未注册的，点去重共享 track）。
fn merge_submodel_into(
    merged: &mut Sfm,
    mut sub: Sfm,
    scale: f64,
    rot: Matrix3<f64>,
    trans: Vector3<f64>,
) {
    // 变换子模型相机位姿：C' = s R C + t，R' = R R_sim^T
    for cam in sub.cameras.iter_mut().flatten() {
        cam.r = cam.r * rot.transpose();
        cam.t = scale * cam.t - cam.r * trans;
    }
    // 变换子模型 3D 点
    for p in sub.points.iter_mut() {
        if p.track_id != usize::MAX {
            p.xyz = scale * (rot * p.xyz) + trans;
        }
    }

    // 合并相机（只补未注册的）
    for i in 0..merged.num_images {
        if merged.cameras[i].is_none() && sub.cameras[i].is_some() {
            merged.cameras[i] = sub.cameras[i].clone();
        }
    }
    // 合并点（去重共享 track）
    for (tid, &pidx_sub) in &sub.track_to_point {
        if sub.point_is_valid(pidx_sub) && !merged.track_to_point.contains_key(tid) {
            merged.points.push(sub.points[pidx_sub].clone());
            let new_pidx = merged.points.len() - 1;
            merged.track_to_point.insert(*tid, new_pidx);
        }
    }
}

/// 合并多个子模型：全局联合配准（姿态图）。
fn merge_submodels(mut submodels: Vec<Sfm>) -> Sfm {
    let n = submodels.len();
    if n <= 1 {
        return submodels.into_iter().max_by_key(submodel_point_count).unwrap();
    }

    // ============ 0) 调平：把每个子模型的「上」方向对齐到共同参考 ============
    // 各子模型坐标系由各自种子对决定，俯仰/滚转任意；农田近乎平面又约束不住这些角，导致
    // 姿态图里 z 漂移。这里先按「相机平均视向」调平：R^T e_z（相机 +z 在世界里的方向）对
    // 正视相机≈竖直向上，把所有子模型的这个方向对齐到共同参考，消除任意俯仰/滚转，让姿态
    // 图只需对齐 yaw + 平移 + z。
    if ENABLE_LEVELING {
        let mut up_ref = Vector3::zeros();
        let mut total = 0usize;
        for s in &submodels {
            for cam in s.cameras.iter().flatten() {
                up_ref += cam.r.row(2).transpose();
                total += 1;
            }
        }
        if total > 0 {
            up_ref /= total as f64;
            up_ref = up_ref.normalize();
            for s in submodels.iter_mut() {
                // 该子模型的上方向
                let mut up_i = Vector3::zeros();
                let mut cnt = 0usize;
                for cam in s.cameras.iter().flatten() {
                    up_i += cam.r.row(2).transpose();
                    cnt += 1;
                }
                if cnt == 0 {
                    continue;
                }
                up_i /= cnt as f64;
                up_i = up_i.normalize();
                // 调平旋转：把 up_i 转到 up_ref
                let l = rotation_between(&up_i, &up_ref);
                // 世界旋转 X' = L X；相机 R' = R L^T，t' = t
                for cam in s.cameras.iter_mut().flatten() {
                    cam.r = cam.r * l.transpose();
                }
                for p in s.points.iter_mut() {
                    if p.track_id != usize::MAX {
                        p.xyz = l * p.xyz;
                    }
                }
            }
        }
    }

    // ============ 全局联合配准（姿态图） ============
    // 不再贪心两两合并（那会形成一条长链、误差逐级累积、把航线位置带乱）。改为：
    //   1) 对每对共享锚点足够的子模型，用鲁棒刚体 RANSAC 估计相对变换（边）；
    //   2) 以最大子模型为根，BFS 生成树求每个子模型的全局位姿（初始）；
    //   3) 迭代平均：每个子模型用所有邻居的位姿提议取平均，把误差全局摊开；
    //   4) 按全局位姿把子模型都变换到根坐标系并合并。

    // 1) 建边：adj[i] = [(j, R_ij, t_ij, n_inliers)]，其中 X_j ≈ R_ij * X_i + t_ij（固定尺度 1）
    // 1) 建边：adj[i] = [(j, s_ij, R_ij, t_ij, n_inliers)]，X_j ≈ s_ij * R_ij * X_i + t_ij。
    // 尺度自由估计（各子模型本质矩阵单位基线在东西向/南北向种子上差 ~10%），并用范围保护。
    let mut adj: Vec<Vec<(usize, f64, Matrix3<f64>, Vector3<f64>, usize)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (src, dst, n_cams) = collect_merge_anchors(&submodels[j], &submodels[i]);
            if src.len() < 4 {
                continue;
            }
            if let Some((s, r, t, n_inl)) = robust_sim3(&src, &dst, None) {
                if n_inl >= 4 && n_cams >= 2 && (0.5..=2.0).contains(&s) {
                    let inv_s = 1.0 / s;
                    adj[i].push((j, s, r, t, n_inl));
                    adj[j].push((i, inv_s, r.transpose(), -(r.transpose() * t) * inv_s, n_inl));
                }
            }
        }
    }

    // 2) 以点数最多的子模型为根，BFS 生成树求初始全局位姿 (s_g, R_g, t_g)：
    //    X_g = s_g[i] * R_g[i] * X_i + t_g[i]。根 (1, I, 0)。
    let root = (0..n)
        .max_by_key(|&i| submodel_point_count(&submodels[i]))
        .unwrap();
    let mut scale_g: Vec<f64> = vec![1.0; n];
    let mut rot_g: Vec<Matrix3<f64>> = vec![Matrix3::identity(); n];
    let mut trans_g: Vec<Vector3<f64>> = vec![Vector3::zeros(); n];
    let mut visited = vec![false; n];
    visited[root] = true;
    let mut queue = vec![root];
    let mut qi = 0usize;
    while qi < queue.len() {
        let i = queue[qi];
        qi += 1;
        for &(j, s_ij, r_ij, t_ij, _) in &adj[i] {
            if !visited[j] {
                visited[j] = true;
                // 边 i→j: X_j = s_ij r_ij X_i + t_ij；由 X_g = s_i R_i X_i + t_i = s_j R_j X_j + t_j
                // 得 R_j = R_i r_ij^T，s_j = s_i / s_ij，t_j = t_i - s_j R_j t_ij。
                rot_g[j] = rot_g[i] * r_ij.transpose();
                scale_g[j] = scale_g[i] / s_ij;
                trans_g[j] = trans_g[i] - scale_g[j] * rot_g[j] * t_ij;
                queue.push(j);
            }
        }
    }

    // 3) 迭代平均（根固定，Gauss-Seidel）：相似变换平均（旋转 Karcher 均值 + 尺度/平移加权
    // 平均），把链式漂移与尺度不一致全局摊开。
    for _ in 0..10 {
        for i in 0..n {
            if i == root || !visited[i] {
                continue;
            }
            let mut acc_r = Matrix3::zeros();
            let mut acc_s = 0.0f64;
            let mut acc_t = Vector3::zeros();
            let mut total_w = 0.0f64;
            for &(j, s_ij, r_ij, t_ij, w) in &adj[i] {
                // 由邻居 j 提议 i：X_g = s_j R_j (s_ij r_ij X_i + t_ij) + t_j
                // => R_i = R_j r_ij，s_i = s_j s_ij，t_i = s_j R_j t_ij + t_j。
                let wf = w as f64;
                acc_r += (rot_g[j] * r_ij) * wf;
                acc_s += (scale_g[j] * s_ij) * wf;
                acc_t += (scale_g[j] * rot_g[j] * t_ij + trans_g[j]) * wf;
                total_w += wf;
            }
            if total_w <= 0.0 {
                continue;
            }
            let svd = SVD::new(acc_r, true, true);
            if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
                let mut r_avg = u * v_t;
                if r_avg.determinant() < 0.0 {
                    r_avg = -r_avg;
                }
                rot_g[i] = r_avg;
                scale_g[i] = acc_s / total_w;
                trans_g[i] = acc_t / total_w;
            }
        }
    }

    // 4) 按相似变换（尺度+旋转+平移）把子模型变换到根坐标系并合并。
    let mut subs: Vec<Option<Sfm>> = submodels.into_iter().map(Some).collect();
    for i in 0..n {
        if i == root || !visited[i] {
            continue;
        }
        let s = subs[i].as_mut().unwrap();
        for cam in s.cameras.iter_mut().flatten() {
            cam.r = cam.r * rot_g[i].transpose();
            cam.t = scale_g[i] * cam.t - cam.r * trans_g[i];
        }
        for p in s.points.iter_mut() {
            if p.track_id != usize::MAX {
                p.xyz = scale_g[i] * (rot_g[i] * p.xyz) + trans_g[i];
            }
        }
    }
    let mut merged = subs[root].take().unwrap();
    for i in 0..n {
        if i == root {
            continue;
        }
        if let Some(sub) = subs[i].take() {
            merge_submodel_into(&mut merged, sub, 1.0, Matrix3::identity(), Vector3::zeros());
        }
    }
    println!(
        "  Global registration: {}/{} sub-models connected (root {} imgs)",
        visited.iter().filter(|&&v| v).count(),
        n,
        merged.num_registered()
    );
    merged
}

/// 多子模型重建：从多个强连接图像对分别启动增量重建，再按共享 track 合并。
fn reconstruct_multi_model(
    sfm: &mut Sfm,
    index: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
) {
    let min_num_inliers = INIT_MIN_NUM_INLIERS;
    let min_tri_angle = INIT_MIN_TRI_ANGLE_DEG.to_radians();

    // 1. 收集子模型种子（强连接图像对）
    let mut seeds: Vec<((usize, usize), usize)> = index
        .iter()
        .filter(|&(_, v)| v.len() >= SUB_MODEL_MIN_MATCHES)
        .map(|(&(a, b), v)| ((a, b), v.len()))
        .collect();
    seeds.sort_by_key(|&(_, n)| std::cmp::Reverse(n));

    // 从一个种子对重建子模型（含增量增长）；两视图初始化失败则返回 None。
    let num_images = sfm.num_images;
    let intrinsics = sfm.intrinsics.clone();
    let build_submodel = |a: usize, b: usize| -> Option<Sfm> {
        let mut sub = Sfm::new(num_images, intrinsics.clone());
        let geometry = sub.estimate_initial_two_view_geometry(
            a,
            b,
            index,
            features,
            min_num_inliers,
            min_tri_angle,
        )?;
        if sub.initialize_with_pair(a, b, &geometry, tracks, features, index) != SfmStatus::Success {
            return None;
        }
        grow_sub_model(&mut sub, index, features, tracks, paths);
        Some(sub)
    };

    // 2. 第一轮：主（尽量不相交）子模型。记录每张图归属哪个主子模型。
    let mut submodel_of = vec![None; num_images];
    let mut submodels: Vec<Sfm> = Vec::new();

    for &((a, b), _n) in &seeds {
        if submodel_of[a].is_some() || submodel_of[b].is_some() {
            continue;
        }
        let sub = match build_submodel(a, b) {
            Some(s) => s,
            None => continue,
        };
        let reg = sub.registered_images();
        let idx = submodels.len();
        for img in &reg {
            submodel_of[*img] = Some(idx);
        }
        let np = sub.points.iter().filter(|p| p.track_id != usize::MAX).count();
        println!(
            "  Sub-model seeded ({},{}) -> {} images, {} points",
            a,
            b,
            reg.len(),
            np
        );
        submodels.push(sub);
    }

    let n_primary = submodels.len();

    // 3. 第二轮：桥接子模型。两端已分属不同主子模型的强连接对（正是第一轮被跳过的
    //    接缝对）作为桥种子，从接缝向外扩张，与两侧主模型共享相机，从而让合并阶段
    //    有可靠的相机中心锚点，避免退化成薄接缝上的弱 Sim3。
    if n_primary >= 2 {
        // 把接缝种子对按 (主子模型 x, y) 分组；seeds 已按匹配数降序，每组内保留强弱顺序。
        let mut seam_seeds: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
        for &((a, b), _n) in &seeds {
            let (x, y) = match (submodel_of[a], submodel_of[b]) {
                (Some(x), Some(y)) if x < n_primary && y < n_primary && x != y => {
                    (x.min(y), x.max(y))
                }
                _ => continue,
            };
            seam_seeds.entry((x, y)).or_default().push((a, b));
        }

        if !seam_seeds.is_empty() {
            println!(
                "  Bridge pass: {} seam pair(s) between primary sub-models",
                seam_seeds.len()
            );
        }

        for ((x, y), cands) in &seam_seeds {
            let mut built_ok = false;
            for &(a, b) in cands {
                let sub = match build_submodel(a, b) {
                    Some(s) => s,
                    // 这个种子两视图初始化失败，换同一对子模型里下一个较弱种子再试
                    None => continue,
                };
                let reg = sub.registered_images();
                let np = sub.points.iter().filter(|p| p.track_id != usize::MAX).count();
                println!(
                    "  Bridge sub-model ({},{}) connects sub-models {} & {} -> {} images, {} points",
                    a,
                    b,
                    x,
                    y,
                    reg.len(),
                    np
                );
                submodels.push(sub);
                built_ok = true;
                break;
            }
            if !built_ok {
                println!(
                    "  Bridge FAILED for sub-model pair ({},{}) — all {} seam seed(s) failed two-view init",
                    x,
                    y,
                    cands.len()
                );
            }
        }
    }

    if submodels.is_empty() {
        reconstruct(sfm, index, features, tracks, paths);
        return;
    }

    println!("\n  Merging {} sub-models...", submodels.len());
    *sfm = merge_submodels(submodels);

    // 合并后做一次全局 BA：两个半区之间若有残余尺度/漂移，靠接缝处的共享观测联合优化
    // 校正（合并的闭式 Sim3 无法利用全部观测，只用了共享 track 的点位置）。
    if !DISABLE_GLOBAL_BA_DIAG {
        sfm.iterative_refinement(tracks, features, true, None);
    }
    sfm.update_point_errors(tracks, features);
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

/// 可选：用图片 EXIF GPS 把重建对齐到米制 ENU（东/北/高）。
///
/// 不依赖 GPS 的通用路径照常工作：有 ≥3 张带 GPS 的已注册相机才做 Umeyama 相似变换
/// （尺度+旋转+平移），把相机中心对齐到 GPS ENU 坐标，从而给重建一个米制尺度；无 GPS
/// 时跳过，输出仍为任意尺度。这是纯后处理（一个全局相似变换），不影响 SfM 结构本身。
#[allow(dead_code)]
fn align_to_gps(sfm: &mut Sfm, paths: &[String]) {
    let mut gps: Vec<Option<(f64, f64, f64)>> = vec![None; sfm.num_images];
    let mut lats: Vec<f64> = Vec::new();
    let mut lons: Vec<f64> = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        if let Ok(data) = fs::read(p) {
            if let Some((lat, lon, alt)) = read_exif_gps(&data) {
                gps[i] = Some((lat, lon, alt));
                lats.push(lat);
                lons.push(lon);
            }
        }
    }
    if lats.len() < 3 {
        println!("  GPS align: only {} images have GPS (<3), keep arbitrary scale", lats.len());
        return;
    }
    let lat0 = lats.iter().sum::<f64>() / lats.len() as f64;
    let lon0 = lons.iter().sum::<f64>() / lons.len() as f64;
    let m_lat = 111320.0;
    let m_lon = 111320.0 * lat0.to_radians().cos();

    let mut src: Vec<Vector3<f64>> = Vec::new();
    let mut dst: Vec<Vector3<f64>> = Vec::new();
    for (i, cam) in sfm.cameras.iter().enumerate() {
        if let (Some(cam), Some((lat, lon, alt))) = (cam, gps[i]) {
            src.push(camera_center(&cam.r, &cam.t));
            dst.push(Vector3::new((lon - lon0) * m_lon, (lat - lat0) * m_lat, alt));
        }
    }
    if src.len() < 3 {
        println!("  GPS align: only {} registered cams have GPS, skip", src.len());
        return;
    }
    let (scale, rot, trans) = match umeyama_sim3(&src, &dst) {
        Some(v) => v,
        None => {
            println!("  GPS align: Umeyama degenerate, skip");
            return;
        }
    };

    // 点变换 X' = scale * rot * X + trans；相机 R' = R * rot^T，t' = scale*t - R'*trans
    for cam in sfm.cameras.iter_mut().flatten() {
        cam.r = cam.r * rot.transpose();
        cam.t = scale * cam.t - cam.r * trans;
    }
    for p in sfm.points.iter_mut() {
        if p.track_id != usize::MAX {
            p.xyz = scale * (rot * p.xyz) + trans;
        }
    }
    println!(
        "  GPS align: {} registered cams aligned to ENU, scale={:.3}",
        src.len(),
        scale
    );
}

pub(crate) fn run_incremental_sfm(
    matches: &[Match],
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
    out: &crate::types::Paths,
) {
    let first_img = image::open(&paths[0]).expect("Failed to open first image");
    let (w, h) = (first_img.width(), first_img.height());
    let mut intrinsics = CameraIntrinsics::from_image_with_exif(&paths[0], w, h);
    let (fw, fh) = feature_dimensions(w, h);
    if fw != w || fh != h {
        let sx = fw as f64 / w as f64;
        let sy = fh as f64 / h as f64;
        intrinsics.k[(0, 0)] *= sx;
        intrinsics.k[(1, 1)] *= sx;
        intrinsics.k[(0, 2)] *= sx;
        intrinsics.k[(1, 2)] *= sy;
    }
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
    reconstruct_multi_model(&mut sfm, &index, features, tracks, paths);

    // 补注册：暂禁用（怀疑其 grow_sub_model 触发的 BA 把尺度重新缩塌，见 2026-11-17 诊断）。
    // 原逻辑：清零未注册图 reg_trials，在合并后的完整模型上再跑一轮增量注册捡回边缘图。
    // let unregistered_before: usize = (0..sfm.num_images)
    //     .filter(|&img| sfm.cameras[img].is_none())
    //     .count();
    // if unregistered_before > 0 {
    //     for img in 0..sfm.num_images {
    //         if sfm.cameras[img].is_none() {
    //             sfm.reg_trials[img] = 0;
    //         }
    //     }
    //     grow_sub_model(&mut sfm, &index, features, tracks, paths);
    //     let unregistered_after: usize = (0..sfm.num_images)
    //         .filter(|&img| sfm.cameras[img].is_none())
    //         .count();
    //     println!(
    //         "  Post-merge re-registration: {} -> {} unregistered",
    //         unregistered_before, unregistered_after
    //     );
    // }

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

    // 约定尺度归一化：把相机轨迹的 RMS 半径归一到 1（同时平移使相机质心回到原点）。
    // 单目 SfM 的绝对尺度不可观测，用这个确定性约定让不同运行的输出数值稳定、可比较
    // （不用 GPS、不改变结构，只做全局相似变换）。
    {
        let centers: Vec<Vector3<f64>> = sfm
            .cameras
            .iter()
            .flatten()
            .map(|c| camera_center(&c.r, &c.t))
            .collect();
        if centers.len() >= 2 {
            let n = centers.len() as f64;
            let mean = centers.iter().fold(Vector3::zeros(), |a, b| a + *b) / n;
            let rms = (centers.iter().map(|c| (c - mean).norm_squared()).sum::<f64>() / n).sqrt();
            if rms > 1e-9 {
                let s = 1.0 / rms;
                for p in sfm.points.iter_mut() {
                    if p.track_id != usize::MAX {
                        p.xyz = (p.xyz - mean) * s;
                    }
                }
                for cam in sfm.cameras.iter_mut().flatten() {
                    cam.t = (cam.t + cam.r * mean) * s;
                }
                println!("  Normalized camera trajectory RMS radius to 1 (scale={:.4})", s);
            }
        }
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

    save_sparse_reconstruction(&sfm, features, tracks, paths, out);
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

/// 局部 z 一致性过滤：把点分到粗格（约 40 格），每格求中位 z 并做几轮扩散填充空格，
/// 再删掉偏离局部中位表面过远的离散点（跨航线假 track 三角化出的离群点）。只删 z 离群、
/// 不碰 x/y，保留地形起伏。
fn filter_local_z_outliers(points: &mut Vec<(Vector3<f64>, u8, u8, u8)>) -> usize {
    let n = points.len();
    if n < 200 {
        return 0;
    }
    let mut mn = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut mx = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
    for (p, _, _, _) in points.iter() {
        mn = mn.inf(p);
        mx = mx.sup(p);
    }
    let span = (mx - mn).norm().max(1.0);
    let cell = (span / 40.0).max(0.05);
    let nx = (((mx.x - mn.x) / cell).ceil() as usize).max(1) + 1;
    let ny = (((mx.y - mn.y) / cell).ceil() as usize).max(1) + 1;
    let mut grid: Vec<Vec<f64>> = vec![Vec::new(); nx * ny];
    for (p, _, _, _) in points.iter() {
        let gx = (((p.x - mn.x) / cell).floor() as usize).min(nx - 1);
        let gy = (((p.y - mn.y) / cell).floor() as usize).min(ny - 1);
        grid[gy * nx + gx].push(p.z);
    }
    let mut medz: Vec<Option<f64>> = vec![None; nx * ny];
    for (i, g) in grid.iter().enumerate() {
        if g.len() >= 5 {
            let mut zs = g.clone();
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            medz[i] = Some(zs[zs.len() / 2]);
        }
    }
    // 扩散填充空格（3 轮 8 邻域平均）
    for _ in 0..3 {
        let old = medz.clone();
        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                if medz[idx].is_some() {
                    continue;
                }
                let mut s = 0.0;
                let mut c = 0usize;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let xx = x as i64 + dx;
                        let yy = y as i64 + dy;
                        if xx >= 0 && yy >= 0 && xx < nx as i64 && yy < ny as i64 {
                            if let Some(v) = old[(yy * nx as i64 + xx) as usize] {
                                s += v;
                                c += 1;
                            }
                        }
                    }
                }
                if c > 0 {
                    medz[idx] = Some(s / c as f64);
                }
            }
        }
    }
    // 阈值：偏离局部中位 3 倍中位绝对偏差（对每点算 |z - medz|，取中位 MAD）
    let mut devs: Vec<f64> = Vec::new();
    for (p, _, _, _) in points.iter() {
        let gx = (((p.x - mn.x) / cell).floor() as usize).min(nx - 1);
        let gy = (((p.y - mn.y) / cell).floor() as usize).min(ny - 1);
        if let Some(m) = medz[gy * nx + gx] {
            devs.push((p.z - m).abs());
        }
    }
    if devs.len() < 50 {
        return 0;
    }
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = devs[devs.len() / 2].max(0.02);
    let threshold = (6.0 * mad).max(0.15);
    let before = points.len();
    points.retain(|(p, _, _, _)| {
        let gx = (((p.x - mn.x) / cell).floor() as usize).min(nx - 1);
        let gy = (((p.y - mn.y) / cell).floor() as usize).min(ny - 1);
        match medz[gy * nx + gx] {
            Some(m) => (p.z - m).abs() <= threshold,
            None => true,
        }
    });
    let removed = before - points.len();
    if removed > 0 {
        println!("  Local-z filter: {} pts removed (z dev > {:.2})", removed, threshold);
    }
    removed
}

/// 温和的局部一致性过滤：粗格（约 60 格）中位表面 + 扩散，只删偏离局部表面超过大阈值
/// （0.5 单位 ≈ 40m）的明显飞散假点。房子（高出地面 5~10m）不会被删，北部房区/葡萄园保留。
fn filter_gentle_outliers(points: &mut Vec<(Vector3<f64>, u8, u8, u8)>) -> usize {
    let n = points.len();
    if n < 200 {
        return 0;
    }
    let mut mn = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut mx = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
    for (p, _, _, _) in points.iter() {
        mn = mn.inf(p);
        mx = mx.sup(p);
    }
    let ext = mx - mn;
    let cell = (ext.x.max(ext.y) / 60.0).max(0.01);
    let nx = ((ext.x / cell).ceil() as usize).max(2) + 1;
    let ny = ((ext.y / cell).ceil() as usize).max(2) + 1;
    let mut grid: Vec<Vec<f64>> = vec![Vec::new(); nx * ny];
    for (p, _, _, _) in points.iter() {
        let i = ((p.x - mn.x) / cell).floor() as usize;
        let j = ((p.y - mn.y) / cell).floor() as usize;
        if i < nx && j < ny {
            grid[j * nx + i].push(p.z);
        }
    }
    let mut medz: Vec<f64> = vec![f64::NAN; nx * ny];
    for (k, g) in grid.iter().enumerate() {
        if !g.is_empty() {
            let mut g = g.clone();
            g.sort_by(|a, b| a.partial_cmp(b).unwrap());
            medz[k] = g[g.len() / 2];
        }
    }
    for _ in 0..(nx + ny) {
        let mut next = medz.clone();
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if !medz[k].is_nan() {
                    continue;
                }
                let mut s = 0.0;
                let mut c = 0i32;
                for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                    let ni = i as i64 + di;
                    let nj = j as i64 + dj;
                    if ni >= 0 && nj >= 0 && ni < nx as i64 && nj < ny as i64 {
                        let nk = nj as usize * nx + ni as usize;
                        if !medz[nk].is_nan() {
                            s += medz[nk];
                            c += 1;
                        }
                    }
                }
                if c > 0 {
                    next[k] = s / c as f64;
                }
            }
        }
        medz = next;
    }
    let threshold = 0.5;
    let before = points.len();
    // 1) z 局部一致性：删偏离局部表面太远的点
    points.retain(|(p, _, _, _)| {
        let i = ((p.x - mn.x) / cell).floor() as usize;
        let j = ((p.y - mn.y) / cell).floor() as usize;
        if i >= nx || j >= ny {
            return false;
        }
        let m = medz[j * nx + i];
        if m.is_nan() {
            return false;
        }
        (p.z - m).abs() <= threshold
    });
    // 2) 3D 距离百分位：删离质心最远的 1%（飞散假点在 x/y 上也远离主群）
    if points.len() >= 200 {
        let c = points
            .iter()
            .fold(Vector3::zeros(), |a, (p, _, _, _)| a + *p)
            / points.len() as f64;
        let mut dists: Vec<f64> = points.iter().map(|(p, _, _, _)| (p - c).norm()).collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cutoff = dists[(dists.len() as f64 * 0.99) as usize];
        points.retain(|(p, _, _, _)| (p - c).norm() <= cutoff);
    }
    let removed = before - points.len();
    if removed > 0 {
        println!("  Gentle filter: {} pts removed (z dev > {:.2} or 1% farthest)", removed, threshold);
    }
    removed
}

fn save_sparse_reconstruction(
    sfm: &Sfm,
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
    out: &crate::types::Paths,
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

    // 温和过滤：只删明显飞散/孤立的假点（重复纹理假 track），保留地面与房子。
    let _ = filter_gentle_outliers(&mut valid_pts);

    // 规定到所有图像中心点（相机中心）包围盒 + FOV 余量的正射范围内：相机覆盖区外的点删掉。
    {
        let mut cmin = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut cmax = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
        let mut cam_n = 0usize;
        for cam in sfm.cameras.iter().flatten() {
            let c = cam.r.transpose() * (-cam.t);
            cmin = cmin.inf(&c);
            cmax = cmax.sup(&c);
            cam_n += 1;
        }
        if cam_n > 0 {
            // 只保留相机中心包围盒以内的数据（多视角覆盖区）；包围盒外只有边缘相机单视角
            // 看到、精度无保证，直接删。不加 FOV 余量。
            let margin = 0.0;
            let before = valid_pts.len();
            valid_pts.retain(|(p, _, _, _)| {
                p.x >= cmin.x - margin && p.x <= cmax.x + margin
                    && p.y >= cmin.y - margin && p.y <= cmax.y + margin
            });
            let removed = before - valid_pts.len();
            if removed > 0 {
                println!(
                    "  Ortho-range filter: {} pts removed (outside camera bbox)",
                    removed
                );
            }
        }
    }

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

    fs::write(out.sparse_ply(), &ply).unwrap();

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
        out.camera_poses_json(),
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
