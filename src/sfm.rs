use std::collections::{BTreeMap, HashMap};
use std::fs;

use image::RgbImage;
use nalgebra::{self as na, Matrix3, Matrix3x4, Vector3, SVD};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

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

pub(crate) const RANSAC_ITERATIONS: usize = 2000;
pub(crate) const RANSAC_THRESHOLD_PX: f64 = 4.0;

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

        self.adjust_global_bundle_fixed(
            tracks,
            features,
            LossType::SoftL1,
            BA_GLOBAL_MAX_ITERATIONS,
            Some(first),
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

pub(crate) fn run_incremental_sfm(
    matches: &[Match],
    features: &[ImageFeatures],
    tracks: &[Track],
    paths: &[String],
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
