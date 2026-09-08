use std::collections::{BTreeMap, HashMap};
use std::fs;

use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::sfm::CameraIntrinsics;
use crate::types::RANSAC_SEED;
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
pub(crate) fn run_mvs_dense(paths: &[String]) {
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
    let dedup_vox: Vec<((i64, i64, i64), f64, f64, f64, u8, u8, u8)> = vox
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
