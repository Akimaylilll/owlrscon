use std::collections::{BTreeMap, HashMap};
use std::fs;

use cubecl::prelude::*;
use image::{Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::imgproc::feature_dimensions;
use crate::sfm::CameraIntrinsics;
use crate::types::RANSAC_SEED;

/// NCC 核每组线程数（wgpu 单维工作组上限 65535，单维线程上限通常 256）
const CUBE_THREADS: u32 = 256;

/// 每个参考相机用于多视图代价聚合的邻帧数。
const N_NBR: usize = 8;

/// 网格三角化步长：深度图全分辨率(5472×3648)下每相机可产生大量三角形，55 相机累计会 OOM。
/// 按步长抽取像素，把面片数量降到可接受范围（步长 8 → 约 1/64）。
const MESH_STEP: usize = 8;

/// 纹理图集边长上限（避免烘焙阶段分配过大的图集位图）。
const ATLAS_MAX_SIDE: u32 = 8192;

/// 是否生成带纹理的三角网格（mesh.obj + mesh_texture.png）。
/// 暂时关闭：当前重点是 dense.ply 的点云质量，网格化耗时且会干扰调优节奏。
const ENABLE_MESH: bool = false;

/// 核内 popcount（SWAR，cubecl 无内置 count_ones）。
#[cube]
fn popcount(mut x: u32) -> u32 {
    x = x - ((x >> 1) & 0x55555555u32);
    x = (x & 0x33333333u32) + ((x >> 2) & 0x33333333u32);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu32;
    x = x + (x >> 8);
    x = x + (x >> 16);
    x & 0x3Fu32
}

/// 单像素对单源图的 Census 代价（cubecl 辅助，内联）。源图/位姿为打包数组，用 base 偏移取第 k 个源。
#[allow(unused_assignments)]
#[cube]
fn census_for_pixel(
    ref_census: &Array<u32>,
    src_census: &Array<u32>,
    src_base: usize,
    r_sr: &Array<f32>,
    r_base: usize,
    t_sr: &Array<f32>,
    t_base: usize,
    f_src: &Array<f32>,
    f_base: usize,
    f_ref: f32,
    cx: f32,
    cy: f32,
    iw: usize,
    ih: usize,
    half: usize,
    x: usize,
    y: usize,
    d_plane: f32,
    n0: f32,
    n1: f32,
    n2: f32,
) -> f32 {
    let fs = f_src[f_base];
    let mut hamming = 0u32;
    let mut n = 0.0f32;

    let x_s = i64::cast_from(x);
    let y_s = i64::cast_from(y);
    let half_s = i64::cast_from(half);
    let iw_s = i64::cast_from(iw);
    let ih_s = i64::cast_from(ih);

    for dy in -half_s..=half_s {
        let ry = y_s + dy;
        if ry >= 0 && ry < ih_s {
            for dx in -half_s..=half_s {
                let rx = x_s + dx;
                if rx >= 0 && rx < iw_s {
                    let rxf = f32::cast_from(rx);
                    let ryf = f32::cast_from(ry);
                    let u_r = (rxf - cx) / f_ref;
                    let v_r = (ryf - cy) / f_ref;
                    let dot = n0 * u_r + n1 * v_r + n2;

                    let rr0 = r_sr[r_base] * u_r
                        + r_sr[r_base + 1] * v_r
                        + r_sr[r_base + 2]
                        + t_sr[t_base] * dot / d_plane;
                    let rr1 = r_sr[r_base + 3] * u_r
                        + r_sr[r_base + 4] * v_r
                        + r_sr[r_base + 5]
                        + t_sr[t_base + 1] * dot / d_plane;
                    let rr2 = r_sr[r_base + 6] * u_r
                        + r_sr[r_base + 7] * v_r
                        + r_sr[r_base + 8]
                        + t_sr[t_base + 2] * dot / d_plane;

                    if rr2.abs() > 1e-9f32 {
                        let sx = fs * rr0 / rr2 + cx;
                        let sy = fs * rr1 / rr2 + cy;
                        let sxi = i64::cast_from(sx + 0.5f32);
                        let syi = i64::cast_from(sy + 0.5f32);
                        if sxi >= 0 && sxi < iw_s && syi >= 0 && syi < ih_s {
                            let ridx = usize::cast_from(ry) * iw + usize::cast_from(rx);
                            let sidx = usize::cast_from(syi) * iw + usize::cast_from(sxi);
                            let sbase = src_base + sidx * 3;
                            hamming += popcount(ref_census[ridx * 3] ^ src_census[sbase])
                                + popcount(ref_census[ridx * 3 + 1] ^ src_census[sbase + 1])
                                + popcount(ref_census[ridx * 3 + 2] ^ src_census[sbase + 2]);
                            n += 1.0f32;
                        }
                    }
                }
            }
        }
    }

    let mut result = 0.0f32;
    if n < 9.0f32 {
        result = -1.0f32;
    } else {
        result = 1.0f32 - f32::cast_from(hamming) / (n * 80.0f32);
    }
    result
}

/// 单像素对多个源图的 Census 相似度均值（cubecl 辅助，内联）。无效返回 -1.0。
#[allow(unused_assignments)]
#[cube]
fn census_cost(
    ref_census: &Array<u32>,
    src_census: &Array<u32>,
    n_px: usize,
    r_sr: &Array<f32>,
    t_sr: &Array<f32>,
    f_src: &Array<f32>,
    f_ref: f32,
    cx: f32,
    cy: f32,
    iw: usize,
    ih: usize,
    half: usize,
    n_nbr: usize,
    x: usize,
    y: usize,
    d: f32,
    n0: f32,
    n1: f32,
    n2: f32,
) -> f32 {
    let xf = f32::cast_from(x);
    let yf = f32::cast_from(y);
    let u = (xf - cx) / f_ref;
    let v = (yf - cy) / f_ref;
    let d_plane = d * (n0 * u + n1 * v + n2);
    let mut result = 0.0f32;
    if d_plane.abs() < 1e-9f32 {
        result = -1.0f32;
    } else {
        let mut total = 0.0f32;
        let mut cnt = 0.0f32;
        for k in 0..n_nbr {
            let sim = census_for_pixel(
                ref_census,
                src_census,
                k * n_px * 3,
                r_sr,
                k * 9,
                t_sr,
                k * 3,
                f_src,
                k,
                f_ref,
                cx,
                cy,
                iw,
                ih,
                half,
                x,
                y,
                d_plane,
                n0,
                n1,
                n2,
            );
            if sim > -0.5f32 {
                total += sim;
                cnt += 1.0f32;
            }
        }
        if cnt == 0.0f32 {
            result = 0.0f32;
        } else {
            result = total / cnt;
        }
    }
    result
}

/// 核内哈希 RNG（xorshift32）。
#[cube]
fn xorshift32(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// PatchMatch 红黑传播 + 扰动（cubecl 内核）。
#[cube(launch_unchecked)]
fn pm_propagate(
    ref_census: &Array<u32>,
    src_census: &Array<u32>,
    r_sr: &Array<f32>,
    t_sr: &Array<f32>,
    f_src: &Array<f32>,
    depth: &mut Array<f32>,
    normal: &mut Array<f32>,
    cost: &mut Array<f32>,
    f_ref: f32,
    cx: f32,
    cy: f32,
    iw: usize,
    ih: usize,
    half: usize,
    n_nbr: usize,
    dx: i32,
    dy: i32,
    color: u32,
    dmin: f32,
    dmax: f32,
    d_perturb: f32,
    n_perturb: f32,
    seed: u32,
) {
    let idx = ABSOLUTE_POS;
    let n_px = iw * ih;
    if idx < n_px {
        let x = idx % iw;
        let y = idx / iw;
        if (x + y) % 2 == color as usize {
            let mut best_d = depth[idx];
            let mut bn0 = normal[idx * 3];
            let mut bn1 = normal[idx * 3 + 1];
            let mut bn2 = normal[idx * 3 + 2];
            let mut best_cost = cost[idx];

            // 传播候选：读邻像素 (x+dx, y+dy)
            let px = x as i64 + dx as i64;
            let py = y as i64 + dy as i64;
            if px >= 0 && px < iw as i64 && py >= 0 && py < ih as i64 {
                let pidx = (py as usize) * iw + px as usize;
                let pd = depth[pidx];
                let pn0 = normal[pidx * 3];
                let pn1 = normal[pidx * 3 + 1];
                let pn2 = normal[pidx * 3 + 2];
                let c = census_cost(
                    ref_census, src_census, n_px, r_sr, t_sr, f_src, f_ref, cx, cy, iw, ih, half,
                    n_nbr, x, y, pd, pn0, pn1, pn2,
                );
                if c > best_cost {
                    best_d = pd;
                    bn0 = pn0;
                    bn1 = pn1;
                    bn2 = pn2;
                    best_cost = c;
                }
            }

            // 扰动候选
            let mut state = xorshift32(seed ^ ((idx as u64) * 2654435761u64) as u32);
            let r1 =
                f32::cast_from(xorshift32(state) & 0x00FF_FFFFu32) / f32::cast_from(0x00FF_FFFFu32);
            state = xorshift32(state);
            let r2 =
                f32::cast_from(xorshift32(state) & 0x00FF_FFFFu32) / f32::cast_from(0x00FF_FFFFu32);
            state = xorshift32(state);
            let r3 =
                f32::cast_from(xorshift32(state) & 0x00FF_FFFFu32) / f32::cast_from(0x00FF_FFFFu32);

            let pd = (best_d + (r1 * 2.0f32 - 1.0f32) * d_perturb).clamp(dmin, dmax);
            let pn0 = (bn0 + (r2 * 2.0f32 - 1.0f32) * n_perturb).clamp(-0.9f32, 0.9f32);
            let pn1 = (bn1 + (r3 * 2.0f32 - 1.0f32) * n_perturb).clamp(-0.9f32, 0.9f32);
            let pn2 = (1.0f32 - pn0 * pn0 - pn1 * pn1).sqrt();

            let c = census_cost(
                ref_census, src_census, n_px, r_sr, t_sr, f_src, f_ref, cx, cy, iw, ih, half,
                n_nbr, x, y, pd, pn0, pn1, pn2,
            );
            if c > best_cost {
                best_d = pd;
                bn0 = pn0;
                bn1 = pn1;
                bn2 = pn2;
                best_cost = c;
            }

            depth[idx] = best_d;
            normal[idx * 3] = bn0;
            normal[idx * 3 + 1] = bn1;
            normal[idx * 3 + 2] = bn2;
            cost[idx] = best_cost;
        }
    }
}

/// MVS 使用的相机（位姿 + 焦距 + Census 码 + 紧凑灰度）。
/// 不长期驻留彩色图/全精度灰度，全分辨率下用 Census 码(4B)+灰度(1B) 控制内存。
struct RegCam {
    r: Matrix3<f64>,
    t: Vector3<f64>,
    f: f64,
    f_full: f64,
    gray: Vec<u8>,
    path: String,
}

/// 对灰度图（u8）计算 9×9 窗口（80 邻域）Census 码，每个像素 3 个 u32（96 位，用 80 位）。
/// 边界外邻域按「不大于中心」处理（bit=0）。
fn compute_census(gray: &[u8], iw: usize, ih: usize) -> Vec<u32> {
    use rayon::prelude::*;
    let n = iw * ih;
    let mut out = vec![0u32; 3 * n];
    out.par_chunks_mut(iw * 3).enumerate().for_each(|(y, row)| {
        for x in 0..iw {
            let center = gray[y * iw + x];
            let mut c0 = 0u32;
            let mut c1 = 0u32;
            let mut c2 = 0u32;
            let mut bit = 0u32;
            for dy in -4i64..=4 {
                for dx in -4i64..=4 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    let b = if nx >= 0 && nx < iw as i64 && ny >= 0 && ny < ih as i64 {
                        (gray[(ny as usize) * iw + nx as usize] > center) as u32
                    } else {
                        0
                    };
                    if bit < 32 {
                        c0 |= b << bit;
                    } else if bit < 64 {
                        c1 |= b << (bit - 32);
                    } else {
                        c2 |= b << (bit - 64);
                    }
                    bit += 1;
                }
            }
            row[x * 3] = c0;
            row[x * 3 + 1] = c1;
            row[x * 3 + 2] = c2;
        }
    });
    out
}

/// 深度/代价统计（诊断用，每相机打印）。
fn depth_stats_line(depth: &[f32], cost: &[f32]) -> String {
    let mut valid: Vec<f64> = Vec::new();
    let mut cost_sum = 0.0f64;
    let mut cost_cnt = 0usize;
    for (i, &d) in depth.iter().enumerate() {
        if d > 0.0 {
            valid.push(d as f64);
        }
        if cost[i] > -1.0 {
            cost_sum += cost[i] as f64;
            cost_cnt += 1;
        }
    }
    let valid_ratio = valid.len() as f64 / depth.len() as f64 * 100.0;
    if valid.is_empty() {
        return "valid=0.0%".to_string();
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let median = valid[valid.len() / 2];
    let var = valid.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / valid.len() as f64;
    let cost_mean = if cost_cnt > 0 {
        cost_sum / cost_cnt as f64
    } else {
        0.0
    };
    format!(
        "mean={:.3} median={:.3} std={:.4} min={:.3} max={:.3} valid={:.1}% | cost_mean={:.3}",
        mean,
        median,
        var.sqrt(),
        valid[0],
        valid[valid.len() - 1],
        valid_ratio,
        cost_mean
    )
}

/// 深度直方图（诊断用，`bins` 个等宽桶的百分比分布）。均匀分布 → 各桶接近相等；收敛 → 少数桶尖峰。
fn depth_histogram(depth: &[f32], bins: usize) -> String {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut total = 0usize;
    for &d in depth {
        if d > 0.0 {
            mn = mn.min(d);
            mx = mx.max(d);
            total += 1;
        }
    }
    if total == 0 || !mn.is_finite() || (mx - mn) < 1e-6 {
        return "empty".to_string();
    }
    let mut counts = vec![0usize; bins];
    for &d in depth {
        if d > 0.0 {
            let b = (((d - mn) / (mx - mn)) * bins as f32) as usize;
            counts[b.min(bins - 1)] += 1;
        }
    }
    let mut s = String::new();
    for &c in &counts {
        s.push_str(&format!("{:3.0}% ", c as f64 / total as f64 * 100.0));
    }
    s.trim_end().to_string()
}

/// 稀疏点云统计离群移除（PCL statistical outlier removal）：
/// 对每个点求 k 近邻平均距离，剔除平均距离 > 全局均值 + std_mult * 标准差 的点。
/// 用体素网格做 k-NN，避免 O(n²)。
fn sparse_statistical_filter(
    points: &[Vector3<f64>],
    k_nn: usize,
    std_mult: f64,
) -> Vec<Vector3<f64>> {
    let n = points.len();
    if n < 50 {
        return points.to_vec();
    }
    // 由包围盒与点数估计平均间距，作为体素单元尺寸
    let mut mn = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut mx = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
    for p in points {
        mn = mn.inf(p);
        mx = mx.sup(p);
    }
    let extent = mx - mn;
    let volume = extent.x.abs() * extent.y.abs() * extent.z.abs();
    let cell = (volume / n as f64).cbrt().max(1e-3);

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        grid.entry((
            (p.x / cell) as i64,
            (p.y / cell) as i64,
            (p.z / cell) as i64,
        ))
        .or_default()
        .push(i);
    }

    let mut mean_dists: Vec<f64> = vec![f64::INFINITY; n];
    for (i, p) in points.iter().enumerate() {
        let cx = (p.x / cell) as i64;
        let cy = (p.y / cell) as i64;
        let cz = (p.z / cell) as i64;
        let mut dists: Vec<f64> = Vec::new();
        'outer: for rad in 1i64..=8 {
            for dz in -rad..=rad {
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        if dx.abs() < rad && dy.abs() < rad && dz.abs() < rad {
                            continue;
                        }
                        if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in ids {
                                if j == i {
                                    continue;
                                }
                                let q = points[j];
                                dists.push(
                                    ((p.x - q.x).powi(2)
                                        + (p.y - q.y).powi(2)
                                        + (p.z - q.z).powi(2))
                                    .sqrt(),
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
        if !dists.is_empty() {
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
    if finite.is_empty() {
        return points.to_vec();
    }
    let mean_md = finite.iter().sum::<f64>() / finite.len() as f64;
    let var_md = finite.iter().map(|d| (d - mean_md).powi(2)).sum::<f64>() / finite.len() as f64;
    let threshold = mean_md + std_mult * var_md.sqrt();

    points
        .iter()
        .enumerate()
        .filter(|(i, _)| mean_dists[*i] < threshold)
        .map(|(_, p)| *p)
        .collect()
}

/// 为相机 si 生成稀疏深度图：把稀疏点投影到该相机，散射后做邻域扩散，未覆盖处用中位深度填充。
/// 返回完整深度图（无 0 值），用作 PatchMatch 的初始深度先验，让迭代从真实表面附近开始。
fn sparse_depth_map(
    si: usize,
    cams: &[RegCam],
    sparse_pts: &[Vector3<f64>],
    iw: usize,
    ih: usize,
    cx: f64,
    cy: f64,
) -> Vec<f32> {
    let n = iw * ih;
    let mut depth = vec![0.0f32; n];
    let cam = &cams[si];
    let f = cam.f;

    // 1) 散射：每个稀疏点落到对应像素（保留更近的深度，避免楼房屋顶被地面点覆盖）
    for pt in sparse_pts {
        let xc = cam.r * pt + cam.t;
        if xc.z <= 0.5 {
            continue;
        }
        let u = (f * xc.x / xc.z + cx).round();
        let v = (f * xc.y / xc.z + cy).round();
        if u < 0.0 || u >= iw as f64 || v < 0.0 || v >= ih as f64 {
            continue;
        }
        let idx = (v as usize) * iw + u as usize;
        let d = xc.z as f32;
        if depth[idx] == 0.0 || d < depth[idx] {
            depth[idx] = d;
        }
    }

    // 2) 邻域扩散：把稀疏深度向四周扩散，形成粗糙先验
    for _ in 0..8 {
        let mut next = depth.clone();
        for y in 0..ih {
            for x in 0..iw {
                let idx = y * iw + x;
                if depth[idx] != 0.0 {
                    continue;
                }
                let mut sum = 0.0f32;
                let mut cnt = 0;
                if x > 0 && depth[idx - 1] != 0.0 {
                    sum += depth[idx - 1];
                    cnt += 1;
                }
                if x + 1 < iw && depth[idx + 1] != 0.0 {
                    sum += depth[idx + 1];
                    cnt += 1;
                }
                if y > 0 && depth[idx - iw] != 0.0 {
                    sum += depth[idx - iw];
                    cnt += 1;
                }
                if y + 1 < ih && depth[idx + iw] != 0.0 {
                    sum += depth[idx + iw];
                    cnt += 1;
                }
                if cnt > 0 {
                    next[idx] = sum / cnt as f32;
                }
            }
        }
        depth = next;
    }

    // 3) 仍未覆盖的像素用非零中位深度填充
    let mut nz: Vec<f32> = depth.iter().copied().filter(|&d| d > 0.0).collect();
    if nz.is_empty() {
        return depth;
    }
    nz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = nz[nz.len() / 2];
    for d in depth.iter_mut() {
        if *d == 0.0 {
            *d = median;
        }
    }
    depth
}

/// PatchMatch 立体匹配（GPU 版，红黑传播 + 扰动 + 稀疏深度初始化）：
/// 用 80-bit Census(9×9, 3×u32) + 连续法向 + 8 邻帧，传播机制把好深度从纹理区扩散到低纹理区。
/// 返回 (深度, 法向, 代价)。
fn patch_match_image_gpu(
    si: usize,
    cams: &[RegCam],
    neighbors: &[usize],
    dmin: f64,
    dmax: f64,
    iw: usize,
    ih: usize,
    cx: f64,
    cy: f64,
    num_iterations: usize,
    sparse_depth: &[f32],
) -> (Vec<f32>, Vec<[f32; 3]>, Vec<f32>) {
    let client = crate::backend::client();
    let n = iw * ih;
    let n_nbr = neighbors.len();
    let patch_half = 1usize; // 3×3 Census 聚合 patch

    let dmin_f = dmin as f32;
    let dmax_f = dmax as f32;

    // 实时计算 Census 码（9×9 80bit，3×u32/像素）：参考图 + 邻帧打包，避免常驻全部相机
    let ref_census = compute_census(&cams[si].gray, iw, ih);
    let mut src_census_packed = vec![0u32; n_nbr * 3 * n];
    let mut r_sr = vec![0.0f32; n_nbr * 9];
    let mut t_sr = vec![0.0f32; n_nbr * 3];
    let mut f_src = vec![0.0f32; n_nbr];
    for (k, &ri) in neighbors.iter().enumerate() {
        let src_c = compute_census(&cams[ri].gray, iw, ih);
        src_census_packed[k * 3 * n..(k + 1) * 3 * n].copy_from_slice(&src_c);
        let (r, t) = relative_pose_mvs(&cams[si].r, &cams[si].t, &cams[ri].r, &cams[ri].t);
        for i in 0..3 {
            for j in 0..3 {
                r_sr[k * 9 + i * 3 + j] = r[(i, j)] as f32;
            }
            t_sr[k * 3 + i] = t[i] as f32;
        }
        f_src[k] = cams[ri].f as f32;
    }

    // 稀疏深度先验初始化 + 连续法向初始化（小抖动，给传播/扰动留探索空间）
    let mut rng = StdRng::seed_from_u64(RANSAC_SEED ^ (si as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let mut depth = vec![0.0f32; n];
    let mut normal_soa = vec![[0.0f32, 0.0f32, 1.0f32]; n];
    for i in 0..n {
        let base = if sparse_depth[i] > 0.0 {
            sparse_depth[i]
        } else {
            (dmin_f + dmax_f) * 0.5
        };
        let jitter = if sparse_depth[i] > 0.0 { 0.01 } else { 0.05 };
        depth[i] = (base * (1.0 + rng.gen_range(-1.0f32..1.0f32) * jitter)).clamp(dmin_f, dmax_f);
        let nx = rng.gen_range(-0.2f32..0.2f32);
        let ny = rng.gen_range(-0.2f32..0.2f32);
        let nz = (1.0f32 - nx * nx - ny * ny).sqrt();
        normal_soa[i] = [nx, ny, nz];
    }

    // 展平法向为交错 f32
    let mut normal_flat = vec![0.0f32; 3 * n];
    for (i, nrm) in normal_soa.iter().enumerate() {
        normal_flat[i * 3] = nrm[0];
        normal_flat[i * 3 + 1] = nrm[1];
        normal_flat[i * 3 + 2] = nrm[2];
    }
    drop(normal_soa);

    // 上传常驻缓冲（depth/normal 为读改写，先写初值）
    let depth_handle = client.create_from_slice(f32::as_bytes(&depth));
    let normal_handle = client.create_from_slice(f32::as_bytes(&normal_flat));
    let cost_handle = client.empty(n * core::mem::size_of::<f32>());
    let ref_census_handle = client.create_from_slice(u32::as_bytes(&ref_census));
    let src_census_handle = client.create_from_slice(u32::as_bytes(&src_census_packed));
    let r_handle = client.create_from_slice(f32::as_bytes(&r_sr));
    let t_handle = client.create_from_slice(f32::as_bytes(&t_sr));
    let f_handle = client.create_from_slice(f32::as_bytes(&f_src));
    drop(src_census_packed);
    drop(ref_census);
    drop(normal_flat);
    drop(depth);

    // 2D grid：全分辨率下 n/256 可能超过 wgpu 单维 65535 上限，用 x/y 两维分摊
    let groups = n.div_ceil(CUBE_THREADS as usize);
    let gx = 512u32;
    let gy = groups.div_ceil(gx as usize) as u32;
    let cube_count = CubeCount::Static(gx, gy, 1);
    let cube_dim = CubeDim::new_1d(CUBE_THREADS);

    // 红黑迭代：传播 + 扰动
    for iter in 0..num_iterations {
        let d_perturb = (dmax_f - dmin_f) * 0.5f32.powi(iter as i32);
        let n_perturb = 0.5f32.powi(iter as i32 + 1);
        for dir in 0..4u32 {
            let (dx, dy) = match dir {
                0 => (-1i32, 0i32),
                1 => (1, 0),
                2 => (0, -1),
                _ => (0, 1),
            };
            for color in 0..2u32 {
                let seed = (RANSAC_SEED as u32)
                    .wrapping_add(si as u32)
                    .wrapping_mul(0x9e3779b9)
                    .wrapping_add(iter as u32 * 131)
                    .wrapping_add(dir * 17)
                    .wrapping_add(color * 7);
                unsafe {
                    pm_propagate::launch_unchecked::<crate::backend::BackendRuntime>(
                        &client,
                        cube_count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(ref_census_handle.clone(), 3 * n),
                        ArrayArg::from_raw_parts(src_census_handle.clone(), n_nbr * 3 * n),
                        ArrayArg::from_raw_parts(r_handle.clone(), n_nbr * 9),
                        ArrayArg::from_raw_parts(t_handle.clone(), n_nbr * 3),
                        ArrayArg::from_raw_parts(f_handle.clone(), n_nbr),
                        ArrayArg::from_raw_parts(depth_handle.clone(), n),
                        ArrayArg::from_raw_parts(normal_handle.clone(), 3 * n),
                        ArrayArg::from_raw_parts(cost_handle.clone(), n),
                        cams[si].f as f32,
                        cx as f32,
                        cy as f32,
                        iw,
                        ih,
                        patch_half,
                        n_nbr,
                        dx,
                        dy,
                        color,
                        dmin_f,
                        dmax_f,
                        d_perturb,
                        n_perturb,
                        seed,
                    );
                }
            }
        }
    }

    // 读回
    let depth_bytes = client.read_one(depth_handle).expect("read depth");
    let normal_bytes = client.read_one(normal_handle).expect("read normal");
    let cost_bytes = client.read_one(cost_handle).expect("read cost");
    let depth = f32::from_bytes(&depth_bytes).to_vec();
    let normal_flat_out = f32::from_bytes(&normal_bytes).to_vec();
    let cost = f32::from_bytes(&cost_bytes).to_vec();

    let mut normal = vec![[0.0f32, 0.0f32, 1.0f32]; n];
    for i in 0..n {
        normal[i] = [
            normal_flat_out[i * 3],
            normal_flat_out[i * 3 + 1],
            normal_flat_out[i * 3 + 2],
        ];
    }

    crate::backend::memory_cleanup();

    (depth, normal, cost)
}

/// 深度图融合（单参考相机）：对每像素反投影为 3D 点，在邻帧间做深度一致性检查，
/// 达到 min_consistent 个一致视角才保留该点（仿 COLMAP StereoFusion 的几何一致性）。
///
/// `nbr_depths`/`nbr_normals` 与 `neighbors` 一一对应。
fn fuse_camera(
    si: usize,
    cams: &[RegCam],
    neighbors: &[usize],
    depth: &[f32],
    normal: &[[f32; 3]],
    costs: &[f32],
    nbr_depths: &[&[f32]],
    nbr_normals: &[&[[f32; 3]]],
    color: &RgbImage,
    iw: usize,
    ih: usize,
    cx: f64,
    cy: f64,
    cost_min: f32,
    depth_err_px: f64,
    min_consistent: usize,
    max_normal_error: f64,
) -> Vec<(f64, f64, f64, u8, u8, u8)> {
    let max_normal_cos = max_normal_error.cos();
    let f = cams[si].f;
    let r_w = cams[si].r.transpose();
    let mut fused: Vec<(f64, f64, f64, u8, u8, u8)> = Vec::new();
    for y in 0..ih {
        for x in 0..iw {
            let idx = y * iw + x;
            let d = depth[idx];
            if d <= 0.0 || costs[idx] < cost_min {
                continue;
            }
            let d_f = d as f64;
            let u = (x as f64 - cx) / f;
            let v = (y as f64 - cy) / f;
            let xc = Vector3::new(u * d_f, v * d_f, d_f);
            let xw = r_w * (xc - cams[si].t);

            // 参考视角的法向（相机系 → 世界系）
            let nc = normal[idx];
            let nw = r_w * Vector3::new(nc[0] as f64, nc[1] as f64, nc[2] as f64);

            // 几何一致性：投影到邻帧，比较深度 + 法向夹角
            let mut consistent = 0usize;
            for (k, &ri) in neighbors.iter().enumerate() {
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
                let d2 = nbr_depths[k][j];
                if d2 <= 0.0 {
                    continue;
                }
                if (d2 as f64 - xc2.z).abs() / xc2.z < depth_err_px {
                    // 法向一致性：两视角估计的法向（世界系）夹角要小
                    let nc2 = nbr_normals[k][j];
                    let nw2 = cams[ri].r.transpose()
                        * Vector3::new(nc2[0] as f64, nc2[1] as f64, nc2[2] as f64);
                    if nw.dot(&nw2) > max_normal_cos {
                        consistent += 1;
                    }
                }
            }
            if consistent < min_consistent {
                continue;
            }
            let pix = color.get_pixel(x as u32, y as u32);
            fused.push((xw.x, xw.y, xw.z, pix[0], pix[1], pix[2]));
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

/// 深度图落盘目录。
const DM_DIR: &str = crate::types::MVS_DM_DIR;
/// 融合阶段内存中的图缓存上限（张）。
const FUSION_CACHE_CAP: usize = 16;

/// 颜色感知双边平滑：窗口内按灰度相似度加权平均深度。
/// 平滑平坦区（去噪），保留楼房边缘（灰度差大处权重低，不混合）。
fn bilateral_smooth(depth: &[f32], gray: &[u8], iw: usize, ih: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let radius = 3usize;
    let sigma = 20.0f32;
    let sigma2 = 2.0 * sigma * sigma;
    let mut smoothed = depth.to_vec();
    smoothed
        .par_chunks_mut(iw)
        .enumerate()
        .for_each(|(y, row)| {
            if y < radius || y >= ih - radius {
                return;
            }
            for x in radius..iw - radius {
                let idx = y * iw + x;
                let d0 = depth[idx];
                if d0 <= 0.0 {
                    continue;
                }
                let g0 = gray[idx] as f32;
                let mut wsum = 0.0f32;
                let mut dsum = 0.0f32;
                for dy in -(radius as i64)..=(radius as i64) {
                    for dx in -(radius as i64)..=(radius as i64) {
                        let j = (y as i64 + dy) as usize * iw + (x as i64 + dx) as usize;
                        let d = depth[j];
                        if d <= 0.0 {
                            continue;
                        }
                        let g = gray[j] as f32;
                        let diff = g - g0;
                        let w = (-diff * diff / sigma2).exp();
                        wsum += w;
                        dsum += w * d;
                    }
                }
                if wsum > 1e-9 {
                    row[x] = dsum / wsum;
                }
            }
        });
    smoothed
}

/// 单相机深度图写盘：depth(n) + normal(n×3) + costs(n)，全部 f32。
fn write_depth_map(si: usize, depth: &[f32], normal: &[[f32; 3]], costs: &[f32]) {
    let mut flat: Vec<f32> = Vec::with_capacity(depth.len() * 5);
    flat.extend_from_slice(depth);
    for nrm in normal {
        flat.extend_from_slice(nrm);
    }
    flat.extend_from_slice(costs);
    let _ = fs::create_dir_all(DM_DIR);
    fs::write(format!("{}/{}.bin", DM_DIR, si), f32::as_bytes(&flat)).unwrap();
}

/// 单相机深度图读盘。
fn read_depth_map(si: usize, n: usize) -> (Vec<f32>, Vec<[f32; 3]>, Vec<f32>) {
    let buf = fs::read(format!("{}/{}.bin", DM_DIR, si)).unwrap();
    let flat: &[f32] = f32::from_bytes(&buf);
    let depth = flat[..n].to_vec();
    let normal = (0..n)
        .map(|i| {
            let o = n + i * 3;
            [flat[o], flat[o + 1], flat[o + 2]]
        })
        .collect();
    let costs = flat[n + n * 3..].to_vec();
    (depth, normal, costs)
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
    let scale = 1u32;
    let (iw, ih) = (ow / scale, oh / scale);
    let intrinsics = CameraIntrinsics::from_image_dimensions(ow, oh);
    let f_default = intrinsics.k[(0, 0)];
    // SfM 在特征分辨率(最长边≤2000px)下标定内参并写进 camera_poses.json，
    // MVS 加载全分辨率图像，需把 JSON 里的 f/cx/cy 换算回全分辨率，否则内参自相矛盾。
    let (fw, fh) = feature_dimensions(ow, oh);
    let sx = ow as f64 / fw as f64;
    let sy = oh as f64 / fh as f64;
    let jcx = poses[0]["cx"].as_f64().unwrap_or(intrinsics.k[(0, 2)]);
    let jcy = poses[0]["cy"].as_f64().unwrap_or(intrinsics.k[(1, 2)]);
    let cx_full = jcx * sx;
    let cy_full = jcy * sy;
    let cx = cx_full / scale as f64;
    let cy = cy_full / scale as f64;

    let mut cams: Vec<RegCam> = Vec::new();
    for pv in poses {
        let fname = pv["image"].as_str().unwrap();
        let img_idx = paths.iter().position(|p| p.contains(fname)).unwrap();
        let ra = pv["R"].as_array().unwrap();
        let ta = pv["t"].as_array().unwrap();
        // 使用稀疏重建阶段每图标定出的焦距（特征分辨率），换算到全分辨率，保证与 sparse.ply 几何一致
        let f_full = pv["f"].as_f64().unwrap_or(f_default) * sx;
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
        // 全分辨率下只保留紧凑灰度(1B)，Census 码在 PatchMatch 阶段按需从灰度实时计算，
        // 释放彩色图与全精度灰度，避免 24-bit Census 码(4B/像素)常驻 55 相机撑爆内存。
        let gray_u8: Vec<u8> = gray.iter().map(|&g| g.clamp(0.0, 255.0) as u8).collect();
        cams.push(RegCam {
            r,
            t,
            f: f_full / scale as f64,
            f_full,
            gray: gray_u8,
            path: paths[img_idx].clone(),
        });
    }
    if cams.len() < 2 {
        return;
    }
    let nc = cams.len();

    let n_nbr = N_NBR.min(nc - 1);
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
                    sparse_pts.push(Vector3::new(x, y, z));
                }
            }
        }
        // 统计离群移除：清掉 sparse 里散落在长方体中的离群点，只保留主体表面，
        // 避免深度范围/包围盒/distance-to-sparse 被离群点撑宽，导致 dense 变成长方体。
        let sparse_before = sparse_pts.len();
        sparse_pts = sparse_statistical_filter(&sparse_pts, 8, 1.0);
        println!(
            "  sparse outlier filter: {} -> {} points (kept {:.0}%)",
            sparse_before,
            sparse_pts.len(),
            sparse_pts.len() as f64 / sparse_before as f64 * 100.0
        );

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
                    // 用 2%~98% 分位（而非 25%~75%）覆盖完整场景深度，避免漏掉近处/远处的表面
                    let dlo = ds[ds.len() / 50].max(0.3);
                    let dhi = ds[ds.len() * 49 / 50].min(500.0);
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

    let num_iterations = 10usize; // PatchMatch 迭代次数（传播 + 扰动）
    let cost_min = 0.6f32; // 最小 Census 相似度（低于此剔除；随机匹配≈0.5）
    let min_consistent = 3usize; // 融合时最少一致视角数（提高以剔除漂浮的球状噪声）
    let depth_err_px = 0.01f64; // 深度一致性相对误差（收紧以剔除跨视角不一致的漂点）
    let max_normal_error = 60.0f64.to_radians(); // 法向一致性最大夹角（放宽）

    println!(
        "Downsampled {}x{} f={:.0}, PatchMatch stereo (iter={}, {} views)...",
        iw, ih, cams[0].f, num_iterations, n_nbr
    );

    // PatchMatch 逐参考图像估计深度/法向（80-bit Census + 连续法向，传播 + 扰动，并行）。
    // 全分辨率 + 9×9 Census(3×u32) + 8 邻帧下，每相机 GPU 缓冲约 2.5GB，
    // 用保守估计把并行度压到 1，保证显存安全。
    let mvs_threads = crate::backend::parallelism(iw as usize * ih as usize * 300 + (1 << 20));
    println!("  MVS parallelism: {} threads", mvs_threads);
    println!(
        "  memory before PatchMatch: {}",
        crate::backend::memory_usage()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(mvs_threads)
        .build()
        .expect("build mvs thread pool");

    let pb = ProgressBar::new(nc as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] PatchMatch {bar:40} {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    pool.install(|| {
        (0..nc).into_par_iter().for_each(|si| {
            let (dlo, dhi) = cam_depths[si];
            let sd = sparse_depth_map(si, &cams, &sparse_pts, iw as usize, ih as usize, cx, cy);

            // 设备丢失（TDR / 驱动重置）会以 panic 的形式从 wgpu 的 Device::poll 抛出，
            // 且无法被「预检测」——只能提交后才知道。全分辨率下 MVS 并行度=1，因此这里
            // 捕获 panic、重建设备并重试该相机，把间歇性丢失从致命崩溃降级为可恢复的重试；
            // 重试耗尽才恢复原始 panic（保留原始 backtrace）。
            const DEVICE_LOST_MAX_RETRIES: usize = 2;
            let mut attempt = 0usize;
            let (d, n, c) = loop {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    patch_match_image_gpu(
                        si,
                        &cams,
                        &neighbors[si],
                        dlo * 0.5,
                        dhi * 1.4,
                        iw as usize,
                        ih as usize,
                        cx,
                        cy,
                        num_iterations,
                        &sd,
                    )
                }));
                match outcome {
                    Ok(result) => break result,
                    Err(_) if attempt < DEVICE_LOST_MAX_RETRIES => {
                        attempt += 1;
                        eprintln!(
                            "[MVS] cam {}: GPU 设备丢失（device lost），重建设备后重试 ({}/{})",
                            si, attempt, DEVICE_LOST_MAX_RETRIES
                        );
                        crate::backend::reset_client();
                    }
                    Err(payload) => {
                        eprintln!(
                            "[MVS] cam {}: GPU 设备丢失（device lost），重试 {} 次后仍失败，放弃",
                            si, DEVICE_LOST_MAX_RETRIES
                        );
                        std::panic::resume_unwind(payload);
                    }
                }
            };

            // 诊断日志（每相机）
            println!("[DEBUG] cam {} depth {}", si, depth_stats_line(&d, &c));
            println!("[DEBUG] cam {} hist  {}", si, depth_histogram(&d, 10));
            let d = bilateral_smooth(&d, &cams[si].gray, iw as usize, ih as usize);
            write_depth_map(si, &d, &n, &c);
            pb.inc(1);
        });
    });
    pb.finish_and_clear();
    println!(
        "  depth maps computed + median filtered -> {} ({} cameras)",
        DM_DIR, nc
    );

    // 融合：逐相机从盘读回（带 LRU 缓存），反投影 + 邻帧深度一致性
    let n_px = iw as usize * ih as usize;
    let mut all_pts: Vec<(f64, f64, f64, u8, u8, u8)> = Vec::new();
    let mut cache: HashMap<usize, (Vec<f32>, Vec<[f32; 3]>, Vec<f32>)> = HashMap::new();
    for si in 0..nc {
        let mut needed: Vec<usize> = Vec::with_capacity(neighbors[si].len() + 1);
        needed.push(si);
        needed.extend_from_slice(&neighbors[si]);
        for &m in &needed {
            if !cache.contains_key(&m) {
                cache.insert(m, read_depth_map(m, n_px));
            }
        }
        {
            let (d_si, n_si, c_si) = &cache[&si];
            let nbr_depths: Vec<&[f32]> = neighbors[si]
                .iter()
                .map(|&ri| cache[&ri].0.as_slice())
                .collect();
            let nbr_normals: Vec<&[[f32; 3]]> = neighbors[si]
                .iter()
                .map(|&ri| cache[&ri].1.as_slice())
                .collect();
            // 全分辨率下彩色图不常驻内存，融合时按需从磁盘重读
            let color = image::open(&cams[si].path)
                .unwrap()
                .resize_exact(iw as u32, ih as u32, image::imageops::FilterType::Lanczos3)
                .to_rgb8();
            all_pts.extend(fuse_camera(
                si,
                &cams,
                &neighbors[si],
                d_si,
                n_si,
                c_si,
                &nbr_depths,
                &nbr_normals,
                &color,
                iw as usize,
                ih as usize,
                cx,
                cy,
                cost_min,
                depth_err_px,
                min_consistent,
                max_normal_error,
            ));
        }
        // LRU 淘汰：超过上限时淘汰距 si 最远且不在当前邻域中的图
        while cache.len() > FUSION_CACHE_CAP {
            let evict = cache
                .keys()
                .filter(|&&k| k != si && !neighbors[si].contains(&k))
                .map(|&k| (k, (k as i64 - si as i64).abs()))
                .max_by_key(|&(_, d)| d)
                .map(|(k, _)| k);
            match evict {
                Some(k) => {
                    cache.remove(&k);
                }
                None => break,
            }
        }
    }
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

    // 网格化暂时关闭：专注 dense.ply 的点云质量。需要时把 ENABLE_MESH 改回 true。
    if !ENABLE_MESH {
        return;
    }

    // ===== 网格化：稀疏点云距离约束（逐相机读盘后内联过滤）=====
    let (sgrid, smax_dist, scell): (Option<HashMap<(i64, i64, i64), Vec<usize>>>, f64, f64) =
        if sparse_pts.is_empty() {
            (None, 0.0, 0.0)
        } else {
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
            (Some(sgrid), max_dist, cell)
        };

    // ===== 贴图模型：深度图网格三角化 + 单张纹理图集烘焙 =====
    println!("\n===== Building textured mesh =====");

    struct Face {
        v: [Vector3<f64>; 3],
        uv: [(f64, f64); 3],
        cam: usize,
    }
    let mut faces: Vec<Face> = Vec::new();
    for si in 0..nc {
        let (mut depth, _normal, costs) = read_depth_map(si, n_px);
        let r_w = cams[si].r.transpose();
        // 稀疏距离过滤（内联，替代原来的整图过滤）
        if let Some(sgrid) = &sgrid {
            let f = cams[si].f;
            for y in 0..ih {
                for x in 0..iw {
                    let idx = (y * iw + x) as usize;
                    let d = depth[idx];
                    if d <= 0.0 {
                        continue;
                    }
                    let d_f = d as f64;
                    let u = (x as f64 - cx) / f;
                    let v = (y as f64 - cy) / f;
                    let xc = Vector3::new(u * d_f, v * d_f, d_f);
                    let xw = r_w * (xc - cams[si].t);
                    let cxg = (xw.x / scell) as i64;
                    let cyg = (xw.y / scell) as i64;
                    let czg = (xw.z / scell) as i64;
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
                                            < smax_dist * smax_dist
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
                        depth[idx] = 0.0;
                    }
                }
            }
        }

        let mut wg: Vec<Vec<Option<Vector3<f64>>>> = vec![vec![None; iw as usize]; ih as usize];
        let mut dg: Vec<Vec<f64>> = vec![vec![0.0; iw as usize]; ih as usize];
        for y in 0..ih {
            for x in 0..iw {
                let d = depth[(y * iw + x) as usize];
                let score = costs[(y * iw + x) as usize];
                if d <= 0.0 || score < cost_min {
                    continue;
                }
                let d_f = d as f64;
                let u = (x as f64 - cx) / cams[si].f;
                let v = (y as f64 - cy) / cams[si].f;
                let xc = Vector3::new(u * d_f, v * d_f, d_f);
                let xw = r_w * (xc - cams[si].t);
                wg[y as usize][x as usize] = Some(xw);
                dg[y as usize][x as usize] = d_f;
            }
        }
        for y in (0..ih as usize - 1).step_by(MESH_STEP) {
            for x in (0..iw as usize - 1).step_by(MESH_STEP) {
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
    atlas_side = atlas_side.max(512).min(ATLAS_MAX_SIDE).next_power_of_two();
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
