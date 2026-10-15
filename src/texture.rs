//! 给 DSM 网格贴真实图像：UV 图集（atlas）+ 深度图遮挡剔除。
//!
//! 输入：DSM 网格顶点/三角形（世界坐标）+ camera_poses.json + 全分辨率原图 +
//! MVS 深度图缓存（cache/mvs_dm/{si}.bin）。
//! 输出：`{prefix}.glb`（单一二进制 glTF 2.0，网格 + 内嵌图集纹理）。
//!
//! 两趟内存安全算法：第一趟逐相机（只载一张深度图）给每个三角形选最佳视角并做
//! 深度图遮挡测试；第二趟逐相机（只载一张 RGB 图）把分配给该相机的三角形光栅化进
//! 图集。全程内存 O(面数) + 单相机，不随相机数线性堆内存。

use std::fs;

use image::{imageops::FilterType, DynamicImage, ImageFormat, RgbImage};
use nalgebra::{Matrix3, Vector3};

/// 每个三角形图集 patch 的最小/最大边长（纹素，不含 margin）。
const MIN_PATCH: u32 = 8;
const MAX_PATCH: u32 = 64;
/// patch 之间的间隔（纹素），避免双线性采样时相邻 patch 串色。
const PATCH_MARGIN: u32 = 2;
/// patch 内部三角形到边界的最小留白（纹素），避免贴到填充色边缘。
const PATCH_INSET: u32 = 1;
/// 图集长边上限与总面积预算（纹素数）。超过预算会自动缩小全局贴图尺度。
const ATLAS_MAX_SIDE: u32 = 4096;
const ATLAS_TARGET: f64 = (ATLAS_MAX_SIDE as f64) * (ATLAS_MAX_SIDE as f64);
/// 全局贴图尺度（1.0 = 投影面积 1:1 出图；总面积超预算时自动缩小以适配）。
const TEX_SCALE: f64 = 1.0;
/// 遮挡测试：顶点深度显著大于深度图记录值（近处有遮挡物）即判定被遮挡。
/// 容差先放宽（10% + 2.0m）：DSM 经中值滤波/结构清理与原始 MVS 深度图有系统性出入，
/// 过紧会把可见面误判为被遮挡（结果整片灰）。后续再收缩。
const OCC_TOL_REL: f64 = 0.10;
const OCC_TOL_ABS: f64 = 2.0;
/// 与 MVS 融合一致的置信度阈值：cost 低于此的深度图像素视为未知，跳过遮挡测试。
const COST_MIN: f32 = 0.6;
/// 视锥边界留白（像素）：顶点离图像边缘太近视为不可靠（避免取到图像外）。
const FRUSTUM_BORDER: f64 = 2.0;
/// 每台相机的白平衡/曝光增益上下限（把各相机亮度归一到全局平均，抑制相邻相机接缝）。
const GAIN_MIN: f64 = 0.5;
const GAIN_MAX: f64 = 2.0;

/// 纹理贴图用的相机：世界→相机位姿 + 全分辨率内参 + 原图路径。
struct Cam {
    r: Matrix3<f64>,
    t: Vector3<f64>,
    center: Vector3<f64>,
    f: f64,
    cx: f64,
    cy: f64,
    path: String,
    w: u32,
    h: u32,
}

/// 每个三角形的视角分配结果。
struct FaceAssign {
    cam: Option<usize>,
    /// 入射角余弦（法向·视线），越大越正对。
    score: f64,
    /// 所选相机内的投影像素坐标（三顶点）。
    proj: [[f64; 2]; 3],
    /// 所选相机内的投影像素面积。
    area: f64,
}

/// 世界点投影到相机：返回 (像素 u, 像素 v, 深度 z)。
#[inline]
fn project(c: &Cam, p: &Vector3<f64>) -> (f64, f64, f64) {
    let x = c.r * p + c.t;
    let u = c.f * x.x / x.z + c.cx;
    let v = c.f * x.y / x.z + c.cy;
    (u, v, x.z)
}

/// 投影三角形的像素面积（2D 叉积的一半）。
fn tri_area_px(p: &[[f64; 2]; 3]) -> f64 {
    let ax = p[1][0] - p[0][0];
    let ay = p[1][1] - p[0][1];
    let bx = p[2][0] - p[0][0];
    let by = p[2][1] - p[0][1];
    (ax * by - ay * bx).abs() * 0.5
}

/// 从 camera_poses.json 加载相机（与 mvs.rs 一致：feature-res f/cx/cy → 全分辨率）。
fn load_cams(paths: &[String]) -> Result<Vec<Cam>, String> {
    let s = fs::read_to_string("camera_poses.json").map_err(|e| format!("camera_poses.json: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| e.to_string())?;
    let poses = v["poses"].as_array().ok_or("no poses array")?;
    if poses.len() < 2 {
        return Err("need >=2 cameras".into());
    }

    let first_img = image::open(&paths[0]).map_err(|e| e.to_string())?;
    let (ow, oh) = (first_img.width(), first_img.height());
    let (fw, fh) = crate::imgproc::feature_dimensions(ow, oh);
    let sx = ow as f64 / fw as f64;
    let sy = oh as f64 / fh as f64;
    // 与 mvs.rs 完全一致：f 逐相机（feature-res → 全分辨率），cx/cy 取 poses[0] 全局值。
    let f_default = ow.max(oh) as f64;
    let cx_full = poses[0]["cx"].as_f64().unwrap_or(ow as f64 / 2.0) * sx;
    let cy_full = poses[0]["cy"].as_f64().unwrap_or(oh as f64 / 2.0) * sy;

    let mut cams = Vec::with_capacity(poses.len());
    for pv in poses {
        let fname = pv["image"].as_str().unwrap();
        let img_idx = paths
            .iter()
            .position(|p| p.contains(fname))
            .ok_or_else(|| format!("image '{fname}' not in dir"))?;
        let ra = pv["R"].as_array().ok_or("R missing")?;
        let ta = pv["t"].as_array().ok_or("t missing")?;
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
        let center = -r.transpose() * t;
        cams.push(Cam {
            r,
            t,
            center,
            f: f_full,
            cx: cx_full,
            cy: cy_full,
            path: paths[img_idx].clone(),
            w: ow,
            h: oh,
        });
    }
    Ok(cams)
}

/// 只读深度图缓存的前 n 个 f32（depth）与第 4n 起 n 个 f32（cost），跳过法向，节省内存。
fn read_depth_cost(si: usize, n: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = fs::File::open(format!("{}/{}.bin", crate::types::MVS_DM_DIR, si)).ok()?;
    let mut dbuf = vec![0u8; n * 4];
    f.read_exact(&mut dbuf).ok()?;
    // 布局：depth(n) + normal(3n) + cost(n)，cost 从第 4n 个 f32（字节 16n）起。
    f.seek(SeekFrom::Start((4 * n * 4) as u64)).ok()?;
    let mut cbuf = vec![0u8; n * 4];
    f.read_exact(&mut cbuf).ok()?;
    Some((bytes_to_f32(&dbuf), bytes_to_f32(&cbuf)))
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 双线性采样（坐标已 clamp 到 [0, w-1]×[0, h-1]）。
fn sample_rgb(img: &RgbImage, x: f64, y: f64) -> [u8; 3] {
    let w = img.width() as f64;
    let h = img.height() as f64;
    let x = x.clamp(0.0, w - 1.0);
    let y = y.clamp(0.0, h - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let x0i = x0 as u32;
    let y0i = y0 as u32;
    let x1i = (x0i + 1).min(img.width() - 1);
    let y1i = (y0i + 1).min(img.height() - 1);
    let p00 = img.get_pixel(x0i, y0i);
    let p10 = img.get_pixel(x1i, y0i);
    let p01 = img.get_pixel(x0i, y1i);
    let p11 = img.get_pixel(x1i, y1i);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = lerp(p00[c] as f64, p10[c] as f64, fx);
        let bot = lerp(p01[c] as f64, p11[c] as f64, fx);
        out[c] = lerp(top, bot, fy).round() as u8;
    }
    out
}

/// 应用每相机白平衡/曝光增益（逐通道相乘并截断）。
#[inline]
fn apply_gain(rgb: [u8; 3], gain: [f64; 3]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = ((rgb[c] as f64) * gain[c]).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// 一张 RGB 图的逐通道平均色（确定性下采样统计，最多约 1M 像素，均值不受影响）。
fn mean_color(img: &RgbImage) -> [f64; 3] {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let n = w * h;
    let step = (n / 1_000_000).max(1);
    let mut acc = [0.0f64; 3];
    let mut cnt = 0u64;
    let mut i = 0usize;
    while i < n {
        let p = img.get_pixel((i % w) as u32, (i / w) as u32);
        acc[0] += p[0] as f64;
        acc[1] += p[1] as f64;
        acc[2] += p[2] as f64;
        cnt += 1;
        i += step;
    }
    let cnt = cnt.max(1) as f64;
    [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt]
}

/// 解析 mesh.rs 写出的 OBJ（`v x y z [r g b]` / `f a b c`），返回 (顶点, 三角形)。
/// 贴图模块只消费 mesh.obj 这个磁盘产物，与 mesh.rs 内部实现解耦。
fn load_mesh_obj(path: &str) -> Option<(Vec<Vector3<f64>>, Vec<[usize; 3]>)> {
    let s = fs::read_to_string(path).ok()?;
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("v ") {
            let mut it = rest.split_whitespace();
            let x: f64 = it.next()?.parse().ok()?;
            let y: f64 = it.next()?.parse().ok()?;
            let z: f64 = it.next()?.parse().ok()?;
            verts.push(Vector3::new(x, y, z));
        } else if let Some(rest) = t.strip_prefix("f ") {
            let mut idx = [0usize; 3];
            let mut ok = true;
            for (k, tok) in rest.split_whitespace().take(3).enumerate() {
                // 面顶点引用可能是 "v"、"v/vt"、"v//vn"、"v/vt/vn"，只取第一个字段。
                let v = tok.split('/').next().unwrap_or(tok);
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => idx[k] = n - 1,
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                tris.push(idx);
            }
        }
    }
    if verts.is_empty() || tris.is_empty() {
        return None;
    }
    Some((verts, tris))
}

/// 贴图独立入口：读取 mesh.obj（网格阶段产物）+ camera_poses.json + 原图 + MVS 深度图，
/// 写出 `mesh_textured.glb`。与 MVS 管线解耦，只消费其磁盘产物。
pub fn run_texture(paths: &[String]) {
    let Some((verts, tris)) = load_mesh_obj("mesh.obj") else {
        eprintln!("  [texture] mesh.obj 不存在或为空，请先跑 -stage=mesh 生成网格");
        return;
    };
    texture_mesh(paths, &verts, &tris, "mesh_textured");
}

/// 给 DSM 网格贴真实图像，写出 `{out_prefix}.glb`（内嵌图集纹理）。
fn texture_mesh(
    paths: &[String],
    verts: &[Vector3<f64>],
    tris: &[[usize; 3]],
    out_prefix: &str,
) {
    let nt = tris.len();
    if nt == 0 || verts.is_empty() {
        println!("  [texture] 空网格，跳过纹理贴图");
        return;
    }

    let cams = match load_cams(paths) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  [texture] 无法加载相机: {e}，跳过纹理贴图");
            return;
        }
    };
    let nc = cams.len();

    println!("\n===== Texturing DSM mesh with {} cameras =====", nc);

    // 预计算每个三角形的质心与几何法向（DSM 三角均朝上 +z）。
    let mut centroids = Vec::with_capacity(nt);
    let mut normals = Vec::with_capacity(nt);
    for t in tris {
        let a = verts[t[0]];
        let b = verts[t[1]];
        let c = verts[t[2]];
        centroids.push((a + b + c) / 3.0);
        normals.push((b - a).cross(&(c - a)));
    }

    // ---- 第一趟：逐相机选最佳视角 + 深度图遮挡测试（多数票，宽松容差） ----
    let mut assign: Vec<FaceAssign> = (0..nt)
        .map(|_| FaceAssign {
            cam: None,
            score: -1.0,
            proj: [[0.0; 2]; 3],
            area: 0.0,
        })
        .collect();

    // 诊断：每个面在「任一台相机」下是否通过过视锥（用于归类未分配原因）。
    let mut seen_frustum = vec![false; nt]; // 通过视锥（含深度>0）
    let mut cams_no_depth = 0usize;

    for ci in 0..nc {
        let cam = &cams[ci];
        let n_px = (cam.w as usize) * (cam.h as usize);
        let Some((depth, cost)) = read_depth_cost(ci, n_px) else {
            cams_no_depth += 1; // 该相机无深度图，跳过
            continue;
        };
        let w = cam.w as f64;
        let h = cam.h as f64;
        let wu = cam.w as usize;

        for ti in 0..nt {
            let centroid = centroids[ti];
            let nrm = normals[ti];
            let to_cam = cam.center - centroid;
            let dist = to_cam.norm();
            if dist <= 1e-9 {
                continue;
            }
            // 入射角余弦（>0 朝上、<0 朝下）。DSM 是单层薄片、两侧都可见，
            // 不做背面剔除（z 轴翻转的重建里 cos 全为负，硬剔除会把所有顶面判死）。
            let cos_inc = nrm.normalize().dot(&(to_cam / dist));
            let inc = cos_inc.abs(); // 越接近 1 越正对，用于排序

            // 投影三顶点，要求全部在视锥内且深度 > 0。
            let mut proj = [[0.0f64; 2]; 3];
            let mut deps = [0.0f64; 3];
            let mut in_frustum = true;
            for k in 0..3 {
                let (u, v, d) = project(cam, &verts[tris[ti][k]]);
                if d <= 0.0
                    || u < FRUSTUM_BORDER
                    || v < FRUSTUM_BORDER
                    || u > w - FRUSTUM_BORDER
                    || v > h - FRUSTUM_BORDER
                {
                    in_frustum = false;
                    break;
                }
                proj[k] = [u, v];
                deps[k] = d;
            }
            if !in_frustum {
                continue;
            }
            seen_frustum[ti] = true;

            // 遮挡测试：三顶点 + 质心；只有全部点（4/4）被挡才放弃该相机。
            // 单点深度噪声 / DSM 与原始深度图的轻微出入不再整面判死。
            let (cu, cv, cd) = project(cam, &centroid);
            let mut occ_count = 0usize;
            for k in 0..4 {
                let (u, v, d) = if k < 3 {
                    (proj[k][0], proj[k][1], deps[k])
                } else {
                    (cu, cv, cd)
                };
                let xi = u.round() as i64;
                let yi = v.round() as i64;
                if xi < 0 || yi < 0 || xi >= cam.w as i64 || yi >= cam.h as i64 {
                    occ_count += 1;
                    continue;
                }
                let di = (yi as usize) * wu + xi as usize;
                let dm = depth[di];
                if dm > 0.0
                    && cost[di] >= COST_MIN
                    && d > dm as f64 * (1.0 + OCC_TOL_REL) + OCC_TOL_ABS
                {
                    occ_count += 1;
                }
            }
            if occ_count >= 4 {
                continue; // 全部采样点都被挡：该相机确实看不到此面
            }

            let area = tri_area_px(&proj);
            // 正对度（|入射角余弦|）主导，面积在相近时打破平局（更大 = 更清晰）。
            let better = match assign[ti].cam {
                None => true,
                Some(_) => {
                    if inc > assign[ti].score + 1e-3 {
                        true
                    } else if (inc - assign[ti].score).abs() <= 1e-3
                        && area > assign[ti].area
                    {
                        true
                    } else {
                        false
                    }
                }
            };
            if better {
                assign[ti] = FaceAssign {
                    cam: Some(ci),
                    score: inc,
                    proj,
                    area,
                };
            }
        }
    }

    let assigned = assign.iter().filter(|a| a.cam.is_some()).count();

    // ---- 第二趟：兜底 —— 对第一趟未分配的面，去掉遮挡测试再选一次（只看视锥）。
    // DSM 是 2.5D 高度场，近正下视几乎无自遮挡；兜底保证「至少被一台相机看到」的面都能贴上。
    let mut fallback = 0usize;
    for ti in 0..nt {
        if assign[ti].cam.is_some() {
            continue;
        }
        let mut best: Option<(usize, f64, f64, [[f64; 2]; 3])> = None;
        for ci in 0..nc {
            let cam = &cams[ci];
            let to_cam = cam.center - centroids[ti];
            let dist = to_cam.norm();
            if dist <= 1e-9 {
                continue;
            }
            let inc = normals[ti].normalize().dot(&(to_cam / dist)).abs();
            let mut proj = [[0.0f64; 2]; 3];
            let mut ok = true;
            for k in 0..3 {
                let (u, v, d) = project(cam, &verts[tris[ti][k]]);
                if d <= 0.0
                    || u < FRUSTUM_BORDER
                    || v < FRUSTUM_BORDER
                    || u > cam.w as f64 - FRUSTUM_BORDER
                    || v > cam.h as f64 - FRUSTUM_BORDER
                {
                    ok = false;
                    break;
                }
                proj[k] = [u, v];
            }
            if !ok {
                continue;
            }
            let area = tri_area_px(&proj);
            let better = match best {
                None => true,
                Some((_, bs, ba, _)) => {
                    if inc > bs + 1e-3 {
                        true
                    } else if (inc - bs).abs() <= 1e-3 && area > ba {
                        true
                    } else {
                        false
                    }
                }
            };
            if better {
                best = Some((ci, inc, area, proj));
            }
        }
        if let Some((ci, score, area, proj)) = best {
            assign[ti] = FaceAssign {
                cam: Some(ci),
                score,
                proj,
                area,
            };
            fallback += 1;
        }
    }

    let assigned_final = assign.iter().filter(|a| a.cam.is_some()).count();
    if cams_no_depth > 0 {
        println!("  [warn] {} 台相机无深度图，第一趟跳过", cams_no_depth);
    }
    println!(
        "  best-view assignment: {} strict + {} fallback = {} / {} faces",
        assigned, fallback, assigned_final, nt
    );
    // 归类最终仍未分配面的原因（只统计，不影响结果）。
    let mut r_frustum = 0usize;
    let mut r_occ = 0usize;
    for ti in 0..nt {
        if assign[ti].cam.is_some() {
            continue;
        }
        if !seen_frustum[ti] {
            r_frustum += 1;
        } else {
            r_occ += 1;
        }
    }
    if assigned_final < nt {
        println!(
            "  untextured: {} (frustum {} / occluded {})",
            nt - assigned_final,
            r_frustum,
            r_occ
        );
    }

    // ---- patch 尺寸：边长 = clamp(sqrt(面积) * scale, MIN, MAX)，全局 scale 适配图集预算 ----
    let mut side: Vec<u32> = vec![MIN_PATCH; nt];
    let mut sqrt_area = vec![0.0f64; nt];
    for ti in 0..nt {
        if assign[ti].cam.is_some() {
            sqrt_area[ti] = assign[ti].area.sqrt();
        }
    }
    let raw_total: f64 = sqrt_area
        .iter()
        .map(|&s| s.clamp(MIN_PATCH as f64, MAX_PATCH as f64).powi(2))
        .sum();
    let scale = if raw_total > ATLAS_TARGET {
        TEX_SCALE * (ATLAS_TARGET / raw_total).sqrt()
    } else {
        TEX_SCALE
    };
    for ti in 0..nt {
        side[ti] = (sqrt_area[ti] * scale).clamp(MIN_PATCH as f64, MAX_PATCH as f64) as u32;
    }

    // ---- shelf 打包：按有效边长降序放进固定宽度的货架 ----
    // 先求图集宽度（≈ sqrt(总面积)，向上取 2 的幂，限制在 [512, 4096]）。
    let total_eff_area: f64 = side
        .iter()
        .map(|&s| (s + 2 * PATCH_MARGIN) as f64 * (s + 2 * PATCH_MARGIN) as f64)
        .sum();
    let atlas_w = (total_eff_area.sqrt().ceil() as u32)
        .clamp(512, ATLAS_MAX_SIDE)
        .next_power_of_two();

    let mut order: Vec<usize> = (0..nt).filter(|&ti| side[ti] > 0).collect();
    order.sort_by_key(|&ti| std::cmp::Reverse((side[ti], ti))); // 稳定 + 确定性

    let mut patch_ox = vec![0u32; nt];
    let mut patch_oy = vec![0u32; nt];
    let mut shelf_x: u32 = 0;
    let mut shelf_y: u32 = 0;
    let mut shelf_h: u32 = 0;
    for &ti in &order {
        let eff = side[ti] + 2 * PATCH_MARGIN;
        if shelf_x + eff > atlas_w {
            shelf_y += shelf_h;
            shelf_x = 0;
            shelf_h = 0;
        }
        if eff > shelf_h {
            shelf_h = eff;
        }
        patch_ox[ti] = shelf_x + PATCH_MARGIN;
        patch_oy[ti] = shelf_y + PATCH_MARGIN;
        shelf_x += eff;
    }
    let atlas_h = (shelf_y + shelf_h).max(1);
    let aw = atlas_w as usize;
    let ah = atlas_h as usize;
    println!(
        "  atlas {}x{} px, {} patches, tex scale {:.3}",
        atlas_w, atlas_h, order.len(), scale
    );

    // ---- 第二趟：逐相机把分配到的三角形光栅化进图集（单相机内存） ----
    let mut atlas: Vec<u8> = vec![128u8; aw * ah * 3];
    let mut cam_faces: Vec<Vec<usize>> = vec![Vec::new(); nc];
    for ti in 0..nt {
        if let Some(ci) = assign[ti].cam {
            cam_faces[ci].push(ti);
        }
    }

    // ---- 色彩补偿：逐相机平均色 → 全局平均 → 每相机增益。
    // 道路/湖面等大块匀色区域对相机间曝光/白平衡差最敏感：统一亮度后斑块感大幅下降。
    let mut cam_gain: Vec<[f64; 3]> = vec![[1.0; 3]; nc];
    {
        let mut cam_mean: Vec<[f64; 3]> = vec![[0.0; 3]; nc];
        let mut cam_used = vec![false; nc];
        for ci in 0..nc {
            if cam_faces[ci].is_empty() {
                continue;
            }
            if let Ok(im) = image::open(&cams[ci].path) {
                let im = im
                    .resize_exact(cams[ci].w, cams[ci].h, FilterType::Lanczos3)
                    .to_rgb8();
                cam_mean[ci] = mean_color(&im);
                cam_used[ci] = true;
            }
        }
        let mut global = [0.0f64; 3];
        let mut used_cnt = 0usize;
        for ci in 0..nc {
            if cam_used[ci] {
                global[0] += cam_mean[ci][0];
                global[1] += cam_mean[ci][1];
                global[2] += cam_mean[ci][2];
                used_cnt += 1;
            }
        }
        if used_cnt > 0 {
            for c in 0..3 {
                global[c] /= used_cnt as f64;
            }
            for ci in 0..nc {
                if cam_used[ci] {
                    for c in 0..3 {
                        cam_gain[ci][c] =
                            (global[c] / cam_mean[ci][c].max(1.0)).clamp(GAIN_MIN, GAIN_MAX);
                    }
                }
            }
        }
        println!(
            "  color gain: {} cams, global mean ({:.0},{:.0},{:.0})",
            used_cnt, global[0], global[1], global[2]
        );
    }

    for ci in 0..nc {
        if cam_faces[ci].is_empty() {
            continue;
        }
        let img = match image::open(&cams[ci].path) {
            Ok(im) => im.resize_exact(cams[ci].w, cams[ci].h, FilterType::Lanczos3).to_rgb8(),
            Err(e) => {
                eprintln!("  [texture] 读图 {} 失败: {}", cams[ci].path, e);
                continue;
            }
        };
        for &ti in &cam_faces[ci] {
            rasterize_face(
                &img,
                &assign[ti].proj,
                patch_ox[ti],
                patch_oy[ti],
                side[ti],
                aw,
                &mut atlas,
                cam_gain[ci],
            );
        }
    }

    // ---- 写出 GLB（单一二进制 glTF，内嵌图集纹理） ----
    let glb_path = format!("{out_prefix}.glb");
    write_glb(
        verts,
        tris,
        &patch_ox,
        &patch_oy,
        &side,
        atlas_w,
        atlas_h,
        atlas,
        &glb_path,
    );
    println!(
        "  -> {} ({} faces, {} vertices, atlas {}x{})",
        glb_path, nt, nt * 3, atlas_w, atlas_h
    );
}

/// 把 f32 切片按小端序追加进字节缓冲。
fn push_f32s(out: &mut Vec<u8>, vals: &[f32]) {
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// 写出 GLB（glTF 2.0 二进制）：
/// - 网格：每面独立 3 顶点（不共享），POSITION/NORMAL/TEXCOORD_0 + mode=4 非索引三角形；
/// - 纹理：图集以 PNG 字节内嵌进 BIN buffer（images[].bufferView + mimeType image/png）。
/// glTF 约定 UV (0,0) 在纹理左上角，因此 v = ty/atlas_h（无需 OBJ 式翻转）。
fn write_glb(
    verts: &[Vector3<f64>],
    tris: &[[usize; 3]],
    patch_ox: &[u32],
    patch_oy: &[u32],
    side: &[u32],
    atlas_w: u32,
    atlas_h: u32,
    atlas: Vec<u8>,
    out_path: &str,
) {
    let nt = tris.len();
    let nv = nt * 3;

    // ---- 顶点 / 法向 / UV ----
    let mut positions: Vec<f32> = Vec::with_capacity(nv * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(nv * 3);
    let mut uvs: Vec<f32> = Vec::with_capacity(nv * 2);
    let mut pmin = [f64::MAX; 3];
    let mut pmax = [f64::MIN; 3];
    let inset = PATCH_INSET as f64;
    let aw = atlas_w as f64;
    let ah = atlas_h as f64;

    for ti in 0..nt {
        let a = verts[tris[ti][0]];
        let b = verts[tris[ti][1]];
        let c = verts[tris[ti][2]];
        let n = (b - a).cross(&(c - a)).normalize();
        let corners = [
            (patch_ox[ti] as f64 + inset, patch_oy[ti] as f64 + inset),
            (
                patch_ox[ti] as f64 + side[ti] as f64 - inset,
                patch_oy[ti] as f64 + inset,
            ),
            (
                patch_ox[ti] as f64 + inset,
                patch_oy[ti] as f64 + side[ti] as f64 - inset,
            ),
        ];
        let vs = [a, b, c];
        for k in 0..3 {
            let v = vs[k];
            positions.push(v.x as f32);
            positions.push(v.y as f32);
            positions.push(v.z as f32);
            normals.push(n.x as f32);
            normals.push(n.y as f32);
            normals.push(n.z as f32);
            let (tx, ty) = corners[k];
            uvs.push((tx / aw) as f32);
            uvs.push((ty / ah) as f32);
            pmin[0] = pmin[0].min(v.x);
            pmin[1] = pmin[1].min(v.y);
            pmin[2] = pmin[2].min(v.z);
            pmax[0] = pmax[0].max(v.x);
            pmax[1] = pmax[1].max(v.y);
            pmax[2] = pmax[2].max(v.z);
        }
    }

    // ---- 图集 PNG 编码（内嵌） ----
    let img = RgbImage::from_raw(atlas_w, atlas_h, atlas).expect("atlas buffer size mismatch");
    let dyn_img = DynamicImage::ImageRgb8(img);
    let mut png = Vec::new();
    dyn_img
        .write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)
        .expect("failed to encode atlas png");
    let png_len = png.len();
    let png_padded = (png_len + 3) & !3;
    png.resize(png_padded, 0);

    // ---- BIN buffer 布局（全部 4 字节对齐） ----
    let pos_len = nv * 12;
    let nrm_len = nv * 12;
    let uv_len = nv * 8;
    let pos_off = 0usize;
    let nrm_off = pos_off + pos_len;
    let uv_off = nrm_off + nrm_len;
    let img_off = uv_off + uv_len;
    let bin_len = img_off + png_padded;

    let mut bin: Vec<u8> = Vec::with_capacity(bin_len);
    push_f32s(&mut bin, &positions);
    push_f32s(&mut bin, &normals);
    push_f32s(&mut bin, &uvs);
    bin.extend_from_slice(&png);

    // ---- glTF JSON ----
    let json = serde_json::json!({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "material": 0,
            "mode": 4
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "doubleSided": true
        }],
        "textures": [{"source": 0, "sampler": 0}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "buffers": [{"byteLength": bin_len}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": nrm_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv_off, "byteLength": uv_len, "target": 34962},
            {"buffer": 0, "byteOffset": img_off, "byteLength": png_len}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3",
             "min": [pmin[0], pmin[1], pmin[2]],
             "max": [pmax[0], pmax[1], pmax[2]]},
            {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": nv, "type": "VEC2"}
        ]
    });
    let json_str = serde_json::to_string(&json).expect("serialize gltf json");
    let json_bytes = json_str.as_bytes();
    let json_padded = (json_bytes.len() + 3) & !3;

    // ---- GLB 容器：12 字节头 + JSON chunk + BIN chunk ----
    let total = 12 + 8 + json_padded + 8 + bin.len();
    let mut out: Vec<u8> = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_padded as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(json_bytes);
    for _ in json_bytes.len()..json_padded {
        out.push(b' ');
    }
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(&bin);

    fs::write(out_path, &out).expect("failed to write glb");
}

/// 把三角形光栅化进它在图集中的 patch：patch 内右三角（v0 左上、v1 右上、v2 左下），
/// 重心坐标插值出源图投影像素坐标后双线性采样。
fn rasterize_face(
    img: &RgbImage,
    proj: &[[f64; 2]; 3],
    ox: u32,
    oy: u32,
    side: u32,
    aw: usize,
    atlas: &mut [u8],
    gain: [f64; 3],
) {
    let inset = PATCH_INSET as f64;
    let l = (side as f64 - 2.0 * inset).max(1.0);
    let (ax, ay) = (proj[0][0], proj[0][1]);
    let (bx, by) = (proj[1][0], proj[1][1]);
    let (cx, cy) = (proj[2][0], proj[2][1]);
    // patch 先用三角形质心色填满，避免斜边外的灰底在双线性采样时串色。
    let fill = apply_gain(
        sample_rgb(img, (ax + bx + cx) / 3.0, (ay + by + cy) / 3.0),
        gain,
    );
    let p0x = ox as f64 + inset;
    let p0y = oy as f64 + inset;

    for ty in oy..oy + side {
        let v = (ty as f64 - p0y) / l;
        for tx in ox..ox + side {
            let u = (tx as f64 - p0x) / l;
            let o = (ty as usize * aw + tx as usize) * 3;
            if u < 0.0 || v < 0.0 || u + v > 1.0 {
                atlas[o] = fill[0];
                atlas[o + 1] = fill[1];
                atlas[o + 2] = fill[2];
                continue;
            }
            let sx = ax + u * (bx - ax) + v * (cx - ax);
            let sy = ay + u * (by - ay) + v * (cy - ay);
            let rgb = apply_gain(sample_rgb(img, sx, sy), gain);
            atlas[o] = rgb[0];
            atlas[o + 1] = rgb[1];
            atlas[o + 2] = rgb[2];
        }
    }
}
