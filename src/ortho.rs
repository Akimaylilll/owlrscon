//! 正射影像（orthophoto / true ortho）生成：把 DSM 场景渲染成一张固定地面分辨率的
//! 二维影像，用 DSM 高度做真射校正（消除建筑倾斜），多相机加权羽化 + 逐相机色彩增益
//! 抑制接缝，深度图遮挡剔除，输出带地理参考的 GeoTIFF。
//!
//! 输入：mesh.obj（DSM 网格）+ camera_poses.json + 全分辨率原图 + MVS 深度图缓存。
//! 输出：`ortho.tif`（GeoTIFF，ModelPixelScale/ModelTiepoint 地理参考）+ `ortho_preview.png`。
//!
//! 与 texture.rs 解耦、但复用同一套相机模型/内参约定/遮挡容差。内存安全：逐相机只载
//! 一张深度图 + 一张 RGB 图，把颜色与权重累加进全局缓冲，全程 O(输出像素) + 单相机。
//!
//! 注意：坐标是 SfM 局部系（无 GCP/EPSG），地理参考只是「像素→局部米」仿射，无 CRS。

use std::fs;

use image::{imageops::FilterType, DynamicImage, ImageFormat, RgbImage};
use nalgebra::{Matrix3, Vector3};

/// 视锥边界留白（像素）：像素离图像边缘太近视为不可靠。
const FRUSTUM_BORDER: f64 = 2.0;
/// 遮挡测试容差（与 texture.rs 一致，先放宽）：深度显著大于深度图记录值即判被挡。
const OCC_TOL_REL: f64 = 0.10;
const OCC_TOL_ABS: f64 = 2.0;
/// 与 MVS 融合一致的置信度阈值。
const COST_MIN: f32 = 0.6;
/// 逐相机白平衡/曝光增益上下限。
const GAIN_MIN: f64 = 0.5;
const GAIN_MAX: f64 = 2.0;
/// 输出总像素上限：超出自适应放宽 GSD，保护内存（4G 下限）。
const MAX_ORTHO_PX: u64 = 64_000_000;
const OUT_PREFIX: &str = "ortho";

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

/// 世界点投影到相机：返回 (像素 u, 像素 v, 深度 z)。
#[inline]
fn project(c: &Cam, p: &Vector3<f64>) -> (f64, f64, f64) {
    let x = c.r * p + c.t;
    let u = c.f * x.x / x.z + c.cx;
    let v = c.f * x.y / x.z + c.cy;
    (u, v, x.z)
}

/// 从 camera_poses.json 加载相机（与 mvs.rs / texture.rs 一致：f 逐相机、cx/cy 取 poses[0]）。
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

/// 只读深度图缓存的 depth(n) 与 cost(第 4n 个 f32 起 n 个)，跳过法向，节省内存。
fn read_depth_cost(si: usize, n: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = fs::File::open(format!("{}/{}.bin", crate::types::MVS_DM_DIR, si)).ok()?;
    let mut dbuf = vec![0u8; n * 4];
    f.read_exact(&mut dbuf).ok()?;
    f.seek(SeekFrom::Start((4 * n * 4) as u64)).ok()?;
    let mut cbuf = vec![0u8; n * 4];
    f.read_exact(&mut cbuf).ok()?;
    let mut depth = Vec::with_capacity(n);
    let mut cost = Vec::with_capacity(n);
    for c in dbuf.chunks_exact(4) {
        depth.push(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
    }
    for c in cbuf.chunks_exact(4) {
        cost.push(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
    }
    Some((depth, cost))
}

/// 解析 mesh.obj（`v x y z [r g b]` / `f a b c`），返回 (顶点, 三角形)。
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

/// 应用逐相机白平衡/曝光增益。
#[inline]
fn apply_gain(rgb: [u8; 3], gain: [f64; 3]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = ((rgb[c] as f64) * gain[c]).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// 一张 RGB 图的逐通道平均色（确定性下采样统计，最多约 1M 像素）。
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

/// 中位数（会排序传入切片）。
fn median(vals: &mut Vec<f64>) -> f64 {
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) * 0.5
    }
}

/// 正交光栅化 DSM 三角形到高度缓冲（取最大 z，即顶面优先）。
fn rasterize_height(
    verts: &[Vector3<f64>],
    tris: &[[usize; 3]],
    x0: f64,
    y0: f64,
    gsd: f64,
    w: usize,
    h: usize,
    hgt: &mut [f32],
) {
    for t in tris {
        let a = verts[t[0]];
        let b = verts[t[1]];
        let c = verts[t[2]];
        let px = |v: &Vector3<f64>| (v.x - x0) / gsd;
        let py = |v: &Vector3<f64>| (y0 - v.y) / gsd;
        let (ax, ay) = (px(&a), py(&a));
        let (bx, by) = (px(&b), py(&b));
        let (cx, cy) = (px(&c), py(&c));
        let i0 = ax.min(bx).min(cx).floor().max(0.0) as usize;
        let i1 = ax.max(bx).max(cx).ceil().min(w as f64 - 1.0) as usize;
        let j0 = ay.min(by).min(cy).floor().max(0.0) as usize;
        let j1 = ay.max(by).max(cy).ceil().min(h as f64 - 1.0) as usize;
        let denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if denom.abs() < 1e-12 {
            continue;
        }
        for j in j0..=j1 {
            let wy = y0 - (j as f64 + 0.5) * gsd;
            for i in i0..=i1 {
                let wx = x0 + (i as f64 + 0.5) * gsd;
                let w0 = ((b.y - c.y) * (wx - c.x) + (c.x - b.x) * (wy - c.y)) / denom;
                let w1 = ((c.y - a.y) * (wx - c.x) + (a.x - c.x) * (wy - c.y)) / denom;
                let w2 = 1.0 - w0 - w1;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let z = (w0 * a.z + w1 * b.z + w2 * c.z) as f32;
                    let k = j * w + i;
                    if z > hgt[k] {
                        hgt[k] = z;
                    }
                }
            }
        }
    }
}

/// 相机在地面（ortho 像素系）的保守足迹：4 个像角在场景深度范围 [dmin,dmax] 的反投影包围盒。
fn footprint(
    cam: &Cam,
    corners: &[Vector3<f64>],
    x0: f64,
    y0: f64,
    gsd: f64,
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize)> {
    let mut dmin = f64::INFINITY;
    let mut dmax = f64::NEG_INFINITY;
    for p in corners {
        let d = (cam.r * p + cam.t).z;
        if d < dmin {
            dmin = d;
        }
        if d > dmax {
            dmax = d;
        }
    }
    if dmax <= 0.0 {
        return None;
    }
    if dmin <= 0.0 {
        dmin = 1e-3;
    }
    let cw = cam.w as f64;
    let ch = cam.h as f64;
    let img_corners = [(0.0, 0.0), (cw - 1.0, 0.0), (cw - 1.0, ch - 1.0), (0.0, ch - 1.0)];
    let mut ix0 = i64::MAX;
    let mut iy0 = i64::MAX;
    let mut ix1 = i64::MIN;
    let mut iy1 = i64::MIN;
    for &(u, v) in &img_corners {
        for &d in &[dmin, dmax] {
            let xc = d * (u - cam.cx) / cam.f;
            let yc = d * (v - cam.cy) / cam.f;
            let p = cam.r.transpose() * Vector3::new(xc, yc, d) + cam.center;
            let px = ((p.x - x0) / gsd).floor() as i64;
            let py = ((y0 - p.y) / gsd).floor() as i64;
            ix0 = ix0.min(px);
            ix1 = ix1.max(px);
            iy0 = iy0.min(py);
            iy1 = iy1.max(py);
        }
    }
    let x0i = (ix0 - 2).max(0) as usize;
    let y0i = (iy0 - 2).max(0) as usize;
    let x1i = ((ix1 + 2) as usize).min(w - 1);
    let y1i = ((iy1 + 2) as usize).min(h - 1);
    if x0i > x1i || y0i > y1i {
        return None;
    }
    Some((x0i, y0i, x1i, y1i))
}

/// 手写最小无压缩基线 TIFF + GeoTIFF 地理参考标签（ModelPixelScale / ModelTiepoint）。
/// 布局：8 字节头 + IFD（升序 tag）+ 额外数据区 + 像素数据（单 strip，行从上到下）。
fn write_geotiff(path: &str, w: u32, h: u32, gsd: f64, x0: f64, y0: f64, rgb: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    const RATIONAL: u16 = 5;
    const DOUBLE: u16 = 12;

    let strip_bytes = (w as u64 * h as u64 * 3) as u32;

    // 条目：(tag, type, count, value4)。需要额外存储的先用占位 offset，稍后回填。
    let mut ent: Vec<(u16, u16, u32, [u8; 4])> = Vec::new();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut blob_at: Vec<usize> = Vec::new();

    // 内联值条目。
    ent.push((256, LONG, 1, w.to_le_bytes())); // ImageWidth
    ent.push((257, LONG, 1, h.to_le_bytes())); // ImageLength
    ent.push((259, SHORT, 1, 1u32.to_le_bytes())); // Compression = none
    ent.push((262, SHORT, 1, 2u32.to_le_bytes())); // PhotometricInterpretation = RGB
    let strip_off_entry = ent.len();
    ent.push((273, LONG, 1, 0u32.to_le_bytes())); // StripOffsets（回填）
    ent.push((277, SHORT, 1, 3u32.to_le_bytes())); // SamplesPerPixel
    ent.push((278, LONG, 1, h.to_le_bytes())); // RowsPerStrip
    ent.push((279, LONG, 1, strip_bytes.to_le_bytes())); // StripByteCounts
    ent.push((284, SHORT, 1, 1u32.to_le_bytes())); // PlanarConfiguration = chunky
    ent.push((296, SHORT, 1, 1u32.to_le_bytes())); // ResolutionUnit = none

    // BitsPerSample = [8,8,8]（3×SHORT，放额外区）。
    let mut b = vec![0u8; 6];
    b[0..2].copy_from_slice(&8u16.to_le_bytes());
    b[2..4].copy_from_slice(&8u16.to_le_bytes());
    b[4..6].copy_from_slice(&8u16.to_le_bytes());
    blobs.push(b);
    blob_at.push(ent.len());
    ent.push((258, SHORT, 3, 0u32.to_le_bytes()));

    // XResolution = 1/1。
    let mut xr = Vec::with_capacity(8);
    xr.extend_from_slice(&1u32.to_le_bytes());
    xr.extend_from_slice(&1u32.to_le_bytes());
    blobs.push(xr);
    blob_at.push(ent.len());
    ent.push((282, RATIONAL, 1, 0u32.to_le_bytes()));

    // YResolution = 1/1。
    let mut yr = Vec::with_capacity(8);
    yr.extend_from_slice(&1u32.to_le_bytes());
    yr.extend_from_slice(&1u32.to_le_bytes());
    blobs.push(yr);
    blob_at.push(ent.len());
    ent.push((283, RATIONAL, 1, 0u32.to_le_bytes()));

    // ModelPixelScaleTag = [gsd, gsd, 0]（3×DOUBLE）。
    let mut ps = Vec::with_capacity(24);
    ps.extend_from_slice(&gsd.to_le_bytes());
    ps.extend_from_slice(&gsd.to_le_bytes());
    ps.extend_from_slice(&0.0f64.to_le_bytes());
    blobs.push(ps);
    blob_at.push(ent.len());
    ent.push((33550, DOUBLE, 3, 0u32.to_le_bytes()));

    // ModelTiepointTag = [0,0,0, x0, y0, 0]（6×DOUBLE）。
    let mut tp = Vec::with_capacity(48);
    for v in [0.0f64, 0.0, 0.0, x0, y0, 0.0] {
        tp.extend_from_slice(&v.to_le_bytes());
    }
    blobs.push(tp);
    blob_at.push(ent.len());
    ent.push((33922, DOUBLE, 6, 0u32.to_le_bytes()));

    // 布局：算 IFD 大小 → 依次分配额外区偏移 → 回填 strip 偏移。
    let n = ent.len() as u32;
    let ifd_off: u32 = 8;
    let ifd_size = 2 + n * 12 + 4;
    let mut data_off = ifd_off + ifd_size;
    for k in 0..blobs.len() {
        ent[blob_at[k]].3 = data_off.to_le_bytes();
        data_off += blobs[k].len() as u32;
    }
    let strip_off = data_off;
    ent[strip_off_entry].3 = strip_off.to_le_bytes();

    // tag 升序（规范友好）。
    ent.sort_by_key(|e| e.0);

    let mut out = Vec::with_capacity(strip_off as usize + rgb.len());
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&ifd_off.to_le_bytes());
    out.extend_from_slice(&(n as u16).to_le_bytes());
    for (tag, typ, count, val) in &ent {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&typ.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(val);
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0
    for b in &blobs {
        out.extend_from_slice(b);
    }
    out.extend_from_slice(rgb);

    let mut f = fs::File::create(path)?;
    f.write_all(&out)?;
    Ok(())
}

/// 正射影像独立入口：读 mesh.obj + camera_poses.json + 原图 + 深度图，写 `ortho.tif`。
pub fn run_ortho(paths: &[String]) {
    let Some((verts, tris)) = load_mesh_obj("mesh.obj") else {
        eprintln!("  [ortho] mesh.obj 不存在或为空，请先跑 -stage=mesh 生成网格");
        return;
    };
    let cams = match load_cams(paths) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  [ortho] 无法加载相机: {e}，跳过正射影像");
            return;
        }
    };
    generate_ortho(&cams, &verts, &tris);
}

fn generate_ortho(cams: &[Cam], verts: &[Vector3<f64>], tris: &[[usize; 3]]) {
    let nc = cams.len();
    if verts.is_empty() || tris.is_empty() {
        println!("  [ortho] 空网格，跳过");
        return;
    }

    // 包围盒 + 场景中心。
    let (mut xmin, mut xmax, mut ymin, mut ymax, mut zmin, mut zmax) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let mut center = Vector3::zeros();
    for v in verts {
        xmin = xmin.min(v.x);
        xmax = xmax.max(v.x);
        ymin = ymin.min(v.y);
        ymax = ymax.max(v.y);
        zmin = zmin.min(v.z);
        zmax = zmax.max(v.z);
        center += v;
    }
    center /= verts.len() as f64;

    // GSD：逐相机在场景中心处的像素尺度 d/f，取中位数（≈ 原始图像分辨率）。
    let mut gsds: Vec<f64> = Vec::with_capacity(nc);
    for c in cams {
        let d = (c.r * center + c.t).z;
        if d > 0.0 {
            gsds.push(d / c.f);
        }
    }
    let mut gsd = median(&mut gsds);
    if !(gsd > 0.0) || gsd.is_nan() {
        gsd = 1.0;
    }

    let mut w = ((xmax - xmin) / gsd).ceil().max(1.0) as u64;
    let mut h = ((ymax - ymin) / gsd).ceil().max(1.0) as u64;
    if w.saturating_mul(h) > MAX_ORTHO_PX {
        let scale = (w as f64 * h as f64 / MAX_ORTHO_PX as f64).sqrt();
        gsd *= scale;
        w = ((xmax - xmin) / gsd).ceil().max(1.0) as u64;
        h = ((ymax - ymin) / gsd).ceil().max(1.0) as u64;
        println!("  [ortho] 输出过大，GSD 放宽到 {:.4} m/px 以限制内存", gsd);
    }
    let (w, h) = (w as usize, h as usize);
    let x0 = xmin;
    let y0 = ymax;

    println!(
        "\n===== Orthophoto: {}x{} px @ {:.4} m/px, {} cameras =====",
        w, h, gsd, nc
    );

    // 高度场：正交光栅化 DSM。
    let mut hgt = vec![f32::NEG_INFINITY; w * h];
    rasterize_height(verts, tris, x0, y0, gsd, w, h, &mut hgt);
    let covered = hgt.iter().filter(|&&z| z != f32::NEG_INFINITY).count();
    println!("  height field: {} / {} px covered", covered, w * h);

    // 逐相机增益：平均色 → 全局平均 → 增益（与 texture.rs 同款，统一曝光/白平衡）。
    let mut gain: Vec<[f64; 3]> = vec![[1.0; 3]; nc];
    {
        let mut cam_mean: Vec<[f64; 3]> = vec![[0.0; 3]; nc];
        let mut used = vec![false; nc];
        for ci in 0..nc {
            if let Ok(im) = image::open(&cams[ci].path) {
                let im = im
                    .resize_exact(cams[ci].w, cams[ci].h, FilterType::Lanczos3)
                    .to_rgb8();
                cam_mean[ci] = mean_color(&im);
                used[ci] = true;
            }
        }
        let mut g = [0.0f64; 3];
        let mut cnt = 0usize;
        for ci in 0..nc {
            if used[ci] {
                g[0] += cam_mean[ci][0];
                g[1] += cam_mean[ci][1];
                g[2] += cam_mean[ci][2];
                cnt += 1;
            }
        }
        if cnt > 0 {
            for c in 0..3 {
                g[c] /= cnt as f64;
            }
            for ci in 0..nc {
                if used[ci] {
                    for c in 0..3 {
                        gain[ci][c] = (g[c] / cam_mean[ci][c].max(1.0)).clamp(GAIN_MIN, GAIN_MAX);
                    }
                }
            }
        }
        println!(
            "  color gain: {} cams, global mean ({:.0},{:.0},{:.0})",
            cnt, g[0], g[1], g[2]
        );
    }

    // 场景包围盒 8 角（用于相机足迹）。
    let mut corners: Vec<Vector3<f64>> = Vec::with_capacity(8);
    for &x in &[xmin, xmax] {
        for &y in &[ymin, ymax] {
            for &z in &[zmin, zmax] {
                corners.push(Vector3::new(x, y, z));
            }
        }
    }

    // 累积缓冲：r,g,b,w（f32）。
    let mut acc = vec![0.0f32; w * h * 4];
    let (mut c_no_footprint, mut c_no_depth, mut c_no_img) = (0usize, 0usize, 0usize);
    let mut c_accumulated = 0usize;
    for ci in 0..nc {
        let cam = &cams[ci];
        let Some((bx0, by0, bx1, by1)) = footprint(cam, &corners, x0, y0, gsd, w, h) else {
            c_no_footprint += 1;
            continue;
        };
        let n = (cam.w as usize) * (cam.h as usize);
        let Some((depth, cost)) = read_depth_cost(ci, n) else {
            c_no_depth += 1;
            continue;
        };
        let img = match image::open(&cam.path) {
            Ok(im) => im.resize_exact(cam.w, cam.h, FilterType::Lanczos3).to_rgb8(),
            Err(_) => {
                c_no_img += 1;
                continue;
            }
        };
        let g = gain[ci];
        let half_w = cam.w as f64 * 0.5;
        let half_h = cam.h as f64 * 0.5;
        let mut wrote_any = false;
        for j in by0..=by1 {
            let wy = y0 - (j as f64 + 0.5) * gsd;
            let row = j * w;
            for i in bx0..=bx1 {
                let idx = row + i;
                let z = hgt[idx];
                if z == f32::NEG_INFINITY {
                    continue;
                }
                let p = Vector3::new(x0 + (i as f64 + 0.5) * gsd, wy, z as f64);
                let (u, v, d) = project(cam, &p);
                if d <= 0.0
                    || u < FRUSTUM_BORDER
                    || v < FRUSTUM_BORDER
                    || u > cam.w as f64 - 1.0 - FRUSTUM_BORDER
                    || v > cam.h as f64 - 1.0 - FRUSTUM_BORDER
                {
                    continue;
                }
                // 遮挡：深度显著大于深度图记录值 → 该相机看不到此点。
                let xi = u.round() as i64;
                let yi = v.round() as i64;
                if xi >= 0 && yi >= 0 && xi < cam.w as i64 && yi < cam.h as i64 {
                    let di = yi as usize * cam.w as usize + xi as usize;
                    let dm = depth[di];
                    if dm > 0.0 && cost[di] >= COST_MIN && d > dm as f64 * (1.0 + OCC_TOL_REL) + OCC_TOL_ABS {
                        continue;
                    }
                }
                let dist = (cam.center - p).norm();
                if dist <= 1e-9 {
                    continue;
                }
                // 入射角余弦（相对 +z 即"向上"）。z 轴翻转的重建里相机在场景下方、
                // cos 全为负，硬剔除会把所有像素判死（与 texture.rs 同款教训）。权重取
                // cos²（符号无关），只按正对程度加权，不做符号剔除。
                let cos_inc = (cam.center.z - p.z) / dist;
                // 权重：正对度² × 图像中心径向衰减²（相邻相机重叠处自然羽化）。
                let dx = (u - cam.cx) / half_w;
                let dy = (v - cam.cy) / half_h;
                let r2 = dx * dx + dy * dy;
                let falloff = (1.0 - r2.min(1.0)).max(0.0);
                let wgt = (cos_inc * cos_inc) * (falloff * falloff);
                if wgt <= 0.0 {
                    continue;
                }
                let rgb = apply_gain(sample_rgb(&img, u, v), g);
                for c in 0..3 {
                    acc[idx * 4 + c] += wgt as f32 * rgb[c] as f32;
                }
                acc[idx * 4 + 3] += wgt as f32;
                wrote_any = true;
            }
        }
        if wrote_any {
            c_accumulated += 1;
        }
    }
    println!(
        "  accumulate: {} cams wrote pixels (skipped {} no-footprint / {} no-depth / {} no-img)",
        c_accumulated, c_no_footprint, c_no_depth, c_no_img
    );

    // 归一化合成 RGB。
    let mut rgb = vec![0u8; w * h * 3];
    for p in 0..w * h {
        let wgt = acc[p * 4 + 3];
        if wgt > 0.0 {
            for c in 0..3 {
                let v = (acc[p * 4 + c] / wgt).round().clamp(0.0, 255.0);
                rgb[p * 3 + c] = v as u8;
            }
        }
    }
    let colored = rgb
        .chunks_exact(3)
        .filter(|px| px[0] > 0 || px[1] > 0 || px[2] > 0)
        .count();
    println!("  colored: {} / {} px", colored, w * h);

    let tif = format!("{}.tif", OUT_PREFIX);
    match write_geotiff(&tif, w as u32, h as u32, gsd, x0, y0, &rgb) {
        Ok(()) => println!(
            "  wrote {} ({:.1} MB, extent {:.1}x{:.1} m)",
            tif,
            rgb.len() as f64 / 1e6,
            xmax - xmin,
            ymax - ymin
        ),
        Err(e) => eprintln!("  [ortho] 写 GeoTIFF 失败: {e}"),
    }

    let png = format!("{}_preview.png", OUT_PREFIX);
    if let Some(img) = RgbImage::from_raw(w as u32, h as u32, rgb) {
        if let Ok(mut f) = fs::File::create(&png) {
            let _ = DynamicImage::ImageRgb8(img).write_to(&mut f, ImageFormat::Png);
        }
    }
}
