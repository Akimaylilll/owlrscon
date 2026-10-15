//! 网格重建：从稠密点云（dense.ply）用 2.5D DSM 高度场重建表面。
//!
//! 与 MVS 完全解耦：本模块只读取点云坐标，直接在 (x,y) 网格上重建表面高度
//! （每格取中位数高度，稳健抗噪声），三角化后着色写 OBJ。对航拍地形这种
//! "基本 2.5D、几乎无墙面点"的数据，这是唯一能稳定产出单一连通网格的方法。

use std::collections::HashMap;
use std::fs;

use nalgebra::Vector3;

/// DSM（高度场）重建：2D 网格长边最大节点数（决定 cell 大小，越大越精细）。
/// 只重建 x/y 平面网格，内存是 O(nx·ny)，远小于体素法的 O(n³)，可放心取较大值。
/// 384 对应 cell≈0.04（dense.ply 中值约 5 点/格），比 256 翻一倍三角面又不至于使
/// 每格点数过低（再细到 512 会重新冒出离群针尖）。
const DSM_MAX_SIDE: usize = 384;

/// DSM 足迹膨胀轮数：把可靠格点向外扩几格，闭合小孔洞、给边界留一圈余量。
/// 取 2：太小闭合不了屋顶孔洞，太大又会把稀疏离群簇放大成假柱子（"石林"状针尖）。
const DSM_FILL_RADIUS: usize = 2;

/// DSM 高度扩散迭代上限（实际上几轮即收敛，仅作安全阀）。
const DSM_FILL_ITERS: usize = 64;

/// 每格最少点数：低于此值的格点视为不可靠（稀疏离群簇），按空洞处理、由邻域扩散填平。
/// dense.ply 中值约 9 点/格，3 点能干净地剔除 1~2 点的孤立离群簇，又不误删真实表面。
const DSM_MIN_PTS: u32 = 3;

/// DSM 高度场中值滤波核大小（k×k）与遍数：压掉逐格 ±0.15 的深度噪声尖峰，
/// 同时保留比核宽得多的屋顶平台与地形。
/// 3×3×1 实测即可把 384 网格的长边针尖全部压到 0，而 5×5×2 会过度平滑、
/// 抹掉屋顶/墙沿细节（用户反馈"精度太差"），故收敛到最轻的 3×3×1。
const DSM_MED_K: usize = 3;
const DSM_MED_PASSES: usize = 1;

/// 中值滤波的边保护容差：邻点与中心高度差超过该值即视为跨越陡边（墙/台阶），
/// 不纳入该格的中值统计——中值滤波因此只压平面噪声、不抹平竖直墙。
/// 取 0.5：与 [`DSM_STRUCT_THRESH`] 一致；深度噪声 ±0.15 远小于此、真实墙高差远大于此。
const DSM_MED_EDGE_TOL: f64 = 0.5;

/// 结构清理：高度超出局部地面该阈值即视为"抬升结构"。
const DSM_STRUCT_THRESH: f64 = 0.5;

/// 结构清理：抬升连通块背后支撑点少于该值即判定为离群针尖，整体压掉。
/// 真实屋顶有数百~数千点；5~39 点的孤立柱是 MVS 误匹配簇。
const DSM_MIN_STRUCT_PTS: u32 = 40;

/// 稀疏尖峰压平：支撑点不足（< DSM_MIN_PTS）的格点，若比其 8 邻域最低值高出该
/// 高度差，即判定为"针状柱体"格点并压掉（置 NEG_INFINITY，随后由平缓空洞填补
/// 回地面）。这些格点本身几乎无真实点支撑，是中值滤波把屋顶/地形边缘扩散到
/// 空腔上造出的假尖。实测把长边 >1.0 的针面从 52 条压到 0 条。
const DSM_SPIKE_DROP: f64 = 1.0;

/// 平缓空洞填补：内部空腔（不与网格边界相连）的边界高度跨度低于该值才填，
/// 跨度大的崖面缺口（填了反而造出陡斜面）保持为洞。1.1 能填平缓坡缺口
/// （如 268 格的缓坡、跨度 ~1.0）又避开真正的崖面（跨度 ~1.18+）。
const DSM_HOLE_RANGE: f64 = 1.1;

/// DSM（高度场）重建参数：把点云当作 2.5D 地形表面，按 (x,y) 网格取每格中位数高度
/// （稳健，抗逐格噪声尖峰），剔除稀疏格点后直接三角化成单一连通表面。航拍地形的房屋
/// 屋顶会沿竖直网格边（陡崖）与地面相连，天然消除"屋顶悬浮、模型分裂成多块"的问题。
pub struct DsmOptions {
    /// 2D 网格长边最大节点数（决定 cell 大小，越大越精细）。
    pub max_side: usize,
    /// 足迹膨胀轮数：闭合点云覆盖中的小孔洞并给边界留一圈余量。
    pub fill_radius: usize,
}

impl Default for DsmOptions {
    fn default() -> Self {
        Self {
            max_side: DSM_MAX_SIDE,
            fill_radius: DSM_FILL_RADIUS,
        }
    }
}

/// 给网格顶点着色：对每个顶点在带色点云里找最近邻，取其颜色。
/// 空间哈希加速最近邻（网格顶点都在表面附近，通常一两圈内即可命中）。
fn colorize_vertices(
    verts: &[Vector3<f64>],
    pts: &[(f64, f64, f64)],
    cols: &[(u8, u8, u8)],
) -> Vec<(u8, u8, u8)> {
    let n = pts.len();
    let (mut lo, mut hi) = (
        Vector3::new(f64::MAX, f64::MAX, f64::MAX),
        Vector3::new(f64::MIN, f64::MIN, f64::MIN),
    );
    for &(x, y, z) in pts {
        lo = lo.inf(&Vector3::new(x, y, z));
        hi = hi.sup(&Vector3::new(x, y, z));
    }
    let ext = hi - lo;
    let cell = ((ext.x * ext.y * ext.z).max(1e-12) / n as f64).max(1e-12).cbrt();
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, &(x, y, z)) in pts.iter().enumerate() {
        grid.entry((
            (x / cell).floor() as i64,
            (y / cell).floor() as i64,
            (z / cell).floor() as i64,
        ))
        .or_default()
        .push(i);
    }
    verts
        .iter()
        .map(|v| {
            let (cx, cy, cz) = (
                (v.x / cell).floor() as i64,
                (v.y / cell).floor() as i64,
                (v.z / cell).floor() as i64,
            );
            let mut best = f64::MAX;
            let mut bi = usize::MAX;
            for rad in 0i64..=32 {
                let mut found = false;
                for dz in -rad..=rad {
                    for dy in -rad..=rad {
                        for dx in -rad..=rad {
                            if dx.abs() < rad && dy.abs() < rad && dz.abs() < rad {
                                continue;
                            }
                            if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &j in ids {
                                    let d = (v.x - pts[j].0).powi(2)
                                        + (v.y - pts[j].1).powi(2)
                                        + (v.z - pts[j].2).powi(2);
                                    if d < best {
                                        best = d;
                                        bi = j;
                                        found = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if found {
                    break;
                }
            }
            if bi == usize::MAX {
                (128u8, 128u8, 128u8)
            } else {
                cols[bi]
            }
        })
        .collect()
}

/// 写 OBJ。若提供 `colors`（与顶点一一对应），写入 `v x y z r g b` 顶点色（r/g/b∈[0,1]）。
fn write_mesh_obj(
    verts: &[Vector3<f64>],
    tris: &[[usize; 3]],
    colors: Option<&[(u8, u8, u8)]>,
    path: &str,
) {
    let mut s = String::with_capacity(verts.len() * 40 + tris.len() * 24);
    s.push_str("# DSM height-field mesh\n");
    if let Some(cols) = colors {
        debug_assert_eq!(cols.len(), verts.len());
        for (v, c) in verts.iter().zip(cols.iter()) {
            s.push_str(&format!(
                "v {:.6} {:.6} {:.6} {:.4} {:.4} {:.4}\n",
                v.x,
                v.y,
                v.z,
                c.0 as f64 / 255.0,
                c.1 as f64 / 255.0,
                c.2 as f64 / 255.0
            ));
        }
    } else {
        for v in verts {
            s.push_str(&format!("v {:.6} {:.6} {:.6}\n", v.x, v.y, v.z));
        }
    }
    for t in tris {
        s.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
    }
    fs::write(path, &s).expect("Failed to write mesh");
}

/// 只保留最大连通分量，丢弃游离的小碎片（重建尾部总有一些边界孤立片）。
fn keep_largest_component(
    verts: Vec<Vector3<f64>>,
    tris: Vec<[usize; 3]>,
) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    if tris.is_empty() {
        return (verts, tris);
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
    for t in &tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    let mut seen = vec![false; verts.len()];
    let mut best: Vec<usize> = Vec::new();
    for s in 0..verts.len() {
        if seen[s] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![s];
        seen[s] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &w in &adj[u] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    let mut remap: Vec<Option<usize>> = vec![None; verts.len()];
    let mut out_v: Vec<Vector3<f64>> = Vec::with_capacity(best.len());
    for &v in &best {
        remap[v] = Some(out_v.len());
        out_v.push(verts[v]);
    }
    let mut out_f: Vec<[usize; 3]> = Vec::new();
    for t in &tris {
        if let (Some(a), Some(b), Some(c)) = (remap[t[0]], remap[t[1]], remap[t[2]]) {
            out_f.push([a, b, c]);
        }
    }
    (out_v, out_f)
}

/// 高度扩散填洞：把足迹内仍为空（NEG_INFINITY）的格点填成其 8 邻域有限值的平均。
/// 就地 Gauss-Seidel，收敛即停（安全阀 DSM_FILL_ITERS 轮）。
fn dsm_fill(hgt: &mut [f64], foot: &[bool], nx: usize, ny: usize) {
    let idx = |i: usize, j: usize| i * (ny + 1) + j;
    for _ in 0..DSM_FILL_ITERS {
        let mut changed = false;
        for i in 0..=nx {
            for j in 0..=ny {
                let k = idx(i, j);
                if !foot[k] || hgt[k] != f64::NEG_INFINITY {
                    continue;
                }
                let mut s = 0.0;
                let mut c = 0u32;
                for di in -1isize..=1 {
                    for dj in -1isize..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        if ni < 0 || nj < 0 || ni > nx as isize || nj > ny as isize {
                            continue;
                        }
                        let nk = idx(ni as usize, nj as usize);
                        if hgt[nk] != f64::NEG_INFINITY {
                            s += hgt[nk];
                            c += 1;
                        }
                    }
                }
                if c > 0 {
                    hgt[k] = s / c as f64;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// 高度场 k×k 中值滤波（多遍）：压掉逐格噪声尖峰，但**跨陡边不混值**——只把与中心
/// 高度差 ≤ [`DSM_MED_EDGE_TOL`] 的邻点纳入统计，屋顶/地面不互相拉低/抬高，竖直墙
/// 得以保留。空（NEG_INFINITY）格点既不参与统计、也不会被填充。逐遍读取快照、写回，
/// 保证确定性。
fn dsm_median_filter(hgt: &mut [f64], nx: usize, ny: usize, k: usize, passes: usize) {
    let idx = |i: usize, j: usize| i * (ny + 1) + j;
    let r = (k / 2) as isize;
    let mut buf: Vec<f64> = Vec::with_capacity(k * k);
    for _ in 0..passes {
        let src = hgt.to_vec();
        for i in 0..=nx {
            for j in 0..=ny {
                let kk = idx(i, j);
                let c = src[kk];
                if c == f64::NEG_INFINITY {
                    continue;
                }
                // 中心始终纳入：陡边上的孤立格点没有同侧邻点，若被邻居中值改写会
                // 反过来抹掉墙顶/墙脚。取 [中心 + 同侧邻点] 的中值 → 平处平滑、陡边保边。
                buf.clear();
                buf.push(c);
                for di in -r..=r {
                    for dj in -r..=r {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        if ni < 0 || nj < 0 || ni > nx as isize || nj > ny as isize {
                            continue;
                        }
                        let nk = idx(ni as usize, nj as usize);
                        let v = src[nk];
                        if v != f64::NEG_INFINITY && (v - c).abs() <= DSM_MED_EDGE_TOL {
                            buf.push(v);
                        }
                    }
                }
                buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
                hgt[kk] = buf[buf.len() / 2];
            }
        }
    }
}

/// 高度场（DSM）重建：把点云当作 2.5D 地形表面重建。
///
/// 不依赖法向、不做隐式场，而是直接在 (x,y) 网格上重建表面高度。对航拍地形这种
/// "基本 2.5D、几乎无墙面点"的数据，这是能稳定产出**单一连通**网格的方法：房屋屋顶
/// 是抬高的平台，墙面由相邻格点间的竖直边天然补齐，屋顶因此与地面连成一体。
///
/// 步骤：包围盒 → 每格取**中位数**高度（而非 max-z，稳健抗逐格离群）→ 剔除稀疏格点
/// （点数 < DSM_MIN_PTS）→ 足迹膨胀闭合小孔 → 高度扩散填洞 → k×k 边保护中值滤波压
/// 噪声尖峰（跨陡边不混值、不抹竖直墙）→ 结构清理（压掉支撑点过少的孤立抬升柱，
/// 消除"石林/针状面"）→ 稀疏尖峰压平
/// （压掉支撑不足却高出 8 邻域的针状柱体格点）→ 平缓空洞填补（填平内部边界高差小的
/// 道路/缓坡缺口，崖面缺口保持为洞）→ 网格三角化 → 最大连通分量 → 顶点着色 →
/// 写 OBJ。全程确定性（固定遍历顺序）。
pub fn build_mesh_dsm(
    points: &[(f64, f64, f64)],
    colors: Option<&[(u8, u8, u8)]>,
    opts: &DsmOptions,
    out_path: &str,
) {
    let n = points.len();
    if n < 4 {
        println!("  点云过小（{} 点），跳过 DSM 重建", n);
        return;
    }
    println!("\n===== Building DSM height-field mesh from {} points =====", n);

    // ---- 1) x/y 包围盒 + 2D 网格 ----
    let (mut xmin, mut xmax, mut ymin, mut ymax) =
        (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for &(x, y, _) in points {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    let extx = xmax - xmin;
    let exty = ymax - ymin;
    let max_ext = extx.max(exty);
    if max_ext <= 1e-9 {
        println!("  点云 x/y 包围盒退化，跳过");
        return;
    }
    let cell = max_ext / opts.max_side as f64;
    let nx = ((extx / cell).ceil() as usize).max(2);
    let ny = ((exty / cell).ceil() as usize).max(2);
    // 顶点网格 (nx+1)×(ny+1)；体素 (i,j) 覆盖 [i,i+1]×[j,j+1]。
    let idx = |i: usize, j: usize| i * (ny + 1) + j;
    let total = (nx + 1) * (ny + 1);
    println!("  bbox x[{:.2},{:.2}] y[{:.2},{:.2}], cell={:.4}, grid {}x{}",
        xmin, xmax, ymin, ymax, cell, nx, ny);

    // ---- 2) 每格收集 z 求中位数 + 点数 ----
    let mut zs: Vec<Vec<f64>> = vec![Vec::new(); total];
    let mut cnt = vec![0u32; total];
    for &(x, y, z) in points {
        let i = ((x - xmin) / cell).floor() as isize;
        let j = ((y - ymin) / cell).floor() as isize;
        if i < 0 || j < 0 {
            continue;
        }
        let i = (i as usize).min(nx);
        let j = (j as usize).min(ny);
        let k = idx(i, j);
        zs[k].push(z);
        cnt[k] += 1;
    }
    // 每格中位数高度（稳健）；点数不足的格视为不可靠 → 留空，随后由邻域扩散填平。
    let mut hgt = vec![f64::NEG_INFINITY; total];
    for k in 0..total {
        if cnt[k] >= DSM_MIN_PTS {
            let zv = &mut zs[k];
            zv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            hgt[k] = zv[zv.len() / 2];
        }
    }
    drop(zs);
    // 足迹 = 可靠格点（点数达阈值的格）。
    let mut foot: Vec<bool> = cnt.iter().map(|&c| c >= DSM_MIN_PTS).collect();

    // ---- 3) 足迹膨胀（闭合小孔洞 + 边界留边） ----
    for _ in 0..opts.fill_radius {
        let mut nf = foot.clone();
        for i in 0..=nx {
            for j in 0..=ny {
                let k = idx(i, j);
                if nf[k] {
                    continue;
                }
                let has = (i > 0 && foot[idx(i - 1, j)])
                    || (i < nx && foot[idx(i + 1, j)])
                    || (j > 0 && foot[idx(i, j - 1)])
                    || (j < ny && foot[idx(i, j + 1)]);
                if has {
                    nf[k] = true;
                }
            }
        }
        foot = nf;
    }

    // ---- 4) 高度扩散填洞 ----
    dsm_fill(&mut hgt, &foot, nx, ny);

    // ---- 5) k×k 边保护中值滤波：压掉逐格噪声尖峰，跨陡边不混值（不抹竖直墙） ----
    dsm_median_filter(&mut hgt, nx, ny, DSM_MED_K, DSM_MED_PASSES);

    // ---- 6) 结构清理：压掉支撑点过少的孤立抬升柱（"石林/针状面"根因） ----
    //   局部地面 = 有限高度的 30% 分位；高出 DSM_STRUCT_THRESH 即视为抬升结构。
    let mut vals: Vec<f64> = hgt
        .iter()
        .filter(|&&v| v != f64::NEG_INFINITY)
        .cloned()
        .collect();
    if !vals.is_empty() {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let g0 = vals[((vals.len() as f64) * 0.3) as usize];
        let thresh = g0 + DSM_STRUCT_THRESH;
        // 抬升格点 4 邻域连通标号，累计每个块背后的原始支撑点数。
        let mut label = vec![0usize; total];
        let mut comp_pts: Vec<u32> = Vec::new();
        let mut cid = 0usize;
        for i in 0..=nx {
            for j in 0..=ny {
                let k = idx(i, j);
                if !foot[k] || hgt[k] == f64::NEG_INFINITY || hgt[k] <= thresh || label[k] != 0 {
                    continue;
                }
                cid += 1;
                let mut pts = 0u32;
                let mut stack = vec![k];
                label[k] = cid;
                while let Some(u) = stack.pop() {
                    pts += cnt[u];
                    let ui = u / (ny + 1);
                    let uj = u % (ny + 1);
                    let mut nbs = [usize::MAX; 4];
                    let mut nn = 0usize;
                    if ui > 0 {
                        nbs[nn] = idx(ui - 1, uj);
                        nn += 1;
                    }
                    if ui < nx {
                        nbs[nn] = idx(ui + 1, uj);
                        nn += 1;
                    }
                    if uj > 0 {
                        nbs[nn] = idx(ui, uj - 1);
                        nn += 1;
                    }
                    if uj < ny {
                        nbs[nn] = idx(ui, uj + 1);
                        nn += 1;
                    }
                    for m in 0..nn {
                        let v = nbs[m];
                        if foot[v]
                            && hgt[v] != f64::NEG_INFINITY
                            && hgt[v] > thresh
                            && label[v] == 0
                        {
                            label[v] = cid;
                            stack.push(v);
                        }
                    }
                }
                comp_pts.push(pts);
            }
        }
        let mut removed = 0usize;
        let removed_blocks = comp_pts.iter().filter(|&&p| p < DSM_MIN_STRUCT_PTS).count();
        for k in 0..total {
            if label[k] != 0 && comp_pts[label[k] - 1] < DSM_MIN_STRUCT_PTS {
                hgt[k] = f64::NEG_INFINITY;
                removed += 1;
            }
        }
        if removed > 0 {
            println!(
                "  DSM 结构清理：压掉 {} 个孤立抬升格点（{} 块，支撑点 < {}）",
                removed, removed_blocks, DSM_MIN_STRUCT_PTS
            );
            // 重新扩散填平被压掉的格点（它们仍在足迹内，由邻域真实地面插值）。
            dsm_fill(&mut hgt, &foot, nx, ny);
        }
    }

    // 最后一遍轻度中值滤波，抹平结构清理后重新扩散填平处的细微边界。
    dsm_median_filter(&mut hgt, nx, ny, 3, 1);

    // ---- 6b) 稀疏尖峰压平：支撑点不足（< DSM_MIN_PTS）却比 8 邻域最低值高出
    //          DSM_SPIKE_DROP 以上的格点 = 针状柱体。这些格点几乎无真实点支撑，
    //          是中值滤波把屋顶/地形边缘扩散到空腔上造出的假尖；压掉后由下一步
    //          平缓空洞填补回地面。快照读、就地写，保证确定性。 ----
    {
        let src = hgt.clone();
        let mut capped = 0usize;
        for i in 0..=nx {
            for j in 0..=ny {
                let k = idx(i, j);
                if src[k] == f64::NEG_INFINITY || cnt[k] >= DSM_MIN_PTS {
                    continue;
                }
                let mut mn = f64::INFINITY;
                for di in -1isize..=1 {
                    for dj in -1isize..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        if ni < 0 || nj < 0 || ni > nx as isize || nj > ny as isize {
                            continue;
                        }
                        let nk = idx(ni as usize, nj as usize);
                        if src[nk] != f64::NEG_INFINITY {
                            mn = mn.min(src[nk]);
                        }
                    }
                }
                if mn != f64::INFINITY && src[k] - mn > DSM_SPIKE_DROP {
                    hgt[k] = f64::NEG_INFINITY;
                    capped += 1;
                }
            }
        }
        if capped > 0 {
            println!("  DSM 稀疏尖峰压平：压掉 {} 个针状柱体格点", capped);
        }
    }

    // ---- 6c) 平缓空洞填补：内部空腔（4 邻域连通、不贴网格边界）中边界高度跨度
    //          低于 DSM_HOLE_RANGE 者填平（道路/缓坡缺口）；跨度大的崖面缺口保持为洞，
    //          避免填出陡斜面。无任何有限邻点的孤立空腔填不动，仍留作洞。 ----
    {
        let mut visited = vec![false; total];
        let mut fill_mask = vec![false; total];
        let mut to_fill = 0usize;
        let mut filled_comp = 0usize;
        let mut skipped = 0usize;
        for i0 in 0..=nx {
            for j0 in 0..=ny {
                let k0 = idx(i0, j0);
                if visited[k0] || hgt[k0] != f64::NEG_INFINITY {
                    continue;
                }
                let mut cells: Vec<usize> = Vec::new();
                let mut stack = vec![k0];
                visited[k0] = true;
                let mut touches_border = false;
                while let Some(k) = stack.pop() {
                    let ci = k / (ny + 1);
                    let cj = k % (ny + 1);
                    cells.push(k);
                    if ci == 0 || ci == nx || cj == 0 || cj == ny {
                        touches_border = true;
                    }
                    let mut nbs = [usize::MAX; 4];
                    let mut nn = 0usize;
                    if ci > 0 {
                        nbs[nn] = idx(ci - 1, cj);
                        nn += 1;
                    }
                    if ci < nx {
                        nbs[nn] = idx(ci + 1, cj);
                        nn += 1;
                    }
                    if cj > 0 {
                        nbs[nn] = idx(ci, cj - 1);
                        nn += 1;
                    }
                    if cj < ny {
                        nbs[nn] = idx(ci, cj + 1);
                        nn += 1;
                    }
                    for m in 0..nn {
                        let v = nbs[m];
                        if !visited[v] && hgt[v] == f64::NEG_INFINITY {
                            visited[v] = true;
                            stack.push(v);
                        }
                    }
                }
                if touches_border {
                    continue;
                }
                // 空腔边界 = 其所有 8 邻域中的有限高度；跨度 = max-min。
                let mut bmin = f64::INFINITY;
                let mut bmax = f64::NEG_INFINITY;
                for &k in &cells {
                    let ci = k / (ny + 1);
                    let cj = k % (ny + 1);
                    for di in -1isize..=1 {
                        for dj in -1isize..=1 {
                            if di == 0 && dj == 0 {
                                continue;
                            }
                            let ni = ci as isize + di;
                            let nj = cj as isize + dj;
                            if ni < 0 || nj < 0 || ni > nx as isize || nj > ny as isize {
                                continue;
                            }
                            let nk = idx(ni as usize, nj as usize);
                            if hgt[nk] != f64::NEG_INFINITY {
                                bmin = bmin.min(hgt[nk]);
                                bmax = bmax.max(hgt[nk]);
                            }
                        }
                    }
                }
                if bmin != f64::INFINITY && (bmax - bmin) < DSM_HOLE_RANGE {
                    for &k in &cells {
                        fill_mask[k] = true;
                        to_fill += 1;
                    }
                    filled_comp += 1;
                } else {
                    skipped += 1;
                }
            }
        }
        if to_fill > 0 {
            dsm_fill(&mut hgt, &fill_mask, nx, ny);
            println!(
                "  DSM 平缓空洞填补：填平 {} 个内部空腔格点（{} 处）",
                to_fill, filled_comp
            );
        }
        if skipped > 0 {
            println!("  DSM 保留 {} 处陡崖/孤立缺口为洞", skipped);
        }
    }

    // ---- 7) 网格三角化 ----
    //   顶点取所有**有限高度**格点（足迹内 + 平缓空洞填补出的足迹外格点），而非只看足迹：
    //   道路/缓坡缺口等空洞本就在足迹之外，只有把它们纳入顶点才能让空洞真正闭合。
    //   仍为 NEG_INFINITY 的格点（边界外、崖面缺口、孤立空腔）不产生顶点 → 保持为洞。
    let mut vid: Vec<Option<usize>> = vec![None; total];
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    for i in 0..=nx {
        for j in 0..=ny {
            let k = idx(i, j);
            if hgt[k] != f64::NEG_INFINITY {
                vid[k] = Some(verts.len());
                verts.push(Vector3::new(
                    xmin + i as f64 * cell,
                    ymin + j as f64 * cell,
                    hgt[k],
                ));
            }
        }
    }
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1));
            if let (Some(va), Some(vb), Some(vc), Some(vd)) = (vid[a], vid[b], vid[c], vid[d]) {
                // 两三角形均逆时针（朝上 +z），光照/朝向一致。
                tris.push([va, vb, vd]);
                tris.push([va, vd, vc]);
            }
        }
    }
    println!("  DSM: {} vertices, {} triangles", verts.len(), tris.len());
    if tris.is_empty() {
        println!("  空网格，跳过");
        return;
    }

    // ---- 8) 最大连通分量 ----
    let (verts, tris) = keep_largest_component(verts, tris);

    // ---- 9) 顶点着色 + 写 OBJ ----
    let vertex_colors: Option<Vec<(u8, u8, u8)>> = colors.and_then(|cols| {
        if cols.len() == n {
            Some(colorize_vertices(&verts, points, cols))
        } else {
            None
        }
    });
    write_mesh_obj(&verts, &tris, vertex_colors.as_deref(), out_path);
    println!(
        "  -> {} ({} vertices, {} faces{})",
        out_path,
        verts.len(),
        tris.len(),
        if vertex_colors.is_some() { ", with vertex colors" } else { "" }
    );
}

/// 读取 ASCII PLY：返回 (点坐标, 法向, 颜色)。法向/颜色为 `None` 表示该 PLY
/// 不含对应属性（旧版 dense.ply 无法向）。
pub fn load_ply_xyz_normal(
    path: &str,
) -> (
    Vec<(f64, f64, f64)>,
    Option<Vec<(f64, f64, f64)>>,
    Option<Vec<(u8, u8, u8)>>,
) {
    let s = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[mesh] 读取 {} 失败: {}", path, e);
            return (Vec::new(), None, None);
        }
    };
    // 解析头：按属性名定位 x/y/z/nx/ny/nz/red/green/blue 的列索引。
    let mut props: Vec<String> = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t == "end_header" {
            break;
        }
        if let Some(rest) = t.strip_prefix("property ") {
            if let Some(name) = rest.split_whitespace().last() {
                props.push(name.to_string());
            }
        }
    }
    let pos = |name: &str| props.iter().position(|p| p.as_str() == name);
    let (Some(ix), Some(iy), Some(iz)) = (pos("x"), pos("y"), pos("z")) else {
        eprintln!("[mesh] {} 头缺少 xyz 属性", path);
        return (Vec::new(), None, None);
    };
    let has_normal = pos("nx").is_some() && pos("ny").is_some() && pos("nz").is_some();
    let (inx, iny, inz) = (pos("nx").unwrap_or(0), pos("ny").unwrap_or(0), pos("nz").unwrap_or(0));
    let has_color = pos("red").is_some() && pos("green").is_some() && pos("blue").is_some();
    let (ir, ig, ib) = (pos("red").unwrap_or(0), pos("green").unwrap_or(0), pos("blue").unwrap_or(0));

    let mut pts = Vec::new();
    let mut nrm = Vec::new();
    let mut col = Vec::new();
    let mut in_header = true;
    let max_idx = ix.max(iy).max(iz);
    for line in s.lines() {
        if in_header {
            if line.trim() == "end_header" {
                in_header = false;
            }
            continue;
        }
        let v: Vec<&str> = line.split_whitespace().collect();
        if v.len() <= max_idx {
            continue;
        }
        let (Ok(x), Ok(y), Ok(z)) = (v[ix].parse(), v[iy].parse(), v[iz].parse()) else {
            continue;
        };
        pts.push((x, y, z));
        if has_normal && v.len() > inx.max(iny).max(inz) {
            let nx: f64 = v[inx].parse().unwrap_or(0.0);
            let ny: f64 = v[iny].parse().unwrap_or(0.0);
            let nz: f64 = v[inz].parse().unwrap_or(0.0);
            nrm.push((nx, ny, nz));
        }
        if has_color && v.len() > ir.max(ig).max(ib) {
            let r: u8 = v[ir].parse().unwrap_or(0);
            let g: u8 = v[ig].parse().unwrap_or(0);
            let b: u8 = v[ib].parse().unwrap_or(0);
            col.push((r, g, b));
        }
    }
    let normals = if has_normal && nrm.len() == pts.len() {
        Some(nrm)
    } else {
        None
    };
    let colors = if has_color && col.len() == pts.len() {
        Some(col)
    } else {
        None
    };
    (pts, normals, colors)
}
