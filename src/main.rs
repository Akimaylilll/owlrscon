mod backend;
mod imgproc;
mod matching;
mod mesh;
mod mvs;
mod ortho;
mod sfm;
mod texture;
mod tracks;
mod types;

use matching::{extract_features, match_all_pairs};
use mvs::run_mvs_dense;
use sfm::run_incremental_sfm;
use tracks::{build_tracks, visualize_track_coverage_graph};
use types::{Paths, RATIO_THRESHOLD};

use std::fs;
use std::time::Instant;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Features,
    Matches,
    Tracks,
    Coverage,
    Sfm,
    Mvs,
    Mesh,
    Texture,
    Ortho,
    All,
}

fn parse_stage(s: &str) -> Stage {
    match s.to_lowercase().as_str() {
        "features" | "sift" => Stage::Features,
        "matches" | "match" => Stage::Matches,
        "tracks" | "track" => Stage::Tracks,
        "coverage" => Stage::Coverage,
        "sfm" => Stage::Sfm,
        "mvs" => Stage::Mvs,
        "mesh" | "obj" => Stage::Mesh,
        "texture" | "tex" => Stage::Texture,
        "ortho" | "orthophoto" => Stage::Ortho,
        "all" => Stage::All,
        other => {
            eprintln!("Unknown stage '{}', defaulting to 'mvs'", other);
            Stage::Mvs
        }
    }
}

/// 文件修改时间（不存在或不可读返回 None）。
fn file_mtime(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// 文件 a 的修改时间是否严格晚于 b（用于「产物比输入新 → 跳过」的增量判断）。
fn is_newer_than(a: &str, b: &str) -> bool {
    match (file_mtime(a), file_mtime(b)) {
        (Some(ta), Some(tb)) => ta > tb,
        _ => false,
    }
}

fn parse_args() -> (String, i32, Stage, String, bool) {
    let mut dir = String::from("./images");
    let mut out = String::from("./out");
    let mut first_octave: i32 = 0;
    let mut stage = Stage::Mvs;
    let mut force = false;

    let args: Vec<String> = std::env::args().collect();
    for arg in args.iter().skip(1) {
        let (key, value) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg.as_str(), None),
        };

        match key {
            "-dir" => match value {
                Some(v) => dir = v.to_string(),
                None => eprintln!("Missing value for -dir"),
            },
            "-out" | "-output" | "-o" => match value {
                Some(v) => out = v.to_string(),
                None => eprintln!("Missing value for -out"),
            },
            "-first_octave" => match value {
                Some(v) => {
                    first_octave = v.parse().unwrap_or_else(|_| {
                        eprintln!("Invalid value for -first_octave: {}", v);
                        0
                    });
                }
                None => eprintln!("Missing value for -first_octave"),
            },
            "-stage" => match value {
                Some(v) => stage = parse_stage(v),
                None => eprintln!("Missing value for -stage"),
            },
            "-force" | "-f" => force = true,
            other => {
                eprintln!("Unknown argument: {}", other);
            }
        }
    }

    (dir, first_octave, stage, out, force)
}

/// 从 dense.ply（稠密点云）重建网格 → mesh.obj。
/// 只用 dense 做主表面源。不用 sparse 补边：sparse 在 dense 覆盖区外（图像右侧）只有少量
/// 低质量点，硬补会把 mesh/ortho 撑出一块「异常数据」。dense 内部的覆盖空洞由 DSM 的
/// 扩散填洞补平。
fn build_mesh_from_dense_ply(out: &Paths) {
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    if let Ok(ply) = std::fs::read_to_string(out.dense_ply()) {
        for line in ply.lines().skip(10) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(x), Ok(y), Ok(z)) = (parts[0].parse(), parts[1].parse(), parts[2].parse())
                {
                    pts.push((x, y, z));
                }
            }
        }
    }
    if pts.is_empty() {
        eprintln!("dense.ply 为空或不存在");
        return;
    }
    println!("\n===== Building mesh from {} dense points =====", pts.len());

    let opts = crate::mesh::DsmOptions {
        max_side: 160,
        fill_radius: 3,
        min_pts: 1,
    };
    crate::mesh::build_mesh_dsm(&pts, None, &opts, &out.mesh_obj());
}

fn main() {
    let total_start = Instant::now();

    let (image_dir, first_octave, stage, out_dir, force) = parse_args();

    backend::init();

    let out = Paths::new(&out_dir);
    out.ensure_dirs();

    let paths = load_image_paths(&image_dir);
    if paths.is_empty() {
        eprintln!("No images found in {} directory", image_dir);
        return;
    }
    println!("Found {} images\n", paths.len());

    let features = extract_features(&paths, first_octave, &out.features_cache_dir());
    if stage <= Stage::Features {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    let matches = match_all_pairs(&features, RATIO_THRESHOLD, &out.matches_cache_dir());
    if stage <= Stage::Matches {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    // 主入口：SfI初始化→增量注册→BA→MVS稠密重建
    let tracks = build_tracks(&matches, &features);
    if stage <= Stage::Tracks {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    visualize_track_coverage_graph(&tracks, &features, &paths, "coverage_graph.png");
    if stage <= Stage::Coverage {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    // SfM 增量跳过：sparse.ply + camera_poses.json 都比 matches 缓存新（说明 SfM 已跑过、
    // matches 没变），跳过 SfM。`-force` 强制重跑。
    let sfm_fresh = !force
        && is_newer_than(&out.sparse_ply(), &out.matches_cache_dir())
        && is_newer_than(&out.camera_poses_json(), &out.matches_cache_dir());
    if sfm_fresh {
        println!("  [skip] SfM: sparse.ply/camera_poses.json 已是最新（-force 强制重跑）");
    } else {
        run_incremental_sfm(&matches, &features, &tracks, &paths, &out);
    }
    if stage <= Stage::Sfm {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    // MVS 增量跳过：dense.ply 比 camera_poses.json + sparse.ply 都新 → 跳过 MVS（1.5h 重活）。
    let mvs_fresh = !force
        && is_newer_than(&out.dense_ply(), &out.camera_poses_json())
        && is_newer_than(&out.dense_ply(), &out.sparse_ply());
    if mvs_fresh {
        println!("  [skip] MVS: dense.ply 已是最新（-force 强制重跑）");
    } else {
        run_mvs_dense(&paths, &out);
    }
    if stage <= Stage::Mvs {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    // `-stage=mesh` / `-stage=texture` / `-stage=all`：MVS 之后接着做网格重建。
    build_mesh_from_dense_ply(&out);
    if stage <= Stage::Mesh {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    // 贴图与正射影像相互独立，都只依赖 mesh 阶段产物（mesh.obj + 相机 + 深度图），
    // 按 stage 显式调度，互不依赖：-stage=texture 只贴图、-stage=ortho 只正射、-stage=all 都做。
    match stage {
        Stage::Texture | Stage::All => crate::texture::run_texture(&paths, &out),
        _ => {}
    }
    match stage {
        Stage::Ortho | Stage::All => crate::ortho::run_ortho(&paths, &out),
        _ => {}
    }
    let total_elapsed = total_start.elapsed();
    println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
}
