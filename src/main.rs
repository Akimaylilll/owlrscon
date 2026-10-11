mod backend;
mod imgproc;
mod matching;
mod mvs;
mod sfm;
mod tracks;
mod types;

use matching::{extract_features, match_all_pairs};
use mvs::run_mvs_dense;
use sfm::run_incremental_sfm;
use tracks::{build_tracks, visualize_track_coverage_graph};
use types::RATIO_THRESHOLD;

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
}

fn parse_stage(s: &str) -> Stage {
    match s.to_lowercase().as_str() {
        "features" | "sift" => Stage::Features,
        "matches" | "match" => Stage::Matches,
        "tracks" | "track" => Stage::Tracks,
        "coverage" => Stage::Coverage,
        "sfm" => Stage::Sfm,
        "mvs" | "all" => Stage::Mvs,
        other => {
            eprintln!("Unknown stage '{}', defaulting to 'mvs'", other);
            Stage::Mvs
        }
    }
}

fn parse_args() -> (String, i32, Stage) {
    let mut dir = String::from("./images");
    let mut first_octave: i32 = 0;
    let mut stage = Stage::Mvs;

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
            other => {
                eprintln!("Unknown argument: {}", other);
            }
        }
    }

    (dir, first_octave, stage)
}

fn main() {
    let total_start = Instant::now();

    backend::init();

    let (image_dir, first_octave, stage) = parse_args();

    let paths = load_image_paths(&image_dir);
    if paths.is_empty() {
        eprintln!("No images found in {} directory", image_dir);
        return;
    }
    println!("Found {} images\n", paths.len());

    let features = extract_features(&paths, first_octave);
    if stage <= Stage::Features {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    let matches = match_all_pairs(&features, RATIO_THRESHOLD);
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

    run_incremental_sfm(&matches, &features, &tracks, &paths);
    if stage <= Stage::Sfm {
        let total_elapsed = total_start.elapsed();
        println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
        return;
    }

    // MVS Dense Reconstruction
    // 计算两相机间相对位姿
    run_mvs_dense(&paths);
    let total_elapsed = total_start.elapsed();
    println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f32());
}
