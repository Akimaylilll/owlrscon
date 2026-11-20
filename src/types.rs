use sift_cubecl::KeyPoint;

pub(crate) const RATIO_THRESHOLD: f32 = 0.8;
pub(crate) const DISTANCE_THRESHOLD: f32 = 0.7;
pub(crate) const GUIDED_EPIPOLAR_THRESHOLD_PX: f64 = 4.0;
pub(crate) const MIN_TRACK_LENGTH: usize = 2;
pub(crate) const RANSAC_SEED: u64 = 42;
pub(crate) const SNAKE_GAP_THRESHOLD: i64 = 3;

/// 缓存与中间产物统一放在 `cache/` 目录下（已由 `Paths` 结构体接管，迁移到 `out/cache/`）。
/// 这些常量保留仅作向后兼容参考。
#[allow(dead_code)]
pub(crate) const FEATURES_CACHE_DIR: &str = "cache/features_cache";
#[allow(dead_code)]
pub(crate) const MATCHES_CACHE_DIR: &str = "cache/matches_cache";
#[allow(dead_code)]
pub(crate) const MVS_DM_DIR: &str = "cache/mvs_dm";

/// 输出与缓存目录：所有最终产物（sparse/dense.ply、camera_poses.json、mesh.obj、
/// mesh_textured.glb、ortho.tif 等）统一放在同一个输出根目录 `out/` 下，避免散落在
/// 工作目录根。默认 `./out`，可用 `-out` 覆盖。
#[derive(Clone)]
pub(crate) struct Paths {
    pub(crate) out: String,
}

impl Paths {
    pub(crate) fn new(out: &str) -> Self {
        let out = out.trim().trim_end_matches('/').trim_end_matches('\\');
        let out = if out.is_empty() { "." } else { out };
        Self { out: out.to_string() }
    }

    /// 拼接输出根目录下的相对路径。
    fn join(&self, name: &str) -> String {
        if self.out == "." {
            name.to_string()
        } else {
            format!("{}/{}", self.out, name)
        }
    }

    // ---- 最终产物 ----
    pub(crate) fn sparse_ply(&self) -> String {
        self.join("sparse.ply")
    }
    pub(crate) fn dense_ply(&self) -> String {
        self.join("dense.ply")
    }
    pub(crate) fn camera_poses_json(&self) -> String {
        self.join("camera_poses.json")
    }
    pub(crate) fn mesh_obj(&self) -> String {
        self.join("mesh.obj")
    }
    pub(crate) fn mesh_textured_glb(&self) -> String {
        self.join("mesh_textured.glb")
    }
    pub(crate) fn ortho_tif(&self) -> String {
        self.join("ortho.tif")
    }
    pub(crate) fn ortho_preview_png(&self) -> String {
        self.join("ortho_preview.png")
    }
    pub(crate) fn coverage_graph_png(&self) -> String {
        self.join("coverage_graph.png")
    }

    // ---- 中间缓存 ----
    pub(crate) fn features_cache_dir(&self) -> String {
        self.join("cache/features_cache")
    }
    pub(crate) fn matches_cache_dir(&self) -> String {
        self.join("cache/matches_cache")
    }
    pub(crate) fn mvs_dm_dir(&self) -> String {
        self.join("cache/mvs_dm")
    }

    /// 预创建所有缓存子目录（输出根目录由各写文件的 `create_dir_all` 兜底）。
    pub(crate) fn ensure_dirs(&self) {
        let _ = std::fs::create_dir_all(self.features_cache_dir());
        let _ = std::fs::create_dir_all(self.matches_cache_dir());
        let _ = std::fs::create_dir_all(self.mvs_dm_dir());
    }
}

/// 单张图像的 SIFT 特征集合：图像路径 + 关键点列表 + 描述子列表（一一对应）。
pub(crate) struct ImageFeatures {
    pub(crate) path: String,
    pub(crate) keypoints: Vec<KeyPoint>,
    pub(crate) descriptors: Vec<Vec<f32>>,
}
