use sift_cubecl::KeyPoint;

pub(crate) const RATIO_THRESHOLD: f32 = 0.8;
pub(crate) const DISTANCE_THRESHOLD: f32 = 0.7;
pub(crate) const GUIDED_EPIPOLAR_THRESHOLD_PX: f64 = 4.0;
pub(crate) const MIN_TRACK_LENGTH: usize = 2;
pub(crate) const RANSAC_SEED: u64 = 42;
pub(crate) const SNAKE_GAP_THRESHOLD: i64 = 3;

/// 缓存与中间产物统一放在 `cache/` 目录下，避免散落在工作目录根。
/// 换名字时只需改这里的 `cache` 前缀。
pub(crate) const FEATURES_CACHE_DIR: &str = "cache/features_cache";
pub(crate) const MATCHES_CACHE_DIR: &str = "cache/matches_cache";
pub(crate) const MVS_DM_DIR: &str = "cache/mvs_dm";

/// 单张图像的 SIFT 特征集合：图像路径 + 关键点列表 + 描述子列表（一一对应）。
pub(crate) struct ImageFeatures {
    pub(crate) path: String,
    pub(crate) keypoints: Vec<KeyPoint>,
    pub(crate) descriptors: Vec<Vec<f32>>,
}
