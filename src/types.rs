use sift_rs::KeyPoint;

pub(crate) const RATIO_THRESHOLD: f32 = 0.8;
pub(crate) const DISTANCE_THRESHOLD: f32 = 0.7;
pub(crate) const GUIDED_EPIPOLAR_THRESHOLD_PX: f64 = 4.0;
pub(crate) const MAX_KEYPOINTS_PER_IMAGE: usize = 5000;
pub(crate) const SLIDING_WINDOW: usize = 10;
pub(crate) const MIN_TRACK_LENGTH: usize = 2;
pub(crate) const RANSAC_SEED: u64 = 42;
pub(crate) const SNAKE_GAP_THRESHOLD: i64 = 3;

/// 单张图像的 SIFT 特征集合：图像路径 + 关键点列表。
pub(crate) struct ImageFeatures {
    pub(crate) path: String,
    pub(crate) keypoints: Vec<KeyPoint>,
}
