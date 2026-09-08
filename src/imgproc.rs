pub(crate) const MAX_IMAGE_DIMENSION: u32 = 2000;

/// 计算特征提取时降采样后的图像尺寸（最长边不超过 [`MAX_IMAGE_DIMENSION`]）。
///
/// # Arguments
///
/// * `w` - 原图宽度
/// * `h` - 原图高度
pub(crate) fn feature_dimensions(w: u32, h: u32) -> (u32, u32) {
    let longest = w.max(h);
    if longest > MAX_IMAGE_DIMENSION {
        let scale = MAX_IMAGE_DIMENSION as f32 / longest as f32;
        (
            (w as f32 * scale).round() as u32,
            (h as f32 * scale).round() as u32,
        )
    } else {
        (w, h)
    }
}
