//! Resolution scaling (`downscale_2x` / `upscale_2x` / `bilinear_scale`).

// Resolution Scaling
// ---------------------------------------------------------------------------

/// Nearest-neighbor downscale by a factor of 2.
#[must_use]
pub fn downscale_2x(src: &[u8], src_w: u32, src_h: u32) -> (Vec<u8>, u32, u32) {
    let dst_w = src_w / 2;
    let dst_h = src_h / 2;
    let mut dst = vec![0_u8; (dst_w * dst_h) as usize];
    for y in 0..dst_h {
        for x in 0..dst_w {
            dst[(y * dst_w + x) as usize] = src[(y * 2 * src_w + x * 2) as usize];
        }
    }
    (dst, dst_w, dst_h)
}

/// Nearest-neighbor upscale by a factor of 2.
#[must_use]
pub fn upscale_2x(src: &[u8], src_w: u32, src_h: u32) -> (Vec<u8>, u32, u32) {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0_u8; (dst_w * dst_h) as usize];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = x / 2;
            let sy = y / 2;
            dst[(y * dst_w + x) as usize] = src[(sy * src_w + sx) as usize];
        }
    }
    (dst, dst_w, dst_h)
}

/// Bilinear downscale to arbitrary resolution.
#[must_use]
pub fn bilinear_scale(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0_u8; (dst_w * dst_h) as usize];
    let x_ratio = if dst_w > 1 {
        (src_w - 1) as f64 / (dst_w - 1) as f64
    } else {
        0.0
    };
    let y_ratio = if dst_h > 1 {
        (src_h - 1) as f64 / (dst_h - 1) as f64
    } else {
        0.0
    };

    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_x = x as f64 * x_ratio;
            let src_y = y as f64 * y_ratio;
            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let xf = src_x - src_x.floor();
            let yf = src_y - src_y.floor();

            let tl = f64::from(src[(y0 * src_w + x0) as usize]);
            let tr = f64::from(src[(y0 * src_w + x1) as usize]);
            let bl = f64::from(src[(y1 * src_w + x0) as usize]);
            let br = f64::from(src[(y1 * src_w + x1) as usize]);

            let val = (br * xf).mul_add(
                yf,
                (bl * (1.0 - xf)).mul_add(
                    yf,
                    (tl * (1.0 - xf)).mul_add(1.0 - yf, tr * xf * (1.0 - yf)),
                ),
            );
            dst[(y * dst_w + x) as usize] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}
