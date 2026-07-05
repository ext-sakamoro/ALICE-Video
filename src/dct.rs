//! DCT & Inverse DCT 8x8.

use core::f64::consts::PI;
// DCT & Inverse DCT (8x8)
// ---------------------------------------------------------------------------

/// 8x8 block size used in DCT.
pub const BLOCK_SIZE: usize = 8;

/// Compute the 2D DCT-II of an 8x8 block.
#[must_use]
pub fn dct_8x8(block: &[f64; 64]) -> [f64; 64] {
    let mut result = [0.0_f64; 64];
    for u in 0..BLOCK_SIZE {
        for v in 0..BLOCK_SIZE {
            let cu = if u == 0 {
                1.0 / core::f64::consts::SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                1.0 / core::f64::consts::SQRT_2
            } else {
                1.0
            };
            let mut sum = 0.0;
            for x in 0..BLOCK_SIZE {
                for y in 0..BLOCK_SIZE {
                    let pixel = block[x * BLOCK_SIZE + y];
                    let cos_x =
                        ((2 * x + 1) as f64 * u as f64 * PI / (2 * BLOCK_SIZE) as f64).cos();
                    let cos_y =
                        ((2 * y + 1) as f64 * v as f64 * PI / (2 * BLOCK_SIZE) as f64).cos();
                    sum += pixel * cos_x * cos_y;
                }
            }
            result[u * BLOCK_SIZE + v] = 0.25 * cu * cv * sum;
        }
    }
    result
}

/// Compute the 2D IDCT (inverse DCT-II) of an 8x8 block.
#[must_use]
pub fn idct_8x8(coeffs: &[f64; 64]) -> [f64; 64] {
    let mut result = [0.0_f64; 64];
    for x in 0..BLOCK_SIZE {
        for y in 0..BLOCK_SIZE {
            let mut sum = 0.0;
            for u in 0..BLOCK_SIZE {
                for v in 0..BLOCK_SIZE {
                    let cu = if u == 0 {
                        1.0 / core::f64::consts::SQRT_2
                    } else {
                        1.0
                    };
                    let cv = if v == 0 {
                        1.0 / core::f64::consts::SQRT_2
                    } else {
                        1.0
                    };
                    let cos_x =
                        ((2 * x + 1) as f64 * u as f64 * PI / (2 * BLOCK_SIZE) as f64).cos();
                    let cos_y =
                        ((2 * y + 1) as f64 * v as f64 * PI / (2 * BLOCK_SIZE) as f64).cos();
                    sum += cu * cv * coeffs[u * BLOCK_SIZE + v] * cos_x * cos_y;
                }
            }
            result[x * BLOCK_SIZE + y] = 0.25 * sum;
        }
    }
    result
}
