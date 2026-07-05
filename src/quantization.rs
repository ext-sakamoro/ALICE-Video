//! Quantization (`quantize` / `dequantize` + `QUANT_MATRIX_LUMA`).

// Quantization
// ---------------------------------------------------------------------------

/// Standard JPEG-like luminance quantization matrix.
pub const QUANT_MATRIX_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Quantize a DCT block with a given quality factor (1-100).
#[must_use]
pub fn quantize(dct_block: &[f64; 64], quality: u8) -> [i16; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 {
        5000.0 / f64::from(q)
    } else {
        2.0f64.mul_add(-f64::from(q), 200.0)
    };

    let mut result = [0_i16; 64];
    for i in 0..64 {
        let qval = (f64::from(QUANT_MATRIX_LUMA[i]).mul_add(scale, 50.0) / 100.0).max(1.0);
        result[i] = (dct_block[i] / qval).round() as i16;
    }
    result
}

/// Dequantize a quantized block.
#[must_use]
pub fn dequantize(quantized: &[i16; 64], quality: u8) -> [f64; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 {
        5000.0 / f64::from(q)
    } else {
        2.0f64.mul_add(-f64::from(q), 200.0)
    };

    let mut result = [0.0_f64; 64];
    for i in 0..64 {
        let qval = (f64::from(QUANT_MATRIX_LUMA[i]).mul_add(scale, 50.0) / 100.0).max(1.0);
        result[i] = f64::from(quantized[i]) * qval;
    }
    result
}
