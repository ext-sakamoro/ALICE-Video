//! Codec pipeline: encode/decode helpers.

use crate::dct::{dct_8x8, idct_8x8};
use crate::quantization::{dequantize, quantize};
use crate::rle::{rle_decode, rle_encode, RlePair};
use crate::zigzag::{inverse_zigzag, zigzag_scan};

// Codec Pipeline: encode / decode helpers
// ---------------------------------------------------------------------------

/// Encode a single 8x8 luma block through the full pipeline:
/// DCT -> Quantize -> Zigzag -> RLE.
#[must_use]
pub fn encode_block(block: &[f64; 64], quality: u8) -> Vec<RlePair> {
    let dct = dct_8x8(block);
    let quantized = quantize(&dct, quality);
    let zigzag = zigzag_scan(&quantized);
    rle_encode(&zigzag)
}

/// Decode a single 8x8 luma block:
/// RLE -> Inverse Zigzag -> Dequantize -> IDCT.
#[must_use]
pub fn decode_block(rle: &[RlePair], quality: u8) -> [f64; 64] {
    let zigzag = rle_decode(rle);
    let quantized = inverse_zigzag(&zigzag);
    let dequantized = dequantize(&quantized, quality);
    idct_8x8(&dequantized)
}

/// Compute PSNR between two blocks.
#[must_use]
pub fn psnr(original: &[f64; 64], reconstructed: &[f64; 64]) -> f64 {
    let mse: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / 64.0;
    if mse < 1e-10 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Compute frame-level PSNR between two luma buffers.
///
/// # Panics
///
/// Panics if `original` and `reconstructed` have different lengths.
#[must_use]
pub fn frame_psnr(original: &[i16], reconstructed: &[i16]) -> f64 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f64;
    let mse: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| {
            let d = f64::from(*a) - f64::from(*b);
            d * d
        })
        .sum::<f64>()
        / n;
    if mse < 1e-10 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}
