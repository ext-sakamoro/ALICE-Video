//! Analytic oracles — closed-form checks for the codec laws in ALICE-Video
//! (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms, published tables (JPEG / JFIF /
//! IJG / BT.601) or an independent reference written in this file (separable
//! DCT matrix, zigzag generator), never from the crate function under test.
//!
//! Oracle sources:
//! - DCT-II 8×8 (JPEG orthonormal): F = C·X·Cᵀ with the DCT matrix built here,
//!   Parseval Σx² = ΣF², idct∘dct = id, constant c ⇒ F₀₀ = 8c, a cosine
//!   basis image ⇒ a single coefficient of 4
//! - quantisation (IJG `jpeg_quality_scaling`): quality 50 reproduces the
//!   luma table exactly, quality 100 ⇒ every step 1, |x − deq(q(x))| ≤ step/2
//! - zigzag: the JPEG diagonal traversal generated independently
//! - BT.601 / JFIF full-range: Y = 0.299R + 0.587G + 0.114B (rounded), grey
//!   round-trips exactly with U = V = 128, JFIF primaries, |Δ| ≤ 2 round trip
//! - scaling: nearest ×2 then ÷2 is the identity, align-corners bilinear
//!   reproduces a linear gradient exactly
//! - motion: a shifted reference is found by full search with the exact
//!   shift, compensation with that vector reproduces the block, bidirectional
//!   compensation is the average
//! - bitrate: bits/frame = kbps·1000/fps, average kbps = bits·fps/(n·1000)
//! - Huffman: dyadic frequencies give code lengths log₂(total/f), Kraft sum
//!   = 1, prefix-free, H ≤ L < H + 1, decode∘encode = id
//! - RLE / pipeline: closed-form pairs, EOB, quality 100 reconstructs a
//!   constant block within 1/16, PSNR = 20 log10(255) for MSE 1
//! - container: `[size:4 BE][fourcc][payload]`, GOP counts

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};

use alice_video::bitrate::{BitrateController, RateControlMode};
use alice_video::codec::{decode_block, encode_block, frame_psnr, psnr};
use alice_video::container::{BoxType, ContainerBox, ContainerFile};
use alice_video::dct::{dct_8x8, idct_8x8};
use alice_video::frame::{FrameType, GopStructure};
use alice_video::huffman::HuffmanTable;
use alice_video::motion::{
    bidir_compensate_block, full_search_motion_estimation, motion_compensate_block, MotionVector,
};
use alice_video::pixel_format::{RgbPixel, YuvPixel};
use alice_video::quantization::{dequantize, quantize, QUANT_MATRIX_LUMA};
use alice_video::rle::{rle_decode, rle_encode, RlePair};
use alice_video::scaling::{bilinear_scale, downscale_2x, upscale_2x};
use alice_video::zigzag::{inverse_zigzag, zigzag_scan, ZIGZAG_ORDER};

/// Deterministic LCG in [lo, hi).
fn lcg(seed: &mut u64, lo: f64, hi: f64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    lo + (*seed >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
}

// ───────────────────────── DCT ────────────────────────────────────────────

/// Orthonormal 8-point DCT-II matrix: C[u][x] = c(u)/2 · cos((2x+1)uπ/16).
fn dct_matrix() -> [[f64; 8]; 8] {
    let mut c = [[0.0; 8]; 8];
    for u in 0..8 {
        let cu = if u == 0 { 1.0 / SQRT_2 } else { 1.0 };
        for x in 0..8 {
            c[u][x] = cu / 2.0 * (((2 * x + 1) as f64) * u as f64 * PI / 16.0).cos();
        }
    }
    c
}

#[test]
fn dct_8x8_is_the_orthonormal_jpeg_transform() {
    let c = dct_matrix();
    // the reference matrix is orthonormal
    for i in 0..8 {
        for j in 0..8 {
            let dot: f64 = (0..8).map(|k| c[i][k] * c[j][k]).sum();
            assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
    let mut seed = 1u64;
    let mut x = [0.0; 64];
    for v in &mut x {
        *v = lcg(&mut seed, -128.0, 128.0);
    }
    // F = C X Cᵀ
    let mut expected = [0.0; 64];
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0.0;
            for i in 0..8 {
                for j in 0..8 {
                    s += c[u][i] * x[i * 8 + j] * c[v][j];
                }
            }
            expected[u * 8 + v] = s;
        }
    }
    let f = dct_8x8(&x);
    for k in 0..64 {
        assert!(
            (f[k] - expected[k]).abs() < 1e-9,
            "F[{k}] = {} vs {}",
            f[k],
            expected[k]
        );
    }
    // Parseval and inverse
    let ex: f64 = x.iter().map(|v| v * v).sum();
    let ef: f64 = f.iter().map(|v| v * v).sum();
    assert!((ex - ef).abs() < 1e-9 * ex, "Parseval {ex} vs {ef}");
    let back = idct_8x8(&f);
    for k in 0..64 {
        assert!((back[k] - x[k]).abs() < 1e-9, "idct∘dct at {k}");
    }
    // constant c ⇒ F₀₀ = 8c, everything else 0
    let f = dct_8x8(&[37.5; 64]);
    assert!((f[0] - 300.0).abs() < 1e-9, "DC {}", f[0]);
    assert!(f[1..].iter().all(|v| v.abs() < 1e-9));
    // basis image (u, v) = (2, 5) ⇒ F₂₅ = 4, everything else 0
    let mut basis = [0.0; 64];
    for i in 0..8 {
        for j in 0..8 {
            basis[i * 8 + j] = (((2 * i + 1) as f64) * 2.0 * PI / 16.0).cos()
                * (((2 * j + 1) as f64) * 5.0 * PI / 16.0).cos();
        }
    }
    let f = dct_8x8(&basis);
    for k in 0..64 {
        let expected = if k == 2 * 8 + 5 { 4.0 } else { 0.0 };
        assert!((f[k] - expected).abs() < 1e-9, "basis F[{k}] = {}", f[k]);
    }
}

// ───────────────────────── quantisation ───────────────────────────────────

#[test]
fn quantisation_follows_the_ijg_quality_scaling() {
    // IJG: scale = q < 50 ? 5000/q : 200 − 2q; step = ⌊(Q·scale + 50)/100⌋, ≥ 1
    let ijg_step = |i: usize, q: u8| -> f64 {
        let scale = if q < 50 {
            5000 / u32::from(q)
        } else {
            200 - 2 * u32::from(q)
        };
        ((u32::from(QUANT_MATRIX_LUMA[i]) * scale + 50) / 100).max(1) as f64
    };
    let ones = [1_i16; 64];
    // quality 50 reproduces the standard luminance table exactly
    let steps = dequantize(&ones, 50);
    for i in 0..64 {
        assert_eq!(
            steps[i],
            f64::from(QUANT_MATRIX_LUMA[i]),
            "quality 50 step {i}"
        );
    }
    // quality 100 ⇒ every step 1 (lossless up to rounding)
    assert!(dequantize(&ones, 100).iter().all(|&s| s == 1.0));
    for q in [1u8, 10, 25, 50, 75, 90, 95, 100] {
        let steps = dequantize(&ones, q);
        for i in 0..64 {
            assert_eq!(steps[i], ijg_step(i, q), "quality {q} step {i}");
        }
    }
    // quantise / dequantise: |x − deq(q(x))| ≤ step/2, and higher quality never coarser
    let mut seed = 9u64;
    let mut x = [0.0; 64];
    for v in &mut x {
        *v = lcg(&mut seed, -1024.0, 1024.0);
    }
    for q in [1u8, 25, 50, 75, 100] {
        let back = dequantize(&quantize(&x, q), q);
        let steps = dequantize(&ones, q);
        for i in 0..64 {
            assert!(
                (back[i] - x[i]).abs() <= steps[i] / 2.0 + 1e-9,
                "q{q} coefficient {i}"
            );
        }
    }
    let (s25, s75) = (dequantize(&ones, 25), dequantize(&ones, 75));
    assert!(s25.iter().zip(&s75).all(|(a, b)| a >= b));
    assert_eq!(quantize(&x, 0), quantize(&x, 1), "quality clamps to 1");
    assert_eq!(
        quantize(&x, 200),
        quantize(&x, 100),
        "quality clamps to 100"
    );
}

// ───────────────────────── zigzag ─────────────────────────────────────────

#[test]
fn zigzag_order_is_the_jpeg_diagonal_traversal() {
    // generate independently: diagonals d = r + c, odd d runs down-left
    // (row ascending), even d runs up-right (row descending)
    let mut order = Vec::with_capacity(64);
    for d in 0..15usize {
        let cells: Vec<usize> = (0..8)
            .filter(|&r| d >= r && d - r < 8)
            .map(|r| r * 8 + (d - r))
            .collect();
        if d % 2 == 1 {
            order.extend(cells);
        } else {
            order.extend(cells.into_iter().rev());
        }
    }
    assert_eq!(&order[..], &ZIGZAG_ORDER[..]);
    let mut seen = [false; 64];
    for &i in &ZIGZAG_ORDER {
        assert!(!seen[i], "permutation");
        seen[i] = true;
    }
    let mut block = [0_i16; 64];
    for (i, v) in block.iter_mut().enumerate() {
        *v = i as i16 * 3 - 90;
    }
    assert_eq!(inverse_zigzag(&zigzag_scan(&block)), block);
    let z = zigzag_scan(&block);
    assert_eq!(
        (z[0], z[1], z[2], z[3]),
        (block[0], block[1], block[8], block[16])
    );
}

// ───────────────────────── colour ─────────────────────────────────────────

#[test]
fn rgb_yuv_conversion_is_bt601_full_range_with_rounding() {
    let reference = |r: f64, g: f64, b: f64| -> (u8, u8, u8) {
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let u = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
        let v = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
        (
            y.round().clamp(0.0, 255.0) as u8,
            u.round().clamp(0.0, 255.0) as u8,
            v.round().clamp(0.0, 255.0) as u8,
        )
    };
    // grey is luma only: Y = g, U = V = 128, and it round-trips exactly
    for g in 0..=255u8 {
        let yuv = RgbPixel::new(g, g, g).to_yuv();
        assert_eq!((yuv.y, yuv.u, yuv.v), (g, 128, 128), "grey {g}");
        let back = yuv.to_rgb();
        assert_eq!((back.r, back.g, back.b), (g, g, g), "grey {g} round trip");
    }
    // JFIF primaries
    assert_eq!(
        RgbPixel::new(255, 0, 0).to_yuv(),
        YuvPixel::new(76, 85, 255),
        "red"
    );
    assert_eq!(
        RgbPixel::new(0, 255, 0).to_yuv(),
        YuvPixel::new(150, 44, 21),
        "green"
    );
    assert_eq!(
        RgbPixel::new(0, 0, 255).to_yuv(),
        YuvPixel::new(29, 255, 107),
        "blue"
    );
    // every conversion equals the rounded reference, and the round trip is within 2
    for r in (0..=255).step_by(15) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(15) {
                let yuv = RgbPixel::new(r, g, b).to_yuv();
                let (ey, eu, ev) = reference(f64::from(r), f64::from(g), f64::from(b));
                assert_eq!((yuv.y, yuv.u, yuv.v), (ey, eu, ev), "rgb({r},{g},{b})");
                let back = yuv.to_rgb();
                let err = (i32::from(back.r) - i32::from(r))
                    .abs()
                    .max((i32::from(back.g) - i32::from(g)).abs())
                    .max((i32::from(back.b) - i32::from(b)).abs());
                assert!(err <= 2, "rgb({r},{g},{b}) round trip error {err}");
            }
        }
    }
    // inverse: R = Y + 1.402(V−128), G = Y − 0.344136(U−128) − 0.714136(V−128), B = Y + 1.772(U−128)
    let inverse = |y: f64, u: f64, v: f64| -> (u8, u8, u8) {
        let (u, v) = (u - 128.0, v - 128.0);
        let r = y + 1.402 * v;
        let g = y - 0.344_136 * u - 0.714_136 * v;
        let b = y + 1.772 * u;
        (
            r.round().clamp(0.0, 255.0) as u8,
            g.round().clamp(0.0, 255.0) as u8,
            b.round().clamp(0.0, 255.0) as u8,
        )
    };
    for y in (0..=255).step_by(17) {
        for u in (0..=255).step_by(15) {
            for v in (0..=255).step_by(15) {
                let rgb = YuvPixel::new(y, u, v).to_rgb();
                let (er, eg, eb) = inverse(f64::from(y), f64::from(u), f64::from(v));
                assert_eq!((rgb.r, rgb.g, rgb.b), (er, eg, eb), "yuv({y},{u},{v})");
            }
        }
    }
}

// ───────────────────────── scaling ────────────────────────────────────────

#[test]
fn scaling_reproduces_gradients_and_nearest_round_trips() {
    let (w, h) = (17u32, 5u32);
    // horizontal gradient 0, 16, …, 256→255 clamp avoided: 0..=16 × 15 = 0..=240
    let src: Vec<u8> = (0..h)
        .flat_map(|_| (0..w).map(|x| (x * 15) as u8))
        .collect();
    // align-corners bilinear: dst x maps to src x·(w−1)/(dw−1) ⇒ value 15·that
    for dw in [9u32, 17, 33, 2, 1] {
        let dst = bilinear_scale(&src, w, h, dw, h);
        assert_eq!(dst.len(), (dw * h) as usize);
        for y in 0..h {
            for x in 0..dw {
                let sx = if dw > 1 {
                    f64::from(x) * f64::from(w - 1) / f64::from(dw - 1)
                } else {
                    0.0
                };
                let expected = (15.0 * sx).round() as u8;
                assert_eq!(dst[(y * dw + x) as usize], expected, "dw {dw} at ({x},{y})");
            }
        }
    }
    // a constant image scales to the same constant at any size
    let flat = vec![77u8; (w * h) as usize];
    assert!(bilinear_scale(&flat, w, h, 40, 30).iter().all(|&v| v == 77));
    // nearest: up then down is the identity, down picks the even samples
    let (up, uw, uh) = upscale_2x(&src, w, h);
    assert_eq!((uw, uh), (2 * w, 2 * h));
    for y in 0..uh {
        for x in 0..uw {
            assert_eq!(
                up[(y * uw + x) as usize],
                src[((y / 2) * w + x / 2) as usize]
            );
        }
    }
    let (down, dw, dh) = downscale_2x(&up, uw, uh);
    assert_eq!((dw, dh), (w, h));
    assert_eq!(down, src);
    let (d2, _, _) = downscale_2x(&src, w, h);
    assert_eq!(d2[1], src[2]);
}

// ───────────────────────── motion ─────────────────────────────────────────

#[test]
fn full_search_recovers_a_pure_translation_exactly() {
    let (w, h) = (32u32, 32u32);
    let mut seed = 5u64;
    let reference: Vec<i16> = (0..w * h)
        .map(|_| lcg(&mut seed, 0.0, 256.0) as i16)
        .collect();
    let (dx, dy) = (3i32, -2i32);
    // current(x, y) = reference(x + dx, y + dy)
    let current: Vec<i16> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let rx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                let ry = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                (ry, rx)
            })
        })
        .map(|(ry, rx)| reference[(ry * w + rx) as usize])
        .collect();
    let mv = full_search_motion_estimation(&current, &reference, w, h, 8, 8, 8, 7);
    assert_eq!(mv, MotionVector::new(dx as i16, dy as i16));
    let block = motion_compensate_block(&reference, w, h, 8, 8, 8, mv);
    for row in 0..8u32 {
        for col in 0..8u32 {
            assert_eq!(
                block[(row * 8 + col) as usize],
                current[((8 + row) * w + 8 + col) as usize]
            );
        }
    }
    // zero motion on identical frames, and compensation clamps at the border
    assert_eq!(
        full_search_motion_estimation(&reference, &reference, w, h, 16, 16, 8, 4),
        MotionVector::new(0, 0)
    );
    let edge = motion_compensate_block(&reference, w, h, 0, 0, 4, MotionVector::new(-10, -10));
    assert!(edge.iter().all(|&v| v == reference[0]));
    // bidirectional = average
    let other: Vec<i16> = reference.iter().map(|&v| 255 - v).collect();
    let avg = bidir_compensate_block(
        &reference,
        &other,
        w,
        h,
        4,
        4,
        4,
        MotionVector::new(0, 0),
        MotionVector::new(0, 0),
    );
    for row in 0..4u32 {
        for col in 0..4u32 {
            let a = reference[((4 + row) * w + 4 + col) as usize];
            let expected = (i32::from(a) + i32::from(255 - a)) / 2;
            assert_eq!(i32::from(avg[(row * 4 + col) as usize]), expected);
        }
    }
    let v = MotionVector::new(3, -4);
    assert_eq!(v.magnitude_sq(), 25);
    assert_eq!(v.add(MotionVector::new(1, 1)), MotionVector::new(4, -3));
    assert_eq!(v.half(), MotionVector::new(1, -2));
}

// ───────────────────────── bitrate ────────────────────────────────────────

#[test]
fn bitrate_controller_arithmetic_and_feedback_direction() {
    let mut rc = BitrateController::new(RateControlMode::Cbr, 3_000, 30);
    assert_eq!(rc.target_bits_per_frame(), 100_000);
    for _ in 0..60 {
        rc.report_frame(100_000);
    }
    assert_eq!(rc.average_bitrate_kbps(), 3_000, "2 s at target");
    assert_eq!(rc.current_quality(), 50, "on target ⇒ quality unchanged");
    // over budget by > 25 % lowers quality by 2 per frame down to min, under raises to max
    for _ in 0..100 {
        rc.report_frame(200_000);
    }
    assert_eq!(rc.current_quality(), rc.min_quality);
    for _ in 0..100 {
        rc.report_frame(10_000);
    }
    assert_eq!(rc.current_quality(), rc.max_quality);
    assert_eq!(
        BitrateController::new(RateControlMode::Vbr, 500, 0).target_bits_per_frame(),
        0
    );
    let mut cq = BitrateController::new(RateControlMode::Cq, 500, 25);
    cq.report_frame(10_000_000);
    assert_eq!(cq.current_quality(), 50, "CQ never adapts");
}

// ───────────────────────── Huffman ────────────────────────────────────────

#[test]
fn huffman_codes_are_prefix_free_and_optimal() {
    // dyadic frequencies ⇒ code lengths log₂(16/f): 1, 2, 3, 4, 4
    let freqs: HashMap<u8, u32> = [(b'a', 8), (b'b', 4), (b'c', 2), (b'd', 1), (b'e', 1)]
        .into_iter()
        .collect();
    let t = HuffmanTable::build(&freqs);
    assert_eq!(t.symbol_count(), 5);
    for (sym, expected_len) in [(b'a', 1), (b'b', 2), (b'c', 3), (b'd', 4), (b'e', 4)] {
        assert_eq!(
            t.get_code(sym).unwrap().len(),
            expected_len,
            "len({})",
            sym as char
        );
    }
    let kraft: f64 = freqs
        .keys()
        .map(|s| 0.5f64.powi(t.get_code(*s).unwrap().len() as i32))
        .sum();
    assert!((kraft - 1.0).abs() < 1e-12, "Kraft = 1 (full tree)");
    // prefix-free
    let codes: Vec<&Vec<bool>> = freqs.keys().map(|s| t.get_code(*s).unwrap()).collect();
    for a in &codes {
        for b in &codes {
            assert!(
                a == b || !(b.len() > a.len() && b[..a.len()] == a[..]),
                "prefix"
            );
        }
    }
    // non-dyadic: H ≤ L < H + 1 and round trip
    let data: Vec<u8> = (0..1000u32).map(|i| (i * i % 7) as u8).collect();
    let mut f: HashMap<u8, u32> = HashMap::new();
    for &b in &data {
        *f.entry(b).or_insert(0) += 1;
    }
    let t = HuffmanTable::build(&f);
    let bits = t.encode(&data);
    let n = data.len() as f64;
    let entropy: f64 = f
        .values()
        .map(|&c| {
            let p = f64::from(c) / n;
            -p * p.log2()
        })
        .sum();
    let avg = bits.len() as f64 / n;
    assert!(
        entropy <= avg + 1e-9 && avg < entropy + 1.0,
        "H = {entropy}, L = {avg}"
    );
    assert_eq!(t.decode(&bits), data);
    // single symbol gets a 1-bit code, empty table encodes nothing
    let one: HashMap<u8, u32> = [(7u8, 3u32)].into_iter().collect();
    assert_eq!(HuffmanTable::build(&one).encode(&[7, 7]).len(), 2);
    assert!(HuffmanTable::build(&HashMap::new()).encode(&[1]).is_empty());
}

// ───────────────────────── RLE / pipeline ─────────────────────────────────

#[test]
fn rle_pairs_and_the_block_pipeline_match_closed_forms() {
    let mut z = [0_i16; 64];
    z[0] = 12;
    z[3] = -5;
    z[10] = 7;
    z[63] = 1;
    let pairs = rle_encode(&z);
    assert_eq!(
        pairs,
        vec![
            RlePair {
                zero_run: 0,
                value: 12
            },
            RlePair {
                zero_run: 2,
                value: -5
            },
            RlePair {
                zero_run: 6,
                value: 7
            },
            RlePair {
                zero_run: 52,
                value: 1
            },
            RlePair {
                zero_run: 0,
                value: 0
            },
        ]
    );
    assert_eq!(rle_decode(&pairs), z);
    assert_eq!(
        rle_encode(&[0; 64]),
        vec![RlePair {
            zero_run: 0,
            value: 0
        }]
    );
    assert_eq!(rle_decode(&[]), [0; 64]);
    // quality 100: a constant block c has F₀₀ = 8c, step 1 ⇒ |error| ≤ 1/16
    let block = [113.3; 64];
    let back = decode_block(&encode_block(&block, 100), 100);
    for v in &back {
        assert!((v - 113.3).abs() <= 1.0 / 16.0 + 1e-9, "constant block {v}");
    }
    // a DC-only block quantises to exactly one RLE pair (+ EOB)
    let enc = encode_block(&[80.0; 64], 50);
    assert_eq!(
        enc,
        vec![
            RlePair {
                zero_run: 0,
                value: 40
            },
            RlePair {
                zero_run: 0,
                value: 0
            }
        ],
        "640/16"
    );
    // PSNR: MSE 1 ⇒ 20·log10(255) = 48.13 dB; identical ⇒ ∞; MSE 255² ⇒ 0 dB
    let plus_one: [f64; 64] = std::array::from_fn(|i| block[i] + 1.0);
    assert!((psnr(&block, &plus_one) - 20.0 * 255f64.log10()).abs() < 1e-9);
    assert_eq!(psnr(&block, &block), f64::INFINITY);
    let a: Vec<i16> = vec![0; 100];
    let b: Vec<i16> = vec![255; 100];
    assert!(frame_psnr(&a, &b).abs() < 1e-9, "0 dB");
    let c: Vec<i16> = (0..100).map(|i| if i % 4 == 0 { 2 } else { 0 }).collect();
    // MSE = 4·25/100 = 1
    assert!((frame_psnr(&a, &c) - 20.0 * 255f64.log10()).abs() < 1e-9);
}

// ───────────────────────── container / GOP ────────────────────────────────

#[test]
fn container_boxes_and_gop_patterns() {
    let b = ContainerBox::new(BoxType::Mdat, vec![1, 2, 3, 4, 5]);
    let bytes = b.serialize();
    assert_eq!(
        bytes,
        vec![0, 0, 0, 13, b'm', b'd', b'a', b't', 1, 2, 3, 4, 5]
    );
    let (back, used) = ContainerBox::parse(&bytes).unwrap();
    assert_eq!(
        (back.box_type, back.payload, used),
        (BoxType::Mdat, vec![1, 2, 3, 4, 5], 13)
    );
    assert!(ContainerBox::parse(&bytes[..12]).is_none(), "truncated");
    assert!(
        ContainerBox::parse(&[0, 0, 0, 4, b'f', b'r', b'e', b'e']).is_none(),
        "size < 8"
    );
    let mut file = ContainerFile::new();
    file.add_box(ContainerBox::new(BoxType::Ftyp, b"isom".to_vec()));
    file.add_box(b.clone());
    file.add_box(ContainerBox::new(BoxType::Custom(*b"xyz "), vec![]));
    let ser = file.serialize();
    assert_eq!(ser.len(), 12 + 13 + 8);
    assert_eq!(file.total_size(), ser.len());
    let parsed = ContainerFile::parse(&ser);
    assert_eq!(parsed.boxes.len(), 3);
    assert_eq!(parsed.find_boxes(&BoxType::Mdat).len(), 1);
    assert_eq!(parsed.boxes[2].box_type, BoxType::Custom(*b"xyz "));
    assert_eq!(BoxType::from_fourcc(*b"moov").fourcc(), *b"moov");
    // GOP: IBBP with 2 B and 4 P ⇒ 1 + 4·3 = 13 frames, counts 1 / 4 / 8, cyclic index
    let gop = GopStructure::ibbp(2, 4);
    assert_eq!(gop.len(), 13);
    let counts = gop.frame_type_counts();
    assert_eq!(
        (
            counts[&FrameType::I],
            counts[&FrameType::P],
            counts[&FrameType::B]
        ),
        (1, 4, 8)
    );
    assert_eq!(gop.frame_type_at(0), FrameType::I);
    assert_eq!(gop.frame_type_at(3), FrameType::P);
    assert_eq!(gop.frame_type_at(13), FrameType::I);
    assert_eq!(
        GopStructure::ip_only(10).frame_type_counts()[&FrameType::P],
        9
    );
    assert!(GopStructure::intra_only(4)
        .pattern
        .iter()
        .all(|&t| t == FrameType::I));
}
