//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::format_collect,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::iter_cloned_collect,
    clippy::suboptimal_flops,
    clippy::should_panic_without_expect,
    clippy::manual_range_contains
)]

use crate::bitrate::*;
use crate::codec::*;
use crate::container::*;
use crate::dct::*;
use crate::frame::*;
use crate::huffman::*;
use crate::image::*;
use crate::motion::*;
use crate::pixel_format::*;
use crate::quantization::*;
use crate::rle::*;
use crate::scaling::*;
use crate::zigzag::*;
use std::collections::HashMap;

// -- Pixel format tests --

#[test]
fn test_rgb_to_yuv_black() {
    let rgb = RgbPixel::new(0, 0, 0);
    let yuv = rgb.to_yuv();
    assert_eq!(yuv.y, 0);
    assert_eq!(yuv.u, 128);
    assert_eq!(yuv.v, 128);
}

#[test]
fn test_rgb_to_yuv_white() {
    let rgb = RgbPixel::new(255, 255, 255);
    let yuv = rgb.to_yuv();
    assert_eq!(yuv.y, 255);
    assert!(yuv.u >= 127 && yuv.u <= 129);
    assert!(yuv.v >= 127 && yuv.v <= 129);
}

#[test]
fn test_yuv_to_rgb_neutral() {
    let yuv = YuvPixel::new(128, 128, 128);
    let rgb = yuv.to_rgb();
    assert_eq!(rgb.r, 128);
    assert_eq!(rgb.g, 128);
    assert_eq!(rgb.b, 128);
}

#[test]
fn test_rgb_yuv_roundtrip() {
    let original = RgbPixel::new(100, 150, 200);
    let yuv = original.to_yuv();
    let back = yuv.to_rgb();
    assert!((i16::from(original.r) - i16::from(back.r)).abs() <= 3);
    assert!((i16::from(original.g) - i16::from(back.g)).abs() <= 3);
    assert!((i16::from(original.b) - i16::from(back.b)).abs() <= 3);
}

#[test]
fn test_yuv_pixel_new() {
    let p = YuvPixel::new(16, 128, 128);
    assert_eq!(p.y, 16);
    assert_eq!(p.u, 128);
    assert_eq!(p.v, 128);
}

#[test]
fn test_rgb_pixel_new() {
    let p = RgbPixel::new(255, 0, 128);
    assert_eq!(p.r, 255);
    assert_eq!(p.g, 0);
    assert_eq!(p.b, 128);
}

#[test]
fn test_yuv_to_rgb_black() {
    let yuv = YuvPixel::new(0, 128, 128);
    let rgb = yuv.to_rgb();
    assert_eq!(rgb.r, 0);
    assert_eq!(rgb.g, 0);
    assert_eq!(rgb.b, 0);
}

#[test]
fn test_yuv_to_rgb_clamp() {
    let yuv = YuvPixel::new(255, 0, 255);
    let rgb = yuv.to_rgb();
    // Should not panic; verify we got valid values
    let _ = rgb.r;
    let _ = rgb.g;
    let _ = rgb.b;
}

// -- YUV420 Image tests --

#[test]
fn test_yuv420_new() {
    let img = Yuv420Image::new(16, 16);
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.y_plane.len(), 256);
    assert_eq!(img.u_plane.len(), 64);
    assert_eq!(img.v_plane.len(), 64);
}

#[test]
fn test_yuv420_odd_dimensions() {
    let img = Yuv420Image::new(15, 15);
    assert_eq!(img.y_plane.len(), 225);
    assert_eq!(img.u_plane.len(), 64); // 8*8
}

#[test]
fn test_yuv420_set_get_luma() {
    let mut img = Yuv420Image::new(8, 8);
    img.set_luma(3, 4, 200);
    assert_eq!(img.get_luma(3, 4), 200);
}

#[test]
fn test_yuv420_chroma() {
    let img = Yuv420Image::new(8, 8);
    assert_eq!(img.get_chroma_u(0, 0), 128);
    assert_eq!(img.get_chroma_v(0, 0), 128);
}

#[test]
fn test_yuv420_byte_size() {
    let img = Yuv420Image::new(16, 16);
    assert_eq!(img.byte_size(), 256 + 64 + 64);
}

// -- RGB Image tests --

#[test]
fn test_rgb_image_new() {
    let img = RgbImage::new(4, 4);
    assert_eq!(img.data.len(), 48);
}

#[test]
fn test_rgb_image_set_get() {
    let mut img = RgbImage::new(4, 4);
    let p = RgbPixel::new(10, 20, 30);
    img.set_pixel(2, 3, p);
    assert_eq!(img.get_pixel(2, 3), p);
}

// -- Frame & GOP tests --

#[test]
fn test_frame_constant() {
    let f = VideoFrame::constant(FrameType::I, 8, 8, 128);
    assert_eq!(f.pixel_count(), 64);
    assert!(f.luma.iter().all(|&v| v == 128));
}

#[test]
fn test_frame_new() {
    let data = vec![0_i16; 16];
    let f = VideoFrame::new(FrameType::P, 4, 4, data);
    assert_eq!(f.frame_type, FrameType::P);
    assert_eq!(f.width, 4);
    assert_eq!(f.height, 4);
}

#[test]
fn test_gop_ibbp() {
    let gop = GopStructure::ibbp(2, 3);
    assert_eq!(gop.pattern[0], FrameType::I);
    assert_eq!(gop.pattern[1], FrameType::B);
    assert_eq!(gop.pattern[2], FrameType::B);
    assert_eq!(gop.pattern[3], FrameType::P);
    assert_eq!(gop.len(), 10); // I + 3*(BB+P)
}

#[test]
fn test_gop_intra_only() {
    let gop = GopStructure::intra_only(5);
    assert_eq!(gop.len(), 5);
    assert!(gop.pattern.iter().all(|&f| f == FrameType::I));
}

#[test]
fn test_gop_ip_only() {
    let gop = GopStructure::ip_only(4);
    assert_eq!(gop.pattern[0], FrameType::I);
    assert_eq!(gop.pattern[1], FrameType::P);
    assert_eq!(gop.pattern[3], FrameType::P);
}

#[test]
fn test_gop_is_empty() {
    let gop = GopStructure { pattern: vec![] };
    assert!(gop.is_empty());
}

#[test]
fn test_gop_frame_type_counts() {
    let gop = GopStructure::ibbp(2, 2);
    let counts = gop.frame_type_counts();
    assert_eq!(*counts.get(&FrameType::I).unwrap(), 1);
    assert_eq!(*counts.get(&FrameType::B).unwrap(), 4);
    assert_eq!(*counts.get(&FrameType::P).unwrap(), 2);
}

#[test]
fn test_gop_frame_type_at_cyclic() {
    let gop = GopStructure::ip_only(3);
    assert_eq!(gop.frame_type_at(0), FrameType::I);
    assert_eq!(gop.frame_type_at(3), FrameType::I); // cyclic
}

// -- DCT tests --

#[test]
fn test_dct_idct_roundtrip() {
    let mut block = [0.0_f64; 64];
    for (i, val) in block.iter_mut().enumerate() {
        *val = (i as f64 * 3.7).sin() * 100.0;
    }
    let dct = dct_8x8(&block);
    let reconstructed = idct_8x8(&dct);
    for i in 0..64 {
        assert!(
            (block[i] - reconstructed[i]).abs() < 0.5,
            "Mismatch at {i}: {} vs {}",
            block[i],
            reconstructed[i]
        );
    }
}

#[test]
fn test_dct_dc_only() {
    let block = [100.0_f64; 64];
    let dct = dct_8x8(&block);
    // DC coefficient should be large, AC should be near zero
    assert!(dct[0].abs() > 100.0);
    for &val in &dct[1..] {
        assert!(val.abs() < 1e-6, "AC coefficient not zero: {val}");
    }
}

#[test]
fn test_dct_zero_block() {
    let block = [0.0_f64; 64];
    let dct = dct_8x8(&block);
    for &val in &dct {
        assert!(val.abs() < 1e-10);
    }
}

#[test]
fn test_idct_zero_block() {
    let coeffs = [0.0_f64; 64];
    let result = idct_8x8(&coeffs);
    for &val in &result {
        assert!(val.abs() < 1e-10);
    }
}

// -- Quantization tests --

#[test]
fn test_quantize_dequantize() {
    let mut block = [0.0_f64; 64];
    block[0] = 1000.0;
    block[1] = 500.0;
    let q = quantize(&block, 50);
    let dq = dequantize(&q, 50);
    // DC should be close
    assert!((block[0] - dq[0]).abs() < f64::from(QUANT_MATRIX_LUMA[0]));
}

#[test]
fn test_quantize_high_quality() {
    let block = [100.0_f64; 64];
    let q = quantize(&block, 95);
    // High quality = less aggressive quantization
    assert!(q[0] != 0);
}

#[test]
fn test_quantize_low_quality() {
    let mut block = [0.0_f64; 64];
    block[63] = 10.0;
    let q = quantize(&block, 1);
    // Low quality: high-frequency should be quantized to zero
    assert_eq!(q[63], 0);
}

#[test]
fn test_quantize_clamp_quality() {
    // quality 0 is clamped to 1, quality 255 to 100
    let block = [100.0_f64; 64];
    let q1 = quantize(&block, 0);
    let q2 = quantize(&block, 1);
    assert_eq!(q1, q2);
}

// -- Zigzag tests --

#[test]
fn test_zigzag_inverse_roundtrip() {
    let mut block = [0_i16; 64];
    for (i, val) in block.iter_mut().enumerate() {
        *val = i as i16;
    }
    let zz = zigzag_scan(&block);
    let back = inverse_zigzag(&zz);
    assert_eq!(block, back);
}

#[test]
fn test_zigzag_first_element() {
    let mut block = [0_i16; 64];
    block[0] = 42;
    let zz = zigzag_scan(&block);
    assert_eq!(zz[0], 42);
}

#[test]
fn test_zigzag_order_valid() {
    let mut seen = [false; 64];
    for &idx in &ZIGZAG_ORDER {
        assert!(!seen[idx], "Duplicate index in zigzag order");
        seen[idx] = true;
    }
}

// -- RLE tests --

#[test]
fn test_rle_all_zeros() {
    let block = [0_i16; 64];
    let rle = rle_encode(&block);
    assert_eq!(rle.len(), 1);
    assert_eq!(
        rle[0],
        RlePair {
            zero_run: 0,
            value: 0
        }
    );
}

#[test]
fn test_rle_single_value() {
    let mut block = [0_i16; 64];
    block[0] = 42;
    let rle = rle_encode(&block);
    assert_eq!(
        rle[0],
        RlePair {
            zero_run: 0,
            value: 42
        }
    );
}

#[test]
fn test_rle_roundtrip() {
    let mut block = [0_i16; 64];
    block[0] = 10;
    block[3] = -5;
    block[10] = 20;
    let rle = rle_encode(&block);
    let decoded = rle_decode(&rle);
    assert_eq!(block, decoded);
}

#[test]
fn test_rle_encode_decode_complex() {
    let mut block = [0_i16; 64];
    block[0] = 100;
    block[1] = -50;
    block[5] = 25;
    block[63] = 1;
    let rle = rle_encode(&block);
    let decoded = rle_decode(&rle);
    assert_eq!(block, decoded);
}

// -- Huffman tests --

#[test]
fn test_huffman_single_symbol() {
    let mut freq = HashMap::new();
    freq.insert(65, 10);
    let table = HuffmanTable::build(&freq);
    assert_eq!(table.symbol_count(), 1);
    let encoded = table.encode(&[65, 65, 65]);
    let decoded = table.decode(&encoded);
    assert_eq!(decoded, vec![65, 65, 65]);
}

#[test]
fn test_huffman_two_symbols() {
    let mut freq = HashMap::new();
    freq.insert(0, 10);
    freq.insert(1, 5);
    let table = HuffmanTable::build(&freq);
    assert_eq!(table.symbol_count(), 2);
}

#[test]
fn test_huffman_roundtrip() {
    let data = b"aabbccddaabb";
    let mut freq = HashMap::new();
    for &b in data.iter() {
        *freq.entry(b).or_insert(0) += 1;
    }
    let table = HuffmanTable::build(&freq);
    let bits = table.encode(data);
    let decoded = table.decode(&bits);
    assert_eq!(decoded, data.to_vec());
}

#[test]
fn test_huffman_empty() {
    let freq = HashMap::new();
    let table = HuffmanTable::build(&freq);
    assert_eq!(table.symbol_count(), 0);
}

#[test]
fn test_huffman_compression() {
    // Skewed distribution should compress
    let mut freq = HashMap::new();
    freq.insert(0, 1000);
    freq.insert(1, 1);
    let table = HuffmanTable::build(&freq);
    let code0 = table.get_code(0).unwrap();
    let code1 = table.get_code(1).unwrap();
    // More frequent symbol should have shorter or equal code
    assert!(code0.len() <= code1.len());
}

#[test]
fn test_huffman_four_symbols() {
    let mut freq = HashMap::new();
    freq.insert(b'a', 50);
    freq.insert(b'b', 30);
    freq.insert(b'c', 15);
    freq.insert(b'd', 5);
    let table = HuffmanTable::build(&freq);
    let data = b"aabcd";
    let bits = table.encode(data);
    let decoded = table.decode(&bits);
    assert_eq!(decoded, data.to_vec());
}

// -- Motion Vector tests --

#[test]
fn test_mv_new() {
    let mv = MotionVector::new(3, -4);
    assert_eq!(mv.dx, 3);
    assert_eq!(mv.dy, -4);
}

#[test]
fn test_mv_magnitude_sq() {
    let mv = MotionVector::new(3, 4);
    assert_eq!(mv.magnitude_sq(), 25);
}

#[test]
fn test_mv_add() {
    let a = MotionVector::new(1, 2);
    let b = MotionVector::new(3, 4);
    let c = a.add(b);
    assert_eq!(c.dx, 4);
    assert_eq!(c.dy, 6);
}

#[test]
fn test_mv_half() {
    let mv = MotionVector::new(4, -6);
    let h = mv.half();
    assert_eq!(h.dx, 2);
    assert_eq!(h.dy, -3);
}

#[test]
fn test_mv_default() {
    let mv = MotionVector::default();
    assert_eq!(mv.dx, 0);
    assert_eq!(mv.dy, 0);
}

// -- Motion Estimation tests --

#[test]
fn test_me_identical_frames() {
    let frame: Vec<i16> = (0..64).collect();
    let mv = full_search_motion_estimation(&frame, &frame, 8, 8, 0, 0, 4, 2);
    assert_eq!(mv.dx, 0);
    assert_eq!(mv.dy, 0);
}

#[test]
fn test_me_shifted_frame() {
    let w = 16_u32;
    let h = 16_u32;
    let mut current = vec![0_i16; (w * h) as usize];
    let mut reference = vec![0_i16; (w * h) as usize];
    // Place a pattern in reference at (2,2) and in current at (4,4)
    for r in 0..4 {
        for c in 0..4 {
            reference[((2 + r) * w + (2 + c)) as usize] = 100;
            current[((4 + r) * w + (4 + c)) as usize] = 100;
        }
    }
    let mv = full_search_motion_estimation(&current, &reference, w, h, 4, 4, 4, 4);
    assert_eq!(mv.dx, -2);
    assert_eq!(mv.dy, -2);
}

#[test]
fn test_motion_compensate() {
    let w = 8_u32;
    let h = 8_u32;
    let reference: Vec<i16> = (0..64).collect();
    let block = motion_compensate_block(&reference, w, h, 0, 0, 4, MotionVector::new(0, 0));
    assert_eq!(block.len(), 16);
    assert_eq!(block[0], 0);
    assert_eq!(block[1], 1);
}

#[test]
fn test_motion_compensate_with_mv() {
    let w = 8_u32;
    let h = 8_u32;
    let mut reference = vec![0_i16; 64];
    reference[(2 * w + 3) as usize] = 99;
    let block = motion_compensate_block(&reference, w, h, 2, 1, 2, MotionVector::new(1, 1));
    assert_eq!(block[0], 99);
}

#[test]
fn test_bidir_compensate() {
    let w = 8_u32;
    let h = 8_u32;
    let fwd = vec![100_i16; 64];
    let bwd = vec![200_i16; 64];
    let block = bidir_compensate_block(
        &fwd,
        &bwd,
        w,
        h,
        0,
        0,
        4,
        MotionVector::new(0, 0),
        MotionVector::new(0, 0),
    );
    // Average of 100 and 200 = 150
    assert!(block.iter().all(|&v| v == 150));
}

// -- Resolution Scaling tests --

#[test]
fn test_downscale_2x() {
    let src = vec![
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let (dst, dw, dh) = downscale_2x(&src, 4, 4);
    assert_eq!(dw, 2);
    assert_eq!(dh, 2);
    assert_eq!(dst.len(), 4);
    assert_eq!(dst[0], 10);
}

#[test]
fn test_upscale_2x() {
    let src = vec![10, 20, 30, 40];
    let (dst, dw, dh) = upscale_2x(&src, 2, 2);
    assert_eq!(dw, 4);
    assert_eq!(dh, 4);
    assert_eq!(dst.len(), 16);
    assert_eq!(dst[0], 10);
    assert_eq!(dst[1], 10);
}

#[test]
fn test_bilinear_scale_identity() {
    let src = vec![100_u8; 16];
    let dst = bilinear_scale(&src, 4, 4, 4, 4);
    assert!(dst.iter().all(|&v| v == 100));
}

#[test]
fn test_bilinear_scale_downscale() {
    let src = vec![50_u8; 64];
    let dst = bilinear_scale(&src, 8, 8, 4, 4);
    assert_eq!(dst.len(), 16);
    assert!(dst.iter().all(|&v| v == 50));
}

#[test]
fn test_bilinear_scale_upscale() {
    let src = vec![128_u8; 4];
    let dst = bilinear_scale(&src, 2, 2, 4, 4);
    assert_eq!(dst.len(), 16);
    assert!(dst.iter().all(|&v| v == 128));
}

// -- Bitrate Control tests --

#[test]
fn test_bitrate_controller_new() {
    let bc = BitrateController::new(RateControlMode::Cbr, 5000, 30);
    assert_eq!(bc.quality, 50);
    assert_eq!(bc.frames_encoded, 0);
}

#[test]
fn test_target_bits_per_frame() {
    let bc = BitrateController::new(RateControlMode::Cbr, 3000, 30);
    assert_eq!(bc.target_bits_per_frame(), 100_000);
}

#[test]
fn test_bitrate_report_frame_cbr_increase() {
    let mut bc = BitrateController::new(RateControlMode::Cbr, 3000, 30);
    // Report a small frame -> quality should go up
    bc.report_frame(10_000);
    assert!(bc.quality > 50);
}

#[test]
fn test_bitrate_report_frame_cbr_decrease() {
    let mut bc = BitrateController::new(RateControlMode::Cbr, 3000, 30);
    // Report a huge frame -> quality should go down
    bc.report_frame(500_000);
    assert!(bc.quality < 50);
}

#[test]
fn test_bitrate_cq_no_change() {
    let mut bc = BitrateController::new(RateControlMode::Cq, 3000, 30);
    bc.report_frame(500_000);
    assert_eq!(bc.quality, 50);
}

#[test]
fn test_average_bitrate_zero() {
    let bc = BitrateController::new(RateControlMode::Cbr, 3000, 30);
    assert_eq!(bc.average_bitrate_kbps(), 0);
}

#[test]
fn test_bitrate_quality_bounds() {
    let mut bc = BitrateController::new(RateControlMode::Cbr, 100, 30);
    for _ in 0..200 {
        bc.report_frame(10_000_000);
    }
    assert!(bc.quality >= bc.min_quality);

    let mut bc2 = BitrateController::new(RateControlMode::Cbr, 1_000_000, 30);
    for _ in 0..200 {
        bc2.report_frame(1);
    }
    assert!(bc2.quality <= bc2.max_quality);
}

#[test]
fn test_rate_control_modes() {
    assert_eq!(RateControlMode::Cbr, RateControlMode::Cbr);
    assert_ne!(RateControlMode::Cbr, RateControlMode::Vbr);
    assert_ne!(RateControlMode::Vbr, RateControlMode::Cq);
}

// -- Container Format tests --

#[test]
fn test_box_type_fourcc() {
    assert_eq!(BoxType::Ftyp.fourcc(), *b"ftyp");
    assert_eq!(BoxType::Moov.fourcc(), *b"moov");
    assert_eq!(BoxType::Trak.fourcc(), *b"trak");
    assert_eq!(BoxType::Mdat.fourcc(), *b"mdat");
    assert_eq!(BoxType::Free.fourcc(), *b"free");
}

#[test]
fn test_box_type_from_fourcc() {
    assert_eq!(BoxType::from_fourcc(*b"ftyp"), BoxType::Ftyp);
    assert_eq!(BoxType::from_fourcc(*b"moov"), BoxType::Moov);
    assert_eq!(BoxType::from_fourcc(*b"xxxx"), BoxType::Custom(*b"xxxx"));
}

#[test]
fn test_container_box_serialize() {
    let b = ContainerBox::new(BoxType::Ftyp, vec![1, 2, 3, 4]);
    let data = b.serialize();
    assert_eq!(data.len(), 12);
    assert_eq!(&data[4..8], b"ftyp");
}

#[test]
fn test_container_box_parse() {
    let b = ContainerBox::new(BoxType::Mdat, vec![0xAA, 0xBB]);
    let data = b.serialize();
    let (parsed, consumed) = ContainerBox::parse(&data).unwrap();
    assert_eq!(consumed, 10);
    assert_eq!(parsed.box_type, BoxType::Mdat);
    assert_eq!(parsed.payload, vec![0xAA, 0xBB]);
}

#[test]
fn test_container_box_total_size() {
    let b = ContainerBox::new(BoxType::Free, vec![0; 100]);
    assert_eq!(b.total_size(), 108);
}

#[test]
fn test_container_file_roundtrip() {
    let mut file = ContainerFile::new();
    file.add_box(ContainerBox::new(BoxType::Ftyp, vec![1, 2, 3]));
    file.add_box(ContainerBox::new(BoxType::Moov, vec![4, 5]));
    file.add_box(ContainerBox::new(BoxType::Mdat, vec![6, 7, 8, 9]));

    let data = file.serialize();
    let parsed = ContainerFile::parse(&data);
    assert_eq!(parsed.boxes.len(), 3);
    assert_eq!(parsed.boxes[0].box_type, BoxType::Ftyp);
    assert_eq!(parsed.boxes[1].box_type, BoxType::Moov);
    assert_eq!(parsed.boxes[2].box_type, BoxType::Mdat);
}

#[test]
fn test_container_file_find_boxes() {
    let mut file = ContainerFile::new();
    file.add_box(ContainerBox::new(BoxType::Mdat, vec![1]));
    file.add_box(ContainerBox::new(BoxType::Mdat, vec![2]));
    file.add_box(ContainerBox::new(BoxType::Moov, vec![3]));
    let found = file.find_boxes(&BoxType::Mdat);
    assert_eq!(found.len(), 2);
}

#[test]
fn test_container_file_total_size() {
    let mut file = ContainerFile::new();
    file.add_box(ContainerBox::new(BoxType::Ftyp, vec![0; 4]));
    assert_eq!(file.total_size(), 12);
}

#[test]
fn test_container_file_default() {
    let file = ContainerFile::default();
    assert!(file.boxes.is_empty());
}

#[test]
fn test_container_parse_empty() {
    let file = ContainerFile::parse(&[]);
    assert!(file.boxes.is_empty());
}

#[test]
fn test_container_parse_truncated() {
    let result = ContainerBox::parse(&[0, 0, 0]);
    assert!(result.is_none());
}

// -- Codec Pipeline tests --

#[test]
fn test_encode_decode_block() {
    let mut block = [0.0_f64; 64];
    for (i, val) in block.iter_mut().enumerate() {
        *val = ((i % 8) as f64 * 30.0).min(255.0);
    }
    let rle = encode_block(&block, 80);
    let decoded = decode_block(&rle, 80);
    let p = psnr(&block, &decoded);
    assert!(p > 20.0, "PSNR too low: {p}");
}

#[test]
fn test_psnr_identical() {
    let block = [100.0_f64; 64];
    assert_eq!(psnr(&block, &block), f64::INFINITY);
}

#[test]
fn test_psnr_different() {
    let a = [100.0_f64; 64];
    let mut b = [100.0_f64; 64];
    b[0] = 200.0;
    let p = psnr(&a, &b);
    assert!(p > 0.0 && p < 100.0);
}

#[test]
fn test_frame_psnr_identical() {
    let data = vec![128_i16; 64];
    assert_eq!(frame_psnr(&data, &data), f64::INFINITY);
}

#[test]
fn test_frame_psnr_different() {
    let a = vec![100_i16; 64];
    let b = vec![110_i16; 64];
    let p = frame_psnr(&a, &b);
    assert!(p > 20.0);
}

// -- PixelFormat enum tests --

#[test]
fn test_pixel_format_eq() {
    assert_eq!(PixelFormat::Yuv420, PixelFormat::Yuv420);
    assert_ne!(PixelFormat::Yuv420, PixelFormat::Rgb);
}

#[test]
fn test_pixel_format_clone() {
    let fmt = PixelFormat::Rgb;
    let fmt2 = fmt;
    assert_eq!(fmt, fmt2);
}

// -- Frame type tests --

#[test]
fn test_frame_type_eq() {
    assert_eq!(FrameType::I, FrameType::I);
    assert_ne!(FrameType::I, FrameType::P);
    assert_ne!(FrameType::P, FrameType::B);
}

// -- Additional edge case tests --

#[test]
fn test_rle_full_nonzero() {
    let block = [1_i16; 64];
    let rle = rle_encode(&block);
    let decoded = rle_decode(&rle);
    assert_eq!(block, decoded);
}

#[test]
fn test_quantize_matrix_nonzero() {
    for &val in &QUANT_MATRIX_LUMA {
        assert!(val > 0);
    }
}

#[test]
fn test_dct_energy_conservation() {
    // Parseval's theorem: energy in spatial ~= energy in frequency (scaled)
    let mut block = [0.0_f64; 64];
    for (i, val) in block.iter_mut().enumerate() {
        *val = (i as f64).sin() * 50.0;
    }
    let spatial_energy: f64 = block.iter().map(|x| x * x).sum();
    let dct = dct_8x8(&block);
    let freq_energy: f64 = dct.iter().map(|x| x * x).sum();
    // Ratio should be close to 1/16 for our normalization
    let ratio = freq_energy / spatial_energy;
    assert!(
        ratio > 0.01 && ratio < 100.0,
        "Energy ratio out of range: {ratio}"
    );
}

#[test]
fn test_encode_block_high_quality() {
    let block = [128.0_f64; 64];
    let rle = encode_block(&block, 100);
    let decoded = decode_block(&rle, 100);
    let p = psnr(&block, &decoded);
    assert!(p > 30.0);
}

#[test]
fn test_mv_zero_magnitude() {
    let mv = MotionVector::new(0, 0);
    assert_eq!(mv.magnitude_sq(), 0);
}

#[test]
fn test_container_box_custom_type() {
    let custom = BoxType::Custom(*b"test");
    assert_eq!(custom.fourcc(), *b"test");
}

#[test]
fn test_container_multiple_parse() {
    let mut file = ContainerFile::new();
    for i in 0..10 {
        file.add_box(ContainerBox::new(BoxType::Mdat, vec![i]));
    }
    let data = file.serialize();
    let parsed = ContainerFile::parse(&data);
    assert_eq!(parsed.boxes.len(), 10);
}

#[test]
fn test_bilinear_scale_1x1() {
    let src = vec![200_u8];
    let dst = bilinear_scale(&src, 1, 1, 1, 1);
    assert_eq!(dst, vec![200]);
}

#[test]
fn test_bitrate_controller_fps_zero() {
    let bc = BitrateController::new(RateControlMode::Cbr, 3000, 0);
    assert_eq!(bc.target_bits_per_frame(), 0);
    assert_eq!(bc.average_bitrate_kbps(), 0);
}

#[test]
fn test_vbr_mode() {
    let mut bc = BitrateController::new(RateControlMode::Vbr, 3000, 30);
    bc.report_frame(500_000);
    // VBR still adjusts quality
    assert!(bc.quality < 50);
}

#[test]
fn test_rgb_to_yuv_red() {
    let rgb = RgbPixel::new(255, 0, 0);
    let yuv = rgb.to_yuv();
    // Red: Y ~ 76, U ~ 84, V ~ 255
    assert!(yuv.y > 50 && yuv.y < 100);
}

#[test]
fn test_gop_ibbp_single_p() {
    let gop = GopStructure::ibbp(2, 1);
    assert_eq!(gop.len(), 4); // I, B, B, P
}

#[test]
fn test_me_zero_block() {
    let current = vec![0_i16; 64];
    let reference = vec![0_i16; 64];
    let mv = full_search_motion_estimation(&current, &reference, 8, 8, 0, 0, 4, 2);
    assert_eq!(mv.dx, 0);
    assert_eq!(mv.dy, 0);
}

#[test]
fn test_downscale_upscale_size() {
    let src = vec![128_u8; 64];
    let (down, dw, dh) = downscale_2x(&src, 8, 8);
    assert_eq!(dw, 4);
    assert_eq!(dh, 4);
    let (up, uw, uh) = upscale_2x(&down, dw, dh);
    assert_eq!(uw, 8);
    assert_eq!(uh, 8);
    assert_eq!(up.len(), 64);
}

#[test]
fn test_frame_timestamp() {
    let mut f = VideoFrame::constant(FrameType::I, 8, 8, 0);
    f.timestamp_ms = 1000;
    assert_eq!(f.timestamp_ms, 1000);
}

#[test]
fn test_huffman_many_symbols() {
    let mut freq = HashMap::new();
    for i in 0..=255 {
        freq.insert(i, 256 - u32::from(i));
    }
    let table = HuffmanTable::build(&freq);
    assert_eq!(table.symbol_count(), 256);
    let data: Vec<u8> = (0..50).collect();
    let bits = table.encode(&data);
    let decoded = table.decode(&bits);
    assert_eq!(decoded, data);
}
