#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_wrap)]

//! ALICE-Video: Video codec and processing library.
//!
//! Provides frame types (I/P/B), GOP structure, motion compensation,
//! DCT/quantization, entropy coding, pixel formats (YUV420, RGB),
//! resolution scaling, bitrate control, and container format basics.

use core::f64::consts::PI;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pixel Formats
// ---------------------------------------------------------------------------

/// Supported pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// YUV 4:2:0 planar
    Yuv420,
    /// RGB interleaved
    Rgb,
}

/// A single YUV pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvPixel {
    pub y: u8,
    pub u: u8,
    pub v: u8,
}

/// A single RGB pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl YuvPixel {
    #[must_use]
    pub const fn new(y: u8, u: u8, v: u8) -> Self {
        Self { y, u, v }
    }

    /// Convert YUV to RGB.
    #[must_use]
    pub fn to_rgb(self) -> RgbPixel {
        let y = f64::from(self.y);
        let u = f64::from(self.u) - 128.0;
        let v = f64::from(self.v) - 128.0;
        let r = 1.402f64.mul_add(v, y).clamp(0.0, 255.0) as u8;
        let g = 0.714_136f64
            .mul_add(-v, 0.344_136f64.mul_add(-u, y))
            .clamp(0.0, 255.0) as u8;
        let b = 1.772f64.mul_add(u, y).clamp(0.0, 255.0) as u8;
        RgbPixel { r, g, b }
    }
}

impl RgbPixel {
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert RGB to YUV.
    #[must_use]
    pub fn to_yuv(self) -> YuvPixel {
        let r = f64::from(self.r);
        let g = f64::from(self.g);
        let b = f64::from(self.b);
        let y = 0.114f64
            .mul_add(b, 0.299f64.mul_add(r, 0.587 * g))
            .clamp(0.0, 255.0) as u8;
        let u = (0.5f64.mul_add(b, (-0.168_736f64).mul_add(r, -(0.331_264 * g))) + 128.0)
            .clamp(0.0, 255.0) as u8;
        let v = (0.081_312f64.mul_add(-b, 0.5f64.mul_add(r, -(0.418_688 * g))) + 128.0)
            .clamp(0.0, 255.0) as u8;
        YuvPixel { y, u, v }
    }
}

// ---------------------------------------------------------------------------
// YUV420 Plane representation
// ---------------------------------------------------------------------------

/// YUV 4:2:0 planar image. Chroma planes are half resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420Image {
    pub width: u32,
    pub height: u32,
    pub y_plane: Vec<u8>,
    pub u_plane: Vec<u8>,
    pub v_plane: Vec<u8>,
}

impl Yuv420Image {
    /// Create a new blank YUV420 image.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let luma_size = (width * height) as usize;
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        let chroma_size = (chroma_w * chroma_h) as usize;
        Self {
            width,
            height,
            y_plane: vec![0; luma_size],
            u_plane: vec![128; chroma_size],
            v_plane: vec![128; chroma_size],
        }
    }

    /// Get luma value at (x, y).
    #[must_use]
    pub fn get_luma(&self, x: u32, y: u32) -> u8 {
        self.y_plane[(y * self.width + x) as usize]
    }

    /// Set luma value at (x, y).
    pub fn set_luma(&mut self, x: u32, y: u32, val: u8) {
        self.y_plane[(y * self.width + x) as usize] = val;
    }

    /// Get chroma U at chroma coordinates.
    #[must_use]
    pub fn get_chroma_u(&self, cx: u32, cy: u32) -> u8 {
        let cw = self.width.div_ceil(2);
        self.u_plane[(cy * cw + cx) as usize]
    }

    /// Get chroma V at chroma coordinates.
    #[must_use]
    pub fn get_chroma_v(&self, cx: u32, cy: u32) -> u8 {
        let cw = self.width.div_ceil(2);
        self.v_plane[(cy * cw + cx) as usize]
    }

    /// Total byte size of this image.
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        self.y_plane.len() + self.u_plane.len() + self.v_plane.len()
    }
}

/// RGB image buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl RgbImage {
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width * height * 3) as usize],
        }
    }

    /// Get pixel at (x, y).
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> RgbPixel {
        let idx = ((y * self.width + x) * 3) as usize;
        RgbPixel {
            r: self.data[idx],
            g: self.data[idx + 1],
            b: self.data[idx + 2],
        }
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, p: RgbPixel) {
        let idx = ((y * self.width + x) * 3) as usize;
        self.data[idx] = p.r;
        self.data[idx + 1] = p.g;
        self.data[idx + 2] = p.b;
    }
}

// ---------------------------------------------------------------------------
// Frame Types & GOP
// ---------------------------------------------------------------------------

/// Video frame type in a GOP structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// Intra-coded frame (keyframe)
    I,
    /// Predicted frame (forward reference)
    P,
    /// Bidirectional predicted frame
    B,
}

/// A video frame with its type and luma data.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub frame_type: FrameType,
    pub width: u32,
    pub height: u32,
    pub luma: Vec<i16>,
    pub timestamp_ms: u64,
}

impl VideoFrame {
    /// Create a new frame from raw luma.
    #[must_use]
    pub const fn new(frame_type: FrameType, width: u32, height: u32, luma: Vec<i16>) -> Self {
        Self {
            frame_type,
            width,
            height,
            luma,
            timestamp_ms: 0,
        }
    }

    /// Create a frame with all pixels set to a constant.
    #[must_use]
    pub fn constant(frame_type: FrameType, width: u32, height: u32, value: i16) -> Self {
        Self {
            frame_type,
            width,
            height,
            luma: vec![value; (width * height) as usize],
            timestamp_ms: 0,
        }
    }

    /// Pixel count.
    #[must_use]
    pub const fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }
}

/// GOP (Group of Pictures) structure definition.
#[derive(Debug, Clone)]
pub struct GopStructure {
    /// Pattern of frame types, e.g. [I, B, B, P, B, B, P]
    pub pattern: Vec<FrameType>,
}

impl GopStructure {
    /// Create an IBBP GOP with the specified number of B-frames between P-frames.
    #[must_use]
    pub fn ibbp(b_count: usize, p_count: usize) -> Self {
        let mut pattern = vec![FrameType::I];
        for _ in 0..p_count {
            for _ in 0..b_count {
                pattern.push(FrameType::B);
            }
            pattern.push(FrameType::P);
        }
        Self { pattern }
    }

    /// Create an I-only GOP of given length.
    #[must_use]
    pub fn intra_only(length: usize) -> Self {
        Self {
            pattern: vec![FrameType::I; length],
        }
    }

    /// Create an IP-only GOP.
    #[must_use]
    pub fn ip_only(length: usize) -> Self {
        let mut pattern = vec![FrameType::I];
        for _ in 1..length {
            pattern.push(FrameType::P);
        }
        Self { pattern }
    }

    /// GOP length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.pattern.len()
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pattern.is_empty()
    }

    /// Count of each frame type.
    #[must_use]
    pub fn frame_type_counts(&self) -> HashMap<FrameType, usize> {
        let mut counts = HashMap::new();
        for &ft in &self.pattern {
            *counts.entry(ft).or_insert(0) += 1;
        }
        counts
    }

    /// Get frame type at a given index in the stream (cyclic).
    #[must_use]
    pub fn frame_type_at(&self, index: usize) -> FrameType {
        self.pattern[index % self.pattern.len()]
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Zigzag scan
// ---------------------------------------------------------------------------

/// Zigzag scan order for an 8x8 block.
pub const ZIGZAG_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Reorder a block into zigzag order.
#[must_use]
pub fn zigzag_scan(block: &[i16; 64]) -> [i16; 64] {
    let mut result = [0_i16; 64];
    for (i, &idx) in ZIGZAG_ORDER.iter().enumerate() {
        result[i] = block[idx];
    }
    result
}

/// Inverse zigzag: from zigzag order back to block order.
#[must_use]
pub fn inverse_zigzag(zigzag: &[i16; 64]) -> [i16; 64] {
    let mut result = [0_i16; 64];
    for (i, &idx) in ZIGZAG_ORDER.iter().enumerate() {
        result[idx] = zigzag[i];
    }
    result
}

// ---------------------------------------------------------------------------
// Entropy Coding: Run-Length Encoding
// ---------------------------------------------------------------------------

/// A run-length encoded pair: (run of zeros, value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RlePair {
    pub zero_run: u16,
    pub value: i16,
}

/// Run-length encode a zigzag-scanned block.
/// Encodes runs of zeros followed by a nonzero value.
/// End-of-block is signaled by (0, 0).
#[must_use]
pub fn rle_encode(zigzag: &[i16; 64]) -> Vec<RlePair> {
    let mut result = Vec::new();
    let mut zero_count: u16 = 0;
    for &val in zigzag {
        if val == 0 {
            zero_count += 1;
        } else {
            result.push(RlePair {
                zero_run: zero_count,
                value: val,
            });
            zero_count = 0;
        }
    }
    // End-of-block marker
    result.push(RlePair {
        zero_run: 0,
        value: 0,
    });
    result
}

/// Decode RLE pairs back to a 64-element zigzag array.
#[must_use]
pub fn rle_decode(pairs: &[RlePair]) -> [i16; 64] {
    let mut result = [0_i16; 64];
    let mut pos = 0;
    for pair in pairs {
        if pair.zero_run == 0 && pair.value == 0 {
            break; // EOB
        }
        pos += pair.zero_run as usize;
        if pos < 64 {
            result[pos] = pair.value;
            pos += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Entropy Coding: Huffman (simplified)
// ---------------------------------------------------------------------------

/// A node in a Huffman tree.
#[derive(Debug, Clone)]
enum HuffmanNode {
    Leaf {
        symbol: u8,
        freq: u32,
    },
    Internal {
        freq: u32,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl HuffmanNode {
    const fn freq(&self) -> u32 {
        match self {
            Self::Leaf { freq, .. } | Self::Internal { freq, .. } => *freq,
        }
    }
}

/// A Huffman code table mapping symbols to bit strings.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    codes: HashMap<u8, Vec<bool>>,
}

impl HuffmanTable {
    /// Build a Huffman table from symbol frequencies.
    ///
    /// # Panics
    ///
    /// Panics if the internal node list becomes inconsistent (should not happen).
    #[must_use]
    pub fn build(frequencies: &HashMap<u8, u32>) -> Self {
        if frequencies.is_empty() {
            return Self {
                codes: HashMap::new(),
            };
        }

        if frequencies.len() == 1 {
            let mut codes = HashMap::new();
            for &sym in frequencies.keys() {
                codes.insert(sym, vec![false]);
            }
            return Self { codes };
        }

        let mut nodes: Vec<HuffmanNode> = frequencies
            .iter()
            .map(|(&symbol, &freq)| HuffmanNode::Leaf { symbol, freq })
            .collect();

        while nodes.len() > 1 {
            nodes.sort_by_key(|n| std::cmp::Reverse(n.freq()));
            let right = nodes.pop().unwrap();
            let left = nodes.pop().unwrap();
            nodes.push(HuffmanNode::Internal {
                freq: left.freq() + right.freq(),
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        let mut codes = HashMap::new();
        if let Some(root) = nodes.into_iter().next() {
            Self::build_codes(&root, &mut Vec::new(), &mut codes);
        }

        Self { codes }
    }

    fn build_codes(node: &HuffmanNode, prefix: &mut Vec<bool>, codes: &mut HashMap<u8, Vec<bool>>) {
        match node {
            HuffmanNode::Leaf { symbol, .. } => {
                codes.insert(*symbol, prefix.clone());
            }
            HuffmanNode::Internal { left, right, .. } => {
                prefix.push(false);
                Self::build_codes(left, prefix, codes);
                prefix.pop();
                prefix.push(true);
                Self::build_codes(right, prefix, codes);
                prefix.pop();
            }
        }
    }

    /// Encode a byte slice into a bit vector.
    #[must_use]
    pub fn encode(&self, data: &[u8]) -> Vec<bool> {
        let mut bits = Vec::new();
        for &byte in data {
            if let Some(code) = self.codes.get(&byte) {
                bits.extend_from_slice(code);
            }
        }
        bits
    }

    /// Decode a bit vector back to bytes.
    #[must_use]
    pub fn decode(&self, bits: &[bool]) -> Vec<u8> {
        let reverse: HashMap<Vec<bool>, u8> = self
            .codes
            .iter()
            .map(|(&sym, code)| (code.clone(), sym))
            .collect();

        let mut result = Vec::new();
        let mut current = Vec::new();
        for &bit in bits {
            current.push(bit);
            if let Some(&sym) = reverse.get(&current) {
                result.push(sym);
                current.clear();
            }
        }
        result
    }

    /// Number of symbols in the table.
    #[must_use]
    pub fn symbol_count(&self) -> usize {
        self.codes.len()
    }

    /// Get the code for a symbol.
    #[must_use]
    pub fn get_code(&self, symbol: u8) -> Option<&Vec<bool>> {
        self.codes.get(&symbol)
    }
}

// ---------------------------------------------------------------------------
// Motion Vector & Compensation
// ---------------------------------------------------------------------------

/// A 2D motion vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    pub dx: i16,
    pub dy: i16,
}

impl MotionVector {
    #[must_use]
    pub const fn new(dx: i16, dy: i16) -> Self {
        Self { dx, dy }
    }

    /// Squared magnitude.
    #[must_use]
    pub const fn magnitude_sq(&self) -> i32 {
        self.dx as i32 * self.dx as i32 + self.dy as i32 * self.dy as i32
    }

    /// Add two vectors.
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        Self {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
        }
    }

    /// Half-pixel interpolation vector for B-frame averaging.
    #[must_use]
    pub const fn half(self) -> Self {
        Self {
            dx: self.dx / 2,
            dy: self.dy / 2,
        }
    }
}

/// Block-based motion estimation using full search (brute force) on luma.
///
/// Returns the best motion vector for a block at `(bx, by)` with given `block_size`.
/// Search range is `[-search_range, search_range]`.
#[must_use]
pub fn full_search_motion_estimation(
    current: &[i16],
    reference: &[i16],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    block_size: u32,
    search_range: i16,
) -> MotionVector {
    let mut best_mv = MotionVector::new(0, 0);
    let mut best_sad = i64::MAX;

    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let sad = compute_sad(
                current, reference, width, height, bx, by, block_size, dx, dy,
            );
            if sad < best_sad {
                best_sad = sad;
                best_mv = MotionVector::new(dx, dy);
            }
        }
    }
    best_mv
}

/// Compute Sum of Absolute Differences for a block.
fn compute_sad(
    current: &[i16],
    reference: &[i16],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    block_size: u32,
    dx: i16,
    dy: i16,
) -> i64 {
    let mut sad: i64 = 0;
    for row in 0..block_size {
        for col in 0..block_size {
            let cx = bx + col;
            let cy = by + row;
            let rx = i32::from(dx) + cx as i32;
            let ry = i32::from(dy) + cy as i32;

            if cx >= width
                || cy >= height
                || rx < 0
                || ry < 0
                || rx >= width as i32
                || ry >= height as i32
            {
                sad += 255;
                continue;
            }

            let c_idx = (cy * width + cx) as usize;
            let r_idx = (ry as u32 * width + rx as u32) as usize;
            sad += i64::from((current[c_idx] - reference[r_idx]).abs());
        }
    }
    sad
}

/// Apply motion compensation: reconstruct a block from reference using a motion vector.
#[must_use]
pub fn motion_compensate_block(
    reference: &[i16],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    block_size: u32,
    mv: MotionVector,
) -> Vec<i16> {
    let mut block = vec![0_i16; (block_size * block_size) as usize];
    for row in 0..block_size {
        for col in 0..block_size {
            let rx = bx as i32 + col as i32 + i32::from(mv.dx);
            let ry = by as i32 + row as i32 + i32::from(mv.dy);
            let rx_c = rx.clamp(0, width as i32 - 1) as u32;
            let ry_c = ry.clamp(0, height as i32 - 1) as u32;
            block[(row * block_size + col) as usize] = reference[(ry_c * width + rx_c) as usize];
        }
    }
    block
}

/// Bidirectional motion compensation: average of forward and backward.
#[must_use]
pub fn bidir_compensate_block(
    ref_fwd: &[i16],
    ref_bwd: &[i16],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    block_size: u32,
    mv_fwd: MotionVector,
    mv_bwd: MotionVector,
) -> Vec<i16> {
    let fwd = motion_compensate_block(ref_fwd, width, height, bx, by, block_size, mv_fwd);
    let bwd = motion_compensate_block(ref_bwd, width, height, bx, by, block_size, mv_bwd);
    fwd.iter()
        .zip(bwd.iter())
        .map(|(&f, &b)| (i32::from(f) + i32::from(b)) as i16 / 2)
        .collect()
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Bitrate Control
// ---------------------------------------------------------------------------

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant Bitrate
    Cbr,
    /// Variable Bitrate
    Vbr,
    /// Constant Quality (CRF-like)
    Cq,
}

/// Bitrate controller state.
#[derive(Debug, Clone)]
pub struct BitrateController {
    pub mode: RateControlMode,
    pub target_bitrate_kbps: u32,
    pub fps: u32,
    pub bits_used: u64,
    pub frames_encoded: u32,
    pub quality: u8,
    pub min_quality: u8,
    pub max_quality: u8,
}

impl BitrateController {
    /// Create a new bitrate controller.
    #[must_use]
    pub const fn new(mode: RateControlMode, target_bitrate_kbps: u32, fps: u32) -> Self {
        Self {
            mode,
            target_bitrate_kbps,
            fps,
            bits_used: 0,
            frames_encoded: 0,
            quality: 50,
            min_quality: 10,
            max_quality: 95,
        }
    }

    /// Target bits per frame.
    #[must_use]
    pub fn target_bits_per_frame(&self) -> u64 {
        if self.fps == 0 {
            return 0;
        }
        u64::from(self.target_bitrate_kbps) * 1000 / u64::from(self.fps)
    }

    /// Average bitrate so far in kbps.
    #[must_use]
    pub fn average_bitrate_kbps(&self) -> u64 {
        if self.frames_encoded == 0 || self.fps == 0 {
            return 0;
        }
        let seconds = f64::from(self.frames_encoded) / f64::from(self.fps);
        if seconds <= 0.0 {
            return 0;
        }
        (self.bits_used as f64 / seconds / 1000.0) as u64
    }

    /// Report a frame's encoded size and adjust quality.
    pub fn report_frame(&mut self, bits: u64) {
        self.bits_used += bits;
        self.frames_encoded += 1;

        if self.mode == RateControlMode::Cq {
            return;
        }

        let target = self.target_bits_per_frame();
        if target == 0 {
            return;
        }

        if bits > target + target / 4 {
            // Too many bits: lower quality
            self.quality = self.quality.saturating_sub(2).max(self.min_quality);
        } else if bits < target.saturating_sub(target / 4) {
            // Too few bits: raise quality
            self.quality = (self.quality + 2).min(self.max_quality);
        }
    }

    /// Current quality value.
    #[must_use]
    pub const fn current_quality(&self) -> u8 {
        self.quality
    }
}

// ---------------------------------------------------------------------------
// Container Format Basics
// ---------------------------------------------------------------------------

/// Simple container box types (inspired by ISO BMFF / MP4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoxType {
    /// File type box
    Ftyp,
    /// Movie header
    Moov,
    /// Track
    Trak,
    /// Media data
    Mdat,
    /// Free space
    Free,
    /// Custom / unknown
    Custom([u8; 4]),
}

impl BoxType {
    /// 4-byte identifier.
    #[must_use]
    pub const fn fourcc(&self) -> [u8; 4] {
        match self {
            Self::Ftyp => *b"ftyp",
            Self::Moov => *b"moov",
            Self::Trak => *b"trak",
            Self::Mdat => *b"mdat",
            Self::Free => *b"free",
            Self::Custom(c) => *c,
        }
    }

    /// Parse from 4 bytes.
    #[must_use]
    pub const fn from_fourcc(cc: [u8; 4]) -> Self {
        match &cc {
            b"ftyp" => Self::Ftyp,
            b"moov" => Self::Moov,
            b"trak" => Self::Trak,
            b"mdat" => Self::Mdat,
            b"free" => Self::Free,
            _ => Self::Custom(cc),
        }
    }
}

/// A container box with type, size, and payload.
#[derive(Debug, Clone)]
pub struct ContainerBox {
    pub box_type: BoxType,
    pub payload: Vec<u8>,
}

impl ContainerBox {
    /// Create a new box.
    #[must_use]
    pub const fn new(box_type: BoxType, payload: Vec<u8>) -> Self {
        Self { box_type, payload }
    }

    /// Total size: 8 bytes header + payload.
    #[must_use]
    pub const fn total_size(&self) -> u32 {
        8 + self.payload.len() as u32
    }

    /// Serialize to bytes: [size:4][type:4][payload].
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let size = self.total_size();
        let mut bytes = Vec::with_capacity(size as usize);
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(&self.box_type.fourcc());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// Parse a box from bytes. Returns `(box, bytes_consumed)`.
    #[must_use]
    pub fn parse(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 8 {
            return None;
        }
        let size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < size || size < 8 {
            return None;
        }
        let fourcc = [data[4], data[5], data[6], data[7]];
        let box_type = BoxType::from_fourcc(fourcc);
        let payload = data[8..size].to_vec();
        Some((Self { box_type, payload }, size))
    }
}

/// A simple container file composed of boxes.
#[derive(Debug, Clone)]
pub struct ContainerFile {
    pub boxes: Vec<ContainerBox>,
}

impl ContainerFile {
    /// Create a new empty container.
    #[must_use]
    pub const fn new() -> Self {
        Self { boxes: Vec::new() }
    }

    /// Add a box.
    pub fn add_box(&mut self, b: ContainerBox) {
        self.boxes.push(b);
    }

    /// Serialize the entire file.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for b in &self.boxes {
            bytes.extend_from_slice(&b.serialize());
        }
        bytes
    }

    /// Parse boxes from bytes.
    #[must_use]
    pub fn parse(data: &[u8]) -> Self {
        let mut boxes = Vec::new();
        let mut offset = 0;
        while offset < data.len() {
            if let Some((b, consumed)) = ContainerBox::parse(&data[offset..]) {
                boxes.push(b);
                offset += consumed;
            } else {
                break;
            }
        }
        Self { boxes }
    }

    /// Total byte size.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.boxes.iter().map(|b| b.total_size() as usize).sum()
    }

    /// Find boxes by type.
    #[must_use]
    pub fn find_boxes(&self, box_type: &BoxType) -> Vec<&ContainerBox> {
        self.boxes
            .iter()
            .filter(|b| b.box_type == *box_type)
            .collect()
    }
}

impl Default for ContainerFile {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
