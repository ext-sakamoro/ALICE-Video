//! Frame types + GOP (`FrameType` / `VideoFrame` / `GopStructure`).

use std::collections::HashMap;

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
