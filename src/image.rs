//! YUV420 + RGB image containers (`Yuv420Image` / `RgbImage`).

use crate::pixel_format::RgbPixel;

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
