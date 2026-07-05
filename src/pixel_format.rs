//! Pixel formats (`PixelFormat` / `YuvPixel` / `RgbPixel`).

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
