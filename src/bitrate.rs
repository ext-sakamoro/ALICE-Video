//! Bitrate control (`RateControlMode` / `BitrateController`).

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
