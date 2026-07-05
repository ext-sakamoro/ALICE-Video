//! Motion vector + compensation (`MotionVector` / `full_search_motion_estimation` / `motion_compensate_block` / `bidir_compensate_block`).

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
