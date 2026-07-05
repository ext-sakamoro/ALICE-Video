//! Zigzag scan (`ZIGZAG_ORDER` / `zigzag_scan` / `inverse_zigzag`).

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
