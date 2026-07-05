//! Entropy coding: Run-Length Encoding (`RlePair` / `rle_encode` / `rle_decode`).

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
