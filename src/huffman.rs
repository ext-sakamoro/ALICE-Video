//! Entropy coding: Huffman (simplified) (`HuffmanTable`).

use std::collections::HashMap;

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
