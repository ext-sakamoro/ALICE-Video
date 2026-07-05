//! Container format basics (`BoxType` / `ContainerBox` / `ContainerFile`).

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

    /// Serialize to bytes: `[size:4][type:4][payload]`.
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
