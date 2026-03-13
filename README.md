**English** | [日本語](README_JP.md)

# ALICE-Video

Video codec and processing library for the A.L.I.C.E. ecosystem. Implements core video pipeline components in pure Rust.

## Features

- **Pixel Formats** — YUV 4:2:0 planar and RGB interleaved with bidirectional conversion
- **Frame Types** — I-frame, P-frame, B-frame classification with GOP structure
- **Motion Compensation** — Block matching, motion vector estimation and prediction
- **DCT & Quantization** — 8x8 Discrete Cosine Transform with configurable quantization tables
- **Entropy Coding** — Variable-length encoding for compressed bitstream output
- **Resolution Scaling** — Bilinear and nearest-neighbor resampling
- **Bitrate Control** — CBR/VBR rate control with quality targeting
- **Container Format** — Basic container muxing/demuxing support

## Architecture

```
Raw Frames (RGB/YUV)
  │
  ├── pixel       — Format conversion (YUV↔RGB)
  ├── frame       — I/P/B frame classification, GOP
  ├── motion      — Block matching, motion vectors
  ├── dct         — DCT transform, quantization
  ├── entropy     — Variable-length coding
  ├── scale       — Resolution resampling
  ├── bitrate     — Rate control (CBR/VBR)
  └── container   — Mux/demux
```

## Usage

```rust
use alice_video::{RgbPixel, YuvPixel};

let rgb = RgbPixel::new(255, 128, 64);
let yuv = rgb.to_yuv();
let back = yuv.to_rgb();
```

## License

MIT OR Apache-2.0
