//! ALICE-Video: Video codec and processing library.
//!
//! Provides frame types (I/P/B), GOP structure, motion compensation,
//! DCT/quantization, entropy coding, pixel formats (YUV420, RGB),
//! resolution scaling, bitrate control, and container format basics.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::wildcard_imports,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    clippy::float_cmp
)]

pub mod bitrate;
pub mod codec;
pub mod container;
pub mod dct;
pub mod frame;
pub mod huffman;
pub mod image;
pub mod motion;
pub mod pixel_format;
pub mod prelude;
pub mod quantization;
pub mod rle;
pub mod scaling;
pub mod zigzag;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::bitrate::*;
pub use crate::codec::*;
pub use crate::container::*;
pub use crate::dct::*;
pub use crate::frame::*;
pub use crate::huffman::*;
pub use crate::image::*;
pub use crate::motion::*;
pub use crate::pixel_format::*;
pub use crate::quantization::*;
pub use crate::rle::*;
pub use crate::scaling::*;
pub use crate::zigzag::*;
