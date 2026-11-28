//! Inkling 2 text decoder on Ammon and Goldy.
//!
//! The Python harness in `tools/ref` is the parity target. This crate owns the
//! checkpoint name map, the decoder layer, and the only depth-tape schedule.
//! Ammon stays model-agnostic.

pub mod names;
pub mod packed;
pub mod tape;
pub mod tokenizer;

#[cfg(feature = "gpu")]
pub mod generate;
#[cfg(feature = "gpu")]
pub mod layer;
#[cfg(feature = "gpu")]
pub mod model;

/// Path of the reference harness, relative to the crate root.
pub const REFERENCE_DIR: &str = "tools/ref";
