//! Inkling 2 and Inkling 2 Mini text decoders.
//!
//! The Python reference harness in `tools/ref` is the parity target: random
//! FP32 weights, eager transformers forward, token-by-token decode. Ammon and
//! Goldy are not linked yet. There is no vision tower and no multi-token
//! prediction head.

/// Path of the reference harness, relative to the crate root.
pub const REFERENCE_DIR: &str = "tools/ref";
