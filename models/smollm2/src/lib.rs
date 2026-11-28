//! SmolLM2-135M on Ammon and Goldy: a Hugging Face Llama checkpoint with
//! grouped-query attention, RoPE base 100000, and tied embeddings.

pub mod config;
pub mod layer;
pub mod weights;

#[cfg(feature = "gpu")]
pub mod generate;
#[cfg(feature = "gpu")]
pub mod model;

pub use config::{Config, SMOLLM2_135M_CONFIG};
pub use layer::LayerSpec;
pub use weights::{interleave_rope_rows, HostWeights, SlabBuilder, WeightMap, WeightSlice};

/// Hugging Face repository and revision `tools/fetch_assets.py` pins.
pub const REPO: &str = "HuggingFaceTB/SmolLM2-135M";
pub const REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";

/// `assets/` beside this crate's manifest, where `tools/fetch_assets.py` writes.
pub fn assets_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets")
}
