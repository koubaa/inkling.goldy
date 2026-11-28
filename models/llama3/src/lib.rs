//! llama3.cuda TinyStories generator on Ammon / Goldy (CUDA / Metal).

pub mod bench;
pub mod checkpoint;
pub mod tokenizer;

#[cfg(feature = "gpu")]
pub mod generate;
#[cfg(feature = "gpu")]
pub mod model;

#[cfg(feature = "gpu")]
pub use checkpoint::LayerWeights;
pub use checkpoint::{
    checkpoint_file_bytes, Checkpoint, Config, LayerWeightOffsets, ModelShape, WeightLayout,
    LLAMA3_CUDA_COMMIT, LLAMA_CPP_COMMIT, STORIES110M_SHA256, STORIES15M_SHA256, STORIES42M_SHA256,
    TOKENIZER_SHA256,
};
pub use tokenizer::{apply_dream_prompt_patch, printable_piece, Tokenizer};
