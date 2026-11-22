//! Inkling text tokenizer.
//!
//! The file is the Hugging Face `tokenizer.json` shipped with
//! `tiny-random/inkling-2` at revision `0a57c72e4e5b68aafe1d983dbb62980344a5fcd7`
//! (sha256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`).
//! That is the same fast tokenizer transformers 5.24.0 loads for the Inkling
//! chat models. Encoding does not add special tokens that were not already in
//! the string.

use anyhow::{Context, Result};
use std::path::Path;

pub struct InklingTokenizer {
    inner: tokenizers::Tokenizer,
}

impl InklingTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("load Inkling tokenizer {}", path.display()))?;
        Ok(Self { inner })
    }

    /// Token ids for `text`, without a leading special token.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("encode: {err}"))?;
        Ok(encoded.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|err| anyhow::anyhow!("decode: {err}"))
    }
}
