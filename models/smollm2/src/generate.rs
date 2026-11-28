//! Greedy generation from a text prompt.

use crate::model::Model;
use ammon::{sample_argmax, HfTokenizer};
use anyhow::{ensure, Result};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Generation {
    pub prompt_ids: Vec<u32>,
    pub tokens: Vec<u32>,
    pub text: String,
    /// Generated tokens per second after the first, which carries the prompt.
    pub tokens_per_second: f64,
}

/// Feed `prompt_ids` from position 0, then append up to `max_new_tokens` argmax
/// tokens. Stops early at `stop` or when the cache is full.
pub fn greedy(
    model: &mut Model,
    prompt_ids: &[u32],
    max_new_tokens: u32,
    stop: Option<u32>,
) -> Result<(Vec<u32>, f64)> {
    ensure!(!prompt_ids.is_empty(), "the prompt has no tokens");
    let last = prompt_ids.len() - 1;
    for (pos, &token) in prompt_ids[..last].iter().enumerate() {
        model.advance(token, pos as u32)?;
    }
    let mut tokens = Vec::new();
    let mut token = prompt_ids[last];
    let mut pos = last as u32;
    let mut started = None;
    while (tokens.len() as u32) < max_new_tokens && pos < model.seq_len() {
        let logits = model.step_view(token, pos)?;
        token = sample_argmax(&logits) as u32;
        drop(logits);
        if stop == Some(token) {
            break;
        }
        tokens.push(token);
        pos += 1;
        started.get_or_insert_with(Instant::now);
    }
    let rate = match started {
        Some(t) if tokens.len() > 1 => (tokens.len() - 1) as f64 / t.elapsed().as_secs_f64(),
        _ => 0.0,
    };
    Ok((tokens, rate))
}

pub fn generate(
    model: &mut Model,
    tokenizer: &HfTokenizer,
    prompt: &str,
    max_new_tokens: u32,
) -> Result<Generation> {
    let prompt_ids = tokenizer.encode(prompt)?;
    let stop = model.config.eos_token;
    let (tokens, tokens_per_second) = greedy(model, &prompt_ids, max_new_tokens, stop)?;
    let text = tokenizer.decode(&tokens)?;
    Ok(Generation {
        prompt_ids,
        tokens,
        text,
        tokens_per_second,
    })
}
