//! Prefill, then greedy decode.
//!
//! Prefill feeds the prompt tokens and keeps the logits of the last one.
//! Each new token is the argmax of those logits. The argmax matches
//! transformers' greedy `generate` (`do_sample=False`): the first maximum,
//! ties broken toward the lower index.

use crate::model::TextModel;
use ammon::sample_argmax;
use anyhow::{ensure, Result};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct GreedyDecode {
    pub prompt: Vec<u32>,
    pub new_tokens: Vec<u32>,
    /// Wall time of the prompt forwards.
    pub prefill_secs: f64,
    /// Wall time of the forwards after the prompt. The first new token comes
    /// from the prefill logits, so this covers `new_tokens.len().saturating_sub(1)` steps.
    pub decode_secs: f64,
    pub decode_steps: u32,
    /// One entry per forward, prompt first. The first entry includes pipeline compile.
    pub step_secs: Vec<f64>,
}

impl GreedyDecode {
    pub fn prefill_tokens_per_sec(&self) -> f64 {
        per_sec(self.prompt.len() as f64, self.prefill_secs)
    }

    pub fn decode_tokens_per_sec(&self) -> f64 {
        per_sec(f64::from(self.decode_steps), self.decode_secs)
    }

    pub fn decode_latency_secs(&self) -> f64 {
        if self.decode_steps == 0 {
            0.0
        } else {
            self.decode_secs / f64::from(self.decode_steps)
        }
    }
}

fn per_sec(count: f64, secs: f64) -> f64 {
    if secs > 0.0 {
        count / secs
    } else {
        0.0
    }
}

/// Teacher-force `prompt`, then append `max_new` greedy tokens.
pub fn greedy(model: &mut TextModel, prompt: &[u32], max_new: u32) -> Result<GreedyDecode> {
    ensure!(!prompt.is_empty(), "greedy decode needs a prompt token");
    let mut step_secs = Vec::with_capacity(prompt.len() + max_new as usize);
    let started = Instant::now();
    let step_started = Instant::now();
    let mut logits = model.step(prompt[0], 0)?;
    step_secs.push(step_started.elapsed().as_secs_f64());
    for (index, token) in prompt.iter().enumerate().skip(1) {
        let step_started = Instant::now();
        logits = model.step(*token, index as u32)?;
        step_secs.push(step_started.elapsed().as_secs_f64());
    }
    let prefill_secs = started.elapsed().as_secs_f64();

    let mut new_tokens = Vec::with_capacity(max_new as usize);
    let mut pos = prompt.len() as u32;
    let decode_started = Instant::now();
    let mut decode_steps = 0u32;
    for step in 0..max_new {
        let next = sample_argmax(&logits);
        ensure!(next >= 0, "argmax returned {next}");
        let next = next as u32;
        new_tokens.push(next);
        if step + 1 == max_new {
            break;
        }
        let step_started = Instant::now();
        logits = model.step(next, pos)?;
        step_secs.push(step_started.elapsed().as_secs_f64());
        pos += 1;
        decode_steps += 1;
    }
    let decode_secs = decode_started.elapsed().as_secs_f64();
    Ok(GreedyDecode {
        prompt: prompt.to_vec(),
        new_tokens,
        prefill_secs,
        decode_secs,
        decode_steps,
        step_secs,
    })
}
