//! Prefill, then greedy decode.
//!
//! Prefill feeds the prompt tokens and keeps the logits of the last one.
//! Each new token is the argmax of those logits. The argmax matches
//! transformers' greedy `generate` (`do_sample=False`): the first maximum,
//! ties broken toward the lower index.

use crate::model::TextModel;
use anyhow::Result;

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
    /// `top1 - top2` at each new token.
    pub margins: Vec<f32>,
    /// Logits at the last prompt position.
    pub last_logits: Vec<f32>,
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
///
/// The loop lives in [`ammon::greedy_decode`].
pub fn greedy(model: &mut TextModel, prompt: &[u32], max_new: u32) -> Result<GreedyDecode> {
    let report = ammon::greedy_decode(model, prompt, max_new)?;
    Ok(GreedyDecode {
        prompt: prompt.to_vec(),
        new_tokens: report.new_tokens,
        prefill_secs: report.prefill_secs,
        decode_secs: report.decode_secs,
        decode_steps: report.decode_steps,
        step_secs: report.step_secs,
        margins: report.margins,
        last_logits: report.last_logits,
    })
}
