//! The `config.json` fields a Llama checkpoint needs, checked against what this
//! crate records: SiLU-gated MLPs, no biases, no RoPE scaling.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub max_positions: u32,
    pub rope_theta: f32,
    /// Whether the checkpoint's Q/K rows already rotate adjacent pairs. Hugging
    /// Face Llama checkpoints rotate halves (`rotate_half`) and say `false`.
    pub rope_interleaved: bool,
    pub rms_eps: f32,
    pub tie_word_embeddings: bool,
    pub eos_token: Option<u32>,
}

impl Config {
    pub fn read_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read model config {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("model config {}", path.display()))
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(text).context("parse config.json")?;
        let model_type = json["model_type"].as_str().unwrap_or_default();
        if model_type != "llama" {
            bail!("model_type {model_type:?} is not llama");
        }
        let act = json["hidden_act"].as_str().unwrap_or("silu");
        if act != "silu" {
            bail!("hidden_act {act:?} is not silu");
        }
        for flag in ["attention_bias", "mlp_bias"] {
            if json[flag].as_bool().unwrap_or(false) {
                bail!("{flag} is set; this decoder has no projection biases");
            }
        }
        if !json["rope_scaling"].is_null() {
            bail!("rope_scaling is not supported");
        }
        let hidden = field(&json, "hidden_size")?;
        let q_heads = field(&json, "num_attention_heads")?;
        let kv_heads = match json.get("num_key_value_heads") {
            Some(Value::Null) | None => q_heads,
            Some(_) => field(&json, "num_key_value_heads")?,
        };
        let head_dim = match json.get("head_dim") {
            Some(Value::Null) | None => hidden / q_heads.max(1),
            Some(_) => field(&json, "head_dim")?,
        };
        let config = Self {
            hidden,
            intermediate: field(&json, "intermediate_size")?,
            layers: field(&json, "num_hidden_layers")?,
            q_heads,
            kv_heads,
            head_dim,
            vocab: field(&json, "vocab_size")?,
            max_positions: field(&json, "max_position_embeddings")?,
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            rope_interleaved: json["rope_interleaved"].as_bool().unwrap_or(false),
            rms_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            tie_word_embeddings: json["tie_word_embeddings"].as_bool().unwrap_or(false),
            eos_token: json["eos_token_id"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok()),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let nonzero = [
            self.hidden,
            self.intermediate,
            self.layers,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.vocab,
            self.max_positions,
        ];
        if nonzero.contains(&0) {
            bail!("config has zero widths: {self:?}");
        }
        if !self.q_heads.is_multiple_of(self.kv_heads) {
            bail!(
                "{} query heads do not group over {} kv heads",
                self.q_heads,
                self.kv_heads
            );
        }
        if self.q_heads * self.head_dim != self.hidden {
            bail!(
                "{} heads of {} do not span hidden {}",
                self.q_heads,
                self.head_dim,
                self.hidden
            );
        }
        if !self.head_dim.is_multiple_of(2) {
            bail!("head_dim {} is odd; RoPE rotates pairs", self.head_dim);
        }
        Ok(())
    }

    pub fn kv_dim(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

fn field(json: &Value, name: &str) -> Result<u32> {
    let value = json[name]
        .as_u64()
        .with_context(|| format!("config.json has no integer {name}"))?;
    u32::try_from(value).with_context(|| format!("{name} = {value} does not fit u32"))
}

/// `config.json` of HuggingFaceTB/SmolLM2-135M at the pinned revision.
pub const SMOLLM2_135M_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "vocab_size": 49152
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_smollm2_135m() {
        let config = Config::from_json(SMOLLM2_135M_CONFIG).unwrap();
        assert_eq!(
            config,
            Config {
                hidden: 576,
                intermediate: 1536,
                layers: 30,
                q_heads: 9,
                kv_heads: 3,
                head_dim: 64,
                vocab: 49152,
                max_positions: 8192,
                rope_theta: 100_000.0,
                rope_interleaved: false,
                rms_eps: 1e-5,
                tie_word_embeddings: true,
                eos_token: Some(0),
            }
        );
        assert_eq!(config.kv_dim(), 192);
    }

    #[test]
    fn rejects_what_the_graph_does_not_record() {
        let with = |from: &str, to: &str| {
            let text = SMOLLM2_135M_CONFIG.replace(from, to);
            assert_ne!(text, SMOLLM2_135M_CONFIG, "{from}");
            Config::from_json(&text)
        };
        assert!(with("\"num_key_value_heads\": 3", "\"num_key_value_heads\": 2").is_err());
        assert!(with("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"").is_err());
        assert!(with("\"attention_bias\": false", "\"attention_bias\": true").is_err());
        assert!(with(
            "\"rope_scaling\": null",
            "\"rope_scaling\": {\"type\": \"linear\"}"
        )
        .is_err());
        assert!(with("\"model_type\": \"llama\"", "\"model_type\": \"mistral\"").is_err());
    }
}
