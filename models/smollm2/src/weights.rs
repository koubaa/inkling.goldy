//! Host packing of a Hugging Face Llama checkpoint onto f32 weight slabs.
//!
//! The checkpoint stores BF16. Each tensor is widened to f32 (exact) and packed
//! in checkpoint order onto slabs of at most [`SLAB_ELEMENTS`] values, so every
//! slab binds within Vulkan's storage-buffer range on lavapipe (128 MiB). A
//! tensor never straddles two slabs.
//!
//! Hugging Face Llama rotates each head's halves (`rotate_half`): dimension `i`
//! pairs with `i + head / 2`. Ammon rotates adjacent pairs `(2i, 2i + 1)` at
//! the same frequency. Interleaving the rows of `q_proj` and `k_proj` per head
//! maps one onto the other, and since Q and K move together their dot products
//! are unchanged. V and the output projection keep their order.

use crate::config::Config;
use crate::layer::LayerSpec;
use ammon::checkpoint::MappedCheckpoint;
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;

/// Largest slab, in f32 values: 128 MiB.
pub const SLAB_ELEMENTS: u32 = 1 << 25;

/// Where one tensor lives: a slab, an element offset into it, and its shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSlice {
    pub slab: usize,
    pub offset: u32,
    pub dims: Vec<u32>,
}

impl WeightSlice {
    pub fn numel(&self) -> u32 {
        self.dims.iter().product()
    }

    #[cfg(feature = "gpu")]
    pub fn view<'a>(&self, slabs: &'a [goldy::Tensor]) -> Result<goldy::TensorView<'a>> {
        let slab = slabs
            .get(self.slab)
            .with_context(|| format!("weight slab {} is not uploaded", self.slab))?;
        let shape = goldy::TensorShape::from_dims(&self.dims)?;
        Ok(goldy::TensorView::packed_at(
            slab.buffer(),
            goldy::TensorDType::F32,
            u64::from(self.offset),
            shape,
        )?)
    }
}

/// Packs tensors onto slabs of at most `limit` values each.
pub struct SlabBuilder {
    limit: u32,
    slabs: Vec<Vec<f32>>,
}

impl SlabBuilder {
    pub fn new(limit: u32) -> Self {
        Self {
            limit,
            slabs: Vec::new(),
        }
    }

    pub fn push(&mut self, values: &[f32], dims: &[u32]) -> Result<WeightSlice> {
        let numel: u64 = dims.iter().map(|&d| u64::from(d)).product();
        ensure!(
            numel == values.len() as u64,
            "tensor {dims:?} needs {numel} values, got {}",
            values.len()
        );
        ensure!(
            numel <= u64::from(self.limit),
            "tensor {dims:?} ({numel} values) exceeds the {}-value slab",
            self.limit
        );
        let fits = self
            .slabs
            .last()
            .is_some_and(|slab| slab.len() as u64 + numel <= u64::from(self.limit));
        if !fits {
            self.slabs.push(Vec::new());
        }
        let slab = self.slabs.len() - 1;
        let current = &mut self.slabs[slab];
        let offset = current.len() as u32;
        current.extend_from_slice(values);
        Ok(WeightSlice {
            slab,
            offset,
            dims: dims.to_vec(),
        })
    }

    pub fn finish(self) -> Vec<Vec<f32>> {
        self.slabs
    }
}

/// Interleave each head's rows from `[first half | second half]` into
/// adjacent pairs: output row `2i` is input row `i`, row `2i + 1` is row
/// `i + head_dim / 2`.
pub fn interleave_rope_rows(rows: &[f32], heads: u32, head_dim: u32, cols: u32) -> Vec<f32> {
    let (heads, head_dim, cols) = (heads as usize, head_dim as usize, cols as usize);
    assert_eq!(rows.len(), heads * head_dim * cols);
    let half = head_dim / 2;
    let mut out = vec![0.0; rows.len()];
    for head in 0..heads {
        let base = head * head_dim;
        for i in 0..half {
            for (dst, src) in [(2 * i, i), (2 * i + 1, i + half)] {
                let to = (base + dst) * cols;
                let from = (base + src) * cols;
                out[to..to + cols].copy_from_slice(&rows[from..from + cols]);
            }
        }
    }
    out
}

/// Every tensor's slice. `lm_head` is `embed` when the embeddings are tied.
#[derive(Clone, Debug)]
pub struct WeightMap {
    pub embed: WeightSlice,
    pub layers: Vec<LayerSpec>,
    pub final_norm: WeightSlice,
    pub lm_head: WeightSlice,
}

/// A checkpoint widened to f32 and packed, ready to upload.
pub struct HostWeights {
    pub config: Config,
    pub slabs: Vec<Vec<f32>>,
    pub map: WeightMap,
}

impl HostWeights {
    /// `config.json` and `model.safetensors` from `dir`.
    pub fn read_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::read_path(dir.join("config.json"))?;
        let path = dir.join("model.safetensors");
        let checkpoint = MappedCheckpoint::open(&path)
            .with_context(|| format!("open checkpoint {}", path.display()))?;
        Self::pack(config, &checkpoint, SLAB_ELEMENTS)
    }

    pub fn pack(config: Config, checkpoint: &MappedCheckpoint, limit: u32) -> Result<Self> {
        config.validate()?;
        let mut builder = SlabBuilder::new(limit);
        let (hidden, inter, kv) = (config.hidden, config.intermediate, config.kv_dim());
        let mut take = |name: &str, dims: &[u32], rope_heads: Option<u32>| -> Result<WeightSlice> {
            let tensor = checkpoint.get(name)?;
            let want: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            if tensor.shape != want {
                bail!("`{name}` is {:?}, expected {dims:?}", tensor.shape);
            }
            let mut values = checkpoint.read_f32(name)?;
            if let Some(heads) = rope_heads {
                if !config.rope_interleaved {
                    values = interleave_rope_rows(&values, heads, config.head_dim, dims[1]);
                }
            }
            builder.push(&values, dims)
        };

        let embed = take("model.embed_tokens.weight", &[config.vocab, hidden], None)?;
        let mut layers = Vec::with_capacity(config.layers as usize);
        for layer in 0..config.layers {
            let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
            layers.push(LayerSpec {
                attn_norm: take(&name("input_layernorm.weight"), &[hidden], None)?,
                query: take(
                    &name("self_attn.q_proj.weight"),
                    &[hidden, hidden],
                    Some(config.q_heads),
                )?,
                key: take(
                    &name("self_attn.k_proj.weight"),
                    &[kv, hidden],
                    Some(config.kv_heads),
                )?,
                value: take(&name("self_attn.v_proj.weight"), &[kv, hidden], None)?,
                output: take(&name("self_attn.o_proj.weight"), &[hidden, hidden], None)?,
                mlp_norm: take(&name("post_attention_layernorm.weight"), &[hidden], None)?,
                gate: take(&name("mlp.gate_proj.weight"), &[inter, hidden], None)?,
                up: take(&name("mlp.up_proj.weight"), &[inter, hidden], None)?,
                down: take(&name("mlp.down_proj.weight"), &[hidden, inter], None)?,
            });
        }
        let final_norm = take("model.norm.weight", &[hidden], None)?;
        let lm_head = if config.tie_word_embeddings {
            embed.clone()
        } else {
            take("lm_head.weight", &[config.vocab, hidden], None)?
        };
        Ok(Self {
            config,
            slabs: builder.finish(),
            map: WeightMap {
                embed,
                layers,
                final_norm,
                lm_head,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaving_pairs_each_heads_halves() {
        // Two heads of four, one column: rows [a0 a1 a2 a3 | b0 b1 b2 b3].
        let rows = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let out = interleave_rope_rows(&rows, 2, 4, 1);
        assert_eq!(out, [0.0, 2.0, 1.0, 3.0, 10.0, 12.0, 11.0, 13.0]);
    }

    /// The interleaved rows under adjacent-pair RoPE give the same scores as the
    /// stored rows under `rotate_half`.
    #[test]
    fn interleaved_pair_rope_matches_rotate_half_scores() {
        let (head, theta, pos) = (8usize, 100_000.0f64, 13.0f64);
        let q: Vec<f64> = (0..head).map(|i| (i as f64 * 0.37).sin()).collect();
        let k: Vec<f64> = (0..head).map(|i| (i as f64 * 0.91).cos()).collect();
        let angle = |pair: usize| pos * theta.powf(-((2 * pair) as f64) / head as f64);
        let rotate_half = |v: &[f64]| {
            let half = head / 2;
            let mut out = vec![0.0; head];
            for i in 0..half {
                let (c, s) = (angle(i).cos(), angle(i).sin());
                out[i] = v[i] * c - v[i + half] * s;
                out[i + half] = v[i + half] * c + v[i] * s;
            }
            out
        };
        let adjacent = |v: &[f64]| {
            let mut out = vec![0.0; head];
            for i in 0..head / 2 {
                let (c, s) = (angle(i).cos(), angle(i).sin());
                out[2 * i] = v[2 * i] * c - v[2 * i + 1] * s;
                out[2 * i + 1] = v[2 * i] * s + v[2 * i + 1] * c;
            }
            out
        };
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let widen = |v: Vec<f32>| v.into_iter().map(f64::from).collect::<Vec<_>>();
        let qi = widen(interleave_rope_rows(&as_f32(&q), 1, head as u32, 1));
        let ki = widen(interleave_rope_rows(&as_f32(&k), 1, head as u32, 1));
        let (q, k) = (widen(as_f32(&q)), widen(as_f32(&k)));
        let want = dot(&rotate_half(&q), &rotate_half(&k));
        let got = dot(&adjacent(&qi), &adjacent(&ki));
        assert!((want - got).abs() < 1e-12, "{want} vs {got}");
    }

    #[test]
    fn slabs_split_without_straddling() {
        let mut builder = SlabBuilder::new(10);
        let a = builder.push(&[1.0; 6], &[2, 3]).unwrap();
        let b = builder.push(&[2.0; 4], &[4]).unwrap();
        let c = builder.push(&[3.0; 2], &[2]).unwrap();
        assert!(builder.push(&[0.0; 11], &[11]).is_err());
        assert_eq!((a.slab, a.offset), (0, 0));
        assert_eq!((b.slab, b.offset), (0, 6));
        assert_eq!((c.slab, c.offset), (1, 0));
        let slabs = builder.finish();
        assert_eq!(slabs.len(), 2);
        assert_eq!(slabs[0].len(), 10);
        assert_eq!(slabs[1], [3.0, 3.0]);
    }
}
