//! SmolLM2 decode on one retained Goldy scheme per token.
//!
//! Ammon records named groups (`embed`, `layerN/attn`, `layerN/ffn`, `tail`)
//! through one shared module set: every layer uses the same
//! [`CausalAttentionBlock`] and [`SwiGluBlock`], fed that layer's views of the
//! weight slabs. Persistent state is the residual `x`, the logits, the
//! `DecodeStep` parcel, and Ammon's [`KvCache`] (one key and one value tensor
//! across all layers). Everything else is scheme-scoped temporary scratch.

use crate::config::Config;
use crate::weights::{HostWeights, WeightMap, SLAB_ELEMENTS};
use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::{
    configure_inference_scheme, AutoregressiveModel, CausalAttentionBlock, Embedding, KvCache,
    Linear, LogitsView, RmsNorm, SwiGluBlock,
};
use anyhow::{ensure, Context, Result};
use goldy::{
    Buffer, DepositTransaction, FusionReport, HostSink, MemoryExchange, ReplayStats, Runtime,
    Scheme, Tensor, TensorDType, TensorShape,
};
use std::path::Path;

/// Positions the KV cache holds. Attention cost per token grows with the
/// positions decoded so far, not with this capacity.
pub const DEFAULT_SEQ_LEN: u32 = 2048;

/// Device state the recorded scheme binds besides the weights.
struct DecodeState {
    step: Buffer,
    x: Tensor,
    logits: Tensor,
    cache: KvCache,
}

/// The shared module set. Modules hold prepared kernels and configuration;
/// recording reuses them for every layer.
struct DecoderModules {
    embedding: Embedding,
    attention: CausalAttentionBlock,
    mlp: SwiGluBlock,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl DecoderModules {
    fn new(runtime: &Runtime, config: &Config, seq_len: u32) -> Result<Self> {
        Ok(Self {
            embedding: Embedding::new(runtime)?,
            attention: CausalAttentionBlock::with_eps(
                runtime,
                config.hidden,
                config.q_heads,
                seq_len,
                config.rms_eps,
            )?
            .with_rope_theta(config.rope_theta)?,
            mlp: SwiGluBlock::with_eps(
                runtime,
                config.hidden,
                config.intermediate,
                config.rms_eps,
            )?,
            final_norm: RmsNorm::with_eps(runtime, config.hidden, config.rms_eps)?,
            lm_head: Linear::new(runtime)?,
        })
    }
}

pub struct Model {
    pub config: Config,
    seq_len: u32,
    _slabs: Vec<Tensor>,
    _state: DecodeState,
    _modules: DecoderModules,
    worker: Scheme,
    deposit: DepositTransaction,
    logits_sink: HostSink,
}

impl Model {
    /// Read `config.json` and `model.safetensors` from `dir` and load them on a
    /// new runtime.
    pub fn load_dir(dir: impl AsRef<Path>, seq_len: u32) -> Result<Self> {
        let weights = HostWeights::read_dir(dir)?;
        let runtime = create_runtime()?;
        Self::load_on(&runtime, &weights, seq_len)
    }

    pub fn load_on(runtime: &Runtime, weights: &HostWeights, seq_len: u32) -> Result<Self> {
        let config = weights.config;
        ensure!(
            seq_len > 0 && seq_len <= config.max_positions,
            "seq_len {seq_len} must be in 1..={}",
            config.max_positions
        );
        let cache_values =
            u64::from(config.layers) * u64::from(seq_len) * u64::from(config.kv_dim());
        ensure!(
            cache_values <= u64::from(SLAB_ELEMENTS),
            "a {seq_len}-position KV cache needs {cache_values} values per tensor, over the \
             {SLAB_ELEMENTS}-value storage-buffer slab"
        );
        let ctx = runtime.create_context().context("create GPU context")?;
        let slabs = weights
            .slabs
            .iter()
            .map(|values| {
                Tensor::from_f32(runtime, TensorShape::vector(values.len() as u32), values)
                    .context("upload weight slab")
            })
            .collect::<Result<Vec<_>>>()?;
        let state = DecodeState {
            step: DecodeStep::parcel(runtime)?,
            x: Tensor::zeros(
                runtime,
                TensorShape::vector(config.hidden),
                TensorDType::F32,
            )?,
            logits: Tensor::zeros(runtime, TensorShape::vector(config.vocab), TensorDType::F32)?,
            cache: KvCache::new(
                runtime,
                config.layers,
                seq_len,
                config.kv_heads,
                config.head_dim,
            )?,
        };
        let modules = DecoderModules::new(runtime, &config, seq_len)?;

        let mut worker = Scheme::new(&ctx);
        configure_inference_scheme(&mut worker);
        // Fused decode is faster once its kernels compile; `GOLDY_FUSION=0` opts out.
        if std::env::var_os("GOLDY_FUSION").is_none() {
            worker.set_automatic_fusion(true);
        }
        let exchange = MemoryExchange::new(&ctx);
        let deposit =
            exchange.bind_deposit(&mut worker, DecodeStep::deposit_target(&state.step))?;
        record(&mut worker, &modules, &state, &slabs, &weights.map, &config)?;
        let logits_sink = exchange.bind_host_sink(&mut worker, state.logits.buffer())?;

        Ok(Self {
            config,
            seq_len,
            _slabs: slabs,
            _state: state,
            _modules: modules,
            worker,
            deposit,
            logits_sink,
        })
    }

    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    fn submit(&mut self, token: u32, pos: u32) -> Result<goldy::Submission> {
        ensure!(
            pos < self.seq_len,
            "position {pos} is past the {}-position cache",
            self.seq_len
        );
        ensure!(
            token < self.config.vocab,
            "token {token} is out of vocabulary"
        );
        (&self.deposit
            << &DecodeStep {
                token,
                position: pos,
            })?;
        Ok(self.worker.submit()?)
    }

    pub fn advance(&mut self, token: u32, pos: u32) -> Result<()> {
        self.submit(token, pos)?;
        Ok(())
    }

    pub fn step_view(&mut self, token: u32, pos: u32) -> Result<LogitsView> {
        let mut submission = self.submit(token, pos)?;
        // The sink copy is part of the retained scheme, ordered after the LM head.
        let logits = (&mut submission >> &self.logits_sink).take::<f32>()?;
        Ok(LogitsView::Host(logits))
    }

    pub fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        Ok(self.step_view(token, pos)?.into_vec())
    }

    pub fn replay_stats(&self) -> ReplayStats {
        self.worker.replay_stats()
    }

    /// Whether specialization or fusion compiles are outstanding; see
    /// [`Scheme::compiles_pending`].
    pub fn compiles_pending(&self) -> bool {
        self.worker.compiles_pending()
    }

    pub fn fusion_report(&self) -> FusionReport {
        self.worker.fusion_report()
    }

    /// Dispatches and other nodes submitted per token, after fusion.
    pub fn executed_node_count(&self) -> usize {
        self.worker.executed_node_count()
    }
}

fn record(
    worker: &mut Scheme,
    modules: &DecoderModules,
    state: &DecodeState,
    slabs: &[Tensor],
    map: &WeightMap,
    config: &Config,
) -> Result<()> {
    modules.embedding.record_group(
        worker,
        "embed",
        map.embed.view(slabs)?,
        &state.step,
        state.x.view(),
    )?;
    for (layer, spec) in map.layers.iter().enumerate() {
        modules.attention.record_group(
            worker,
            format!("layer{layer}/attn"),
            state.x.view(),
            spec.attention(slabs)?,
            state.cache.layer(layer as u32)?,
            &state.step,
        )?;
        modules.mlp.record_group(
            worker,
            format!("layer{layer}/ffn"),
            state.x.view(),
            spec.mlp(slabs)?,
        )?;
    }
    let final_norm = map.final_norm.view(slabs)?;
    let lm_head = map.lm_head.view(slabs)?;
    worker.group("tail", |scheme| {
        let normed =
            scheme.temporary_tensor(TensorShape::vector(config.hidden), TensorDType::F32)?;
        modules
            .final_norm
            .record(scheme, state.x.view(), final_norm, normed)?;
        modules
            .lm_head
            .record(scheme, lm_head, normed, state.logits.view())
    })?;
    Ok(())
}

impl AutoregressiveModel for Model {
    fn vocab_size(&self) -> usize {
        self.config.vocab as usize
    }

    fn max_seq_len(&self) -> u32 {
        self.seq_len
    }

    fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        Model::step(self, token, pos)
    }

    fn step_view(&mut self, token: u32, pos: u32) -> Result<LogitsView> {
        Model::step_view(self, token, pos)
    }

    fn advance(&mut self, token: u32, pos: u32) -> Result<()> {
        Model::advance(self, token, pos)
    }
}
