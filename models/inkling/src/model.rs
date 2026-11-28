//! One Inkling 2 Mini forward: embeddings, 48 visits, final norm, and the LM head.
//!
//! The schedule in [`crate::tape`] is the only tape owner. A boundary layer
//! writes its input into the tape row [`blocks_before`] already names; this
//! module does not append a second copy. Later passes copy the message-key
//! rows out of the previous pass, mix them with that pass's exit state, then
//! restore the pre-loop hidden state.

use crate::layer::{DecodeLayer, DecodeLayerScalars, DecodeLayerWeights};
use crate::names::{load_decode_layer, DecodeLayerSpec};
use crate::tape::{blocks_before, message_key_start, LoopSchedule};
use ammon::{configure_inference_scheme, DecodeLayerState, DepthMix, Embedding, Linear, RmsNorm, Softcap};
use anyhow::{bail, Context, Result};
use goldy::{
    Buffer, BufferKind, Context as GoldyContext, GoldyError, MemoryExchange, Runtime, Scheme,
    Tensor, TensorDType, TensorKernels, TensorShape, TensorView,
};
use safetensors::Dtype;
use safetensors::SafeTensors;

const DEPTH_RMS_EPS: f32 = 1e-6;

/// Widths and loop grid for one random-init Mini forward.
#[derive(Clone, Copy, Debug)]
pub struct ForwardSpec {
    pub schedule: LoopSchedule,
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    pub rel_extent: u32,
    pub window: u32,
    pub conv_kernel: u32,
    pub experts: u32,
    pub top_k: u32,
    pub intermediate: u32,
    pub gate_rank: u32,
    pub vocab: u32,
    pub seq: u32,
    pub scalars: DecodeLayerScalars,
    pub logit_cap: f32,
}

struct Visit {
    layer: u32,
    pass: u32,
}

/// Device buffers the recorded schemes read. Several fields are never named
/// after `record` returns; dropping them would make the next submit stale.
#[allow(dead_code)]
struct Graph {
    ctx: GoldyContext,
    step: Buffer,
    deposit: goldy::DepositTransaction,
    /// Embed, every visit, the loop copies, and the LM head. One submit per token.
    token: Scheme,
    logits_sink: goldy::HostSink,
    states: Vec<DecodeLayerState>,
    h: Tensor,
    h2: Tensor,
    h_snap: Tensor,
    exit: Tensor,
    embed_raw: Tensor,
    tape: Tensor,
    msg_keys: Tensor,
    messages_buf: Tensor,
    final_mix: Tensor,
    final_hidden: Tensor,
    pre_logits: Tensor,
    logits: Tensor,
}

/// Device weights and one recorded token of the Mini schedule.
pub struct TextModel {
    graph: Option<Graph>,
    layers: Vec<DecodeLayer>,
    weights: Vec<DecodeLayerWeights>,
    spec: ForwardSpec,
    runtime: Runtime,
    embed_table: Tensor,
    embed_norm_w: Tensor,
    final_query: Tensor,
    final_norm_w: Tensor,
    message_query: Tensor,
    lm_head_w: Tensor,
    embedder: Embedding,
    norm: RmsNorm,
    head: Linear,
    softcap: Softcap,
    depth: DepthMix,
    ops: TensorKernels,
}

impl ammon::AutoregressiveModel for TextModel {
    fn vocab_size(&self) -> usize {
        self.spec.vocab as usize
    }

    fn max_seq_len(&self) -> u32 {
        self.spec.seq
    }

    fn step(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        TextModel::step(self, token, pos)
    }
}

impl TextModel {
    pub fn new(runtime: &Runtime, spec: ForwardSpec, weights: &[u8]) -> Result<Self> {
        if spec.schedule.visits() == 0 || spec.seq == 0 {
            bail!("forward needs a schedule and a sequence");
        }
        let even = spec.schedule.loop_start % 2 == 0
            && spec.schedule.n_loop() % 2 == 0
            && (spec.schedule.layers - spec.schedule.loop_end) % 2 == 0;
        if !even {
            bail!("each visit segment must be even so hidden returns to the same buffer");
        }
        let file = SafeTensors::deserialize(weights).context("model safetensors")?;
        let hidden = spec.hidden;
        let embed_raw = take_f32(&file, "model.embed_tokens.weight")?;
        if embed_raw.0.len() != 2
            || embed_raw.0[1] != hidden as usize
            || embed_raw.0[0] < spec.vocab as usize
        {
            bail!("embed table {:?} is not [>=vocab, {hidden}]", embed_raw.0);
        }
        let embed_table = shaped(runtime, &embed_raw.1, &[embed_raw.0[0] as u32, hidden])?;
        let embed_norm_w = shaped(
            runtime,
            &take(
                &file,
                "model.embed_tokens.embed_norm.weight",
                &[hidden as usize],
            )?,
            &[hidden],
        )?;
        let final_norm_w = shaped(
            runtime,
            &take(&file, "model.norm.weight", &[hidden as usize])?,
            &[hidden],
        )?;
        let lm_head_w = shaped(
            runtime,
            &take(
                &file,
                "lm_head.weight",
                &[spec.vocab as usize, hidden as usize],
            )?,
            &[spec.vocab, hidden],
        )?;
        let final_query = query(runtime, &file, "model.final_input.query", 1, hidden)?;
        let message_query = if spec.schedule.n_loop() == 0 {
            zeros(runtime, &[1, hidden])?
        } else {
            query(
                runtime,
                &file,
                "model.message_input.query",
                spec.schedule.n_loop() * 4,
                hidden,
            )?
        };

        let mut layers = Vec::new();
        let mut layer_weights = Vec::new();
        for layer in 0..spec.schedule.layers {
            let full = spec.schedule.is_full(layer);
            let in_loop = layer >= spec.schedule.loop_start && layer < spec.schedule.loop_end;
            let layer_spec = DecodeLayerSpec {
                full,
                dynamic_conv: full,
                layer_index: layer,
                boundary: spec.schedule.is_boundary(layer),
                messages: in_loop,
                hidden,
                q_heads: spec.q_heads,
                kv_heads: spec.kv_heads,
                head_dim: spec.head_dim,
                d_rel: spec.d_rel,
                rel_extent: if full { spec.rel_extent } else { spec.window },
                conv_kernel: spec.conv_kernel,
                post_kernel: spec.conv_kernel,
                experts: spec.experts,
                top_k: spec.top_k,
                intermediate: spec.intermediate,
                gate_rank: spec.gate_rank,
                window: if full { 0 } else { spec.window },
            };
            let mut scalars = spec.scalars;
            if !full {
                scalars.log_alpha = 0.0;
            }
            let host = load_decode_layer(weights, &format!("model.layers.{layer}."), layer_spec)
                .with_context(|| format!("layer {layer}"))?;
            layer_weights.push(DecodeLayerWeights::from_spec(runtime, layer_spec, &host)?);
            layers.push(DecodeLayer::new(runtime, layer_spec, scalars)?);
        }

        Ok(Self {
            graph: None,
            layers,
            weights: layer_weights,
            spec,
            runtime: runtime.clone(),
            embed_table,
            embed_norm_w,
            final_query,
            final_norm_w,
            message_query,
            lm_head_w,
            embedder: Embedding::new(runtime)?,
            norm: RmsNorm::with_eps(runtime, spec.scalars.rms_eps)?,
            head: Linear::new(runtime)?,
            softcap: Softcap::new(runtime, spec.logit_cap)?,
            depth: DepthMix::new(runtime)?,
            ops: TensorKernels::new(runtime)?,
        })
    }

    pub fn spec(&self) -> ForwardSpec {
        self.spec
    }

    /// Record every visit once. Later tokens replay these schemes.
    pub fn bind(&mut self, ctx: &GoldyContext) -> Result<()> {
        if self.graph.is_some() {
            return Ok(());
        }
        self.graph = Some(self.record(ctx)?);
        Ok(())
    }

    /// Submits per token, executed dispatch nodes per token, and whether a
    /// background fusion or specialization compile is still outstanding.
    ///
    /// One scheme records the whole token, and the logits read is that submit.
    pub fn replay_census(&self) -> Option<(usize, usize, bool)> {
        let scheme = &self.graph.as_ref()?.token;
        Some((1, scheme.executed_node_count(), scheme.compiles_pending()))
    }

    /// Replay counters for the token scheme: records, topology records, clean
    /// submits, resubmit hits and retained evictions.
    pub fn replay_totals(&self) -> Option<[u64; 5]> {
        let stats = self.graph.as_ref()?.token.replay_stats();
        Some([
            stats.records,
            stats.topology_records,
            stats.clean_submits,
            stats.resubmit_hits,
            stats.retained_evictions,
        ])
    }

    /// Fusion report of the token scheme when `index` is a visit in the schedule.
    ///
    /// Visits are groups inside that one scheme, so the report is the whole token,
    /// not a single visit.
    pub fn visit_fusion(&self, index: usize) -> Option<(goldy::FusionReport, usize)> {
        if index >= self.spec.schedule.visits() as usize {
            return None;
        }
        let scheme = &self.graph.as_ref()?.token;
        Some((scheme.fusion_report(), scheme.executed_node_count()))
    }

    /// Run one token. `position` is the absolute position, also the KV slot.
    pub fn step(&mut self, token: u32, position: u32) -> Result<Vec<f32>> {
        let graph = self
            .graph
            .as_mut()
            .context("bind the forward before stepping")?;
        graph.states[0]
            .deposit_position(&graph.deposit, token, position)
            .context("deposit token")?;
        let mut sub = graph.token.submit().context("token")?;
        let logits = (&mut sub >> &graph.logits_sink)
            .take::<f32>()
            .context("logits")?;
        Ok(logits.to_vec())
    }

    /// Retention counters for the one recorded token scheme.
    pub fn replay_stats(&self) -> goldy::ReplayStats {
        self.graph
            .as_ref()
            .map(|graph| graph.token.replay_stats())
            .unwrap_or_default()
    }

    fn record(&self, ctx: &GoldyContext) -> Result<Graph> {
        let spec = self.spec;
        let hidden = spec.hidden;
        let schedule = spec.schedule;
        let order = visit_order(schedule);
        if order.len() != schedule.visits() as usize {
            bail!("visit list {} != {}", order.len(), schedule.visits());
        }
        let (n_keys, key_start) = if schedule.n_loop() == 0 {
            (0, 0)
        } else {
            let end_len = blocks_before(schedule, schedule.loop_end, 0);
            let key_start = message_key_start(schedule);
            if end_len < key_start {
                bail!("message key slice starts past the loop tape");
            }
            (end_len - key_start, key_start)
        };
        let tape_rows = blocks_before(schedule, schedule.layers, 0).max(1);
        let cap = spec.seq.max(spec.window);
        let runtime = &self.runtime;
        let h = zeros(runtime, &[hidden])?;
        let h2 = zeros(runtime, &[hidden])?;
        let embed_raw = zeros(runtime, &[hidden])?;
        let h_snap = zeros(runtime, &[hidden])?;
        let exit = zeros(runtime, &[hidden])?;
        let tape = zeros(runtime, &[tape_rows, hidden])?;
        let msg_keys = zeros(runtime, &[n_keys.max(1), hidden])?;
        let n_loop = schedule.n_loop();
        let messages_buf = zeros(runtime, &[n_loop, 4, hidden])?;
        let final_mix = zeros(runtime, &[1, hidden])?;
        let final_hidden = zeros(runtime, &[hidden])?;
        let pre_logits = zeros(runtime, &[spec.vocab])?;
        let logits = zeros(runtime, &[spec.vocab])?;

        let mut states = Vec::with_capacity(order.len());
        for visit in &order {
            let full = schedule.is_full(visit.layer);
            let state = if full {
                DecodeLayerState::full(
                    runtime,
                    cap,
                    spec.kv_heads,
                    spec.head_dim,
                    spec.q_heads,
                    spec.conv_kernel,
                )?
            } else {
                DecodeLayerState::sliding(
                    runtime,
                    spec.window,
                    cap,
                    spec.kv_heads,
                    spec.head_dim,
                    spec.q_heads,
                    spec.conv_kernel,
                )?
            };
            states.push(state.with_post_conv(runtime, hidden)?);
        }

        let step = runtime.acquire_buffer_with_data(
            &[ammon::kernels::DecodeStep {
                token: 0,
                position: 0,
            }],
            BufferKind::Scattered,
        )?;
        let exchange = MemoryExchange::new(ctx);
        let mut token = Scheme::new(ctx);
        configure_inference_scheme(&mut token);
        token.set_automatic_fusion(true);
        let deposit = exchange.bind_deposit(
            &mut token,
            ammon::kernels::DecodeStep::deposit_target(&step),
        )?;
        let embedder = &self.embedder;
        let embed_table = self.embed_table.view();
        let embed_norm_w = self.embed_norm_w.view();
        let norm = &self.norm;
        token.group("embed", |scheme| {
            embedder.record(scheme, embed_table, &step, embed_raw.view())?;
            norm.record(scheme, embed_raw.view(), embed_norm_w, h.view())?;
            Ok(())
        })?;

        let mut src_is_h = true;
        let mut visit_at = 0usize;
        let record_visit =
            |token: &mut Scheme, index: usize, visit: &Visit, src_is_h: bool| -> Result<bool> {
                let n_blocks = blocks_before(schedule, visit.layer, visit.pass);
                let (src, dst) = if src_is_h {
                    (h.view(), h2.view())
                } else {
                    (h2.view(), h.view())
                };
                let messages = if visit.pass > 0 {
                    Some(message_views(
                        &messages_buf,
                        visit.layer - schedule.loop_start,
                        hidden,
                    )?)
                } else {
                    None
                };
                let layer = &self.layers[visit.layer as usize];
                let weights = &self.weights[visit.layer as usize];
                let state = &states[index];
                token.group(format!("visit-{index}"), |scheme| {
                    layer.record(
                        scheme,
                        src,
                        tape.view(),
                        n_blocks,
                        messages,
                        weights,
                        state,
                        &step,
                        dst,
                    )
                })?;
                Ok(!src_is_h)
            };
        for layer in 0..schedule.loop_start {
            let visit = &order[visit_at];
            debug_assert_eq!(visit.layer, layer);
            src_is_h = record_visit(&mut token, visit_at, visit, src_is_h)?;
            visit_at += 1;
        }
        let ops = &self.ops;
        token.group("snap", |scheme| {
            ops.recorder(scheme)
                .copy("snap_hidden", h.view(), h_snap.view())
        })?;
        let depth = &self.depth;
        let scale = (f64::from(hidden)).powf(-0.5) as f32;
        for pass in 0..schedule.passes {
            if pass > 0 {
                token.group(format!("keys-{pass}"), |scheme| {
                    for slot in 0..n_keys {
                        let src = tape
                            .view()
                            .narrow(0, key_start + slot, 1)?
                            .reshape(&[hidden])?;
                        depth.store_row(scheme, src, msg_keys.view(), slot)?;
                    }
                    Ok(())
                })?;
                let message_query = self.message_query.view();
                token.group(format!("messages-{pass}"), |scheme| {
                    // Every loop layer's four message queries read the same keys and
                    // partial, so one mix covers them all.
                    if n_loop == 0 {
                        return Ok(());
                    }
                    depth.mix(
                        scheme,
                        msg_keys.view(),
                        exit.view(),
                        message_query,
                        messages_buf.view().reshape(&[n_loop * 4, hidden])?,
                        n_keys,
                        DEPTH_RMS_EPS,
                        scale,
                    )
                })?;
                token.group(format!("restore-{pass}"), |scheme| {
                    ops.recorder(scheme)
                        .copy("restore_hidden", h_snap.view(), h.view())
                })?;
            }
            for _layer in schedule.loop_start..schedule.loop_end {
                let visit = &order[visit_at];
                src_is_h = record_visit(&mut token, visit_at, visit, src_is_h)?;
                visit_at += 1;
            }
            token.group(format!("exit-{pass}"), |scheme| {
                ops.recorder(scheme)
                    .copy("pass_exit", h.view(), exit.view())
            })?;
        }
        for layer in schedule.loop_end..schedule.layers {
            let visit = &order[visit_at];
            debug_assert_eq!(visit.layer, layer);
            src_is_h = record_visit(&mut token, visit_at, visit, src_is_h)?;
            visit_at += 1;
        }
        if visit_at != order.len() {
            bail!("recorded {visit_at} visits, schedule has {}", order.len());
        }
        let n_final = blocks_before(schedule, schedule.layers, 0);
        let final_query = self.final_query.view();
        let final_norm_w = self.final_norm_w.view();
        let lm_head_w = self.lm_head_w.view();
        let head = &self.head;
        let softcap = &self.softcap;
        token.group("finish", |scheme| {
            depth.mix(
                scheme,
                tape.view(),
                h.view(),
                final_query,
                final_mix.view(),
                n_final,
                DEPTH_RMS_EPS,
                scale,
            )?;
            norm.record(
                scheme,
                final_mix.view().reshape(&[hidden])?,
                final_norm_w,
                final_hidden.view(),
            )?;
            head.record(scheme, lm_head_w, final_hidden.view(), pre_logits.view())?;
            softcap.record(scheme, pre_logits.view(), logits.view())?;
            Ok(())
        })?;
        let logits_sink = exchange
            .bind_host_sink(&mut token, logits.buffer())
            .context("logits sink")?;

        Ok(Graph {
            ctx: ctx.clone(),
            step,
            deposit,
            token,
            logits_sink,
            states,
            h,
            h2,
            h_snap,
            exit,
            embed_raw,
            tape,
            msg_keys,
            messages_buf,
            final_mix,
            final_hidden,
            pre_logits,
            logits,
        })
    }
}

fn visit_order(schedule: LoopSchedule) -> Vec<Visit> {
    let mut order = Vec::new();
    for layer in 0..schedule.loop_start {
        order.push(Visit { layer, pass: 0 });
    }
    for pass in 0..schedule.passes {
        for layer in schedule.loop_start..schedule.loop_end {
            order.push(Visit { layer, pass });
        }
    }
    for layer in schedule.loop_end..schedule.layers {
        order.push(Visit { layer, pass: 0 });
    }
    order
}

fn message_views(
    messages: &Tensor,
    local: u32,
    hidden: u32,
) -> Result<[TensorView<'_>; 4], GoldyError> {
    let row = messages.view().narrow(0, local, 1)?.reshape(&[4, hidden])?;
    Ok([
        row.narrow(0, 0, 1)?.reshape(&[hidden])?,
        row.narrow(0, 1, 1)?.reshape(&[hidden])?,
        row.narrow(0, 2, 1)?.reshape(&[hidden])?,
        row.narrow(0, 3, 1)?.reshape(&[hidden])?,
    ])
}

fn zeros(runtime: &Runtime, dims: &[u32]) -> Result<Tensor> {
    Ok(Tensor::zeros(
        runtime,
        TensorShape::from_dims(dims)?,
        TensorDType::F32,
    )?)
}

fn shaped(runtime: &Runtime, data: &[f32], dims: &[u32]) -> Result<Tensor> {
    Tensor::from_f32(runtime, TensorShape::from_dims(dims)?, data).context("upload weight")
}

fn take_f32(file: &SafeTensors<'_>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = file
        .tensor(name)
        .map_err(|err| anyhow::anyhow!("missing `{name}`: {err}"))?;
    if view.dtype() != Dtype::F32 {
        bail!("`{name}` is {:?}, expected F32", view.dtype());
    }
    Ok((view.shape().to_vec(), bytes_f32(view.data())))
}

fn take(file: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = file
        .tensor(name)
        .map_err(|err| anyhow::anyhow!("missing `{name}` ({shape:?}): {err}"))?;
    if view.dtype() != Dtype::F32 {
        bail!("`{name}` is {:?}, expected F32", view.dtype());
    }
    if view.shape() != shape {
        bail!("`{name}` has shape {:?}, expected {shape:?}", view.shape());
    }
    Ok(bytes_f32(view.data()))
}

fn query(
    runtime: &Runtime,
    file: &SafeTensors<'_>,
    name: &str,
    rows: u32,
    hidden: u32,
) -> Result<Tensor> {
    let view = file
        .tensor(name)
        .map_err(|err| anyhow::anyhow!("missing `{name}`: {err}"))?;
    if view.dtype() != Dtype::F32 {
        bail!("`{name}` is {:?}, expected F32", view.dtype());
    }
    let data = bytes_f32(view.data());
    let shape = view.shape();
    let n = (rows as usize) * (hidden as usize);
    let ok = data.len() == n
        && (shape == [rows as usize, hidden as usize]
            || shape == [n]
            || shape == [(rows / 4) as usize, 4, hidden as usize]
            || (rows == 1 && (shape == [hidden as usize] || shape == [1, hidden as usize])));
    if !ok {
        bail!("`{name}` has shape {shape:?}, expected [{rows}, {hidden}]");
    }
    shaped(runtime, &data, &[rows, hidden])
}

/// Read a flat or nested config object.
///
/// Each field is the last `"name": number` in the text, so a nested
/// `text_config` overrides a top-level copy. Hugging Face aliases
/// (`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
/// `num_key_value_heads`, `intermediate_size`, `num_local_experts`,
/// `num_experts_per_tok`, `vocab_size`, `max_position_embeddings`,
/// `sliding_window`, `rms_norm_eps`) are accepted beside the short names.
pub fn spec_from_config(text: &str) -> Result<ForwardSpec> {
    let number = |names: &[&str]| -> Result<f64> {
        let mut found = None;
        for name in names {
            let mut rest = text;
            let needle = format!("\"{name}\"");
            while let Some(at) = rest.find(&needle) {
                let after = &rest[at + needle.len()..];
                let after = after.trim_start();
                let Some(after) = after.strip_prefix(':') else {
                    rest = &rest[at + needle.len()..];
                    continue;
                };
                let after = after.trim_start();
                let end = after
                    .find(|ch: char| {
                        !(ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E'))
                    })
                    .unwrap_or(after.len());
                if let Ok(value) = after[..end].parse::<f64>() {
                    found = Some(value);
                }
                rest = &rest[at + needle.len()..];
            }
        }
        found.with_context(|| format!("config is missing {}", names[0]))
    };
    let whole = |names: &[&str]| -> Result<u32> {
        let value = number(names)?;
        if value < 0.0 || value.fract() != 0.0 {
            bail!("config {} is not a whole number ({value})", names[0]);
        }
        Ok(value as u32)
    };
    Ok(ForwardSpec {
        schedule: LoopSchedule {
            layers: whole(&["num_hidden_layers", "layers"])?,
            block: whole(&["block"])?,
            loop_start: whole(&["loop_start"])?,
            loop_end: whole(&["loop_end"])?,
            passes: whole(&["passes"])?,
        },
        hidden: whole(&["hidden_size", "hidden"])?,
        q_heads: whole(&["num_attention_heads", "q_heads"])?,
        kv_heads: whole(&["num_key_value_heads", "kv_heads"])?,
        head_dim: whole(&["head_dim"])?,
        d_rel: whole(&["d_rel"])?,
        rel_extent: whole(&["rel_extent"])?,
        window: whole(&["sliding_window", "window"])?,
        conv_kernel: whole(&["conv_kernel"])?,
        experts: whole(&["num_local_experts", "experts"])?,
        top_k: whole(&["num_experts_per_tok", "top_k"])?,
        intermediate: whole(&["intermediate_size", "intermediate"])?,
        gate_rank: whole(&["gate_rank"])?,
        vocab: whole(&["vocab_size", "vocab"])?,
        seq: whole(&["max_position_embeddings", "seq"])?,
        scalars: DecodeLayerScalars {
            rms_eps: number(&["rms_norm_eps", "rms_eps"])? as f32,
            qk_clip: number(&["qk_clip"])? as f32,
            rel_clip: number(&["rel_clip"])? as f32,
            route_scale: number(&["route_scale"])? as f32,
            gate_cap: number(&["gate_cap"])? as f32,
            up_cap: number(&["up_cap"])? as f32,
            log_alpha: number(&["log_alpha"])? as f32,
            log_floor: number(&["log_floor"])? as f32,
        },
        logit_cap: number(&["logit_cap"])? as f32,
    })
}

/// Header, prompt tokens, and teacher-forced logits from `mini_forward.bin`.
pub fn read_golden(bytes: &[u8]) -> Result<(ForwardSpec, Vec<u32>, Vec<f32>)> {
    let mut at = 0usize;
    let u32_at = |at: &mut usize| -> Result<u32> {
        let bytes = bytes
            .get(*at..*at + 4)
            .context("golden truncated")?
            .try_into()
            .unwrap();
        *at += 4;
        Ok(u32::from_le_bytes(bytes))
    };
    let f32_at = |at: &mut usize| -> Result<f32> {
        let bytes = bytes
            .get(*at..*at + 4)
            .context("golden truncated")?
            .try_into()
            .unwrap();
        *at += 4;
        Ok(f32::from_le_bytes(bytes))
    };
    let magic = u32_at(&mut at)?;
    if magic != 0x494E_4B4D {
        bail!("golden magic {magic:#x}");
    }
    let version = u32_at(&mut at)?;
    if version != 1 {
        bail!("golden version {version}");
    }
    let hidden = u32_at(&mut at)?;
    let layers = u32_at(&mut at)?;
    let block = u32_at(&mut at)?;
    let loop_start = u32_at(&mut at)?;
    let loop_end = u32_at(&mut at)?;
    let passes = u32_at(&mut at)?;
    let mut spec = ForwardSpec {
        schedule: LoopSchedule {
            layers,
            block,
            loop_start,
            loop_end,
            passes,
        },
        hidden,
        q_heads: u32_at(&mut at)?,
        kv_heads: u32_at(&mut at)?,
        head_dim: u32_at(&mut at)?,
        d_rel: u32_at(&mut at)?,
        rel_extent: u32_at(&mut at)?,
        window: u32_at(&mut at)?,
        conv_kernel: u32_at(&mut at)?,
        experts: u32_at(&mut at)?,
        top_k: u32_at(&mut at)?,
        intermediate: u32_at(&mut at)?,
        gate_rank: u32_at(&mut at)?,
        vocab: u32_at(&mut at)?,
        seq: u32_at(&mut at)?,
        scalars: crate::layer::DecodeLayerScalars {
            rms_eps: f32_at(&mut at)?,
            qk_clip: f32_at(&mut at)?,
            rel_clip: f32_at(&mut at)?,
            route_scale: f32_at(&mut at)?,
            gate_cap: f32_at(&mut at)?,
            up_cap: f32_at(&mut at)?,
            log_alpha: 0.0,
            log_floor: 0.0,
        },
        logit_cap: f32_at(&mut at)?,
    };
    spec.scalars.log_alpha = f32_at(&mut at)?;
    spec.scalars.log_floor = f32_at(&mut at)?;
    let mut tokens = Vec::with_capacity(spec.seq as usize);
    for _ in 0..spec.seq {
        tokens.push(u32_at(&mut at)?);
    }
    let mut logits = Vec::with_capacity((spec.seq * spec.vocab) as usize);
    for _ in 0..spec.seq * spec.vocab {
        logits.push(f32_at(&mut at)?);
    }
    if at != bytes.len() {
        bail!("golden has {} trailing bytes", bytes.len() - at);
    }
    Ok((spec, tokens, logits))
}

fn bytes_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::spec_from_config;

    #[test]
    fn config_prefers_the_nested_copy() {
        let text = r#"{
            "hidden_size": 1,
            "text_config": {
                "num_hidden_layers": 12, "block": 6, "loop_start": 12, "loop_end": 12, "passes": 0,
                "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "d_rel": 8, "rel_extent": 16, "sliding_window": 8,
                "conv_kernel": 4, "num_local_experts": 8, "num_experts_per_tok": 2,
                "intermediate_size": 64, "gate_rank": 1, "vocab_size": 250,
                "max_position_embeddings": 8, "rms_norm_eps": 1e-5, "qk_clip": 0.15,
                "rel_clip": 0.35, "route_scale": 8, "gate_cap": 0.05, "up_cap": 0.05,
                "log_alpha": 0.1, "log_floor": 4, "logit_cap": 0.05
            }
        }"#;
        let spec = spec_from_config(text).unwrap();
        assert_eq!(spec.hidden, 64);
        assert_eq!(spec.schedule.layers, 12);
        assert_eq!(spec.vocab, 250);
        assert!((spec.scalars.rms_eps - 1e-5).abs() < 1e-8);
    }
}
