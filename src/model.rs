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
use ammon::{DecodeLayerState, DepthMix, Embedding, Linear, RmsNorm, Softcap};
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
    upload: Scheme,
    embed: Scheme,
    visits: Vec<Scheme>,
    snap: Scheme,
    restore: Scheme,
    copy_keys: Scheme,
    messages: Scheme,
    copy_exit: Scheme,
    finish: Scheme,
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
        let message_query = query(
            runtime,
            &file,
            "model.message_input.query",
            spec.schedule.n_loop() * 4,
            hidden,
        )?;

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

    /// Run one token. `position` is the absolute position, also the KV slot.
    pub fn step(&mut self, token: u32, position: u32) -> Result<Vec<f32>> {
        let spec = self.spec;
        let graph = self
            .graph
            .as_mut()
            .context("bind the forward before stepping")?;
        graph.states[0]
            .deposit_position(&graph.deposit, token, position)
            .context("deposit token")?;
        let _ = graph.upload.submit().context("upload token")?;
        let _ = graph.embed.submit().context("embed")?;
        let mut visit = 0usize;
        for _layer in 0..spec.schedule.loop_start {
            let _ = graph.visits[visit].submit().context("pre-loop visit")?;
            visit += 1;
        }
        let _ = graph.snap.submit().context("snapshot pre-loop hidden")?;
        for pass in 0..spec.schedule.passes {
            if pass > 0 {
                let _ = graph.copy_keys.submit().context("copy message keys")?;
                let _ = graph.messages.submit().context("message mix")?;
                let _ = graph.restore.submit().context("restore pre-loop hidden")?;
            }
            for _layer in spec.schedule.loop_start..spec.schedule.loop_end {
                let _ = graph.visits[visit].submit().context("loop visit")?;
                visit += 1;
            }
            let _ = graph.copy_exit.submit().context("save pass exit")?;
        }
        for _layer in spec.schedule.loop_end..spec.schedule.layers {
            let _ = graph.visits[visit].submit().context("post-loop visit")?;
            visit += 1;
        }
        if visit != graph.visits.len() {
            bail!("ran {visit} visits, recorded {}", graph.visits.len());
        }
        let _ = graph.finish.submit().context("final norm and lm head")?;
        let mut read = Scheme::new(&graph.ctx);
        let mut sub = read.submit().context("read logits")?;
        let bytes = (&mut sub >> graph.logits.buffer())
            .take::<u8>()
            .context("logits bytes")?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    fn record(&self, ctx: &GoldyContext) -> Result<Graph> {
        let spec = self.spec;
        let hidden = spec.hidden;
        let schedule = spec.schedule;
        let order = visit_order(schedule);
        if order.len() != schedule.visits() as usize {
            bail!("visit list {} != {}", order.len(), schedule.visits());
        }
        let end_len = blocks_before(schedule, schedule.loop_end, 0);
        let key_start = message_key_start(schedule);
        if end_len < key_start {
            bail!("message key slice starts past the loop tape");
        }
        let n_keys = end_len - key_start;
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
        let mut upload = Scheme::new(ctx);
        let deposit = MemoryExchange::new(ctx).bind_deposit(
            &mut upload,
            ammon::kernels::DecodeStep::deposit_target(&step),
        )?;

        let mut embed = Scheme::new(ctx);
        self.embedder
            .record(&mut embed, self.embed_table.view(), &step, embed_raw.view())?;
        self.norm.record(
            &mut embed,
            embed_raw.view(),
            self.embed_norm_w.view(),
            h.view(),
        )?;

        let mut visits = Vec::with_capacity(order.len());
        let mut src_is_h = true;
        for (index, visit) in order.iter().enumerate() {
            let n_blocks = blocks_before(schedule, visit.layer, visit.pass);
            let mut scheme = Scheme::new(ctx);
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
            self.layers[visit.layer as usize].record(
                &mut scheme,
                src,
                tape.view(),
                n_blocks,
                messages,
                &self.weights[visit.layer as usize],
                &states[index],
                &step,
                dst,
            )?;
            visits.push(scheme);
            src_is_h = !src_is_h;
        }

        let mut snap = Scheme::new(ctx);
        self.ops
            .recorder(&mut snap)
            .copy("snap_hidden", h.view(), h_snap.view())?;
        let mut restore = Scheme::new(ctx);
        self.ops
            .recorder(&mut restore)
            .copy("restore_hidden", h_snap.view(), h.view())?;
        let mut copy_exit = Scheme::new(ctx);
        self.ops
            .recorder(&mut copy_exit)
            .copy("pass_exit", h.view(), exit.view())?;

        let mut copy_keys = Scheme::new(ctx);
        for slot in 0..n_keys {
            let src = tape
                .view()
                .narrow(0, key_start + slot, 1)?
                .reshape(&[hidden])?;
            self.depth
                .store_row(&mut copy_keys, src, msg_keys.view(), slot)?;
        }
        let scale = (f64::from(hidden)).powf(-0.5) as f32;
        let mut messages = Scheme::new(ctx);
        for local in 0..n_loop {
            let queries = self
                .message_query
                .view()
                .narrow(0, local * 4, 4)?
                .reshape(&[4, hidden])?;
            let out = messages_buf
                .view()
                .narrow(0, local, 1)?
                .reshape(&[4, hidden])?;
            self.depth.mix(
                &mut messages,
                msg_keys.view(),
                exit.view(),
                queries,
                out,
                n_keys,
                DEPTH_RMS_EPS,
                scale,
            )?;
        }

        let n_final = blocks_before(schedule, schedule.layers, 0);
        let mut finish = Scheme::new(ctx);
        self.depth.mix(
            &mut finish,
            tape.view(),
            h.view(),
            self.final_query.view(),
            final_mix.view(),
            n_final,
            DEPTH_RMS_EPS,
            scale,
        )?;
        self.norm.record(
            &mut finish,
            final_mix.view().reshape(&[hidden])?,
            self.final_norm_w.view(),
            final_hidden.view(),
        )?;
        self.head.record(
            &mut finish,
            self.lm_head_w.view(),
            final_hidden.view(),
            pre_logits.view(),
        )?;
        self.softcap
            .record(&mut finish, pre_logits.view(), logits.view())?;

        Ok(Graph {
            ctx: ctx.clone(),
            step,
            deposit,
            upload,
            embed,
            visits,
            snap,
            restore,
            copy_keys,
            messages,
            copy_exit,
            finish,
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
