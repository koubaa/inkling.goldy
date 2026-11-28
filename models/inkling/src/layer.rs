//! One Inkling 2 / loop decoder layer, recorded once and replayed per token.
//!
//! Order matches the pinned decoder layer: depth mix (skipped on layer 0),
//! optional message gates, input RMSNorm, self-attention, post-attention short
//! conv, residual (the layer input is added only off a block boundary), MLP
//! depth mix, optional MLP message gate, post-attention RMSNorm, routed
//! experts, post-MLP short conv, and the residual add. KV, the key/value
//! convolutions, and the post-attention and post-MLP convolutions all live on
//! the visit's [`DecodeLayerState`]. Intra-visit activations are scheme-scoped
//! temporaries on `TransientPool`; this module retains no GPU buffers.

use crate::names::{DecodeLayerSpec, DecodeLayerTensors, MessageGateTensors};
use ammon::{
    DecodeAttention, DecodeAttentionKind, DecodeAttentionWeights, DecodeLayerState, DecodeMoe,
    DepthMix, Linear, LogPositionScale, RmsNorm, ShortConv, Softcap, DEPTH_RMS_EPS,
};
use anyhow::{bail, Context};
use goldy::{Buffer, GoldyError, Runtime, Scheme, Tensor, TensorDType, TensorShape, TensorView};
use std::sync::Arc;

/// Scalar hyperparameters that are config fields, not checkpoint tensors.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLayerScalars {
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub rms_eps: f32,
    pub route_scale: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    /// `0` disables log-position scaling. Full layers in the reference set it.
    pub log_alpha: f32,
    pub log_floor: f32,
}

/// Contiguous F32 slab builder so a whole model (or standalone layer) uploads one weight buffer.
#[derive(Default)]
pub(crate) struct WeightSlabBuilder {
    values: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct WeightSlice {
    offset: u64,
    shape: TensorShape,
}

impl WeightSlice {
    #[inline]
    pub(crate) fn view<'a>(&self, slab: &'a Tensor) -> TensorView<'a> {
        TensorView::packed_at(slab.buffer(), TensorDType::F32, self.offset, self.shape)
            .expect("WeightSlice validated against slab length at pack time")
    }
}

impl WeightSlabBuilder {
    pub(crate) fn push(&mut self, data: &[f32], dims: &[u32]) -> anyhow::Result<WeightSlice> {
        let shape = TensorShape::from_dims(dims).context("weight tensor shape")?;
        let expected = usize::try_from(shape.numel().context("weight tensor numel")?)
            .context("numel usize")?;
        if data.len() != expected {
            bail!(
                "weight slice has {} f32 elements, shape {:?} expects {}",
                data.len(),
                dims,
                expected
            );
        }
        let offset = u64::try_from(self.values.len()).context("weight offset overflow")?;
        self.values.extend_from_slice(data);
        Ok(WeightSlice { offset, shape })
    }

    pub(crate) fn finish(self, runtime: &Runtime) -> anyhow::Result<Arc<Tensor>> {
        let len = u32::try_from(self.values.len().max(1)).context("weight slab exceeds u32")?;
        let values = if self.values.is_empty() {
            vec![0.0f32; 1]
        } else {
            self.values
        };
        let slab = Tensor::from_f32(runtime, TensorShape::vector(len), &values)
            .context("upload packed weight slab")?;
        Ok(Arc::new(slab))
    }
}

#[derive(Clone)]
struct GateSlices {
    down: WeightSlice,
    forget: WeightSlice,
    message: WeightSlice,
}

#[derive(Clone)]
pub(crate) struct PackedLayerSlices {
    wq: WeightSlice,
    wk: WeightSlice,
    wv: WeightSlice,
    wo: WeightSlice,
    wgate: WeightSlice,
    wsink: WeightSlice,
    wrel: WeightSlice,
    rel_proj: WeightSlice,
    q_scale: WeightSlice,
    rel_scale: WeightSlice,
    k_conv: WeightSlice,
    v_conv: WeightSlice,
    k_conv_proj: Option<WeightSlice>,
    v_conv_proj: Option<WeightSlice>,
    input_norm: Option<WeightSlice>,
    q_norm: Option<WeightSlice>,
    k_norm: Option<WeightSlice>,
    v_norm: Option<WeightSlice>,
    post_norm: WeightSlice,
    post_attn_conv: WeightSlice,
    post_mlp_conv: WeightSlice,
    attn_query: Option<WeightSlice>,
    mlp_query: WeightSlice,
    router: WeightSlice,
    router_bias: WeightSlice,
    gate_up: WeightSlice,
    down: WeightSlice,
    q_gate: Option<GateSlices>,
    k_gate: Option<GateSlices>,
    v_gate: Option<GateSlices>,
    mlp_gate: Option<GateSlices>,
}

/// Device weights for [`DecodeLayer::record`], sliced from a shared contiguous F32 slab.
pub struct DecodeLayerWeights {
    slab: Arc<Tensor>,
    slices: PackedLayerSlices,
}

impl DecodeLayerWeights {
    pub(crate) fn pack(
        builder: &mut WeightSlabBuilder,
        spec: DecodeLayerSpec,
        host: &DecodeLayerTensors,
    ) -> anyhow::Result<PackedLayerSlices> {
        let hidden = spec.hidden;
        let q_dim = spec.q_heads * spec.head_dim;
        let kv_dim = spec.kv_heads * spec.head_dim;
        let queries = if spec.full { 3 } else { 1 };
        let conv_dims = if spec.dynamic_conv {
            [spec.conv_kernel, kv_dim]
        } else {
            [kv_dim, spec.conv_kernel]
        };
        fn pack_gate(
            builder: &mut WeightSlabBuilder,
            g: &MessageGateTensors,
            gate_rank: u32,
            hidden: u32,
        ) -> anyhow::Result<GateSlices> {
            Ok(GateSlices {
                down: builder.push(&g.down, &[gate_rank, 2 * hidden])?,
                forget: builder.push(&g.forget, &[hidden, gate_rank])?,
                message: builder.push(&g.message, &[hidden, gate_rank])?,
            })
        }
        let wq = builder.push(&host.wq, &[q_dim, hidden])?;
        let wk = builder.push(&host.wk, &[kv_dim, hidden])?;
        let wv = builder.push(&host.wv, &[kv_dim, hidden])?;
        let wo = builder.push(&host.wo, &[hidden, q_dim])?;
        let wgate = builder.push(&host.wgate, &[q_dim, hidden])?;
        let wsink = builder.push(&host.wsink, &[spec.q_heads, hidden])?;
        let wrel = builder.push(&host.wrel, &[spec.q_heads * spec.d_rel, hidden])?;
        let rel_proj = builder.push(&host.rel_proj, &[spec.d_rel, spec.rel_extent])?;
        let q_scale = builder.push(&host.q_scale, &[spec.q_heads, spec.head_dim])?;
        let rel_scale = builder.push(&host.rel_scale, &[spec.q_heads, spec.d_rel])?;
        let k_conv = builder.push(&host.k_conv, &conv_dims)?;
        let v_conv = builder.push(&host.v_conv, &conv_dims)?;
        let k_conv_proj = host
            .k_conv_proj
            .as_ref()
            .map(|d| builder.push(d, &[spec.kv_heads * spec.conv_kernel, hidden]))
            .transpose()?;
        let v_conv_proj = host
            .v_conv_proj
            .as_ref()
            .map(|d| builder.push(d, &[spec.kv_heads * spec.conv_kernel, hidden]))
            .transpose()?;
        let input_norm = host
            .input_norm
            .as_ref()
            .map(|d| builder.push(d, &[hidden]))
            .transpose()?;
        let q_norm = host
            .q_norm
            .as_ref()
            .map(|d| builder.push(d, &[hidden]))
            .transpose()?;
        let k_norm = host
            .k_norm
            .as_ref()
            .map(|d| builder.push(d, &[hidden]))
            .transpose()?;
        let v_norm = host
            .v_norm
            .as_ref()
            .map(|d| builder.push(d, &[hidden]))
            .transpose()?;
        let post_norm = builder.push(&host.post_norm, &[hidden])?;
        let post_attn_conv = builder.push(&host.post_attn_conv, &[hidden, spec.post_kernel])?;
        let post_mlp_conv = builder.push(&host.post_mlp_conv, &[hidden, spec.post_kernel])?;
        let attn_query = host
            .attn_query
            .as_ref()
            .map(|d| builder.push(d, &[queries, hidden]))
            .transpose()?;
        let mlp_query = builder.push(&host.mlp_query, &[1, hidden])?;
        let router = builder.push(&host.router, &[spec.experts, hidden])?;
        let router_bias = builder.push(&host.router_bias, &[spec.experts])?;
        let gate_up = builder.push(
            &host.gate_up,
            &[spec.experts, 2 * spec.intermediate, hidden],
        )?;
        let down = builder.push(&host.down, &[spec.experts, hidden, spec.intermediate])?;
        let q_gate = host
            .q_gate
            .as_ref()
            .map(|g| pack_gate(builder, g, spec.gate_rank, hidden))
            .transpose()?;
        let k_gate = host
            .k_gate
            .as_ref()
            .map(|g| pack_gate(builder, g, spec.gate_rank, hidden))
            .transpose()?;
        let v_gate = host
            .v_gate
            .as_ref()
            .map(|g| pack_gate(builder, g, spec.gate_rank, hidden))
            .transpose()?;
        let mlp_gate = host
            .mlp_gate
            .as_ref()
            .map(|g| pack_gate(builder, g, spec.gate_rank, hidden))
            .transpose()?;

        Ok(PackedLayerSlices {
            wq,
            wk,
            wv,
            wo,
            wgate,
            wsink,
            wrel,
            rel_proj,
            q_scale,
            rel_scale,
            k_conv,
            v_conv,
            k_conv_proj,
            v_conv_proj,
            input_norm,
            q_norm,
            k_norm,
            v_norm,
            post_norm,
            post_attn_conv,
            post_mlp_conv,
            attn_query,
            mlp_query,
            router,
            router_bias,
            gate_up,
            down,
            q_gate,
            k_gate,
            v_gate,
            mlp_gate,
        })
    }

    pub(crate) fn from_packed(slab: Arc<Tensor>, slices: PackedLayerSlices) -> Self {
        Self { slab, slices }
    }

    pub fn from_spec(
        runtime: &Runtime,
        spec: DecodeLayerSpec,
        host: &DecodeLayerTensors,
    ) -> anyhow::Result<Self> {
        let mut builder = WeightSlabBuilder::default();
        let slices = Self::pack(&mut builder, spec, host)?;
        let slab = builder.finish(runtime)?;
        Ok(Self::from_packed(slab, slices))
    }

    fn views(&self) -> DecodeAttentionWeights<'_> {
        let slab = self.slab.as_ref();
        let s = &self.slices;
        DecodeAttentionWeights {
            wq: s.wq.view(slab),
            wk: s.wk.view(slab),
            wv: s.wv.view(slab),
            wo: s.wo.view(slab),
            wgate: s.wgate.view(slab),
            wsink: s.wsink.view(slab),
            wrel: s.wrel.view(slab),
            rel_proj: s.rel_proj.view(slab),
            q_scale: s.q_scale.view(slab),
            rel_scale: s.rel_scale.view(slab),
            k_conv: s.k_conv.view(slab),
            v_conv: s.v_conv.view(slab),
            k_conv_proj: s.k_conv_proj.as_ref().map(|w| w.view(slab)),
            v_conv_proj: s.v_conv_proj.as_ref().map(|w| w.view(slab)),
        }
    }
}

/// Stateless child modules shared across every decoder layer in a model.
///
/// Mirrors `llama3`'s `DecoderModules`: pipelines and unit buffers are prepared
/// once on the runtime, while per-layer structural flags (`full`, `layer_index`,
/// `boundary`, `messages`) and scalars stay plain host descriptors.
pub struct SharedLayerModules {
    sliding_attn: Option<DecodeAttention>,
    full_attn: Option<DecodeAttention>,
    moe: DecodeMoe,
    norm: RmsNorm,
    linear: Linear,
    tanh: Softcap,
    depth: DepthMix,
    post: ShortConv,
    ops: goldy::TensorKernels,
    hidden: u32,
    gate_rank: u32,
    scale: f32,
}

fn build_attention(
    runtime: &Runtime,
    spec: DecodeLayerSpec,
    scalars: DecodeLayerScalars,
) -> anyhow::Result<DecodeAttention> {
    let kind = if spec.full {
        DecodeAttentionKind::Full
    } else {
        if spec.window == 0 {
            anyhow::bail!("sliding decoder layer needs a window");
        }
        DecodeAttentionKind::Sliding {
            window: spec.window,
        }
    };
    let log = if scalars.log_alpha != 0.0 {
        Some(LogPositionScale {
            alpha: scalars.log_alpha,
            floor: scalars.log_floor,
        })
    } else {
        None
    };
    DecodeAttention::new(
        runtime,
        kind,
        spec.q_heads,
        spec.kv_heads,
        spec.head_dim,
        spec.hidden,
        spec.d_rel,
        spec.rel_extent,
        spec.conv_kernel,
        spec.dynamic_conv,
        scalars.qk_clip,
        scalars.rel_clip,
        scalars.rms_eps,
        log,
    )
}

fn same_attn_config(
    a: (DecodeLayerSpec, DecodeLayerScalars),
    b: (DecodeLayerSpec, DecodeLayerScalars),
) -> bool {
    a.0.full == b.0.full
        && a.0.dynamic_conv == b.0.dynamic_conv
        && a.0.hidden == b.0.hidden
        && a.0.q_heads == b.0.q_heads
        && a.0.kv_heads == b.0.kv_heads
        && a.0.head_dim == b.0.head_dim
        && a.0.d_rel == b.0.d_rel
        && a.0.rel_extent == b.0.rel_extent
        && a.0.conv_kernel == b.0.conv_kernel
        && a.0.window == b.0.window
        && a.1.qk_clip.to_bits() == b.1.qk_clip.to_bits()
        && a.1.rel_clip.to_bits() == b.1.rel_clip.to_bits()
        && a.1.rms_eps.to_bits() == b.1.rms_eps.to_bits()
        && a.1.log_alpha.to_bits() == b.1.log_alpha.to_bits()
        && (a.1.log_alpha == 0.0 || a.1.log_floor.to_bits() == b.1.log_floor.to_bits())
}

fn same_shared_config(
    a: (DecodeLayerSpec, DecodeLayerScalars),
    b: (DecodeLayerSpec, DecodeLayerScalars),
) -> bool {
    a.0.hidden == b.0.hidden
        && a.0.post_kernel == b.0.post_kernel
        && a.0.experts == b.0.experts
        && a.0.top_k == b.0.top_k
        && a.0.intermediate == b.0.intermediate
        && a.0.gate_rank == b.0.gate_rank
        && a.1.rms_eps.to_bits() == b.1.rms_eps.to_bits()
        && a.1.route_scale.to_bits() == b.1.route_scale.to_bits()
        && a.1.gate_cap.to_bits() == b.1.gate_cap.to_bits()
        && a.1.up_cap.to_bits() == b.1.up_cap.to_bits()
}

impl SharedLayerModules {
    pub fn new(
        runtime: &Runtime,
        configs: &[(DecodeLayerSpec, DecodeLayerScalars)],
    ) -> anyhow::Result<Self> {
        let &(first_spec, first_scalars) = configs
            .first()
            .context("SharedLayerModules requires at least one layer spec")?;
        let mut sliding_cfg: Option<(DecodeLayerSpec, DecodeLayerScalars)> = None;
        let mut full_cfg: Option<(DecodeLayerSpec, DecodeLayerScalars)> = None;
        for &(spec, scalars) in configs {
            if !same_shared_config((first_spec, first_scalars), (spec, scalars)) {
                anyhow::bail!(
                    "layer {} does not match shared model dimensions/scalars",
                    spec.layer_index
                );
            }
            let slot = if spec.full {
                &mut full_cfg
            } else {
                &mut sliding_cfg
            };
            if let Some(prev) = *slot {
                if !same_attn_config(prev, (spec, scalars)) {
                    anyhow::bail!(
                        "layer {} attention config differs from earlier {} layer",
                        spec.layer_index,
                        if spec.full { "full" } else { "sliding" }
                    );
                }
            } else {
                *slot = Some((spec, scalars));
            }
        }
        let sliding_attn = sliding_cfg
            .map(|(spec, scalars)| build_attention(runtime, spec, scalars))
            .transpose()?;
        let full_attn = full_cfg
            .map(|(spec, scalars)| build_attention(runtime, spec, scalars))
            .transpose()?;
        let hidden = first_spec.hidden;
        Ok(Self {
            sliding_attn,
            full_attn,
            moe: DecodeMoe::new(
                runtime,
                first_spec.experts,
                first_spec.top_k,
                hidden,
                first_spec.intermediate,
                first_scalars.route_scale,
                first_scalars.gate_cap,
                first_scalars.up_cap,
            )?,
            norm: RmsNorm::with_eps(runtime, hidden, first_scalars.rms_eps)?,
            linear: Linear::new(runtime)?,
            tanh: Softcap::new(runtime, hidden, 1.0)?,
            depth: DepthMix::new(runtime)?,
            post: ShortConv::stateless(runtime, hidden, first_spec.post_kernel)?,
            ops: goldy::TensorKernels::new(runtime)?,
            hidden,
            gate_rank: first_spec.gate_rank,
            scale: (f64::from(hidden)).powf(-0.5) as f32,
        })
    }

    pub(crate) fn norm(&self) -> &RmsNorm {
        &self.norm
    }

    pub(crate) fn linear(&self) -> &Linear {
        &self.linear
    }

    pub(crate) fn depth(&self) -> &DepthMix {
        &self.depth
    }

    pub(crate) fn ops(&self) -> &goldy::TensorKernels {
        &self.ops
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// Record one layer visit using `spec` and per-visit `weights` / `state`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_layer(
        &self,
        scheme: &mut Scheme,
        spec: DecodeLayerSpec,
        hidden: TensorView<'_>,
        blocks: TensorView<'_>,
        n_blocks: u32,
        messages: Option<[TensorView<'_>; 4]>,
        weights: &DecodeLayerWeights,
        state: &DecodeLayerState,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let h = self.hidden;
        if spec.hidden != h || spec.gate_rank != self.gate_rank {
            return Err(GoldyError::Validation(
                "decoder layer spec does not match shared layer modules".into(),
            ));
        }
        if hidden.shape().dims() != [h] || out.shape().dims() != [h] {
            return Err(GoldyError::Validation(
                "decoder layer hidden and out are [hidden]".into(),
            ));
        }
        if blocks.shape().rank() != 2 || blocks.shape().dim(1)? != h {
            return Err(GoldyError::Validation(
                "decoder layer blocks are [capacity, hidden]".into(),
            ));
        }
        let capacity = blocks.shape().dim(0)?;
        let mlp_blocks = n_blocks + u32::from(spec.boundary);
        if mlp_blocks > capacity {
            return Err(GoldyError::Validation(
                "decoder layer block tape is shorter than the boundary append".into(),
            ));
        }
        if messages.is_some() && !spec.messages {
            return Err(GoldyError::Validation(
                "decoder layer has no message gates for this visit".into(),
            ));
        }
        let attn = if spec.full {
            self.full_attn.as_ref().ok_or_else(|| {
                GoldyError::Validation("shared layer modules have no full attention".into())
            })?
        } else {
            self.sliding_attn.as_ref().ok_or_else(|| {
                GoldyError::Validation("shared layer modules have no sliding attention".into())
            })?
        };
        let f = TensorDType::F32;
        let vec_h = || TensorShape::vector(h);
        let slab = weights.slab.as_ref();
        let ws = &weights.slices;

        let (qn, kn, vn) = self
            .prepare_attention_streams(scheme, spec, hidden, blocks, n_blocks, messages, weights)?;
        let attn_out = scheme.temporary_tensor(vec_h(), f)?;
        attn.record(scheme, qn, kn, vn, weights.views(), state, step, attn_out)?;
        let post_attn = state.post_attn().ok_or_else(|| {
            GoldyError::Validation("visit is missing the post-attention conv ring".into())
        })?;
        let conv_attn = scheme.temporary_tensor(vec_h(), f)?;
        self.post.record_at_state(
            scheme,
            attn_out,
            ws.post_attn_conv.view(slab),
            post_attn,
            step,
            conv_attn,
        )?;
        let mlp_partial = if spec.boundary {
            self.depth.store_row(scheme, hidden, blocks, n_blocks)?;
            conv_attn
        } else {
            let residual = scheme.temporary_tensor(vec_h(), f)?;
            self.depth.add(scheme, hidden, conv_attn, residual)?;
            residual
        };
        let mlp_depth = scheme.temporary_tensor(TensorShape::matrix(1, h), f)?;
        self.mix(
            scheme,
            blocks,
            mlp_partial,
            ws.mlp_query.view(slab),
            mlp_depth,
            mlp_blocks,
        )?;
        let mixed = mlp_depth.reshape(&[h])?;
        // Pass 0 of a loop layer owns the gates and does not run them.
        let mlp_src = if let Some(msgs) = messages {
            let gate = ws.mlp_gate.as_ref().ok_or_else(|| {
                GoldyError::Validation("decoder layer is missing the MLP message gate".into())
            })?;
            let mlp = scheme.temporary_tensor(vec_h(), f)?;
            self.record_gate(scheme, mixed, msgs[3], gate, slab, mlp)?;
            mlp
        } else {
            mixed
        };
        let mlp_norm = scheme.temporary_tensor(vec_h(), f)?;
        self.norm
            .record(scheme, mlp_src, ws.post_norm.view(slab), mlp_norm)?;
        let moe_out = scheme.temporary_tensor(vec_h(), f)?;
        self.moe.record(
            scheme,
            mlp_norm,
            ws.router.view(slab),
            ws.router_bias.view(slab),
            ws.gate_up.view(slab),
            ws.down.view(slab),
            moe_out,
        )?;
        let post_mlp = state.post_mlp().ok_or_else(|| {
            GoldyError::Validation("visit is missing the post-MLP conv ring".into())
        })?;
        let conv_mlp = scheme.temporary_tensor(vec_h(), f)?;
        self.post.record_at_state(
            scheme,
            moe_out,
            ws.post_mlp_conv.view(slab),
            post_mlp,
            step,
            conv_mlp,
        )?;
        self.depth.add(scheme, mlp_partial, conv_mlp, out)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_attention_streams(
        &self,
        scheme: &mut Scheme,
        spec: DecodeLayerSpec,
        hidden: TensorView<'_>,
        blocks: TensorView<'_>,
        n_blocks: u32,
        messages: Option<[TensorView<'_>; 4]>,
        weights: &DecodeLayerWeights,
    ) -> Result<
        (
            TensorView<'static>,
            TensorView<'static>,
            TensorView<'static>,
        ),
        GoldyError,
    > {
        let h = self.hidden;
        let f = TensorDType::F32;
        let vec_h = || TensorShape::vector(h);
        let slab = weights.slab.as_ref();
        let ws = &weights.slices;

        let depth_out = if spec.layer_index > 0 {
            let queries = ws.attn_query.as_ref().ok_or_else(|| {
                GoldyError::Validation("decoder layer is missing attention_input.query".into())
            })?;
            let nq = if spec.full { 3 } else { 1 };
            let depth_out = scheme.temporary_tensor(TensorShape::matrix(nq, h), f)?;
            self.mix(
                scheme,
                blocks,
                hidden,
                queries.view(slab).narrow(0, 0, nq)?,
                depth_out,
                n_blocks,
            )?;
            Some(depth_out)
        } else {
            None
        };
        let row = |index: u32| -> Result<TensorView<'static>, GoldyError> {
            depth_out
                .expect("checked layer_index > 0")
                .narrow(0, index, 1)?
                .reshape(&[h])
        };

        let qn = scheme.temporary_tensor(vec_h(), f)?;
        let kn = scheme.temporary_tensor(vec_h(), f)?;
        let vn = scheme.temporary_tensor(vec_h(), f)?;

        if let Some(msgs) = messages {
            let src = |index: u32| -> Result<TensorView<'_>, GoldyError> {
                if spec.layer_index == 0 {
                    Ok(hidden)
                } else if spec.full {
                    row(index)
                } else {
                    row(0)
                }
            };
            let q = scheme.temporary_tensor(vec_h(), f)?;
            let k = scheme.temporary_tensor(vec_h(), f)?;
            let v = scheme.temporary_tensor(vec_h(), f)?;
            self.gate_into(scheme, src(0)?, msgs[0], ws.q_gate.as_ref(), slab, q)?;
            self.gate_into(scheme, src(1)?, msgs[1], ws.k_gate.as_ref(), slab, k)?;
            self.gate_into(scheme, src(2)?, msgs[2], ws.v_gate.as_ref(), slab, v)?;
            self.norm_stream(scheme, q, Self::q_weight(spec.full, weights)?, qn)?;
            self.norm_stream(scheme, k, Self::k_weight(spec.full, weights)?, kn)?;
            self.norm_stream(scheme, v, Self::v_weight(spec.full, weights)?, vn)?;
            return Ok((qn, kn, vn));
        }
        if spec.layer_index == 0 {
            if spec.full {
                self.norm_stream(scheme, hidden, Self::q_weight(true, weights)?, qn)?;
                self.norm_stream(scheme, hidden, Self::k_weight(true, weights)?, kn)?;
                self.norm_stream(scheme, hidden, Self::v_weight(true, weights)?, vn)?;
            } else {
                let w = Self::q_weight(false, weights)?;
                self.norm_stream(scheme, hidden, w, qn)?;
                self.norm_stream(scheme, hidden, w, kn)?;
                self.norm_stream(scheme, hidden, w, vn)?;
            }
            return Ok((qn, kn, vn));
        }
        if spec.full {
            self.norm_stream(scheme, row(0)?, Self::q_weight(true, weights)?, qn)?;
            self.norm_stream(scheme, row(1)?, Self::k_weight(true, weights)?, kn)?;
            self.norm_stream(scheme, row(2)?, Self::v_weight(true, weights)?, vn)?;
        } else {
            let w = Self::q_weight(false, weights)?;
            let src = row(0)?;
            self.norm_stream(scheme, src, w, qn)?;
            self.norm_stream(scheme, src, w, kn)?;
            self.norm_stream(scheme, src, w, vn)?;
        }
        Ok((qn, kn, vn))
    }

    fn q_weight(full: bool, weights: &DecodeLayerWeights) -> Result<TensorView<'_>, GoldyError> {
        let slab = weights.slab.as_ref();
        if full {
            weights
                .slices
                .q_norm
                .as_ref()
                .map(|w| w.view(slab))
                .ok_or_else(|| {
                    GoldyError::Validation("full decoder layer is missing q_input_layernorm".into())
                })
        } else {
            weights
                .slices
                .input_norm
                .as_ref()
                .map(|w| w.view(slab))
                .ok_or_else(|| {
                    GoldyError::Validation(
                        "sliding decoder layer is missing input_layernorm".into(),
                    )
                })
        }
    }

    fn k_weight(full: bool, weights: &DecodeLayerWeights) -> Result<TensorView<'_>, GoldyError> {
        let slab = weights.slab.as_ref();
        if full {
            weights
                .slices
                .k_norm
                .as_ref()
                .map(|w| w.view(slab))
                .ok_or_else(|| {
                    GoldyError::Validation("full decoder layer is missing k_input_layernorm".into())
                })
        } else {
            Self::q_weight(false, weights)
        }
    }

    fn v_weight(full: bool, weights: &DecodeLayerWeights) -> Result<TensorView<'_>, GoldyError> {
        let slab = weights.slab.as_ref();
        if full {
            weights
                .slices
                .v_norm
                .as_ref()
                .map(|w| w.view(slab))
                .ok_or_else(|| {
                    GoldyError::Validation("full decoder layer is missing v_input_layernorm".into())
                })
        } else {
            Self::q_weight(false, weights)
        }
    }

    fn norm_stream(
        &self,
        scheme: &mut Scheme,
        x: TensorView<'_>,
        weight: TensorView<'_>,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.norm.record(scheme, x, weight, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_into(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        message: TensorView<'_>,
        gate: Option<&GateSlices>,
        slab: &Tensor,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let gate = gate.ok_or_else(|| {
            GoldyError::Validation("decoder layer is missing a message gate".into())
        })?;
        self.record_gate(scheme, hidden, message, gate, slab, out)
    }

    fn record_gate(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        message: TensorView<'_>,
        gate: &GateSlices,
        slab: &Tensor,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        let h = self.hidden;
        let f = TensorDType::F32;
        let vec_h = || TensorShape::vector(h);
        let cat = scheme.temporary_tensor(TensorShape::matrix(2, h), f)?;
        let rank = scheme.temporary_tensor(TensorShape::vector(self.gate_rank.max(1)), f)?;
        let forget_logits = scheme.temporary_tensor(vec_h(), f)?;
        let forget = scheme.temporary_tensor(vec_h(), f)?;
        let gate_logits = scheme.temporary_tensor(vec_h(), f)?;
        let gate_act = scheme.temporary_tensor(vec_h(), f)?;
        let forgotten = scheme.temporary_tensor(vec_h(), f)?;
        let kept = scheme.temporary_tensor(vec_h(), f)?;
        let gained = scheme.temporary_tensor(vec_h(), f)?;

        let hidden_row = cat.narrow(0, 0, 1)?.reshape(&[h])?;
        let message_row = cat.narrow(0, 1, 1)?.reshape(&[h])?;
        self.norm.record_unweighted(scheme, hidden, hidden_row)?;
        self.norm.record_unweighted(scheme, message, message_row)?;
        let flat = cat.reshape(&[h * 2])?;
        self.linear
            .record(scheme, gate.down.view(slab), flat, rank)?;
        self.linear
            .record(scheme, gate.forget.view(slab), rank, forget_logits)?;
        self.tanh.record(scheme, forget_logits, forget)?;
        self.linear
            .record(scheme, gate.message.view(slab), rank, gate_logits)?;
        self.tanh.record(scheme, gate_logits, gate_act)?;
        let mut rec = self.ops.recorder(scheme);
        rec.mul_into("msg_forget_h", forget, hidden, forgotten)?;
        rec.sub_into("msg_keep", hidden, forgotten, kept)?;
        rec.mul_into("msg_gain_m", gate_act, message, gained)?;
        rec.add_into("msg_mix", kept, gained, out)
    }

    fn mix(
        &self,
        scheme: &mut Scheme,
        blocks: TensorView<'_>,
        partial: TensorView<'_>,
        queries: TensorView<'_>,
        out: TensorView<'_>,
        n_blocks: u32,
    ) -> Result<(), GoldyError> {
        self.depth.mix(
            scheme,
            blocks,
            partial,
            queries,
            out,
            n_blocks,
            DEPTH_RMS_EPS,
            self.scale,
        )
    }
}

/// Stateless child modules for one standalone decoder layer configuration.
pub struct DecodeLayer {
    spec: DecodeLayerSpec,
    modules: SharedLayerModules,
}

impl DecodeLayer {
    pub fn new(
        runtime: &Runtime,
        spec: DecodeLayerSpec,
        scalars: DecodeLayerScalars,
    ) -> anyhow::Result<Self> {
        let modules = SharedLayerModules::new(runtime, &[(spec, scalars)])?;
        Ok(Self { spec, modules })
    }

    pub fn spec(&self) -> DecodeLayerSpec {
        self.spec
    }

    /// Record the layer. `blocks` is `[capacity, hidden]` and the first
    /// `n_blocks` rows are valid. A boundary layer writes the layer input into
    /// row `n_blocks` so the MLP mix can see it; that write stays in `blocks`.
    ///
    /// `messages`, when this layer was built with them, is query, key, value,
    /// and MLP, each `[hidden]`. `step.position` is the token position.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        scheme: &mut Scheme,
        hidden: TensorView<'_>,
        blocks: TensorView<'_>,
        n_blocks: u32,
        messages: Option<[TensorView<'_>; 4]>,
        weights: &DecodeLayerWeights,
        state: &DecodeLayerState,
        step: &Buffer,
        out: TensorView<'_>,
    ) -> Result<(), GoldyError> {
        self.modules.record_layer(
            scheme, self.spec, hidden, blocks, n_blocks, messages, weights, state, step, out,
        )
    }
}
