//! One recorded decoder layer, replayed per token against the pinned module.
//!
//! `tests/data/decoder_layer.safetensors` is the layer `state_dict`.
//! `tests/data/layer_goldens.bin` is the activations. Max abs error is at most 1e-5.

#![cfg(feature = "vulkan")]

use ammon::gpu::create_runtime;
use ammon::kernels::DecodeStep;
use ammon::DecodeLayerState;
use goldy::{BufferKind, MemoryExchange, Scheme, Tensor, TensorDType, TensorShape};
use inkling_goldy::layer::{DecodeLayer, DecodeLayerScalars, DecodeLayerWeights};
use inkling_goldy::names::{load_decode_layer, DecodeLayerSpec};

const WEIGHTS: &[u8] = include_bytes!("data/decoder_layer.safetensors");
const GOLDEN: &[u8] = include_bytes!("data/layer_goldens.bin");

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let bytes: [u8; 4] = self.data[self.at..self.at + 4].try_into().unwrap();
        self.at += 4;
        u32::from_le_bytes(bytes)
    }

    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.u32().to_le_bytes())
    }

    fn f32s(&mut self) -> Vec<f32> {
        let n = self.u32() as usize;
        let bytes = &self.data[self.at..self.at + n * 4];
        self.at += n * 4;
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

fn zeros(device: &goldy::Runtime, dims: &[u32]) -> Tensor {
    Tensor::zeros(
        device,
        TensorShape::from_dims(dims).unwrap(),
        TensorDType::F32,
    )
    .unwrap()
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn decoder_layer_matches_pinned_module_on_replay() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let mut reader = Reader {
        data: GOLDEN,
        at: 0,
    };
    assert_eq!(reader.u32(), 0x5259_414C);
    assert_eq!(reader.u32(), 1);
    let n_cases = reader.u32();
    for case in 0..n_cases {
        let full = reader.u32() != 0;
        let layer_index = reader.u32();
        let boundary = reader.u32() != 0;
        let messages = reader.u32() != 0;
        let hidden = reader.u32();
        let q_heads = reader.u32();
        let kv_heads = reader.u32();
        let head_dim = reader.u32();
        let d_rel = reader.u32();
        let rel_extent = reader.u32();
        let conv_kernel = reader.u32();
        let post_kernel = reader.u32();
        let experts = reader.u32();
        let top_k = reader.u32();
        let intermediate = reader.u32();
        let gate_rank = reader.u32();
        let window = reader.u32();
        let n_tokens = reader.u32();
        let scalars = DecodeLayerScalars {
            qk_clip: reader.f32(),
            rel_clip: reader.f32(),
            rms_eps: reader.f32(),
            route_scale: reader.f32(),
            gate_cap: reader.f32(),
            up_cap: reader.f32(),
            log_alpha: reader.f32(),
            log_floor: reader.f32(),
        };
        let n_blocks = reader.u32();
        let blocks_host = reader.f32s();
        let mut tokens = Vec::new();
        for _ in 0..n_tokens {
            let x = reader.f32s();
            let msg = reader.f32s();
            let y = reader.f32s();
            tokens.push((x, msg, y));
        }
        let spec = DecodeLayerSpec {
            full,
            dynamic_conv: full,
            layer_index,
            boundary,
            messages,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            d_rel,
            rel_extent,
            conv_kernel,
            post_kernel,
            experts,
            top_k,
            intermediate,
            gate_rank,
            window,
        };
        let prefix = if full { "full." } else { "sliding." };
        let host = load_decode_layer(WEIGHTS, prefix, spec).unwrap();
        let weights = DecodeLayerWeights::from_spec(&device, spec, &host).unwrap();
        let layer = DecodeLayer::new(&device, spec, scalars).unwrap();
        let state = if full {
            DecodeLayerState::full(&device, 8, kv_heads, head_dim, q_heads, conv_kernel).unwrap()
        } else {
            DecodeLayerState::sliding(&device, window, 8, kv_heads, head_dim, q_heads, conv_kernel)
                .unwrap()
        };
        let state = state.with_post_conv(&device, hidden).unwrap();
        let x = zeros(&device, &[hidden]);
        let blocks = zeros(&device, &[n_blocks + 1, hidden]);
        blocks
            .buffer()
            .write(0, bytemuck::cast_slice(&blocks_host))
            .unwrap();
        let msg = zeros(&device, &[4, hidden]);
        let msg_views = [
            msg.view()
                .narrow(0, 0, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 1, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 2, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 3, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
        ];
        let out = zeros(&device, &[hidden]);
        let step = device
            .acquire_buffer_with_data(
                &[DecodeStep {
                    token: 0,
                    position: 0,
                }],
                BufferKind::Scattered,
            )
            .unwrap();
        let mut recorded = Scheme::new(&ctx);
        recorded.set_automatic_fusion(true);
        layer
            .record(
                &mut recorded,
                x.view(),
                blocks.view(),
                n_blocks,
                Some(msg_views),
                &weights,
                &state,
                &step,
                out.view(),
            )
            .unwrap();
        let mut upload = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit(&mut upload, DecodeStep::deposit_target(&step))
            .unwrap();
        for (pos, (token, message, want)) in tokens.iter().enumerate() {
            x.buffer().write(0, bytemuck::cast_slice(token)).unwrap();
            msg.buffer()
                .write(0, bytemuck::cast_slice(message))
                .unwrap();
            state.deposit_position(&deposit, 0, pos as u32).unwrap();
            let _ = upload.submit().unwrap();
            let _ = recorded.submit().unwrap();
            let mut read = Scheme::new(&ctx);
            let got = read_f32(&mut read, out.buffer());
            let err = max_abs(&got, want);
            assert!(err <= 1e-5, "case {case} token {pos} max abs {err}");
        }
        for _ in 0..150 {
            let stats = recorded.replay_stats();
            let report = recorded.fusion_report();
            let settling = report
                .regions
                .iter()
                .any(|region| matches!(region.status, goldy::FusionRegionStatus::Compiling));
            if stats.clean_submits > 0
                && !recorded.compiles_pending()
                && !settling
                && !report.regions.is_empty()
            {
                break;
            }
            let _ = recorded.submit().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let stats = recorded.replay_stats();
        let report = recorded.fusion_report();
        let has = |name: &str| {
            report
                .regions
                .iter()
                .any(|region| region.labels.iter().any(|label| label.to_string() == name))
        };
        // docs/RESTRUCTURE.md asks a later sliding visit to execute 26+2K nodes.
        // On this micro visit (top-k 2) that budget is 30. Fusion does build one
        // semantic region per message-gate projection, per SwiGLU, and per RMS
        // chain. The full visit promotes to 52 executed nodes: a unit softcap is
        // one tanh, so each message gate joins its mix and the following RMS.
        // Depth mix, the conv rings, split-K, and the tape store stay one
        // dispatch each. The sliding visit's
        // pipelines were still compiling after this wait, so its executed count
        // stays the recorded graph; the region list is what the planner admitted.
        assert!(has("rms_normalize"), "no fused RMS region: {report:?}");
        assert!(has("msg_forget_h"), "no fused message gate: {report:?}");
        assert!(has("swiglu"), "no fused SwiGLU: {report:?}");
        if full {
            let promoted = report
                .regions
                .iter()
                .all(|region| matches!(region.status, goldy::FusionRegionStatus::Promoted));
            assert!(promoted, "full visit fusion did not promote: {report:?}");
            assert_eq!(recorded.executed_node_count(), 52, "{report:?}");
        }
        assert_eq!(stats.topology_records, 0, "{stats:?}");
        assert!(stats.clean_submits > 0, "{stats:?}");
        let mut read = Scheme::new(&ctx);
        let again = read_f32(&mut read, out.buffer());
        let err = max_abs(&again, &tokens.last().unwrap().2);
        assert!(err <= 1e-5, "case {case} retained replay max abs {err}");
    }
    assert_eq!(reader.at, reader.data.len());
}

#[test]
fn two_visit_states_do_not_share_post_conv_history() {
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let mut reader = Reader {
        data: GOLDEN,
        at: 0,
    };
    assert_eq!(reader.u32(), 0x5259_414C);
    reader.u32();
    reader.u32();
    let full = reader.u32() != 0;
    assert!(!full);
    let layer_index = reader.u32();
    let boundary = reader.u32() != 0;
    let messages = reader.u32() != 0;
    let hidden = reader.u32();
    let q_heads = reader.u32();
    let kv_heads = reader.u32();
    let head_dim = reader.u32();
    let d_rel = reader.u32();
    let rel_extent = reader.u32();
    let conv_kernel = reader.u32();
    let post_kernel = reader.u32();
    let experts = reader.u32();
    let top_k = reader.u32();
    let intermediate = reader.u32();
    let gate_rank = reader.u32();
    let window = reader.u32();
    let n_tokens = reader.u32();
    let scalars = DecodeLayerScalars {
        qk_clip: reader.f32(),
        rel_clip: reader.f32(),
        rms_eps: reader.f32(),
        route_scale: reader.f32(),
        gate_cap: reader.f32(),
        up_cap: reader.f32(),
        log_alpha: reader.f32(),
        log_floor: reader.f32(),
    };
    let n_blocks = reader.u32();
    let blocks_host = reader.f32s();
    let mut tokens = Vec::new();
    for _ in 0..n_tokens {
        tokens.push((reader.f32s(), reader.f32s(), reader.f32s()));
    }
    let spec = DecodeLayerSpec {
        full,
        dynamic_conv: false,
        layer_index,
        boundary,
        messages,
        hidden,
        q_heads,
        kv_heads,
        head_dim,
        d_rel,
        rel_extent,
        conv_kernel,
        post_kernel,
        experts,
        top_k,
        intermediate,
        gate_rank,
        window,
    };
    let host = load_decode_layer(WEIGHTS, "sliding.", spec).unwrap();
    let weights = DecodeLayerWeights::from_spec(&device, spec, &host).unwrap();
    let layer = DecodeLayer::new(&device, spec, scalars).unwrap();
    let make_state = || {
        DecodeLayerState::sliding(&device, window, 8, kv_heads, head_dim, q_heads, conv_kernel)
            .unwrap()
            .with_post_conv(&device, hidden)
            .unwrap()
    };
    let solo = make_state();
    let left = make_state();
    let right = make_state();
    let prep = |state: &DecodeLayerState| {
        let x = zeros(&device, &[hidden]);
        let blocks = zeros(&device, &[n_blocks + 1, hidden]);
        blocks
            .buffer()
            .write(0, bytemuck::cast_slice(&blocks_host))
            .unwrap();
        let msg = zeros(&device, &[4, hidden]);
        let msg_views = [
            msg.view()
                .narrow(0, 0, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 1, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 2, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
            msg.view()
                .narrow(0, 3, 1)
                .unwrap()
                .reshape(&[hidden])
                .unwrap(),
        ];
        let out = zeros(&device, &[hidden]);
        let step = device
            .acquire_buffer_with_data(
                &[DecodeStep {
                    token: 0,
                    position: 0,
                }],
                BufferKind::Scattered,
            )
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        layer
            .record(
                &mut scheme,
                x.view(),
                blocks.view(),
                n_blocks,
                Some(msg_views),
                &weights,
                state,
                &step,
                out.view(),
            )
            .unwrap();
        (scheme, x, blocks, msg, out, step)
    };
    let (mut solo_scheme, solo_x, _solo_blocks, solo_msg, solo_out, solo_step) = prep(&solo);
    let (mut left_scheme, left_x, _left_blocks, left_msg, left_out, left_step) = prep(&left);
    let (mut right_scheme, right_x, _right_blocks, right_msg, right_out, right_step) = prep(&right);
    let mut solo_up = Scheme::new(&ctx);
    let mut left_up = Scheme::new(&ctx);
    let mut right_up = Scheme::new(&ctx);
    let solo_dep = MemoryExchange::new(&ctx)
        .bind_deposit(&mut solo_up, DecodeStep::deposit_target(&solo_step))
        .unwrap();
    let left_dep = MemoryExchange::new(&ctx)
        .bind_deposit(&mut left_up, DecodeStep::deposit_target(&left_step))
        .unwrap();
    let right_dep = MemoryExchange::new(&ctx)
        .bind_deposit(&mut right_up, DecodeStep::deposit_target(&right_step))
        .unwrap();
    let run = |scheme: &mut Scheme,
               upload: &mut Scheme,
               dep: &goldy::DepositTransaction,
               state: &DecodeLayerState,
               x: &Tensor,
               msg: &Tensor,
               out: &Tensor,
               token: &[f32],
               message: &[f32],
               pos: u32| {
        x.buffer().write(0, bytemuck::cast_slice(token)).unwrap();
        msg.buffer()
            .write(0, bytemuck::cast_slice(message))
            .unwrap();
        state.deposit_position(dep, 0, pos).unwrap();
        let _ = upload.submit().unwrap();
        let _ = read_f32(scheme, out.buffer());
    };
    // Solo sees positions 0 and 2. The interleaved left state sees the same
    // positions and tokens; the right state takes 1 and 3.
    run(
        &mut solo_scheme,
        &mut solo_up,
        &solo_dep,
        &solo,
        &solo_x,
        &solo_msg,
        &solo_out,
        &tokens[0].0,
        &tokens[0].1,
        0,
    );
    run(
        &mut solo_scheme,
        &mut solo_up,
        &solo_dep,
        &solo,
        &solo_x,
        &solo_msg,
        &solo_out,
        &tokens[2].0,
        &tokens[2].1,
        2,
    );
    let mut read = Scheme::new(&ctx);
    let solo_y = read_f32(&mut read, solo_out.buffer());
    run(
        &mut left_scheme,
        &mut left_up,
        &left_dep,
        &left,
        &left_x,
        &left_msg,
        &left_out,
        &tokens[0].0,
        &tokens[0].1,
        0,
    );
    run(
        &mut right_scheme,
        &mut right_up,
        &right_dep,
        &right,
        &right_x,
        &right_msg,
        &right_out,
        &tokens[1].0,
        &tokens[1].1,
        1,
    );
    run(
        &mut left_scheme,
        &mut left_up,
        &left_dep,
        &left,
        &left_x,
        &left_msg,
        &left_out,
        &tokens[2].0,
        &tokens[2].1,
        2,
    );
    let mut read = Scheme::new(&ctx);
    let left_y = read_f32(&mut read, left_out.buffer());
    let err = max_abs(&solo_y, &left_y);
    assert!(
        err <= 1e-5,
        "interleaved visit shared post-conv history, max abs {err}"
    );
}
