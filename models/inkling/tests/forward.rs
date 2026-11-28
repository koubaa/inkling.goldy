//! Full Mini forward, token by token, against the transformers 5.24.0 logits.
//!
//! `tests/data/mini_model.safetensors` is the module state dict.
//! `tests/data/mini_forward.bin` is the compact golden: header, token ids, logits.

#![cfg(feature = "gpu")]

use ammon::gpu::create_serial_runtime as create_runtime;
use inkling_goldy::layer::DecodeLayerScalars;
use inkling_goldy::model::{ForwardSpec, TextModel};
use inkling_goldy::tape::LoopSchedule;

const WEIGHTS: &[u8] = include_bytes!("data/mini_model.safetensors");
const GOLDEN: &[u8] = include_bytes!("data/mini_forward.bin");

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
        let bytes: [u8; 4] = self.data[self.at..self.at + 4].try_into().unwrap();
        self.at += 4;
        f32::from_le_bytes(bytes)
    }
}

#[test]
fn mini_forward_matches_the_reference_logits() {
    let mut reader = Reader {
        data: GOLDEN,
        at: 0,
    };
    assert_eq!(reader.u32(), 0x494E_4B4D);
    assert_eq!(reader.u32(), 1);
    let hidden = reader.u32();
    let layers = reader.u32();
    let block = reader.u32();
    let loop_start = reader.u32();
    let loop_end = reader.u32();
    let passes = reader.u32();
    let spec = ForwardSpec {
        schedule: LoopSchedule {
            layers,
            block,
            loop_start,
            loop_end,
            passes,
        },
        hidden,
        q_heads: reader.u32(),
        kv_heads: reader.u32(),
        head_dim: reader.u32(),
        d_rel: reader.u32(),
        rel_extent: reader.u32(),
        window: reader.u32(),
        conv_kernel: reader.u32(),
        experts: reader.u32(),
        top_k: reader.u32(),
        intermediate: reader.u32(),
        gate_rank: reader.u32(),
        vocab: reader.u32(),
        seq: reader.u32(),
        scalars: DecodeLayerScalars {
            rms_eps: reader.f32(),
            qk_clip: reader.f32(),
            rel_clip: reader.f32(),
            route_scale: reader.f32(),
            gate_cap: reader.f32(),
            up_cap: reader.f32(),
            log_alpha: 0.0,
            log_floor: 0.0,
        },
        logit_cap: reader.f32(),
    };
    // pack_mini.py writes the logit cap before the log-scale pair.
    let mut spec = spec;
    spec.scalars.log_alpha = reader.f32();
    spec.scalars.log_floor = reader.f32();
    assert_eq!(
        spec.schedule.visits(),
        48,
        "Mini is 24 layers with the loop run three times"
    );
    let mut tokens = Vec::new();
    for _ in 0..spec.seq {
        tokens.push(reader.u32());
    }
    let mut want = Vec::new();
    for _ in 0..spec.seq * spec.vocab {
        want.push(reader.f32());
    }
    assert_eq!(reader.at, reader.data.len());

    // The predictor re-records every warmup step. Replay is about retained
    // command buffers, so this test pins specialization off.
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    let device = create_runtime().expect("goldy runtime");
    let ctx = device.create_context().unwrap();
    let mut model = TextModel::new(&device, spec, WEIGHTS).unwrap();
    model.bind(&ctx).unwrap();
    let mut got = Vec::new();
    for (position, token) in tokens.iter().enumerate() {
        got.extend(model.step(*token, position as u32).unwrap());
    }
    let err = max_abs(&got, &want);
    let rms = (want
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / want.len() as f64)
        .sqrt();
    let limit = 1.0e-5_f64.max(1.0e-4 * rms);
    assert!(
        err <= limit,
        "mini logits max abs {err}, limit {limit}, rms {rms}"
    );
    let stats = model.replay_stats();
    assert!(
        stats.resubmit_hits > 0,
        "token scheme did not replay: {stats:?}"
    );
    assert_eq!(stats.retained_evictions, 0, "{stats:?}");
}

fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
}
