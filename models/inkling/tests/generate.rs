//! Greedy continuation of the committed Mini prompt, token for token with
//! transformers 5.24.0 `generate(..., do_sample=False)`.

#![cfg(feature = "gpu")]

use ammon::gpu::create_serial_runtime as create_runtime;
use inkling_goldy::generate::greedy;
use inkling_goldy::model::{read_golden, TextModel};

const WEIGHTS: &[u8] = include_bytes!("data/mini_model.safetensors");
const GOLDEN: &[u8] = include_bytes!("data/mini_forward.bin");

#[test]
fn greedy_matches_transformers_on_the_mini_prompt() {
    let (spec, tokens, logits) = read_golden(GOLDEN).unwrap();
    let prompt = &tokens[..2];
    let device = create_runtime().expect("runtime");
    let ctx = device.create_context().unwrap();
    let mut model = TextModel::new(&device, spec, WEIGHTS).unwrap();
    model.bind(&ctx).unwrap();
    let decoded = greedy(&mut model, prompt, 2).unwrap();
    let vocab = spec.vocab as usize;
    let want = &logits[vocab..2 * vocab];
    let err = decoded
        .last_logits
        .iter()
        .zip(want)
        .map(|(got, want)| (f64::from(*got) - f64::from(*want)).abs())
        .fold(0.0_f64, f64::max);
    let rms = (want
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / want.len() as f64)
        .sqrt();
    let limit = 1.0e-5_f64.max(1.0e-4 * rms);
    assert!(
        err <= limit,
        "last-prompt logits max abs {err}, limit {limit}"
    );
    // A token id is evidence only when top1 beats top2 by more than the logit
    // tolerance. The known Mini continuation is [30, 30]; a closer race is
    // checked by the logits above.
    let expected = [30, 30];
    for ((got, margin), want) in decoded
        .new_tokens
        .iter()
        .zip(&decoded.margins)
        .zip(expected)
    {
        if *margin > 1.0e-4 {
            assert_eq!(*got, want, "margin {margin}");
        }
    }
    assert_eq!(decoded.decode_steps, 1);
    let replay = decoded
        .step_secs
        .iter()
        .skip(1)
        .copied()
        .fold(f64::MAX, f64::min);
    eprintln!(
        "mini first {:.3} s replay {:.3} s ({:.2} tok/s) steps {:?} margins {:?}",
        decoded.step_secs[0],
        replay,
        1.0 / replay,
        decoded.step_secs,
        decoded.margins
    );
}

#[test]
fn greedy_matches_transformers_on_inkling2_micro() {
    use inkling_goldy::layer::DecodeLayerScalars;
    use inkling_goldy::model::ForwardSpec;
    use inkling_goldy::tape::LoopSchedule;
    // inkling2/micro: 12 layers, no loop. Prompt is the reference dump's first two ids.
    let spec = ForwardSpec {
        schedule: LoopSchedule {
            layers: 12,
            block: 6,
            loop_start: 12,
            loop_end: 12,
            passes: 0,
        },
        hidden: 64,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 32,
        d_rel: 8,
        rel_extent: 16,
        window: 8,
        conv_kernel: 4,
        experts: 8,
        top_k: 2,
        intermediate: 64,
        gate_rank: 1,
        vocab: 250,
        seq: 8,
        scalars: DecodeLayerScalars {
            rms_eps: 1e-5,
            qk_clip: 0.15,
            rel_clip: 0.35,
            route_scale: 8.0,
            gate_cap: 0.05,
            up_cap: 0.05,
            log_alpha: 0.1,
            log_floor: 4.0,
        },
        logit_cap: 0.05,
    };
    let device = create_runtime().expect("runtime");
    let ctx = device.create_context().unwrap();
    let mut model = TextModel::new(
        &device,
        spec,
        include_bytes!("data/micro_model.safetensors"),
    )
    .unwrap();
    model.bind(&ctx).unwrap();
    let decoded = greedy(&mut model, &[197, 247], 2).unwrap();
    let expected = [79, 79];
    for ((got, margin), want) in decoded
        .new_tokens
        .iter()
        .zip(&decoded.margins)
        .zip(expected)
    {
        if *margin > 1.0e-4 {
            assert_eq!(*got, want, "margin {margin}");
        }
    }
    eprintln!("micro margins {:?}", decoded.margins);
    let replay = decoded
        .step_secs
        .iter()
        .skip(1)
        .copied()
        .fold(f64::MAX, f64::min);
    eprintln!(
        "micro first {:.3} s replay {:.3} s ({:.2} tok/s) steps {:?}",
        decoded.step_secs[0],
        replay,
        1.0 / replay,
        decoded.step_secs
    );
}

#[test]
fn prefill_scheme_skips_head_and_matches_step_by_step_on_micro() {
    use inkling_goldy::layer::DecodeLayerScalars;
    use inkling_goldy::model::ForwardSpec;
    use inkling_goldy::tape::LoopSchedule;

    let spec = ForwardSpec {
        schedule: LoopSchedule {
            layers: 12,
            block: 6,
            loop_start: 12,
            loop_end: 12,
            passes: 0,
        },
        hidden: 64,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 32,
        d_rel: 8,
        rel_extent: 16,
        window: 8,
        conv_kernel: 4,
        experts: 8,
        top_k: 2,
        intermediate: 64,
        gate_rank: 1,
        vocab: 250,
        seq: 8,
        scalars: DecodeLayerScalars {
            rms_eps: 1e-5,
            qk_clip: 0.15,
            rel_clip: 0.35,
            route_scale: 8.0,
            gate_cap: 0.05,
            up_cap: 0.05,
            log_alpha: 0.1,
            log_floor: 4.0,
        },
        logit_cap: 0.05,
    };
    let weights = include_bytes!("data/micro_model.safetensors");
    let prompt = [197u32, 247, 13, 30];

    let device = create_runtime().expect("runtime");
    let ctx = device.create_context().unwrap();

    let mut prefill_model = TextModel::new(&device, spec, weights).unwrap();
    prefill_model.bind(&ctx).unwrap();
    let (_, prefill_nodes, _) = prefill_model.prefill_replay_census().unwrap();
    let (_, decode_nodes, _) = prefill_model.replay_census().unwrap();
    assert!(
        prefill_nodes < decode_nodes,
        "prefill_token ({prefill_nodes} nodes) must skip finish + HostSink vs token ({decode_nodes} nodes)"
    );

    for (pos, &tok) in prompt[..prompt.len() - 1].iter().enumerate() {
        prefill_model.advance(tok, pos as u32).unwrap();
    }
    let got_last = prefill_model
        .step(prompt[prompt.len() - 1], (prompt.len() - 1) as u32)
        .unwrap();
    assert!(
        prefill_model.prefill_replay_stats().records >= 1,
        "prefill_token should execute prompt tokens: {:?}",
        prefill_model.prefill_replay_stats()
    );
    assert_eq!(
        prefill_model.replay_stats().records,
        1,
        "decode token scheme should record once on the final prompt token"
    );
    assert_eq!(
        prefill_model.replay_stats().clean_submits,
        0,
        "decode token scheme should run only on the final prompt token"
    );

    let mut step_model = TextModel::new(&device, spec, weights).unwrap();
    step_model.bind(&ctx).unwrap();
    let mut want_last = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        want_last = step_model.step(tok, pos as u32).unwrap();
    }

    let err = got_last
        .iter()
        .zip(&want_last)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err <= 1.0e-5,
        "prefill advance + final step drifted from step-by-step forward: max_abs={err}"
    );
}
