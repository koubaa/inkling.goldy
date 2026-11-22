//! Greedy continuation of the committed Mini prompt, token for token with
//! transformers 5.24.0 `generate(..., do_sample=False)`.

#![cfg(feature = "vulkan")]

use ammon::gpu::create_runtime;
use inkling_goldy::generate::greedy;
use inkling_goldy::model::{read_golden, TextModel};

const WEIGHTS: &[u8] = include_bytes!("data/mini_model.safetensors");
const GOLDEN: &[u8] = include_bytes!("data/mini_forward.bin");

#[test]
fn greedy_matches_transformers_on_the_mini_prompt() {
    let (spec, tokens, _) = read_golden(GOLDEN).unwrap();
    let prompt = &tokens[..2];
    let device = create_runtime().expect("runtime");
    let ctx = device.create_context().unwrap();
    let mut model = TextModel::new(&device, spec, WEIGHTS).unwrap();
    model.bind(&ctx).unwrap();
    let decoded = greedy(&mut model, prompt, 2).unwrap();
    // transformers Inkling2LoopForCausalLM.generate on this checkpoint,
    // prompt [30, 13], max_new_tokens=2, do_sample=False.
    assert_eq!(decoded.new_tokens, vec![30, 30], "{decoded:?}");
    assert_eq!(decoded.decode_steps, 1);
    let replay = decoded
        .step_secs
        .iter()
        .skip(1)
        .copied()
        .fold(f64::MAX, f64::min);
    eprintln!(
        "mini first {:.3} s replay {:.3} s ({:.2} tok/s) steps {:?}",
        decoded.step_secs[0],
        replay,
        1.0 / replay,
        decoded.step_secs
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
    assert_eq!(decoded.new_tokens, vec![79, 79], "{decoded:?}");
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
