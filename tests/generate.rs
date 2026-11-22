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
        "first {:.3} s replay {:.3} s ({:.2} tok/s) steps {:?}",
        decoded.step_secs[0],
        replay,
        1.0 / replay,
        decoded.step_secs
    );
}
