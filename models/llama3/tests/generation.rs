//! End-to-end greedy generation vs the published llama3.cuda TinyStories sample.
//!
//! Needs `python tools/fetch_assets.py --only required`. The test fails, rather
//! than skips, when the checkpoint or tokenizer is missing.

#![cfg(feature = "gpu")]

use llama3_goldy::checkpoint::Checkpoint;
use llama3_goldy::generate::{generate, DREAM_STORY};
use llama3_goldy::model::Model;
use llama3_goldy::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn asset(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name);
    assert!(
        path.exists(),
        "missing {}; run `python tools/fetch_assets.py --only required` in models/llama3",
        path.display()
    );
    path
}

fn assert_dream(model: &mut Model, tokenizer: &Tokenizer, pass: &str) {
    let out = generate(model, tokenizer, "I have a dream", 50).expect("generate");
    assert_eq!(
        out.text, DREAM_STORY,
        "{pass}: generated text must match the llama3.cuda README sample\n--- got ---\n{}\n--- expected ---\n{}",
        out.text, DREAM_STORY
    );
}

#[test]
fn stories15m_i_have_a_dream() {
    let ckpt = Checkpoint::read_path(asset("stories15M.bin")).expect("checkpoint");
    assert_eq!(ckpt.config.dim, 288);
    assert_eq!(ckpt.config.n_layers, 6);
    assert_eq!(
        ckpt.config.n_kv_heads, ckpt.config.n_heads,
        "stories15M is MHA"
    );
    let tokenizer =
        Tokenizer::from_path(asset("tokenizer.bin"), ckpt.config.vocab_size()).expect("tokenizer");
    let mut model = Model::load(&ckpt).expect("load model on GPU");

    // The first pass runs before any background specialization or fusion compile
    // lands, so it checks the recorded graph as authored.
    assert_dream(&mut model, &tokenizer, "first pass");

    let settle = Instant::now();
    while model.compiles_pending() && settle.elapsed() < Duration::from_secs(300) {
        assert_dream(&mut model, &tokenizer, "settling pass");
    }
    assert!(
        !model.compiles_pending(),
        "compiles still pending after {:?}",
        settle.elapsed()
    );
    // Promoted specializations and fused regions replace the authored dispatches.
    assert_dream(&mut model, &tokenizer, "settled pass");

    let stats = model.replay_stats();
    let fusion = model.fusion_report();
    eprintln!(
        "settled in {:?}: {} fused regions, {} executed nodes per token; {stats:?}",
        settle.elapsed(),
        fusion.regions.len(),
        model.executed_node_count()
    );
    assert_eq!(
        stats.topology_records, 0,
        "DecodeStep deposit must not dirty worker topology"
    );
    assert_eq!(stats.fusion_fallbacks, 0, "{stats:?}");
    assert_eq!(stats.fusion_compile_failures, 0, "{stats:?}");
    assert!(
        !fusion.regions.is_empty(),
        "automatic fusion should form regions; {fusion:?}"
    );
    #[cfg(not(feature = "metal"))]
    assert!(
        stats.resubmit_hits > 0,
        "steady-state submits should hit retained replay; got {stats:?}"
    );
}
