//! Greedy decode and logits against Hugging Face transformers.
//!
//! `tests/data/reference.json` is `tools/ref/reference.py`'s float32
//! `LlamaForCausalLM` run of the pinned checkpoint: for each prompt, the greedy
//! continuation and the top logits at every position. The model decodes each
//! prompt one token at a time from position 0 and must pick the same tokens,
//! with every reference top logit within [`LOGIT_ATOL`]. That holds on the
//! first pass, before any specialization or fusion compile lands, and again
//! once they have settled.
//!
//! Needs `python3 tools/fetch_assets.py`; fails, rather than skips, without it.

#![cfg(feature = "gpu")]

use ammon::sample_argmax;
use serde_json::Value;
use smollm2_goldy::model::Model;
use std::time::{Duration, Instant};

/// Largest absolute logit difference allowed at the reference's top ids.
const LOGIT_ATOL: f32 = 5e-3;
const SEQ_LEN: u32 = 32;

struct Position {
    top_ids: Vec<u32>,
    top_logits: Vec<f32>,
}

struct Case {
    prompt: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    positions: Vec<Position>,
}

fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn fixture() -> Vec<Case> {
    let json: Value = serde_json::from_str(include_str!("data/reference.json")).unwrap();
    assert_eq!(json["revision"], smollm2_goldy::REVISION);
    assert_eq!(json["dtype"], "float32");
    json["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| Case {
            prompt: case["prompt"].as_str().unwrap().to_string(),
            prompt_ids: u32s(&case["prompt_ids"]),
            greedy_ids: u32s(&case["greedy_ids"]),
            positions: case["positions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| Position {
                    top_ids: u32s(&p["top_ids"]),
                    top_logits: p["top_logits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect(),
                })
                .collect(),
        })
        .collect()
}

/// Decode every case; return the largest logit error seen.
fn check_pass(model: &mut Model, cases: &[Case], pass: &str) -> f32 {
    let mut worst = 0.0f32;
    for case in cases {
        let mut token = case.prompt_ids[0];
        let mut generated = Vec::new();
        for (pos, want) in case.positions.iter().enumerate() {
            let logits = model.step_view(token, pos as u32).expect("decode step");
            for (&id, &reference) in want.top_ids.iter().zip(&want.top_logits) {
                let err = (logits[id as usize] - reference).abs();
                worst = worst.max(err);
                assert!(
                    err <= LOGIT_ATOL,
                    "{pass}: {:?} position {pos} token {id}: logit {} vs reference {reference}",
                    case.prompt,
                    logits[id as usize]
                );
            }
            let argmax = sample_argmax(&logits) as u32;
            drop(logits);
            token = match case.prompt_ids.get(pos + 1) {
                Some(&next) => next,
                None => {
                    generated.push(argmax);
                    argmax
                }
            };
        }
        // The last generated token is chosen from the last position's logits.
        assert_eq!(
            generated, case.greedy_ids,
            "{pass}: greedy continuation of {:?} diverged from transformers",
            case.prompt
        );
    }
    worst
}

#[test]
fn greedy_decode_matches_transformers() {
    let cases = fixture();
    assert!(!cases.is_empty());
    for case in &cases {
        assert_eq!(
            case.positions.len(),
            case.prompt_ids.len() + case.greedy_ids.len() - 1
        );
    }
    let mut model = Model::load_dir(smollm2_goldy::assets_dir(), SEQ_LEN)
        .unwrap_or_else(|e| panic!("{e:#}; run `python3 tools/fetch_assets.py` in models/smollm2"));
    assert_eq!(model.config.q_heads, 9);
    assert_eq!(model.config.kv_heads, 3);
    assert!(model.config.tie_word_embeddings);

    let first = check_pass(&mut model, &cases, "first pass");
    let settle = Instant::now();
    let mut settling = 0.0f32;
    while model.compiles_pending() && settle.elapsed() < Duration::from_secs(600) {
        settling = settling.max(check_pass(&mut model, &cases, "settling pass"));
    }
    assert!(
        !model.compiles_pending(),
        "compiles still pending after {:?}",
        settle.elapsed()
    );
    let settled = check_pass(&mut model, &cases, "settled pass");

    let stats = model.replay_stats();
    let fusion = model.fusion_report();
    let policy = ammon::inference_arithmetic_policy().unwrap();
    println!(
        "REFERENCE smollm2 policy={policy:?} cases={} max_abs_logit_err first={first:.2e} \
         settling={settling:.2e} settled={settled:.2e} greedy=match settle={:?} \
         fused_regions={} executed_nodes/token={}",
        cases.len(),
        settle.elapsed(),
        fusion.regions.len(),
        model.executed_node_count()
    );
    assert_eq!(stats.topology_records, 0, "{stats:?}");
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
