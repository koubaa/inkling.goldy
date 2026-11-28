//! Steady-state decode timing for stories15M. Ignored by default; run with
//! `cargo test --release --features vulkan --test bench -- --ignored --nocapture`.
//!
//! Each step feeds the previous argmax and cycles positions `0..seq_len`, so the
//! attention length averages over the whole context. Timing starts once no
//! specialization or fusion compile is outstanding. `host_cpu` is the stepping
//! thread's CPU time per token, so it separates submit cost from waiting on the
//! device. `AMMON_BENCH_SETTLE_STEPS=<n>` steps untimed after promotion before
//! timing, and `AMMON_BENCH_TIMED_STEPS=<n>` times `n` steps instead of `seq_len`.

#![cfg(feature = "gpu")]

use ammon::sample_argmax;
use llama3_goldy::checkpoint::Checkpoint;
use llama3_goldy::model::Model;
use std::path::Path;
use std::time::{Duration, Instant};

/// Nanoseconds this thread has spent on a CPU (Linux schedstat).
fn thread_cpu_ns() -> u64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(0)
}

fn env_steps(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn bench_stories15m() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/stories15M.bin");
    let ckpt = Checkpoint::read_path(&path).expect("fetch assets with tools/fetch_assets.py");
    let mut model = Model::load(&ckpt).expect("load model on GPU");
    let seq = ckpt.config.max_seq_len() as u32;
    let mut token = 1u32;
    let mut pos = 0u32;
    let mut step = |model: &mut Model| {
        let t = Instant::now();
        let logits = model.step(token, pos).unwrap();
        token = sample_argmax(&logits) as u32;
        pos = (pos + 1) % seq;
        if pos == 0 {
            token = 1;
        }
        t.elapsed().as_secs_f64()
    };

    let warm = Instant::now();
    let mut warm_steps = 0u32;
    while warm.elapsed() < Duration::from_secs(300) {
        step(&mut model);
        warm_steps += 1;
        if !model.compiles_pending() && warm_steps >= 2 * seq {
            break;
        }
    }
    let settle = env_steps("AMMON_BENCH_SETTLE_STEPS", 0);
    for _ in 0..settle {
        step(&mut model);
    }
    assert!(
        !model.compiles_pending(),
        "a compile is still pending after {settle} settle steps"
    );
    let timed = env_steps("AMMON_BENCH_TIMED_STEPS", seq);
    let before = model.replay_stats();
    let cpu_before = thread_cpu_ns();
    let mut times: Vec<f64> = (0..timed).map(|_| step(&mut model)).collect();
    let host_ms = (thread_cpu_ns() - cpu_before) as f64 / 1e6 / f64::from(timed);
    let after = model.replay_stats();
    let records = after.records - before.records;
    assert_eq!(records, 0, "the timed window re-recorded a scheme");
    let raw: Vec<String> = times.iter().map(|t| format!("{:.4}", t * 1e3)).collect();
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
    println!(
        "BENCH stories15M arithmetic_policy={} lp_threads={} warm_steps={warm_steps} \
         settle={settle} timed={timed} records={records} executed_nodes/token={} fused_regions={} \
         clean_submits/token={:.1} tok/s p10 {:.2} median {:.2} p90 {:.2} host_cpu {host_ms:.2} ms",
        std::env::var("AMMON_ARITHMETIC_POLICY").unwrap_or_else(|_| "default".into()),
        std::env::var("LP_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        model.executed_node_count(),
        model.fusion_report().regions.len(),
        (after.clean_submits - before.clean_submits) as f64 / f64::from(timed),
        1.0 / at(0.9),
        1.0 / at(0.5),
        1.0 / at(0.1),
    );
    println!("STEP_MS stories15M {}", raw.join(","));
}
