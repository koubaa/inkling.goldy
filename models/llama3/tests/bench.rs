//! Steady-state decode timing for stories15M. Ignored by default; run with
//! `cargo test --release --features vulkan --test bench -- --ignored --nocapture`.
//!
//! Each step feeds the previous argmax and cycles positions `0..seq_len`, so the
//! attention length averages over the whole context. Timing starts once no
//! specialization or fusion compile is outstanding. `host_cpu` is the stepping
//! thread's CPU time per token, so it separates submit cost from waiting on the
//! device.

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
    let pending = model.compiles_pending();
    let before = model.replay_stats();
    let cpu_before = thread_cpu_ns();
    let mut times: Vec<f64> = (0..seq).map(|_| step(&mut model)).collect();
    let host_ms = (thread_cpu_ns() - cpu_before) as f64 / 1e6 / f64::from(seq);
    let after = model.replay_stats();
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "BENCH stories15M arithmetic_policy={} lp_threads={} warm_steps={warm_steps} pending={pending} \
         executed_nodes/token={} fused_regions={} records={} clean_submits/token={:.1} \
         mean {:.2} ms ({:.1} tok/s) median {:.2} ms min {:.2} ms host_cpu {host_ms:.2} ms",
        std::env::var("AMMON_ARITHMETIC_POLICY").unwrap_or_else(|_| "default".into()),
        std::env::var("LP_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        model.executed_node_count(),
        model.fusion_report().regions.len(),
        after.records - before.records,
        (after.clean_submits - before.clean_submits) as f64 / f64::from(seq),
        mean * 1e3,
        1.0 / mean,
        times[times.len() / 2] * 1e3,
        times[0] * 1e3,
    );
}
