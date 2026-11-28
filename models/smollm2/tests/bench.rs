//! Steady-state decode timing for SmolLM2-135M under Relaxed arithmetic. Ignored
//! by default; run with
//! `cargo test --release --features <backend> --test bench -- --ignored --nocapture`.
//!
//! Each step feeds the previous argmax and cycles positions `0..BENCH_SEQ`, so
//! the attention length averages over the whole context. Timing starts once no
//! specialization or fusion compile is outstanding, and the timed window must
//! not record. `host_cpu` is the stepping thread's CPU time per token, so it
//! separates submit cost from waiting on the device.

#![cfg(feature = "gpu")]

use ammon::sample_argmax;
use smollm2_goldy::model::Model;
use std::time::{Duration, Instant};

/// Context the bench cycles over, the same as the TinyStories bench.
const BENCH_SEQ: u32 = 256;
/// `<|endoftext|>`, which also starts each cycle.
const START: u32 = 0;

/// Nanoseconds this thread has spent on a CPU (Linux schedstat).
fn thread_cpu_ns() -> u64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(0)
}

#[test]
#[ignore]
fn bench_smollm2_135m() {
    let policy = ammon::inference_arithmetic_policy().unwrap();
    assert_eq!(
        policy,
        goldy::ArithmeticPolicy::Relaxed,
        "model benchmarks run under Relaxed arithmetic; unset {}",
        ammon::ARITHMETIC_POLICY_ENV
    );
    let mut model = Model::load_dir(smollm2_goldy::assets_dir(), BENCH_SEQ)
        .expect("fetch assets with tools/fetch_assets.py");
    let mut token = START;
    let mut pos = 0u32;
    let mut step = |model: &mut Model| {
        let t = Instant::now();
        let logits = model.step_view(token, pos).unwrap();
        token = sample_argmax(&logits) as u32;
        drop(logits);
        pos = (pos + 1) % BENCH_SEQ;
        if pos == 0 {
            token = START;
        }
        t.elapsed().as_secs_f64()
    };

    let warm = Instant::now();
    let mut warm_steps = 0u32;
    while warm.elapsed() < Duration::from_secs(600) {
        step(&mut model);
        warm_steps += 1;
        if !model.compiles_pending() && warm_steps >= BENCH_SEQ {
            break;
        }
    }
    let settle_steps = std::env::var("AMMON_BENCH_SETTLE_STEPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(64);
    for _ in 0..settle_steps {
        step(&mut model);
    }
    while warm.elapsed() < Duration::from_secs(600) {
        if !model.compiles_pending() {
            break;
        }
        step(&mut model);
    }
    assert!(
        !model.compiles_pending(),
        "compiles still pending after warmup and settle"
    );
    let timed_steps = std::env::var("AMMON_BENCH_TIMED_STEPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(BENCH_SEQ);
    let before = model.replay_stats();
    let cpu_before = thread_cpu_ns();
    let mut times: Vec<f64> = (0..timed_steps).map(|_| step(&mut model)).collect();
    let host_ms = (thread_cpu_ns() - cpu_before) as f64 / 1e6 / f64::from(timed_steps);
    let after = model.replay_stats();
    let records = after.records - before.records;
    assert_eq!(
        records, 0,
        "steady-state window recorded {records} times (not amortized)"
    );
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let p10 = times[times.len() / 10];
    let p90 = times[(times.len() * 9) / 10];
    println!(
        "BENCH smollm2-135M policy={policy:?} lp_threads={} warm_steps={warm_steps} settle_steps={settle_steps} \
         timed_steps={timed_steps} executed_nodes/token={} fused_regions={} records={records} \
         clean_submits/token={:.1} median {:.2} ms ({:.1} tok/s) p10 {:.2} ms ({:.1} tok/s) \
         p90 {:.2} ms ({:.1} tok/s) mean {:.2} ms min {:.2} ms host_cpu {host_ms:.2} ms",
        std::env::var("LP_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        model.executed_node_count(),
        model.fusion_report().regions.len(),
        (after.clean_submits - before.clean_submits) as f64 / f64::from(timed_steps),
        median * 1e3,
        1.0 / median,
        p10 * 1e3,
        1.0 / p10,
        p90 * 1e3,
        1.0 / p90,
        mean * 1e3,
        times[0] * 1e3,
    );
}
