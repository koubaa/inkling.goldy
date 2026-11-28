//! Steady-state replay timing. Ignored by default; run with
//! `cargo test --release --features <backend> --test bench -- --ignored --nocapture --test-threads=1`.
//!
//! Each step cycles positions `0..seq`, so attention length averages over the window.
//! Timing starts once no scheme has a fusion or specialization compile outstanding; on
//! lavapipe that takes hundreds of steps, so earlier tok/s figures are not steady state.
//! `INKLING_FUSION_REPORT=<visit>` prints the token scheme's fusion regions
//! (visits share that scheme) when `visit` is in range, and
//! `INKLING_HOLD_SECS=<n>` keeps stepping for `n` seconds after warm-up (for `perf -p`),
//! touching `/tmp/inkling_steady` when it starts. `host_cpu` is the stepping thread's CPU
//! time per token, so it separates submit cost from waiting on the device.

#![cfg(feature = "gpu")]

use ammon::gpu::create_runtime;
use inkling_goldy::layer::DecodeLayerScalars;
use inkling_goldy::model::{read_golden, ForwardSpec, TextModel};
use inkling_goldy::tape::LoopSchedule;
use std::time::{Duration, Instant};

fn micro_spec() -> ForwardSpec {
    ForwardSpec {
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
    }
}

/// Nanoseconds this thread has spent on a CPU (Linux schedstat).
fn thread_cpu_ns() -> u64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(0)
}

fn run(name: &str, spec: ForwardSpec, weights: &[u8]) {
    let device = create_runtime().expect("runtime");
    let ctx = device.create_context().unwrap();
    let mut model = TextModel::new(&device, spec, weights).unwrap();
    model.bind(&ctx).unwrap();
    let seq = spec.seq;
    let mut pos = 0u32;
    let mut step = |model: &mut TextModel| {
        let t = Instant::now();
        model.step(1 + pos % 7, pos).unwrap();
        pos = (pos + 1) % seq;
        t.elapsed().as_secs_f64()
    };
    let warm = Instant::now();
    let mut warm_steps = 0;
    while warm.elapsed() < Duration::from_secs(240) {
        step(&mut model);
        warm_steps += 1;
        let (_, _, pending) = model.replay_census().unwrap();
        if !pending && warm_steps >= 2 * seq {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let (schemes, nodes, pending) = model.replay_census().unwrap();
    if let Ok(which) = std::env::var("INKLING_FUSION_REPORT") {
        let index: usize = which.parse().unwrap();
        if let Some((report, executed)) = model.visit_fusion(index) {
            println!("token scheme (visit {index}) executes {executed} nodes");
            for region in &report.regions {
                let labels: Vec<String> = region.labels.iter().map(|l| l.to_string()).collect();
                println!(
                    "  region {:?} {:?} n={} [{}] schedule={:?} cost={:?}",
                    region.status,
                    region.tier,
                    region.nodes.len(),
                    labels.join(", "),
                    region.schedule,
                    region.cost
                );
            }
            for rejected in &report.rejected {
                println!("  rejected {rejected:?}");
            }
        }
    }
    if let Some(hold) = std::env::var("INKLING_HOLD_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        std::fs::write("/tmp/inkling_steady", b"1").unwrap();
        let until = Instant::now() + Duration::from_secs(hold);
        let mut n = 0u64;
        while Instant::now() < until {
            step(&mut model);
            n += 1;
        }
        println!("held {n} steps");
    }
    let before = model.replay_totals().unwrap();
    let cpu_before = thread_cpu_ns();
    let mut times: Vec<f64> = (0..4 * seq).map(|_| step(&mut model)).collect();
    let host_ms = (thread_cpu_ns() - cpu_before) as f64 / 1e6 / f64::from(4 * seq);
    let after = model.replay_totals().unwrap();
    let per = |i: usize| (after[i] - before[i]) as f64 / f64::from(4 * seq);
    println!(
        "per token: records {:.1} topology_records {:.1} clean_submits {:.1} resubmit_hits {:.1} evictions {:.1}",
        per(0),
        per(1),
        per(2),
        per(3),
        per(4)
    );
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "BENCH {name} arithmetic_policy={} lp_threads={} warm_steps={warm_steps} \
         pending={pending} submits/token={schemes} executed_nodes/token={nodes} \
         mean {:.2} ms ({:.2} tok/s) median {:.2} ms min {:.2} ms host_cpu {host_ms:.2} ms",
        std::env::var("AMMON_ARITHMETIC_POLICY").unwrap_or_else(|_| "default".into()),
        std::env::var("LP_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        mean * 1e3,
        1.0 / mean,
        times[times.len() / 2] * 1e3,
        times[0] * 1e3,
    );
}

#[test]
#[ignore]
fn bench_micro() {
    run(
        "micro",
        micro_spec(),
        include_bytes!("data/micro_model.safetensors"),
    );
}

#[test]
#[ignore]
fn bench_mini() {
    let (spec, _, _) = read_golden(include_bytes!("data/mini_forward.bin")).unwrap();
    run("mini", spec, include_bytes!("data/mini_model.safetensors"));
}
