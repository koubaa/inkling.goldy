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

/// Nanoseconds this thread has spent on a CPU (`CLOCK_THREAD_CPUTIME_ID`).
#[cfg(unix)]
fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid writable `timespec` for `clock_gettime`.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc == 0 {
        (ts.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(ts.tv_nsec as u64)
    } else {
        0
    }
}

#[cfg(not(unix))]
fn thread_cpu_ns() -> u64 {
    0
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
        let logits = model.step_view(1 + pos % 7, pos).unwrap();
        std::hint::black_box(&*logits);
        drop(logits);
        pos = (pos + 1) % seq;
        t.elapsed().as_secs_f64()
    };
    let warm = Instant::now();
    let mut warm_steps = 0u32;
    while warm.elapsed() < Duration::from_secs(240) {
        step(&mut model);
        warm_steps += 1;
        let (_, _, pending) = model.replay_census().unwrap();
        if !pending && warm_steps >= 2 * seq {
            break;
        }
    }
    let settle_steps = std::env::var("AMMON_BENCH_SETTLE_STEPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(|| (4 * seq).max(32));
    for _ in 0..settle_steps {
        step(&mut model);
    }
    while warm.elapsed() < Duration::from_secs(240) {
        let (_, _, pending) = model.replay_census().unwrap();
        if !pending {
            break;
        }
        step(&mut model);
    }
    let (schemes, nodes, pending) = model.replay_census().unwrap();
    assert!(!pending, "compiles still pending after warmup and settle");
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
    let timed_steps = std::env::var("AMMON_BENCH_TIMED_STEPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(if name == "micro" { 128 } else { 64 });
    let before = model.replay_totals().unwrap();
    let cpu_before = thread_cpu_ns();
    let mut times: Vec<f64> = (0..timed_steps).map(|_| step(&mut model)).collect();
    let host_ms = (thread_cpu_ns() - cpu_before) as f64 / 1e6 / f64::from(timed_steps);
    let after = model.replay_totals().unwrap();
    let records = after[0] - before[0];
    let topology_records = after[1] - before[1];
    assert_eq!(
        records, 0,
        "steady-state window recorded {records} times (not amortized)"
    );
    assert_eq!(
        topology_records, 0,
        "steady-state window re-recorded topology {topology_records} times"
    );
    let per = |i: usize| (after[i] - before[i]) as f64 / f64::from(timed_steps);
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
    let median = times[times.len() / 2];
    let p10 = times[times.len() / 10];
    let p90 = times[(times.len() * 9) / 10];
    println!(
        "BENCH {name} lp_threads={} warm_steps={warm_steps} settle_steps={settle_steps} timed_steps={timed_steps} \
         submits/token={schemes} executed_nodes/token={nodes} records={records} \
         median {:.2} ms ({:.2} tok/s) mean {:.2} ms ({:.2} tok/s) p10 {:.2} ms p90 {:.2} ms min {:.2} ms host_cpu {host_ms:.2} ms",
        std::env::var("LP_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        median * 1e3,
        1.0 / median,
        mean * 1e3,
        1.0 / mean,
        p10 * 1e3,
        p90 * 1e3,
        times[0] * 1e3,
    );

    let prefill_warm = Instant::now();
    let mut prefill_warm_steps = 0u32;
    while prefill_warm.elapsed() < Duration::from_secs(120) {
        for _ in 0..seq.max(4) {
            model.advance(1 + pos % 7, pos).unwrap();
            pos = (pos + 1) % seq;
            prefill_warm_steps += 1;
        }
        ctx.wait_until_idle().unwrap();
        let (_, _, pending) = model.prefill_replay_census().unwrap();
        if !pending && prefill_warm_steps >= 2 * seq {
            break;
        }
    }
    for _ in 0..settle_steps {
        model.advance(1 + pos % 7, pos).unwrap();
        pos = (pos + 1) % seq;
    }
    ctx.wait_until_idle().unwrap();
    while prefill_warm.elapsed() < Duration::from_secs(120) {
        let (_, _, pending) = model.prefill_replay_census().unwrap();
        if !pending {
            break;
        }
        model.advance(1 + pos % 7, pos).unwrap();
        pos = (pos + 1) % seq;
        ctx.wait_until_idle().unwrap();
    }
    let (prefill_schemes, prefill_nodes, prefill_pending) = model.prefill_replay_census().unwrap();
    assert!(
        !prefill_pending,
        "prefill compiles still pending after warmup and settle"
    );
    let prefill_before = model.prefill_replay_totals().unwrap();
    let prefill_cpu_before = thread_cpu_ns();
    let prefill_wall = Instant::now();
    for _ in 0..timed_steps {
        model.advance(1 + pos % 7, pos).unwrap();
        pos = (pos + 1) % seq;
    }
    ctx.wait_until_idle().unwrap();
    let prefill_elapsed = prefill_wall.elapsed().as_secs_f64();
    let prefill_host_ms =
        (thread_cpu_ns() - prefill_cpu_before) as f64 / 1e6 / f64::from(timed_steps);
    let prefill_after = model.prefill_replay_totals().unwrap();
    let prefill_records = prefill_after[0] - prefill_before[0];
    let prefill_topology = prefill_after[1] - prefill_before[1];
    assert_eq!(
        prefill_records, 0,
        "prefill steady-state window recorded {prefill_records} times"
    );
    assert_eq!(
        prefill_topology, 0,
        "prefill steady-state window re-recorded topology {prefill_topology} times"
    );
    let prefill_mean = prefill_elapsed / f64::from(timed_steps);
    println!(
        "BENCH {name}_prefill lp_threads={} warm_steps={prefill_warm_steps} settle_steps={settle_steps} timed_steps={timed_steps} \
         submits/token={prefill_schemes} executed_nodes/token={prefill_nodes} records={prefill_records} \
         mean {:.2} ms ({:.2} tok/s) host_cpu {prefill_host_ms:.2} ms",
        std::env::var("LP_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        prefill_mean * 1e3,
        1.0 / prefill_mean,
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

#[cfg(unix)]
#[test]
fn thread_cpu_timer_advances_under_host_work() {
    let start = thread_cpu_ns();
    let mut acc = 1u64;
    let wall = Instant::now();
    while wall.elapsed() < Duration::from_millis(5) {
        for i in 1..10_000u64 {
            acc = std::hint::black_box(acc.wrapping_mul(6364136223846793005).wrapping_add(i));
        }
    }
    std::hint::black_box(acc);
    let end = thread_cpu_ns();
    assert!(
        end > start && start > 0,
        "CLOCK_THREAD_CPUTIME_ID did not advance: start={start}ns end={end}ns"
    );
}
