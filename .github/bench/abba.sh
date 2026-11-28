#!/bin/bash
# Paired steady-state decode benchmark of two builds, A and B, under Ammon's
# inference arithmetic policy (relaxed).
#
# usage: abba.sh BIN_DIR ROUNDS [MODEL...]
#
# BIN_DIR holds bench test binaries named {a,b}-inkling and {a,b}-llama3. Each run
# is a fresh process. Round r runs A B B A (B A A B on even rounds), so drift on a
# shared runner falls on both builds alike. MODEL is micro, mini or stories15m
# (default all three).
set -euo pipefail
bin_dir=$1
rounds=$2
shift 2
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(micro mini stories15m)
export AMMON_ARITHMETIC_POLICY=relaxed

for model in "${models[@]}"; do
  case $model in
    micro) exe=inkling; test=bench_micro ;;
    mini) exe=inkling; test=bench_mini ;;
    stories15m) exe=llama3; test=bench_stories15m ;;
    *) echo "unknown model $model" >&2; exit 2 ;;
  esac
  for round in $(seq "$rounds"); do
    if [ $((round % 2)) -eq 1 ]; then order=(a b b a); else order=(b a a b); fi
    for variant in "${order[@]}"; do
      "$bin_dir/$variant-$exe" --ignored --nocapture --test-threads=1 --exact "$test" 2>&1 \
        | grep -E '^(BENCH|STEP_MS)|panicked' | sed "s/^/RUN variant=$variant round=$round /" || true
    done
  done
done
