#!/bin/bash
# Paired decode benchmark of two builds, A and B, under both arithmetic policies.
#
# usage: abba.sh BIN_DIR ROUNDS [MODEL...]
#
# BIN_DIR holds bench test binaries named {a,b}-inkling and {a,b}-llama3. Each run
# is a fresh process. Round r runs A B B A (B A A B on even rounds) for each
# policy, and the policy order alternates too, so drift on a shared runner falls
# on both builds alike. MODEL is micro, mini or stories15m (default all three).
set -euo pipefail
bin_dir=$1
rounds=$2
shift 2
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(micro mini stories15m)

for model in "${models[@]}"; do
  case $model in
    micro) exe=inkling; test=bench_micro ;;
    mini) exe=inkling; test=bench_mini ;;
    stories15m) exe=llama3; test=bench_stories15m ;;
    *) echo "unknown model $model" >&2; exit 2 ;;
  esac
  for round in $(seq "$rounds"); do
    if [ $((round % 2)) -eq 1 ]; then
      policies=(match-recorded relaxed)
      order=(a b b a)
    else
      policies=(relaxed match-recorded)
      order=(b a a b)
    fi
    for policy in "${policies[@]}"; do
      for variant in "${order[@]}"; do
        line=$(AMMON_ARITHMETIC_POLICY=$policy "$bin_dir/$variant-$exe" \
          --ignored --nocapture --test-threads=1 --exact "$test" 2>&1 | grep -o 'BENCH.*')
        echo "RUN variant=$variant policy=$policy round=$round $line"
      done
    done
  done
done
