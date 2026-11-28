"""Summarize abba.sh output as a markdown table.

usage: python3 summarize.py LOG [A_NAME B_NAME]

Per model and build: executed nodes per token, runs, timed steps, and the p10,
median and p90 of per-step tok/s over every timed step of every run, plus the
range of the per-run medians. The change row compares B's median against A's.
"""

import re
import statistics
import sys
from collections import defaultdict

log = sys.argv[1]
names = {
    "a": sys.argv[2] if len(sys.argv) > 2 else "A",
    "b": sys.argv[3] if len(sys.argv) > 3 else "B",
}

steps = defaultdict(list)
run_medians = defaultdict(list)
nodes = defaultdict(set)
failures = []
for line in open(log):
    m = re.match(r"RUN variant=(\w) round=(\d+) (.*)", line.rstrip("\n"))
    if not m:
        continue
    variant, _, rest = m.groups()
    if rest.startswith("STEP_MS "):
        _, model, values = rest.split(" ", 2)
        rates = [1000.0 / float(v) for v in values.split(",")]
        steps[(model, variant)].extend(rates)
        run_medians[(model, variant)].append(statistics.median(rates))
    elif rest.startswith("BENCH "):
        model = rest.split()[1]
        n = re.search(r"executed_nodes/token=(\d+)", rest)
        if n:
            nodes[(model, variant)].add(int(n.group(1)))
    else:
        failures.append(f"{names[variant]}: {rest}")


def quantile(sorted_values, q):
    return sorted_values[round((len(sorted_values) - 1) * q)]


print("| model | build | nodes/token | runs | steps | p10 tok/s | median tok/s | p90 tok/s | run medians |")
print("|---|---|---|---|---|---|---|---|---|")
for model in sorted({k[0] for k in steps}):
    medians = {}
    for variant in ("a", "b"):
        rates = sorted(steps.get((model, variant), []))
        if not rates:
            continue
        medians[variant] = quantile(rates, 0.5)
        runs = run_medians[(model, variant)]
        print(
            f"| {model} | {names[variant]} | {'/'.join(map(str, sorted(nodes[(model, variant)])))} "
            f"| {len(runs)} | {len(rates)} | {quantile(rates, 0.1):.1f} | {medians[variant]:.1f} "
            f"| {quantile(rates, 0.9):.1f} | {min(runs):.1f}-{max(runs):.1f} |"
        )
    if len(medians) == 2:
        change = 100.0 * (medians["b"] / medians["a"] - 1.0)
        print(f"| {model} | median change | | | | | {change:+.1f}% | | |")
for failure in failures:
    print(f"\nFAILED {failure}")
