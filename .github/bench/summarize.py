"""Summarize abba.sh output as a markdown table.

usage: python3 summarize.py LOG [A_NAME B_NAME]

Per model and policy: executed nodes per token, then the mean, median and minimum
step time over every timed step of every run (each run's mean, median and min
combined as mean of means, median of medians and min of mins), and B's change
against A.
"""

import re
import statistics
import sys
from collections import defaultdict

log = sys.argv[1]
names = {"a": sys.argv[2] if len(sys.argv) > 2 else "A", "b": sys.argv[3] if len(sys.argv) > 3 else "B"}

runs = defaultdict(list)
pattern = re.compile(
    r"RUN variant=(\w) policy=([\w-]+) round=\d+ BENCH (\S+) .*?"
    r"executed_nodes/token=(\d+).*?mean ([\d.]+) ms .*?median ([\d.]+) ms min ([\d.]+) ms"
)
for line in open(log):
    m = pattern.search(line)
    if m:
        variant, policy, model, nodes, mean, median, low = m.groups()
        runs[(model, policy, variant)].append((int(nodes), float(mean), float(median), float(low)))


def tps(ms):
    return 1000.0 / ms


print("| model | policy | build | runs | nodes/token | mean ms (tok/s) | median ms (tok/s) | min ms (tok/s) |")
print("|---|---|---|---|---|---|---|---|")
for model, policy in sorted({(k[0], k[1]) for k in runs}):
    rows = {}
    for variant in ("a", "b"):
        samples = runs.get((model, policy, variant))
        if not samples:
            continue
        nodes = sorted({s[0] for s in samples})
        mean = statistics.mean(s[1] for s in samples)
        median = statistics.median(s[2] for s in samples)
        low = min(s[3] for s in samples)
        rows[variant] = (mean, median, low)
        print(
            f"| {model} | {policy} | {names[variant]} | {len(samples)} | {'/'.join(map(str, nodes))} "
            f"| {mean:.2f} ({tps(mean):.1f}) | {median:.2f} ({tps(median):.1f}) | {low:.2f} ({tps(low):.1f}) |"
        )
    if len(rows) == 2:
        delta = [100.0 * (rows["a"][i] / rows["b"][i] - 1.0) for i in range(3)]
        print(
            f"| {model} | {policy} | tok/s change | | | {delta[0]:+.1f}% | {delta[1]:+.1f}% | {delta[2]:+.1f}% |"
        )
