#!/usr/bin/env python3
"""Plot results_cpu.csv (from bench_cpu.c) -> runtime_cpu.png
Usage: python3 plot.py [results_cpu.csv] [runtime_cpu.png]
"""

import csv
import statistics
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "results_cpu.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "runtime_cpu.png"

trials = defaultdict(list)
with open(src) as f:
    for row in csv.DictReader(f):
        trials[int(row["N"])].append(float(row["time_s"]))

ns = np.array(sorted(trials))
med = np.array([statistics.median(trials[n]) for n in ns])
lo = med - np.array([min(trials[n]) for n in ns])
hi = np.array([max(trials[n]) for n in ns]) - med

slope = np.polyfit(np.log(ns), np.log(med), 1)[0]
print(f"Log-log slope: {slope:.2f} (ideal O(N^3) = 3.00)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.errorbar(ns, med, yerr=[lo, hi], fmt="o-", capsize=3, label="CPU (median, min-max)")
ax1.set_xlabel("N")
ax1.set_ylabel("Runtime (s)")
ax1.set_title("Runtime vs N")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.loglog(ns, med, "o-", label=f"CPU (slope = {slope:.2f})")
ax2.loglog(ns, med[0] * (ns / ns[0]) ** 3, "--", color="gray", label="O(N³) reference")
ax2.set_xlabel("N")
ax2.set_ylabel("Runtime (s)")
ax2.set_title("Runtime vs N (log-log)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

fig.suptitle("CPU Matrix Multiplication")
fig.tight_layout()
fig.savefig(dst, dpi=150)
print(f"Plot saved to {dst}")
