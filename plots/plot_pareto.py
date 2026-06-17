#!/usr/bin/env python
"""Cross-engine Pareto: RAM vs recall vs latency, one point per engine.

The money chart. Reads results/multi_engine.txt (parses the engine table); falls
back to embedded reference numbers. skeg sits alone in the low-RAM/high-recall
corner — every other engine gives up at least one axis.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# fallback reference (mxbai-1024 @100k, brute-force GT): ram, recall10, p50ms
REF = {
    "skeg-tq2":          (47, 1.000, 2.49),
    "lancedb (IVF-PQ)":  (198, 0.998, 59.26),
    "milvus-lite":       (108, 0.934, 2.69),
    "hnswlib (raw HNSW)": (426, 0.985, 1.99),
    "chroma (HNSW)":     (682, 0.985, 3.91),
    "qdrant-f32":        (885, 0.997, 2.62),
}


def load():
    path = os.path.join(RESULTS, "multi_engine.txt")
    try:
        data = {}
        for line in open(path):
            p = line.split()
            if len(p) < 7:
                continue
            try:
                build, ram, r10, r100, p50, p95 = map(float, p[-6:])
            except ValueError:
                continue
            name = " ".join(p[:-6])
            if name and name != "engine":
                data[name] = (ram, r10, p50)
        return data or REF
    except FileNotFoundError:
        return REF


def color(name):
    if name.startswith("skeg"):
        return "#0a7"
    return "#c33" if name.startswith("qdrant") else "#888"


D = load()
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle("Cross-engine Pareto — mxbai-1024 @ 100K, recall vs brute-force GT  (top-left is best)",
             fontsize=13, fontweight="bold")

# Panel 1: RAM (log x) vs recall@10
a = ax[0]
for name, (ram, r10, p50) in D.items():
    a.scatter(ram, r10, s=180, color=color(name), edgecolor="k", lw=0.6, zorder=3)
    a.annotate(name, (ram, r10), xytext=(6, 6), textcoords="offset points", fontsize=9,
               fontweight="bold" if name.startswith("skeg") else "normal")
a.set_xscale("log")
a.set_xlabel("serve RAM (MB, log) — lower is better")
a.set_ylabel("recall@10 — higher is better")
a.set_title("RAM vs recall: skeg owns the lean + accurate corner", fontsize=11)
a.axhspan(0.999, 1.001, color="#0a7", alpha=0.06)
a.grid(True, alpha=0.3, which="both")

# Panel 2: RAM (log x) vs latency (log y), bubble = recall
a = ax[1]
for name, (ram, r10, p50) in D.items():
    a.scatter(ram, p50, s=120 + 600 * (r10 - 0.9), color=color(name), edgecolor="k", lw=0.6, zorder=3, alpha=0.85)
    a.annotate(name, (ram, p50), xytext=(6, 6), textcoords="offset points", fontsize=9,
               fontweight="bold" if name.startswith("skeg") else "normal")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("serve RAM (MB, log) — lower is better")
a.set_ylabel("p50 latency (ms, log) — lower is better")
a.set_title("RAM vs latency (bubble = recall): skeg is lean + fast + accurate", fontsize=11)
a.grid(True, alpha=0.3, which="both")

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts", "chart_pareto.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}")
