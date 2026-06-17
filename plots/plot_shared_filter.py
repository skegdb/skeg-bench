#!/usr/bin/env python
"""Filtered apples-to-apples: skeg shared+filter vs qdrant shared+filter.

Both engines: ONE index holding all tenants, each query scoped by a per-tenant
filter. The honest head-to-head on the SAME multi-tenant strategy. Every config
returned leaks=0 (no cross-tenant bleed). 100k vectors/tenant, mxbai-1024.

  metric vs #tenants -> LINES (peak RAM, serve RSS, recall)
  magnitude @5 tenants -> BAR (peak RAM)
green = skeg tiers, red = qdrant. Data = measured. recall vs brute-force GT.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T = [3, 5]
# config -> per-tenant-count (peak MB, serve MB, recall, p50ms)  [all leaks=0]
D = {
    "skeg-int8": [(588, 132, 1.000, 5.31), (979, 467, 0.998, 2.22)],
    "skeg-tq4":  [(816, 195, 1.000, 2.33), (1544, 246, 0.998, 4.26)],
    "skeg-tq2":  [(707, 145, 1.000, 2.13), (1471, 222, 0.999, 2.33)],
    "skeg-tq1":  [(703, 229, 1.000, 6.04), (1123, 225, 0.999, 7.00)],
    "qdrant":    [(3388, 1667, 0.967, 4.50), (4131, 2213, 0.941, 5.67)],
}
LINE = {
    "skeg-int8": ("#0a7", "-", "o"), "skeg-tq4": ("#0a7", "--", "s"),
    "skeg-tq2": ("#13b18a", "-", "^"), "skeg-tq1": ("#6cba3a", "--", "v"),
    "qdrant": ("#c33", "-", "D"),
}
I = dict(peak=0, serve=1, recall=2, p50=3)


def lines(a, key, title, ylab, logy, ylim=None):
    for cfg in D:
        col, ls, mk = LINE[cfg]
        a.plot(T, [D[cfg][j][I[key]] for j in range(2)], ls, color=col, marker=mk, lw=2, ms=8, label=cfg)
    if logy:
        a.set_yscale("log")
    if ylim:
        a.set_ylim(*ylim)
    a.set_title(title, fontsize=11, fontweight="bold")
    a.set_xlabel("number of tenants (100k each)")
    a.set_ylabel(ylab)
    a.set_xticks(T); a.set_xticklabels([f"{t} tenants\n({t*100}k total)" for t in T])
    a.grid(True, alpha=0.3, which="both")
    a.legend(fontsize=8)


fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Filtered apples-to-apples: skeg shared+filter vs qdrant shared+filter  (100k/tenant, mxbai-1024, leaks=0 all)",
             fontsize=12.5, fontweight="bold")

lines(ax[0][0], "peak", "Peak RAM vs #tenants — lower is better", "MB (log)", True)
lines(ax[0][1], "serve", "Steady serve RSS vs #tenants — lower is better", "MB (log)", True)
lines(ax[1][0], "recall", "recall@10 vs #tenants — qdrant decays to 0.94 (higher is better)", "recall@10", False, (0.93, 1.005))

a = ax[1][1]
cfgs = list(D.keys())
x = np.arange(len(cfgs)); w = 0.4
peak = [D[c][1][I["peak"]] for c in cfgs]
serve = [D[c][1][I["serve"]] for c in cfgs]
b1 = a.bar(x - w/2, peak, w, color="#4878b0", edgecolor="k", lw=0.4, label="peak (build)")
b2 = a.bar(x + w/2, serve, w, color="#f0a04b", edgecolor="k", lw=0.4, label="serve (steady RSS)")
for bars in (b1, b2):
    for b in bars:
        a.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.0f}",
               ha="center", va="bottom", fontsize=8, rotation=90)
a.set_title("RAM @5 tenants — peak vs serve RSS (lower is better)", fontsize=11, fontweight="bold")
a.set_ylabel("MB"); a.set_ylim(top=max(peak)*1.18)
a.set_xticks(x); a.set_xticklabels(cfgs, rotation=20, ha="right", fontsize=9)
a.grid(True, alpha=0.3, axis="y"); a.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("charts", exist_ok=True)
out = "charts/chart_shared_filter.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")

