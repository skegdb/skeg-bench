#!/usr/bin/env python
"""Phase A single-tenant chart: skeg tiers vs qdrant (mxbai-1024).

Everything here is a metric-vs-N scaling relationship, so everything is a LINE.
  RAM: BOTH peak (during build) and steady serve RSS, each across all N.
  recall: BOTH depths (@10 and @100), across all N.
green family = skeg tiers, red family = qdrant. The two never overlap on log-y.

Data = measured Phase A (200 real queries, brute-force GT). RAM via `ps -o rss`,
identical for both engines.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = [100_000, 200_000, 500_000]
NLAB = ["100k", "200k", "500k"]
# config -> per-N (build_s, peak MB, serve MB, recall@10, recall@100, p50ms, p95ms)
D = {
    "skeg-int8":   [(59.9, 274, 274, 1.0000, 1.0000, 2.59, 2.85), (179.0, 387, 137, 1.0000, 1.0000, 2.62, 2.91), (591.9, 794, 198, 0.9995, 0.9997, 4.98, 17.17)],
    "skeg-tq4":    [(67.0, 213, 72, 1.0000, 1.0000, 2.63, 3.01), (180.2, 347, 131, 1.0000, 0.9999, 2.72, 3.17), (586.7, 998, 109, 0.9995, 0.9998, 3.49, 8.35)],
    "skeg-tq2":    [(69.5, 203, 48, 1.0000, 1.0000, 3.05, 6.61), (182.9, 333, 76, 1.0000, 1.0000, 2.66, 3.13), (600.3, 901, 76, 1.0000, 0.9997, 4.70, 12.42)],
    "skeg-tq1":    [(70.1, 141, 75, 1.0000, 0.9999, 5.49, 8.69), (184.9, 347, 49, 1.0000, 0.9997, 6.16, 8.82), (738.3, 1021, 96, 1.0000, 0.9993, 6.55, 11.51)],
    "qdrant-f32":  [(63.0, 1006, 768, 0.9965, 0.9786, 2.64, 3.12), (127.4, 1801, 1010, 0.9930, 0.9782, 3.22, 6.81), (315.5, 5436, 2331, 0.9875, 0.9559, 3.02, 4.99)],
    "qdrant-int8": [(60.1, 1095, 841, 0.9595, 0.9653, 2.36, 2.61), (118.5, 1601, 1311, 0.9560, 0.9635, 2.38, 2.92), (323.7, 5051, 2563, 0.9455, 0.9543, 2.36, 4.45)],
}
LINE = {
    "skeg-int8":   ("#0a7", "-",  "o"),
    "skeg-tq4":    ("#0a7", "--", "s"),
    "skeg-tq2":    ("#13b18a", "-",  "^"),
    "skeg-tq1":    ("#6cba3a", "--", "v"),
    "qdrant-f32":  ("#c33", "-",  "o"),
    "qdrant-int8": ("#e8902a", "--", "s"),
}
I = dict(build=0, peak=1, serve=2, r10=3, r100=4, p50=5, p95=6)

# Prefer a fresh run's results/; fall back to the embedded reference numbers.
try:
    from _parse import parse_singletenant
    _Ns, _d = parse_singletenant()
    if _d:
        _ord = ["build", "peak", "serve", "r10", "r100", "p50", "p95"]
        N = _Ns
        NLAB = [f"{n // 1000}k" if n >= 1000 else str(n) for n in _Ns]
        D = {c: [tuple(_d[c][n][k] for k in _ord) for n in _Ns] for c in _d}
except (FileNotFoundError, KeyError):
    pass


def lines(a, key, title, ylab, logy, ylim=None):
    for cfg in D:
        col, ls, mk = LINE.get(cfg, ("#888", "-", "o"))
        a.plot(N, [D[cfg][j][I[key]] for j in range(len(N))], ls, color=col, marker=mk,
               lw=2, ms=7, label=cfg)
    if logy:
        a.set_yscale("log")
    if ylim:
        a.set_ylim(*ylim)
    a.set_title(title, fontsize=11, fontweight="bold")
    a.set_xlabel("corpus size (vectors)")
    a.set_ylabel(ylab)
    a.set_xticks(N); a.set_xticklabels(NLAB)
    a.grid(True, alpha=0.3, which="both")
    a.legend(fontsize=8, ncol=2)


fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase A single-tenant: skeg tiers vs qdrant  (mxbai-1024, 200 real queries, brute-force GT)",
             fontsize=14, fontweight="bold")

lines(ax[0][0], "peak",  "Peak RAM during build vs N — lower is better", "MB (log)", True)
lines(ax[0][1], "serve", "Steady serve RSS vs N — lower is better", "MB (log)", True)
lines(ax[1][0], "r10",   "recall@10 vs N — higher is better", "recall@10", False, (0.93, 1.005))
lines(ax[1][1], "r100",  "recall@100 vs N — higher is better", "recall@100", False, (0.93, 1.005))

# RAM gap callout at the largest N, computed from whatever data is loaded.
if len(N) >= 2 and "qdrant-f32" in D and "skeg-tq2" in D:
    qd = D["qdrant-f32"][-1][I["peak"]]
    sk = D["skeg-tq2"][-1][I["peak"]]
    ax[0][0].annotate(f"skeg {sk:.0f}  vs  qdrant {qd:.0f}  -> {qd / sk:.1f}x",
                      xy=(N[-1], qd), xytext=(N[len(N) // 3], qd * 0.65),
                      fontsize=9, color="#c33", fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="#c33"))

fig.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("charts", exist_ok=True)
out = "charts/chart_singletenant.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")


# --- companion figure: RAM as BARS across phases (N), peak and serve ---
import numpy as np
CFGS = list(D.keys())
NSKEG = sum(c.startswith("skeg") for c in CFGS)
NCOL = ["#7bb8e0", "#f0a04b", "#5cb874"]  # 100k / 200k / 500k -> the legend


def ram_pair(a, ni, title):
    """x = config; peak and serve RSS as ADJACENT columns per config."""
    x = np.arange(len(CFGS)); w = 0.4
    a.axvspan(-0.5, NSKEG - 0.5, color="#0a7", alpha=0.06)
    a.axvspan(NSKEG - 0.5, len(CFGS) - 0.5, color="#c33", alpha=0.06)
    peak = [D[c][ni][I["peak"]] for c in CFGS]
    serve = [D[c][ni][I["serve"]] for c in CFGS]
    b1 = a.bar(x - w/2, peak, w, color="#4878b0", edgecolor="k", linewidth=0.3, label="peak (build)")
    b2 = a.bar(x + w/2, serve, w, color="#f0a04b", edgecolor="k", linewidth=0.3, label="serve (steady RSS)")
    for bars in (b1, b2):
        for b in bars:
            a.text(b.get_x() + b.get_width()/2, b.get_height() + 40, f"{b.get_height():.0f}",
                   ha="center", va="bottom", fontsize=7, rotation=90)
    a.set_title(title, fontsize=12, fontweight="bold")
    a.set_ylabel("MB")
    a.set_xticks(x); a.set_xticklabels(CFGS, rotation=25, ha="right", fontsize=9)
    a.legend(fontsize=10); a.grid(True, alpha=0.3, axis="y")
    top = a.get_ylim()[1]
    a.text((NSKEG - 1) / 2, top * 0.97, "SKEG", ha="center", va="top",
           fontsize=11, color="#085", fontweight="bold", alpha=0.55)
    a.text((NSKEG + len(CFGS) - 1) / 2, top * 0.97, "QDRANT", ha="center", va="top",
           fontsize=11, color="#a22", fontweight="bold", alpha=0.55)


fig2, ax2 = plt.subplots(1, 1, figsize=(13, 7))
fig2.suptitle(f"Phase A RAM @{NLAB[-1]} — peak vs serve RSS side by side, linear  (mxbai-1024)",
              fontsize=14, fontweight="bold")
ram_pair(ax2, len(N) - 1, f"Peak vs serve RSS @{NLAB[-1]} (MB) per config — lower is better")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
out2 = "charts/chart_singletenant_ram.png"
fig2.savefig(out2, dpi=130)
print(f"wrote {out2}")
