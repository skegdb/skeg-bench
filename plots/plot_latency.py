#!/usr/bin/env python
"""Latency across both benchmark phases — skeg tiers vs qdrant (mxbai-1024).

Top row = Phase A (single-tenant) p50 and p95 vs corpus size N.
Bottom row = Phase B (multi-tenant isolation) p50 and p95 vs #tenants.
Latency vs the scaling axis is a relationship -> LINES.
Caveat: skeg=RESP3, qdrant=HTTP; transports differ, read trends not absolutes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Phase A: config -> per-N (p50, p95) at N=100k/200k/500k
NA = [100_000, 200_000, 500_000]
A = {
    "skeg-int8":   [(2.59, 2.85), (2.62, 2.91), (4.98, 17.17)],
    "skeg-tq4":    [(2.63, 3.01), (2.72, 3.17), (3.49, 8.35)],
    "skeg-tq2":    [(3.05, 6.61), (2.66, 3.13), (4.70, 12.42)],
    "skeg-tq1":    [(5.49, 8.69), (6.16, 8.82), (6.55, 11.51)],
    "qdrant-f32":  [(2.64, 3.12), (3.22, 6.81), (3.02, 4.99)],
    "qdrant-int8": [(2.36, 2.61), (2.38, 2.92), (2.36, 4.45)],
}
# Phase B: config -> per-tenant-count (p50, p95) at 3/5 tenants
TB = [3, 5]
B = {
    "skeg-int8":     [(3.96, 12.40), (4.31, 25.80)],
    "skeg-tq4":      [(4.53, 19.79), (4.61, 21.96)],
    "skeg-tq2":      [(4.34, 20.64), (3.74, 18.51)],
    "skeg-tq1":      [(7.76, 18.75), (6.53, 27.69)],
    "qdrant-coll":   [(7.10, 71.70), (10.22, 87.03)],
    "qdrant-shared": [(7.48, 49.90), (8.05, 50.98)],
}
STY = {
    "skeg-int8": ("#0a7", "-", "o"), "skeg-tq4": ("#0a7", "--", "s"),
    "skeg-tq2": ("#13b18a", "-", "^"), "skeg-tq1": ("#6cba3a", "--", "v"),
    "qdrant-f32": ("#c33", "-", "o"), "qdrant-int8": ("#e8902a", "--", "s"),
    "qdrant-coll": ("#e8902a", "-", "o"), "qdrant-shared": ("#c33", "-", "D"),
}


# Prefer fresh runs; fall back to embedded reference. A=single-tenant, B=multi.
try:
    from _parse import parse_singletenant, parse_multitenant
    _na, _ad = parse_singletenant()
    if _ad:
        NA = _na
        A = {c: [(_ad[c][n]["p50"], _ad[c][n]["p95"]) for n in _na] for c in _ad}
    _tb, _bd = parse_multitenant()
    if _bd:
        TB = _tb
        B = {c: [(_bd[c][t]["p50"], _bd[c][t]["p95"]) for t in _tb] for c in _bd}
except (FileNotFoundError, KeyError):
    pass


def panel(a, data, xs, idx, title, xlab, xticklab):
    for cfg in data:
        col, ls, mk = STY.get(cfg, ("#888", "-", "o"))
        a.plot(xs, [data[cfg][j][idx] for j in range(len(xs))], ls, color=col,
               marker=mk, lw=2, ms=7, label=cfg)
    a.set_title(title, fontsize=11, fontweight="bold")
    a.set_xlabel(xlab); a.set_ylabel("ms")
    a.set_xticks(xs); a.set_xticklabels(xticklab)
    a.grid(True, alpha=0.3); a.legend(fontsize=8, ncol=2)


fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Query latency across both phases — skeg tiers vs qdrant  (mxbai-1024; skeg RESP3 vs qdrant HTTP)",
             fontsize=13, fontweight="bold")

_alab = [f"{n // 1000}k" if n >= 1000 else str(n) for n in NA]
_blab = [str(t) for t in TB]
panel(ax[0][0], A, NA, 0, "Phase A single-tenant — p50 vs N (lower is better)", "corpus size", _alab)
panel(ax[0][1], A, NA, 1, "Phase A single-tenant — p95 vs N (lower is better)", "corpus size", _alab)
panel(ax[1][0], B, TB, 0, "Phase B multi-tenant — p50 vs #tenants (lower is better)", "tenants", _blab)
panel(ax[1][1], B, TB, 1, "Phase B multi-tenant — p95 vs #tenants (lower is better)", "tenants", _blab)

fig.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("charts", exist_ok=True)
out = "charts/chart_latency.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
