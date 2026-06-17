#!/usr/bin/env python
"""Phase B multi-tenant density chart: skeg per-tenant isolation vs qdrant.

100k vectors/tenant (mxbai-1024). x = number of tenants. The wedge story:
skeg's RAM stays ~flat as tenants grow (each vindex builds/serves independently,
bounded by the largest tenant), while qdrant climbs with the total.

  metric vs #tenants (scaling)  -> LINES (peak RAM, serve RSS, recall)
  magnitude at the stress point -> BAR  (peak RAM @5 tenants)

green family = skeg tiers (physical isolation, leak-free by construction),
orange = qdrant per-collection, red = qdrant shared+filter.
Data = measured Phase B. RAM via `ps -o rss`. recall vs brute-force GT.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T = [3, 5]  # tenants
# config -> per-tenant-count (peak MB, serve MB, recall, p50ms, p95ms)
D = {
    "skeg-int8":   [(273, 50, 1.000, 3.96, 12.40), (249, 54, 1.000, 4.31, 25.80)],
    "skeg-tq4":    [(262, 74, 1.000, 4.53, 19.79), (285, 189, 1.000, 4.61, 21.96)],
    "skeg-tq2":    [(201, 65, 1.000, 4.34, 20.64), (231, 33, 1.000, 3.74, 18.51)],
    "skeg-tq1":    [(168, 27, 1.000, 7.76, 18.75), (200, 162, 1.000, 6.53, 27.69)],
    "qdrant-coll":   [(1299, 1054, 0.990, 7.10, 71.70), (1818, 1155, 0.996, 10.22, 87.03)],
    "qdrant-shared": [(2887, 979, 0.962, 7.48, 49.90), (3494, 2538, 0.957, 8.05, 50.98)],
}
LINE = {
    "skeg-int8":   ("#0a7", "-",  "o"),
    "skeg-tq4":    ("#0a7", "--", "s"),
    "skeg-tq2":    ("#13b18a", "-",  "^"),
    "skeg-tq1":    ("#6cba3a", "--", "v"),
    "qdrant-coll":   ("#e8902a", "-", "o"),
    "qdrant-shared": ("#c33", "-", "D"),
}
I = dict(peak=0, serve=1, recall=2, p50=3, p95=4)


def lines(a, key, title, ylab, logy, ylim=None):
    for cfg in D:
        col, ls, mk = LINE[cfg]
        a.plot(T, [D[cfg][j][I[key]] for j in range(2)], ls, color=col, marker=mk,
               lw=2, ms=8, label=cfg)
    if logy:
        a.set_yscale("log")
    if ylim:
        a.set_ylim(*ylim)
    a.set_title(title, fontsize=11, fontweight="bold")
    a.set_xlabel("number of tenants (100k vectors each)")
    a.set_ylabel(ylab)
    a.set_xticks(T); a.set_xticklabels([f"{t} tenants\n({t*100}k total)" for t in T])
    a.grid(True, alpha=0.3, which="both")
    a.legend(fontsize=8, ncol=2)


fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase B multi-tenant density: skeg per-tenant isolation vs qdrant  (100k vectors/tenant, mxbai-1024)",
             fontsize=13, fontweight="bold")

lines(ax[0][0], "peak", "Peak RAM vs #tenants — skeg flat, qdrant climbs (lower is better)", "MB (log)", True)
ax[0][0].annotate("skeg ~flat\n(bounded by largest tenant)", xy=(5, 230), xytext=(3.25, 470),
                  fontsize=9, color="#085", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#085"))
lines(ax[0][1], "serve", "Steady serve RSS vs #tenants (lower is better)", "MB (log)", True)
lines(ax[1][0], "recall", "recall@10 vs #tenants — skeg 1.0, qdrant lower (higher is better)", "recall@10", False, (0.94, 1.005))

# magnitude bars @5 tenants: peak AND serve RSS, side by side per config
a = ax[1][1]
cfgs = list(D.keys())
x = np.arange(len(cfgs)); w = 0.4
peak = [D[c][1][I["peak"]] for c in cfgs]   # 5 tenants
serve = [D[c][1][I["serve"]] for c in cfgs]
b1 = a.bar(x - w/2, peak, w, color="#4878b0", edgecolor="k", lw=0.4, label="peak (build)")
b2 = a.bar(x + w/2, serve, w, color="#f0a04b", edgecolor="k", lw=0.4, label="serve (steady RSS)")
for bars in (b1, b2):
    for b in bars:
        a.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.0f}",
               ha="center", va="bottom", fontsize=7.5, rotation=90)
a.set_title("RAM @5 tenants — peak vs serve RSS (lower is better)", fontsize=11, fontweight="bold")
a.set_ylabel("MB"); a.set_ylim(top=max(peak)*1.18)
a.set_xticks(x); a.set_xticklabels(cfgs, rotation=25, ha="right", fontsize=9)
a.grid(True, alpha=0.3, axis="y"); a.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("charts", exist_ok=True)
out = "charts/chart_multitenant.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
