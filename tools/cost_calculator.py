#!/usr/bin/env python
"""What the RAM gap costs per month.

skeg keeps the full f32 vectors on disk and only a quantized tier + graph
resident. Qdrant (HNSW, default) keeps the f32 vectors in RAM. That difference
is the whole bill. This turns it into dollars.

Per-vector resident RAM is computed from the quantization math (dim-aware), not
a single benchmark point, so you can sanity-check it:

  skeg-<tier> = tier_bytes(dim) + graph_bytes        (f32 lives on disk)
  qdrant-f32  = dim*4 (f32 in RAM) + hnsw_bytes

  tier_bytes: int8=dim, tq4=dim/2, tq2=dim/4, tq1=dim/8
  graph/hnsw: ~a few hundred bytes/vector of neighbour links

Usage:
  python tools/cost_calculator.py --vectors 50_000_000 --dim 1024 --tier tq2 --price 4
  (price = USD per GB-month of RAM; ~3-5 on the big clouds)
"""
import argparse

GRAPH_BYTES = 160      # skeg Vamana neighbour links + ids, per vector (R~32)
HNSW_BYTES = 220       # qdrant HNSW links, per vector (m~16)
TIER_DIVISOR = {"int8": 1, "tq4": 2, "tq2": 4, "tq1": 8}  # bytes = dim/divisor


def skeg_bytes(dim, tier):
    return dim / TIER_DIVISOR[tier] + GRAPH_BYTES


def qdrant_f32_bytes(dim):
    return dim * 4 + HNSW_BYTES


def gib(b):
    return b / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=lambda s: int(s.replace("_", "")), default=10_000_000)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--tier", choices=list(TIER_DIVISOR), default="tq2")
    ap.add_argument("--price", type=float, default=4.0, help="USD per GB-month of RAM")
    ap.add_argument("--tenants", type=int, default=1, help="informational; RAM is bounded by total here")
    a = ap.parse_args()

    sk = skeg_bytes(a.dim, a.tier) * a.vectors
    qd = qdrant_f32_bytes(a.dim) * a.vectors
    sk_gib, qd_gib = gib(sk), gib(qd)
    sk_cost, qd_cost = sk_gib * a.price, qd_gib * a.price

    print(f"# {a.vectors:,} vectors x {a.dim}-dim, skeg tier {a.tier}, RAM @ ${a.price}/GB-month")
    if a.tenants > 1:
        print(f"# ({a.tenants} tenants — with skeg's per-tenant isolation, build RAM is bounded by the largest tenant)")
    print()
    print(f"| engine | resident RAM | $/month | $/year |")
    print(f"|--------|-------------:|--------:|-------:|")
    print(f"| skeg-{a.tier} | {sk_gib:,.1f} GiB | ${sk_cost:,.0f} | ${sk_cost*12:,.0f} |")
    print(f"| qdrant-f32 | {qd_gib:,.1f} GiB | ${qd_cost:,.0f} | ${qd_cost*12:,.0f} |")
    print()
    print(f"-> {qd/sk:.1f}x less RAM with skeg = ${qd_cost-sk_cost:,.0f}/month saved "
          f"(${(qd_cost-sk_cost)*12:,.0f}/year), {(1-sk/qd)*100:.0f}% lower")


if __name__ == "__main__":
    main()
