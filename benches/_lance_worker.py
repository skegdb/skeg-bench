"""LanceDB worker — runs in its own process so RSS is isolated and comparable.

Args: <corpus.npy> <queries.npy> <workdir> <nprobes>. Builds an IVF-PQ index
(disk-first, like skeg's quantized tier), queries top-100, self-measures RSS
AFTER freeing the in-memory corpus (so the number is lance's serving footprint,
not the numpy copy), and prints one JSON line:
{rss_mib, baseline_mib, build_s, disk_mib, p50_us, p95_us, got}.
The parent computes recall from `got`.
"""
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _common import rss


baseline = rss()  # python + numpy, before any data
import lancedb  # noqa: E402

cp, qp, work, nprobes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
refine = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # f32 re-rank factor (0 = off)
corpus = np.load(cp)
queries = np.load(qp)
n, dim = corpus.shape

db = lancedb.connect(f"{work}/db")
t0 = time.time()
tbl = db.create_table("cmp", [{"id": i, "vector": corpus[i].tolist()} for i in range(n)])
num_part = max(1, int(math.sqrt(n)))     # IVF: ~sqrt(n) partitions
# PQ sub-vectors must DIVIDE the dimension: pick the largest divisor <= dim/8.
num_sub = max(d for d in range(1, max(1, dim // 8) + 1) if dim % d == 0)
tbl.create_index(metric="cosine", num_partitions=num_part, num_sub_vectors=num_sub, index_type="IVF_PQ")
build_s = time.time() - t0
disk_mib = sum(f.stat().st_size for f in Path(f"{work}/db").rglob("*") if f.is_file()) / (1024 * 1024)

probes = nprobes if nprobes else 40
qlist = [v.tolist() for v in queries]
del corpus  # free the numpy copy so RSS reflects lance's serving footprint
gc.collect()

def query(v):
    q = tbl.search(v).metric("cosine").limit(100).nprobes(probes)
    if refine:
        q = q.refine_factor(refine)
    return q.to_list()


for v in qlist[:32]:
    query(v)

got, lat = [], []
for v in qlist:
    s = time.time()
    rows = query(v)
    lat.append((time.time() - s) * 1e6)
    got.append([int(r["id"]) for r in rows])

a = sorted(lat)
pick = lambda p: a[min(len(a) - 1, int(len(a) * p))]
print(json.dumps({
    "rss_mib": round(rss(), 1),
    "baseline_mib": round(baseline, 1),
    "build_s": round(build_s, 2),
    "disk_mib": round(disk_mib, 1),
    "p50_us": round(pick(0.50), 1),
    "p95_us": round(pick(0.95), 1),
    "got": got,
}))
