"""hnswlib worker — the raw in-RAM HNSW algorithm (the kernel Qdrant/Chroma wrap),
cosine. A reference point: what HNSW costs with no server around it. Subprocess.

Args: <corpus.npy> <queries.npy> <workdir> [ef]. JSON out:
{rss_mib, baseline_mib, build_s, p50_us, p95_us, got}.
"""
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np


def rss_mib():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                              capture_output=True, text=True).stdout.strip()) / 1024.0


baseline = rss_mib()
import hnswlib  # noqa: E402

cp, qp, work = sys.argv[1], sys.argv[2], sys.argv[3]
ef = int(sys.argv[4]) if len(sys.argv) > 4 else 100
corpus = np.load(cp)
queries = np.load(qp)
n, dim = corpus.shape

idx = hnswlib.Index(space="cosine", dim=dim)
idx.init_index(max_elements=n, ef_construction=200, M=16)
t0 = time.time()
idx.add_items(corpus, np.arange(n))
build_s = time.time() - t0
idx.set_ef(ef)

q = queries.copy()
del corpus
gc.collect()

got, lat = [], []
for v in q:
    s = time.time()
    labels, _ = idx.knn_query(v, k=100)
    lat.append((time.time() - s) * 1e6)
    got.append([int(x) for x in labels[0]])

a = sorted(lat)
pick = lambda p: a[min(len(a) - 1, int(len(a) * p))]
print(json.dumps({"rss_mib": round(rss_mib(), 1), "baseline_mib": round(baseline, 1),
                  "build_s": round(build_s, 2), "p50_us": round(pick(0.50), 1),
                  "p95_us": round(pick(0.95), 1), "got": got}))
