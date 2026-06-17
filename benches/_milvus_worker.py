"""Milvus Lite worker — embedded, cosine. Subprocess for RSS isolation.

Args: <corpus.npy> <queries.npy> <workdir>. JSON out:
{rss_mib, baseline_mib, build_s, p50_us, p95_us, got}.
"""
import gc
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def rss_mib():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                              capture_output=True, text=True).stdout.strip()) / 1024.0


baseline = rss_mib()
from pymilvus import MilvusClient  # noqa: E402

cp, qp, work = sys.argv[1], sys.argv[2], sys.argv[3]
corpus = np.load(cp)
queries = np.load(qp)
n, dim = corpus.shape

client = MilvusClient(f"{work}/milvus.db")
client.create_collection("cmp", dimension=dim, metric_type="COSINE", auto_id=False)
t0 = time.time()
B = 2000
for s in range(0, n, B):
    e = min(s + B, n)
    client.insert("cmp", data=[{"id": int(i), "vector": corpus[i].tolist()} for i in range(s, e)])
build_s = time.time() - t0

qlist = [q.tolist() for q in queries]
del corpus
gc.collect()

got, lat = [], []
for q in qlist:
    s = time.time()
    res = client.search("cmp", data=[q], limit=100, output_fields=["id"])
    lat.append((time.time() - s) * 1e6)
    got.append([int(h["id"]) for h in res[0]])

a = sorted(lat)
pick = lambda p: a[min(len(a) - 1, int(len(a) * p))]
print(json.dumps({"rss_mib": round(rss_mib(), 1), "baseline_mib": round(baseline, 1),
                  "build_s": round(build_s, 2), "p50_us": round(pick(0.50), 1),
                  "p95_us": round(pick(0.95), 1), "got": got}))
