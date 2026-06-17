"""Chroma worker — embedded (HNSW via hnswlib), cosine. Subprocess for RSS.

Args: <corpus.npy> <queries.npy> <workdir>. JSON out:
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
import chromadb  # noqa: E402

cp, qp, work = sys.argv[1], sys.argv[2], sys.argv[3]
corpus = np.load(cp)
queries = np.load(qp)
n, dim = corpus.shape

client = chromadb.PersistentClient(path=f"{work}/chroma")
col = client.create_collection("cmp", metadata={"hnsw:space": "cosine"})
t0 = time.time()
B = 5000  # chroma caps add batch size
for s in range(0, n, B):
    e = min(s + B, n)
    col.add(ids=[str(i) for i in range(s, e)], embeddings=corpus[s:e].tolist())
build_s = time.time() - t0

qlist = [q.tolist() for q in queries]
del corpus
gc.collect()

got, lat = [], []
for q in qlist:
    s = time.time()
    res = col.query(query_embeddings=[q], n_results=100)
    lat.append((time.time() - s) * 1e6)
    got.append([int(x) for x in res["ids"][0]])

a = sorted(lat)
pick = lambda p: a[min(len(a) - 1, int(len(a) * p))]
print(json.dumps({"rss_mib": round(rss_mib(), 1), "baseline_mib": round(baseline, 1),
                  "build_s": round(build_s, 2), "p50_us": round(pick(0.50), 1),
                  "p95_us": round(pick(0.95), 1), "got": got}))
