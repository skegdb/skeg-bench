#!/usr/bin/env python
"""Cross-engine single-tenant comparison at a fixed N: skeg vs LanceDB vs Qdrant.

The honest disk-first trio: skeg (graph + TurboQuant tier) and LanceDB (IVF-PQ)
are both low-RAM disk-first engines; Qdrant-f32 is the RAM-resident baseline.
Same corpus, same brute-force ground truth, RAM via `ps -o rss`.

LanceDB is embedded, so it runs in a worker subprocess and self-measures RSS
after freeing the in-memory corpus; we report its marginal RAM (over the python
baseline) for a fair comparison with the server engines.

Env: SKEG_RESP3_BIN, SKEG_CORPUS, SKEG_QUERIES, N (default 100000), NQ (200),
SKEG_TIER (tq2), LANCE_NPROBES (40).
"""
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np
from _common import free_port, wait_tcp, rss, load_npy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEG = os.environ["SKEG_RESP3_BIN"]
QDRANT = os.path.join(ROOT, "vendor", "qdrant")
N = int(os.environ.get("N", "100000"))
NQ = int(os.environ.get("NQ", "200"))
TIER = os.environ.get("SKEG_TIER", "tq2")
# LanceDB IVF-PQ tuned for a fair fight: nprobes + f32 refine give recall@10 1.0
# (the same as skeg) so the comparison is RAM/latency at matched recall, not a
# straw-man. Lower these and lance gets faster but its recall collapses.
NPROBES = int(os.environ.get("LANCE_NPROBES", "150"))
REFINE = int(os.environ.get("LANCE_REFINE", "10"))


def gt(corpus, queries, k=100):
    cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    return np.argsort(-(qn @ cn.T), axis=1)[:, :k]


def recalls(got, truth):
    r10 = np.mean([len(set(got[i][:10]) & set(int(x) for x in truth[i][:10])) / 10 for i in range(len(got))])
    r100 = np.mean([len(set(got[i][:100]) & set(int(x) for x in truth[i])) / 100 for i in range(len(got))])
    return float(r10), float(r100)


def pcts(lat_s):
    a = np.array(lat_s) * 1000
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def run_skeg(corpus, queries, truth):
    import redis
    port = free_port(); data = tempfile.mkdtemp(prefix="me-skeg-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_tcp(port); r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=900)
        dim = corpus.shape[1]
        r.execute_command("SKEG.VINDEX.CREATE", "idx", str(dim), TIER, "disk")
        t0 = time.time()
        for s in range(0, N, 256):
            a = ["SKEG.VMSET", "idx"]
            for i in range(s, min(s + 256, N)):
                a += [str(i), corpus[i].tobytes(), ""]
            r.execute_command(*a)
        r.execute_command("SKEG.VINDEX.CONSOLIDATE", "idx")
        build = time.time() - t0
        serve = max(rss(p.pid) for _ in range(5))
        got, lat = [], []
        for qi in range(NQ):
            t = time.time()
            res = r.execute_command("SKEG.VSEARCH", "idx", "100", "200", queries[qi].tobytes())
            lat.append(time.time() - t)
            got.append([int(res[j]) for j in range(0, len(res), 2)])
        r10, r100 = recalls(got, truth); p50, p95 = pcts(lat)
        return dict(build=build, ram=serve, r10=r10, r100=r100, p50=p50, p95=p95)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


def run_qdrant(corpus, queries, truth):
    from qdrant_client import QdrantClient, models
    http, grpc = free_port(), free_port(); storage = tempfile.mkdtemp(prefix="me-qd-")
    env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
           "QDRANT__STORAGE__STORAGE_PATH": storage, "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([QDRANT], env=env, cwd=storage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_tcp(http); time.sleep(1.0); cl = QdrantClient(host="127.0.0.1", port=http)
        dim = corpus.shape[1]
        cl.create_collection("c", vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        t0 = time.time()
        for s in range(0, N, 256):
            cl.upsert("c", points=[models.PointStruct(id=s + j, vector=corpus[s + j].tolist())
                                   for j in range(min(256, N - s))])
        while not str(cl.get_collection("c").status).lower().endswith("green"):
            time.sleep(0.3)
        build = time.time() - t0
        serve = max(rss(p.pid) for _ in range(5))
        got, lat = [], []
        for qi in range(NQ):
            t = time.time()
            pts = cl.query_points("c", query=queries[qi].tolist(), limit=100).points
            lat.append(time.time() - t); got.append([pt.id for pt in pts])
        r10, r100 = recalls(got, truth); p50, p95 = pcts(lat)
        return dict(build=build, ram=serve, r10=r10, r100=r100, p50=p50, p95=p95)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(storage, ignore_errors=True)


def run_worker(corpus, queries, truth, worker, extra=()):
    """Generic runner for embedded engines (lance/chroma/milvus/hnsw): each runs
    in its own process and self-measures RSS over a python baseline."""
    work = tempfile.mkdtemp(prefix="me-")
    cf, qf = os.path.join(work, "c.npy"), os.path.join(work, "q.npy")
    np.save(cf, corpus[:N]); np.save(qf, queries[:NQ])
    try:
        out = subprocess.run(["python", os.path.join(os.path.dirname(__file__), worker),
                              cf, qf, work, *map(str, extra)], capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  {worker} failed:", out.stderr.strip()[-300:]); return None
        j = json.loads(out.stdout.strip().splitlines()[-1])
        r10, r100 = recalls(j["got"], truth)
        return dict(build=j["build_s"], ram=j["rss_mib"] - j["baseline_mib"],
                    r10=r10, r100=r100, p50=j["p50_us"] / 1000, p95=j["p95_us"] / 1000)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    corpus = load_npy(os.environ["SKEG_CORPUS"], N)
    queries = load_npy(os.environ["SKEG_QUERIES"], NQ)
    truth = gt(corpus, queries)
    print(f"Cross-engine @ N={N} ({corpus.shape[1]}-dim), {NQ} queries, brute-force GT. RAM via ps -o rss.")
    print("(LanceDB RAM = marginal over python baseline; skeg/qdrant = server serve RSS.)\n")
    # Which engines to run (env ENGINES, comma-separated; default all available).
    want = os.environ.get("ENGINES", "skeg,lancedb,chroma,milvus,hnswlib,qdrant").split(",")
    candidates = {
        "skeg": ("skeg-" + TIER, lambda: run_skeg(corpus, queries, truth)),
        "lancedb": ("lancedb (IVF-PQ)", lambda: run_worker(corpus, queries, truth, "_lance_worker.py", [NPROBES, REFINE])),
        "chroma": ("chroma (HNSW)", lambda: run_worker(corpus, queries, truth, "_chroma_worker.py")),
        "milvus": ("milvus-lite", lambda: run_worker(corpus, queries, truth, "_milvus_worker.py")),
        "hnswlib": ("hnswlib (raw HNSW)", lambda: run_worker(corpus, queries, truth, "_hnsw_worker.py")),
        "qdrant": ("qdrant-f32", lambda: run_qdrant(corpus, queries, truth)),
    }
    rows = [(candidates[e][0], candidates[e][1]()) for e in want if e in candidates]
    print(f"  {'engine':<18} {'build_s':>7} {'serve RAM':>9} {'r@10':>6} {'r@100':>6} {'p50ms':>6} {'p95ms':>6}")
    for name, m in rows:
        if not m:
            print(f"  {name:<18}  (failed)"); continue
        print(f"  {name:<18} {m['build']:>7.1f} {m['ram']:>9.0f} {m['r10']:>6.3f} {m['r100']:>6.3f} {m['p50']:>6.2f} {m['p95']:>6.2f}")


if __name__ == "__main__":
    main()
