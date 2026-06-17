#!/usr/bin/env python
"""Single-tenant scaling: skeg vs qdrant at 100k / 200k / 500k (mxbai-1024).

One index per engine, growing corpus. Captures the FULL metric set per cell:
  build_s     ingest + index time (skeg VMSET+consolidate; qdrant upsert+green)
  RSS_load    PEAK process RSS during build (ps -o rss)
  RSS_serve   steady process RSS after build
  recall@10   vs brute-force top-10 (held-out queries)
  recall@100  vs brute-force top-100
  p50/p95     query latency ms (skeg=RESP3, qdrant=HTTP - transports differ)

Configs:
  skeg            disk vindex (f32 on disk, int8 tier + graph in RAM).
  qdrant-f32      HNSW default, vectors f32 in RAM.
  qdrant-int8     HNSW + scalar int8 quantization (same tier as skeg, fair RAM cmp).

Env: SKEG_RESP3_BIN, SKEG_CORPUS(>=500k npy), SKEG_QUERIES. SCALES="100000,200000,500000" NQ=200.
One engine at a time (clean RSS).
"""
import os, time, socket, subprocess, tempfile, shutil
import numpy as np
from _common import free_port, wait_tcp, rss, load_npy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEG = os.environ["SKEG_RESP3_BIN"]
QDRANT = os.path.join(ROOT, "vendor", "qdrant")
DIM = int(os.environ.get("DIM", "1024"))
SCALES = [int(x) for x in os.environ.get("SCALES", "100000,200000,500000").split(",")]
NQ = int(os.environ.get("NQ", "200"))


def gt(corpus, queries, kmax=100):
    cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    return np.argsort(-(qn @ cn.T), axis=1)[:, :kmax]


def recalls(got, truth):
    r10 = np.mean([len(set(got[qi][:10]) & set(int(x) for x in truth[qi][:10])) / 10 for qi in range(len(got))])
    r100 = np.mean([len(set(got[qi][:100]) & set(int(x) for x in truth[qi])) / 100 for qi in range(len(got))])
    return float(r10), float(r100)


def pcts(lat):
    a = np.array(lat) * 1000
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def run_skeg(corpus, queries, truth, N, tier="int8"):
    import redis
    port = free_port(); data = tempfile.mkdtemp(prefix="st-skeg-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    try:
        assert wait_tcp(port); r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=900)
        r.execute_command("SKEG.VINDEX.CREATE", "idx", str(DIM), tier, "disk")
        t0 = time.time()
        for s in range(0, N, 256):
            a = ["SKEG.VMSET", "idx"]
            for i in range(s, min(s + 256, N)): a += [str(i), corpus[i].tobytes(), ""]
            r.execute_command(*a)
            if (s // 256) % 30 == 0: peak = max(peak, rss(p.pid))
        r.execute_command("SKEG.VINDEX.CONSOLIDATE", "idx"); peak = max(peak, rss(p.pid))
        build = time.time() - t0
        serve = max(rss(p.pid) for _ in range(5))
        for qi in range(min(20, NQ)): r.execute_command("SKEG.VSEARCH", "idx", "100", "200", queries[qi].tobytes())
        got, lat = [], []
        for qi in range(NQ):
            t = time.time()
            res = r.execute_command("SKEG.VSEARCH", "idx", "100", "200", queries[qi].tobytes())
            lat.append(time.time() - t); got.append([int(res[j]) for j in range(0, len(res), 2)])
        r10, r100 = recalls(got, truth); p50, p95 = pcts(lat)
        return dict(build=build, load=peak, serve=serve, r10=r10, r100=r100, p50=p50, p95=p95)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


def run_qdrant(corpus, queries, truth, N, quant):
    from qdrant_client import QdrantClient, models
    http, grpc = free_port(), free_port(); storage = tempfile.mkdtemp(prefix="st-qd-")
    env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
           "QDRANT__STORAGE__STORAGE_PATH": storage, "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([QDRANT], env=env, cwd=storage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    try:
        assert wait_tcp(http); time.sleep(1.0); cl = QdrantClient(host="127.0.0.1", port=http)
        qcfg = None
        if quant:
            qcfg = models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, always_ram=True))
        cl.create_collection("c", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
                             quantization_config=qcfg)
        t0 = time.time()
        for s in range(0, N, 256):
            cl.upsert("c", points=[models.PointStruct(id=s + j, vector=corpus[s + j].tolist())
                                   for j in range(min(256, N - s))])
            if (s // 256) % 30 == 0: peak = max(peak, rss(p.pid))
        while not str(cl.get_collection("c").status).lower().endswith("green"):
            time.sleep(0.3); peak = max(peak, rss(p.pid))
        build = time.time() - t0
        serve = max(rss(p.pid) for _ in range(5))
        for qi in range(min(20, NQ)): cl.query_points("c", query=queries[qi].tolist(), limit=100)
        got, lat = [], []
        for qi in range(NQ):
            t = time.time()
            pts = cl.query_points("c", query=queries[qi].tolist(), limit=100).points
            lat.append(time.time() - t); got.append([pt.id for pt in pts])
        r10, r100 = recalls(got, truth); p50, p95 = pcts(lat)
        return dict(build=build, load=peak, serve=serve, r10=r10, r100=r100, p50=p50, p95=p95)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(storage, ignore_errors=True)


def main():
    full = load_npy(os.environ["SKEG_CORPUS"], max(SCALES))
    queries = load_npy(os.environ["SKEG_QUERIES"], NQ)
    print(f"Single-tenant scaling (mxbai-1024, {NQ} queries). RSS via ps -o rss.\n")
    # skeg in every disk tier (int8 default + the TurboQuant ladder) + qdrant.
    skeg_tiers = [t for t in os.environ.get("SKEG_TIERS", "int8,tq4,tq2,tq1").split(",") if t]
    for N in SCALES:
        corpus = full[:N]
        truth = gt(corpus, queries)
        print(f"=== N={N} ===  {'config':<13} {'build_s':>7} {'RSS_load':>8} {'RSS_serve':>9} {'r@10':>6} {'r@100':>6} {'p50ms':>6} {'p95ms':>6}", flush=True)
        def emit(name, m):
            print(f"  {name:<13} {m['build']:>7.1f} {m['load']:>8.0f} {m['serve']:>9.0f} "
                  f"{m['r10']:>6.4f} {m['r100']:>6.4f} {m['p50']:>6.2f} {m['p95']:>6.2f}", flush=True)
        for t in skeg_tiers:
            emit(f"skeg-{t}", run_skeg(corpus, queries, truth, N, tier=t))
        emit("qdrant-f32", run_qdrant(corpus, queries, truth, N, quant=False))
        emit("qdrant-int8", run_qdrant(corpus, queries, truth, N, quant=True))


if __name__ == "__main__":
    main()
