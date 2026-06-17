#!/usr/bin/env python
"""Query latency, idle and under concurrent load: skeg vs qdrant.

A quiet "victim" tenant's query latency is measured two ways: (a) system idle,
(b) while N threads hammer a different "noisy" tenant. We report the absolute
p50/p95 in both states — the question is which engine serves faster, and whether
it stays fast when the box is busy.

  skeg   - one index per tenant + a VSEARCH worker pool (--workers)
  qdrant - one shared collection + per-tenant filter

(Note: under enough concurrency both degrade by a similar factor — it's
CPU-bound, not an architecture story. The takeaway here is absolute latency:
skeg runs lower, idle and under load.)

Self-contained (synthetic vectors). Env: SKEG_RESP3_BIN, NS (8), M (20000),
DIM (256), NOISE_THREADS (8), WORKERS (8), QUERIES (200).
"""
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np

SKEG = os.environ["SKEG_RESP3_BIN"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT = os.path.join(ROOT, "vendor", "qdrant")
NS = int(os.environ.get("NS", "8"))
M = int(os.environ.get("M", "20000"))
DIM = int(os.environ.get("DIM", "256"))
NOISE = int(os.environ.get("NOISE_THREADS", "8"))
WORKERS = int(os.environ.get("WORKERS", "8"))
NQ = int(os.environ.get("QUERIES", "200"))
VICTIM, NOISY = 0, 1


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_tcp(port, t=30):
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close(); return True
        except OSError:
            time.sleep(0.05)
    return False


def pcts(lat):
    a = np.array(lat) * 1000
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def measure(victim_query, noise_query, label):
    """victim p50/p95 idle, then with NOISE threads hammering the noisy tenant."""
    idle = [(_t(victim_query)) for _ in range(NQ)]
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                noise_query()
            except Exception:
                pass
    threads = [threading.Thread(target=loop, daemon=True) for _ in range(NOISE)]
    for th in threads:
        th.start()
    time.sleep(0.5)  # let the noise ramp
    loaded = [(_t(victim_query)) for _ in range(NQ)]
    stop.set()
    for th in threads:
        th.join(timeout=1)
    ip50, ip95 = pcts(idle)
    lp50, lp95 = pcts(loaded)
    print(f"  {label:<22} idle p50/p95 {ip50:5.2f}/{ip95:5.2f}ms   under-load p50/p95 {lp50:5.2f}/{lp95:5.2f}ms")
    return dict(ip50=ip50, ip95=ip95, lp50=lp50, lp95=lp95)


def _t(fn):
    t = time.time(); fn(); return time.time() - t


def run_skeg(corpus):
    import redis
    port = free_port(); data = tempfile.mkdtemp(prefix="nn-skeg-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}", "--workers", str(WORKERS)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_tcp(port)
        r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=120)
        for t in range(NS):
            r.execute_command("SKEG.VINDEX.CREATE", f"t{t}", str(DIM), "tq2", "disk")
            for s in range(0, M, 512):
                a = ["SKEG.VMSET", f"t{t}"]
                for i in range(s, min(s + 512, M)):
                    a += [str(i), corpus[t][i].tobytes(), ""]
                r.execute_command(*a)
            r.execute_command("SKEG.VINDEX.CONSOLIDATE", f"t{t}")
        rv = redis.Redis(host="127.0.0.1", port=port)
        rn = redis.Redis(host="127.0.0.1", port=port)
        qv = corpus[VICTIM][0].tobytes()
        qn = corpus[NOISY][0].tobytes()
        return measure(lambda: rv.execute_command("SKEG.VSEARCH", f"t{VICTIM}", "10", "200", qv),
                       lambda: rn.execute_command("SKEG.VSEARCH", f"t{NOISY}", "10", "200", qn),
                       f"skeg (per-tenant, w={WORKERS})")
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


def run_qdrant(corpus):
    from qdrant_client import QdrantClient, models
    http, grpc = free_port(), free_port(); storage = tempfile.mkdtemp(prefix="nn-qd-")
    env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
           "QDRANT__STORAGE__STORAGE_PATH": storage, "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([QDRANT], env=env, cwd=storage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_tcp(http); time.sleep(1.0)
        cl = QdrantClient(host="127.0.0.1", port=http)
        cl.create_collection("shared", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
        cl.create_payload_index("shared", field_name="tenant", field_schema="integer")
        for t in range(NS):
            for s in range(0, M, 512):
                cl.upsert("shared", points=[models.PointStruct(id=t * M + s + j, vector=corpus[t][s + j].tolist(),
                          payload={"tenant": t}) for j in range(min(512, M - s))])
        while not str(cl.get_collection("shared").status).lower().endswith("green"):
            time.sleep(0.3)
        cv = QdrantClient(host="127.0.0.1", port=http)
        cn = QdrantClient(host="127.0.0.1", port=http)
        fv = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=VICTIM))])
        fn = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=NOISY))])
        qv, qn = corpus[VICTIM][0].tolist(), corpus[NOISY][0].tolist()
        return measure(lambda: cv.query_points("shared", query=qv, limit=10, query_filter=fv),
                       lambda: cn.query_points("shared", query=qn, limit=10, query_filter=fn),
                       "qdrant (shared+filter)")
    finally:
        p.terminate(); p.wait(); shutil.rmtree(storage, ignore_errors=True)


def main():
    rng = np.random.default_rng(7)
    corpus = [rng.standard_normal((M, DIM)).astype("<f4") for _ in range(NS)]
    print(f"Latency: {NS} tenants x {M} ({DIM}-dim), {NOISE} noise threads on tenant {NOISY}, victim {VICTIM}.")
    print("Lower is better.\n")
    sk = run_skeg(corpus)
    qd = run_qdrant(corpus)
    print(f"\n-> skeg vs qdrant p95: idle {qd['ip95'] / sk['ip95']:.1f}x faster, "
          f"under load {qd['lp95'] / sk['lp95']:.1f}x faster "
          f"(skeg {sk['ip95']:.2f}/{sk['lp95']:.2f}ms vs qdrant {qd['ip95']:.2f}/{qd['lp95']:.2f}ms)")


if __name__ == "__main__":
    main()
