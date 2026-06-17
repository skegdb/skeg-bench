#!/usr/bin/env python
"""Drive ONE engine to ingest + serve MNIST, inside a memory-capped container.

The kernel decides the verdict: if the engine fits the cgroup cap, this exits 0
(PASS); if it exceeds, the OOM killer terminates it and the container exits 137.
There is no soft RSS comparison here — this is the authoritative version.

  ENGINE=skeg|qdrant  NS=<tenants>  M=<vectors/tenant>  DIM=<dim>  SKEG_TIER=<tier>
"""
import os
import socket
import subprocess
import sys
import time

import numpy as np

ENGINE = os.environ.get("ENGINE", "skeg")
DIM = int(os.environ.get("DIM", "784"))
NS = int(os.environ.get("NS", "1"))
M = int(os.environ.get("M", "60000"))
TIER = os.environ.get("SKEG_TIER", "tq2")
CORPUS = os.environ.get("SKEG_CORPUS", "/data/mnist_corpus_60k.npy")
QUERIES = os.environ.get("SKEG_QUERIES", "/data/mnist_queries_200.npy")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_tcp(port, t=60):
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close(); return True
        except OSError:
            time.sleep(0.1)
    return False


def run_skeg(corpus, queries):
    import redis
    port = free_port()
    p = subprocess.Popen([os.environ["SKEG_RESP3_BIN"], "--data-dir", "/tmp/skeg", "--addr", f"127.0.0.1:{port}"])
    assert wait_tcp(port)
    r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=900)
    for t in range(NS):
        r.execute_command("SKEG.VINDEX.CREATE", f"idx{t}", str(DIM), TIER, "disk")
        sl = corpus[t * M:(t + 1) * M]
        for s in range(0, M, 256):
            a = ["SKEG.VMSET", f"idx{t}"]
            for i in range(s, min(s + 256, M)):
                a += [str(i), sl[i].tobytes(), ""]
            r.execute_command(*a)
        r.execute_command("SKEG.VINDEX.CONSOLIDATE", f"idx{t}")
    for t in range(NS):
        r.execute_command("SKEG.VSEARCH", f"idx{t}", "10", "200", queries[0].tobytes())
    p.terminate(); p.wait()


def run_qdrant(corpus, queries):
    from qdrant_client import QdrantClient, models
    http, grpc = free_port(), free_port()
    env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
           "QDRANT__STORAGE__STORAGE_PATH": "/tmp/qdrant", "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen(["/usr/local/bin/qdrant"], env=env)
    assert wait_tcp(http); time.sleep(1.0)
    cl = QdrantClient(host="127.0.0.1", port=http)
    for t in range(NS):
        cl.create_collection(f"t{t}", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
        sl = corpus[t * M:(t + 1) * M]
        for s in range(0, M, 256):
            cl.upsert(f"t{t}", points=[models.PointStruct(id=s + j, vector=sl[s + j].tolist())
                                       for j in range(min(256, M - s))])
        while not str(cl.get_collection(f"t{t}").status).lower().endswith("green"):
            time.sleep(0.3)
    for t in range(NS):
        cl.query_points(f"t{t}", query=queries[0].tolist(), limit=10)
    p.terminate(); p.wait()


def main():
    corpus = np.load(CORPUS)[:NS * M].astype("<f4")
    queries = np.load(QUERIES).astype("<f4")
    print(f"[{ENGINE}] ingesting {NS}x{M}={NS*M} vectors ({DIM}-dim) under the container cap...", flush=True)
    (run_skeg if ENGINE == "skeg" else run_qdrant)(corpus, queries)
    print(f"[{ENGINE}] PASS — fit under the cap, served queries.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # any crash is a fail, but the OOM path is the kernel killing us (137)
        print(f"[{ENGINE}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
