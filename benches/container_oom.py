#!/usr/bin/env python
"""Tight-container demo: how many tenants fit under a fixed RAM cap.

Inspired by brinicle's "256MB container, qdrant OOMKilled" table, with skeg's
own angle: the cap holds MULTIPLE isolated tenants, not one corpus. We ingest
`NS` tenants x `M` vectors into each engine, measure peak RSS (build) and steady
serve RSS, and verdict PASS if the engine stays under `RAM_CAP_MB`.

On macOS there is no hard cgroup cap, so this measures peak RSS and compares it
to the cap (a process that exceeds the cap WOULD be OOMKilled in a real
`docker run --memory=<cap>m` container). The authoritative run is exactly that
docker invocation on Linux; this script is the local, scriptable proxy and
emits the same PASS / WOULD-OOM table.

Env: SKEG_RESP3_BIN, SKEG_CORPUS, SKEG_QUERIES, RAM_CAP_MB (default 512),
NS (tenants, default 5), M (vectors/tenant, default 100000), SKEG_TIER
(default tq2), DIM (default 1024).
"""
import os
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEG = os.environ["SKEG_RESP3_BIN"]
QDRANT = os.path.join(ROOT, "vendor", "qdrant")
DIM = int(os.environ.get("DIM", "1024"))
CAP = int(os.environ.get("RAM_CAP_MB", "512"))
NS = int(os.environ.get("NS", "5"))
M = int(os.environ.get("M", "100000"))
TIER = os.environ.get("SKEG_TIER", "tq2")
QPT = 10


def load_npy(path, limit):
    with open(path, "rb") as f:
        assert f.read(6) == b"\x93NUMPY"
        f.read(2)
        hlen = int.from_bytes(f.read(2), "little")
        hdr = f.read(hlen).decode()
        cols = int(hdr.split("'shape':")[1].split(",")[1].split(")")[0])
        data = np.frombuffer(f.read(limit * cols * 4), dtype="<f4")
    return data.reshape(limit, cols).copy()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_tcp(port, t=60):
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def rss_mib(pid):
    o = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(o.stdout.strip() or 0) / 1024


def serve_skeg(corpus, queries):
    import redis

    port = free_port()
    data = tempfile.mkdtemp(prefix="oom-skeg-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    try:
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
                if (s // 256) % 20 == 0:
                    peak = max(peak, rss_mib(p.pid))
            r.execute_command("SKEG.VINDEX.CONSOLIDATE", f"idx{t}")
            peak = max(peak, rss_mib(p.pid))
        # confirm it actually serves under the cap
        ok = True
        for t in range(NS):
            for qi in range(QPT):
                res = r.execute_command("SKEG.VSEARCH", f"idx{t}", "10", "200", queries[qi].tobytes())
                ok = ok and len(res) > 0
        serve = max(rss_mib(p.pid) for _ in range(5))
        return peak, serve, ok
    finally:
        p.terminate()
        p.wait()
        shutil.rmtree(data, ignore_errors=True)


def serve_qdrant(corpus, queries):
    from qdrant_client import QdrantClient, models

    http, grpc = free_port(), free_port()
    storage = tempfile.mkdtemp(prefix="oom-qd-")
    env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(http),
           "QDRANT__SERVICE__GRPC_PORT": str(grpc),
           "QDRANT__STORAGE__STORAGE_PATH": storage, "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([QDRANT], env=env, cwd=storage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    try:
        assert wait_tcp(http)
        time.sleep(1.0)
        cl = QdrantClient(host="127.0.0.1", port=http)
        for t in range(NS):
            cl.create_collection(f"t{t}", vectors_config=models.VectorParams(
                size=DIM, distance=models.Distance.COSINE))
            sl = corpus[t * M:(t + 1) * M]
            for s in range(0, M, 256):
                cl.upsert(f"t{t}", points=[models.PointStruct(id=s + j, vector=sl[s + j].tolist())
                                           for j in range(min(256, M - s))])
                if (s // 256) % 20 == 0:
                    peak = max(peak, rss_mib(p.pid))
        for t in range(NS):
            while not str(cl.get_collection(f"t{t}").status).lower().endswith("green"):
                time.sleep(0.3)
                peak = max(peak, rss_mib(p.pid))
        ok = True
        for t in range(NS):
            for qi in range(QPT):
                pts = cl.query_points(f"t{t}", query=queries[qi].tolist(), limit=10).points
                ok = ok and len(pts) > 0
        serve = max(rss_mib(p.pid) for _ in range(5))
        return peak, serve, ok
    finally:
        p.terminate()
        p.wait()
        shutil.rmtree(storage, ignore_errors=True)


def verdict(peak):
    # A process whose peak RSS exceeds the cap would be OOMKilled in a
    # `docker run --memory=<cap>m` container.
    return "PASS" if peak <= CAP else "WOULD-OOM"


def main():
    total = NS * M
    corpus = load_npy(os.environ["SKEG_CORPUS"], total)
    queries = load_npy(os.environ["SKEG_QUERIES"], QPT)
    print(f"# Tight-container demo: {NS} tenants x {M} = {total} vectors ({DIM}-dim), cap {CAP} MB")
    print(f"# skeg tier: {TIER}. Peak RSS vs cap = PASS / WOULD-OOM (docker --memory={CAP}m on Linux is authoritative).\n")

    sk_peak, sk_serve, sk_ok = serve_skeg(corpus, queries)
    qd_peak, qd_serve, qd_ok = serve_qdrant(corpus, queries)

    print(f"| engine | tenants | vectors | peak RSS (MB) | serve RSS (MB) | serves? | verdict @ {CAP}MB |")
    print("|---|---|---|---|---|---|---|")
    print(f"| skeg-{TIER} (isolated) | {NS} | {total} | {sk_peak:.0f} | {sk_serve:.0f} | {'yes' if sk_ok else 'NO'} | {verdict(sk_peak)} |")
    print(f"| qdrant (per-collection) | {NS} | {total} | {qd_peak:.0f} | {qd_serve:.0f} | {'yes' if qd_ok else 'NO'} | {verdict(qd_peak)} |")
    print(f"\n-> skeg peak {sk_peak:.0f}MB vs qdrant {qd_peak:.0f}MB = {qd_peak / max(sk_peak, 1):.1f}x")


if __name__ == "__main__":
    main()
