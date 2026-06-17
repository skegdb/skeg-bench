#!/usr/bin/env python
"""Single-process QPS saturation: how many queries/sec one skeg process serves,
inline (--workers 0) vs with the VSEARCH worker pool (--workers N).

Settles the "one shard tops out near 640 QPS, scale with processes not cores"
claim: if the worker pool lifts the ceiling, the claim is stale.

The load generator uses client *processes*, not threads, so Python's GIL does
not cap the offered load — what saturates is skeg, not the benchmark. Each
client process loops VSEARCH for a fixed window; QPS = total / elapsed. Sweeps
concurrency to find the peak. Self-contained (synthetic vectors).

Env: SKEG_RESP3_BIN, M (50000), DIM (256), SECS (3), CONC (1,2,4,8,16,32).
"""
import os
import shutil
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import redis


def _client_load(args):
    """Run in a separate process: hammer VSEARCH for `secs`, return count."""
    port, dim, secs, seed = args
    rng = np.random.default_rng(seed)
    qs = [rng.standard_normal(dim).astype("<f4").tobytes() for _ in range(64)]
    r = redis.Redis(host="127.0.0.1", port=port)
    n, i, end = 0, 0, time.time() + secs
    while time.time() < end:
        r.execute_command("SKEG.VSEARCH", "idx", "10", "200", qs[i % 64])
        n += 1; i += 1
    return n

SKEG = os.environ["SKEG_RESP3_BIN"]
M = int(os.environ.get("M", "50000"))
DIM = int(os.environ.get("DIM", "256"))
SECS = float(os.environ.get("SECS", "3"))
CONC = [int(x) for x in os.environ.get("CONC", "1,2,4,8,16,32").split(",")]


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


def peak_qps(port):
    # Each client process loops for exactly SECS *internally* (its window starts
    # after imports/connect), so the aggregate denominator is SECS, not the
    # wall-clock that would include process-spawn overhead. The pool is warmed
    # by a throwaway round first so spawn cost never lands in a measured window.
    with ProcessPoolExecutor(max_workers=max(CONC)) as ex:
        list(ex.map(_client_load, [(port, DIM, 0.3, 999 + w) for w in range(max(CONC))]))  # warm
        best = (0, 0.0)
        for c in CONC:
            counts = list(ex.map(_client_load, [(port, DIM, SECS, w) for w in range(c)]))
            qps = sum(counts) / SECS
            print(f"    {c:>2} client procs: {qps:>8.0f} QPS")
            if qps > best[1]:
                best = (c, qps)
    return best


def run(workers, corpus):
    port = free_port(); data = tempfile.mkdtemp(prefix="tp-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}", "--workers", str(workers)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_tcp(port)
        r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=120)
        r.execute_command("SKEG.VINDEX.CREATE", "idx", str(DIM), "tq2", "disk")
        for s in range(0, M, 512):
            a = ["SKEG.VMSET", "idx"]
            for i in range(s, min(s + 512, M)):
                a += [str(i), corpus[i].tobytes(), ""]
            r.execute_command(*a)
        r.execute_command("SKEG.VINDEX.CONSOLIDATE", "idx")
        print(f"  --workers {workers}:")
        return peak_qps(port)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


def main():
    rng = np.random.default_rng(3)
    corpus = rng.standard_normal((M, DIM)).astype("<f4")
    queries = rng.standard_normal((256, DIM)).astype("<f4")
    print(f"Throughput: {M} vectors x {DIM}-dim, tq2, {SECS}s windows. One skeg process.\n")
    c0, q0 = run(0, corpus)
    c8, q8 = run(8, corpus)
    print(f"\n-> peak QPS  inline(w=0) {q0:.0f} @ conc {c0}   |   pool(w=8) {q8:.0f} @ conc {c8}"
          f"   ({q8 / max(q0, 1):.1f}x from the worker pool)")


if __name__ == "__main__":
    main()
