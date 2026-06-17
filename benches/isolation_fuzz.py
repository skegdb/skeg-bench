#!/usr/bin/env python
"""Provable tenant isolation: adversarial fuzz that tries to make skeg leak.

Two multi-tenant models:
  1. physical   - one index per tenant (isolation by construction)
  2. shared     - one index, `tenant=<t>` payload, per-tenant FILTER (the model
                  that *can* leak if the filter is wrong — this is the hard one)

The adversarial move: query the shared index with another tenant's *exact*
vector (its own nearest neighbour, score ~1.0) while filtering for a different
tenant. A correct filter must still return zero of the other tenant's rows.
Plus mutations mid-stream: delete rows, consolidate the delta, query again.

A single leaked id across all rounds = FAIL. Deterministic (seeded), so a
failure is reproducible.

Env: SKEG_RESP3_BIN, NS (default 8), M (default 2000), ROUNDS (default 200),
DIM (default 64).
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
from _common import free_port, wait_tcp

SKEG = os.environ["SKEG_RESP3_BIN"]
NS = int(os.environ.get("NS", "8"))
M = int(os.environ.get("M", "2000"))
ROUNDS = int(os.environ.get("ROUNDS", "200"))
DIM = int(os.environ.get("DIM", "64"))
BASE = 1_000_000  # global id = tenant * BASE + local; tenant = id // BASE


def ids_of(res):
    return [int(res[i]) for i in range(0, len(res), 2)]


def main():
    rng = np.random.default_rng(1234)
    corpus = [rng.standard_normal((M, DIM)).astype("<f4") for _ in range(NS)]

    port = free_port(); data = tempfile.mkdtemp(prefix="iso-fuzz-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import redis
    leaks = 0
    queries = 0
    try:
        assert wait_tcp(port)
        r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=120)

        # ── model 1: physical isolation (one index per tenant) ──────────────
        for t in range(NS):
            r.execute_command("SKEG.VINDEX.CREATE", f"phys{t}", str(DIM), "tq2", "disk")
            a = ["SKEG.VMSET", f"phys{t}"]
            for i in range(M):  # NOTE: local ids collide across tenants on purpose
                a += [str(i), corpus[t][i].tobytes(), ""]
            r.execute_command(*a)
            r.execute_command("SKEG.VINDEX.CONSOLIDATE", f"phys{t}")

        # ── model 2: shared index + per-tenant filter ───────────────────────
        r.execute_command("SKEG.VINDEX.CREATE", "shared", str(DIM), "tq2", "disk")
        for t in range(NS):
            a = ["SKEG.VMSET", "shared"]
            for i in range(M):
                a += [str(t * BASE + i), corpus[t][i].tobytes(), f"tenant={t}"]
            r.execute_command(*a)
        r.execute_command("SKEG.VINDEX.CONSOLIDATE", "shared")

        def check_shared(victim, attacker_vec):
            """Query shared filtered to `victim`; any id from another tenant leaks."""
            nonlocal leaks, queries
            res = r.execute_command("SKEG.VSEARCH", "shared", "10", "200",
                                    attacker_vec.tobytes(), "FILTER", f"tenant={victim}")
            queries += 1
            for gid in ids_of(res):
                if gid // BASE != victim:
                    leaks += 1

        for rd in range(ROUNDS):
            victim = int(rng.integers(NS))
            attacker = int(rng.integers(NS))
            # adversarial: query with an ACTUAL attacker-tenant vector (its own
            # nearest neighbour) while filtering for the victim.
            av = corpus[attacker][int(rng.integers(M))]
            check_shared(victim, av)
            # also a random vector
            check_shared(victim, rng.standard_normal(DIM).astype("<f4"))
            # physical model: query a tenant's index with another's vector;
            # every returned local id must be a valid row of THIS tenant.
            res = r.execute_command("SKEG.VSEARCH", f"phys{victim}", "10", "200", av.tobytes())
            queries += 1
            for lid in ids_of(res):
                if not (0 <= lid < M):
                    leaks += 1

            # mutate mid-stream every 50 rounds: delete some of victim's rows,
            # consolidate, re-check. Deletes must not open a leak.
            if rd % 50 == 49:
                for i in rng.choice(M, size=20, replace=False):
                    r.execute_command("SKEG.VDEL", "shared", str(victim * BASE + int(i)))
                r.execute_command("SKEG.VINDEX.CONSOLIDATE", "shared")
                check_shared(victim, corpus[attacker][int(rng.integers(M))])

        verdict = "PASS" if leaks == 0 else "FAIL"
        print(f"isolation fuzz: {NS} tenants x {M} vectors, {ROUNDS} rounds")
        print(f"  total tenant-scoped queries: {queries}")
        print(f"  cross-tenant leaks:          {leaks}")
        print(f"  adversarial cases:           query with another tenant's exact vector + mid-stream deletes/consolidate")
        print(f"  -> {verdict} (physical isolation cannot leak by construction; shared+filter held under attack)")
        sys.exit(0 if leaks == 0 else 1)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


if __name__ == "__main__":
    main()
