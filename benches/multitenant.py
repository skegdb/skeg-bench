#!/usr/bin/env python
"""Part C of the comparative bench: multi-tenant DENSITY (the wedge).

For a growing number of ISOLATED tenants (each = M disjoint vectors from the 1m
wiki corpus, mxbai-1024), compare three ways to serve them:

  skeg            ONE skeg-resp3 process, N isolated vindexes (idx_0..idx_{N-1}).
  qdrant-coll     ONE qdrant, N collections (one per tenant) - physical isolation.
  qdrant-shared   ONE qdrant, ONE collection, tenant_id in the payload + a per-query
                  `must: tenant_id=t` filter - logical isolation (a missing filter leaks).

Per config:
  RSS_load(MiB)  PEAK resident memory of the engine process DURING ingest/indexing
                 (qdrant's HNSW build + segment optimizer spike well above steady).
  RSS_serve(MiB) steady resident memory AFTER load (the serving footprint).
  recall@10      mean per-tenant recall vs brute-force top-10 over that tenant's slice.
  p50/p95(ms)    per-tenant query latency; skeg=RESP3, qdrant=HTTP - transports differ.
  leak           (qdrant-shared) #queries that returned another tenant's vector (must be 0).

Both RAM numbers matter: serve = what it costs to run; load-peak = whether the build
even fits the machine (qdrant peaks ~2.3x its steady; at scale it hits the RAM ceiling).

Env: SKEG_RESP3_BIN, SKEG_CORPUS(1m npy), SKEG_QUERIES. NS="3,10,30" M=20000 QPT=20.
One engine at a time (clean RSS).
"""
import os, time, socket, subprocess, tempfile, shutil
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SKEG = os.environ["SKEG_RESP3_BIN"]
QDRANT = os.path.join(ROOT, "vendor", "qdrant")
DIM, K = 1024, 10
NS = [int(x) for x in os.environ.get("NS", "3,10,30").split(",")]
M = int(os.environ.get("M", "20000"))
QPT = int(os.environ.get("QPT", "20"))


def load_npy(path, limit):
    with open(path, "rb") as f:
        assert f.read(6) == b"\x93NUMPY"; f.read(2)
        hlen = int.from_bytes(f.read(2), "little"); hdr = f.read(hlen).decode()
        cols = int(hdr.split("'shape':")[1].split(",")[1].split(")")[0])
        data = np.frombuffer(f.read(limit * cols * 4), dtype="<f4")
    return data.reshape(limit, cols).copy()


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


def rss_mib(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) / 1024


def steady(pid):
    return max(rss_mib(pid) for _ in range(5))


def gt_for(corpus, queries, t):
    sub = corpus[t * M:(t + 1) * M]
    sn = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    top = np.argsort(-(qn @ sn.T), axis=1)[:, :K]
    return [set(int(x) for x in top[qi]) for qi in range(len(queries))]


def qstats(recs, lat):
    a = np.array(lat) * 1000
    return float(np.mean(recs)), float(np.percentile(a, 50)), float(np.percentile(a, 95))


def run_skeg(corpus, queries, N, tier="int8"):
    import redis
    port = free_port(); data = tempfile.mkdtemp(prefix="mt-skeg-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    try:
        assert wait_tcp(port); r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=600)
        for t in range(N):
            r.execute_command("SKEG.VINDEX.CREATE", f"idx{t}", str(DIM), tier, "disk")
            sl = corpus[t * M:(t + 1) * M]; B = 256
            for s in range(0, M, B):
                a = ["SKEG.VMSET", f"idx{t}"]
                for i in range(s, min(s + B, M)): a += [str(i), sl[i].tobytes(), ""]
                r.execute_command(*a)
                if (s // B) % 20 == 0: peak = max(peak, rss_mib(p.pid))
            r.execute_command("SKEG.VINDEX.CONSOLIDATE", f"idx{t}")
            peak = max(peak, rss_mib(p.pid))
        serve = steady(p.pid)
        recs, lat = [], []
        for t in range(N):
            gt = gt_for(corpus, queries[:QPT], t); got = []
            for qi in range(QPT):
                t0 = time.time()
                res = r.execute_command("SKEG.VSEARCH", f"idx{t}", str(K), "200", queries[qi].tobytes())
                lat.append(time.time() - t0)
                got.append([int(res[j]) for j in range(0, len(res), 2)])
            recs.append(np.mean([len(set(got[qi]) & gt[qi]) / K for qi in range(QPT)]))
        return (peak, serve, *qstats(recs, lat))
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


def run_skeg_shared_filter(corpus, queries, N, tier="int8"):
    """One vindex, all tenants tagged payload `tenant=<t>`, queried with a
    per-tenant FILTER. Apples-to-apples vs qdrant shared+filter, incl. leak."""
    import redis
    port = free_port(); data = tempfile.mkdtemp(prefix="mt-skf-")
    p = subprocess.Popen([SKEG, "--data-dir", data, "--addr", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0.0
    try:
        assert wait_tcp(port); r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=600)
        r.execute_command("SKEG.VINDEX.CREATE", "shared", str(DIM), tier, "disk")
        for t in range(N):
            sl = corpus[t * M:(t + 1) * M]; B = 256
            for s in range(0, M, B):
                a = ["SKEG.VMSET", "shared"]
                for i in range(s, min(s + B, M)): a += [str(t * M + i), sl[i].tobytes(), f"tenant={t}"]
                r.execute_command(*a)
                if (s // B) % 20 == 0: peak = max(peak, rss_mib(p.pid))
        r.execute_command("SKEG.VINDEX.CONSOLIDATE", "shared"); peak = max(peak, rss_mib(p.pid))
        serve = steady(p.pid)
        recs, lat, leak = [], [], 0
        for t in range(N):
            gt = gt_for(corpus, queries[:QPT], t); got = []
            for qi in range(QPT):
                t0 = time.time()
                res = r.execute_command("SKEG.VSEARCH", "shared", str(K), "200",
                                        queries[qi].tobytes(), "FILTER", f"tenant={t}")
                lat.append(time.time() - t0)
                ids = [int(res[j]) for j in range(0, len(res), 2)]
                if any(gid // M != t for gid in ids): leak += 1
                got.append([gid - t * M for gid in ids])
            recs.append(np.mean([len(set(got[qi]) & gt[qi]) / K for qi in range(QPT)]))
        return (peak, serve, *qstats(recs, lat), leak)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(data, ignore_errors=True)


def _qdrant(prefix):
    from qdrant_client import QdrantClient, models
    http, grpc = free_port(), free_port(); storage = tempfile.mkdtemp(prefix=prefix)
    env = {**os.environ, "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
           "QDRANT__STORAGE__STORAGE_PATH": storage, "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([QDRANT], env=env, cwd=storage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_tcp(http); time.sleep(1.0)
    return p, storage, QdrantClient(host="127.0.0.1", port=http), models


def run_qdrant_coll(corpus, queries, N):
    p, storage, cl, models = _qdrant("mt-qc-"); peak = 0.0
    try:
        for t in range(N):
            cl.create_collection(f"t{t}", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
            sl = corpus[t * M:(t + 1) * M]
            for s in range(0, M, 256):
                cl.upsert(f"t{t}", points=[models.PointStruct(id=s + j, vector=sl[s + j].tolist())
                                           for j in range(min(256, M - s))])
                if (s // 256) % 20 == 0: peak = max(peak, rss_mib(p.pid))
        for t in range(N):
            while not str(cl.get_collection(f"t{t}").status).lower().endswith("green"):
                time.sleep(0.3); peak = max(peak, rss_mib(p.pid))
        serve = steady(p.pid)
        recs, lat = [], []
        for t in range(N):
            gt = gt_for(corpus, queries[:QPT], t); got = []
            for qi in range(QPT):
                t0 = time.time()
                pts = cl.query_points(f"t{t}", query=queries[qi].tolist(), limit=K).points
                lat.append(time.time() - t0); got.append([pt.id for pt in pts])
            recs.append(np.mean([len(set(got[qi]) & gt[qi]) / K for qi in range(QPT)]))
        return (peak, serve, *qstats(recs, lat))
    finally:
        p.terminate(); p.wait(); shutil.rmtree(storage, ignore_errors=True)


def run_qdrant_shared(corpus, queries, N):
    p, storage, cl, models = _qdrant("mt-qs-"); peak = 0.0
    try:
        cl.create_collection("shared", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
        cl.create_payload_index("shared", field_name="tenant", field_schema="integer")
        for t in range(N):
            sl = corpus[t * M:(t + 1) * M]
            for s in range(0, M, 256):
                cl.upsert("shared", points=[models.PointStruct(id=t * M + s + j, vector=sl[s + j].tolist(),
                          payload={"tenant": t}) for j in range(min(256, M - s))])
                if (s // 256) % 20 == 0: peak = max(peak, rss_mib(p.pid))
        while not str(cl.get_collection("shared").status).lower().endswith("green"):
            time.sleep(0.3); peak = max(peak, rss_mib(p.pid))
        serve = steady(p.pid)
        recs, lat, leak = [], [], 0
        for t in range(N):
            gt = gt_for(corpus, queries[:QPT], t); got = []
            flt = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=t))])
            for qi in range(QPT):
                t0 = time.time()
                pts = cl.query_points("shared", query=queries[qi].tolist(), limit=K, query_filter=flt).points
                lat.append(time.time() - t0)
                if any(pt.id // M != t for pt in pts): leak += 1
                got.append([pt.id - t * M for pt in pts])
            recs.append(np.mean([len(set(got[qi]) & gt[qi]) / K for qi in range(QPT)]))
        return (peak, serve, *qstats(recs, lat), leak)
    finally:
        p.terminate(); p.wait(); shutil.rmtree(storage, ignore_errors=True)


def main():
    corpus = load_npy(os.environ["SKEG_CORPUS"], max(NS) * M)
    queries = load_npy(os.environ["SKEG_QUERIES"], QPT)
    print(f"Multi-tenant density: {M} vectors/tenant (mxbai-1024), {QPT} queries/tenant.")
    print("RSS_load = PEAK during ingest/indexing; RSS_serve = steady after load (both ps -o rss).")
    print("Latency caveat: skeg=RESP3, qdrant=HTTP - transports differ; RAM+recall are apples-to-apples.\n")
    skeg_tiers = [t for t in os.environ.get("SKEG_TIERS", "int8,tq4,tq2,tq1").split(",") if t]
    # MODE=filter: focused apples-to-apples on the FILTERED path only -
    # skeg shared+filter (per tier) vs qdrant shared+filter, both with leak check.
    if os.environ.get("MODE") == "filter":
        for N in NS:
            print(f"=== N={N} tenants  ({N*M} vectors total, {M}/tenant) - SHARED+FILTER ===")
            print(f"  {'config':<24} {'RSS_load':>8} {'RSS_serve':>9} {'recall':>6} {'p50ms':>6} {'p95ms':>6}  leaks", flush=True)
            for tier in skeg_tiers:
                s = run_skeg_shared_filter(corpus, queries, N, tier=tier)
                print(f"  {('skeg-'+tier+' shared+filter'):<24} {s[0]:>8.0f} {s[1]:>9.0f} {s[2]:>6.3f} {s[3]:>6.2f} {s[4]:>6.2f}  {s[5]}", flush=True)
            qs = run_qdrant_shared(corpus, queries, N)
            print(f"  {'qdrant shared+filter':<24} {qs[0]:>8.0f} {qs[1]:>9.0f} {qs[2]:>6.3f} {qs[3]:>6.2f} {qs[4]:>6.2f}  {qs[5]}\n", flush=True)
        return
    for N in NS:
        print(f"=== N={N} tenants  ({N*M} vectors total, {M}/tenant) ===")
        hdr = f"  {'config':<22} {'RSS_load':>8} {'RSS_serve':>9} {'recall':>6} {'p50ms':>6} {'p95ms':>6}  notes"
        print(hdr, flush=True)
        sk_peaks = {}
        for tier in skeg_tiers:
            sk = run_skeg(corpus, queries, N, tier=tier)
            sk_peaks[tier] = sk[0]
            print(f"  {('skeg-'+tier+' (N vindexes)'):<22} {sk[0]:>8.0f} {sk[1]:>9.0f} {sk[2]:>6.3f} {sk[3]:>6.2f} {sk[4]:>6.2f}  1 process, isolated", flush=True)
        qc = run_qdrant_coll(corpus, queries, N)
        print(f"  {'qdrant per-collection':<22} {qc[0]:>8.0f} {qc[1]:>9.0f} {qc[2]:>6.3f} {qc[3]:>6.2f} {qc[4]:>6.2f}  {N} collections", flush=True)
        qs = run_qdrant_shared(corpus, queries, N)
        print(f"  {'qdrant shared+filter':<22} {qs[0]:>8.0f} {qs[1]:>9.0f} {qs[2]:>6.3f} {qs[3]:>6.2f} {qs[4]:>6.2f}  leaks={qs[5]}", flush=True)
        best = min(sk_peaks.values())
        print(f"  -> peak RAM: skeg-best {best:.0f}MB vs qdrant coll {qc[0]:.0f} ({qc[0]/best:.1f}x) / shared {qs[0]:.0f} ({qs[0]/best:.1f}x)\n", flush=True)


if __name__ == "__main__":
    main()
