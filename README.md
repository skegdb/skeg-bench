# skeg-bench

**Reproducible benchmarks: [skeg](https://github.com/skegdb/skeg) vs [Qdrant](https://github.com/qdrant/qdrant), on RAM, recall, and multi-tenant density.**

Every number in skeg's published charts comes from this suite. It's public so you can re-run it — not take our word for it.

---

## TL;DR — the headline

On **MNIST 60k in a 256 MB container** (the exact test others use to show Qdrant getting OOM-killed):

| engine | peak RAM | recall@10 | p50 | verdict @ 256 MB |
|--------|---------:|----------:|----:|:----------------:|
| **skeg** (tq2) | **156 MB** | **1.000** | 2.1 ms | ✅ **PASS** |
| Qdrant (f32) | 423 MB | 1.000 | 2.3 ms | ❌ **OOM** |
| Qdrant (int8) | 460 MB | 0.914 | 2.1 ms | ❌ **OOM** |

Same recall as Qdrant at **2.7× less RAM** — and when Qdrant quantizes to save memory, its recall collapses to 0.91 while skeg's stays at 1.0.

It gets better with **more tenants**: split the same 60k across 5 isolated tenants and skeg's peak RAM *drops* to 123 MB (each tenant is built independently), while Qdrant climbs. RAM bounded by your largest tenant, not your total.

---

## Quickstart (one command)

```sh
git clone https://github.com/skegdb/skeg-bench && cd skeg-bench
./quickstart.sh
```

`quickstart.sh` installs the Python deps, downloads a Qdrant binary, fetches
MNIST, and runs the container demo. ~3 minutes, prints the table above.

Want the authoritative OOM (real cgroup cap, not an RSS estimate)?

```sh
docker/run-oom-demo.sh        # runs each engine under `docker run --memory=256m`
```

`PASS` = stayed under the cap. `OOM` = exit code 137, killed by the kernel.

---

## What's in the box

```
quickstart.sh        one-command setup + demo
runner.py            run any phase: singletenant | multitenant | filter | container | plots | all
benches/             the four measurement scripts
plots/               chart renderers
examples/            how to USE skeg (ingest, search, multi-tenant, filtered) — runnable
docker/              ready-to-run images + the authoritative memory-capped OOM demo
data/                where corpora live (fetched, not committed)
charts/             reference figures
```

Run a single phase:

```sh
python runner.py container      # the tight-RAM demo
python runner.py singletenant   # RAM/recall/latency vs corpus size
python runner.py multitenant    # density vs tenant count (physical isolation)
python runner.py filter         # shared-index + per-tenant filter, with leak check
python runner.py all            # everything, re-renders charts/
```

---

## The full picture (reference runs)

### MNIST 60k, 784-dim, recall vs brute-force ground truth

| config | peak RAM | recall@10 | recall@100 | p50 | p95 |
|--------|---------:|----------:|-----------:|----:|----:|
| skeg-int8 | 134 | 1.000 | 1.000 | 2.1 | 2.4 |
| skeg-tq4  | 171 | 1.000 | 1.000 | 2.1 | 3.6 |
| **skeg-tq2** | **156** | **1.000** | **1.000** | 2.1 | 2.5 |
| skeg-tq1  | 116 | 1.000 | 0.999 | 3.2 | 4.5 |
| qdrant-f32  | 423 | 1.000 | 0.999 | 2.3 | 2.7 |
| qdrant-int8 | 460 | 0.914 | 0.955 | 2.1 | 2.3 |

> MNIST is highly clustered, so recall@10 is ~1.0 for everyone *except* Qdrant's
> int8 quantization. The story here is RAM (2.7× lower at equal recall) and the
> fact that skeg's quantization keeps recall where Qdrant's loses it.

### mxbai-1024 embeddings — where the gap widens

- **Single-tenant 100k–500k:** skeg holds recall@10 ~1.0 at **5–7× lower peak RAM** than Qdrant-f32. Qdrant builds ~2× faster at 500k (rebuild vs incremental HNSW) — though at 60k skeg is actually faster; the build gap is size-dependent.
- **Multi-tenant (3–5 tenants × 100k):** skeg's RAM stays ~flat as tenants grow; **9–17× lower peak RAM** at 5 tenants, recall 1.0, zero cross-tenant leak.
- **Shared index + per-tenant filter (apples-to-apples vs Qdrant's model):** skeg wins recall (1.0 vs 0.94–0.97) and RAM (3–4× peak, 5–10× serve), `leaks=0` on both.

See `charts/` for the figures.

---

## Methodology (read before quoting numbers)

- **Recall is real.** Ground truth is exact brute-force cosine top-k over the
  same corpus; recall@10/@100 are measured against it with held-out queries. No
  proxy, no self-reported recall.
- **RAM is measured identically** for both engines via `ps -o rss`. `peak` is the
  max during build/ingest (the stable, quotable number); `serve` is the steady
  state after load (allocator-noisy on skeg's small footprint — lead with peak).
- **Latency caveat.** skeg is queried over RESP3, Qdrant over HTTP. Transports
  differ, so read latency as a trend, not an absolute. RAM and recall are
  apples-to-apples.
- **Fair Qdrant config.** `qdrant-f32` (HNSW, vectors in RAM) is the baseline.
  `qdrant-int8` adds scalar quantization with `always_ram=True` (keeps f32
  originals resident) — included to show that quantizing Qdrant raises RAM *and*
  drops recall, not to stack the deck.
- **The container verdict.** `quickstart.sh`/`runner.py container` compares peak
  RSS to the cap (a process over the cap *would* be OOM-killed). For the
  authoritative kernel-enforced version, `docker/run-oom-demo.sh` runs each
  engine under a real `docker run --memory` cgroup cap and reads the exit code.

## Honest scope

Single-node, up to 1M vectors, mxbai-1024 + MNIST-784. Not distributed, not
billion-scale, not a throughput-saturation test. skeg's wedge is high-density
multi-tenant serving — that is what this measures, and only that.
