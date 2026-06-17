# Results

Reference numbers from this suite. Reproduce any of them with
`python runner.py <phase>` (see [README](README.md)). Measured single-machine on
Apple Silicon (M1); RAM ratios are hardware-independent, absolute latencies and
build times are not. Recall is always against exact brute-force cosine ground
truth. RAM via `ps -o rss`.

## Cross-engine, single-tenant (mxbai-1024 @ 100K)

`runner.py engines` — every engine at a reasonable default config (LanceDB tuned
to recall 1.0 for a fair fight; the HNSW engines could trade latency for higher
recall but none would match skeg on RAM *and* recall together).

| engine | serve RAM | recall@10 | recall@100 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| **skeg-tq2** | **47 MB** | **1.000** | **1.000** | 2.49 | 3.57 |
| lancedb (IVF-PQ) | 198 MB | 0.998 | 0.991 | 59.26 | 98.14 |
| milvus-lite | 108 MB | 0.934 | 0.880 | 2.69 | 4.34 |
| hnswlib (raw HNSW) | 426 MB | 0.985 | 0.925 | 1.99 | 2.58 |
| chroma (HNSW) | 682 MB | 0.985 | 0.919 | 3.91 | 5.64 |
| qdrant-f32 | 885 MB | 0.997 | 0.981 | 2.62 | 3.40 |

skeg is the only engine that is simultaneously leanest, most accurate, and fast.
See `charts/chart_pareto.png`.

### Standard dataset — GloVe-100-angular @ 500K

`SKEG_CORPUS=data/glove_corpus.npy ... runner.py engines` — the canonical cosine
ann-benchmarks dataset (1.18M × 100-dim word vectors; skeg ranks by normalized
squared-L2, equivalent to cosine, so GloVe-angular is the right metric fit, not
the L2-native SIFT/GIST). A genuinely hard distribution — recall drops for
everyone — but skeg still leads on **both** recall and RAM, beating even
full-precision Qdrant (its on-disk f32 re-rank recovers what quantization loses):

| engine | serve RAM | recall@10 | recall@100 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| **skeg-tq2** | **31 MB** | **0.975** | **0.966** | 2.3 |
| lancedb (IVF-PQ) | 946 MB | 0.909 | 0.729 | 14.4 |
| qdrant-f32 | 611 MB | 0.899 | 0.823 | 2.0 |
| hnswlib (raw HNSW) | 344 MB | 0.838 | 0.709 | 0.5 |
| chroma (HNSW) | 616 MB | 0.818 | 0.696 | 2.6 |
| milvus-lite | 118 MB | 0.776 | 0.729 | 1.4 |

(HNSW engines at default `ef`; raising it trades latency for recall but not the
RAM gap. GloVe fetched by `quickstart.sh` / the snippet in `data/README.md`.)

## Single-tenant scaling (mxbai-1024, skeg tiers vs Qdrant)

`runner.py singletenant` — peak RAM during build / recall / latency at 100K–500K.

| config | peak RAM @500K | recall@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| skeg-int8 | 794 MB | 0.9995 | 4.98 | 17.17 |
| skeg-tq4 | 998 MB | 0.9995 | 3.49 | 8.35 |
| **skeg-tq2** | 901 MB | 1.0000 | 4.70 | 12.42 |
| skeg-tq1 | 1021 MB | 1.0000 | 6.55 | 11.51 |
| qdrant-f32 | 5436 MB | 0.9875 | 3.02 | 4.99 |
| qdrant-int8 | 5051 MB | 0.9455 | 2.36 | 4.45 |

Steady serve RSS is far lower (skeg 76–198 MB vs qdrant 2331–2563 MB @500K);
peak is the stable, quotable number. `charts/chart_singletenant*.png`.

## Multi-tenant density (100K vectors / tenant)

`runner.py multitenant` — skeg's RAM is bounded by the largest tenant, not the
total. Leak counts are 0 by construction (physical isolation) and 0 measured
(shared + filter).

| 5 tenants × 100K | peak RAM | recall | leaks |
| --- | ---: | ---: | ---: |
| **skeg (per-tenant index)** | **~200 MB** | **1.000** | 0 |
| qdrant (collection per tenant) | 1818 MB | 0.996 | 0 |
| qdrant (shared + filter) | 3494 MB | 0.957 | 0 |

9–17× less RAM at higher recall. `charts/chart_multitenant.png`.

### Shared index + per-tenant filter (apples-to-apples)

`runner.py filter` — both engines: one index, per-tenant filter.

| 5 tenants × 100K | peak RAM | serve RSS | recall | leaks |
| --- | ---: | ---: | ---: | ---: |
| **skeg-tq2 shared+filter** | 1471 MB | **222 MB** | **0.999** | 0 |
| qdrant shared+filter | 4131 MB | 2213 MB | 0.941 | 0 |

`charts/chart_shared_filter.png`.

## Tight container — MNIST 60K, 256 MB cap

`runner.py container` — the test others use to show Qdrant OOM-killed.

| engine | peak RAM | recall@10 | verdict @ 256 MB |
| --- | ---: | ---: | :---: |
| **skeg-tq2** | **156 MB** | 1.000 | ✅ PASS |
| qdrant-f32 | 473 MB | 1.000 | ❌ OOM |

MNIST recall/latency (all skeg tiers hold recall@10 1.0 on raw pixels; Qdrant's
int8 quantization collapses to 0.914):

| config | peak RAM | recall@10 | recall@100 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| skeg-tq2 | 156 MB | 1.000 | 1.000 | 2.07 |
| qdrant-f32 | 423 MB | 1.000 | 0.999 | 2.33 |
| qdrant-int8 | 460 MB | 0.914 | 0.955 | 2.05 |

The authoritative kernel-enforced version: `docker/run-oom-demo.sh`.

## Tenant isolation — adversarial leak-fuzz

`runner.py isolation` — query a tenant's index with another tenant's *exact*
vector (its own nearest neighbour) while filtering for the victim, plus
mid-stream deletes and consolidation.

```text
8 tenants × 2000 vectors, 200 rounds, 604 tenant-scoped queries
cross-tenant leaks: 0   ->   PASS
```

## Latency under load

`runner.py latency` — a victim tenant's p95, idle and while a noisy tenant is
hammered by 8 threads.

| engine | idle p95 | under-load p95 |
| --- | ---: | ---: |
| **skeg (per-tenant, workers)** | **0.91 ms** | **2.84 ms** |
| qdrant (shared + filter) | 1.70 ms | 5.12 ms |

skeg is ~2× lower latency idle and under load. (Both degrade by a similar factor
under CPU saturation — this is a speed result, not an isolation claim.)

## Throughput

`runner.py throughput` — single-process QPS, GIL-free (client *processes*),
inline vs the `--workers` pool. 50K × 256-dim, tq2.

| | peak QPS (1 process) |
| --- | ---: |
| inline (`--workers 0`) | ~1,870 |
| pool (`--workers 8`) | ~1,560 |

Honest reading: QPS is noisy (±15%) and strongly dimension-dependent (heavier
vectors → fewer QPS). The takeaways are directional: one process scales with
concurrency to a per-process ceiling well above any single-shard folklore
number, and the worker pool does not raise that ceiling here. For a real figure,
measure at your own dimension/tier.

## What it costs

`python tools/cost_calculator.py --vectors 50_000_000 --dim 1024 --tier tq2 --price 4`

| 50M × 1024-dim @ $4/GB-month | resident RAM | $/year |
| --- | ---: | ---: |
| **skeg-tq2** | 19.4 GiB | **$930** |
| qdrant-f32 | 201 GiB | $9,647 |

90% lower memory cost at matched recall.
