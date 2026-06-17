# skeg-bench

Reproducible benchmark suite comparing [skeg](https://github.com/skegdb/skeg)
against [Qdrant](https://github.com/qdrant/qdrant) on the axes skeg is built
for: RAM density, recall, and multi-tenant isolation.

Every number in skeg's published charts comes from this suite. The harness is
public so the claims are reproducible, not self-reported.

## What it measures

| phase | script | question |
|-------|--------|----------|
| A — single-tenant scaling | `benches/singletenant.py` | RAM / recall / latency vs corpus size (100k–500k), skeg tiers vs qdrant |
| B — multi-tenant density | `benches/multitenant.py` | RAM / recall / latency vs tenant count; physical isolation **and** shared+filter |
| demo — tight container | `benches/container_oom.py` | pass/fail: serve N tenants in a fixed RAM cap where qdrant OOMs |

`benches/multitenant.py` runs two modes:
- default: skeg one-vindex-per-tenant (physical isolation) vs qdrant per-collection and shared+filter
- `MODE=filter`: skeg shared+filter vs qdrant shared+filter, apples-to-apples, with a cross-tenant **leak** check

## Methodology (read before quoting numbers)

- **Recall is real.** Ground truth is exact brute-force cosine top-k over the
  same corpus; recall@10 and recall@100 are measured against it with held-out
  real queries (mxbai-embed-large, 1024-dim). No proxy, no self-reported recall.
- **RAM is measured identically for both engines** via `ps -o rss`. Two numbers:
  `RSS_load` is the peak during build/ingest; `RSS_serve` is the steady state
  after load. Peak is the stable metric; steady-serve is allocator-noisy on
  skeg's small footprint, so lead with peak when in doubt.
- **Latency caveat.** skeg is queried over RESP3, qdrant over HTTP. Transports
  differ, so read latency as a trend, not an absolute head-to-head. RAM and
  recall are apples-to-apples.
- **Qdrant configs.** `qdrant-f32` (HNSW, vectors in RAM) is the fair baseline.
  `qdrant-int8` adds scalar quantization with `always_ram=True`, which keeps the
  f32 originals resident — worse on both RAM and recall, included only to show
  that quantizing qdrant does not close the gap.
- **Hardware.** Results in the repo were collected single-machine. Record the
  machine in `results/` when you run; the RAM ratios are hardware-independent,
  absolute latencies and build times are not.

## Running

```sh
pip install -r requirements.txt
# place a Qdrant binary at vendor/qdrant (see data/README.md)
# generate or fetch corpus + query .npy into data/ (see data/README.md)

python runner.py all          # everything, emits charts/
python runner.py singletenant # one phase
python runner.py multitenant
python runner.py filter        # multitenant shared+filter mode
python runner.py container     # the tight-RAM demo
python runner.py plots         # re-render charts from results/
```

Env knobs (see each script header): `SKEG_RESP3_BIN`, `SKEG_CORPUS`,
`SKEG_QUERIES`, `SCALES`, `NS`, `M`, `QPT`, `SKEG_TIERS`, `RAM_CAP_MB`.

## What the charts show (reference run, single-machine)

- **single-tenant:** skeg holds recall@10 ~1.0 at 5–7× lower peak RAM than
  qdrant-f32; qdrant builds ~2× faster (rebuild vs incremental HNSW).
- **multi-tenant:** skeg's RAM stays ~flat as tenants grow (bounded by the
  largest tenant); 9–17× lower peak RAM at 5 tenants, recall 1.0, zero leak.
- **shared+filter:** skeg beats qdrant on recall (1.0 vs 0.94–0.97) and RAM
  (3–4× peak, 5–10× serve), leaks=0 on both.

## Honest scope

Single-node, up to 1M vectors, one embedding model (mxbai-1024) plus a
secondary distribution (MiniLM-384) for the quant gates. Not distributed, not
billion-scale, not a throughput-saturation test. skeg's wedge is high-density
multi-tenant serving, and that is what this measures.
