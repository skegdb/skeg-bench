# Benchmark data

Corpora and query sets are not committed (large, regenerable). Each is a
`.npy` float32 array, row-major, shape `(n, dim)`.

## What you need

| env var | file | shape |
|---------|------|-------|
| `SKEG_CORPUS` | corpus `.npy` | `(>=500_000, 1024)` for the full single-tenant sweep |
| `SKEG_QUERIES` | query `.npy` | `(>=200, 1024)`, held out from the corpus |

The reference runs use **mxbai-embed-large** (1024-dim) embeddings of a wiki
chunk corpus. The quant gates additionally use **MiniLM** (384-dim) as a second
distribution. Any real embedding set works as long as queries are held out from
the corpus (so recall is not trivially 1.0).

## Generating

Embed a text corpus with your model of choice and `np.save` the float32 matrix.
Keep the last few hundred rows aside as the query set. Larger is better for the
density story — 500k single-tenant and 5×100k multi-tenant are the headline
points; 1M mono-tenant is the stress point.

## Qdrant binary

Place a Qdrant server binary at `vendor/qdrant` (downloaded from
[Qdrant releases](https://github.com/qdrant/qdrant/releases) or built from
source). The harness boots a fresh instance per config on an ephemeral port,
so no running Qdrant service is required.
