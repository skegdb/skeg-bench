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

## Standard dataset: GloVe-100-angular

The canonical cosine ann-benchmarks set (1.18M × 100-dim). skeg ranks by
normalized squared-L2 (≡ cosine), so use an angular dataset, not L2-native
SIFT/GIST. Fetch once into `data/`:

```python
import urllib.request, h5py, numpy as np
req = urllib.request.Request("https://ann-benchmarks.com/glove-100-angular.hdf5",
                             headers={"User-Agent": "Mozilla/5.0"})
open("data/glove-100-angular.hdf5", "wb").write(urllib.request.urlopen(req).read())
f = h5py.File("data/glove-100-angular.hdf5")
np.save("data/glove_corpus.npy", np.asarray(f["train"], dtype="<f4"))
np.save("data/glove_queries.npy", np.asarray(f["test"], dtype="<f4")[:200])
```

Then `SKEG_CORPUS=data/glove_corpus.npy SKEG_QUERIES=data/glove_queries.npy
N=500000 python runner.py engines`.

## Qdrant binary

Place a Qdrant server binary at `vendor/qdrant` (downloaded from
[Qdrant releases](https://github.com/qdrant/qdrant/releases) or built from
source). The harness boots a fresh instance per config on an ephemeral port,
so no running Qdrant service is required.
