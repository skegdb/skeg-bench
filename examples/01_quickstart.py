#!/usr/bin/env python
"""Create an index, insert vectors, search. The skeg hello-world."""
import os

import numpy as np
import redis

DIM = 8
host, _, port = os.environ.get("SKEG_ADDR", "127.0.0.1:6379").partition(":")
r = redis.Redis(host=host, port=int(port or 6379))

# A disk index with the tq2 tier (the recommended sweet spot).
r.execute_command("SKEG.VINDEX.CREATE", "demo", str(DIM), "tq2", "disk")

# Insert 100 random unit-ish vectors. Vectors go on the wire as raw f32 bytes.
rng = np.random.default_rng(0)
data = rng.standard_normal((100, DIM)).astype("<f4")
for i, v in enumerate(data):
    r.execute_command("SKEG.VSET", "demo", str(i), v.tobytes())
r.execute_command("SKEG.VINDEX.CONSOLIDATE", "demo")  # fold the delta into the graph

# Search: k=5 nearest to vector 0 (it should be its own top hit).
query = data[0].tobytes()
res = r.execute_command("SKEG.VSEARCH", "demo", "5", "64", query)
# Results come back flat: [id, score, id, score, ...]
hits = [(int(res[i]), float(res[i + 1])) for i in range(0, len(res), 2)]
print("top-5 nearest to vector 0:")
for vid, score in hits:
    print(f"  id={vid:<3} score={score:.4f}")
assert hits[0][0] == 0, "a vector is its own nearest neighbor"
print("\nOK")
