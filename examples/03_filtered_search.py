"""Filtered search: one index, vectors tagged with payload, queries scoped by a
filter. The alternative multi-tenant model (shared index + `tenant=` filter),
and the way to do faceted / metadata-constrained search in general.
"""
import os

import numpy as np
import redis

DIM, N = 8, 200
host, _, port = os.environ.get("SKEG_ADDR", "127.0.0.1:6379").partition(":")
r = redis.Redis(host=host, port=int(port or 6379))
rng = np.random.default_rng(2)

r.execute_command("SKEG.VINDEX.CREATE", "catalog", str(DIM), "tq2", "disk")
# Payload is whitespace-separated key=value tokens. A repeated key is multi-valued.
cats = ["shoes", "shirts", "hats"]
for i in range(N):
    v = rng.standard_normal(DIM).astype("<f4").tobytes()
    payload = f"category={cats[i % 3]} price={(i % 10) * 10}"
    r.execute_command("SKEG.VSET", "catalog", str(i), v, "PAYLOAD", payload)
r.execute_command("SKEG.VINDEX.CONSOLIDATE", "catalog")

q = rng.standard_normal(DIM).astype("<f4").tobytes()

def search(label, *filter_parts):
    args = ["SKEG.VSEARCH", "catalog", "5", "64", q]
    if filter_parts:
        args += ["FILTER", " ".join(filter_parts)]
    res = r.execute_command(*args)
    ids = [int(res[i]) for i in range(0, len(res), 2)]
    print(f"{label:<38} -> ids {ids}")
    return ids

search("unfiltered")
search("category = shoes", "category = shoes")
search("shoes AND price < 50", "category = shoes AND price < 50")
search("hats OR shirts, price BETWEEN 20 AND 60",
       "(category = hats OR category = shirts) AND price BETWEEN 20 AND 60")
print("\nOK — results respect the filter (verify against the payloads above)")
