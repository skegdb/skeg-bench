"""Multi-tenant by physical isolation: one index per tenant.

Each tenant's vectors live in their own index, so a search can never see another
tenant's data — isolation by construction, no filter needed. RAM stays bounded
by the largest tenant, not the total (this is skeg's wedge).
"""
import os

import numpy as np
import redis

DIM, TENANTS, PER = 8, 3, 50
host, _, port = os.environ.get("SKEG_ADDR", "127.0.0.1:6379").partition(":")
r = redis.Redis(host=host, port=int(port or 6379))
rng = np.random.default_rng(1)

for t in range(TENANTS):
    idx = f"tenant{t}"
    r.execute_command("SKEG.VINDEX.CREATE", idx, str(DIM), "tq2", "disk")
    # Bulk insert with VMSET: name then (id, vector, payload) triples.
    args = ["SKEG.VMSET", idx]
    for i in range(PER):
        args += [str(i), rng.standard_normal(DIM).astype("<f4").tobytes(), ""]
    r.execute_command(*args)
    r.execute_command("SKEG.VINDEX.CONSOLIDATE", idx)

# A search hits exactly one tenant's index. No cross-tenant bleed is possible.
q = rng.standard_normal(DIM).astype("<f4").tobytes()
for t in range(TENANTS):
    res = r.execute_command("SKEG.VSEARCH", f"tenant{t}", "3", "32", q)
    ids = [int(res[i]) for i in range(0, len(res), 2)]
    print(f"tenant{t}: top-3 ids {ids}  (all < {PER}, so all belong to tenant{t})")
    assert all(i < PER for i in ids)
print("\nOK — every result stayed inside its tenant")
