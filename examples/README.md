# Using skeg

Small, runnable examples of the skeg vector API over RESP3 (it speaks the Redis
wire protocol, so any Redis client works — these use `redis-py`).

## Start a server

```sh
# from the skeg repo:
cargo run --release --bin skeg-resp3 -- --data-dir /tmp/skeg-demo --addr 127.0.0.1:6379
```

Then point the examples at it (defaults to `127.0.0.1:6379`):

```sh
pip install redis numpy
python examples/01_quickstart.py
python examples/02_multi_tenant.py
python examples/03_filtered_search.py
```

Override the address with `SKEG_ADDR=host:port`.

## The API in one screen

| command | meaning |
|---------|---------|
| `SKEG.VINDEX.CREATE name dim kind backend` | create an index. `kind` = `int8 \| tq1 \| tq2 \| tq4`, `backend` = `disk \| flat` |
| `SKEG.VSET name id vector [PAYLOAD blob]` | upsert one vector (raw little-endian f32 bytes), optional payload |
| `SKEG.VMSET name (id vector payload)+` | bulk upsert (concurrent fan-out — the fast ingest path) |
| `SKEG.VSEARCH name k l_search vector [WITHPAYLOAD] [FILTER expr]` | top-k search, optionally returning payloads / filtered |
| `SKEG.VINDEX.CONSOLIDATE name` | fold the write delta into the graph (also happens automatically when idle) |

**Tiers:** `tq2` is the recommended sweet spot — ~1.0 recall at a quarter of
int8's resident bytes. `tq1` is the leanest (best-effort recall below 512-dim).

**Filters:** `key=value`, `key IN (a, b)`, ranges `>= > <= < BETWEEN a AND b`,
`key EXISTS`, combined with `AND` / `OR` / `NOT` and parentheses.
