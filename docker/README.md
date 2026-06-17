# Authoritative OOM demo (Docker)

`runner.py container` measures peak RSS and compares it to a cap — a good local
proxy. This is the real thing: each engine runs under a kernel-enforced
`docker run --memory` cgroup cap, and the **OOM killer** decides the verdict.

```sh
docker/run-oom-demo.sh                 # MNIST 60k, 256 MB cap, single corpus
docker/run-oom-demo.sh 256 5 12000     # same 60k split across 5 isolated tenants
docker/run-oom-demo.sh 512 5 100000    # mxbai-scale: 5x100k @ 512 MB
```

Output:

```
engine     verdict  exit       cap=256MB 1x60000
------------------------------------------------
skeg       PASS     0
qdrant     OOM-KILL 137
```

- **PASS** (exit 0): the engine ingested and served within the cap.
- **OOM-KILL** (exit 137): the kernel killed it for exceeding the cap. This is
  what "Qdrant gets OOM-killed at 256 MB" actually means — not an estimate.

## Notes

- The image clones and builds skeg. Point it at a build with the TurboQuant RW
  tier (0.5.0 or newer):

  ```sh
  SKEG_REF=v0.5.0 docker/run-oom-demo.sh
  ```

- x86-64 only as written (the bundled Qdrant binary is `x86_64-linux`). Adjust
  the `QDRANT_VERSION`/target in the `Dockerfile` for arm64.
- The cap covers ingest **and** serve in one process tree; the server's memory
  counts against it, which is the whole point.
