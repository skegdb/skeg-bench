# Authoritative tight-container OOM demo image.
# Builds skeg from source, bundles Qdrant + MNIST, and drives ONE engine
# (chosen at run time via ENGINE=skeg|qdrant) so a memory-capped
# `docker run --memory=...` lets the kernel decide PASS vs OOM.

# ── build skeg ────────────────────────────────────────────────────────────
FROM rust:1-bookworm AS skeg-build
ARG SKEG_REPO=https://github.com/skegdb/skeg
ARG SKEG_REF=main
RUN git clone --depth 1 --branch "$SKEG_REF" "$SKEG_REPO" /src \
 && cd /src && cargo build --release --bin skeg-resp3
# (SKEG_REF must point at a build with the TurboQuant RW tier — 0.5.0+.)

# ── runtime ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
ARG QDRANT_VERSION=v1.12.4
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Qdrant binary
RUN curl -fsSL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C /usr/local/bin qdrant && chmod +x /usr/local/bin/qdrant

# skeg binary
COPY --from=skeg-build /src/target/release/skeg-resp3 /usr/local/bin/skeg-resp3
ENV SKEG_RESP3_BIN=/usr/local/bin/skeg-resp3

# python deps + MNIST baked in (so runtime is offline)
RUN pip install --no-cache-dir numpy redis qdrant-client scikit-learn
RUN python - <<'PY'
import numpy as np, os
os.makedirs("/data", exist_ok=True)
from sklearn.datasets import fetch_openml
X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
X = np.asarray(X, dtype="<f4")
np.save("/data/mnist_corpus_60k.npy", X[:60000].copy())
np.save("/data/mnist_queries_200.npy", X[60000:60200].copy())
PY

COPY oom_single.py /app/oom_single.py
WORKDIR /app
ENV ENGINE=skeg DIM=784 NS=1 M=60000 SKEG_TIER=tq2
ENTRYPOINT ["python", "oom_single.py"]
