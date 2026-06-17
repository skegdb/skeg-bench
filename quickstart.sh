#!/usr/bin/env bash
# One-command setup + the headline container demo.
#   ./quickstart.sh
# Installs Python deps, fetches a Qdrant binary and MNIST, locates a skeg
# binary, then runs the 256 MB container demo and prints the table.
set -euo pipefail
cd "$(dirname "$0")"

QDRANT_VERSION="${QDRANT_VERSION:-v1.12.4}"
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# ── 1. Python deps ────────────────────────────────────────────────────────
say "Python dependencies"
python3 -m pip install -q -r requirements.txt

# ── 2. skeg binary ────────────────────────────────────────────────────────
say "Locating skeg-resp3 binary"
if [ -z "${SKEG_RESP3_BIN:-}" ]; then
  for cand in ../skeg/target/release/skeg-resp3 ../skeg/target/debug/skeg-resp3; do
    [ -x "$cand" ] && SKEG_RESP3_BIN="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")" && break
  done
fi
if [ -z "${SKEG_RESP3_BIN:-}" ] && [ -d ../skeg ]; then
  echo "building skeg (release)..."; ( cd ../skeg && cargo build -q --release --bin skeg-resp3 )
  SKEG_RESP3_BIN="$(cd ../skeg && pwd)/target/release/skeg-resp3"
fi
if [ -z "${SKEG_RESP3_BIN:-}" ] || [ ! -x "$SKEG_RESP3_BIN" ]; then
  echo "could not find skeg-resp3. Set SKEG_RESP3_BIN=/path/to/skeg-resp3 and re-run." >&2
  echo "(build it from the skeg repo: cargo build --release --bin skeg-resp3)" >&2
  exit 1
fi
export SKEG_RESP3_BIN
echo "skeg: $SKEG_RESP3_BIN"

# ── 3. Qdrant binary ──────────────────────────────────────────────────────
say "Qdrant binary ($QDRANT_VERSION)"
mkdir -p vendor
if [ ! -x vendor/qdrant ]; then
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os-$arch" in
    Darwin-arm64)  tgt="aarch64-apple-darwin" ;;
    Darwin-x86_64) tgt="x86_64-apple-darwin" ;;
    Linux-x86_64)  tgt="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64) tgt="aarch64-unknown-linux-gnu" ;;
    *) echo "unsupported platform $os-$arch; drop a qdrant binary at vendor/qdrant manually" >&2; exit 1 ;;
  esac
  url="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-$tgt.tar.gz"
  echo "downloading $url"
  curl -fsSL "$url" | tar -xz -C vendor qdrant
  chmod +x vendor/qdrant
fi
echo "qdrant: $(pwd)/vendor/qdrant"

# ── 4. MNIST ──────────────────────────────────────────────────────────────
say "MNIST 60k (fetched once into data/)"
if [ ! -f data/mnist_corpus_60k.npy ]; then
  python3 - <<'PY'
import numpy as np, os
os.makedirs("data", exist_ok=True)
from sklearn.datasets import fetch_openml
X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
X = np.asarray(X, dtype="<f4")
np.save("data/mnist_corpus_60k.npy", X[:60000].copy())
np.save("data/mnist_queries_200.npy", X[60000:60200].copy())
print("saved MNIST corpus + queries")
PY
fi

# ── 5. The demo ───────────────────────────────────────────────────────────
say "Container demo: MNIST 60k, 256 MB cap"
SKEG_CORPUS=data/mnist_corpus_60k.npy \
SKEG_QUERIES=data/mnist_queries_200.npy \
DIM=784 RAM_CAP_MB=256 NS=1 M=60000 SKEG_TIER=tq2 \
python3 benches/container_oom.py

say "Done"
echo "Next: python3 runner.py all   (full sweep + charts)   |   docker/run-oom-demo.sh   (kernel-enforced OOM)"
