#!/usr/bin/env bash
# Authoritative OOM demo: run each engine under a real kernel-enforced memory
# cap and read the exit code. PASS = fit; OOM (exit 137) = killed by the kernel.
#
#   docker/run-oom-demo.sh [cap_mb] [tenants] [per_tenant]
#
# Defaults reproduce the MNIST 60k / 256 MB single-corpus scene.
set -euo pipefail
cd "$(dirname "$0")"

CAP="${1:-256}"
NS="${2:-1}"
M="${3:-60000}"
SKEG_REF="${SKEG_REF:-main}"   # must be a build with the TurboQuant RW tier (0.5.0+)

echo "Building image (clones skeg @ ${SKEG_REF}, bundles Qdrant + MNIST)..."
docker build -q --build-arg SKEG_REF="$SKEG_REF" -t skeg-bench-oom . >/dev/null

printf '\n%-10s %-8s %-10s %s\n' "engine" "verdict" "exit" "cap=${CAP}MB ${NS}x${M}"
printf '%s\n' "------------------------------------------------"
for engine in skeg qdrant; do
  set +e
  docker run --rm --memory="${CAP}m" --memory-swap="${CAP}m" \
    -e ENGINE="$engine" -e NS="$NS" -e M="$M" skeg-bench-oom >/tmp/oom-$engine.log 2>&1
  code=$?
  set -e
  if [ "$code" -eq 0 ]; then
    verdict="PASS"
  elif [ "$code" -eq 137 ]; then
    verdict="OOM-KILL"
  else
    verdict="FAIL"
  fi
  printf '%-10s %-8s %-10s %s\n' "$engine" "$verdict" "$code" "(log: /tmp/oom-$engine.log)"
done
