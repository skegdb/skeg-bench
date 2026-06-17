#!/usr/bin/env python
"""Single entry point for the skeg-bench suite.

  python runner.py <phase>

phase: singletenant | multitenant | filter | container | plots | all

Each phase shells out to the matching script with the env it needs, teeing
stdout to results/<phase>.txt. Corpus/query/binary paths come from env
(SKEG_RESP3_BIN, SKEG_CORPUS, SKEG_QUERIES) — see README.md and data/README.md.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

# phase -> (script, extra env). Scripts read SKEG_CORPUS/SKEG_QUERIES/etc. themselves.
PHASES = {
    "singletenant": ("benches/singletenant.py", {}),
    "multitenant": ("benches/multitenant.py", {}),
    "filter": ("benches/multitenant.py", {"MODE": "filter"}),
    "container": ("benches/container_oom.py", {}),
}
PLOTS = ["plot_singletenant.py", "plot_multitenant.py", "plot_shared_filter.py", "plot_latency.py"]


def need(*vars):
    missing = [v for v in vars if not os.environ.get(v)]
    if missing:
        sys.exit(f"set required env: {', '.join(missing)} (see README.md)")


def run_phase(phase):
    script, extra = PHASES[phase]
    need("SKEG_RESP3_BIN", "SKEG_CORPUS", "SKEG_QUERIES")
    os.makedirs(RESULTS, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **extra}
    out = os.path.join(RESULTS, f"{phase}.txt")
    print(f"== {phase} -> {out}")
    with open(out, "w") as f:
        p = subprocess.Popen([sys.executable, os.path.join(ROOT, script)],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    return p.returncode


def run_plots():
    rc = 0
    for s in PLOTS:
        print(f"== plot {s}")
        rc |= subprocess.run([sys.executable, os.path.join(ROOT, "plots", s)], cwd=ROOT).returncode
    return rc


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "plots":
        sys.exit(run_plots())
    if phase == "all":
        rc = 0
        for ph in ("singletenant", "multitenant", "filter", "container"):
            rc |= run_phase(ph)
        rc |= run_plots()
        sys.exit(rc)
    if phase not in PHASES:
        sys.exit(f"unknown phase {phase!r}; want one of: {', '.join(PHASES)} | plots | all")
    sys.exit(run_phase(phase))


if __name__ == "__main__":
    main()
