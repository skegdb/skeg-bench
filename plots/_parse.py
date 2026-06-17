"""Parse runner.py result files into plot-ready dicts.

Each parser returns (axis_values, {config: {axis_value: metrics_dict}}). Plots
call these; if the results file is missing they fall back to embedded reference
numbers, so charts render both from a fresh run and out of the box.
"""
import os
import re

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _norm(raw):
    """Row label -> canonical config key used by the plots."""
    raw = raw.strip()
    if raw.startswith("skeg-"):
        return raw.split()[0]  # "skeg-tq2 (N vindexes)" / "skeg-tq2 shared+filter" -> "skeg-tq2"
    if "per-collection" in raw:
        return "qdrant-coll"
    if raw.startswith("qdrant") and "shared+filter" in raw:
        return "qdrant-shared"
    return raw.split()[0]


def parse_singletenant(name="singletenant.txt"):
    """-> (Ns, {config: {N: {build,peak,serve,r10,r100,p50,p95}}}). Single-token configs."""
    path = os.path.join(RESULTS, name)
    Ns, data, curN = [], {}, None
    for line in open(path):
        m = re.match(r"=== N=(\d+) ===", line)
        if m:
            curN = int(m.group(1))
            if curN not in Ns:
                Ns.append(curN)
            continue
        p = line.split()
        if curN is None or len(p) != 8 or p[0] == "config" or not p[0][0].isalpha():
            continue
        try:
            b, load, serve, r10, r100, p50, p95 = map(float, p[1:])
        except ValueError:
            continue
        data.setdefault(p[0], {})[curN] = dict(
            build=b, peak=load, serve=serve, r10=r10, r100=r100, p50=p50, p95=p95)
    return Ns, data


def _parse_tenant_file(name, want_leaks):
    path = os.path.join(RESULTS, name)
    tenants, data, curT = [], {}, None
    row = re.compile(r"\s+(.+?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)")
    for line in open(path):
        m = re.match(r"=== N=(\d+) tenants", line)
        if m:
            curT = int(m.group(1))
            if curT not in tenants:
                tenants.append(curT)
            continue
        m = row.match(line)
        if curT is None or not m or m.group(1).strip() == "config":
            continue
        cfg = _norm(m.group(1))
        d = dict(peak=int(m.group(2)), serve=int(m.group(3)),
                 recall=float(m.group(4)), p50=float(m.group(5)), p95=float(m.group(6)))
        if want_leaks:
            tail = m.group(7).split()
            d["leaks"] = int(tail[0]) if tail and tail[0].isdigit() else 0
        data.setdefault(cfg, {})[curT] = d
    return tenants, data


def parse_multitenant(name="multitenant.txt"):
    return _parse_tenant_file(name, want_leaks=False)


def parse_filter(name="filter.txt"):
    return _parse_tenant_file(name, want_leaks=True)
