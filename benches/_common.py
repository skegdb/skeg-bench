"""Shared bench helpers: ephemeral ports, server readiness, RSS, and .npy load.

Imported by every bench so the harness boilerplate lives in one place.
"""
import os
import socket
import subprocess
import time

import numpy as np


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_tcp(port, t=60):
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def rss(pid=None):
    """Resident set size in MiB for `pid` (default: this process), via `ps`."""
    pid = os.getpid() if pid is None else pid
    o = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(o.stdout.strip() or 0) / 1024


def load_npy(path, limit):
    """First `limit` rows of a float32 .npy as an (limit, cols) array."""
    with open(path, "rb") as f:
        assert f.read(6) == b"\x93NUMPY"
        f.read(2)
        hlen = int.from_bytes(f.read(2), "little")
        hdr = f.read(hlen).decode()
        cols = int(hdr.split("'shape':")[1].split(",")[1].split(")")[0])
        data = np.frombuffer(f.read(limit * cols * 4), dtype="<f4")
    return data.reshape(limit, cols).copy()
