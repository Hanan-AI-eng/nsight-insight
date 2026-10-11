from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from . import __version__
from .analysis import analyze
from .demo import make_demo
from .loader import TraceError, load
from .report import render


def _ensure_sqlite(path: str) -> str:
    """Accept .nsys-rep directly by calling `nsys export` when it is installed."""
    if not path.endswith(".nsys-rep"):
        return path
    nsys = shutil.which("nsys")
    if not nsys:
        raise TraceError("Got a .nsys-rep file but `nsys` is not on PATH. Run: "
                         f"nsys export --type sqlite {path}")
    out = path[: -len(".nsys-rep")] + ".sqlite"
    subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                    "--output", out, path], check=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="nsight-insight",
        description="Explain GPU idle time in an NVIDIA Nsight Systems trace.")
    ap.add_argument("trace", nargs="?", help=".sqlite from `nsys export --type sqlite` (or a .nsys-rep)")
    ap.add_argument("-o", "--out", default="report.html", help="HTML report path (default: report.html)")
    ap.add_argument("--json", metavar="FILE", help="also write the findings as JSON")
    ap.add_argument("--min-gap-us", type=float, default=100, help="smallest idle gap to report (default 100)")
    ap.add_argument("--max-events", type=int, default=300_000, help="cap on timeline events in the report")
    ap.add_argument("--demo", action="store_true", help="analyse a synthetic trace (no GPU needed)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    try:
        if a.demo:
            a.trace = os.path.join(tempfile.mkdtemp(), "demo.sqlite")
            make_demo(a.trace)
        if not a.trace:
            ap.error("give a trace file or use --demo")
        res = analyze(load(_ensure_sqlite(a.trace)), int(a.min_gap_us * 1000), a.max_events)
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    render(res, a.out, "demo trace" if a.demo else os.path.basename(a.trace))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "events"}, f, indent=2)
    print(f"{res['kernels']} kernels | GPU busy {res['util']:.0%} | "
          f"{res['n_gaps']} idle gaps | {len(res['findings'])} findings -> {a.out}")
    for f in res["findings"]:
        print(f"  [{f['sev']:<4}] {f['title']}")
    return 0
