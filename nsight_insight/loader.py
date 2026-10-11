"""Read an Nsight Systems SQLite export (`nsys export --type sqlite`)."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

COPY_KIND = {1: "H2D", 2: "D2H", 3: "H2A", 4: "A2H", 8: "D2D", 10: "P2P"}
_VERSION_SUFFIX = re.compile(r"_v\d+$")


class TraceError(Exception):
    """Raised when the trace cannot be analysed."""


@dataclass
class Trace:
    kernels: list = field(default_factory=list)   # (start_ns, end_ns, stream, name)
    copies: list = field(default_factory=list)    # (start_ns, end_ns, stream, bytes, kind)
    runtime: list = field(default_factory=list)   # (start_ns, end_ns, api_name)
    nvtx: list = field(default_factory=list)      # (start_ns, end_ns, range_name)


def _rows(con, *queries):
    """Run the first query that works. Column names differ between Nsight versions."""
    for q in queries:
        try:
            return con.execute(q).fetchall()
        except sqlite3.Error:
            continue
    return []


def load(path: str) -> Trace:
    try:
        con = sqlite3.connect(path)
        con.execute("select 1 from sqlite_master limit 1")
    except sqlite3.Error as exc:
        raise TraceError(
            f"{path} is not a readable SQLite file. Export it first with: "
            "nsys export --type sqlite <file>.nsys-rep"
        ) from exc

    names = dict(_rows(con, "select id, value from StringIds"))
    name = lambda i: names.get(i, str(i))

    tr = Trace()
    for s, e, st, n in _rows(
        con,
        "select start, end, streamId, coalesce(demangledName, shortName) from CUPTI_ACTIVITY_KIND_KERNEL",
        "select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL",
    ):
        tr.kernels.append((s, e, st, name(n)))

    for s, e, st, b, k in _rows(
        con, "select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        tr.copies.append((s, e, st, b or 0, COPY_KIND.get(k, "copy")))

    for s, e, n in _rows(con, "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        tr.runtime.append((s, e, _VERSION_SUFFIX.sub("", name(n))))

    for s, e, text, tid in _rows(
        con,
        "select start, end, text, textId from NVTX_EVENTS where end is not null and eventType in (59, 60)",
        "select start, end, text, textId from NVTX_EVENTS where end is not null",
    ):
        label = text or (name(tid) if tid is not None else None)
        if label:
            tr.nvtx.append((s, e, label))

    if not tr.kernels:
        raise TraceError(
            "No CUDA kernels found in this trace. Was a GPU workload profiled with "
            "`nsys profile --trace=cuda,nvtx`?"
        )
    return tr
