"""Find GPU idle time and explain it."""
from __future__ import annotations

import bisect
import re
from collections import defaultdict

TINY_KERNEL_NS = 10_000
BLOCKING_API = re.compile(r"Synchronize|^cudaMemcpy$|^cudaMalloc|^cudaFree|^cuMemcpy[DH]to[DH]$")
MAX_ATTRIBUTED_GAPS = 100


def merge(intervals):
    """Union of (start, end) intervals as a sorted list of [start, end]."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


class IntervalSet:
    """Disjoint intervals with fast overlap queries."""

    def __init__(self, intervals):
        self.merged = merge(intervals)
        self.starts = [m[0] for m in self.merged]

    def overlap(self, s, e):
        i = max(0, bisect.bisect_right(self.starts, s) - 1)
        total = 0
        while i < len(self.merged) and self.merged[i][0] < e:
            total += max(0, min(e, self.merged[i][1]) - max(s, self.merged[i][0]))
            i += 1
        return total


def attribute_gap(gap, runtime, nvtx):
    """Return (blocking_api_or_None, innermost_nvtx_range_or_None) for one idle gap."""
    s, e = gap
    api, best = None, 0
    for rs, re_, name in runtime:
        if re_ < s or rs > e or not BLOCKING_API.search(name):
            continue
        ov = min(e, re_) - max(s, rs)
        if ov > best:
            api, best = name, ov
    api_covers_gap = api is not None and best >= 0.5 * (e - s)

    mid = (s + e) / 2
    inner, inner_len = None, None
    for ns, ne, text in nvtx:
        if ns <= mid <= ne and (inner_len is None or ne - ns < inner_len):
            inner, inner_len = text, ne - ns
    return (api if api_covers_gap else None), inner


def describe_cause(api, rng):
    if api:
        return f"CPU was blocked in {api}" + (f" (inside NVTX range '{rng}')" if rng else "")
    if rng:
        return f"CPU was busy in NVTX range '{rng}'"
    return "no CPU activity recorded here; likely host-side/Python overhead (add NVTX ranges to pin it down)"


def analyze(trace, min_gap_ns=100_000, max_events=300_000):
    kernels, copies = trace.kernels, trace.copies
    t0 = min([k[0] for k in kernels] + [c[0] for c in copies])
    t1 = max([k[1] for k in kernels] + [c[1] for c in copies])
    span = max(1, t1 - t0)

    busy = merge([(k[0], k[1]) for k in kernels] + [(c[0], c[1]) for c in copies])
    busy_ns = sum(e - s for s, e in busy)
    util = busy_ns / span

    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1)
            if busy[i + 1][0] - busy[i][1] >= min_gap_ns]
    gaps.sort(key=lambda g: g[0] - g[1])  # longest first

    # --- why is the GPU idle? -------------------------------------------------
    causes, idle_by_range = {}, defaultdict(int)
    for g in gaps[:MAX_ATTRIBUTED_GAPS]:
        api, rng = attribute_gap(g, trace.runtime, trace.nvtx)
        causes[g] = describe_cause(api, rng)
        if rng:
            idle_by_range[rng] += g[1] - g[0]
    attributed_idle = sum(g[1] - g[0] for g in gaps[:MAX_ATTRIBUTED_GAPS]) or 1

    # --- copy / compute overlap ------------------------------------------------
    kset = IntervalSet([(k[0], k[1]) for k in kernels])
    copy_ns = sum(c[1] - c[0] for c in copies)
    copy_overlap = sum(kset.overlap(c[0], c[1]) for c in copies)

    # --- kernels ---------------------------------------------------------------
    agg = defaultdict(lambda: [0, 0])
    for s, e, _, n in kernels:
        agg[n][0] += 1
        agg[n][1] += e - s
    kernel_ns = sum(v[1] for v in agg.values()) or 1
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:8]
    tiny = sum(1 for k in kernels if k[1] - k[0] < TINY_KERNEL_NS)
    streams = sorted({k[2] for k in kernels})

    # --- findings --------------------------------------------------------------
    F = []
    F.append(("bad" if util < .6 else "warn" if util < .85 else "good",
              f"GPU is busy {util:.0%} of the trace",
              f"{(span - busy_ns) / 1e6:.1f} ms of {span / 1e6:.1f} ms had nothing running on the GPU "
              f"({len(gaps)} gaps longer than {min_gap_ns / 1e3:.0f} us).",
              "Work through the idle gaps below, largest first.", None))
    for g in gaps[:3]:
        F.append(("warn", f"Idle gap of {(g[1] - g[0]) / 1e3:.0f} us at t = {(g[0] - t0) / 1e6:.2f} ms",
                  causes[g] + ".",
                  "Overlap this CPU work with GPU compute (prefetch, async copies, a worker thread).",
                  [(g[0] - t0) / 1e3 - 50, (g[1] - t0) / 1e3 + 50]))
    if idle_by_range:
        name, ns = max(idle_by_range.items(), key=lambda kv: kv[1])
        share = ns / attributed_idle
        if share > .3:
            F.append(("bad" if share > .6 else "warn",
                      f"{share:.0%} of idle time is inside NVTX range '{name}'",
                      f"{ns / 1e6:.1f} ms of GPU idle time while the CPU was in this range.",
                      "This range is the best place to look for the starvation.", None))
    if tiny / len(kernels) > .25:
        F.append(("warn", f"{tiny / len(kernels):.0%} of kernels run under {TINY_KERNEL_NS // 1000} us",
                  f"{tiny} tiny kernels: launch overhead likely dominates their cost.",
                  "Fuse elementwise ops (e.g. torch.compile) or capture the sequence with CUDA Graphs.", None))
    if copies and copy_ns / span > .02:
        ov = copy_overlap / copy_ns
        F.append(("bad" if ov < .2 else "warn" if ov < .6 else "good",
                  f"Only {ov:.0%} of memcpy time overlaps with compute",
                  f"{copy_ns / 1e6:.1f} ms spent copying ({copy_ns / span:.0%} of the trace).",
                  "Use cudaMemcpyAsync on a dedicated stream with pinned host memory.", None))
    if copies and len(streams) == 1:
        F.append(("warn", "All kernels run on a single stream",
                  "Nothing can overlap with anything else.",
                  "Put copies and independent work on separate streams.", None))
    n0, (c0, d0) = top[0]
    if d0 / kernel_ns > .3:
        F.append(("warn", f"One kernel owns {d0 / kernel_ns:.0%} of kernel time",
                  f"{n0[:80]} ({c0} calls).",
                  "Best optimisation target: profile it in detail with Nsight Compute.", None))

    # --- timeline events -------------------------------------------------------
    sync_calls = [r for r in trace.runtime if BLOCKING_API.search(r[2])]
    lanes = (["NVTX ranges"] if trace.nvtx else []) + (["Blocking CUDA API"] if sync_calls else [])
    base = len(lanes)
    lanes += [f"Stream {s}" for s in streams] + (["Memcpy"] if copies else [])
    lane_of = {s: base + i for i, s in enumerate(streams)}
    us = lambda t: round((t - t0) / 1e3, 3)

    events = [[us(k[0]), us(k[1]), lane_of[k[2]], 0, k[3]] for k in kernels]
    events += [[us(c[0]), us(c[1]), base + len(streams), 1, f"{c[4]} {c[3] / 1e6:.2f} MB"] for c in copies]
    if trace.nvtx:
        events += [[us(n[0]), us(n[1]), 0, 2, n[2]] for n in trace.nvtx]
    if sync_calls:
        events += [[us(r[0]), us(r[1]), 1 if trace.nvtx else 0, 3, r[2]] for r in sync_calls]
    events = [e for e in events if e[1] > 0 and e[0] < us(t1) + 1]
    truncated = len(events) > max_events
    if truncated:
        events.sort(key=lambda e: e[0] - e[1])
        events = events[:max_events]
    events.sort(key=lambda e: e[0])

    return {
        "span_us": round(span / 1e3, 3), "util": round(util, 4),
        "kernels": len(kernels), "copies": len(copies), "n_gaps": len(gaps),
        "idle_ms": round((span - busy_ns) / 1e6, 3),
        "lanes": lanes, "events": events, "truncated": truncated,
        "gaps": [[us(g[0]), us(g[1]), causes.get(g, "")] for g in gaps[:200]],
        "findings": [dict(sev=a, title=b, detail=c, fix=d, zoom=z) for a, b, c, d, z in F],
        "top": [dict(name=n, calls=v[0], ms=round(v[1] / 1e6, 3), pct=round(v[1] / kernel_ns, 4)) for n, v in top],
        "idle_by_range": [dict(name=n, ms=round(v / 1e6, 3), pct=round(v / attributed_idle, 4))
                          for n, v in sorted(idle_by_range.items(), key=lambda kv: -kv[1])[:6]],
    }
