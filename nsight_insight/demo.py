"""Generate a synthetic Nsight-style SQLite trace so the tool can be tried without a GPU."""
from __future__ import annotations

import os
import random
import sqlite3

SCHEMA = """
create table StringIds(id integer primary key, value text);
create table CUPTI_ACTIVITY_KIND_KERNEL(start int, end int, streamId int, demangledName int, shortName int);
create table CUPTI_ACTIVITY_KIND_MEMCPY(start int, end int, streamId int, bytes int, copyKind int);
create table CUPTI_ACTIVITY_KIND_RUNTIME(start int, end int, nameId int, globalTid int, correlationId int);
create table NVTX_EVENTS(start int, end int, text text, textId int, eventType int);
"""
KERNELS = ["volta_sgemm_128x64_nn", "implicit_convolve_sgemm", "layer_norm_fwd_kernel",
           "softmax_warp_forward", "at::native::elementwise_kernel<add>"]
SYNC_ID = 10


def make_demo(path: str) -> None:
    rnd = random.Random(7)
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("insert into StringIds values (?,?)",
                    list(enumerate(KERNELS)) + [(SYNC_ID, "cudaStreamSynchronize_v3020")])

    kern = lambda s, e, n: con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?)", (s, e, 7, n, n))
    nvtx = lambda s, e, txt: con.execute("insert into NVTX_EVENTS values (?,?,?,null,59)", (s, e, txt))

    t = 0
    for it in range(30):
        it_start = t
        gap = rnd.randint(300_000, 900_000)
        if it % 5 == 4:   # every 5th step the host blocks on a sync (e.g. loss.item())
            nvtx(t, t + gap, "optimizer.step")
            con.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,1,0)", (t, t + gap, SYNC_ID))
        else:             # otherwise the GPU waits for the data loader
            nvtx(t, t + gap, "dataloader.next")
        t += gap

        d = rnd.randint(150_000, 250_000)   # host-to-device copy, serialised with compute
        con.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?)", (t, t + d, 7, 8_000_000, 1))
        t += d

        fwd = t
        for _ in range(12):
            for n, dur in [(0, 220_000), (1, 160_000), (2, 40_000), (3, 30_000)]:
                dur = int(dur * rnd.uniform(.8, 1.2))
                kern(t, t + dur, n)
                t += dur + rnd.randint(2_000, 9_000)
        nvtx(fwd, t, "forward")

        act = t
        for _ in range(40):                 # burst of tiny elementwise kernels
            dur = rnd.randint(3_000, 8_000)
            kern(t, t + dur, 4)
            t += dur + rnd.randint(4_000, 9_000)
        nvtx(act, t, "activations")
        nvtx(it_start, t, f"iteration {it}")
    con.commit()
    con.close()
