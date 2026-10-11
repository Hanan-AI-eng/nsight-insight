import json
import sqlite3

import pytest

from nsight_insight.analysis import IntervalSet, analyze, merge
from nsight_insight.cli import main
from nsight_insight.demo import make_demo
from nsight_insight.loader import TraceError, load


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("d") / "demo.sqlite")
    make_demo(path)
    return analyze(load(path))


def test_merge_unions_overlaps():
    assert merge([(5, 8), (0, 3), (2, 4)]) == [[0, 4], [5, 8]]


def test_interval_overlap():
    s = IntervalSet([(0, 10), (20, 30)])
    assert s.overlap(5, 25) == 10      # 5 from the first, 5 from the second
    assert s.overlap(10, 20) == 0
    assert s.overlap(-5, 100) == 20


def test_demo_utilisation_and_gaps(demo_result):
    r = demo_result
    assert 0.6 < r["util"] < 0.95
    assert r["n_gaps"] >= 25
    assert r["kernels"] == 30 * (48 + 40)


def test_idle_time_attributed_to_dataloader(demo_result):
    top = demo_result["idle_by_range"][0]
    assert top["name"] == "dataloader.next"
    assert top["pct"] > 0.6


def test_blocking_sync_is_identified(demo_result):
    causes = " ".join(g[2] for g in demo_result["gaps"])
    assert "cudaStreamSynchronize" in causes


def test_findings_cover_expected_problems(demo_result):
    titles = " ".join(f["title"] for f in demo_result["findings"])
    assert "memcpy" in titles
    assert "tiny" in titles.lower() or "under 10 us" in titles
    assert "single stream" in titles


def test_loader_works_with_kernels_only(tmp_path):
    p = str(tmp_path / "k.sqlite")
    con = sqlite3.connect(p)
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL(start int, end int, streamId int, demangledName int, shortName int)")
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (0, 1000, 1, 5, 5)")
    con.commit()
    con.close()
    tr = load(p)
    assert len(tr.kernels) == 1 and tr.copies == [] and tr.nvtx == []
    assert analyze(tr)["util"] == 1.0


def test_loader_rejects_non_sqlite(tmp_path):
    p = tmp_path / "bad.sqlite"
    p.write_text("not a database")
    with pytest.raises(TraceError):
        load(str(p))


def test_loader_rejects_trace_without_kernels(tmp_path):
    p = str(tmp_path / "empty.sqlite")
    sqlite3.connect(p).execute("create table x(a)")
    with pytest.raises(TraceError):
        load(p)


def test_cli_demo_writes_report_and_json(tmp_path):
    out, js = tmp_path / "r.html", tmp_path / "r.json"
    assert main(["--demo", "-o", str(out), "--json", str(js)]) == 0
    html = out.read_text()
    assert "__DATA__" not in html and "nsight" in html
    assert json.loads(js.read_text())["findings"]


def test_cli_missing_trace_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope.sqlite"), "-o", str(tmp_path / "r.html")]) == 1
    assert "error:" in capsys.readouterr().err
