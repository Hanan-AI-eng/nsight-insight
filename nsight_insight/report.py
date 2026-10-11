"""Render the analysis as one self-contained HTML file."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")


def render(result: dict, out: str, title: str) -> None:
    data = json.dumps(result).replace("</", "<\\/")
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data).replace("__TITLE__", safe_title)
    Path(out).write_text(page, encoding="utf-8")
