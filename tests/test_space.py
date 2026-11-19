"""Checks for the Hugging Face Space bundle (space/), which CI cannot deploy."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragmap import api as api_mod

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"


def _name(req: str) -> str:
    return re.split(r"[\s<>=!~\[@;]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")


def test_space_requirements_cover_package_dependencies():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    needed = (
        project["dependencies"] + project["optional-dependencies"]["models"] + project["optional-dependencies"]["nlp"]
    )
    lines = [
        ln for ln in (SPACE / "requirements.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]
    have = {_name(ln) for ln in lines}
    missing = {_name(r) for r in needed} - have
    assert not missing, f"space/requirements.txt is missing {missing}"
    assert "anthropic" not in have, "the public demo must not be able to call a paid API"
    assert "en-core-web-sm" in have


def test_space_readme_front_matter():
    text = (SPACE / "README.md").read_text()
    front = text.split("---")[1]
    assert re.search(r"^sdk: gradio$", front, re.M)
    assert re.search(r"^app_file: app.py$", front, re.M)
    short = re.search(r"^short_description: (.+)$", front, re.M).group(1)
    assert len(short) <= 60  # Hugging Face rejects longer short descriptions
    assert "<!-- PAPERS -->" in text  # filled in by scripts/deploy_space.py


def test_space_paper_candidates_are_arxiv_ids():
    ids = [ln.split("#")[0].strip() for ln in (SPACE / "papers.txt").read_text().splitlines()]
    ids = [i for i in ids if i]
    assert len(ids) >= 3
    assert all(re.fullmatch(r"\d{4}\.\d{4,5}", i) for i in ids)


def test_upload_cap_and_notice(engine, monkeypatch):
    monkeypatch.setattr(api_mod, "MAX_USER_DOCS", 1)
    monkeypatch.setattr(api_mod, "DEMO_NOTICE", "Demo instance")
    client = TestClient(api_mod.create_app(engine=engine))
    assert client.get("/api/config").json()["notice"] == "Demo instance"
    first = client.post("/api/documents", files={"file": ("a.md", b"# A\n\nAlpha beta gamma delta.", "text/plain")})
    assert first.status_code == 200, first.text
    second = client.post("/api/documents", files={"file": ("b.md", b"# B\n\nEpsilon zeta eta theta.", "text/plain")})
    assert second.status_code == 429


OAI_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2307.03172</id>
<authors><author><keyname>Liu</keyname><forenames>Nelson F.</forenames></author>
<author><keyname>Lin</keyname><forenames>Kevin</forenames></author></authors>
<title>Lost in the Middle: How Language
  Models Use Long Contexts</title>
<license>http://creativecommons.org/licenses/by/4.0/</license></arXiv>
</metadata></record></GetRecord></OAI-PMH>"""


def test_paper_license_parsing():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch", ROOT / "scripts" / "space_fetch_papers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rec = mod.parse_record(OAI_SAMPLE)
    assert rec["title"] == "Lost in the Middle: How Language Models Use Long Contexts"
    assert rec["authors"] == ["Nelson F. Liu", "Kevin Lin"]
    assert rec["license"] in mod.CC_BY
    assert "CC BY 4.0" in mod.attribution(rec) and "arXiv:2307.03172" in mod.attribution(rec)
