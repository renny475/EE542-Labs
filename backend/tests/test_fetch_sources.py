from datetime import date

import httpx

from fetch_sources import fetch_sources
from manifest import load_manifest

MANIFEST = """\
- {id: ois-a, title: A, tier: ois, url: https://ois.usc.edu/a/, file: data/ois/a.pdf, kind: pdf, manual: true}
- {id: uscis-b, title: B, tier: uscis, url: https://www.uscis.gov/b, file: data/federal/uscis-b.html, kind: html}
- {id: sevp-c, title: C, tier: sevp, url: https://www.ice.gov/c.pdf, file: data/federal/sevp-c.pdf, kind: pdf}
"""


def make(tmp_path, handler):
    (tmp_path / "data" / "ois").mkdir(parents=True)
    (tmp_path / "data" / "federal").mkdir()
    m = tmp_path / "data" / "sources.yaml"
    m.write_text(MANIFEST)
    seen = []

    def h(request):
        seen.append(request)
        return handler(request)

    return m, tmp_path / "data", httpx.Client(transport=httpx.MockTransport(h)), seen


def test_downloads_missing_federal_files_and_reports_missing_manual(tmp_path):
    def handler(req):
        return httpx.Response(200, content=b"<html>B</html>" if req.url.path == "/b" else b"%PDF-1.4 c")

    m, data_dir, client, seen = make(tmp_path, handler)
    sleeps = []
    rep = fetch_sources(m, data_dir, client, today=date(2026, 10, 31), sleep=sleeps.append)
    assert rep.downloaded == ["uscis-b", "sevp-c"] and rep.missing == ["ois-a"] and rep.failed == []
    assert (data_dir / "federal" / "uscis-b.html").read_bytes() == b"<html>B</html>"
    assert all(r.headers["User-Agent"].startswith("EE542-OIS-Assistant/") for r in seen)
    assert sleeps == [2.0, 2.0]
    assert {s.id: s.fetched for s in load_manifest(m)}["uscis-b"] == date(2026, 10, 31)


def test_present_files_are_skipped_unless_refresh(tmp_path):
    m, data_dir, client, seen = make(tmp_path, lambda r: httpx.Response(200, content=b"x"))
    (data_dir / "federal" / "uscis-b.html").write_bytes(b"old")
    rep = fetch_sources(m, data_dir, client, sleep=lambda s: None)
    assert "uscis-b" in rep.skipped and len(seen) == 1
    rep = fetch_sources(m, data_dir, client, refresh=True, sleep=lambda s: None)
    assert "uscis-b" in rep.downloaded


def test_403_is_failed_with_no_retry(tmp_path):
    m, data_dir, client, seen = make(tmp_path, lambda r: httpx.Response(403))
    rep = fetch_sources(m, data_dir, client, sleep=lambda s: None)
    assert [f[0] for f in rep.failed] == ["uscis-b", "sevp-c"] and len(seen) == 2


def test_check_only_fills_fetched_for_present_manual_file(tmp_path):
    m, data_dir, client, seen = make(tmp_path, lambda r: httpx.Response(200, content=b"x"))
    (data_dir / "ois" / "a.pdf").write_bytes(b"%PDF")
    rep = fetch_sources(m, data_dir, client, check_only=True, sleep=lambda s: None)
    assert seen == [] and "ois-a" not in rep.missing
    assert {s.id: s.fetched for s in load_manifest(m)}["ois-a"] == date.today()
