from datetime import date

import pytest

from manifest import ManifestError, Source, load_manifest, save_manifest, slug_from_url


def test_roundtrip(tmp_path):
    p = tmp_path / "sources.yaml"
    src = [
        Source(
            id="a", title="A", tier="ois", url="https://x/a/", file="data/ois/a.pdf",
            kind="pdf", manual=True, fetched=date(2026, 10, 31), published=None,
        )
    ]
    save_manifest(p, src)
    assert load_manifest(p) == src


def test_duplicate_id_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "- {id: a, title: A, tier: ois, url: u, file: data/ois/a.pdf, kind: pdf}\n"
        "- {id: a, title: B, tier: ois, url: v, file: data/ois/b.pdf, kind: pdf}\n"
    )
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_bad_tier_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("- {id: a, title: A, tier: blog, url: u, file: data/ois/a.pdf, kind: pdf}\n")
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_slug_from_url():
    assert slug_from_url("https://ois.usc.edu/employment/optional-practical-training-opt/") == "optional-practical-training-opt"
    assert slug_from_url("https://www.ice.gov/doclib/sevis/pdf/i983.pdf") == "i983.pdf"


def test_real_manifest_loads_and_files_match_slugs():
    from settings import DATA_DIR

    sources = load_manifest(DATA_DIR / "sources.yaml")
    assert len(sources) >= 45
    for s in sources:
        if s.tier == "ois":
            assert s.manual and s.file == f"data/ois/{slug_from_url(s.url)}.pdf"
        else:
            assert not s.manual and s.file.startswith("data/federal/")
