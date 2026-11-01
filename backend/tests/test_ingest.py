from corpus_db import chunks_missing_embedding, create_tables, load_snapshot, make_engine, models_present
from ingest import ingest


def setup(tmp_path, html="<html><body><main><p>" + "OPT text. " * 300 + "</p></main></body></html>"):
    d = tmp_path / "data"
    (d / "federal").mkdir(parents=True)
    (d / "ois").mkdir()
    (d / "federal" / "x.html").write_text(html)
    (d / "sources.yaml").write_text(
        "- {id: uscis-x, title: X, tier: uscis, url: u, file: data/federal/x.html, kind: html, fetched: 2026-10-31}\n"
        "- {id: ois-a, title: A, tier: ois, url: v, file: data/ois/a.pdf, kind: pdf, manual: true}\n"
    )
    e = make_engine("sqlite://")
    create_tables(e)
    return d / "sources.yaml", d, e


def all_chunks(e):
    with e.begin() as c:
        return chunks_missing_embedding(c, "any-model")


def test_ingest_adds_then_skips_then_replaces(tmp_path):
    m, d, e = setup(tmp_path)
    r1 = ingest(m, d, e)
    assert r1.added == ["uscis-x"] and r1.missing == ["ois-a"]
    assert len(all_chunks(e)) >= 3
    r2 = ingest(m, d, e)
    assert r2.skipped == ["uscis-x"] and r2.added == []
    (d / "federal" / "x.html").write_text("<html><body><main><p>changed</p></main></body></html>")
    r3 = ingest(m, d, e)
    assert r3.replaced == ["uscis-x"] and [t for _, t in all_chunks(e)] == ["changed"]


def test_ingest_reports_missing_files(tmp_path):
    m, d, e = setup(tmp_path)
    (d / "federal" / "x.html").unlink()
    r = ingest(m, d, e)
    assert sorted(r.missing) == ["ois-a", "uscis-x"] and load_snapshot(e, "any").document_count == 0


def test_ingest_document_with_no_text_is_recorded_with_zero_chunks(tmp_path):
    m, d, e = setup(tmp_path, html="<html><body><main></main></body></html>")
    r = ingest(m, d, e)
    assert r.added == ["uscis-x"] and r.empty == ["uscis-x"]
    assert load_snapshot(e, "any").document_count == 1 and all_chunks(e) == []


def test_prune_removes_documents_missing_from_manifest(tmp_path):
    m, d, e = setup(tmp_path)
    ingest(m, d, e)
    m.write_text("- {id: ois-a, title: A, tier: ois, url: v, file: data/ois/a.pdf, kind: pdf, manual: true}\n")
    r = ingest(m, d, e, prune=True)
    assert r.pruned == ["uscis-x"] and load_snapshot(e, "any").document_count == 0



def test_ingest_leaves_vectors_to_embed_corpus(tmp_path):
    m, d, e = setup(tmp_path)
    ingest(m, d, e)
    assert models_present(e) == {} and len(all_chunks(e)) >= 3
