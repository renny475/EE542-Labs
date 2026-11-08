import json
import re

import httpx
import pytest

import knowledge
import manage_knowledge
from config import Settings
from fake_llm import FakeLLM
from knowledge import KnowledgeError, KnowledgeStore, official_url

STONES = (
    "Kidney stones diet. Drink plenty of fluids, mostly water, to help prevent kidney stones. "
    "Limit sodium and avoid foods high in oxalate such as spinach, rhubarb, and nuts."
)
INHALER = (
    "COPD inhaler use. Shake the albuterol inhaler before each puff and rinse your mouth "
    "after using the inhaler. Pursed lip breathing can help during activity."
)


def record(source_id, text, url=None, topic="Kidney stones", title=None):
    return {
        "id": source_id,
        "title": title or source_id.title(),
        "url": url or f"https://medlineplus.gov/{source_id}.html",
        "topic": topic,
        "collected_at": "2026-11-07T00:00:00+00:00",
        "sha256": "0" * 64,
        "text": text,
    }


def test_official_url_allowlist():
    assert official_url("https://medlineplus.gov/kidneystones.html")
    assert official_url("https://www.niddk.nih.gov/health-information/urologic-diseases")
    assert official_url("https://www.cdc.gov/copd/index.html")
    assert official_url("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc")
    assert official_url("https://www.kidney.org/kidney-topics/kidney-stones")
    assert official_url("https://www.upmc.com/healthbeat/urology/x")
    assert official_url("https://kidneystones.uchicago.edu/2015/11/16/how-to-eat-a-low-oxalate-diet/")
    assert not official_url("https://docs.google.com/spreadsheets/d/x")
    assert not official_url("http://medlineplus.gov/kidneystones.html")  # HTTPS only
    assert not official_url("https://medlineplus.gov.evil.example/x")
    assert not official_url("https://evil.example/medlineplus.gov")
    assert not official_url("https://user:pw@medlineplus.gov/x")
    assert not official_url("https://www.webmd.com/kidney-stones")
    assert not official_url("not a url")


async def test_add_and_retrieve_ranks_relevant_chunk_first(tmp_path):
    store = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM())
    assert await store.add(record("stones", STONES)) == 1
    await store.add(record("inhaler", INHALER, topic="COPD"))
    results = await store.retrieve("which foods are high in oxalate for kidney stones", limit=5)
    assert results[0]["source_id"] == "stones"
    assert set(results[0]) == {"source_id", "title", "url", "topic", "content", "score"}
    assert results[0]["url"] == "https://medlineplus.gov/stones.html"
    assert results[0]["topic"] == "Kidney stones"
    assert "oxalate" in results[0]["content"]
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)
    assert len(await store.retrieve("inhaler", limit=1)) == 1


async def test_retrieve_respects_min_similarity(tmp_path):
    store = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM(), min_similarity=0.25)
    await store.add(record("stones", STONES))
    assert await store.retrieve("zebra xylophone quantum") == []
    strict = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM(), min_similarity=0.99)
    assert await strict.retrieve("oxalate kidney stones") == []
    loose = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM(), min_similarity=0.0)
    assert len(await loose.retrieve("oxalate kidney stones")) == 1


async def test_status_not_ready_when_model_changed(tmp_path):
    path = tmp_path / "index.sqlite3"
    empty = KnowledgeStore(path, FakeLLM()).status()
    assert empty["ready"] is False
    assert "not been indexed" in empty["message"]

    store = KnowledgeStore(path, FakeLLM(embedding_model="model-a"))
    await store.add(record("stones", STONES))
    ready = store.status()
    assert ready["ready"] is True
    assert ready["embedding_model"] == "model-a"
    assert ready["chunks"] == 1
    assert ready["topics"] == ["Kidney stones"]

    changed = KnowledgeStore(path, FakeLLM(embedding_model="model-b"))
    status = changed.status()
    assert status["ready"] is False
    assert "Rebuild the index" in status["message"]
    with pytest.raises(KnowledgeError, match="Rebuild the index"):
        await changed.retrieve("oxalate")


async def test_add_rejects_unofficial_url_and_replaces_existing_source(tmp_path):
    store = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM())
    with pytest.raises(KnowledgeError, match="approved"):
        await store.add(record("bad", STONES, url="https://www.webmd.com/stones"))
    long_text = "\n".join(f"Line {n} about drinking water and kidney stones prevention." for n in range(40))
    chunks = await store.add(record("stones", long_text))
    assert chunks > 1
    assert all(len(c["content"]) <= 500 for c in await KnowledgeStore(
        tmp_path / "index.sqlite3", FakeLLM(), min_similarity=0.0).retrieve("water", limit=100))
    await store.add(record("stones", STONES))
    assert store.status()["chunks"] == 1


async def test_failed_embedding_leaves_existing_data(tmp_path):
    store = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM())
    await store.add(record("stones", STONES))

    class Broken(FakeLLM):
        async def embed(self, texts):
            raise RuntimeError("embedding service down")

    with pytest.raises(RuntimeError):
        await KnowledgeStore(tmp_path / "index.sqlite3", Broken()).add(record("stones", INHALER))
    assert store.status()["chunks"] == 1
    assert "oxalate" in (await store.retrieve("oxalate kidney stones"))[0]["content"]


def test_clean_html_keeps_main_text_and_checks_health_relevance():
    body = "<p>" + "Drink water to prevent kidney stones and lower sodium. " * 10 + "</p>"
    html = (
        "<html><body><nav>Menu Home</nav><main><header>Site header</header>"
        f"<h1>Kidney stones \u2014 diet</h1>{body}<script>x=1</script>"
        "<footer>Footer links</footer></main></body></html>"
    )
    text = knowledge.clean_html(html)
    assert text.startswith("Kidney stones - diet")
    assert "\u2014" not in text
    assert "Menu" not in text and "Footer" not in text and "x=1" not in text and "Site header" not in text
    with pytest.raises(ValueError, match="unrelated"):
        knowledge.clean_html("<main>" + "<p>Football scores and weather today.</p>" * 20 + "</main>")
    with pytest.raises(ValueError, match="main/article"):
        knowledge.clean_html("<div>no main region</div>")


def test_dailymed_sections_extracts_named_sections():
    xml = """<?xml version="1.0"?>
    <document xmlns="urn:hl7-org:v3"><component><structuredBody>
      <component><section><code code="34067-9"/><title>1 INDICATIONS</title>
        <text><paragraph>Treats infections.</paragraph></text></section></component>
      <component><section><code code="34076-0"/><title>17 PATIENT COUNSELING INFORMATION</title>
        <text><paragraph>If you miss a dose, take it as soon as you remember \u2014 do not double.</paragraph></text>
        <component><section><code code="0"/><title>17.1 Food</title>
          <text><paragraph>Take with food.</paragraph></text></section></component>
      </section></component>
    </structuredBody></component></document>"""
    text = knowledge.dailymed_sections(xml, ["34076-0"])
    assert text.startswith("17 PATIENT COUNSELING INFORMATION")
    assert "If you miss a dose, take it as soon as you remember - do not double." in text
    assert "17.1 Food" in text and "Take with food." in text
    assert "Treats infections" not in text
    with pytest.raises(ValueError, match="sections"):
        knowledge.dailymed_sections(xml, ["42231-1"])


def write_manifest(directory, items):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(items))


async def test_collect_writes_unreviewed_snapshot_and_skips_curated(tmp_path):
    page = ("<main><h1>Pneumonia</h1>" + "<p>Pneumonia is an infection of the lungs. Rest and fluids help.</p>" * 6
            + "<div class='promo'>Donate today</div></main>")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        assert "AfterCare" in request.headers["user-agent"]
        return httpx.Response(200, text=page)

    write_manifest(tmp_path, [
        {"id": "pneumonia", "title": "Pneumonia", "topic": "Pneumonia", "url": "https://medlineplus.gov/pneumonia.html",
         "exclude": [".promo"]},
        {"id": "note", "title": "Note", "topic": "Kidney stones", "url": "https://www.kidney.org/x", "curated": True},
    ])
    report = await knowledge.collect_sources(tmp_path, transport=httpx.MockTransport(handler))
    assert report[0] == {"id": "pneumonia", "ok": True, "characters": report[0]["characters"]}
    assert report[1]["id"] == "note" and report[1]["skipped"]
    assert not (tmp_path / "sources" / "note.json").exists()
    assert seen == ["https://medlineplus.gov/pneumonia.html"]
    saved = json.loads((tmp_path / "sources" / "pneumonia.json").read_text())
    assert saved["reviewed"] is False
    assert saved["text"].startswith("Pneumonia")
    assert "Donate" not in saved["text"]
    assert len(saved["sha256"]) == 64 and saved["collected_at"]


async def test_collect_rejects_redirect_off_allowlist_and_keeps_old_snapshot(tmp_path):
    def handler(request):
        return httpx.Response(302, headers={"location": "https://evil.example/pneumonia"})

    write_manifest(tmp_path, [
        {"id": "pneumonia", "title": "Pneumonia", "topic": "Pneumonia", "url": "https://medlineplus.gov/pneumonia.html"},
    ])
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "pneumonia.json").write_text(json.dumps({"id": "pneumonia", "text": "old", "reviewed": True}))
    report = await knowledge.collect_sources(tmp_path, transport=httpx.MockTransport(handler))
    assert report[0]["ok"] is False and "approved" in report[0]["error"]
    saved = json.loads((tmp_path / "sources" / "pneumonia.json").read_text())
    assert saved["text"] == "old"
    assert saved["reviewed"] is False
    assert saved["refresh_error"]


async def test_index_sources_indexes_reviewed_and_curated_only(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    reviewed = record("stones", STONES) | {"reviewed": True}
    curated = record("oxalate-note", "High oxalate foods have more than 10 mg per serving.",
                     url="https://www.kidney.org/oxalate") | {"reviewed": True, "curated": True}
    pending = record("inhaler", INHALER, topic="COPD") | {"reviewed": False}
    for item in (reviewed, curated, pending):
        (sources / f"{item['id']}.json").write_text(json.dumps(item))
    path = tmp_path / "index.sqlite3"
    result = await knowledge.index_sources(KnowledgeStore(path, FakeLLM()), tmp_path)
    assert result["indexed"] == {"oxalate-note": 1, "stones": 1}
    assert result["skipped"] == ["inhaler"]
    status = KnowledgeStore(path, FakeLLM()).status()
    assert status["ready"] is True
    assert {d["id"] for d in status["documents"]} == {"stones", "oxalate-note"}
    assert not path.with_suffix(".staging.sqlite3").exists()


async def test_index_sources_failure_keeps_previous_index(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    path = tmp_path / "index.sqlite3"
    store = KnowledgeStore(path, FakeLLM())
    await store.add(record("stones", STONES))
    (sources / "bad.json").write_text(json.dumps(
        record("bad", INHALER, url="https://www.webmd.com/x") | {"reviewed": True}))
    with pytest.raises(KnowledgeError):
        await knowledge.index_sources(store, tmp_path)
    assert {d["id"] for d in store.status()["documents"]} == {"stones"}
    assert not path.with_suffix(".staging.sqlite3").exists()


async def test_index_sources_requires_reviewed_sources(tmp_path):
    (tmp_path / "sources").mkdir()
    with pytest.raises(KnowledgeError, match="reviewed"):
        await knowledge.index_sources(KnowledgeStore(tmp_path / "i.sqlite3", FakeLLM()), tmp_path)


def test_cli_index_without_api_key_fails_clearly(tmp_path, capsys):
    settings = Settings(rag_db_path=str(tmp_path / "index.sqlite3"), llm_api_key="", embedding_api_key="")
    assert manage_knowledge.run(["index"], settings) == 2
    assert "EMBEDDING_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "index.sqlite3").exists()


def test_cli_index_keyless_self_hosted_embeddings_allowed(tmp_path, capsys, monkeypatch):
    calls = []

    async def fake_index(store, data):
        calls.append(store.path)
        return {"indexed": {}, "skipped": []}

    monkeypatch.setattr(manage_knowledge, "index_sources", fake_index)
    settings = Settings(
        rag_db_path=str(tmp_path / "index.sqlite3"), llm_api_base="http://gpu-box:8000/v1",
        llm_model="served-model", llm_api_key="", embedding_api_key="",
    )
    assert manage_knowledge.run(["index"], settings) == 0, capsys.readouterr().err
    assert calls == [tmp_path / "index.sqlite3"]


def test_cli_status_reports_json(tmp_path, capsys):
    settings = Settings(rag_db_path=str(tmp_path / "index.sqlite3"), embedding_model="m1")
    assert manage_knowledge.run(["status"], settings) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ready"] is False and status["embedding_model"] == "m1"


def test_real_library_snapshots_are_valid():
    manifest = json.loads((knowledge.DATA / "manifest.json").read_text())
    ids = [item["id"] for item in manifest]
    assert len(ids) == len(set(ids))
    topics = set()
    for item in manifest:
        assert official_url(item["url"]), item["id"]
        snapshot = json.loads((knowledge.DATA / "sources" / f"{item['id']}.json").read_text())
        assert snapshot["id"] == item["id"]
        assert snapshot["reviewed"] is True, item["id"]
        assert snapshot["url"] == item["url"] and snapshot["topic"] == item["topic"]
        assert snapshot.get("curated", False) == item.get("curated", False)
        assert len(snapshot["text"]) >= 200 and "\u2014" not in snapshot["text"], item["id"]
        topics.add(item["topic"])
    assert set(knowledge.SCENARIO_TOPICS) <= topics
    assert {p.stem for p in (knowledge.DATA / "sources").glob("*.json")} == set(ids)


def test_real_curated_summaries_match_their_source_snapshot():
    """Every curated summary names a snapshot of the same page, and every number in
    the summary appears in that snapshot. The UChicago summary's key quotes are verbatim."""
    sources = knowledge.DATA / "sources"
    curated = [json.loads(p.read_text()) for p in sources.glob("*.json")]
    curated = [c for c in curated if c.get("curated")]
    assert curated
    for summary in curated:
        source = json.loads((sources / f"{summary['summarized_source_id']}.json").read_text())
        assert source["url"] == summary["url"] and not source.get("curated"), summary["id"]
        for number in re.findall(r"\d+(?:\.\d+)?", summary["text"]):
            assert number in source["text"], (summary["id"], number)
    uchicago = json.loads((sources / "curated-uchicago-oxalate-table-use.json").read_text())
    source = json.loads((sources / "uchicago-low-oxalate-diet.json").read_text())
    for quote in ("Less than 100 mg of diet oxalate is good; less than 50 mg is ideal.",
                  "Note that quantity is critical.",
                  "The graph shows mg of oxalate in a common portion."):
        assert quote in uchicago["text"] and quote in source["text"]
    assert not (sources / "curated-high-oxalate-cutoff.json").exists()


def test_clean_html_joins_inline_links_and_honors_selector():
    html = (
        "<html><body><article><ul><li><a href='#'>Start Here</a></li></ul>"
        "<div id='topic-summary'><h3>What is pneumonia?</h3>"
        "<p>Pneumonia is an <a href='/infections.html'>infection</a> in one or both of your lungs."
        "<br>It can be mild or severe, depending on your overall health.</p>"
        + "<p>Rest, drink fluids, and take your medicine as your doctor prescribed.</p>" * 4
        + "<ul><li>Fever</li><li>Cough</li></ul></div></article></body></html>"
    )
    text = knowledge.clean_html(html, selector="#topic-summary")
    lines = text.splitlines()
    assert lines[0] == "What is pneumonia?"
    assert lines[1] == "Pneumonia is an infection in one or both of your lungs."
    assert lines[2] == "It can be mild or severe, depending on your overall health."
    assert lines[-2:] == ["Fever", "Cough"]
    assert "Start Here" not in text
    trimmed = knowledge.clean_html(html, selector="#topic-summary", exclude=["ul"])
    assert "Fever" not in trimmed and "Pneumonia is an infection" in trimmed
    with pytest.raises(ValueError, match="selector"):
        knowledge.clean_html(html, selector="#missing")


def test_clean_html_excluded_inline_citation_keeps_words_apart():
    html = ("<main><p>Nearly 16 million adults have COPD,<span class='cite'><a>1</a></span>and many more "
            "do not know it. Talk with your doctor about symptoms.</p>" + "<p>Quit smoking to protect your lungs.</p>" * 6 + "</main>")
    assert "COPD, and many more" in knowledge.clean_html(html, exclude=[".cite"])


def test_source_line_breaks_inside_a_paragraph_are_not_kept():
    html = ("<main><p>Drink plenty of\n      water every day to help\nprevent kidney stones.</p>"
            + "<p>Limit sodium in your diet.</p>" * 10 + "</main>")
    assert knowledge.clean_html(html).splitlines()[0] == "Drink plenty of water every day to help prevent kidney stones."
    xml = """<document xmlns="urn:hl7-org:v3"><section><code code="34068-7"/><title>DOSAGE</title>
      <text><paragraph>Monitor closely
         <content>(see</content>
         <content>PRECAUTIONS)</content>.</paragraph></text></section></document>"""
    assert knowledge.dailymed_sections(xml, ["34068-7"]).splitlines() == ["DOSAGE", "Monitor closely (see PRECAUTIONS)."]


def test_dailymed_sections_drops_bullet_and_figure_only_lines():
    xml = """<document xmlns="urn:hl7-org:v3"><section><code code="59845-8"/><title>Instructions for Use</title>
      <text><list><item><caption>•</caption><paragraph>Shake the inhaler well.</paragraph></item></list>
      <paragraph>Figure A</paragraph><paragraph>Image</paragraph><paragraph>See Figure A for the parts.</paragraph></text>
    </section></document>"""
    assert knowledge.dailymed_sections(xml, ["59845-8"]).splitlines() == [
        "Instructions for Use", "Shake the inhaler well.", "See Figure A for the parts."]
