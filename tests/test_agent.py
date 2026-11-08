import json

import pytest

import agent
from agent import FALLBACK, Evidence, render_food, run_agent, validate
from fake_llm import FakeLLM
from foods import FoodTable
from knowledge import KnowledgeStore
from models import DischargeDocument, ExtractedItem

DOC_TEXT = (
    "Discharge date: 2026-03-02\n\n"
    "Limit high-oxalate foods to help prevent kidney stones.\n\n"
    "Drink at least 2 liters of water every day.\n\n"
    "Go to the emergency room if you have severe pain or cannot urinate."
)
OXALATE_PASSAGE = (
    "Low oxalate diet. Foods with more than 10 mg of oxalate per serving count as high "
    "oxalate foods. Very high oxalate foods include spinach, rhubarb and most nuts."
)
LIBRARY_URL = "https://www.kidney.org/kidney-topics/low-oxalate-diet"
UCHICAGO_ARTICLE = "https://kidneystones.uchicago.edu/2015/11/16/how-to-eat-a-low-oxalate-diet/"
FDC_URL = "https://fdc.nal.usda.gov/food-details/170567/nutrients"

FOODS = {
    "sources": {
        "fdc-170567": {"title": "USDA FoodData Central: Nuts, almonds (FDC ID 170567)", "url": FDC_URL},
        "fdc-169145": {
            "title": "USDA FoodData Central: Bananas, raw (FDC ID 169145)",
            "url": "https://fdc.nal.usda.gov/food-details/169145/nutrients",
        },
        "uchicago-oxalate-table": {
            "title": "University of Chicago Kidney Stone Program: Updated table of oxalate foods",
            "url": "https://docs.google.com/spreadsheets/d/example",
            "linked_from": UCHICAGO_ARTICLE,
        },
    },
    "foods": [
        {
            "id": "almonds",
            "name": "Almonds",
            "aliases": ["almond", "raw almonds"],
            "serving": {"description": "1 oz (about 23 almonds)", "grams": 28.35},
            "nutrients": {
                "sodium_mg": {"value": 0, "unit": "mg", "source_id": "fdc-170567"},
                "calcium_mg": {"value": 76, "unit": "mg", "source_id": "fdc-170567"},
                "protein_g": {"value": 6.0, "unit": "g", "source_id": "fdc-170567"},
                "phosphorus_mg": {"value": None, "unit": "mg", "source_id": None},
                "oxalate_mg": {
                    "value": 122, "unit": "mg", "source_id": "uchicago-oxalate-table",
                    "source_category": "Very High", "source_food": "Almonds",
                    "source_serving": "1 oz or 22 kernels",
                },
            },
        },
        {
            "id": "banana",
            "name": "Banana",
            "aliases": ["bananas"],
            "serving": {"description": "1 medium", "grams": 118},
            "nutrients": {
                "potassium_mg": {"value": 422, "unit": "mg", "source_id": "fdc-169145"},
                "oxalate_mg": {"value": None, "unit": "mg", "source_id": None},
            },
        },
    ],
}


@pytest.fixture
def foods(tmp_path):
    path = tmp_path / "foods.json"
    path.write_text(json.dumps(FOODS))
    return FoodTable.load(path)


@pytest.fixture
async def store(tmp_path):
    store = KnowledgeStore(tmp_path / "index.sqlite3", FakeLLM(), min_similarity=0.0)
    await store.add({
        "id": "kidney-oxalate", "title": "Low oxalate diet", "url": LIBRARY_URL,
        "topic": "Kidney stones", "collected_at": "2026-11-07T00:00:00+00:00",
        "sha256": "0" * 64, "text": OXALATE_PASSAGE,
    })
    return store


@pytest.fixture
def document():
    return DischargeDocument(id=1, user_id=1, status="confirmed", text=DOC_TEXT)


@pytest.fixture
def items():
    return [
        ExtractedItem(
            id=1, document_id=1, category="diet", status="active",
            text="Limit high-oxalate foods.",
            source_quote="Limit high-oxalate foods to help prevent kidney stones.", details={},
        ),
        ExtractedItem(
            id=2, document_id=1, category="diet", status="rejected",
            text="Eat more spinach.", source_quote="Drink at least 2 liters of water every day.",
            details={},
        ),
    ]


def tool_call(name, args, call_id="call-1"):
    return {
        "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def final(answer, citations, kind="answer", care_team_question=None):
    return {"content": json.dumps({
        "kind": kind, "answer": answer, "citations": citations,
        "care_team_question": care_team_question,
    })}


async def ask(llm, store, foods, document, items, question="Can I eat almonds?", history=(),
              emergency_terms=(), **kw):
    return await run_agent(llm, store, foods, document, items, list(history), question,
                           list(emergency_terms), **kw)


def tool_results(llm, call_index):
    """JSON tool results the model saw at the start of chat call `call_index`."""
    messages = llm.calls[call_index][0]
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def assert_fallback(result, question="Can I eat almonds?"):
    assert result.kind == "not_covered"
    assert result.answer == FALLBACK
    assert result.sources == []
    assert result.flagged is True
    assert result.care_team_question == question
    assert result.care_team_source == "validation_fallback"


async def test_cited_answer_with_food_lookup(store, foods, document, items):
    llm = FakeLLM()
    llm.script(
        tool_call("lookup_food", {"name": "almonds"}),
        tool_call("search_education_library", {"query": "high oxalate"}, call_id="call-2"),
        final(
            "Your document says to limit high-oxalate foods [2]. Almonds have 122 mg of oxalate "
            "in 1 oz [5]. Foods with more than 10 mg per serving count as high [6].",
            [2, 5, 6],
        ),
    )
    result = await ask(llm, store, foods, document, items)

    assert result.kind == "answer"
    assert result.flagged is False and result.care_team_question is None
    assert [s.number for s in result.sources] == [2, 5, 6]
    assert [s.kind for s in result.sources] == ["document", "food", "library"]
    doc_source, food, passage = result.sources
    assert doc_source.title == "Your discharge document" and doc_source.url is None
    assert doc_source.content == "Limit high-oxalate foods to help prevent kidney stones."
    assert food.title == "Almonds"
    assert food.url == UCHICAGO_ARTICLE
    assert "oxalate 122 mg (Very High, per UChicago table serving '1 oz or 22 kernels')" in food.content
    assert FDC_URL in food.content and UCHICAGO_ARTICLE in food.content
    assert passage.title == "Low oxalate diet" and passage.url == LIBRARY_URL
    assert passage.content == OXALATE_PASSAGE

    # Tool results carry evidence numbers, continuing after the 4 document paragraphs.
    assert tool_results(llm, 1)[0]["foods"][0]["evidence_number"] == 5
    assert tool_results(llm, 2)[1]["passages"][0]["evidence_number"] == 6
    tools = {t["function"]["name"] for t in llm.calls[0][1]}
    assert tools == {"search_education_library", "lookup_food", "add_care_team_question"}


async def test_uncited_number_falls_back(store, foods, document, items):
    llm = FakeLLM()
    llm.script(
        tool_call("lookup_food", {"name": "almonds"}),
        final("Almonds have 200 mg of oxalate per serving [5].", [5]),
    )
    assert_fallback(await ask(llm, store, foods, document, items))


async def test_nonexistent_citation_falls_back(store, foods, document, items):
    llm = FakeLLM()
    llm.script(final("Limit high-oxalate foods [9].", [9]))
    assert_fallback(await ask(llm, store, foods, document, items))


async def test_malformed_json_falls_back(store, foods, document, items):
    llm = FakeLLM()
    llm.script({"content": "Almonds are fine [2]."})
    assert_fallback(await ask(llm, store, foods, document, items))


async def test_answer_without_citations_falls_back(store, foods, document, items):
    llm = FakeLLM()
    llm.script(final("Limit high-oxalate foods.", []))
    assert_fallback(await ask(llm, store, foods, document, items))


async def test_not_covered_adds_care_team_question_once(store, foods, document, items):
    with_tool = FakeLLM()
    with_tool.script(
        tool_call("add_care_team_question", {"question": "Can I eat almond flour?",
                                             "reason": "Not in my sources"}),
        tool_call("add_care_team_question", {"question": "Duplicate", "reason": "Again"},
                  call_id="call-2"),
        final("I could not find this, so I added it to your care-team questions.", [],
              kind="not_covered", care_team_question="Something else"),
    )
    result = await ask(with_tool, store, foods, document, items, question="Almond flour?")
    assert result.kind == "not_covered" and result.flagged is True
    assert result.care_team_question == "Can I eat almond flour?"
    assert result.care_team_reason == "Not in my sources"
    assert result.care_team_source == "agent"
    assert tool_results(with_tool, 2)[1]["status"] == "already added"

    without_tool = FakeLLM()
    without_tool.script(final("I could not find this.", [], kind="not_covered"))
    result = await ask(without_tool, store, foods, document, items, question="Almond flour?")
    assert result.flagged is True
    assert result.care_team_question == "Almond flour?"
    assert result.care_team_source == "agent"

    from_json = FakeLLM()
    from_json.script(final("I could not find this.", [], kind="not_covered",
                           care_team_question="Is almond flour high in oxalate?"))
    result = await ask(from_json, store, foods, document, items, question="Almond flour?")
    assert result.care_team_question == "Is almond flour high in oxalate?"


async def test_exhausted_rounds_falls_back(store, foods, document, items):
    llm = FakeLLM()
    llm.script(*[tool_call("lookup_food", {"name": "almonds"}, call_id=f"c{i}") for i in range(3)])
    assert_fallback(await ask(llm, store, foods, document, items, max_rounds=3))
    assert len(llm.calls) == 3


async def test_emergency_terms_passed_to_model(store, foods, document, items):
    llm = FakeLLM()
    llm.script(final("Call 911 now or follow your document's instructions [4].", [4],
                     kind="emergency"))
    result = await ask(llm, store, foods, document, items, question="I have chest pain",
                       emergency_terms=["chest pain"])
    assert result.kind == "emergency"
    user_content = llm.calls[0][0][-1]["content"]
    assert "A possible emergency was detected in the patient's message: chest pain" in user_content

    quiet = FakeLLM()
    quiet.script(final("Limit high-oxalate foods [2].", [2]))
    await ask(quiet, store, foods, document, items)
    assert "possible emergency" not in quiet.calls[0][0][-1]["content"]


async def test_context_has_numbered_document_active_items_and_history(store, foods, document, items):
    llm = FakeLLM()
    llm.script(final("Limit high-oxalate foods [2].", [2]))
    history = [{"role": "user", "content": f"q{i}"} for i in range(8)]
    await ask(llm, store, foods, document, items, history=history)
    messages = llm.calls[0][0]
    assert messages[0]["role"] == "system"
    assert [m["content"] for m in messages[1:-1]] == [f"q{i}" for i in range(2, 8)]
    user_content = messages[-1]["content"]
    assert "[2] Limit high-oxalate foods to help prevent kidney stones." in user_content
    assert "Limit high-oxalate foods." in user_content
    assert "Eat more spinach." not in user_content  # rejected items are not instructions
    assert user_content.rstrip().endswith("Can I eat almonds?")


async def test_library_unavailable_is_a_tool_error(tmp_path, foods, document, items):
    empty = KnowledgeStore(tmp_path / "empty.sqlite3", FakeLLM())
    for library in (None, empty):
        llm = FakeLLM()
        llm.script(
            tool_call("search_education_library", {"query": "oxalate"}),
            final("Your document says to limit high-oxalate foods [2].", [2]),
        )
        result = await ask(llm, library, foods, document, items)
        assert result.kind == "answer"
        assert tool_results(llm, 1) == [
            {"error": "The education library is not available right now."}
        ]


async def test_food_not_found_and_partial_match(store, foods, document, items):
    llm = FakeLLM()
    llm.script(
        tool_call("lookup_food", {"name": "dragon fruit"}),
        tool_call("lookup_food", {"name": "almond flour"}, call_id="call-2"),
        final("The food table does not list almond flour [5].", [5], kind="not_covered"),
    )
    result = await ask(llm, store, foods, document, items)
    missing, partial = tool_results(llm, 2)
    assert missing["foods"] == [] and missing["result"] == "not found"
    entry = partial["foods"][0]
    assert entry["match"] == "partial" and entry["evidence_number"] == 5
    assert entry["content"].startswith("Not in the food table. Closest entry: Almonds, ")
    assert result.sources[0].content == entry["content"]


async def test_bad_tool_arguments_return_errors(store, foods, document, items):
    llm = FakeLLM()
    bad = tool_call("lookup_food", {})
    bad["tool_calls"][0]["function"]["arguments"] = "not json"
    llm.script(
        bad,
        tool_call("delete_everything", {}, call_id="call-2"),
        final("Limit high-oxalate foods [2].", [2]),
    )
    result = await ask(llm, store, foods, document, items)
    assert result.kind == "answer"
    assert "error" in tool_results(llm, 1)[0]
    assert "error" in tool_results(llm, 2)[1]


def evidence(*contents):
    return {
        n: Evidence(number=n, kind="document", title="Your discharge document", url=None, content=c)
        for n, c in enumerate(contents, start=1)
    }


def raw(**fields):
    body = {"kind": "answer", "answer": "", "citations": [], "care_team_question": None}
    body.update(fields)
    return json.dumps(body)


def test_validate_accepts_grouped_markers_and_code_fences():
    ev = evidence("Drink 2 liters of water.", "Walk 10 minutes twice a day.")
    text = "```json\n" + raw(answer="Drink 2 liters [1] and walk 10 minutes [1, 2].",
                             citations=[1, 2]) + "\n```"
    assert validate(text, ev) == ("answer", "Drink 2 liters [1] and walk 10 minutes [1, 2].",
                                  [1, 2], None)


def test_validate_rejections():
    ev = evidence("Drink 2 liters of water.", "Walk 10 minutes twice a day.")
    bad = [
        "not json",
        json.dumps(["a list"]),
        raw(kind="maybe", answer="Drink water [1].", citations=[1]),
        raw(answer="", citations=[1]),
        raw(answer="Drink water [1].", citations=[1, 2]),       # citation without marker
        raw(answer="Drink water [1] and walk [2].", citations=[1]),  # marker without citation
        raw(answer="Drink water [1].", citations=["1"]),
        raw(answer="Drink 3 liters [1].", citations=[1]),       # 3 is in no cited evidence
        raw(answer="Walk 10 minutes [1].", citations=[1]),      # 10 is only in uncited evidence
        raw(kind="out_of_scope", answer="Ask a librarian [7].", citations=[7]),
    ]
    for text in bad:
        with pytest.raises(ValueError):
            validate(text, ev)


def test_validate_other_kinds_without_markers_and_marker_numbers_excluded():
    ev = evidence("Drink 2 liters of water.")
    assert validate(raw(kind="out_of_scope", answer="I can only help with your recovery."), ev)[0] \
        == "out_of_scope"
    assert validate(raw(kind="emergency", answer="Call 911 now."), ev)[0] == "emergency"
    # 1 appears only as a marker number, not in the evidence text.
    assert validate(raw(answer="Drink water [1].", citations=[1]), ev)[2] == [1]


def test_render_food_omits_nulls_and_prefers_fdc_url_without_oxalate(foods):
    banana = foods.lookup("banana")[0]
    content, url = render_food(banana)
    assert content.startswith("Banana, 1 medium, 118 g: potassium 422 mg.")
    assert "oxalate" not in content
    assert url == "https://fdc.nal.usda.gov/food-details/169145/nutrients"
    almonds = foods.lookup("almonds")[0]
    content, url = render_food(almonds)
    assert "phosphorus" not in content
    assert url == UCHICAGO_ARTICLE


def test_system_prompt_states_rules():
    prompt = agent.SYSTEM_PROMPT
    assert "\u2014" not in prompt
    for phrase in ("doctor's document wins", "never from your own knowledge",
                   "Never suggest changing, stopping, adding, or substituting",
                   "add_care_team_question", "Your discharge document does not cover this; here is general information.",
                   "call 911", "decline briefly", "untrusted data", "different food",
                   '"care_team_question"'):
        assert phrase in prompt, phrase
