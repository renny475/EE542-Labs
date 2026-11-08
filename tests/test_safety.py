import sys
from datetime import date
from pathlib import Path

import pytest

from models import ExtractedItem
from safety import EMERGENCY_TERMS, detect_emergency, warning_terms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import seed_demo  # noqa: E402


def item(text, category="warning_signs", status="active", quote=None):
    return ExtractedItem(category=category, text=text, source_quote=quote or text, status=status)


def kidney_stone_warning_items():
    return [
        item(text, quote=quote)
        for category, text, quote, _ in seed_demo.build_items(date(2026, 11, 5))
        if category == "warning_signs"
    ]


def test_detects_chest_pain_any_case():
    assert "chest pain" in detect_emergency("I have CHEST   Pain right now", [])


def test_detects_document_warning_term():
    terms = warning_terms([item("Call 911 for fever over 101.5 F or confusion")])
    assert "confusion" in detect_emergency("my dad has some confusion", terms)


def test_no_false_negative_in_question_form():
    assert detect_emergency("is chest pain normal?", [])


def test_benign_message_no_hit():
    assert detect_emergency("Can I eat toast for breakfast?", []) == []


@pytest.mark.parametrize(
    "msg",
    [
        "I can’t breathe",
        "i CAN'T catch my breath!",
        "having trouble breathing",
        "difficulty breathing since morning",
        "my chest hurts",
        "chest-pain, help",
        "I think I'm having a seizure.",
        "he is coughing up blood",
        "cannot breathe",
    ],
)
def test_variants_detected(msg):
    assert detect_emergency(msg, [])


@pytest.mark.parametrize(
    "msg",
    [
        "the seizures are back", "he overdosed", "he was fainting", "bleeding heavily",
        "my chest is hurting", "chest feels tight", "I feel faint", "cant breath",
        "hard to breathe", "I am struggling to breathe", "bleeding a lot",
        "I want to kill myself", "she is having a stroke", "heart attack",
        "he is gasping", "anaphylaxis", "my throat is closing", "his face is swelling",
        "tongue swollen", "lips swelling", "I feel suicidal", "thinking of suicide",
        "I want to die", "thinking about self harm", "self-harm", "bleeding won't stop",
        "the bleeding will not stop", "I can not breathe", "short of breath",
        "pain in my chest",
    ],
)
def test_realistic_phrasings_detected(msg):
    assert detect_emergency(msg, []), msg


@pytest.mark.parametrize(
    "msg",
    [
        "my lips are turning blue", "my lips are blue", "I can hardly breathe",
        "breathing is hard", "I am choking", "I collapsed",
        "there is blood in my vomit", "blood wont stop coming out",
        "I am bleeding through the bandage", "swallowed too many pills",
        "i dont want to be here anymore", "I want to end it",
        "hives and trouble swallowing", "sudden vision loss", "face is numb",
    ],
)
def test_more_realistic_phrasings_detected(msg):
    assert detect_emergency(msg, []), msg


@pytest.mark.parametrize(
    "msg",
    [
        "I have a little pain at the incision",
        "when can I take a deep breath of fresh air outside",
        "is it ok to feel tired",
        "when can I take a shower",
        "my incision is sore",
    ],
)
def test_benign_probes_negative(msg):
    assert detect_emergency(msg, []) == [], msg


def test_alias_labels_deduped():
    assert detect_emergency("I am short of breath", []) == ["shortness of breath"]
    assert detect_emergency("I cannot breathe", []) == ["can't breathe"]


def test_hit_returns_readable_terms():
    assert detect_emergency("the seizures are back", []) == ["seizure"]


def test_generic_document_words_dropped_but_multiword_and_specific_kept():
    items = [
        item("Call your doctor if you have fever, pain, or bleeding"),
        item("confusion or a new rash"),
    ]
    terms = warning_terms(items)
    for generic in ("fever", "pain", "bleeding", "rash"):
        assert generic not in terms
    assert "confusion" in terms
    assert detect_emergency("what is a fever?", terms) == []


def test_extra_terms_returned_and_deduped():
    hits = detect_emergency("Chest pain and confusion", ["Confusion", "confusion"])
    assert hits.count("confusion") == 1
    assert "chest pain" in hits


def test_all_minimum_terms_present_and_detected():
    for t in EMERGENCY_TERMS:
        assert detect_emergency(f"I have {t} today", []), t


def test_warning_terms_split_filter_dedupe_and_ignore_rejected():
    items = [
        item("redness, swelling; or pus at the wound"),
        item("redness and confusion"),
        item("Call 911 for severe pain", status="rejected"),
        item("take aspirin", category="medications"),
    ]
    assert warning_terms(items) == [
        "redness, swelling; or pus at the wound", "pus at the wound",
        "redness and confusion", "confusion",
    ]


# ---------- the document's own warning signs (kidney-stone seed) ----------

@pytest.mark.parametrize(
    "msg",
    [
        "I cannot keep fluids down",
        "I can't keep fluids down since this morning",
        "I have a fever of 102 and chills",
        "fever of 102 and chills",
        "the pain my medicine does not control",
        "my pain is severe and the medicine doesn't control it",
        "I cannot urinate",
        "I see clots in my urine",
    ],
)
def test_kidney_stone_document_warning_signs_detected(msg):
    terms = warning_terms(kidney_stone_warning_items())
    assert detect_emergency(msg, terms), (msg, terms)


@pytest.mark.parametrize(
    "msg",
    [
        "I keep throwing up",
        "I can't stop vomiting",
        "the vomiting will not stop",
        "I can't keep anything down",
        "I cannot pee",
    ],
)
def test_static_vomiting_and_urination_rules(msg):
    assert detect_emergency(msg, []), msg


@pytest.mark.parametrize(
    "msg",
    [
        "How much fluid should I drink?",
        "Can I take my medicine with food?",
        "Is it normal to urinate often with the stent?",
        "I keep fluids by my bed",
        "What does my document say about fever?",
        "When do I call the clinic?",
        "My medicine controls the pain well",
        "I threw up once yesterday after a big meal",
    ],
)
def test_kidney_stone_document_terms_do_not_fire_on_generic_words(msg):
    terms = warning_terms(kidney_stone_warning_items())
    assert detect_emergency(msg, terms) == [], (msg, detect_emergency(msg, terms))


def test_warning_terms_use_source_quote_too():
    terms = warning_terms([item("Go to the ER if you vomit a lot.", quote="persistent emesis")])
    assert detect_emergency("I have persistent emesis", terms)


def test_document_hit_reports_the_phrase_once():
    terms = warning_terms(kidney_stone_warning_items())
    hits = detect_emergency("I see heavy bleeding and clots", terms)
    # "clots" alone is covered by the longer document phrase that matched.
    assert hits == ["heavy bleeding", "you see heavy bleeding or clots"]
