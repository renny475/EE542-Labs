import pytest

from models import ExtractedItem
from safety import EMERGENCY_TERMS, detect_emergency, warning_terms


def item(text, category="warning_signs", status="active"):
    return ExtractedItem(category=category, text=text, source_quote=text, status=status)


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
    assert warning_terms(items) == ["pus at the wound", "confusion"]
