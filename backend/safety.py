"""Server-side emergency term detection for patient chat messages.

False positives are acceptable; missed emergencies are not. No negation or
question-form parsing is attempted.
"""

import re

from extraction import normalize
from models import ExtractedItem

EMERGENCY_TERMS: tuple[str, ...] = (
    "chest pain",
    "chest pressure",
    "chest hurts",
    "chest tightness",
    "trouble breathing",
    "difficulty breathing",
    "shortness of breath",
    "short of breath",
    "can't breathe",
    "cannot breathe",
    "can not breathe",
    "cant breathe",
    "can't catch my breath",
    "cannot catch my breath",
    "face drooping",
    "slurred speech",
    "weakness on one side",
    "numbness on one side",
    "worst headache",
    "fainted",
    "passed out",
    "unconscious",
    "coughing up blood",
    "vomiting blood",
    "heavy bleeding",
    "bleeding that won't stop",
    "seizure",
    "suicidal",
    "overdose",
    "severe allergic reaction",
    "swelling of the throat",
    "blue lips",
)

_MIN_CHARS = 4
_MAX_WORDS = 6
_SPLIT = re.compile(r"[,;]|\bor\b|\band\b", re.IGNORECASE)


def _canon(s: str) -> str:
    """Normalize, then turn punctuation (except apostrophes) into spaces."""
    s = normalize(s).replace("'", "\0")
    s = re.sub(r"[^\w\0]+", " ", s)
    return s.replace("\0", "'").strip()


def warning_terms(items: list[ExtractedItem]) -> list[str]:
    """Key phrases from active warning_signs items, deduplicated in order."""
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        if it.category != "warning_signs" or it.status == "rejected":
            continue
        for part in _SPLIT.split(it.text):
            phrase = part.strip(" \t\n.:")
            words = phrase.split()
            if not (1 <= len(words) <= _MAX_WORDS) or len(phrase) < _MIN_CHARS:
                continue
            key = _canon(phrase)
            if key and key not in seen:
                seen.add(key)
                out.append(phrase)
    return out


def detect_emergency(message: str, extra_terms: list[str]) -> list[str]:
    """Return matched terms (canonical form, deduplicated, in term order)."""
    text = _canon(message)
    hits: list[str] = []
    for term in (*EMERGENCY_TERMS, *extra_terms):
        needle = _canon(term)
        if not needle or needle in hits:
            continue
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text):
            hits.append(needle)
    return hits
