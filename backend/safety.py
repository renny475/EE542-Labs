"""Server-side emergency term detection for patient chat messages.

False positives are acceptable; missed emergencies are not. No negation or
question-form parsing is attempted.
"""

import re

from extraction import normalize
from models import ExtractedItem

_BREATH_TAIL = (
    r"(?:\w+\s+)?(?:breath\w*|catch\s+(?:my|a|your)\s+breath|get\s+(?:any\s+)?air)"
)
_CHEST = r"(?:my\s+|the\s+)?chest"

# label -> regex, matched against the canonical (lowercase, punctuation-free)
# message. Patterns use a leading word boundary and stems (seiz\w*, faint\w*)
# so inflected forms ("seizures", "overdosed") still hit. Labels are the
# human-readable terms returned to the caller. Err toward detection.
_PATTERNS: dict[str, str] = {
    "chest pain": (
        rf"\bchest\s+(?:(?:is|feels|has|keeps)\s+)?(?:\w+\s+)?"
        rf"(?:pain\w*|ach\w*|hurt\w*|sore)|\bpain\w*\s+(?:in|on)\s+{_CHEST}"
    ),
    "chest pressure": (
        rf"\bchest\s+(?:(?:is|feels|has|keeps)\s+)?(?:\w+\s+)?"
        rf"(?:pressure|tight\w*|heavy|squeez\w*|crush\w*)"
        rf"|\b(?:pressure|tight\w*|squeez\w*|heaviness)\s+(?:in|on)\s+{_CHEST}"
    ),
    "trouble breathing": (
        r"\b(?:trouble|difficult\w*|hard|struggl\w*|labou?red)\s+(?:to\s+|with\s+)?"
        r"(?:\w+\s+)?(?:breath\w*)|\b(?:trouble|difficult\w*|hard|struggl\w*)\s+"
        r"(?:to\s+)?(?:catch\w*|get\w*)\s+(?:my|a|your)?\s*(?:breath|air)"
        r"|\b(?:stopped|not|isn'?t)\s+breathing"
    ),
    "gasping": r"\bgasp\w*",
    "shortness of breath": r"\bshort(?:ness)?\s+of\s+breath|\bout\s+of\s+breath",
    "short of breath": r"\bshort(?:ness)?\s+of\s+breath|\bout\s+of\s+breath",
    "can't breathe": (
        rf"\b(?:can'?t|couldn'?t|unable\s+to|not\s+able\s+to)\s+{_BREATH_TAIL}"
    ),
    "cannot breathe": rf"\bcan\s?not\s+{_BREATH_TAIL}",
    "face drooping": (
        r"\bface\s+(?:is\s+)?(?:droop\w*|sagg\w*)|\bdroop\w*\s+face|\bfacial\s+droop\w*"
    ),
    "stroke": r"\bstroke\b",
    "slurred speech": r"\bslurr\w*\s+(?:speech|words)|\bspeech\s+(?:is\s+)?slurr\w*|\bslurring",
    "weakness on one side": (
        r"\bweak\w*\s+(?:on|in)\s+(?:one|the\s+left|the\s+right|my\s+left|my\s+right)"
    ),
    "numbness on one side": (
        r"\bnumb\w*\s+(?:on|in)\s+(?:one|the\s+left|the\s+right|my\s+left|my\s+right)"
    ),
    "worst headache": r"\bworst\s+(?:\w+\s+)?headache|\bthunderclap",
    "fainted": r"\bfaint\w*",
    "passed out": r"\bpass\w*\s+out|\bblack\w*\s+out",
    "unconscious": r"\bunconscious\w*|\bunresponsive|\bnot\s+responding",
    "coughing up blood": r"\bcough\w*\s+(?:up\s+)?blood",
    "vomiting blood": (
        r"\b(?:vomit\w*|throw\w*\s+up|throw\s+up)\s+(?:up\s+)?blood|\bblood\w*\s+vomit"
    ),
    "heavy bleeding": (
        r"\bheav\w*\s+bleed\w*|\bbleed\w*\s+(?:heav\w*|a\s+lot|badly|profusely)"
        r"|\bhemorrhag\w*|\bhaemorrhag\w*"
    ),
    "bleeding that won't stop": (
        r"\bbleed\w*\s+(?:\w+\s+){0,3}(?:won'?t|will\s+not|doesn'?t|does\s+not|"
        r"not|isn'?t|cannot|can'?t)\s+(?:\w+\s+)?stop\w*"
        r"|\b(?:won'?t|can'?t|cannot|not)\s+(?:\w+\s+)?stop\w*\s+(?:the\s+)?bleed\w*"
    ),
    "seizure": r"\bseiz\w*|\bconvuls\w*",
    "suicidal": r"\bsuicid\w*",
    "want to die": r"\bwant(?:s|ed)?\s+to\s+die|\bdon'?t\s+want\s+to\s+(?:live|be\s+alive)",
    "kill myself": r"\bkill\w*\s+(?:my\s?self|me)\b|\bend\w*\s+(?:my\s+life|it\s+all)",
    "self harm": r"\bself\s?harm\w*|\b(?:hurt|harm|cut)\w*\s+my\s?self",
    "overdose": r"\boverdos\w*|\btook\s+too\s+(?:many|much)",
    "severe allergic reaction": r"\ballergic\s+reaction|\bgoing\s+into\s+anaphyla\w*",
    "anaphylaxis": r"\banaphyla\w*",
    "swelling of the throat": (
        r"\bthroat\s+(?:is\s+|feels\s+)?(?:clos\w*|swell\w*|swollen|tight\w*)"
        r"|\bswell\w*\s+(?:of|in)\s+(?:the\s+|my\s+)?throat|\bswollen\s+throat"
    ),
    "throat closing": r"\bthroat\s+(?:is\s+|feels\s+)?(?:clos\w*|tight\w*)",
    "face swelling": (
        r"\b(?:face|lips?|tongue)\s+(?:is\s+|are\s+|feels?\s+)?(?:swell\w*|swollen)"
        r"|\bswell\w*\s+(?:of|in)\s+(?:the\s+|my\s+)?(?:face|lips?|tongue)"
        r"|\bswollen\s+(?:face|lips?|tongue)"
    ),
    "blue lips": (
        r"\b(?:blue|bluish|purple|grey|gray)\s+(?:lips?|face)|"
        r"\blips?\s+(?:are\s+|turned\s+|turning\s+|look\s+)?(?:blue|bluish|purple)"
    ),
    "heart attack": r"\bheart\s+attack|\bcardiac\s+arrest",
}
_COMPILED = {label: re.compile(rx) for label, rx in _PATTERNS.items()}
EMERGENCY_TERMS: tuple[str, ...] = tuple(_PATTERNS)

# Generic single words that appear in nearly every discharge sheet. As
# document-derived terms they would banner almost every message, so they are
# dropped (multi-word phrases are kept). Trade-off: a patient writing only
# "pain" or "fever" will not be flagged via the document; real emergencies
# are still caught by the static patterns above, which are unaffected.
GENERIC_SINGLE_WORDS = frozenset(
    {
        "pain", "fever", "call", "doctor", "symptoms", "symptom", "swelling",
        "redness", "nausea", "your", "home", "worse", "bleeding", "headache",
        "dizziness", "vomiting", "rash", "cough", "chills", "soreness",
        "discomfort", "infection", "problems", "problem", "signs", "sign",
    }
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
            if len(words) == 1 and key in GENERIC_SINGLE_WORDS:
                continue
            if key and key not in seen:
                seen.add(key)
                out.append(phrase)
    return out


def detect_emergency(message: str, extra_terms: list[str]) -> list[str]:
    """Return matched human-readable terms (deduplicated, static terms first)."""
    text = _canon(message)
    hits: list[str] = []
    for label, rx in _COMPILED.items():
        if rx.search(text):
            hits.append(label)
    for term in extra_terms:
        needle = _canon(term)
        if not needle or needle in hits:
            continue
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text):
            hits.append(needle)
    return hits
