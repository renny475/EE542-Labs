"""Server-side emergency term detection for patient chat messages.

False positives are acceptable; missed emergencies are not. No question-form
parsing is attempted. Document warning signs match by content words (stop
words dropped, crude stemming, negations folded to "not"), so the document's
"you cannot keep fluids down" matches "I can't keep fluids down".
"""

import re

from extraction import normalize
from models import ExtractedItem

_BREATH_TAIL = (
    r"(?:\w+\s+)?(?:breath\w*|catch\s+(?:my|a|your)\s+breath|get\s+(?:any\s+)?air)"
)
_CHEST = r"(?:my\s+|the\s+)?chest"
_CANT = r"(?:can'?t|can\s?not|couldn'?t|unable\s+to|not\s+able\s+to)"

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
        r"|\bhardly\s+(?:\w+\s+)?breath\w*"
        r"|\bbreathing\s+(?:\w+\s+){0,2}(?:hard|difficult\w*|labou?red|shallow|painful)"
    ),
    "gasping": r"\bgasp\w*",
    "choking": r"\bchok\w*",
    "collapsed": r"\bcollaps\w*",
    "trouble swallowing": (
        r"\b(?:trouble|difficult\w*|hard|struggl\w*|unable|can'?t|can\s?not)\s+"
        r"(?:to\s+)?(?:\w+\s+)?swallow\w*"
    ),
    "sudden vision loss": (
        r"\b(?:sudden\w*|abrupt\w*)\s+(?:\w+\s+){0,2}(?:vision|blind\w*)"
        r"|\b(?:lost|losing|loss\s+of)\s+(?:my\s+|the\s+)?(?:vision|sight)"
        r"|\bgoing\s+blind|\bdouble\s+vision"
    ),
    "shortness of breath": r"\bshort(?:ness)?\s+of\s+breath|\bout\s+of\s+breath",
    "short of breath": r"\bshort(?:ness)?\s+of\s+breath|\bout\s+of\s+breath",
    "can't breathe": (
        rf"\b(?:can'?t|can\s?not|couldn'?t|unable\s+to|not\s+able\s+to)\s+{_BREATH_TAIL}"
    ),
    "cannot breathe": "",  # alias of "can't breathe", set below
    "face drooping": (
        r"\bface\s+(?:is\s+)?(?:droop\w*|sagg\w*)|\bdroop\w*\s+face|\bfacial\s+droop\w*"
    ),
    "stroke": r"\bstroke\b",
    "slurred speech": r"\bslurr\w*\s+(?:speech|words)|\bspeech\s+(?:is\s+)?slurr\w*|\bslurring",
    "weakness on one side": (
        r"\bweak\w*\s+(?:on|in)\s+(?:one|the\s+left|the\s+right|my\s+left|my\s+right)"
    ),
    "numbness on one side": (
        r"\b(?:face|arm|leg|hand)\s+(?:is\s+|feels\s+|went\s+|has\s+gone\s+)?numb\w*|"
        r"\bnumb\w*\s+(?:on|in)\s+(?:one|the\s+left|the\s+right|my\s+left|my\s+right)"
    ),
    "worst headache": r"\bworst\s+(?:\w+\s+)?headache|\bthunderclap",
    "fainted": r"\bfaint\w*",
    "passed out": r"\bpass\w*\s+out|\bblack\w*\s+out",
    "unconscious": r"\bunconscious\w*|\bunresponsive|\bnot\s+responding",
    "coughing up blood": r"\bcough\w*\s+(?:up\s+)?blood",
    "vomiting blood": (
        r"\b(?:vomit\w*|throw\w*\s+up|throw\s+up)\s+(?:up\s+)?blood|\bblood\w*\s+vomit"
        r"|\bblood\s+in\s+(?:\w+\s+){0,2}vomit"
    ),
    "heavy bleeding": (
        r"\bheav\w*\s+bleed\w*|\bbleed\w*\s+(?:heav\w*|a\s+lot|badly|profusely)"
        r"|\bhemorrhag\w*|\bhaemorrhag\w*"
        r"|\bbleed\w*\s+through|\bsoak\w*\s+(?:through\s+)?(?:\w+\s+){0,2}(?:bandage|dressing)"
        r"|\bblood\w*\s+(?:is\s+)?soak\w*"
        r"|\bblood\s+in\s+(?:\w+\s+){0,2}(?:stool|urine|poop|pee)"
    ),
    "bleeding that won't stop": (
        r"\b(?:bleed\w*|blood)\s+(?:\w+\s+){0,3}(?:won'?t|will\s+not|doesn'?t|does\s+not|"
        r"not|isn'?t|cannot|can'?t)\s+(?:\w+\s+)?stop\w*"
        r"|\b(?:won'?t|can'?t|cannot|not)\s+(?:\w+\s+)?stop\w*\s+(?:the\s+)?bleed\w*"
    ),
    "seizure": r"\bseiz\w*|\bconvuls\w*",
    "suicidal": r"\bsuicid\w*",
    "want to die": r"\bwant(?:s|ed)?\s+to\s+die|\bdon'?t\s+(?:\w+\s+)?want\s+to\s+(?:\w+\s+)?(?:live|be\s+alive|be\s+here|go\s+on|exist)",
    "kill myself": r"\bkill\w*\s+(?:my\s?self|me)\b|\bend\w*\s+(?:my\s+life|it\s+all)|\bwant\w*\s+to\s+end\s+(?:it|everything)",
    "self harm": r"\bself\s?harm\w*|\b(?:hurt|harm|cut)\w*\s+my\s?self",
    "overdose": r"\boverdos\w*|\b(?:took|taken|swallow\w*|ate|drank)\s+(?:\w+\s+){0,2}too\s+(?:many|much)",
    "severe allergic reaction": r"\ballergic\s+reaction|\bgoing\s+into\s+anaphyla\w*",
    "anaphylaxis": r"\banaphyla\w*",
    "swelling of the throat": (
        r"\bthroat\s+(?:is\s+|feels\s+)?(?:clos\w*|swell\w*|swollen|tight\w*)"
        r"|\bswell\w*\s+(?:of|in)\s+(?:the\s+|my\s+)?throat|\bswollen\s+throat"
    ),
    "throat closing": "",  # alias of "swelling of the throat", set below
    "face swelling": (
        r"\b(?:face|lips?|tongue)\s+(?:is\s+|are\s+|feels?\s+)?(?:swell\w*|swollen)"
        r"|\bswell\w*\s+(?:of|in)\s+(?:the\s+|my\s+)?(?:face|lips?|tongue)"
        r"|\bswollen\s+(?:face|lips?|tongue)"
    ),
    "blue lips": (
        r"\b(?:blue|bluish|purple|grey|gray)\s+(?:lips?|face)|"
        r"\b(?:lips?|face|skin|fingers?)\s+(?:\w+\s+){0,2}(?:blue|bluish|purple|grey|gray)"
    ),
    "heart attack": r"\bheart\s+attack|\bcardiac\s+arrest",
    "vomiting that won't stop": (
        r"\bkeep\w*\s+(?:on\s+)?(?:throw\w*\s+up|vomit\w*|puk\w*)"
        rf"|\b{_CANT}\s+(?:\w+\s+)?stop\s+(?:throw\w*\s+up|vomit\w*|puk\w*)"
        r"|\b(?:vomit\w*|throw\w*\s+up|puk\w*)\s+(?:\w+\s+){0,3}(?:won'?t|will\s+not|"
        r"doesn'?t|does\s+not|not|isn'?t)\s+(?:\w+\s+)?stop\w*"
    ),
    "can't keep fluids down": rf"\b{_CANT}\s+keep\s+(?:\w+\s+){{0,3}}down\b",
    "can't urinate": rf"\b{_CANT}\s+(?:\w+\s+)?(?:urinat\w*|pee|void|pass\s+urine)\b",
}
_PATTERNS["cannot breathe"] = _PATTERNS["can't breathe"]
_PATTERNS["throat closing"] = _PATTERNS["swelling of the throat"]
_PATTERNS["short of breath"] = _PATTERNS["shortness of breath"]
_COMPILED = {label: re.compile(rx) for label, rx in _PATTERNS.items()}
EMERGENCY_TERMS: tuple[str, ...] = tuple(_PATTERNS)

# Generic words that appear in nearly every discharge sheet. A document phrase
# made of only one of them would banner almost every message, so it is dropped
# (the static patterns above still catch real emergencies). A phrase needs a
# specific word ("clots", "urinate") or at least two generic ones ("fever ...
# chills") to count.
GENERIC_SINGLE_WORDS = frozenset(
    {
        "pain", "fever", "call", "doctor", "symptoms", "symptom", "swelling",
        "redness", "nausea", "your", "home", "worse", "bleeding", "headache",
        "dizziness", "vomiting", "rash", "cough", "chills", "soreness",
        "discomfort", "infection", "problems", "problem", "signs", "sign",
    }
)

# Words that carry no symptom of their own in a warning-sign line: function
# words, the "call 911 / go to the ER / call the clinic" frame, and intensity
# words a patient may leave out ("severe pain that your medicine does not
# control" must match "the pain my medicine does not control").
STOP_WORDS = frozenset(
    """a an the and or of to for if in on at with by from as into about over under than then
    so is are was were be been being am i you your my me we our us he she they it its this
    that these those there here which who what when where while do does did doing have has
    had having can could will would should may might must shall get gets got very really too
    also just still now any some all more most much many such severe severely high higher
    low lower heavy heavily new sudden suddenly mild bad call go return er emergency room
    clinic doctor provider office hospital right away immediately please see notice
    experience develop start starts started feel feels felt""".split()
)
# Words that must match but do not make a phrase specific on their own.
WEAK_WORDS = frozenset({"not", "keep", "down", "up", "stop", "come", "make", "take"})
NEGATIONS = frozenset({"no", "not", "never", "cannot", "unable", "cant", "wont", "dont",
                       "doesnt", "didnt", "isnt", "couldnt", "wouldnt", "arent"})

_MIN_CHARS = 4
_SPLIT_CLAUSES = re.compile(r"[,;]")
_SPLIT_WORDS = re.compile(r"[,;]|\bor\b|\band\b", re.IGNORECASE)
_LEADING = re.compile(r"^(?:or|and|if|for)\s+", re.IGNORECASE)


def _canon(s: str) -> str:
    """Normalize, then turn punctuation (except apostrophes) into spaces."""
    s = normalize(s).replace("'", "\0")
    s = re.sub(r"[^\w\0]+", " ", s)
    return s.replace("\0", "'").strip()


def _stem(word: str) -> str:
    """A crude suffix strip, applied the same way to document and message words, so
    "fluids" == "fluid", "urinating" == "urinate", "chills" == "chill"."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]  # "controlled" -> "control", like "control"
    return word


def _words(text: str) -> list[str]:
    """Stemmed words with every negation folded to "not"."""
    out = []
    for word in _canon(text).split():
        if word.endswith("n't") or word in NEGATIONS:
            out.append("not")
        else:
            out.append(_stem(word.replace("'", "")))
    return out


_GENERIC_STEMS = frozenset(_stem(w) for w in GENERIC_SINGLE_WORDS)
_STOP_STEMS = frozenset(STOP_WORDS | {_stem(w) for w in STOP_WORDS})
_WEAK_STEMS = frozenset(_stem(w) for w in WEAK_WORDS)


def _content(phrase: str) -> frozenset[str]:
    """The words a message must contain to match `phrase`: no stop words, no
    numbers or one-letter units (a patient's "102" need not equal the
    document's "101 F")."""
    return frozenset(
        w for w in _words(phrase)
        if len(w) > 1 and not any(c.isdigit() for c in w) and w not in _STOP_STEMS
    )


def _specific_enough(content: frozenset[str]) -> bool:
    strong = content - _GENERIC_STEMS - _WEAK_STEMS
    generic = content & _GENERIC_STEMS
    return any(len(w) >= _MIN_CHARS for w in strong) or len(strong) >= 2 or len(generic) >= 2


def _phrases(text: str):
    """The whole line, its clauses (split on , ;) and its parts (also split on or/and)."""
    yield text
    yield from _SPLIT_CLAUSES.split(text)
    yield from _SPLIT_WORDS.split(text)


def warning_terms(items: list[ExtractedItem]) -> list[str]:
    """Key phrases from active warning_signs items, from the document's own bullet
    (source_quote) first and then the plain-language text. One phrase per distinct
    set of content words, in order."""
    out: list[str] = []
    seen: set[frozenset[str]] = set()
    for it in items:
        if it.category != "warning_signs" or it.status == "rejected":
            continue
        for line in (it.source_quote or "", it.text or ""):
            for part in _phrases(line):
                phrase = _LEADING.sub("", part.strip(" \t\n.:")).strip()
                content = _content(phrase)
                if not content or content in seen or not _specific_enough(content):
                    continue
                seen.add(content)
                out.append(phrase)
    return out


def detect_emergency(message: str, extra_terms: list[str]) -> list[str]:
    """Return matched human-readable terms (deduplicated, static terms first).

    A document term matches when every one of its content words is in the
    message, in any order. A term whose words are all covered by a longer term
    that already matched is not reported again."""
    text = _canon(message)
    hits: list[str] = []
    seen_patterns: set[str] = set()
    for label, rx in _COMPILED.items():
        # Alias labels share one pattern: one phrase yields one label.
        if rx.pattern not in seen_patterns and rx.search(text):
            seen_patterns.add(rx.pattern)
            hits.append(label)
    words = set(_words(message))
    matched: list[frozenset[str]] = []
    terms = [(term, _content(term)) for term in extra_terms]
    for term, content in sorted(terms, key=lambda t: -len(t[1])):
        if not content or not content <= words or any(content <= m for m in matched):
            continue
        matched.append(content)
        label = _canon(term)
        if label not in hits:
            hits.append(label)
    return hits
