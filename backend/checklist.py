"""Pure checklist generation from confirmed extracted items. No LLM, no database.

Item details come from an LLM and are not validated upstream, so every read
here is defensive: bad values fall back to documented defaults and nothing
raises."""
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

SLOT_ORDER = ["morning", "midday", "evening", "bedtime", "anytime"]
TIMED_SLOTS = SLOT_ORDER[:4]
SCHEDULED_CATEGORIES = ("monitoring", "wound_or_device_care")
MAX_DAYS = 3650  # ten years; anything larger is treated as garbage

# Ruling R18: a diet item that states a quantified daily intake target ("drink
# enough fluids to make at least 2.5 liters of urine a day") is something the
# patient acts on every day, so it becomes one "any time" daily item. Diet
# restrictions ("avoid", "limit", "under 2,300 mg a day") stay off the checklist.
_VOLUME = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:liters?|litres?|l|ml|milliliters?|millilitres?|cups?|glass(?:es)?"
    r"|ounces?|oz|quarts?)\b",
    re.IGNORECASE,
)
_DAILY = re.compile(r"\b(?:a|per|each|every)\s+day\b|\bdaily\b|/\s*day\b", re.IGNORECASE)
_RESTRICTION = re.compile(
    r"\b(?:limit\w*|less\s+than|under|no\s+more\s+than|not\s+more\s+than|at\s+most"
    r"|maximum|max|restrict\w*|avoid\w*|cut\s+back|reduce|fewer\s+than|below|up\s+to"
    r"|do\s+not|don'?t|never)\b",
    re.IGNORECASE,
)


@dataclass
class ChecklistSpec:
    extracted_item_id: int
    label: str
    slot: str
    start_date: date
    end_date: Optional[date]
    one_time: bool
    approximate: bool
    undated: bool


def _int(value: Any) -> Optional[int]:
    """Coerce ints, integral floats and numeric strings; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                return None
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return value is True


def _positive_days(value: Any) -> Optional[int]:
    n = _int(value)
    return n if n is not None and 1 <= n <= MAX_DAYS else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _count_slots(count: int) -> list[str]:
    if count <= 1:
        return ["morning"]
    if count == 2:
        return ["morning", "evening"]
    if count == 3:
        return ["morning", "midday", "evening"]
    return list(TIMED_SLOTS)


def _slots(details: dict) -> list[str]:
    given = details.get("times_of_day")
    if isinstance(given, list):
        wanted = {s.strip().lower() for s in given if isinstance(s, str)}
        found = [s for s in TIMED_SLOTS if s in wanted]
        if found:
            return found
    count = _int(details.get("times_per_day"))
    if count is not None and count >= 1:
        return _count_slots(count)
    interval = _int(details.get("interval_hours"))
    if interval is not None and 1 <= interval <= 24:
        return _count_slots(24 // interval)
    return ["anytime"]


def _medication_label(item, details: dict) -> str:
    name = _text(details.get("name"))
    if not name:
        return item.text
    label = " ".join(p for p in (name, _text(details.get("dose"))) if p)
    instructions = _text(details.get("instructions"))
    return f"{label} - {instructions}" if instructions else label


def _parse_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _add_days(anchor: date, days: int) -> Optional[date]:
    try:
        return anchor + timedelta(days=days)
    except OverflowError:
        return None


def is_daily_intake_target(item) -> bool:
    """A diet item with a daily volume to reach, and no restriction wording."""
    text = " ".join(t for t in (item.text, item.source_quote) if isinstance(t, str))
    return bool(_VOLUME.search(text) and _DAILY.search(text) and not _RESTRICTION.search(text))


def build_checklist(items: list, anchor: date) -> list[ChecklistSpec]:
    specs: list[ChecklistSpec] = []
    for item in items:
        if item.status != "active":
            continue
        details = item.details if isinstance(item.details, dict) else {}
        if item.category == "follow_up":
            specs.append(_follow_up(item, details, anchor))
        elif item.category == "diet":
            if is_daily_intake_target(item):
                specs.append(ChecklistSpec(item.id, item.text, "anytime", anchor, None,
                                           False, False, False))
        elif item.category == "medication" or item.category in SCHEDULED_CATEGORIES:
            if item.category == "medication":
                if _bool(details.get("as_needed")):
                    continue
                label = _medication_label(item, details)
            else:
                label = item.text
            end = None
            days = _positive_days(details.get("duration_days"))
            if days is not None:
                end = _add_days(anchor, days - 1)
            for slot in _slots(details):
                specs.append(ChecklistSpec(item.id, label, slot, anchor, end, False, False, False))
    return specs


def _follow_up(item, details: dict, anchor: date) -> ChecklistSpec:
    when = _parse_date(details.get("date"))
    approximate = False
    if when is None:
        days = _int(details.get("days_after_discharge"))
        if days is not None and 0 <= days <= MAX_DAYS:
            when = _add_days(anchor, days)
            approximate = when is not None
    if when is None:
        return ChecklistSpec(item.id, item.text, "anytime", anchor, None, False, False, True)
    return ChecklistSpec(item.id, item.text, "anytime", when, when, True, approximate, False)
