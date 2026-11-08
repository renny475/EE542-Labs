"""Food nutrient table: load `data/foods.json` and look foods up by name.

Every nutrient value is per stated serving and carries a `source_id` that
points into the table's `sources`. A value of None means no published value
was found, and callers must say "unknown" rather than estimate.

Lookup normalizes names the way patients type them: lowercase, punctuation
stripped, and a plural ending removed per word ("ies" -> "y" for words longer
than four letters, "es" after s/x/z/ch/sh/o, otherwise a trailing "s" unless it
is "ss"). Matching is on whole words, so "tea" never matches "steak".

Each result says how it matched: `match` is "exact" when the normalized query
equals the food's normalized name or an alias, and "partial" when one only
contains the other ("almond flour" -> Almonds). A partial match is a different
food until shown otherwise, so callers must not present its numbers as the
asked-about food's. `matched_term` is the name or alias that matched.
"""
import copy
import json
import re
from pathlib import Path

_APOSTROPHES = re.compile(r"['’]")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_ES_STEMS = ("s", "x", "z", "ch", "sh", "o")


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[:-2].endswith(_ES_STEMS):
        return word[:-2]
    if len(word) > 1 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> tuple[str, ...]:
    text = _APOSTROPHES.sub("", text.lower())
    return tuple(_singular(w) for w in _NON_WORD.split(text) if w)


def normalize(text: str) -> str:
    return " ".join(_words(text))


def _contains(outer: tuple[str, ...], inner: tuple[str, ...]) -> bool:
    n = len(inner)
    return any(outer[i:i + n] == inner for i in range(len(outer) - n + 1))


class FoodTable:
    def __init__(self, foods: list[dict], sources: dict[str, dict]):
        self._foods = foods
        self._sources = sources
        self._terms = [
            [(_words(term), term) for term in [food["name"], *food.get("aliases", [])]
             if _words(term)]
            for food in foods
        ]

    @classmethod
    def load(cls, path) -> "FoodTable":
        data = json.loads(Path(path).read_text())
        foods, sources = data["foods"], data["sources"]
        for food in foods:
            for key, entry in food["nutrients"].items():
                if entry["value"] is None:
                    if entry.get("source_id") is not None:
                        raise ValueError(f"{food['id']}.{key}: null value with a source_id")
                elif entry["source_id"] not in sources:
                    raise ValueError(
                        f"{food['id']}.{key}: unknown source_id {entry['source_id']!r}"
                    )
        return cls(foods, sources)

    def lookup(self, query: str, limit: int = 3) -> list[dict]:
        q = _words(query)
        if not q:
            return []
        ranked = []
        for index, terms in enumerate(self._terms):
            best = None
            for words, term in terms:
                if words == q:
                    score = (0, 0)
                elif _contains(words, q) or _contains(q, words):
                    score = (1, abs(len(words) - len(q)))
                else:
                    continue
                if best is None or score < best[0]:
                    best = (score, term)
            if best is not None:
                ranked.append((best[0], index, best[1]))
        ranked.sort()
        return [self._result(self._foods[index], score, term)
                for score, index, term in ranked[:limit]]

    def _result(self, food: dict, score: tuple, term: str) -> dict:
        nutrients = copy.deepcopy(food["nutrients"])
        used = {e["source_id"] for e in nutrients.values() if e["value"] is not None}
        return {
            "id": food["id"],
            "name": food["name"],
            "match": "exact" if score[0] == 0 else "partial",
            "matched_term": term,
            "aliases": list(food.get("aliases", [])),
            "serving": dict(food["serving"]),
            "nutrients": nutrients,
            "sources": {sid: dict(self._sources[sid]) for sid in sorted(used)},
        }
