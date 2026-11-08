import json
from pathlib import Path

import pytest

from foods import FoodTable, normalize

ROOT = Path(__file__).resolve().parents[1]
NUTRIENT_KEYS = {
    "sodium_mg",
    "potassium_mg",
    "calcium_mg",
    "phosphorus_mg",
    "protein_g",
    "carbohydrate_g",
    "sugars_g",
    "oxalate_mg",
}


def nutrients(source_id, oxalate=None, oxalate_source=None):
    out = {key: {"value": 1.0, "unit": key.rsplit("_", 1)[1], "source_id": source_id}
           for key in NUTRIENT_KEYS}
    out["oxalate_mg"] = {"value": oxalate, "unit": "mg", "source_id": oxalate_source}
    return out


FIXTURE = {
    "foods": [
        {
            "id": "almond-butter",
            "name": "Almond butter",
            "aliases": [],
            "serving": {"description": "1 tablespoon", "grams": 16},
            "nutrients": nutrients("fdc-2"),
        },
        {
            "id": "almonds",
            "name": "Almonds",
            "aliases": ["raw almonds"],
            "serving": {"description": "1 oz (22 almonds)", "grams": 28.35},
            "nutrients": nutrients("fdc-1", oxalate=122, oxalate_source="ox"),
        },
        {
            "id": "black-tea",
            "name": "Black tea",
            "aliases": ["tea", "hot tea"],
            "serving": {"description": "1 cup (8 fl oz)", "grams": 237},
            "nutrients": nutrients("fdc-3", oxalate=14, oxalate_source="ox"),
        },
        {
            "id": "steak",
            "name": "Beef steak",
            "aliases": ["steak"],
            "serving": {"description": "3 oz", "grams": 85},
            "nutrients": nutrients("fdc-4"),
        },
        {
            "id": "strawberries",
            "name": "Strawberries",
            "aliases": [],
            "serving": {"description": "1/2 cup", "grams": 72},
            "nutrients": nutrients("fdc-5", oxalate=2, oxalate_source="ox"),
        },
    ],
    "sources": {
        "fdc-1": {"title": "USDA FoodData Central: Nuts, almonds (FDC ID 1)", "url": "https://x/1"},
        "fdc-2": {"title": "USDA FoodData Central: Almond butter (FDC ID 2)", "url": "https://x/2"},
        "fdc-3": {"title": "USDA FoodData Central: Tea (FDC ID 3)", "url": "https://x/3"},
        "fdc-4": {"title": "USDA FoodData Central: Steak (FDC ID 4)", "url": "https://x/4"},
        "fdc-5": {"title": "USDA FoodData Central: Strawberries (FDC ID 5)", "url": "https://x/5"},
        "ox": {"title": "Oxalate table", "url": "https://x/ox"},
    },
}


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "foods.json"
    path.write_text(json.dumps(FIXTURE))
    return FoodTable.load(path)


def ids(results):
    return [r["id"] for r in results]


def test_normalize_rules():
    assert normalize("Almonds!") == "almond"
    assert normalize("  POTATOES, baked ") == "potato baked"
    assert normalize("Strawberries") == normalize("strawberry")
    assert normalize("grapes") == normalize("grape")
    assert normalize("peaches") == normalize("peach")
    assert normalize("glass") == "glass"


def test_lookup_exact_case_insensitive(table):
    assert ids(table.lookup("ALMONDS"))[0] == "almonds"
    assert ids(table.lookup("black TEA"))[0] == "black-tea"
    assert ids(table.lookup("Hot Tea"))[0] == "black-tea"


def test_lookup_plural_and_alias(table):
    assert ids(table.lookup("Almonds"))[0] == "almonds"
    assert ids(table.lookup("almond"))[0] == "almonds"
    assert ids(table.lookup("raw almond"))[0] == "almonds"
    assert ids(table.lookup("almond butter"))[0] == "almond-butter"
    assert ids(table.lookup("strawberry")) == ["strawberries"]


def test_lookup_contains_both_directions(table):
    # Query inside a name: "almond" also finds almond butter, after the exact match.
    assert ids(table.lookup("almond")) == ["almonds", "almond-butter"]
    # Name inside a longer query.
    assert ids(table.lookup("a handful of almonds"))[0] == "almonds"


def test_lookup_matches_whole_words_only(table):
    # "tea" is a substring of "steak" but not a word in it.
    assert "steak" not in ids(table.lookup("tea"))
    assert "black-tea" not in ids(table.lookup("steak"))


def test_lookup_respects_limit(table):
    assert len(table.lookup("almond", limit=1)) == 1


def test_lookup_unknown_returns_empty(table):
    assert table.lookup("dragonfruit") == []
    assert table.lookup("") == []
    assert table.lookup("  ?! ") == []


def test_result_carries_serving_nutrients_and_used_sources_only(table):
    result = table.lookup("almonds")[0]
    assert result["name"] == "Almonds"
    assert result["serving"] == {"description": "1 oz (22 almonds)", "grams": 28.35}
    assert result["nutrients"]["oxalate_mg"]["value"] == 122
    assert set(result["sources"]) == {"fdc-1", "ox"}
    assert result["sources"]["ox"]["url"] == "https://x/ox"


def test_null_value_preserved(table):
    result = table.lookup("almond butter")[0]
    oxalate = result["nutrients"]["oxalate_mg"]
    assert oxalate["value"] is None
    assert oxalate["source_id"] is None
    assert set(result["sources"]) == {"fdc-2"}


def test_results_are_copies(table):
    table.lookup("almonds")[0]["nutrients"]["oxalate_mg"]["value"] = 0
    assert table.lookup("almonds")[0]["nutrients"]["oxalate_mg"]["value"] == 122


def test_load_rejects_unknown_source_id(tmp_path):
    bad = json.loads(json.dumps(FIXTURE))
    bad["foods"][1]["nutrients"]["sodium_mg"]["source_id"] = "missing"
    path = tmp_path / "foods.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing"):
        FoodTable.load(path)


def test_real_foods_json_valid():
    data = json.loads((ROOT / "data" / "foods.json").read_text())
    foods, sources = data["foods"], data["sources"]
    assert 50 <= len(foods) <= 100
    assert len({f["id"] for f in foods}) == len(foods)
    for food in foods:
        assert food["name"] and isinstance(food["aliases"], list)
        assert food["serving"]["description"]
        assert food["serving"]["grams"] > 0
        assert set(food["nutrients"]) == NUTRIENT_KEYS, food["id"]
        for key, entry in food["nutrients"].items():
            if entry["value"] is None:
                continue
            assert entry["source_id"] in sources, (food["id"], key)
            assert entry["unit"] == key.rsplit("_", 1)[1]
    for source in sources.values():
        assert source["title"] and source["url"].startswith("https://")
    text = (ROOT / "data" / "foods.json").read_text()
    assert chr(0x2014) not in text

    table = FoodTable.load(ROOT / "data" / "foods.json")
    required = [
        "almonds", "almond butter", "peanuts", "cashews", "walnuts", "spinach",
        "rhubarb", "beets", "sweet potato", "potato chips", "black tea",
        "dark chocolate", "banana", "orange juice", "canned soup", "deli meat",
        "cheddar cheese", "milk", "white bread", "regular soda",
    ]
    for query in required:
        assert table.lookup(query), query
    almonds = table.lookup("almonds")[0]
    assert almonds["id"] == "almonds"
    assert almonds["nutrients"]["oxalate_mg"]["value"] is not None
