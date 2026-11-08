import importlib.util
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
    assert normalize("pies") == normalize("pie") == "pie"
    assert normalize("fries") == "fry"


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


def test_match_kind_and_matched_term(table):
    first = table.lookup("Almonds")[0]
    assert (first["match"], first["matched_term"]) == ("exact", "Almonds")
    first = table.lookup("raw almond")[0]
    assert (first["match"], first["matched_term"]) == ("exact", "raw almonds")
    butter = table.lookup("almond")[1]
    assert (butter["id"], butter["match"], butter["matched_term"]) == ("almond-butter", "partial", "Almond butter")
    flour = table.lookup("almond flour")
    assert [(r["id"], r["match"]) for r in flour] == [("almonds", "partial")]
    assert flour[0]["matched_term"] == "Almonds"


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


def test_load_rejects_source_id_on_null_value(tmp_path):
    bad = json.loads(json.dumps(FIXTURE))
    bad["foods"][0]["nutrients"]["oxalate_mg"]["source_id"] = "ox"
    path = tmp_path / "foods.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="null value"):
        FoodTable.load(path)


@pytest.mark.parametrize("query, food_id", [
    ("brown rice flour", "brown-rice"),
    ("miso soup", "chicken-noodle-soup"),
    ("almond flour", "almonds"),
    ("sweet potato fries", "sweet-potato"),
    ("spinach dip", "spinach-raw"),
])
def test_real_table_labels_near_misses_partial(query, food_id):
    results = FoodTable.load(ROOT / "data" / "foods.json").lookup(query)
    assert results[0]["id"] == food_id
    assert all(r["match"] == "partial" for r in results)


@pytest.mark.parametrize("query", ["Almonds", "almond"])
def test_real_table_almonds_exact(query):
    first = FoodTable.load(ROOT / "data" / "foods.json").lookup(query)[0]
    assert (first["id"], first["match"]) == ("almonds", "exact")


def load_builder():
    spec = importlib.util.spec_from_file_location("build_foods", ROOT / "data" / "build_foods.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_fdc_item(fdc_id=111, portions=(28.0,), data_type="SR Legacy"):
    def n(number, unit, amount):
        return {"nutrient": {"number": number, "unitName": unit}, "amount": amount}
    return {
        "fdcId": fdc_id,
        "dataType": data_type,
        "description": "Nuts, test",
        "foodPortions": [{"gramWeight": g} for g in portions],
        # Per 100 g. Sugars (269) is missing on purpose.
        "foodNutrients": [n("307", "mg", 1.0), n("306", "mg", 733.0), n("301", "mg", 269.0),
                          n("305", "mg", 481.0), n("203", "g", 21.15), n("205", "g", 21.55),
                          n("999", "g", 5.0)],
    }


FAKE_OXALATE = {
    "source": {"id": "ox", "title": "Oxalate table", "url": "https://x/sheet",
               "linked_from": "https://x/article"},
    "values": [{"food_id": "nut", "oxalate_mg": 122, "source_category": "Very High",
                "source_food": "Nuts", "source_serving": "1 oz"}],
}


def test_build_scales_rounds_and_carries_oxalate_metadata(tmp_path):
    bf = load_builder()
    specs = [bf.food("nut", "Nut", ["nuts"], 111, "1 oz", ounces=1),
             bf.food("seed", "Seed", [], 222, "1 piece", portion=(28.0, 0.5))]
    items = {111: fake_fdc_item(111), 222: fake_fdc_item(222)}
    table = bf.build(items, FAKE_OXALATE, specs)
    nut, seed = table["foods"]
    assert nut["serving"] == {"description": "1 oz", "grams": 28.35}
    values = {k: v["value"] for k, v in nut["nutrients"].items()}
    # 733 mg * 28.35 / 100 = 207.8 -> 208; 21.15 g * 0.2835 = 5.996 -> 6.0
    assert values["potassium_mg"] == 208 and values["sodium_mg"] == 0
    assert values["protein_g"] == 6.0 and values["carbohydrate_g"] == 6.1
    assert nut["nutrients"]["sugars_g"] == {"value": None, "unit": "g", "source_id": None}
    assert nut["nutrients"]["oxalate_mg"] == {
        "value": 122, "unit": "mg", "source_id": "ox", "source_category": "Very High",
        "source_food": "Nuts", "source_serving": "1 oz",
    }
    assert seed["serving"]["grams"] == 14
    assert seed["nutrients"]["oxalate_mg"] == {"value": None, "unit": "mg", "source_id": None}
    assert table["sources"]["ox"] == {
        "title": "Oxalate table", "url": "https://x/sheet", "linked_from": "https://x/article",
    }
    assert table["sources"]["fdc-111"] == {
        "title": "USDA FoodData Central: Nuts, test (FDC ID 111)",
        "url": "https://fdc.nal.usda.gov/food-details/111/nutrients",
    }
    path = tmp_path / "foods.json"
    path.write_text(json.dumps(table))
    assert FoodTable.load(path).lookup("nuts")[0]["id"] == "nut"


def test_build_rejects_unlisted_portion_and_wrong_data_type():
    bf = load_builder()
    spec = [bf.food("seed", "Seed", [], 222, "1 piece", portion=(30.0, 1))]
    with pytest.raises(SystemExit, match="no FDC portion"):
        bf.build({222: fake_fdc_item(222)}, FAKE_OXALATE | {"values": []}, spec)
    spec = [bf.food("seed", "Seed", [], 222, "1 oz", ounces=1)]
    with pytest.raises(SystemExit, match="data type"):
        bf.build({222: fake_fdc_item(222, data_type="Branded")}, FAKE_OXALATE | {"values": []}, spec)


def test_sr_legacy_csv_gives_the_api_item_shape(tmp_path):
    bf = load_builder()
    (tmp_path / "food.csv").write_text(
        '"fdc_id","data_type","description","food_category_id","publication_date"\n'
        '"111","sr_legacy_food","Nuts, test","12","2019-04-01"\n'
        '"999","sr_legacy_food","Other","12","2019-04-01"\n'
    )
    (tmp_path / "food_portion.csv").write_text(
        '"id","fdc_id","seq_num","amount","measure_unit_id","portion_description","modifier",'
        '"gram_weight","data_points","footnote","min_year_acquired"\n'
        '"1","111","1","1","9999","","oz","28.35","","",""\n'
    )
    (tmp_path / "nutrient.csv").write_text(
        '"id","name","unit_name","nutrient_nbr","rank"\n"1092","Potassium, K","MG","306","5700.0"\n'
    )
    (tmp_path / "food_nutrient.csv").write_text(
        '"id","fdc_id","nutrient_id","amount","data_points","derivation_id","min","max","median",'
        '"footnote","min_year_acquired"\n"5","111","1092","733.0","","","","","","",""\n'
    )
    items = bf.load_sr_legacy_csv(tmp_path, [111])
    assert items == {111: {
        "fdcId": 111, "dataType": "SR Legacy", "description": "Nuts, test",
        "foodPortions": [{"gramWeight": 28.35}],
        "foodNutrients": [{"nutrient": {"number": "306", "unitName": "MG"}, "amount": 733.0}],
    }}
    spec = [bf.food("nut", "Nut", [], 111, "1 oz", portion=(28.35, 1))]
    table = bf.build(items, FAKE_OXALATE | {"values": []}, spec)
    assert table["foods"][0]["nutrients"]["potassium_mg"]["value"] == 208
    with pytest.raises(SystemExit, match="does not contain"):
        bf.load_sr_legacy_csv(tmp_path, [111, 222])


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
    for food in foods:
        oxalate = food["nutrients"]["oxalate_mg"]
        if oxalate["value"] is None:
            assert set(oxalate) == {"value", "unit", "source_id"}, food["id"]
        else:
            assert oxalate["source_category"] and oxalate["source_food"], food["id"]
            assert oxalate["source_serving"], food["id"]
            assert sources[oxalate["source_id"]]["linked_from"].startswith(
                "https://kidneystones.uchicago.edu/"
            )
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
