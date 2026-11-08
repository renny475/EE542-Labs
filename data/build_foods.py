"""Build data/foods.json from USDA FoodData Central plus data/oxalate.json.

Usage:
    FDC_API_KEY=... python data/build_foods.py [--cache DIR]
    python data/build_foods.py --sr-legacy-csv DIR

Each food below names one FoodData Central item (SR Legacy unless noted) and a
patient-facing serving. The serving's gram weight comes from a portion that
FoodData Central itself lists for that item (checked at build time), or from a
weight serving (1 oz = 28.35 g). Per-100 g nutrient values are scaled to the
serving and rounded: mg to whole numbers, g to one decimal (serving grams
keep two decimals).

FoodData Central has no oxalate data. Oxalate values come from data/oxalate.json,
where each value is recorded with its source, the food label, serving, row and
category word as printed in that source, and any matching note. Each oxalate
entry in foods.json carries `source_category`, `source_food` and
`source_serving` so a patient can find the row. The category word is the
source table's own label; it is not a mg cutoff and is never recomputed here.
Foods not listed there get oxalate null ("unknown") and no category.

Items are fetched with POST /v1/foods, 20 per request, to stay inside the
DEMO_KEY rate limit. `--cache DIR` stores raw responses so reruns are free.

`--sr-legacy-csv DIR` reads the same items offline from FoodData Central's own
SR Legacy CSV download (FoodData_Central_sr_legacy_food_csv_2018-04.zip from
https://fdc.nal.usda.gov/download-datasets, unzipped), for when the API rate
limit is used up. SR Legacy is a final, unchanging release, so both paths give
the same numbers.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
FDC_API = "https://api.nal.usda.gov/fdc/v1/foods"
BATCH = 20
OZ = 28.35

# key -> (FDC nutrient number, unit)
NUTRIENTS = {
    "sodium_mg": ("307", "mg"),
    "potassium_mg": ("306", "mg"),
    "calcium_mg": ("301", "mg"),
    "phosphorus_mg": ("305", "mg"),
    "protein_g": ("203", "g"),
    "carbohydrate_g": ("205", "g"),
    "sugars_g": ("269", "g"),
}


def food(id, name, aliases, fdc_id, serving, portion=None, ounces=None):
    """portion=(gram_weight, count): `count` times an FDC-listed portion of
    `gram_weight` g. ounces=n: a weight serving of n oz."""
    assert (portion is None) != (ounces is None), id
    return {
        "id": id,
        "name": name,
        "aliases": aliases,
        "fdc_id": fdc_id,
        "serving": serving,
        "portion": portion,
        "ounces": ounces,
    }


FOODS = [
    # Nuts, seeds and nut butters
    food("almonds", "Almonds", ["almond", "raw almonds"], 170567,
         "1 oz (about 23 almonds)", ounces=1),
    food("almond-butter", "Almond butter", ["almond nut butter"], 168588,
         "1 tablespoon", portion=(16, 1)),
    food("almond-milk", "Almond milk, unsweetened", ["almond milk", "almond beverage"], 174832,
         "1 cup", portion=(262, 1)),
    food("peanuts", "Peanuts, dry-roasted, salted", ["peanuts", "peanut", "roasted peanuts"], 174262,
         "1 oz", ounces=1),
    food("peanut-butter", "Peanut butter, smooth", ["peanut butter"], 174266,
         "1 tablespoon", portion=(32, 0.5)),
    food("cashews", "Cashews, dry-roasted, salted", ["cashews", "cashew", "cashew nuts"], 169421,
         "1 oz (about 18 cashews)", ounces=1),
    food("walnuts", "Walnuts", ["walnut", "english walnuts"], 170187,
         "7 walnuts, shelled (about 1 oz)", portion=(28, 1)),
    food("pecans", "Pecans", ["pecan"], 170182, "1 oz", ounces=1),
    food("pistachios", "Pistachios, dry-roasted, salted", ["pistachios", "pistachio"], 169426,
         "1 oz (about 49 kernels)", ounces=1),
    food("mixed-nuts", "Mixed nuts with peanuts, dry-roasted, salted", ["mixed nuts"], 168599,
         "1 oz", ounces=1),
    food("tahini", "Tahini (sesame seed butter)", ["sesame butter", "sesame paste"], 170189,
         "1 tablespoon", portion=(15, 1)),
    # Vegetables
    food("spinach-raw", "Spinach, raw", ["spinach", "raw spinach", "spinach salad"], 168462,
         "1 cup raw", portion=(30, 1)),
    food("spinach-cooked", "Spinach, cooked", ["cooked spinach", "boiled spinach"], 168463,
         "1/2 cup cooked", portion=(180, 0.5)),
    food("rhubarb", "Rhubarb", ["rhubarb stalks"], 167758,
         "1/2 cup diced, raw", portion=(122, 0.5)),
    food("beets", "Beets, cooked", ["beet", "beetroot", "cooked beets"], 169146,
         "1/2 cup sliced, cooked", portion=(85, 1)),
    food("sweet-potato", "Sweet potato, baked", ["sweet potato", "sweet potatoes", "baked sweet potato"],
         168483, "1 cup, baked (flesh only)", portion=(200, 1)),
    food("baked-potato", "Baked potato with skin", ["potato", "potatoes", "baked potato"], 170093,
         "1 medium potato", portion=(173, 1)),
    food("mashed-potatoes", "Mashed potatoes", ["mashed potato"], 168555,
         "1 cup (made with whole milk and butter)", portion=(210, 1)),
    food("broccoli", "Broccoli, raw", ["broccoli", "raw broccoli"], 170379,
         "1/2 cup chopped, raw", portion=(44, 1)),
    food("carrots-cooked", "Carrots, cooked", ["carrots", "carrot", "cooked carrots"], 170394,
         "1/2 cup sliced, cooked", portion=(78, 1)),
    food("celery", "Celery, raw", ["celery", "celery sticks"], 169988,
         "1/2 cup chopped, raw", portion=(101, 0.5)),
    food("tomato", "Tomato, raw", ["tomato", "tomatoes"], 170457,
         "1 medium tomato", portion=(123, 1)),
    food("iceberg-lettuce", "Iceberg lettuce", ["lettuce", "iceberg"], 169248,
         "1 cup shredded", portion=(72, 1)),
    food("kidney-beans", "Kidney beans, cooked", ["kidney beans", "red beans"], 175194,
         "1/2 cup cooked", portion=(177, 0.5)),
    food("navy-beans", "Navy beans, cooked", ["navy beans", "white beans"], 173746,
         "1/2 cup cooked", portion=(182, 0.5)),
    food("tomato-sauce", "Tomato sauce, canned", ["tomato sauce", "pasta sauce"], 170054,
         "1/2 cup", portion=(245, 0.5)),
    food("sauerkraut", "Sauerkraut, canned", ["sauerkraut"], 169279,
         "1/2 cup", portion=(142, 0.5)),
    food("dill-pickle", "Dill pickle", ["pickle", "pickles", "dill pickles"], 168558,
         "1 small spear", portion=(35, 1)),
    food("tofu", "Tofu, firm", ["tofu", "bean curd"], 172475,
         "3.5 oz", ounces=3.5),
    # Fruits
    food("banana", "Banana", ["bananas"], 173944, "1 medium banana", portion=(118, 1)),
    food("apple", "Apple", ["apples"], 171688, "1 medium apple", portion=(182, 1)),
    food("orange", "Orange", ["oranges"], 169097, "1 orange", portion=(131, 1)),
    food("avocado", "Avocado", ["avocados"], 171705, "1 avocado", portion=(201, 1)),
    food("grapes", "Grapes", ["grape", "seedless grapes"], 174683,
         "1/2 cup (about 16 grapes)", portion=(151, 0.5)),
    food("strawberries", "Strawberries", ["strawberry"], 167762,
         "1/2 cup whole", portion=(144, 0.5)),
    food("blueberries", "Blueberries", ["blueberry"], 171711, "1/2 cup", portion=(148, 0.5)),
    food("raspberries", "Raspberries", ["raspberry"], 167755, "1 cup", portion=(123, 1)),
    food("raisins", "Raisins", ["raisin"], 168165, "1 oz", ounces=1),
    food("kiwi", "Kiwi", ["kiwifruit", "kiwis"], 168153, "1 kiwi", portion=(69, 1)),
    food("grapefruit", "Grapefruit", ["grapefruits"], 174673, "1/2 grapefruit", portion=(123, 1)),
    # Drinks
    food("black-tea", "Black tea, brewed", ["black tea", "tea", "hot tea", "iced tea, unsweetened"],
         173227, "1 cup (8 fl oz)", portion=(237, 1)),
    food("green-tea", "Green tea, brewed", ["green tea"], 171917, "1 cup", portion=(245, 1)),
    food("coffee", "Coffee, brewed", ["coffee", "black coffee"], 171890,
         "1 cup (8 fl oz)", portion=(237, 1)),
    food("orange-juice", "Orange juice", ["orange juice", "oj"], 169100,
         "1 cup (8 fl oz)", portion=(249, 1)),
    food("apple-juice", "Apple juice", ["apple juice"], 173933, "6 fl oz", portion=(31, 6)),
    food("tomato-juice", "Tomato juice, canned, salted", ["tomato juice"], 170458,
         "1 cup", portion=(243, 1)),
    food("lemonade", "Lemonade (from frozen concentrate)", ["lemonade"], 173217,
         "1 cup (8 fl oz)", portion=(247, 1)),
    food("regular-soda", "Regular soda (cola)", ["regular soda", "soda", "cola", "pop", "soft drink"],
         174852, "8 fl oz", portion=(30.7, 8)),
    food("diet-soda", "Diet soda (cola)", ["diet soda", "diet cola", "diet pop"], 174850,
         "8 fl oz", portion=(29.6, 8)),
    food("sports-drink", "Sports drink (Gatorade)", ["sports drink", "gatorade"], 173660,
         "8 fl oz", portion=(244, 1)),
    food("beer", "Beer, regular", ["beer"], 168746, "1 can (12 fl oz)", portion=(356, 1)),
    food("soy-milk", "Soy milk", ["soymilk", "soy drink"], 172446, "1 cup", portion=(243, 1)),
    # Dairy and eggs
    food("whole-milk", "Milk, whole", ["milk", "whole milk"], 171265, "1 cup", portion=(244, 1)),
    food("skim-milk", "Milk, fat free (skim)", ["milk", "skim milk", "nonfat milk", "fat free milk"], 171269,
         "1 cup", portion=(245, 1)),
    food("chocolate-milk", "Chocolate milk, reduced fat", ["chocolate milk"], 170880,
         "1 cup", portion=(250, 1)),
    food("yogurt", "Yogurt, plain, low fat", ["yogurt", "plain yogurt", "yoghurt"], 170886,
         "1 cup (8 fl oz)", portion=(245, 1)),
    food("cheddar-cheese", "Cheddar cheese", ["cheddar"], 173414, "1 slice (1 oz)", portion=(28, 1)),
    food("american-cheese", "American cheese", ["processed cheese", "cheese slice"], 171290,
         "1 slice (3/4 oz)", portion=(21, 1)),
    food("mozzarella", "Mozzarella cheese", ["mozzarella"], 170845, "1 oz", ounces=1),
    food("cottage-cheese", "Cottage cheese", ["cottage cheese"], 172179, "1/2 cup", portion=(113, 1)),
    food("ice-cream", "Vanilla ice cream", ["ice cream"], 167575, "1/2 cup", portion=(66, 1)),
    food("egg", "Egg", ["eggs"], 171287, "1 medium egg", portion=(44, 1)),
    # Meat and fish
    food("chicken-breast", "Chicken breast, roasted", ["chicken", "chicken breast"], 171477,
         "3 oz cooked", ounces=3),
    food("ground-beef", "Ground beef (85% lean), cooked", ["ground beef", "hamburger", "beef"], 174032,
         "3 oz cooked", portion=(85, 1)),
    food("salmon", "Salmon, cooked", ["salmon"], 175168, "4 oz cooked", ounces=4),
    food("tuna-canned", "Tuna, canned in water", ["tuna", "canned tuna"], 173709,
         "3.5 oz, drained", ounces=3.5),
    food("bacon", "Bacon, cooked", ["bacon"], 168322, "2 slices", portion=(11.5, 2)),
    food("hot-dog", "Hot dog (beef frankfurter)", ["hot dog", "hot dogs", "frankfurter", "wiener"],
         173862, "1 hot dog (no bun)", portion=(49, 1)),
    food("deli-turkey", "Deli turkey breast, sliced", ["deli meat", "deli turkey", "sliced turkey",
                                                       "lunch meat", "cold cuts"],
         172941, "2 oz (about 3-4 slices)", ounces=2),
    food("deli-ham", "Deli ham, sliced", ["deli meat", "ham", "sliced ham", "lunch meat"], 173864,
         "2 slices (2 oz)", portion=(28, 2)),
    food("bologna", "Bologna (beef)", ["bologna", "deli meat"], 172012, "1 slice", portion=(30, 1)),
    # Breads, grains and cereals
    food("white-bread", "White bread", ["bread", "white bread"], 174924, "1 slice", portion=(29, 1)),
    food("whole-wheat-bread", "Whole wheat bread", ["wheat bread", "whole wheat bread"], 172688,
         "1 slice", portion=(32, 1)),
    food("bagel", "Bagel, plain", ["bagel", "bagels"], 174899, "1 bagel", portion=(99, 1)),
    food("corn-tortilla", "Corn tortilla", ["corn tortillas"], 175036, "1 tortilla", portion=(24, 1)),
    food("flour-tortilla", "Flour tortilla", ["flour tortillas", "tortilla"], 173242,
         "1 medium tortilla (6 inch)", portion=(32, 1)),
    food("white-rice", "White rice, cooked", ["rice", "white rice"], 168878, "1 cup cooked",
         portion=(158, 1)),
    food("brown-rice", "Brown rice, cooked", ["brown rice"], 169704, "1 cup cooked", portion=(202, 1)),
    food("spaghetti", "Spaghetti, cooked", ["pasta", "spaghetti", "noodles"], 169737,
         "1 cup cooked", portion=(124, 1)),
    food("oatmeal", "Oatmeal, cooked with water and salt", ["oatmeal", "oats", "porridge"], 171675,
         "1 cup cooked", portion=(234, 1)),
    food("cheerios", "Cheerios cereal", ["cheerios"], 173884, "1 cup", portion=(28, 1)),
    food("raisin-bran", "Raisin bran cereal", ["raisin bran", "bran cereal"], 171650, "1 cup",
         portion=(59, 1)),
    # Snacks and sweets
    food("potato-chips", "Potato chips, salted", ["potato chips", "chips", "crisps"], 169677,
         "1 oz (about 15-20 chips)", ounces=1),
    food("tortilla-chips", "Tortilla chips, salted", ["tortilla chips", "corn chips"], 167558,
         "1 oz", ounces=1),
    food("pretzels", "Pretzels, hard, salted", ["pretzel"], 167555, "1 oz", ounces=1),
    food("saltine-crackers", "Saltine crackers", ["saltines", "saltine", "crackers", "soda crackers"],
         172746, "1 cracker", portion=(3, 1)),
    food("popcorn", "Popcorn, air-popped", ["popcorn"], 167959, "1 cup", portion=(8, 1)),
    food("french-fries", "French fries (fast food)", ["french fries", "fries", "chips (fries)"], 170698,
         "4 oz", ounces=4),
    food("chocolate-chip-cookie", "Chocolate chip cookie", ["cookie", "cookies",
                                                            "chocolate chip cookies"],
         172820, "1 medium cookie", portion=(10, 1)),
    food("milk-chocolate", "Milk chocolate", ["chocolate", "milk chocolate", "chocolate bar"], 167587,
         "1 oz", ounces=1),
    food("dark-chocolate", "Dark chocolate", ["dark chocolate"], 170274, "1 oz", ounces=1),
    food("hummus", "Hummus", ["houmous", "hummous"], 174289, "1 tablespoon", portion=(15, 1)),
    # Soups and condiments
    food("chicken-noodle-soup", "Chicken noodle soup, canned", ["canned soup", "chicken noodle soup",
                                                                "chicken soup", "soup"],
         172909, "1 cup (prepared with water)", portion=(248, 1)),
    food("tomato-soup", "Tomato soup, canned", ["canned soup", "tomato soup"], 171176,
         "1 cup (prepared with water)", portion=(248, 1)),
    food("soy-sauce", "Soy sauce", ["soy sauce", "shoyu"], 174277, "1 tablespoon", portion=(16, 1)),
    food("ketchup", "Ketchup", ["catsup"], 168556, "1 packet", portion=(9, 1)),
    food("table-salt", "Table salt", ["salt"], 173468, "1 teaspoon", portion=(6, 1)),
    food("sugar", "Sugar, white", ["sugar", "granulated sugar", "table sugar"], 169655,
         "1 teaspoon", portion=(4.2, 1)),
]


def serving_grams(spec):
    if spec["ounces"] is not None:
        grams = spec["ounces"] * OZ
    else:
        gram_weight, count = spec["portion"]
        grams = gram_weight * count
    grams = round(grams, 2)
    return int(grams) if grams.is_integer() else grams


def check_portion(spec, item):
    """The serving's gram weight must come from a portion FDC lists for this item."""
    if spec["portion"] is None:
        return
    gram_weight = spec["portion"][0]
    listed = [p.get("gramWeight") for p in item.get("foodPortions", [])]
    if not any(g is not None and abs(g - gram_weight) < 0.01 for g in listed):
        raise SystemExit(
            f"{spec['id']}: no FDC portion of {gram_weight} g for {spec['fdc_id']} (has {listed})"
        )


def fetch(ids, api_key, cache):
    """Return {fdc_id: item}, fetched 20 at a time."""
    items = {}
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        cached = cache / f"fdc-{'-'.join(map(str, chunk))}.json" if cache else None
        if cached and cached.exists():
            data = json.loads(cached.read_text())
        else:
            resp = httpx.post(
                FDC_API, params={"api_key": api_key}, json={"fdcIds": chunk, "format": "full"},
                timeout=60,
            )
            if resp.status_code != 200:
                raise SystemExit(f"FoodData Central returned {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            print(f"fetched {len(data)} items, rate limit remaining "
                  f"{resp.headers.get('x-ratelimit-remaining')}", file=sys.stderr)
            if cached:
                cached.write_text(json.dumps(data))
        for item in data:
            items[item["fdcId"]] = item
    missing = set(ids) - set(items)
    if missing:
        raise SystemExit(f"FoodData Central did not return: {sorted(missing)}")
    return items


def load_sr_legacy_csv(directory, ids):
    """Return {fdc_id: item} in the API's "full" shape, from the SR Legacy CSVs."""
    directory = Path(directory)
    wanted = {str(i) for i in ids}

    def rows(name):
        with open(directory / name, newline="") as f:
            yield from csv.DictReader(f)

    items = {}
    for row in rows("food.csv"):
        if row["fdc_id"] in wanted:
            items[int(row["fdc_id"])] = {
                "fdcId": int(row["fdc_id"]),
                "dataType": "SR Legacy" if row["data_type"] == "sr_legacy_food" else row["data_type"],
                "description": row["description"],
                "foodPortions": [],
                "foodNutrients": [],
            }
    for row in rows("food_portion.csv"):
        if row["fdc_id"] in wanted:
            items[int(row["fdc_id"])]["foodPortions"].append({"gramWeight": float(row["gram_weight"])})
    nutrients = {
        row["id"]: {"number": row["nutrient_nbr"], "unitName": row["unit_name"]}
        for row in rows("nutrient.csv")
    }
    for row in rows("food_nutrient.csv"):
        if row["fdc_id"] in wanted and row["amount"] != "":
            items[int(row["fdc_id"])]["foodNutrients"].append(
                {"nutrient": nutrients[row["nutrient_id"]], "amount": float(row["amount"])}
            )
    missing = set(ids) - set(items)
    if missing:
        raise SystemExit(f"SR Legacy CSV does not contain: {sorted(missing)}")
    return items


def per_100g(item):
    out = {}
    for entry in item.get("foodNutrients", []):
        nutrient = entry.get("nutrient", {})
        if "amount" in entry:
            out[nutrient.get("number")] = (entry["amount"], nutrient.get("unitName"))
    return out


def scaled(amount, grams, unit):
    value = amount * grams / 100
    return round(value) if unit == "mg" else round(value, 1)


def build(items, oxalate, specs=None):
    specs = FOODS if specs is None else specs
    ox_source = oxalate["source"]
    ox_values = {v["food_id"]: v for v in oxalate["values"]}
    unknown = set(ox_values) - {spec["id"] for spec in specs}
    if unknown:
        raise SystemExit(f"oxalate.json names foods not in the table: {sorted(unknown)}")
    sources = {
        ox_source["id"]: {"title": ox_source["title"], "url": ox_source["url"],
                          "linked_from": ox_source["linked_from"]},
    }
    foods = []
    for spec in specs:
        item = items[spec["fdc_id"]]
        if item.get("dataType") not in ("SR Legacy", "Foundation"):
            raise SystemExit(f"{spec['id']}: unexpected data type {item.get('dataType')}")
        check_portion(spec, item)
        grams = serving_grams(spec)
        source_id = f"fdc-{spec['fdc_id']}"
        # This public page is an Angular app: the server answers deep links with
        # HTTP 404 plus the app shell, and the page renders in a browser. Scripts
        # checking these URLs see 404; open them in a browser instead.
        sources[source_id] = {
            "title": f"USDA FoodData Central: {item['description']} (FDC ID {spec['fdc_id']})",
            "url": f"https://fdc.nal.usda.gov/food-details/{spec['fdc_id']}/nutrients",
        }
        raw = per_100g(item)
        nutrients = {}
        for key, (number, unit) in NUTRIENTS.items():
            if number in raw:
                amount, fdc_unit = raw[number]
                if fdc_unit.lower() != unit:
                    raise SystemExit(f"{spec['id']}: {key} unit {fdc_unit}, expected {unit}")
                nutrients[key] = {"value": scaled(amount, grams, unit), "unit": unit,
                                  "source_id": source_id}
            else:
                nutrients[key] = {"value": None, "unit": unit, "source_id": None}
        ox = ox_values.get(spec["id"])
        if ox is None:
            nutrients["oxalate_mg"] = {"value": None, "unit": "mg", "source_id": None}
        else:
            nutrients["oxalate_mg"] = {
                "value": ox["oxalate_mg"], "unit": "mg", "source_id": ox_source["id"],
                "source_category": ox["source_category"], "source_food": ox["source_food"],
                "source_serving": ox["source_serving"],
            }
        foods.append({
            "id": spec["id"],
            "name": spec["name"],
            "aliases": spec["aliases"],
            "serving": {"description": spec["serving"], "grams": grams},
            "nutrients": nutrients,
        })
    used = {e["source_id"] for f in foods for e in f["nutrients"].values() if e["source_id"]}
    return {"foods": foods, "sources": {k: v for k, v in sources.items() if k in used}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, help="directory for raw FDC responses")
    parser.add_argument("--sr-legacy-csv", type=Path,
                        help="read items from the unzipped SR Legacy CSV download instead of the API")
    parser.add_argument("--out", type=Path, default=HERE / "foods.json")
    args = parser.parse_args()
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    ids = [spec["fdc_id"] for spec in FOODS]
    if len(set(ids)) != len(ids) or len({s["id"] for s in FOODS}) != len(FOODS):
        raise SystemExit("duplicate FDC ID or food id in FOODS")
    oxalate = json.loads((HERE / "oxalate.json").read_text())
    if args.sr_legacy_csv:
        items = load_sr_legacy_csv(args.sr_legacy_csv, ids)
    else:
        items = fetch(ids, os.environ.get("FDC_API_KEY") or "DEMO_KEY", args.cache)
    table = build(items, oxalate)
    args.out.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(table['foods'])} foods to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
