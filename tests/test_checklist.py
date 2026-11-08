from datetime import date

from checklist import build_checklist
from models import ExtractedItem

ANCHOR = date(2026, 11, 5)


def item(category, details=None, text="Item text", status="active", id=1):
    return ExtractedItem(
        id=id, category=category, text=text, source_quote="q", details=details, status=status
    )


def med(**details):
    base = {"name": "Furosemide", "dose": "40 mg"}
    base.update(details)
    return item("medication", base)


def slots(specs):
    return [s.slot for s in specs]


def test_twice_daily_med_gives_morning_evening():
    specs = build_checklist([med(times_per_day=2)], ANCHOR)
    assert slots(specs) == ["morning", "evening"]
    assert specs[0].label == "Furosemide 40 mg"
    assert specs[0].extracted_item_id == 1
    assert specs[0].start_date == ANCHOR
    assert specs[0].end_date is None
    assert not specs[0].one_time and not specs[0].approximate and not specs[0].undated


def test_count_mapping_for_other_values():
    assert slots(build_checklist([med(times_per_day=1)], ANCHOR)) == ["morning"]
    assert slots(build_checklist([med(times_per_day=3)], ANCHOR)) == ["morning", "midday", "evening"]
    assert slots(build_checklist([med(times_per_day=6)], ANCHOR)) == [
        "morning", "midday", "evening", "bedtime"
    ]


def test_interval_8h_gives_three_slots():
    assert slots(build_checklist([med(interval_hours=8)], ANCHOR)) == [
        "morning", "midday", "evening"
    ]


def test_times_of_day_wins_over_count():
    specs = build_checklist([med(times_of_day=["bedtime"], times_per_day=3)], ANCHOR)
    assert slots(specs) == ["bedtime"]


def test_as_needed_med_skipped():
    assert build_checklist([med(as_needed=True, times_per_day=2)], ANCHOR) == []


def test_as_needed_string_is_coerced():
    assert build_checklist([med(as_needed="true", times_per_day=2)], ANCHOR) == []
    assert slots(build_checklist([med(as_needed="false", times_per_day=1)], ANCHOR)) == ["morning"]


def test_duration_sets_end_date():
    specs = build_checklist([med(times_per_day=1, duration_days=7)], ANCHOR)
    assert specs[0].end_date == date(2026, 11, 11)


def test_label_includes_instructions():
    specs = build_checklist([med(times_per_day=1, instructions="with food")], ANCHOR)
    assert specs[0].label == "Furosemide 40 mg - with food"


def test_label_falls_back_to_item_text_without_name():
    it = item("medication", {"times_per_day": 1}, text="Take the blue pill")
    assert build_checklist([it], ANCHOR)[0].label == "Take the blue pill"


def test_no_schedule_gives_anytime():
    assert slots(build_checklist([med()], ANCHOR)) == ["anytime"]
    assert slots(build_checklist([item("medication", None)], ANCHOR)) == ["anytime"]


def test_monitoring_and_wound_care_use_item_text():
    mon = item("monitoring", {"times_per_day": 1}, text="Weigh yourself", id=2)
    wound = item("wound_or_device_care", {"times_of_day": ["evening"]}, text="Change dressing", id=3)
    specs = build_checklist([mon, wound], ANCHOR)
    assert [(s.label, s.slot) for s in specs] == [
        ("Weigh yourself", "morning"),
        ("Change dressing", "evening"),
    ]


def test_follow_up_with_date_is_one_time():
    it = item("follow_up", {"date": "2026-11-20"}, text="See cardiology")
    (spec,) = build_checklist([it], ANCHOR)
    assert spec.one_time and spec.start_date == date(2026, 11, 20)
    assert spec.end_date == date(2026, 11, 20)
    assert not spec.approximate and not spec.undated
    assert spec.label == "See cardiology" and spec.slot == "anytime"


def test_follow_up_relative_is_approximate():
    it = item("follow_up", {"days_after_discharge": 14}, text="See your doctor")
    (spec,) = build_checklist([it], ANCHOR)
    assert spec.one_time and spec.approximate
    assert spec.start_date == date(2026, 11, 19)


def test_follow_up_without_timing_is_undated():
    (spec,) = build_checklist([item("follow_up", {}, text="Schedule a visit")], ANCHOR)
    assert spec.undated and not spec.one_time


def test_diet_items_not_in_checklist():
    cats = ["diet", "activity", "warning_signs", "other"]
    items = [item(c, {"times_per_day": 2}, id=n) for n, c in enumerate(cats)]
    assert build_checklist(items, ANCHOR) == []


def test_rejected_items_excluded():
    assert build_checklist([item("medication", {"times_per_day": 1}, status="rejected")], ANCHOR) == []


def test_garbage_details_never_raise():
    bad = [
        {"times_per_day": "twice", "interval_hours": "x", "duration_days": "soon"},
        {"times_per_day": None, "interval_hours": None, "duration_days": None},
        {"times_of_day": "morning"},
        {"times_of_day": 5},
        {"times_of_day": ["lunch", None, 3]},
        {"times_per_day": [2], "duration_days": {"a": 1}},
        {"duration_days": float("inf"), "times_per_day": float("nan")},
        {"duration_days": 10**12},
        {"as_needed": [1]},
    ]
    for d in bad:
        specs = build_checklist([med(**d), item("follow_up", d), item("monitoring", d)], ANCHOR)
        assert all(s.slot in {"morning", "midday", "evening", "bedtime", "anytime"} for s in specs)
    assert build_checklist([item("medication", "not a dict")], ANCHOR)[0].slot == "anytime"


def test_unknown_slots_fall_back_to_anytime():
    assert slots(build_checklist([med(times_of_day=["lunch", "noon"])], ANCHOR)) == ["anytime"]


def test_unknown_slots_ignored_when_some_valid():
    assert slots(build_checklist([med(times_of_day=["lunch", "evening"])], ANCHOR)) == ["evening"]


def test_numeric_strings_and_floats_are_coerced():
    assert slots(build_checklist([med(times_per_day="2")], ANCHOR)) == ["morning", "evening"]
    assert slots(build_checklist([med(times_per_day=2.0)], ANCHOR)) == ["morning", "evening"]
    assert build_checklist([med(times_per_day=1, duration_days="3")], ANCHOR)[0].end_date == date(
        2026, 11, 7
    )


def test_bad_duration_means_no_end_date():
    for bad in (0, -3, "abc", None, 10**12, True):
        assert build_checklist([med(times_per_day=1, duration_days=bad)], ANCHOR)[0].end_date is None


def test_bad_interval_is_ignored():
    for bad in (0, -8, 25, 48, "x"):
        assert slots(build_checklist([med(interval_hours=bad)], ANCHOR)) == ["anytime"]
    assert slots(build_checklist([med(interval_hours=24)], ANCHOR)) == ["morning"]
    assert slots(build_checklist([med(interval_hours=6)], ANCHOR)) == [
        "morning", "midday", "evening", "bedtime"
    ]


def test_bad_follow_up_values_are_undated():
    for d in ({"date": "next week"}, {"date": 5}, {"days_after_discharge": "soon"},
              {"days_after_discharge": -2}, {"days_after_discharge": 10**12}):
        (spec,) = build_checklist([item("follow_up", d)], ANCHOR)
        assert spec.undated


def test_bad_date_falls_back_to_relative():
    it = item("follow_up", {"date": "garbage", "days_after_discharge": 7})
    (spec,) = build_checklist([it], ANCHOR)
    assert spec.approximate and spec.start_date == date(2026, 11, 12)
