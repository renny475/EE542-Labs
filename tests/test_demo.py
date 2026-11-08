import json
import re
import sys
from pathlib import Path

import pytest

from documents import read_upload

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

import generate_demo  # noqa: E402

# Binding medication list per case: the education library only has labels for these.
ALLOWED_MEDS = {
    "pneumonia": ["amoxicillin-clavulanate", "azithromycin"],
    "cellulitis": ["cephalexin"],
    "uti": ["nitrofurantoin"],
    "copd": ["prednisone", "albuterol inhaler (ventolin hfa)", "tiotropium (spiriva handihaler)"],
    "heart_failure": ["furosemide", "lisinopril", "metoprolol succinate"],
    "diabetes_htn": ["metformin", "amlodipine", "lisinopril"],
    "kidney_stones": ["tamsulosin", "ibuprofen"],
}
EXTRA_DRUGS = ["acetaminophen", "tylenol", "potassium citrate", "aspirin", "insulin", "hydrochlorothiazide"]


def squash(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo-out")
    generate_demo.generate(out)
    return out


def test_all_cases_present_and_generated(out_dir):
    assert sorted(ALLOWED_MEDS) == generate_demo.case_ids()
    for case_id in ALLOWED_MEDS:
        assert (out_dir / f"{case_id}.pdf").is_file()
        assert (out_dir / f"{case_id}.txt").is_file()


@pytest.mark.parametrize("case_id", sorted(ALLOWED_MEDS))
def test_pdf_and_txt_carry_same_content(out_dir, case_id):
    pdf_text = squash(read_upload((out_dir / f"{case_id}.pdf").read_bytes(), "d.pdf"))
    txt_text = squash(read_upload((out_dir / f"{case_id}.txt").read_bytes(), "d.txt"))
    assert "FICTIONAL - FOR DEMO" in pdf_text
    assert re.search(r"Discharge date: [A-Z][a-z]+ \d{1,2}, \d{4}", pdf_text)
    assert "Call 911 or return to the ER if:" in pdf_text
    assert "(fictional)" in pdf_text
    # The PDF repeats the banner on each page; the body text must otherwise match.
    strip = lambda t: t.replace("FICTIONAL - FOR DEMO", "")  # noqa: E731
    assert squash(strip(pdf_text)) == squash(strip(txt_text))
    assert "\u2014" not in txt_text


@pytest.mark.parametrize("case_id", sorted(ALLOWED_MEDS))
def test_medications_stay_inside_allowlist(out_dir, case_id):
    case = json.loads((ROOT / "demo" / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert sorted(m["name"].lower() for m in case["medications"]) == sorted(ALLOWED_MEDS[case_id])
    text = (out_dir / f"{case_id}.txt").read_text(encoding="utf-8").lower()
    for drug in EXTRA_DRUGS:
        assert drug not in text


def test_kidney_stones_required_wording(out_dir):
    text = squash(read_upload((out_dir / "kidney_stones.pdf").read_bytes(), "d.pdf"))
    assert "avoid high-oxalate foods such as nuts, spinach, and rhubarb" in text
    assert "limit sodium to less than 2,300 mg per day" in text
    assert "drink enough fluids to make at least 2.5 liters of urine a day" in text
    assert "tamsulosin 0.4 mg by mouth once daily at bedtime for 14 days" in text.lower()
    assert "ibuprofen" in text.lower() and "as needed" in text
