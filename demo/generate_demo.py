"""Generate the fictional family-medicine demo discharge documents.

    python demo/generate_demo.py [--days-ago 2] [--out demo/out]

Reads demo/cases/*.json and writes <case>.pdf and <case>.txt per case. The TXT
and the PDF carry the same text (the PDF adds a "FICTIONAL - FOR DEMO" page
header and footer). The discharge date is today minus --days-ago; follow-ups
are written as absolute dates, so the committed files in demo/out are dated
relative to the day they were generated: regenerate before a demo. Every logical
line is one sentence and fits on one PDF line, and the document fits one page, so
the app's reader (pypdf + paragraphs) yields the same units as the TXT. All people and events are invented.

render_text() is also used by scripts/seed_demo.py, so the kidney-stones case
has a single source of truth in demo/cases/kidney_stones.json.
"""
import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).resolve().parent
CASES_DIR = HERE / "cases"
DEFAULT_OUT = HERE / "out"
BANNER = "FICTIONAL - FOR DEMO"
LABEL = re.compile(r"^[A-Z][A-Za-z -]{1,28}:(?=\s|$)")


def long_date(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


def load_case(case_id: str) -> dict:
    return json.loads((CASES_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def case_ids() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("*.json"))


def medication_line(med: dict) -> str:
    line = f"{med['name']} {med['dose']} {med['route']} {med['frequency']}"
    duration = med.get("duration")
    if duration == "ongoing":
        line += ", ongoing"
    elif duration:
        line += f" for {duration}"
    line += "."
    if med.get("note"):
        line += f" {med['note']}"
    return line


SENTENCE_END = re.compile(r"(?<=[.!?])(?<!Dr\.) (?=[A-Z])")


def one_sentence_per_line(block: str) -> str:
    """Split each line at sentence ends so no logical line has to wrap in the PDF."""
    lines: list[str] = []
    for line in block.splitlines():
        lines.extend(SENTENCE_END.split(line) if not line.startswith("- ") else [line])
    return "\n".join(lines)


def body_blocks(case: dict, discharge: date) -> list[str]:
    """The document as paragraph blocks (no banner)."""
    admitted = discharge - timedelta(days=case["admit_days"])
    course = case["hospital_course"].format(procedure_date=long_date(discharge - timedelta(days=1)))
    blocks = [
        "\n".join([
            f"{case['hospital']} - Discharge Summary",
            f"Patient: {case['patient']}",
            f"Admission date: {long_date(admitted)}",
            f"Discharge date: {long_date(discharge)}",
            f"Attending: {case['attending']}, {case['service']}",
        ]),
        f"Diagnosis:\n{case['diagnosis']}",
        f"Hospital course:\n{course}",
        "\n".join(["Medications:"] + [medication_line(m) for m in case["medications"]]),
        "\n".join(["Diet:"] + case["diet"]),
        "\n".join(["Activity:"] + case["activity"]),
        "\n".join([f"{case['care_label']}:"] + case["care"]),
        "\n".join(["Monitoring:"] + case["monitoring"]),
        "\n".join(["Follow-up:"] + [
            f"See {f['who']} on {long_date(discharge + timedelta(days=f['days']))} for {f['purpose']}."
            for f in case["follow_up"]
        ]),
        "\n".join(["Call 911 or return to the ER if:"] + [f"- {s}" for s in case["warning_signs"]]),
        "\n".join([f"Call the clinic at {case['clinic_phone']} if:"] + [f"- {s}" for s in case["call_clinic"]]),
    ]
    return [one_sentence_per_line(b) for b in blocks]


def render_text(case: dict, discharge: date, banner: bool = True) -> str:
    blocks = body_blocks(case, discharge)
    if banner:
        blocks = [BANNER] + blocks + [BANNER]
    return "\n\n".join(blocks)


class DemoPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(160, 0, 0)
        self.cell(0, 6, BANNER, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(160, 0, 0)
        self.cell(0, 6, BANNER, align="C")
        self.set_text_color(0, 0, 0)


def draw_block(pdf: FPDF, block: str, is_title_block: bool) -> None:
    line_h = 4.3
    for line_no, line in enumerate(block.splitlines()):
        pdf.set_font("Helvetica", "B" if is_title_block and line_no == 0 else "", 15 if is_title_block and line_no == 0 else 9.5)
        # Bold labels are wider than regular text, so measure with a small safety margin.
        if pdf.get_string_width(line) * 1.03 > pdf.w - pdf.l_margin - pdf.r_margin - 4:
            raise ValueError(f"Line would wrap in the PDF, shorten it: {line!r}")
        if is_title_block and line_no == 0:
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            continue
        if line.startswith("- "):
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, line_h, line, new_x="LMARGIN", new_y="NEXT")
            continue
        match = LABEL.match(line)
        if match:
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.write(line_h, match.group(0))
            line = line[match.end():]
        pdf.set_font("Helvetica", "", 9.5)
        pdf.write(line_h, line)
        pdf.ln(line_h)
    pdf.ln(2)


def render_pdf(case: dict, discharge: date) -> bytes:
    pdf = DemoPDF(format="Letter")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()
    for index, block in enumerate(body_blocks(case, discharge)):
        # Keep each section on one page so a paragraph is never split by the footer.
        with pdf.offset_rendering() as probe:
            draw_block(probe, block, index == 0)
        if probe.page_break_triggered:
            pdf.add_page()
        draw_block(pdf, block, index == 0)
    return bytes(pdf.output())


def generate(out_dir: Path, days_ago: int = 2, today: date | None = None) -> list[Path]:
    discharge = (today or date.today()) - timedelta(days=days_ago)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case_id in case_ids():
        case = load_case(case_id)
        text = render_text(case, discharge)
        text.encode("latin-1")  # the core PDF font only covers Latin-1
        txt_path = out_dir / f"{case_id}.txt"
        txt_path.write_text(text + "\n", encoding="utf-8")
        pdf_path = out_dir / f"{case_id}.pdf"
        pdf_path.write_bytes(render_pdf(case, discharge))
        written += [pdf_path, txt_path]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fictional demo discharge documents.")
    parser.add_argument("--days-ago", type=int, default=2,
                        help="discharge date is today minus this many days (default 2)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()
    for path in generate(args.out, args.days_ago):
        print(path)


if __name__ == "__main__":
    sys.exit(main())
