from pathlib import Path

import pytest

from extract import ExtractError, cfr_paragraph_f, html_to_text, pdf_pages, strip_repeated_lines

HTML = """<html><body><nav><a>Home</a></nav><header>USCIS</header>
<main><h1>OPT</h1><p>Students may apply.</p><ul><li>Pre-completion</li><li>Post-completion</li></ul>
<table><tr><td>Type</td><td>Months</td></tr><tr><td>OPT</td><td>12</td></tr></table></main>
<footer>Privacy</footer><script>x()</script></body></html>"""


def test_html_to_text_keeps_main_content_and_drops_chrome():
    text = html_to_text(HTML)
    assert text.splitlines() == [
        "OPT", "Students may apply.", "Pre-completion", "Post-completion", "Type | Months", "OPT | 12",
    ]


def test_strip_repeated_lines_removes_chrome_print_headers():
    words = {1: "one", 2: "two", 3: "three"}
    pages = [
        (i, f"10/31/26, 2:1{i} PM OPT - OIS\nBody {words[i]}\nhttps://ois.usc.edu/x/ {i}/3") for i in (1, 2, 3)
    ]
    assert strip_repeated_lines(pages) == [(1, "Body one"), (2, "Body two"), (3, "Body three")]


def test_strip_repeated_lines_leaves_short_documents_alone():
    pages = [(1, "Same\nA"), (2, "Same\nB")]
    assert strip_repeated_lines(pages) == pages


def test_cfr_paragraph_f_slices_between_markers():
    text = "(e) Treaty traders\nblah\n(f) Students in colleges, universities\n(1) Admission\n(2) Stuff\n(g) Representatives\nmore"
    assert cfr_paragraph_f(text) == "(f) Students in colleges, universities\n(1) Admission\n(2) Stuff"


def test_cfr_paragraph_f_requires_markers():
    with pytest.raises(ExtractError):
        cfr_paragraph_f("(f) Students in colleges\nno g")


I983 = Path(__file__).resolve().parents[2] / "data" / "federal" / "sevp-form-i-983-instructions.pdf"


@pytest.mark.skipif(not I983.exists(), reason="federal corpus not fetched")
def test_pdf_pages_reads_real_pdf():
    pages = pdf_pages(I983)
    assert pages[0][0] == 1 and len(pages) >= 2
    assert "STEM" in "\n".join(t for _, t in pages)


def test_cfr_paragraph_f_skips_table_of_contents():
    text = (
        "TABLE 1 TO 214.2 SECTION CONTENTS\n(e) Treaty traders\n(f) Students in colleges, universities\n"
        "(g) Representatives\n(h) Temporary employees\n"
        "body of (e)\n(f) Students in colleges, universities, sem-\ninaries\n(1) Admission\n"
        "(g) Representatives to international or-\nganizations\n"
    )
    assert cfr_paragraph_f(text) == "(f) Students in colleges, universities, sem-\ninaries\n(1) Admission"


def test_strip_repeated_lines_catches_alternating_headers_and_varying_digit_widths():
    # Two-sided print: odd pages carry one running head, even pages another,
    # and "Page 9 of 25" has fewer digits than "Page 10 of 25".
    pages = []
    for i in range(1, 11):
        head = "8 CFR Ch. I (1-1-25 Edition)" if i % 2 else "Department of Homeland Security"
        pages.append((i, f"{head}\nParagraph {['a','b','c','d','e','f','g','h','i','j'][i-1]}\nPage {i} of 10"))
    stripped = strip_repeated_lines(pages)
    assert [t for _, t in stripped] == [f"Paragraph {c}" for c in "abcdefghij"]


def test_strip_repeated_lines_keeps_a_heading_that_appears_on_two_pages_of_a_short_document():
    # OIS "Program Completion": page 2 is a jump list of the section titles, page 3 holds
    # the sections themselves. A title seen on two of four pages is content, not chrome.
    chrome = "Office of International Services\nNeed help?"
    pages = [
        (1, f"{chrome}\nProgram Completion (Students)"),
        (2, f"{chrome}\nChoose one of the following actions.\nTransfer to Another Institution\nDepart From the United States"),
        (3, f"{chrome}\nTransfer to Another Institution\nAsk your new school to request a SEVIS transfer."),
        (4, f"{chrome}\nDepart From the United States\nLeave within the 60-day grace period."),
    ]
    out = strip_repeated_lines(pages)
    assert "Transfer to Another Institution" in out[2][1]
    assert "Depart From the United States" in out[3][1]
    assert all("Need help?" not in text and "Office of International Services" not in text for _, text in out)


def test_join_wrapped_lines_repairs_hyphenation_and_mid_sentence_breaks_but_keeps_headings():
    from extract import join_wrapped_lines

    text = (
        "STEM OPT Extension\nWhat Is a STEM OPT Extension?\n"
        "(f) Students in colleges, universities, \nseminaries, conservatories, academic high \n"
        "schools, elementary schools, other aca-\ndemic institutions, and in language \ntraining programs."
    )
    assert join_wrapped_lines(text) == (
        "STEM OPT Extension\nWhat Is a STEM OPT Extension?\n"
        "(f) Students in colleges, universities, seminaries, conservatories, academic high "
        "schools, elementary schools, other academic institutions, and in language training programs."
    )
