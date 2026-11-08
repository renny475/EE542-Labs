"""Dev-only demo seed. Never imported by the app (it lives outside backend/).

Creates, or resets, the user `demo` / `demo-password-1` in the database named by
DATABASE_URL (from the project .env), with one PENDING fictional kidney-stones
discharge document and its extracted items, so the review screen can be seen
without a language model. Every source_quote is a verbatim substring of one
document paragraph; the script checks this before writing anything.

    .venv/bin/python scripts/seed_demo.py            # pending review
    .venv/bin/python scripts/seed_demo.py --confirm  # confirmed, checklist generated

--confirm runs the confirm route's own function, so the checklist is generated
exactly as when the patient presses Confirm. Resetting deletes everything the
demo user owns (documents, checklist, conversation, care-team questions).
"""
import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "demo"))

from sqlalchemy import delete, select  # noqa: E402

from auth import hash_password  # noqa: E402
from config import get_settings  # noqa: E402
from db import init_db, make_engine, make_session_factory  # noqa: E402
from documents import paragraphs  # noqa: E402
from models import (  # noqa: E402
    CareTeamQuestion, ChatMessage, ChecklistCompletion, ChecklistItem,
    DischargeDocument, ExtractedItem, User,
)
import routes_documents  # noqa: E402
from generate_demo import load_case, long_date, render_text  # noqa: E402

USERNAME = "demo"
PASSWORD = "demo-password-1"


def demo_text(discharge: date) -> str:
    """Same document as demo/out/kidney_stones.txt (one source: demo/cases)."""
    return render_text(load_case("kidney_stones"), discharge)


NO_SCHEDULE = {"times_per_day": None, "times_of_day": [], "duration_days": None}


def build_items(discharge: date) -> list[tuple[str, str, str, dict]]:
    """(category, plain-language text, verbatim source_quote, details)."""
    visit = long_date(discharge + timedelta(days=7))
    return [
        ("medication", "Take tamsulosin 0.4 mg by mouth once a day at bedtime for 14 days.",
         "Tamsulosin 0.4 mg by mouth once daily at bedtime for 14 days.",
         {"name": "Tamsulosin", "dose": "0.4 mg", "route": "by mouth",
          "frequency_text": "once daily at bedtime", "times_of_day": ["bedtime"],
          "times_per_day": 1, "interval_hours": None, "as_needed": False,
          "duration_days": 14, "instructions": "at bedtime"}),
        ("medication", "Take ibuprofen 600 mg with food, up to every 8 hours, only if you need it for pain.",
         "Ibuprofen 600 mg by mouth every 8 hours as needed for pain. Take with food.",
         {"name": "Ibuprofen", "dose": "600 mg", "route": "by mouth",
          "frequency_text": "every 8 hours as needed", "times_of_day": [],
          "times_per_day": None, "interval_hours": 8, "as_needed": True,
          "duration_days": None, "instructions": "with food"}),
        ("diet", "Drink enough fluids to make at least 2.5 liters of urine a day.",
         "drink enough fluids to make at least 2.5 liters of urine a day", {}),
        ("diet", "Avoid high-oxalate foods such as nuts, spinach, and rhubarb.",
         "avoid high-oxalate foods such as nuts, spinach, and rhubarb", {}),
        ("diet", "Keep salt (sodium) under 2,300 mg a day.",
         "limit sodium to less than 2,300 mg per day", {}),
        ("activity", "Do not lift anything heavier than 10 pounds until your stent is removed.",
         "No heavy lifting over 10 pounds until your stent is removed.", {}),
        ("wound_or_device_care", "Showers are fine. No baths or swimming while the stent is in.",
         "You may shower; no baths or swimming while the stent is in place.", dict(NO_SCHEDULE)),
        ("monitoring", "Strain all of your urine and save any stone pieces for your follow-up visit.",
         "Strain all of your urine through the strainer provided and save any stone pieces",
         dict(NO_SCHEDULE)),
        ("monitoring", "Check your temperature every evening for 7 days.",
         "Check your temperature every evening for 7 days.",
         {"times_per_day": 1, "times_of_day": ["evening"], "duration_days": 7}),
        ("follow_up", f"See Dr. Priya Raman at the Urology Clinic on {visit} to remove the stent.",
         f"See Dr. Priya Raman, Urology Clinic on {visit} (7 days after discharge) for stent removal.",
         {"who": "Dr. Priya Raman, Urology Clinic", "when_text": visit,
          "date": None, "days_after_discharge": 7}),
        ("follow_up", "See your primary care provider in about 14 days to talk about a 24-hour urine test.",
         f"See your primary care provider on {long_date(discharge + timedelta(days=14))} "
         "(14 days after discharge) for a discussion of a 24-hour urine test.",
         {"who": "Primary care provider", "when_text": long_date(discharge + timedelta(days=14)),
          "date": None, "days_after_discharge": 14}),
        ("warning_signs", "Call 911 or go to the ER for a fever of 101 F or higher with chills.",
         "a fever of 101 F (38.3 C) or higher with chills", {}),
        ("warning_signs", "Call 911 or go to the ER if you cannot keep fluids down.",
         "you cannot keep fluids down", {}),
        ("warning_signs", "Call 911 or go to the ER for severe pain that your medicine does not control.",
         "severe pain that your medicine does not control", {}),
        ("warning_signs", "Call 911 or go to the ER for chest pain, trouble breathing, or fainting.",
         "chest pain, trouble breathing, or fainting", {}),
        ("warning_signs", "Call the clinic if you cannot urinate.",
         "you cannot urinate", {}),
        ("warning_signs", "Call the clinic if you see heavy bleeding or clots.",
         "you see heavy bleeding or clots", {}),
    ]


# Shown in the "could not be verified" list: the document never says this.
UNVERIFIED = [{"category": "diet", "text": "Avoid all dairy products."}]


async def reset_user(session) -> User:
    user = await session.scalar(select(User).where(User.username == USERNAME))
    if user is not None:
        # Explicit deletes: SQLite does not enforce ON DELETE CASCADE by default.
        doc_ids = select(DischargeDocument.id).where(DischargeDocument.user_id == user.id)
        item_ids = select(ChecklistItem.id).where(ChecklistItem.document_id.in_(doc_ids))
        await session.execute(
            delete(ChecklistCompletion).where(ChecklistCompletion.checklist_item_id.in_(item_ids))
        )
        await session.execute(delete(ChecklistItem).where(ChecklistItem.document_id.in_(doc_ids)))
        await session.execute(delete(ExtractedItem).where(ExtractedItem.document_id.in_(doc_ids)))
        await session.execute(delete(ChatMessage).where(ChatMessage.document_id.in_(doc_ids)))
        await session.execute(delete(CareTeamQuestion).where(CareTeamQuestion.user_id == user.id))
        await session.execute(delete(DischargeDocument).where(DischargeDocument.user_id == user.id))
        await session.delete(user)
        await session.flush()
    user = User(username=USERNAME, password_hash=hash_password(PASSWORD))
    session.add(user)
    await session.flush()
    return user


async def seed(confirm: bool) -> None:
    discharge = date.today() - timedelta(days=2)
    text = demo_text(discharge)
    paras = paragraphs(text)
    items = build_items(discharge)
    for _, _, quote, _ in items:
        if not any(quote in p for p in paras):
            raise SystemExit(f"Quote is not inside one paragraph: {quote!r}")

    engine = make_engine(get_settings().database_url)
    try:
        await init_db(engine)
        async with make_session_factory(engine)() as session:
            user = await reset_user(session)
            doc = DischargeDocument(
                user_id=user.id,
                active_user_id=user.id,
                status="pending_review",
                filename="kidney_stones_demo.pdf",
                text=text,
                discharge_date=discharge,
                unverified=UNVERIFIED,
            )
            session.add(doc)
            await session.flush()
            session.add_all(
                ExtractedItem(document_id=doc.id, category=c, text=t, source_quote=q,
                              details=d, status="active")
                for c, t, q, d in items
            )
            await session.commit()
            if confirm:
                await routes_documents.confirm(user=user, session=session)
        state = "confirmed" if confirm else "pending review"
        print(f"Seeded user '{USERNAME}' with a {state} demo document ({len(items)} items).")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the AfterCare demo user (dev only).")
    parser.add_argument("--confirm", action="store_true",
                        help="also confirm the document, generating the checklist")
    asyncio.run(seed(parser.parse_args().confirm))


if __name__ == "__main__":
    main()
