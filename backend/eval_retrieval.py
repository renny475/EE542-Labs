"""Retrieval check per embedding model against a small golden question set.

On-topic questions list the document source_ids that would answer them and
count as a hit when any appears in the top-k (threshold ignored: the number
used to compare embeddings). Off-topic questions (``expect: []``) must be
rejected: their top score has to fall below the model's threshold.
``--calibrate`` picks a threshold per histogram model from those two sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select

import settings
from corpus_db import documents, load_snapshot, make_engine, models_present
from embeddings import Embedder, is_histogram_model, make_embedder
from rag import Retriever

REJECT_MARGIN = 0.01


@dataclass
class EvalQuestion:
    question: str
    expect: list[str]  # [] means off-topic: must be rejected

    @property
    def off_topic(self) -> bool:
        return not self.expect


@dataclass
class EvalResult:
    question: str
    expect: list[str]
    hit_ids: list[str]
    top_score: float
    ok: bool

    @property
    def off_topic(self) -> bool:
        return not self.expect


@dataclass
class EvalSummary:
    model: str
    recall: tuple[int, int]  # on-topic hits / on-topic questions (threshold ignored)
    answered: tuple[int, int]  # on-topic questions whose top score clears the threshold
    rejected: tuple[int, int]  # off-topic questions whose top score is below the threshold
    threshold: float


def load_eval(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [EvalQuestion(question=str(item["question"]), expect=[str(x) for x in item.get("expect") or []]) for item in raw]


def evaluate(
    questions: list[EvalQuestion],
    retriever: Retriever,
    embedder: Embedder,
    source_id_of: Callable[[int], str],
    *,
    top_k: int = 5,
    threshold: float,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for q in questions:
        vec = embedder.embed([q.question])[0]  # one call per question, as the app does
        hits = retriever.search(vec, top_k, min_score=0.0)
        hit_ids: list[str] = []
        for h in hits:
            sid = source_id_of(h.document_id)
            if sid not in hit_ids:
                hit_ids.append(sid)
        top_score = hits[0].score if hits else 0.0
        ok = top_score < threshold if q.off_topic else any(sid in hit_ids for sid in q.expect)
        results.append(EvalResult(q.question, q.expect, hit_ids, top_score, ok))
    return results


def summarise(model: str, results: list[EvalResult], threshold: float) -> EvalSummary:
    on = [r for r in results if not r.off_topic]
    off = [r for r in results if r.off_topic]
    return EvalSummary(
        model=model,
        recall=(sum(r.ok for r in on), len(on)),
        answered=(sum(r.top_score >= threshold for r in on), len(on)),
        rejected=(sum(r.top_score < threshold for r in off), len(off)),
        threshold=threshold,
    )


def calibrate(results: list[EvalResult]) -> float:
    """Threshold that maximises answered on-topic hits plus rejected off-topic questions.

    Candidates are the midpoints between consecutive top scores (plus one cut
    below everything and one above). Ties go to the higher cut, so a toss-up
    prefers rejecting. A single off-topic outlier therefore cannot silence the
    model: rejecting it would have to be worth more than the answers it costs.
    """
    on_scores = [r.top_score for r in results if not r.off_topic and r.ok]
    off_scores = [r.top_score for r in results if r.off_topic]
    scores = sorted(set(on_scores + off_scores))
    if not scores:
        return settings.RAG_MIN_SCORE
    candidates = [scores[0] - REJECT_MARGIN] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + REJECT_MARGIN]
    best_t, best_score = candidates[0], -1
    for t in candidates:
        value = sum(s >= t for s in on_scores) + sum(s < t for s in off_scores)
        if value >= best_score:  # ">=" so a tie moves to the higher cut
            best_t, best_score = t, value
    return best_t


def _print_results(summary: EvalSummary, results: list[EvalResult]) -> None:
    print(f"== {summary.model} (threshold {summary.threshold:.2f})")
    for r in results:
        tag = ("REJ " if r.ok else "LEAK") if r.off_topic else ("OK  " if r.ok else "MISS")
        print(f"{tag} {r.top_score:.2f} {r.question}  ->  {', '.join(r.hit_ids) or '-'}")
    _print_summary(summary)


def _print_summary(s: EvalSummary) -> None:
    print(
        f"recall@{settings.RAG_TOP_K} (no threshold): {s.recall[0]}/{s.recall[1]}  |  with threshold {s.threshold:.2f}: "
        f"answered {s.answered[0]}/{s.answered[1]}, rejected {s.rejected[0]}/{s.rejected[1]}"
    )


def run_model(model: str, engine, questions, id_to_source, thresholds: dict[str, float]) -> tuple[list[EvalResult], float]:
    retriever = Retriever(load_snapshot(engine, model), model)
    embedder = make_embedder(model, api_key=settings.OPENAI_API_KEY, models_dir=settings.MODELS_DIR)
    threshold = settings.threshold_for(model, thresholds)
    return evaluate(questions, retriever, embedder, id_to_source.__getitem__, top_k=settings.RAG_TOP_K, threshold=threshold), threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=settings.EMBEDDING_MODEL)
    parser.add_argument("--all", action="store_true", help="every model present in chunk_embeddings")
    parser.add_argument("--calibrate", action="store_true", help="write per-histogram-model thresholds to data/models/thresholds.json")
    args = parser.parse_args()

    engine = make_engine(settings.SYNC_DATABASE_URL)
    with engine.connect() as conn:
        id_to_source = dict(conn.execute(select(documents.c.id, documents.c.source_id)).all())
    questions = load_eval(settings.DATA_DIR / "eval_questions.yaml")
    thresholds = settings.load_thresholds(settings.MODELS_DIR)
    models = sorted(models_present(engine)) if args.all else [args.model]

    summaries: list[EvalSummary] = []
    any_miss = False
    for model in models:
        results, threshold = run_model(model, engine, questions, id_to_source, thresholds)
        if args.calibrate and is_histogram_model(model):
            threshold = round(calibrate(results), 3)
            thresholds[model] = threshold
            for r in results:  # re-judge the off-topic rows against the new threshold
                if r.off_topic:
                    r.ok = r.top_score < threshold
        summary = summarise(model, results, threshold)
        summaries.append(summary)
        any_miss |= any(not r.ok for r in results if not r.off_topic)
        if not args.all:
            _print_results(summary, results)

    if args.calibrate:
        path = settings.MODELS_DIR / "thresholds.json"
        path.write_text(json.dumps(dict(sorted(thresholds.items())), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if args.all:
        print(f"{'model':28s} {'recall@5':>9s} {'answered':>9s} {'rejected':>9s} {'thr':>5s}")
        for s in summaries:
            print(f"{s.model:28s} {s.recall[0]:>4d}/{s.recall[1]:<4d} {s.answered[0]:>4d}/{s.answered[1]:<4d} {s.rejected[0]:>4d}/{s.rejected[1]:<4d} {s.threshold:5.2f}")
        return 0
    return 1 if any_miss else 0


if __name__ == "__main__":
    sys.exit(main())
