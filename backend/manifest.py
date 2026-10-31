"""The source manifest: what documents make up the corpus and where they live.

``data/sources.yaml`` is a list of mappings; each becomes a :class:`Source`.
``file`` is a path relative to the project root (``data/ois/x.pdf``), so the
on-disk location is ``data_dir.parent / file``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import yaml

TIERS = ("ois", "uscis", "sevp", "cfr")
KINDS = ("pdf", "html")
EXTRACTORS = ("cfr_paragraph_f",)


class ManifestError(ValueError):
    """The manifest file is malformed or violates an invariant."""


@dataclass
class Source:
    id: str
    title: str
    tier: str
    url: str
    file: str
    kind: str
    manual: bool = False
    fetched: date | None = None
    published: date | None = None
    extractor: str | None = None

    def path(self, data_dir: Path) -> Path:
        return data_dir.parent / self.file


_REQUIRED = ("id", "title", "tier", "url", "file", "kind")
_KNOWN = {f.name for f in fields(Source)}


def _parse_date(value: object, field: str, source_id: str) -> date | None:
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ManifestError(f"{source_id}: {field} is not an ISO date: {value!r}") from exc
    raise ManifestError(f"{source_id}: {field} must be a date, got {type(value).__name__}")


def _parse_entry(raw: object, position: int) -> Source:
    if not isinstance(raw, dict):
        raise ManifestError(f"entry {position} is not a mapping")
    source_id = str(raw.get("id", f"<entry {position}>"))
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ManifestError(f"{source_id}: missing field(s) {', '.join(missing)}")
    unknown = set(raw) - _KNOWN
    if unknown:
        raise ManifestError(f"{source_id}: unknown field(s) {', '.join(sorted(unknown))}")
    if raw["tier"] not in TIERS:
        raise ManifestError(f"{source_id}: tier must be one of {TIERS}, got {raw['tier']!r}")
    if raw["kind"] not in KINDS:
        raise ManifestError(f"{source_id}: kind must be one of {KINDS}, got {raw['kind']!r}")
    if not str(raw["file"]).startswith("data/"):
        raise ManifestError(f"{source_id}: file must start with 'data/', got {raw['file']!r}")
    extractor = raw.get("extractor")
    if extractor is not None and extractor not in EXTRACTORS:
        raise ManifestError(f"{source_id}: unknown extractor {extractor!r}")
    return Source(
        id=str(raw["id"]),
        title=str(raw["title"]),
        tier=raw["tier"],
        url=str(raw["url"]),
        file=str(raw["file"]),
        kind=raw["kind"],
        manual=bool(raw.get("manual", False)),
        fetched=_parse_date(raw.get("fetched"), "fetched", source_id),
        published=_parse_date(raw.get("published"), "published", source_id),
        extractor=extractor,
    )


def load_manifest(path: Path) -> list[Source]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        return []
    if not isinstance(data, list):
        raise ManifestError(f"{path}: top level must be a list of entries")
    sources = [_parse_entry(raw, i) for i, raw in enumerate(data)]
    seen: set[str] = set()
    for s in sources:
        if s.id in seen:
            raise ManifestError(f"duplicate source id {s.id!r}")
        seen.add(s.id)
    return sources


def _to_mapping(source: Source) -> dict:
    out: dict = {}
    for key, value in asdict(source).items():
        if value is None or (key == "manual" and value is False):
            continue
        out[key] = value.isoformat() if isinstance(value, date) else value
    return out


def save_manifest(path: Path, sources: list[Source]) -> None:
    text = yaml.safe_dump(
        [_to_mapping(s) for s in sources],
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=None,
        width=1000,
    )
    path.write_text(text, encoding="utf-8")


def slug_from_url(url: str) -> str:
    segments = [seg for seg in urlparse(url).path.split("/") if seg]
    if not segments:
        raise ManifestError(f"cannot derive a slug from {url!r}")
    return segments[-1]
