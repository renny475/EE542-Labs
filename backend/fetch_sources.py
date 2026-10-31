"""Download the non-manual documents listed in ``data/sources.yaml``.

Fetch policy (see the design doc, section 4): a self-identifying User-Agent,
a two-second pause after every request, no retry on a non-200 status, and
never a browser User-Agent. Manual (OIS) entries are only checked for
presence; the user saves those from Chrome by hand.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import httpx

import settings
from manifest import Source, load_manifest, save_manifest

PAUSE_SECONDS = 2.0


@dataclass
class FetchReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def _mtime_date(path: Path) -> date:
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def _download(client: httpx.Client, source: Source, target: Path, today: date) -> str | None:
    """Fetch one source into ``target``. Returns an error string, or None on success."""
    try:
        response = client.get(
            source.url,
            headers={"User-Agent": settings.USER_AGENT},
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"
    if response.status_code != 200:
        return f"HTTP {response.status_code}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    source.fetched = today
    return None


def fetch_sources(
    manifest_path: Path,
    data_dir: Path,
    client: httpx.Client,
    *,
    refresh: bool = False,
    check_only: bool = False,
    today: date | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchReport:
    today = today or date.today()
    sources = load_manifest(manifest_path)
    report = FetchReport()

    for source in sources:
        target = source.path(data_dir)
        if source.manual:
            if target.exists():
                if source.fetched is None:
                    source.fetched = _mtime_date(target)
                report.skipped.append(source.id)
            continue
        if check_only or (target.exists() and not refresh):
            if target.exists():
                report.skipped.append(source.id)
            continue
        error = _download(client, source, target, today)
        sleep(PAUSE_SECONDS)
        if error is None:
            report.downloaded.append(source.id)
        else:
            report.failed.append((source.id, error))

    for source in sources:
        if not source.path(data_dir).exists():
            report.missing.append(source.id)

    save_manifest(manifest_path, sources)
    return report


def _print_report(report: FetchReport) -> None:
    print(f"downloaded ({len(report.downloaded)}): {', '.join(report.downloaded) or '-'}")
    print(f"skipped    ({len(report.skipped)}): {', '.join(report.skipped) or '-'}")
    for source_id, error in report.failed:
        print(f"FAILED  {source_id}: {error}")
    for source_id in report.missing:
        print(f"MISSING {source_id}")
    print(f"failed: {len(report.failed)}, missing: {len(report.missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    parser.add_argument("--check", action="store_true", help="only report missing files, download nothing")
    args = parser.parse_args()

    with httpx.Client(timeout=60) as client:
        report = fetch_sources(
            settings.DATA_DIR / "sources.yaml",
            settings.DATA_DIR,
            client,
            refresh=args.refresh,
            check_only=args.check,
        )
    _print_report(report)
    return 1 if report.failed or report.missing else 0


if __name__ == "__main__":
    sys.exit(main())
