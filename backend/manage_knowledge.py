"""Education library curator commands.

  status   show what is indexed and whether it matches EMBEDDING_MODEL
  collect  fetch every manifest URL into knowledge/sources/ as an unreviewed snapshot
  index    embed every reviewed snapshot and curated summary into the index
"""
import argparse
import asyncio
import json
import sys

from config import Settings, get_settings
from knowledge import DATA, KnowledgeError, KnowledgeStore, collect_sources, index_sources
from llm import LLMClient, LLMError


def run(argv: list[str], settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "collect", "index"])
    args = parser.parse_args(argv)
    settings = settings or get_settings()
    llm = LLMClient(settings)
    try:
        if args.command == "collect":
            report = asyncio.run(collect_sources(DATA))
            print(json.dumps(report, indent=2))
            failed = [r["id"] for r in report if not r["ok"]]
            if failed:
                print("Collection failed for: " + ", ".join(failed), file=sys.stderr)
                return 1
            print("Review each snapshot in knowledge/sources/, then set \"reviewed\": true.", file=sys.stderr)
            return 0
        if args.command == "index" and not settings.embedding_api_key:
            print("Set EMBEDDING_API_KEY (or LLM_API_KEY) in the project .env before indexing.", file=sys.stderr)
            return 2
        store = KnowledgeStore(settings.rag_db_path, llm, settings.rag_min_similarity)
        if args.command == "index":
            result = asyncio.run(index_sources(store, DATA))
            for source_id, chunks in result["indexed"].items():
                print(f"{source_id}: {chunks} chunks")
            if result["skipped"]:
                print("Skipped (not reviewed): " + ", ".join(result["skipped"]), file=sys.stderr)
        print(json.dumps(store.status(), indent=2))
        return 0
    except (KnowledgeError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
