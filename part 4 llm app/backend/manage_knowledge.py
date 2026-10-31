"""Offline curator commands: collect, inspect, approve, index, upload."""
import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from knowledge import DATA, ROOT, KnowledgeStore, collect_sources, extract_document, make_record, save_record

load_dotenv(ROOT / '.env')

async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('collect')
    commands.add_parser('status')
    commands.add_parser('inspect')
    approve = commands.add_parser('approve')
    approve.add_argument('source_ids', nargs='+', help='IDs you have manually reviewed, or all')
    commands.add_parser('index')
    upload = commands.add_parser('import')
    upload.add_argument('path')
    upload.add_argument('--title', required=True)
    upload.add_argument('--url', required=True)
    upload.add_argument('--topic', choices=['F-1','CPT','OPT','STEM OPT','Travel'], required=True)
    args = parser.parse_args()
    store = KnowledgeStore()
    if args.command == 'collect':
        print(json.dumps(await collect_sources(), indent=2))
    elif args.command == 'status':
        print(json.dumps(store.status(), indent=2))
    elif args.command == 'inspect':
        for path in sorted((DATA / 'sources').glob('*.json')):
            record = json.loads(path.read_text())
            print(f"\n=== {record['id']} | {record['url']} | reviewed={record['reviewed']} ===\n{record['text']}")
    elif args.command == 'approve':
        for path in sorted((DATA / 'sources').glob('*.json')):
            record = json.loads(path.read_text())
            if 'all' in args.source_ids or record['id'] in args.source_ids:
                record['reviewed'] = True
                path.write_text(json.dumps(record, indent=2))
                print('Approved:', record['id'])
    elif args.command == 'index':
        records = [json.loads(p.read_text()) for p in sorted((DATA/'sources').glob('*.json'))]
        if not records or any(not r.get('reviewed') for r in records):
            raise ValueError('Inspect the collected sources, then approve them before indexing.')
        # Stage a complete replacement so failed re-indexing cannot mix snapshots/models.
        staging = store.path.with_suffix('.staging.sqlite3')
        staging.unlink(missing_ok=True)
        candidate = KnowledgeStore(staging)
        try:
            for record in records:
                n = await candidate.add(record, record.get('pages', [(None, record['text'])]))
                print(record['id'], n, 'chunks')
            staging.replace(store.path)
        finally:
            staging.unlink(missing_ok=True)
        print(json.dumps(store.status(), indent=2))
    elif args.command == 'import':
        path = Path(args.path)
        data = path.read_bytes()
        record = make_record(data, args.title, args.url, args.topic)
        pages = extract_document(data, path.suffix)
        print('Indexed chunks:', await store.add(record, pages))
        save_record(record, pages)

if __name__ == '__main__':
    asyncio.run(main())
