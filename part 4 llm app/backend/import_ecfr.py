"""Collect the official eCFR F-1 student regulation using its public versioned API.

Federal regulatory text is public-domain government material. USC procedure notes
remain separate; this source does not establish a student's individual eligibility.
"""
import asyncio
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from knowledge import DATA, ROOT

HEADINGS = {
1:'Admission of student',2:'Student maintenance of Form I-20',3:'Admission of the spouse',
4:'Temporary absence',5:'Period of stay',6:'Full course of study',7:'Extension of stay',
8:'School transfer and change of educational objectives',9:'Employment',10:'Practical training',
11:'OPT application and approval process',12:'Reporting while on optional practical training',
13:'Temporary absence from the United States',14:'Effect of strike',15:'Spouse and children',
16:'Reinstatement to student status',17:'Current name and address',18:'Special rules for certain border',
19:'Remittance of the fee',20:'Severability'}
INCLUDE = [1,2,4,5,6,7,8,9,10,11,12,13,14,16,17]


def extract_records(xml, metadata):
    root=ET.fromstring(xml)
    paragraphs=[' '.join(''.join(p.itertext()).split()) for p in root.findall('.//P')]
    start=next(i for i,t in enumerate(paragraphs) if t.startswith('(f) Students in colleges'))
    end=next(i for i,t in enumerate(paragraphs[start+1:],start+1) if t.startswith('(g) Representatives'))
    students=paragraphs[start:end]
    positions={}
    for n,heading in HEADINGS.items():
        pattern=re.compile(r'^\('+str(n)+r'\)\s+'+re.escape(heading),re.I)
        found=[i for i,t in enumerate(students) if pattern.search(t)]
        if len(found)!=1:
            raise ValueError(f'Unexpected eCFR structure for subsection {n}; review before importing.')
        positions[n]=found[0]
    now=datetime.now(timezone.utc).isoformat()
    records=[]
    for n in INCLUDE:
        stop=positions[n+1]
        text='\n\n'.join(students[positions[n]:stop])
        cite=f'8 CFR 214.2(f)({n})'
        title=f'Federal regulation: {cite} — {HEADINGS[n]} (text version {metadata["latest_issue_date"]})'
        body=f'{cite}: {HEADINGS[n]}. Federal rule, not a USC application procedure.\n\n'+text
        topic='OPT' if n in (11,12) else ('Travel' if n in (4,13) else 'F-1')
        if n==10:topic='CPT / OPT / STEM OPT'
        url='https://www.ecfr.gov/on/'+metadata['latest_issue_date']+'/title-8/chapter-I/subchapter-B/part-214/section-214.2#p-214.2(f)('+str(n)+')'
        records.append(dict(id=f'ecfr-f-{n}',title=title,url=url,topic=topic,text=body,
            collected_at=now,sha256=hashlib.sha256(body.encode()).hexdigest(),reviewed=True,
            source_kind='official_federal_regulatory_text',text_version=metadata['latest_issue_date'],
            up_to_date_as_of=metadata['up_to_date_as_of'],citation=cite,
            curation_note='Full selected subsection from the official eCFR XML API, preserving paragraph order; F-2 and other non-F-1 topics omitted from this collection.'))
    return records


async def main():
    async with httpx.AsyncClient(timeout=90,follow_redirects=True) as client:
        response=await client.get('https://www.ecfr.gov/api/versioner/v1/titles.json');response.raise_for_status()
        metadata=next(t for t in response.json()['titles'] if t['number']==8)
        url='https://www.ecfr.gov/api/versioner/v1/full/'+metadata['latest_issue_date']+'/title-8.xml?section=214.2'
        response=await client.get(url);response.raise_for_status()
        xml=response.content
    records=extract_records(xml,metadata)
    directory=DATA/'sources';directory.mkdir(exist_ok=True)
    for record in records:
        (directory/(record['id']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    raw=DATA/'reference_text';raw.mkdir(exist_ok=True)
    (raw/'f1-federal-regulations.txt').write_text('\n\n'.join(r['text'] for r in records)+'\n')
    report=dict(source_api=url,text_version=metadata['latest_issue_date'],up_to_date_as_of=metadata['up_to_date_as_of'],
        collected_at=datetime.now(timezone.utc).isoformat(),xml_sha256=hashlib.sha256(xml).hexdigest(),
        subsections=len(records),words=sum(len(r['text'].split()) for r in records),
        citations=[r['citation'] for r in records])
    (DATA/'federal-collection-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    load_dotenv(ROOT/'.env')
    asyncio.run(main())
