# AfterCare document retrieval

Python 3.10+ component for PDF text extraction, section chunking, document indexing,
source-linked retrieval and full-document export. Checklist generation is handled
by a separate teammate's LLM service. No generative model or Ollama is included.

## Files

- `document_processing.py`: Chunk data structure, PDF reader and section splitter.
- `retriever.py`: BM25 baseline, optional CPU semantic retrieval, saved index,
  retrieval CLI and source-bundle export.
- `test_retriever.py`: 18 deterministic retrieval and integration tests.
- `requirements.txt`: base dependencies (PyMuPDF and NumPy).
- `requirements-semantic.txt`: optional Sentence Transformers encoder dependencies.
- `sample/gout.pdf`: explicitly fictional test fixture. Required by one test and
  the fixture-specific evaluation command.

## Setup

```bash
python3 -m venv "$HOME/.venvs/aftercare"
source "$HOME/.venvs/aftercare/bin/activate"
python -m pip install -r requirements.txt
```

If that environment already exists, activate it; no need to create it again.

## Run the baseline

```bash
python retriever.py build sample/gout.pdf --index rag_index --backend bm25
python retriever.py search --index rag_index --query "How often do I take colchicine?"
python retriever.py eval-gout --index rag_index
python retriever.py export --index rag_index --output source_bundle.json
python -m unittest -v
```

Use a fresh index/output path for each new run. Existing paths are not overwritten.
On the supplied sheet, the expected section ranked first for all 15 development
questions, and all 18 tests passed. This is a small same-document development
benchmark, not generalization, medical correctness or live model validation.

## Teammate integration

```python
from pathlib import Path
from retriever import RetrievalIndex

index = RetrievalIndex.load(Path("rag_index"))
question_context = index.search("When is my repeat kidney blood test?", top_k=3)
all_sources = index.export_all()
```

Search returns original passages, source IDs, page numbers, headings and ranking
scores. Scores are not confidence probabilities. Send all source sections to the
checklist extractor so it can inspect the whole sheet; a top-k subset can miss tasks.
The receiving module should preserve citations, retain conditions, flag ambiguity
and validate generated instructions. The export itself is not a checklist.

## Optional semantic retrieval

```bash
python -m pip install -r requirements-semantic.txt
python retriever.py build sample/gout.pdf --index rag_semantic --backend semantic
python retriever.py search --index rag_semantic --query "What footwear is suitable?"
```

The first use downloads sentence-transformers/all-MiniLM-L6-v2 and dependencies.
It is an embedding encoder running on CPU, not a checklist-generating LLM. This
backend has not been tested with downloaded weights in the build environment.
Evaluate it separately, including paraphrases and questions the sheet cannot answer.
Nearest-neighbor matches alone do not establish that an answer exists. No reliable
semantic score threshold is calibrated. Rebuild indexes when model weights change.

## Current limits

Text PDFs only. Scans, unusual headings, complex columns and cross-page instruction
links need more handling. Sections over 4,500 characters stop processing rather
than being silently cut. The optional encoder also checks its token limit.
BM25 matches words and can miss synonyms or retrieve keyword overlap without a
complete answer. Missing/conflicting instructions are not resolved by this module.
Document-level source separation is implemented; application authentication and
patient authorization are outside this local component.

Generated indexes, source bundles and local data are excluded by `.gitignore`.
Only the explicitly fictional sample PDF is allowed by the PDF ignore exception.

## References

- https://sbert.net/docs/sentence_transformer/usage/usage.html
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://pymupdf.readthedocs.io/en/latest/recipes-text.html
