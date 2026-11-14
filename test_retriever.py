"""Retrieval integration checks. No Ollama or downloaded encoder needed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from document_processing import Chunk, make_chunks, read_pdf
from retriever import RetrievalIndex, evaluate_gout


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("c1", 1, "Medication", "Medication:\nExampleMed 5 mg every 12 hours as needed.\nTake with food."),
            Chunk("c2", 2, "Follow-up", "Follow-up:\nAttend your clinic appointment on December 5."),
            Chunk("c3", 3, "Monitoring", "Monitoring:\nWrite down pain every evening for 7 days."),
        ]
        self.index = RetrievalIndex(self.chunks, "document-A")

    def test_medication_retrieval_keeps_qualifiers_and_citation(self):
        hit = self.index.search("ExampleMed food")['matches'][0]
        self.assertEqual(hit['chunk_id'], 'c1')
        self.assertEqual(hit['page'], 1)
        self.assertEqual(hit['text'], self.chunks[0].text)
        self.assertIn('as needed', hit['text'])
        self.assertEqual(hit['source_id'], 'document-A:c1')

    def test_followup_retrieval(self):
        self.assertEqual(self.index.search("clinic appointment")['matches'][0]['chunk_id'], 'c2')

    def test_numeric_query(self):
        self.assertEqual(self.index.search("every evening 7 days")['matches'][0]['chunk_id'], 'c3')

    def test_unknown_vocabulary_returns_no_lexical_match(self):
        self.assertEqual(self.index.search("reset wifi password")['matches'], [])

    def test_stop_words_only_returns_no_lexical_match(self):
        self.assertEqual(self.index.search("what should I do")['matches'], [])

    def test_blank_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search(" ")

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("food", top_k=0)

    def test_top_k_limits_results(self):
        self.assertEqual(len(self.index.search("every", top_k=1)['matches']), 1)

    def test_save_reload_preserves_actual_search_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            self.index.save(path)
            loaded = RetrievalIndex.load(path)
            self.assertEqual(self.index.search("food"), loaded.search("food"))

    def test_existing_index_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.index.save(Path(directory))

    def test_full_export_preserves_every_section_in_order(self):
        exported = self.index.export_all()
        self.assertEqual([c['id'] for c in exported['sections']], ['c1', 'c2', 'c3'])
        self.assertEqual(exported['pages_received'], [1, 2, 3])
        self.assertEqual([c['text'] for c in exported['sections']], [c.text for c in self.chunks])

    def test_independent_document_indexes_do_not_mix_sources(self):
        other = RetrievalIndex([Chunk('c1', 1, 'Other', 'Other:\nAnnual hearing review.')], 'document-B')
        self.assertEqual(other.search('ExampleMed')['matches'], [])
        self.assertEqual(self.index.search('hearing')['matches'], [])

    def test_nonprobability_scores_are_labelled(self):
        self.assertFalse(self.index.search('food')['score_is_confidence'])

    def test_empty_index_is_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalIndex([], 'document-A')

    def test_duplicate_chunk_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalIndex([self.chunks[0], self.chunks[0]], 'document-A')

    def test_semantic_vector_shape_is_checked_without_loading_model(self):
        with self.assertRaises(ValueError):
            RetrievalIndex(self.chunks, 'A', backend='semantic', vectors=np.array([[1., 0.]]))

    def test_semantic_vectors_must_be_normalized(self):
        with self.assertRaises(ValueError):
            RetrievalIndex(self.chunks, 'A', backend='semantic', vectors=np.ones((3, 2)))

    def test_reference_pdf_retrieval_benchmark(self):
        sample = Path(__file__).parent / 'sample' / 'gout.pdf'
        self.assertTrue(sample.exists(), "Keep sample/gout.pdf in this project directory.")
        chunks = make_chunks(read_pdf(sample))
        index = RetrievalIndex(chunks, 'fictional-gout')
        report = evaluate_gout(index)
        self.assertEqual(len(chunks), 11)
        self.assertEqual(report['query_count'], 15)
        self.assertEqual(report['hit_at_1'], 1.0)
        self.assertEqual(report['hit_at_3'], 1.0)


if __name__ == '__main__':
    unittest.main()
