import numpy as np
import pytest

from histogram_embedding import (
    Artifacts,
    ArtifactsMissing,
    HistogramEmbedder,
    HistogramSpec,
    histogram,
    load_artifacts,
    tokenize,
)

ART = Artifacts(
    vectors=np.eye(4, dtype=np.float32),
    vocab={"opt": 0, "employment": 1, "travel": 2, "visa": 3},
    # Bin 0 sits between opt/employment, bin 1 between travel/visa, so no word is
    # equidistant from both centroids (axis-aligned centroids would tie on two words).
    centroids=np.array([[0.7071, 0.7071, 0, 0], [0, 0, 0.7071, 0.7071]], dtype=np.float32),
    idf={"opt": 1.0, "employment": 2.0, "travel": 1.0, "visa": 3.0},
    chunk_count=10,
)


def test_tokenize_lowercases_filters_stopwords_and_short_tokens():
    # Possessive "student's" becomes "student": the apostrophe ends the token.
    assert tokenize("The F-1 student's OPT and CPT!") == ["f-1", "student", "opt", "cpt"]
    assert "the" not in tokenize("the and of") and tokenize("a an I") == []


def test_spec_parse_and_name_roundtrip():
    s = HistogramSpec.parse("w2v-hist-k256-idf-soft")
    assert (s.k, s.idf, s.soft) == (256, True, True) and s.name == "w2v-hist-k256-idf-soft"
    assert HistogramSpec.parse("w2v-hist-k64").name == "w2v-hist-k64"
    with pytest.raises(ValueError):
        HistogramSpec.parse("w2v-hist-idf")


def test_raw_histogram_counts_nearest_bins_and_normalises():
    h = histogram(["opt", "employment", "travel", "unknownword"], ART, HistogramSpec(2, False, False))
    assert np.allclose(h, [2 / 3, 1 / 3])


def test_idf_histogram_weights_rare_words():
    h = histogram(["opt", "visa"], ART, HistogramSpec(2, True, False))
    assert np.allclose(h, [1.0 / 2, 3.0 / 2])


def test_soft_histogram_spreads_over_nearest_centroids():
    h = histogram(["opt"], ART, HistogramSpec(2, False, True))
    assert h[0] > h[1] > 0 and np.isclose(h.sum(), 1.0)


def test_histogram_embedder_returns_zero_vector_for_unknown_words():
    e = HistogramEmbedder(ART, HistogramSpec(2, False, False))
    assert e.embed(["zzzz qqqq"]) == [[0.0, 0.0]] and e.dim == 2 and e.model == "w2v-hist-k2"


def test_histogram_embedder_output_is_unit_length():
    v = np.array(HistogramEmbedder(ART, HistogramSpec(2, True, True)).embed(["opt travel visa"])[0])
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_load_artifacts_reports_missing_files(tmp_path):
    with pytest.raises(ArtifactsMissing, match="word2vec.kv"):
        load_artifacts(tmp_path, 256)
