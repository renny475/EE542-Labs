import struct

import numpy as np

from embeddings import FakeEmbedder, from_blob, normalise, to_blob


def test_fake_embedder_is_deterministic_and_unit_length():
    e = FakeEmbedder(dim=8)
    a, b = e.embed(["opt", "opt"])
    c = e.embed(["cpt"])[0]
    assert a == b and a != c and len(a) == 8
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6


def test_blob_roundtrip_is_float32_little_endian():
    blob = to_blob([1.0, -2.5, 0.0])
    assert len(blob) == 12 and blob[:4] == struct.pack("<f", 1.0)
    assert from_blob(blob).tolist() == [1.0, -2.5, 0.0]


def test_normalise_rows():
    out = normalise(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], [0, 0])


def test_openai_embedder_bounds_request_time():
    from embeddings import OpenAIEmbedder

    client = OpenAIEmbedder("text-embedding-3-small", "sk-test-not-used")._client
    assert client.timeout == 30.0 and client.max_retries == 2
