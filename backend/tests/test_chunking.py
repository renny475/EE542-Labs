from chunking import chunk_pages


def test_empty_input_gives_no_chunks():
    assert chunk_pages([]) == []
    assert chunk_pages([(1, ""), (2, "   \n")]) == []


def test_chunk_pages_skips_blank_pages_and_attributes_pages():
    p1 = "\n".join(f"Page one line {i}." for i in range(40))  # about 680 chars
    p3 = "\n".join(f"Page three line {i}." for i in range(40))
    chunks = chunk_pages([(1, p1), (2, ""), (3, p3)], chunk_size=300, chunk_overlap=50)
    assert chunks[0].page == 1 and chunks[0].index == 0
    assert chunks[-1].page == 3
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content) <= 300 and c.content.strip() for c in chunks)


def test_consecutive_chunks_overlap():
    text = "\n".join(f"Sentence number {i} about OPT rules." for i in range(100))
    chunks = chunk_pages([(None, text)], chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 5
    tail = chunks[0].content[-60:]
    assert tail.splitlines()[-1] in chunks[1].content
