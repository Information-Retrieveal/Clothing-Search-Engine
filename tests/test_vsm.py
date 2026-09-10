"""
Tests for the ranking engine  (Person A)
========================================

Three things are checked:

  1. THE 0.8 CHECK - reproduce the lecture's lnc.ltc "car / auto insurance"
     worked example (IIR Ch. 6). If our weighting prints 0.8 on the slide's
     own numbers, the scheme is provably correct.

  2. TEN FREE-TEXT QUERIES over the real 100-doc corpus return well-formed,
     correctly ordered results (<=10 hits, score descending, valid docids).

  3. EDGE CASES - an out-of-vocabulary query (silk / leather) and the
     degenerate all-idf-0 query ("festive wear") both return [].

Runnable either way::

    python tests/test_vsm.py          # plain asserts, prints a summary
    pytest tests/test_vsm.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing import load_corpus            # noqa: E402
from inverted_index import build_inverted_index   # noqa: E402
from vsm import cosine_lnc_ltc, rank              # noqa: E402


# --------------------------------------------------------------------------- #
# 1. The 0.8 check - lecture's lnc.ltc worked example (IIR Ch. 6).
# --------------------------------------------------------------------------- #
def test_lecture_example_scores_0_8():
    """query "best car insurance"  vs  doc "car insurance auto insurance"."""
    N = 1_000_000
    df = {"auto": 5000, "best": 50000, "car": 10000, "insurance": 1000}
    query_tf = {"best": 1, "car": 1, "insurance": 1}
    doc_tf = {"car": 1, "insurance": 2, "auto": 1}

    score = cosine_lnc_ltc(query_tf, doc_tf, df, N)
    assert round(score, 2) == 0.80, f"expected 0.80, got {score:.4f}"


# --------------------------------------------------------------------------- #
# Shared corpus fixture for the corpus-level tests.
# --------------------------------------------------------------------------- #
def _load_index():
    docs = load_corpus(os.path.join(_ROOT, "corpus_100.txt"))
    index = build_inverted_index(docs)
    return docs, index


# --------------------------------------------------------------------------- #
# 2. Ten free-text queries -> well-formed, correctly ordered results.
# --------------------------------------------------------------------------- #
FREE_TEXT_QUERIES = [
    "cotton shirt",
    "black t-shirt",
    "denim jeans",
    "printed saree",
    "linen kurta",
    "midi dress",
    "crew neck",
    "slim fit shirt",
    "breathable fabric",
    "cotton",
]


def test_ten_free_text_queries_are_well_formed():
    docs, index = _load_index()
    for query in FREE_TEXT_QUERIES:
        hits = rank(query, index, docs)
        assert 1 <= len(hits) <= 10, f"{query!r}: got {len(hits)} hits"
        # every docid is real
        for docid, _score in hits:
            assert docid in docs, f"{query!r}: unknown docid {docid}"
        # scores are sorted descending, ties broken by docid ascending
        assert hits == sorted(hits, key=lambda p: (-p[1], p[0])), \
            f"{query!r}: results not correctly ordered"


# --------------------------------------------------------------------------- #
# 3. Edge cases.
# --------------------------------------------------------------------------- #
def test_out_of_vocabulary_returns_empty():
    docs, index = _load_index()
    assert rank("silk leather", index, docs) == []


def test_all_idf_zero_returns_empty():
    """"festive" and "wear" occur in every doc -> idf 0 -> query norm 0 -> []."""
    docs, index = _load_index()
    assert rank("festive wear", index, docs) == []


def _run():
    tests = [
        ("0.8 lecture example", test_lecture_example_scores_0_8),
        ("10 free-text queries", test_ten_free_text_queries_are_well_formed),
        ("out-of-vocabulary -> []", test_out_of_vocabulary_returns_empty),
        ("all idf 0 -> []", test_all_idf_zero_returns_empty),
    ]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run()
