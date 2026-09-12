"""
Tests for the NOVELTY - proximity-boosted Smart Search.  [Integration]
=======================================================================

Structural checks only (no hard-coded document IDs):

  * smallest_window computes the correct minimum span.
  * for 'cotton shirt', every exact-phrase document is ranked ABOVE every
    non-phrase document (the query-parser promotion works), and the phrase set
    matches phrase_search.
  * for 'festive wear', plain VSM returns nothing (idf 0) yet Smart Search
    returns results whose top items are the phrase-matching sarees - i.e. the
    novelty rescues the degenerate query.
  * net score >= cosine for every result, and results are sorted by net score.
  * a single-term query falls back to plain VSM ordering.

Run:  python tests/test_smart_search.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from preprocessing import load_corpus                 # noqa: E402
from inverted_index import build_inverted_index       # noqa: E402
from vsm import rank                                   # noqa: E402
from positional_index import build_positional_index   # noqa: E402
from phrase_search import phrase_search                # noqa: E402
from smart_search import smart_search, smallest_window  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = load_corpus(os.path.join(_ROOT, "corpus_100.txt"))
INDEX = build_inverted_index(DOCS)
PINDEX = build_positional_index(DOCS)

_checks = 0


def check(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError("FAILED: " + msg)


def test_smallest_window():
    print("[1] smallest_window")
    check(smallest_window({"a", "b"}, {"a": [0, 5], "b": [1, 9]}) == 2, "adjacent window should be 2")
    check(smallest_window({"a", "b"}, {"a": [2], "b": [6]}) == 5, "span 2..6 should be 5")
    check(smallest_window({"a", "b"}, {"a": [3]}) is None, "missing term -> None")


def test_phrase_docs_ranked_first():
    print("[2] phrase matches promoted above non-phrase (cotton shirt)")
    results = smart_search("cotton shirt", INDEX, PINDEX, DOCS, top_k=100)
    phrase_set = set(phrase_search("cotton shirt", PINDEX))
    flags = [r["docid"] in phrase_set for r in results]
    # every phrase doc appears before every non-phrase doc:
    last_phrase = max((i for i, f in enumerate(flags) if f), default=-1)
    first_non = next((i for i, f in enumerate(flags) if not f), len(flags))
    check(last_phrase < first_non, "a non-phrase doc outranked a phrase doc")
    check({r["docid"] for r in results if r["phrase"]} == phrase_set,
          "smart_search phrase flags disagree with phrase_search")


def test_degenerate_query_rescued():
    print("[3] 'festive wear': VSM empty, Smart Search returns sarees")
    check(rank("festive wear", INDEX, DOCS) == [], "VSM should be empty for festive wear")
    results = smart_search("festive wear", INDEX, PINDEX, DOCS, top_k=10)
    check(len(results) > 0, "Smart Search should still return results")
    top5 = results[:5]
    check(all(r["phrase"] for r in top5), "top-5 should be exact-phrase matches")
    check(all(DOCS[r["docid"]]["category"] == "Saree" for r in top5), "top-5 should be sarees")


def test_scores_wellformed():
    print("[4] net >= cosine and sorted by net")
    results = smart_search("denim jeans", INDEX, PINDEX, DOCS, top_k=10)
    for r in results:
        check(r["score"] >= r["cosine"] - 1e-9, "net score below cosine")
    check(results == sorted(results, key=lambda r: (-r["score"], r["docid"])), "not sorted by net")


def test_single_term_falls_back_to_vsm():
    print("[5] single-term query == plain VSM ordering")
    smart = [r["docid"] for r in smart_search("saree", INDEX, PINDEX, DOCS, top_k=10)]
    vsm = [d for d, _ in rank("saree", INDEX, DOCS, top_k=10)]
    check(smart == vsm, "single-term smart search should match VSM order")


if __name__ == "__main__":
    print(f"Loaded {len(DOCS)} docs.\n")
    for fn in (test_smallest_window, test_phrase_docs_ranked_first,
               test_degenerate_query_rescued, test_scores_wellformed,
               test_single_term_falls_back_to_vsm):
        fn()
    print(f"\nAll Smart Search checks passed ({_checks} assertions).")
