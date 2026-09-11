"""
Part E tests - Positional / phrase / proximity search.  [Person B]
===================================================================

Mandatory coverage from Part E (the phrase/proximity half):
    * at least 5 exact phrase queries
    * at least 3 proximity queries with DIFFERENT values of k
    * at least one query whose term does NOT occur in the corpus
    * an empty-result phrase (words exist but never adjacent)

Per the assignment, we DO NOT hard-code expected document IDs.  Instead every
check asserts a STRUCTURAL property that must hold for a correct positional
engine, e.g.:
    - every doc a phrase returns really does contain that phrase (re-derived
      independently from the corpus, not from the index under test);
    - a phrase's result docs are a subset of its ordered-proximity result;
    - a larger k returns a superset of a smaller k.

Run:  python tests/test_positional.py   (exits non-zero if any check fails)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from preprocessing import load_corpus, preprocess_query   # noqa: E402
from positional_index import build_positional_index       # noqa: E402
from phrase_search import phrase_search, proximity_search  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus_100.txt")

DOCS = load_corpus(CORPUS)
PINDEX = build_positional_index(DOCS)

_checks = 0


def check(condition, message):
    global _checks
    _checks += 1
    if not condition:
        raise AssertionError("FAILED: " + message)


# --- independent ground-truth helpers (do NOT use the index under test) -------
def doc_has_terms(docid, terms):
    """True if the document contains every term at least once."""
    present = {t for t, _pos in DOCS[docid]["terms"]}
    return all(t in present for t in terms)


def doc_has_phrase(docid, terms):
    """Re-derive phrase membership straight from the corpus token stream."""
    pos2term = {pos: t for t, pos in DOCS[docid]["terms"]}
    firsts = [pos for t, pos in DOCS[docid]["terms"] if t == terms[0]]
    return any(all(pos2term.get(p + i) == terms[i] for i in range(len(terms))) for p in firsts)


# ---------------------------------------------------------------------------
# 1. Five exact phrase queries - structural correctness (no hard-coded IDs)
# ---------------------------------------------------------------------------
def test_phrase_queries():
    phrases = ["cotton shirt", "stretch denim", "festive wear",
               "winter wear", "high waist"]
    print("[1] Exact phrase queries")
    for phrase in phrases:
        terms = preprocess_query(phrase)
        res = phrase_search(phrase, PINDEX)
        print(f"    '{phrase}': {len(res)} docs -> {sorted(res)}")

        for docid, starts in res.items():
            # a) returned doc must truly contain all the terms ...
            check(doc_has_terms(docid, terms),
                  f"'{phrase}' returned {docid} which lacks a phrase term")
            # b) ... AND actually contain the phrase (independent re-derivation)
            check(doc_has_phrase(docid, terms),
                  f"'{phrase}' returned {docid} but phrase is not really adjacent")
            # c) every reported start position is a real occurrence
            pos2term = {pos: t for t, pos in DOCS[docid]["terms"]}
            for p in starts:
                check(all(pos2term.get(p + i) == terms[i] for i in range(len(terms))),
                      f"'{phrase}' {docid} bad start pos {p}")
        # d) completeness: no doc that DOES contain the phrase was missed
        for docid in DOCS:
            if doc_has_phrase(docid, terms):
                check(docid in res, f"'{phrase}' missed {docid} which contains the phrase")


# ---------------------------------------------------------------------------
# 2. Three proximity queries with DIFFERENT k
# ---------------------------------------------------------------------------
def test_proximity_queries():
    queries = [("cotton", 2, "shirt"), ("stretch", 4, "denim"), ("winter", 3, "wear")]
    print("[2] Ordered proximity queries (different k)")
    ks = set()
    for t1, k, t2 in queries:
        ks.add(k)
        res = proximity_search(t1, k, t2, PINDEX)
        print(f"    {t1} WITHIN/{k} {t2}: {len(res)} docs -> {sorted(res)}")
        for docid, pairs in res.items():
            check(doc_has_terms(docid, preprocess_query(t1) + preprocess_query(t2)),
                  f"{t1}/{t2} returned {docid} missing a term")
            for a, b in pairs:                       # ordered + within window
                check(0 < b - a <= k, f"{t1} WITHIN/{k} {t2} bad pair {(a, b)} in {docid}")
    check(len(ks) >= 3, "need 3 DIFFERENT k values")


# ---------------------------------------------------------------------------
# 3. Relationship checks: phrase implies proximity; larger k is a superset
# ---------------------------------------------------------------------------
def test_phrase_vs_proximity():
    print("[3] Phrase-vs-proximity relationships")
    phrase_docs = set(phrase_search("cotton shirt", PINDEX))
    near_docs = set(proximity_search("cotton", 3, "shirt", PINDEX))
    # Adjacent phrase => within any k>=1, so phrase docs subset of proximity docs.
    check(phrase_docs <= near_docs, "phrase 'cotton shirt' not a subset of WITHIN/3")
    # Monotonic in k: within/2 is a subset of within/4.
    small = set(proximity_search("cotton", 2, "shirt", PINDEX))
    large = set(proximity_search("cotton", 4, "shirt", PINDEX))
    check(small <= large, "WITHIN/2 not a subset of WITHIN/4")
    print(f"    phrase(cotton shirt)={len(phrase_docs)}  <=  "
          f"WITHIN/3={len(near_docs)};  WITHIN/2={len(small)} <= WITHIN/4={len(large)}")


# ---------------------------------------------------------------------------
# 4. Empty result: words exist in the corpus but never adjacent
# ---------------------------------------------------------------------------
def test_empty_phrase():
    print("[4] Empty-result phrase")
    res = phrase_search("zip closure", PINDEX)   # 'zip' only in jeans, 'closure' only in shirts
    print(f"    'zip closure': {len(res)} docs")
    check(res == {}, "'zip closure' should return no documents")


# ---------------------------------------------------------------------------
# 5. Out-of-vocabulary term (does not occur in the corpus)
# ---------------------------------------------------------------------------
def test_out_of_vocabulary():
    print("[5] Out-of-vocabulary term")
    check(phrase_search("silk saree", PINDEX) == {}, "'silk' is OOV -> phrase must be empty")
    check(proximity_search("silk", 3, "saree", PINDEX) == {}, "'silk' is OOV -> proximity must be empty")
    print("    'silk' absent -> phrase and proximity both empty (as expected)")


if __name__ == "__main__":
    print(f"Loaded {len(DOCS)} docs, {len(PINDEX)} terms.\n")
    for fn in (test_phrase_queries, test_proximity_queries, test_phrase_vs_proximity,
               test_empty_phrase, test_out_of_vocabulary):
        fn()
    print(f"\nAll positional/phrase/proximity checks passed ({_checks} assertions).")
