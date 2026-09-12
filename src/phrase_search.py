"""
Part C - Phrase and proximity search over the positional index.  [Person B]
============================================================================

This is the payoff of the positional index (IIR Ch. 2):

  * phrase_search("cotton shirt")   -> docs where the words are ADJACENT,
    not merely co-occurring in the same document.
  * proximity_search("cotton", 3, "shirt") -> docs where "shirt" occurs after
    "cotton" within k positions  (ordered "cotton WITHIN/3 shirt").

Both use the same preprocessing pipeline as indexing (so query terms line up
with indexed terms) and BOTH RETURN THE ACTUAL MATCHING POSITIONS - the lecture
stresses that a proximity/phrase search should return the positions, not just a
list of docIDs (needed for snippets / evidence).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_query          # noqa: E402
from positional_index import get_postings            # noqa: E402


def phrase_search(phrase, pindex):
    """Exact phrase search via positional intersection.

    Returns {docID: [start_pos, ...]} - each start_pos is where the phrase
    begins in that document.  Empty dict if the phrase never occurs.
    """
    terms = preprocess_query(phrase)
    if not terms:
        return {}

    # Every phrase term must exist in the index, else no document can match.
    postings = []
    for t in terms:
        p = get_postings(pindex, t)
        if not p:
            return {}
        postings.append(p)

    # Candidate documents = those containing ALL phrase terms.
    candidates = set(postings[0])
    for p in postings[1:]:
        candidates &= set(p)

    results = {}
    for docid in candidates:
        # Pre-build position sets for the 2nd..last terms for O(1) lookup.
        later_sets = [set(postings[i][docid]) for i in range(1, len(terms))]
        starts = []
        for p0 in postings[0][docid]:
            # term[i] must sit exactly at p0 + i (consecutive positions).
            if all((p0 + i) in later_sets[i - 1] for i in range(1, len(terms))):
                starts.append(p0)
        if starts:
            results[docid] = sorted(starts)
    return results


def proximity_search(term1, k, term2, pindex):
    """Ordered proximity search: term1 followed by term2 within k positions.

    Matches the "cotton WITHIN/3 shirt" style: keep (p1, p2) with
    0 < p2 - p1 <= k.  Returns {docID: [(pos1, pos2), ...]}.
    """
    q1, q2 = preprocess_query(term1), preprocess_query(term2)
    if not q1 or not q2:
        return {}
    t1, t2 = q1[0], q2[0]

    p1, p2 = get_postings(pindex, t1), get_postings(pindex, t2)
    results = {}
    for docid in set(p1) & set(p2):                  # docs containing both terms
        positions2 = p2[docid]
        pairs = [(a, b) for a in p1[docid] for b in positions2 if 0 < b - a <= k]
        if pairs:
            results[docid] = sorted(pairs)
    return results


if __name__ == "__main__":
    from positional_index import build_positional_index
    from preprocessing import load_corpus

    pindex = build_positional_index(load_corpus())

    print("=== Exact phrase queries ===")
    for phrase in ["cotton shirt", "stretch denim", "festive wear", "winter wear",
                   "regular fit", "breathable fabric", "high waist", "zip closure"]:
        res = phrase_search(phrase, pindex)
        docs = sorted(res)
        print(f"  '{phrase}': {len(docs)} docs -> {docs}")

    print("\n=== Ordered proximity queries (WITHIN/k) ===")
    for a, k, b in [("cotton", 3, "shirt"), ("stretch", 4, "denim"),
                    ("winter", 3, "wear"), ("festive", 4, "kurta")]:
        res = proximity_search(a, k, b, pindex)
        docs = sorted(res)
        print(f"  {a} WITHIN/{k} {b}: {len(docs)} docs -> {docs}")
        if docs:
            d0 = docs[0]
            print(f"      e.g. {d0} matching (pos_{a}, pos_{b}) pairs: {res[d0]}")
