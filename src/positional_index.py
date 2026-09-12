"""
Part C - Positional index.  [Person B]
=======================================

Extends the inverted index so every posting also stores the positions at which
the term occurs in the document (IIR Ch. 2, "Positional indexes"):

    term -> df -> [(docID, tf, [p1, p2, ...]), ...]

We store this as:
    { term: {"df": int, "postings": {docID: [pos, ...]}} }
where tf for a (term, doc) is simply len(positions).

Positions come from the SAME preprocessing pipeline used in Part A
(src/preprocessing.py), and are original token offsets (see the position policy
in that module), so phrase and proximity distances are true text distances.
"""

import sys
import os

# Make the shared pipeline importable whether run from repo root or src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_corpus  # noqa: E402


def build_positional_index(docs):
    """Build the positional index from the analyzed corpus.

    `docs` is the dict returned by preprocessing.load_corpus(): each value has a
    "terms" list of (term, position) pairs. We invert it into
    term -> {"df", "postings": {docID: [positions]}}.
    """
    index = {}
    for docid, rec in docs.items():
        for term, pos in rec["terms"]:
            entry = index.setdefault(term, {"df": 0, "postings": {}})
            positions = entry["postings"].setdefault(docid, [])
            positions.append(pos)

    # df = number of documents in the postings; positions are already in order.
    for term, entry in index.items():
        entry["df"] = len(entry["postings"])
    return index


def get_postings(index, term):
    """Return {docID: [positions]} for a term (empty dict if absent)."""
    entry = index.get(term)
    return entry["postings"] if entry else {}


def term_frequency(index, term, docid):
    """tf of a term in a document = number of stored positions."""
    return len(get_postings(index, term).get(docid, []))


if __name__ == "__main__":
    docs = load_corpus()
    pindex = build_positional_index(docs)
    print(f"Positional index built: {len(pindex)} unique terms over {len(docs)} docs.")

    # Show a sample posting in the (term -> df -> [(docID, tf, [positions])]) form.
    for term in ("cotton", "shirt", "festiv"):
        postings = get_postings(pindex, term)
        df = pindex[term]["df"] if term in pindex else 0
        sample = list(postings.items())[:3]
        shown = [(d, len(p), p) for d, p in sample]
        print(f"\nterm '{term}'  df={df}")
        print(f"  first postings (docID, tf, positions): {shown}")
