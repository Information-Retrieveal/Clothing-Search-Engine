"""
Part A - Inverted index  (Person A: ranking engine)
===================================================

Builds the classic inverted index of IIR Ch. 1 / Ch. 2: one postings list per
term, plus the document frequency (df) that lnc.ltc needs for idf.

Data shape (frozen in CONTRACT.md so Person B's positional index and this one
never collide):

    build_inverted_index(docs) -> dict
        term -> {"df": int, "postings": {docid: tf}}      # tf = raw term count

    * df       = number of DISTINCT documents the term occurs in.
    * postings = docid -> term frequency (raw count) inside that document.

Every term here has already been through the single shared pipeline in
`preprocessing.analyze()` (lower-case -> stop-word removal -> Porter stem), so
the index keys line up exactly with the query terms produced by
`preprocess_query()`.
"""

from collections import Counter


def build_inverted_index(docs):
    """Build ``term -> {"df", "postings"}`` from the analyzed corpus.

    Parameters
    ----------
    docs : dict
        Output of ``preprocessing.load_corpus`` -
        ``docid -> {"category", "title", "text", "terms"}`` where ``terms`` is
        the analyzed ``[(term, position), ...]`` list for that document.

    Returns
    -------
    dict
        ``term -> {"df": int, "postings": {docid: tf}}``.
    """
    index = {}
    for docid, rec in docs.items():
        # Raw term frequency = how many times the term occurs in this document.
        tfs = Counter(term for term, _pos in rec["terms"])
        for term, tf in tfs.items():
            entry = index.get(term)
            if entry is None:
                entry = {"df": 0, "postings": {}}
                index[term] = entry
            entry["df"] += 1              # one more document contains this term
            entry["postings"][docid] = tf
    return index


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import load_corpus

    corpus = os.path.join(os.path.dirname(__file__), "..", "corpus_100.txt")
    docs = load_corpus(corpus)
    index = build_inverted_index(docs)

    print(f"Vocabulary size : {len(index)} distinct terms")
    for term in ("cotton", "shirt", "wear", "festiv"):
        if term in index:
            e = index[term]
            print(f"  {term:10s} df={e['df']:3d}  postings(first 5)="
                  f"{list(e['postings'].items())[:5]}")