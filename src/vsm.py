"""
Part B - Vector Space Model, lnc.ltc cosine ranking  (Person A: ranking engine)
===============================================================================

Ranked retrieval exactly as taught in IIR Ch. 6 (tf-idf, cosine, the SMART
``lnc.ltc`` scheme - the same one used in the lecture's "car / auto insurance"
worked example that scores 0.8; that example is reproduced verbatim in
``tests/test_vsm.py`` to hard-verify the weighting).

Weighting (lnc.ltc, N = number of documents):

    Document term weight (lnc):  w = 1 + log10(tf)          then L2-normalize
                                 -> "l" log tf, "n" no idf, "c" cosine norm.
    Query    term weight (ltc):  w = (1 + log10(tf)) * log10(N / df)
                                 then L2-normalize
                                 -> "l" log tf, "t" idf,   "c" cosine norm.

    Cosine similarity = dot product of the two normalized vectors.

Degenerate case (documented limitation, and the Case-1 demo in the report):
    If every query term has idf 0 - i.e. it occurs in all N documents, e.g.
    "festive wear" over this corpus - the query vector is all zeros and cannot
    be normalized. ``rank`` then returns ``[]`` (free-text VSM genuinely cannot
    rank such a query; the positional engine handles it instead).
"""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_query


def _log_tf(tf):
    """Sub-linear tf scaling  1 + log10(tf)  (the "l" in lnc / ltc)."""
    return 1.0 + math.log10(tf)


def _l2_normalize(vec):
    """Cosine ("c") normalization: divide a term->weight dict by its L2 norm.

    Returns the zero-length vector unchanged (norm 0) - callers guard against
    that case explicitly.
    """
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm == 0.0:
        return dict(vec)
    return {term: w / norm for term, w in vec.items()}


def doc_vector(terms):
    """lnc document vector for one document's analyzed ``[(term, pos), ...]``.

    ``1 + log10(tf)`` per term, no idf, then L2-normalized.
    """
    tfs = Counter(term for term, _pos in terms)
    weights = {term: _log_tf(tf) for term, tf in tfs.items()}
    return _l2_normalize(weights)


def query_weights(query_tf, df, N):
    """ltc query weights (UN-normalized): ``(1 + log10(tf)) * log10(N/df)``.

    ``query_tf`` : term -> tf in the query.   ``df`` : term -> document freq.
    Terms absent from ``df`` (out of vocabulary) contribute nothing.
    """
    weights = {}
    for term, tf in query_tf.items():
        d = df.get(term, 0)
        if d <= 0:
            continue                     # out of vocabulary: no document has it
        weights[term] = _log_tf(tf) * math.log10(N / d)
    return weights


def cosine_lnc_ltc(query_tf, doc_tf, df, N):
    """Full lnc.ltc cosine from raw counts - the exact lecture computation.

    Used by :func:`rank` and directly by the unit test that reproduces the
    lecture's 0.8 worked example.  ``doc_tf`` : term -> tf in the document.
    """
    qv = _l2_normalize(query_weights(query_tf, df, N))
    dv = _l2_normalize({term: _log_tf(tf) for term, tf in doc_tf.items()})
    return sum(w * dv.get(term, 0.0) for term, w in qv.items())


def rank(query, index, docs, top_k=10):
    """Rank documents for a free-text ``query`` by lnc.ltc cosine.

    Parameters
    ----------
    query : str            raw query text (run through the shared pipeline here).
    index : dict           output of ``inverted_index.build_inverted_index``.
    docs  : dict           output of ``preprocessing.load_corpus``.
    top_k : int            maximum number of results (default 10).

    Returns
    -------
    list[(docid, score)]   score descending; ties broken by docid ascending.
                           ``[]`` if the query has no rankable term (all idf 0
                           or fully out of vocabulary) - see module docstring.
    """
    N = len(docs)
    q_terms = preprocess_query(query)
    if not q_terms:
        return []

    df = {term: entry["df"] for term, entry in index.items()}
    qv_raw = query_weights(Counter(q_terms), df, N)
    q_norm = math.sqrt(sum(w * w for w in qv_raw.values()))
    if q_norm == 0.0:
        return []                        # degenerate: nothing to rank on
    qv = {term: w / q_norm for term, w in qv_raw.items()}

    # Candidate docs = those sharing at least one (rankable) query term.
    candidates = set()
    for term in qv:
        candidates.update(index[term]["postings"])

    results = []
    for docid in candidates:
        dv = doc_vector(docs[docid]["terms"])
        score = sum(w * dv.get(term, 0.0) for term, w in qv.items())
        results.append((docid, score))

    results.sort(key=lambda pair: (-pair[1], pair[0]))
    return results[:top_k]


if __name__ == "__main__":
    from inverted_index import build_inverted_index
    from preprocessing import load_corpus

    corpus = os.path.join(os.path.dirname(__file__), "..", "corpus_100.txt")
    docs = load_corpus(corpus)
    index = build_inverted_index(docs)
    for q in ("cotton shirt", "denim jeans", "festive wear", "silk saree"):
        hits = rank(q, index, docs)
        print(f"\nQuery: {q!r}")
        if not hits:
            print("  (no rankable results - all terms idf 0 or out of vocabulary)")
        for docid, score in hits:
            print(f"  {docid}  {score:.4f}  {docs[docid]['title']}")
