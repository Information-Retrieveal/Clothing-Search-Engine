"""
NOVELTY - Proximity-boosted "Smart Search".  [Integration]
===========================================================

Base requirement keeps free-text VSM ranking and phrase/proximity search as two
SEPARATE modes.  The novelty fuses them into one smarter ranker, exactly along
the lines of IIR Ch. 7 ("Scoring and results assembly"):

  * Query parser (IIR 7.2.3): try the query as an exact PHRASE first; phrase
    matches are the most precise, so they are promoted.
  * Query-term proximity (IIR 7.2.2): documents in which the query terms occur
    in a SMALL window (close together) are better than documents where they are
    scattered.  We measure the smallest window containing all query terms.
  * Net score (IIR 7.1.4 / 7.2.3): combine the signals linearly, just like the
    lecture's  net-score(q,d) = cosine(q,d) + other signals.

        net(q, d) = cosine_lnc_ltc(q, d)            (relevance,  Person A)
                  + W_PHRASE * [exact phrase in d]   (arrangement, Person B)
                  + W_PROX  * 1 / smallest_window    (proximity,   Person B)

Why it is a genuine improvement (shown in the report / run_all_queries):
  * `cotton shirt`: plain VSM puts T-shirts at ranks 1-4 (they merely contain
    both words); Smart Search lifts the actual "Cotton Shirt" products to the
    top because the words are adjacent there.
  * `festive wear`: plain VSM returns NOTHING (both terms have idf 0), but Smart
    Search still returns the 10 sarees via the phrase/proximity signal.

This uses BOTH halves of the project (VSM + positional index) and needs only the
data structures already built - no new index.
"""

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_query          # noqa: E402
from vsm import rank                                 # noqa: E402
from phrase_search import phrase_search              # noqa: E402
from positional_index import get_postings            # noqa: E402

# Signal weights.  cosine is ~[0, 0.4] on this corpus, so these let phrase /
# proximity re-order near-ties without swamping a genuinely higher cosine.
W_PHRASE = 0.5     # bonus for an exact adjacent-phrase match
W_PROX = 0.30      # weight on the (1 / smallest-window) proximity signal


def smallest_window(distinct_terms, positions_by_term):
    """Smallest span (in token positions) covering >=1 occurrence of each term.

    positions_by_term: {term: [sorted positions]} for the terms present in the
    document.  Returns None if any query term is missing from the document.
    For an adjacent phrase of n terms the window is exactly n (the minimum).
    (This is the IIR 7.2.2 "smallest window containing all query terms".)
    """
    if any(t not in positions_by_term for t in distinct_terms):
        return None
    events = sorted((p, t) for t in distinct_terms for p in positions_by_term[t])
    need = len(distinct_terms)
    count = defaultdict(int)
    have = 0
    left = 0
    best = None
    for right in range(len(events)):
        _, t_right = events[right]
        if count[t_right] == 0:
            have += 1
        count[t_right] += 1
        while have == need:                          # shrink from the left
            span = events[right][0] - events[left][0] + 1
            if best is None or span < best:
                best = span
            _, t_left = events[left]
            count[t_left] -= 1
            if count[t_left] == 0:
                have -= 1
            left += 1
    return best


def smart_search(query, index, pindex, docs, top_k=10):
    """Return proximity-boosted ranking as a list of result dicts.

    Each result: {docid, score, cosine, phrase (bool), window (int|None)}.
    Sorted by net score descending, ties broken by docID ascending.
    """
    q_terms = preprocess_query(query)
    if not q_terms:
        return []
    distinct = set(q_terms)

    # Relevance signal: lnc.ltc cosine for every doc with a non-zero cosine.
    cosine = dict(rank(query, index, docs, top_k=len(docs)))

    # Positional signals (only meaningful for multi-term queries).
    phrase_hits = set(phrase_search(query, pindex)) if len(distinct) >= 2 else set()
    postings = {t: get_postings(pindex, t) for t in distinct}

    # Candidate set: any doc with cosine, an exact phrase, or all query terms.
    all_terms_docs = None
    for t in distinct:
        docs_with_t = set(postings[t])
        all_terms_docs = docs_with_t if all_terms_docs is None else (all_terms_docs & docs_with_t)
    all_terms_docs = all_terms_docs or set()
    candidates = set(cosine) | phrase_hits | (all_terms_docs if len(distinct) >= 2 else set())

    results = []
    for docid in candidates:
        cos = cosine.get(docid, 0.0)
        phrase_flag = docid in phrase_hits
        window = None
        prox = 0.0
        if len(distinct) >= 2:
            here = {t: postings[t][docid] for t in distinct if docid in postings[t]}
            window = smallest_window(distinct, here)
            if window:
                prox = 1.0 / window
        net = cos + W_PHRASE * (1.0 if phrase_flag else 0.0) + W_PROX * prox
        results.append({
            "docid": docid, "score": net, "cosine": cos,
            "phrase": phrase_flag, "window": window,
        })

    results.sort(key=lambda r: (-r["score"], r["docid"]))
    return results[:top_k]


if __name__ == "__main__":
    from preprocessing import load_corpus
    from inverted_index import build_inverted_index
    from positional_index import build_positional_index

    docs = load_corpus(os.path.join(os.path.dirname(__file__), "..", "corpus_100.txt"))
    index = build_inverted_index(docs)
    pindex = build_positional_index(docs)

    for q in ("cotton shirt", "festive wear", "denim jeans"):
        print(f"\n=== Smart Search: {q!r} ===")
        for i, r in enumerate(smart_search(q, index, pindex, docs, top_k=5), 1):
            tag = "PHRASE" if r["phrase"] else (f"win={r['window']}" if r["window"] else "-")
            print(f"  {i}. {r['docid']}  net={r['score']:.4f} (cos={r['cosine']:.4f}, {tag})"
                  f"  {docs[r['docid']]['category']:10s} {docs[r['docid']]['title']}")
