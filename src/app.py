"""
Part D - Clothing search engine command-line interface.  [Integration]
=======================================================================

Wires BOTH retrieval engines behind one simple menu:

    1. Free-text ranked search      -> Vector Space Model, lnc.ltc cosine  (Person A)
    2. Exact phrase search          -> positional index                    (Person B)
    3. Proximity search (WITHIN/k)  -> positional index                    (Person B)

Requirements covered (Part D):
    * user types a free-text query; top-10 results show docID, category/title
      and the cosine score;
    * a second mode does phrase / proximity search on the positional index;
    * phrase and proximity results PRINT THE MATCHING TERM POSITIONS as
      evidence that the positional index is actually being used.

Run from the repo root:  python src/app.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_corpus, preprocess_query   # noqa: E402
from inverted_index import build_inverted_index           # noqa: E402
from vsm import rank                                       # noqa: E402
from positional_index import build_positional_index       # noqa: E402
from phrase_search import phrase_search, proximity_search  # noqa: E402
from smart_search import smart_search                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(_ROOT, "corpus_100.txt")

_PROXIMITY_RE = re.compile(r"^\s*(.+?)\s+WITHIN\s*/\s*(\d+)\s+(.+?)\s*$", re.IGNORECASE)


def _label(docs, docid):
    """Human-readable '<category> | <title>' for a document."""
    rec = docs[docid]
    return f"{rec['category']:10s} | {rec['title']}"


# --------------------------------------------------------------------------- #
# Mode 1 - free-text ranked retrieval (VSM lnc.ltc)
# --------------------------------------------------------------------------- #
def do_free_text(query, index, docs):
    terms = preprocess_query(query)
    print(f"\n  query terms after preprocessing: {terms}")
    hits = rank(query, index, docs, top_k=10)
    if not hits:
        print("  No rankable results — every query term is out-of-vocabulary or")
        print("  occurs in all documents (idf 0). Try mode 2/3 for such phrases.")
        return
    print(f"  Top {len(hits)} results (lnc.ltc cosine):\n")
    print(f"    {'rank':<5}{'docID':<7}{'score':<9}category / title")
    print(f"    {'-'*4:<5}{'-'*5:<7}{'-'*7:<9}{'-'*30}")
    for i, (docid, score) in enumerate(hits, 1):
        print(f"    {i:<5}{docid:<7}{score:<9.4f}{_label(docs, docid)}")


# --------------------------------------------------------------------------- #
# Mode 2 - exact phrase search (positional index)
# --------------------------------------------------------------------------- #
def do_phrase(phrase, pindex, docs):
    terms = preprocess_query(phrase)
    print(f"\n  phrase terms after preprocessing: {terms}")
    results = phrase_search(phrase, pindex)
    if not results:
        print("  No document contains that exact phrase.")
        return
    print(f"  {len(results)} document(s) contain the phrase "
          f"(showing start positions as evidence):\n")
    for docid in sorted(results):
        starts = results[docid]
        print(f"    {docid}  start_pos={starts}  {_label(docs, docid)}")


# --------------------------------------------------------------------------- #
# Mode 3 - ordered proximity search (positional index)
# --------------------------------------------------------------------------- #
def do_proximity(raw, pindex, docs):
    m = _PROXIMITY_RE.match(raw)
    if not m:
        print("  Format:  <term1> WITHIN/<k> <term2>   e.g.  cotton WITHIN/3 shirt")
        return
    t1, k, t2 = m.group(1), int(m.group(2)), m.group(3)
    print(f"\n  {t1!r} followed by {t2!r} within {k} positions (ordered):")
    results = proximity_search(t1, k, t2, pindex)
    if not results:
        print(f"  No document has '{t1}' then '{t2}' within {k} positions.")
        return
    print(f"  {len(results)} document(s) match "
          f"(showing matching (pos_{t1}, pos_{t2}) pairs as evidence):\n")
    for docid in sorted(results):
        print(f"    {docid}  pairs={results[docid]}  {_label(docs, docid)}")


def do_smart_search(query, index, pindex, docs):
    print(f"\n  query terms after preprocessing: {preprocess_query(query)}")
    results = smart_search(query, index, pindex, docs, top_k=10)
    if not results:
        print("  No results.")
        return
    print("  Smart-ranked results  (net = cosine + phrase-bonus + proximity):\n")
    print(f"    {'rank':<5}{'docID':<7}{'net':<8}{'cosine':<8}{'signal':<9}category / title")
    print(f"    {'-'*4:<5}{'-'*5:<7}{'-'*6:<8}{'-'*6:<8}{'-'*7:<9}{'-'*30}")
    for i, r in enumerate(results, 1):
        signal = "phrase" if r["phrase"] else (f"win={r['window']}" if r["window"] else "-")
        print(f"    {i:<5}{r['docid']:<7}{r['score']:<8.4f}{r['cosine']:<8.4f}{signal:<9}{_label(docs, r['docid'])}")


MENU = """
============================================================
  Clothing Search Engine  (100 products)
============================================================
  1) Free-text ranked search   (VSM, lnc.ltc cosine)
  2) Exact phrase search       (positional index)
  3) Proximity search WITHIN/k (positional index)
  4) Smart search   [NOVELTY]  (VSM + phrase + proximity)
  0) Quit
------------------------------------------------------------"""


def main():
    print("Loading corpus and building indexes ...")
    docs = load_corpus(CORPUS_PATH)
    inv_index = build_inverted_index(docs)
    pos_index = build_positional_index(docs)
    print(f"Ready: {len(docs)} documents, {len(inv_index)} terms indexed.")

    while True:
        print(MENU)
        try:
            choice = input("  Select mode (0-4): ").strip()
        except EOFError:
            break

        if choice == "0":
            break
        elif choice == "1":
            try:
                q = input("  Enter free-text query: ").strip()
            except EOFError:
                break
            if q:
                do_free_text(q, inv_index, docs)
        elif choice == "2":
            try:
                q = input("  Enter exact phrase (e.g. cotton shirt): ").strip()
            except EOFError:
                break
            if q:
                do_phrase(q, pos_index, docs)
        elif choice == "3":
            try:
                q = input("  Enter proximity query (e.g. cotton WITHIN/3 shirt): ").strip()
            except EOFError:
                break
            if q:
                do_proximity(q, pos_index, docs)
        elif choice == "4":
            try:
                q = input("  Enter query for smart search: ").strip()
            except EOFError:
                break
            if q:
                do_smart_search(q, inv_index, pos_index, docs)
        else:
            print("  Invalid choice — pick 0, 1, 2, 3 or 4.")

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
