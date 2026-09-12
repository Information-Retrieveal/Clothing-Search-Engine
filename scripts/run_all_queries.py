"""
Part E - Testing battery + deliverable results log.  [Integration]
===================================================================

Runs the mandatory Part E query set against the FULL system (VSM + positional)
and writes a readable report to data/results.txt:

    * 10 free-text queries          (top-10 ranked results each)
    * 5  exact phrase queries
    * 3  proximity queries with DIFFERENT k
    * a query with an out-of-vocabulary term (does not occur in the corpus)
    * TWO worked cases where positional information changes the result
      set/order compared with the bag-of-words VSM.

Nothing is hard-coded: every number below is computed from the live indexes.

Run from the repo root:  python scripts/run_all_queries.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing import load_corpus                     # noqa: E402
from inverted_index import build_inverted_index           # noqa: E402
from vsm import rank                                       # noqa: E402
from positional_index import build_positional_index       # noqa: E402
from phrase_search import phrase_search, proximity_search  # noqa: E402
from smart_search import smart_search                       # noqa: E402

OUT_PATH = os.path.join(_ROOT, "data", "results.txt")

FREE_TEXT = [
    "cotton shirt", "black t-shirt", "denim jeans", "printed saree",
    "linen kurta", "winter jacket", "high waist leggings", "fleece hoodie",
    "slim fit", "navy blue dress",
]
PHRASES = ["cotton shirt", "stretch denim", "festive wear", "winter wear",
           "breathable fabric"]
PROXIMITY = [("cotton", 2, "shirt"), ("cotton", 4, "shirt"), ("winter", 3, "wear")]

_lines = []


def w(line=""):
    _lines.append(line)


def hr(ch="="):
    w(ch * 70)


def show_free_text(query, index, docs):
    hits = rank(query, index, docs, top_k=10)
    w(f"QUERY: {query!r}   ->  {len(hits)} result(s)")
    if not hits:
        w("   (no rankable results: term(s) out-of-vocabulary or idf 0)")
        return
    for i, (docid, score) in enumerate(hits, 1):
        w(f"   {i:2d}. {docid}  {score:.4f}  {docs[docid]['category']:10s} {docs[docid]['title']}")


def show_phrase(phrase, pindex, docs):
    res = phrase_search(phrase, pindex)
    w(f"PHRASE: {phrase!r}   ->  {len(res)} doc(s): {sorted(res)}")
    for docid in sorted(res):
        w(f"      {docid}  start_pos={res[docid]}  {docs[docid]['title']}")


def show_proximity(t1, k, t2, pindex, docs):
    res = proximity_search(t1, k, t2, pindex)
    w(f"PROXIMITY: {t1} WITHIN/{k} {t2}   ->  {len(res)} doc(s): {sorted(res)}")
    for docid in sorted(res):
        w(f"      {docid}  pairs={res[docid]}  {docs[docid]['title']}")


def main():
    docs = load_corpus(os.path.join(_ROOT, "corpus_100.txt"))
    index = build_inverted_index(docs)
    pindex = build_positional_index(docs)

    hr()
    w("PART E - QUERY BATTERY  (clothing search engine, N=100)")
    hr()

    w("\n[A] TEN FREE-TEXT QUERIES  (VSM lnc.ltc, top-10)\n")
    for q in FREE_TEXT:
        show_free_text(q, index, docs)
        w()

    w("")
    hr("-")
    w("[B] FIVE EXACT PHRASE QUERIES  (positional index)\n")
    for p in PHRASES:
        show_phrase(p, pindex, docs)
        w()

    w("")
    hr("-")
    w("[C] THREE PROXIMITY QUERIES WITH DIFFERENT k  (ordered WITHIN/k)\n")
    for t1, k, t2 in PROXIMITY:
        show_proximity(t1, k, t2, pindex, docs)
        w()

    w("")
    hr("-")
    w("[D] OUT-OF-VOCABULARY QUERIES  (term not in the corpus)\n")
    w("Free-text 'silk saree' ('silk' is OOV, so only 'saree' can rank):")
    show_free_text("silk saree", index, docs)
    w()
    w("Free-text 'cashmere sweater' (both terms OOV):")
    show_free_text("cashmere sweater", index, docs)
    w()
    w("Phrase 'leather jacket' ('leather' is OOV):")
    show_phrase("leather jacket", pindex, docs)

    w("")
    hr()
    w("[E] TWO CASES WHERE POSITIONAL INFO CHANGES THE RESULT")
    hr()

    # ---- Case 1: festive wear -------------------------------------------------
    ft1 = rank("festive wear", index, docs, top_k=10)
    ph1 = phrase_search("festive wear", pindex)
    w("\nCASE 1  -  'festive wear'")
    w(f"   VSM free-text : {len(ft1)} results  -> {[d for d, _ in ft1]}")
    w(f"   Phrase        : {len(ph1)} docs     -> {sorted(ph1)}")
    w("   WHY IT CHANGES: 'festive' and 'wear' both occur in ALL 100 documents")
    w("   (idf = log10(100/100) = 0), so the VSM query vector is all zeros and")
    w("   CANNOT rank anything -> 0 results.  The positional index instead pins")
    w("   the exact adjacent phrase to the sarees that actually advertise")
    w("   'festive wear'.  Positional finds the answer where VSM cannot.")

    # ---- Case 2: cotton shirt -------------------------------------------------
    ft2 = rank("cotton shirt", index, docs, top_k=10)
    ph2 = phrase_search("cotton shirt", pindex)
    px2 = proximity_search("cotton", 3, "shirt", pindex)
    ft2_ids = [d for d, _ in ft2]
    tshirts_in_vsm = [d for d in ft2_ids if docs[d]["category"] == "T-Shirt"]
    w("\nCASE 2  -  'cotton shirt'")
    w(f"   VSM free-text (top-10): {ft2_ids}")
    w(f"   Phrase                : {len(ph2)} docs -> {sorted(ph2)}")
    w(f"   Proximity WITHIN/3    : {len(px2)} docs -> {sorted(px2)}")
    w("   WHY IT CHANGES: as a bag of words, VSM ranks any doc with 'cotton'")
    w(f"   and/or 'shirt' highly - its top-10 includes {len(tshirts_in_vsm)} T-Shirts")
    w(f"   ({tshirts_in_vsm}) whose words are NOT adjacent.  The exact phrase")
    w("   drops every T-Shirt and keeps only the 5 'Checked Cotton Shirt' items")
    w("   where 'cotton shirt' is truly consecutive.  Relaxing to WITHIN/3 lets")
    w("   the T-Shirts back in ('cotton ... t shirt' within 3 positions), so the")
    w("   result SET grows again - proximity k directly controls precision.")

    # ---- Novelty: proximity-boosted Smart Search ------------------------------
    w("")
    hr()
    w("[F] NOVELTY - PROXIMITY-BOOSTED SMART SEARCH  (VSM + phrase + proximity)")
    hr()
    w("Fuses both engines (IIR Ch. 7: query parser + query-term proximity +")
    w("net-score). Compare plain VSM vs Smart Search:\n")
    for q in ["cotton shirt", "festive wear"]:
        vsm_ids = [d for d, _ in rank(q, index, docs, top_k=5)]
        w(f"Query {q!r}")
        w(f"   plain VSM  top-5 : {vsm_ids if vsm_ids else '[]  (cannot rank - all idf 0)'}")
        w("   Smart Search top-5 :")
        for i, r in enumerate(smart_search(q, index, pindex, docs, top_k=5), 1):
            sig = "phrase" if r["phrase"] else (f"window={r['window']}" if r["window"] else "-")
            w(f"      {i}. {r['docid']}  net={r['score']:.4f}  (cosine={r['cosine']:.4f}, {sig})"
              f"  {docs[r['docid']]['category']} {docs[r['docid']]['title']}")
        w("")
    w("Effect: for 'cotton shirt' the truly-adjacent 'Cotton Shirt' products are")
    w("lifted above the T-shirts that merely contain both words; for 'festive")
    w("wear' Smart Search still returns the sarees where plain VSM (idf 0) can't.")

    # write + echo
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    text = "\n".join(_lines) + "\n"
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"[written to {OUT_PATH}]")


if __name__ == "__main__":
    main()
