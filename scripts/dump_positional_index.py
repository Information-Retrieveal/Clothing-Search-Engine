"""
Deliverable dump - Positional-index output.  [Person B]
========================================================

Writes the positional index to a human-readable text file in the conceptual
representation required by Part C:

    term -> df -> [(docID, tf, [p1, p2, ...]), ...]

Run from the repo root:
    python scripts/dump_positional_index.py
-> writes data/positional_index.txt
"""

import sys
import os

# Make src/ importable when run from the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing import load_corpus              # noqa: E402
from positional_index import build_positional_index  # noqa: E402

OUT_PATH = os.path.join(_ROOT, "data", "positional_index.txt")


def dump(pindex, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total_postings = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("POSITIONAL INDEX  (term -> df -> [(docID, tf, [positions]), ...])\n")
        fh.write(f"Vocabulary size: {len(pindex)} terms\n")
        fh.write("=" * 70 + "\n\n")
        for term in sorted(pindex):
            entry = pindex[term]
            fh.write(f"{term}  ->  df={entry['df']}\n")
            for docid in sorted(entry["postings"]):
                positions = entry["postings"][docid]
                total_postings += 1
                fh.write(f"    ({docid}, tf={len(positions)}, {positions})\n")
            fh.write("\n")
    return total_postings


if __name__ == "__main__":
    docs = load_corpus(os.path.join(_ROOT, "corpus_100.txt"))
    pindex = build_positional_index(docs)
    postings = dump(pindex, OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(pindex)} terms, {postings} (term, doc) postings across {len(docs)} docs.")
