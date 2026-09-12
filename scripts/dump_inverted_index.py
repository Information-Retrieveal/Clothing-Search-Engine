"""
Deliverable dump - inverted index  (Person A)
=============================================

Writes the full inverted index to a human-readable text file (one of the
required submission artifacts) and prints a short summary to the console.

For every term, in alphabetical order:

    term    df=<n>    docid:tf  docid:tf  ...

Run from the repository root:

    python scripts/dump_inverted_index.py [output_path]

Default output: ``outputs/inverted_index.txt``.
"""

import os
import sys

# Make ``src/`` importable no matter where the script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing import load_corpus          # noqa: E402
from inverted_index import build_inverted_index  # noqa: E402


def dump_inverted_index(index, out_path):
    """Write ``index`` to ``out_path`` in ``term  df=n  docid:tf ...`` form."""
    total_postings = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Inverted index - {len(index)} distinct terms\n")
        fh.write("# format:  term  df=<docfreq>  docid:tf  docid:tf ...\n\n")
        for term in sorted(index):
            entry = index[term]
            postings = entry["postings"]
            total_postings += len(postings)
            posting_str = "  ".join(
                f"{docid}:{tf}" for docid, tf in sorted(postings.items())
            )
            fh.write(f"{term}\tdf={entry['df']}\t{posting_str}\n")
    return total_postings


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _ROOT, "outputs", "inverted_index.txt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    docs = load_corpus(os.path.join(_ROOT, "corpus_100.txt"))
    index = build_inverted_index(docs)
    total_postings = dump_inverted_index(index, out_path)

    print(f"Documents indexed : {len(docs)}")
    print(f"Distinct terms    : {len(index)}")
    print(f"Total postings    : {total_postings}")
    print(f"Written to        : {out_path}")


if __name__ == "__main__":
    main()
