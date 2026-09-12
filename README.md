# Clothing Search Engine

A small information-retrieval search engine over a corpus of **100 clothing
product descriptions**, built for IR Assignment 1 (CSD358). It supports two
retrieval styles:

- **Ranked free-text search** — Vector Space Model with `lnc.ltc` cosine similarity.
- **Positional search** — exact-phrase and ordered proximity (`WITHIN/k`) matching.

Pure Python 3, one dependency (`nltk`, for the Porter stemmer). No web framework,
no database.

---

## Features

- Inverted index with term frequencies (tf) and document frequencies (df).
- `lnc.ltc` VSM ranking (log-tf document weights, log-tf × idf query weights,
  cosine normalization), top-10 results, ties broken by docID.
- Positional index storing every term occurrence's position.
- Exact phrase search (terms at consecutive positions, not mere co-occurrence).
- Ordered proximity search (`term1 WITHIN/k term2`) returning the matching positions.
- Interactive CLI wiring both engines behind one menu.
- One shared text pipeline (case-fold → tokenize → stop-word removal → Porter stem)
  used identically for documents and queries.

---

## Quickstart

```bash
pip install nltk

python src/app.py                       # interactive search (free-text + phrase/proximity)
```

Example session:

```
1) Free-text ranked search   (VSM, lnc.ltc cosine)
2) Exact phrase search       (positional index)
3) Proximity search WITHIN/k (positional index)

> 1  cotton shirt              -> top-10 ranked docs with cosine scores
> 2  cotton shirt              -> the 5 shirts where the words are adjacent
> 3  cotton WITHIN/3 shirt     -> docs + matching (pos, pos) pairs
```

---

## Repository layout

```
corpus_100.txt              Product corpus (100 documents)
src/
  preprocessing.py          Shared pipeline: tokenize / stopword / stem / positions
  inverted_index.py         Inverted index  {term: {df, postings{doc: tf}}}
  vsm.py                    lnc.ltc cosine ranking, top-10
  positional_index.py       Positional index {term: {df, postings{doc: [pos]}}}
  phrase_search.py          Exact phrase + ordered proximity (WITHIN/k)
  app.py                    Command-line interface (both modes)
scripts/
  dump_inverted_index.py    Writes the inverted-index deliverable
  dump_positional_index.py  Writes the positional-index deliverable
  run_all_queries.py        Runs the full test-query battery -> data/results.txt
tests/
  test_vsm.py               Ranking tests (incl. the lecture 0.8 cosine check)
  test_positional.py        Phrase/proximity tests (structural, no hard-coded IDs)
data/                       Generated index dumps and query-result logs
report/REPORT.md            Detailed writeup: methodology, formulas, results analysis
```

---

## Regenerate the deliverables

```bash
python scripts/dump_inverted_index.py    # -> data/inverted_index.txt
python scripts/dump_positional_index.py  # -> data/positional_index.txt
python scripts/run_all_queries.py        # -> data/results.txt
```

## Run the tests

```bash
python tests/test_vsm.py          # VSM ranking + lecture worked example (0.80)
python tests/test_positional.py   # phrase / proximity correctness
```

---

## How it works (in one paragraph)

Every document and query passes through the same pipeline in `preprocessing.py`,
so query terms always line up with indexed terms. Free-text queries are scored by
cosine similarity between `lnc`-weighted document vectors and `ltc`-weighted query
vectors (`N = 100`). Phrase and proximity queries instead use the positional
index: the phrase search intersects postings at consecutive positions, and the
proximity search keeps position pairs within the requested window. The core
lesson the project demonstrates — VSM ranks on term *presence*, positional
retrieval ranks on term *arrangement* — is worked through in `report/REPORT.md`.

---

## Team

Group of two:

- **Viraja** — ranking engine: inverted index + VSM (Parts A, B)
- **Rishit** — positional engine: positional index + phrase/proximity (Part C)

Part D (CLI) and Part E (testing/report) were done jointly after merge.