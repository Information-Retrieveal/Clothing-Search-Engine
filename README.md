# Clothing Search Engine

A small information retrieval search engine over a corpus of 100 clothing
product descriptions, built for IR Assignment 1 (CSD358). It supports three
retrieval styles:

- Ranked free-text search using the Vector Space Model with `lnc.ltc` cosine similarity.
- Positional search for exact phrases and ordered proximity (`WITHIN/k`).
- Smart search (our novelty), which fuses the two above into one proximity-aware ranker.

It is written in pure Python 3 with a single dependency (`nltk`, used only for the
Porter stemmer). There is no web framework and no database.

## Team

This was done as a group of two.

| Name | Roll number | Role |
|------|-------------|------|
| Viraja | 2410110xxx | Ranking engine: inverted index and VSM (Parts A and B) |
| Rishit Kamboj | 2410110598 | Positional engine: positional index, phrase and proximity search (Part C) |

The command line interface (Part D), the testing battery (Part E) and the
novelty were finished together after both branches were merged.

## Features

- Inverted index with term frequencies (tf) and document frequencies (df).
- `lnc.ltc` VSM ranking (log-tf document weights, log-tf times idf query weights,
  cosine normalisation), top 10 results, ties broken by document id.
- Positional index that stores the position of every term occurrence.
- Exact phrase search (terms must sit at consecutive positions, not just co-occur).
- Ordered proximity search (`term1 WITHIN/k term2`) that also returns the matching positions.
- Smart search (novelty): runs the query as a phrase first, then blends cosine
  relevance with a phrase bonus and a proximity (smallest window) bonus.
- One shared text pipeline (case fold, tokenize, stop-word removal, Porter stem)
  used identically for documents and queries.

## Quickstart

```bash
pip install nltk

python src/app.py
```

The menu offers four modes:

```
1) Free-text ranked search   (VSM, lnc.ltc cosine)
2) Exact phrase search       (positional index)
3) Proximity search WITHIN/k (positional index)
4) Smart search   [NOVELTY]  (VSM + phrase + proximity)

1  cotton shirt              top 10 ranked docs with cosine scores
2  cotton shirt              the 5 shirts where the words are adjacent
3  cotton WITHIN/3 shirt     docs plus the matching (pos, pos) pairs
4  cotton shirt              the real shirts lifted above the t-shirts
```

## Repository layout

```
corpus_100.txt              Product corpus (100 documents)
src/
  preprocessing.py          Shared pipeline: tokenize, stopword, stem, positions
  inverted_index.py         Inverted index  {term: {df, postings{doc: tf}}}
  vsm.py                    lnc.ltc cosine ranking, top 10
  positional_index.py       Positional index {term: {df, postings{doc: [pos]}}}
  phrase_search.py          Exact phrase and ordered proximity (WITHIN/k)
  smart_search.py           Novelty: proximity-boosted ranking (VSM + positional)
  app.py                    Command line interface (all four modes)
scripts/
  dump_inverted_index.py    Writes the inverted-index deliverable
  dump_positional_index.py  Writes the positional-index deliverable
  run_all_queries.py        Runs the full query battery into data/results.txt
  evaluate.py               Precision@k comparison, VSM vs smart search
tests/
  test_vsm.py               Ranking tests (includes the lecture 0.8 cosine check)
  test_positional.py        Phrase and proximity tests (structural, no fixed ids)
  test_smart_search.py      Novelty tests (structural)
data/                       Generated index dumps and result logs
report/REPORT.md            Full write-up: methodology, formulas, results
SCREENSHOTS.pdf             Application screenshots and query evidence
```

## Regenerate the deliverables

```bash
python scripts/dump_inverted_index.py    # data/inverted_index.txt
python scripts/dump_positional_index.py  # data/positional_index.txt
python scripts/run_all_queries.py        # data/results.txt
python scripts/evaluate.py               # data/evaluation.txt
```

## Run the tests

```bash
python tests/test_vsm.py           # VSM ranking plus the lecture example (0.80)
python tests/test_positional.py    # phrase and proximity correctness
python tests/test_smart_search.py  # smart search correctness
```

## How it works

Every document and query goes through the same pipeline in `preprocessing.py`,
so query terms always line up with the indexed terms. Free-text queries are
scored by cosine similarity between `lnc`-weighted document vectors and
`ltc`-weighted query vectors (with N equal to 100). Phrase and proximity queries
use the positional index instead: the phrase search intersects postings at
consecutive positions, and the proximity search keeps position pairs inside the
requested window. Smart search combines both by promoting exact phrase matches
and adding a proximity bonus on top of the cosine score. The main idea the
project shows, that VSM ranks on term presence while positional retrieval ranks
on term arrangement, is worked through in `report/REPORT.md`.
