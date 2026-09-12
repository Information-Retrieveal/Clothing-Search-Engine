# Module Contract (shared foundation)

This file freezes the interfaces and data shapes both branches build against,
so Person A and Person B can work in parallel and merge without conflicts.

## Shared pipeline — `src/preprocessing.py` (on `main`)

```python
analyze(text)          -> list[(term, position)]   # content terms, original offsets
preprocess_query(q)    -> list[term]               # same pipeline, ordered terms
load_corpus(path="corpus_100.txt")
    -> dict: docid -> {"category", "title", "text", "terms"}
       where terms == analyze(title + " " + text)
```
Pipeline = lowercase → split on non-alphanumeric → drop standard English stop
words (stop words still consume a position) → Porter stem.
`N = 100` documents.

## Person A — `feat/vsm` branch

`src/inverted_index.py`
```python
build_inverted_index(docs) -> dict:
    term -> {"df": int, "postings": {docid: tf}}     # tf = raw term count
```
`src/vsm.py` — lnc.ltc, cosine, top-10
```python
rank(query, index, docs, top_k=10) -> list[(docid, score)]  # score desc, tie: docid asc
```
- Document weight (lnc): `1 + log10(tf)`, then L2-normalize per doc. No idf on docs.
- Query weight (ltc): `(1 + log10(tf)) * log10(N/df)`, then L2-normalize.
- Cosine = dot product of the two normalized vectors.
- If every query term has idf 0 (query norm 0) → return `[]` (documented limitation).

## Person B — `rishit` branch

`src/positional_index.py`
```python
build_positional_index(docs) -> dict:
    term -> {"df": int, "postings": {docid: [pos, ...]}}   # tf == len(positions)
```
`src/phrase_search.py`
```python
phrase_search(phrase, pindex)              -> dict: docid -> [start_pos, ...]
proximity_search(term1, k, term2, pindex)  -> dict: docid -> [(pos1, pos2), ...]
```
- Phrase: terms at CONSECUTIVE positions `p, p+1, …` (positional intersection).
- Proximity `A WITHIN/k B` (ordered): positions with `0 < pos(B) - pos(A) <= k`.
- Both return the actual matching positions.

## Integration (on `main`, after both merge)

`src/app.py` (Part D CLI, wires VSM + positional), `scripts/run_all_queries.py`
(Part E battery), `report/REPORT.md`.

`src/smart_search.py` (NOVELTY — proximity-boosted ranking, fuses VSM + positional)
```python
smart_search(query, index, pindex, docs, top_k=10)
    -> list[{docid, score, cosine, phrase, window}]   # net = cosine + 0.5*phrase + 0.3/window
smallest_window(distinct_terms, positions_by_term) -> int | None
```
