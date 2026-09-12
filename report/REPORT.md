# Clothing Search Engine — IR Assignment 1 (CSD358)

A small clothing search engine over a 100-document product corpus, built exactly
along the lines taught in the *Introduction to Information Retrieval* (IIR)
lectures: an inverted index with tf/df, ranked retrieval with the **lnc.ltc**
Vector Space Model, and a **positional index** supporting exact-phrase and
ordered-proximity search.

**Team (group of two):**
- **Person A — ranking engine:** inverted index + VSM (Parts A, B) — *Vraja*
- **Person B — positional engine:** positional index + phrase/proximity (Part C) — *Rishit*
- Part D (CLI) and Part E (testing/report) were done jointly after merge.

---

## 1. How to run

```bash
pip install nltk                       # only dependency (Porter stemmer)

python src/app.py                      # Part D interactive search (both modes)
python scripts/dump_inverted_index.py  # Part A deliverable: inverted-index dump
python scripts/dump_positional_index.py# Part C deliverable: positional-index dump
python scripts/run_all_queries.py      # Part E battery -> data/results.txt

python tests/test_vsm.py               # Person A tests (incl. lecture 0.8 check)
python tests/test_positional.py        # Person B tests (298 assertions)
```

## 2. Repository layout

| File | Part | Owner |
|---|---|---|
| `src/preprocessing.py` | A — shared pipeline (tokenize/stopword/stem/positions) | shared |
| `src/inverted_index.py` | A — inverted index `{term:{df,postings{doc:tf}}}` | A |
| `src/vsm.py` | B — lnc.ltc cosine ranking | A |
| `src/positional_index.py` | C — `{term:{df,postings{doc:[pos]}}}` | B |
| `src/phrase_search.py` | C — exact phrase + ordered `WITHIN/k` | B |
| `src/app.py` | D — CLI (free-text + phrase/proximity) | integration |
| `scripts/dump_*_index.py`, `scripts/run_all_queries.py` | A/C/E deliverables | — |
| `tests/test_vsm.py`, `tests/test_positional.py` | E | A / B |

---

## 3. Part A — Pre-processing

Each document's DOCID, CATEGORY, TITLE and TEXT are parsed; TITLE + TEXT are
indexed. Every piece of text (documents **and** queries) passes through one
shared pipeline (`preprocessing.analyze`):

1. **Case-folding** — lowercase everything (IIR Ch. 2 "case folding").
2. **Tokenize / remove punctuation** — split on any non-alphanumeric run, so
   `t-shirt → t, shirt` and `100% → 100`.
3. **Stop-word removal** — standard English list.
4. **Stemming** — the **Porter stemmer** (`festive → festiv`, `stitching → stitch`).

**Stop-word policy (justification).** We use a standard, well-known English
stop-word list (the NLTK English list) rather than a hand-made one, and apply it
**identically to documents and queries**. We deliberately do **not** add
domain-specific words such as *wear, fabric, festive, winter* to the list, for
two reasons: (a) they are needed as query terms for the Part C phrase/proximity
queries (the lecture makes the same point — *"you need stop words for phrase
queries, e.g. King of Denmark"*), and (b) the **idf** factor in lnc.ltc already
drives collection-wide words to weight 0 for ranking, so removing them by hand is
unnecessary. In this corpus **41 terms occur in all 100 documents** (idf = 0),
e.g. *festiv, wear, regular, colour, garment* — these are neutralised by idf
automatically.

**Position policy.** A token's position is its **original offset** in the token
stream; a stop word consumes a position but is not indexed. This mirrors the
IIR positional-index model ("to₁ be₂ or₃ not₄ to₅ be₆") so phrase/proximity
distances are true text distances. Example — D001 *"Men's Cotton Crew Neck
T-Shirt"* indexes `cotton@2 … shirt@6` (not adjacent), so the phrase
`cotton shirt` correctly does **not** match this T-shirt.

## 4. Part B — Vector Space Model (lnc.ltc, N = 100)

- **Document weight (lnc):** `w = 1 + log10(tf)`, then L2-normalised per document; no idf.
- **Query weight (ltc):** `w = (1 + log10(tf)) · log10(N/df)`, then L2-normalised.
- **Score** = cosine = dot product of the two normalised vectors.
- Return up to **10** docIDs, sorted by **decreasing score**, ties broken by
  **increasing docID**.
- If every query term has idf 0 (query vector all zeros), ranking is undefined and
  we return no results — see Case 1 below.

**Correctness check.** `tests/test_vsm.py` reproduces the lecture's lnc.ltc
worked example (query *"best car insurance"* vs doc *"car insurance auto
insurance"*) and asserts the score is **0.80**, matching the slide. This proves
the weighting is implemented exactly as taught.

## 5. Part C — Positional index, phrase & proximity

The positional index stores, per term, `df` and a postings map
`docID → [positions]` (tf = number of positions), i.e. the
`term → df → [(docID, tf, [p1,p2,…])]` shape from the brief.

- **Exact phrase** (`phrase_search`): positional intersection — the phrase's
  terms must appear at **consecutive** positions `p, p+1, …`, not merely
  co-occur in the same document.
- **Ordered proximity** (`proximity_search`, `A WITHIN/k B`): keep position
  pairs with `0 < pos(B) − pos(A) ≤ k`.
- Both **return the actual matching positions** (evidence the positional index
  is used), which the CLI prints.

## 6. Part D — Application

`src/app.py` offers a menu: (1) free-text ranked search (VSM), (2) exact phrase,
(3) proximity `WITHIN/k`. Results show docID, category/title and cosine score
(mode 1) or the matching positions (modes 2 & 3). See the screenshots in §8.

## 7. Part E — Testing and the positional-vs-VSM comparison

Full logs are in [`../data/results.txt`](../data/results.txt). Coverage: 10
free-text queries, 5 exact phrase queries, 3 proximity queries with different
`k`, and out-of-vocabulary queries (`cashmere sweater` → 0; phrase
`leather jacket` → 0; `silk saree` → sarees, since `silk` is ignored). Automated
tests: `test_vsm.py` (4 checks incl. the 0.8 example) and `test_positional.py`
(298 structural assertions, no hard-coded doc IDs).

### Two cases where positional information changes the result

**Case 1 — `festive wear`.** Free-text VSM returns **0 results**: both *festive*
and *wear* occur in all 100 documents, so `idf = log10(100/100) = 0`, the query
vector is all zeros and nothing can be ranked. The positional index instead pins
the exact adjacent phrase to the **10 sarees** (D005, D015, …, D095) that
actually advertise *"festive wear"*. Positional retrieval succeeds where
bag-of-words VSM structurally cannot.

**Case 2 — `cotton shirt`.** VSM's top-10 mixes **5 T-shirts** (D001, D041, …)
with the checked shirts, because it only cares that *cotton* and *shirt* occur —
not that they are adjacent (in a T-shirt the tokens are `cotton … crew neck …
shirt`). The **exact phrase** drops every T-shirt and returns only the **5**
"Checked Cotton Shirt" products (D002/22/42/62/82) where the words are truly
consecutive. Relaxing to **`cotton WITHIN/3 shirt`** lets the T-shirts back in
(→ 10 docs), and `WITHIN/4` → 15 docs: the proximity window `k` is a direct
precision/recall knob. This is the core lesson — VSM ranks on term presence,
positional retrieval ranks on term *arrangement*.

## 8. Screenshots

Run `python src/app.py` and capture:

- `screenshots/mode1_free_text.png` — mode 1 on `cotton shirt` (top-10 with scores)
- `screenshots/mode1_festive_wear.png` — mode 1 on `festive wear` (no rankable results)
- `screenshots/mode2_phrase.png` — mode 2 on `cotton shirt` (5 docs, start positions shown)
- `screenshots/mode3_proximity.png` — mode 3 on `cotton WITHIN/3 shirt` (position pairs shown)

*(Place the images in `report/screenshots/` and they render above.)*

## 9. Deliverables checklist

- [x] Source code with comments — `src/`, `scripts/`, `tests/`
- [x] Inverted-index output — `data/inverted_index.txt` (via `scripts/dump_inverted_index.py`)
- [x] Positional-index output — `data/positional_index.txt`
- [x] Query results / comparison — `data/results.txt`
- [ ] Screenshots — capture per §8
- [ ] ZIP of all files
