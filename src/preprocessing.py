"""
Part A - Corpus reading and the shared text-processing pipeline.
==================================================================

SHARED FOUNDATION (lives on `main`, used by BOTH halves of the project):

    * Person A  (inverted_index.py, vsm.py)          uses analyze() / preprocess_query()
    * Person B  (positional_index.py, phrase_search) uses analyze() / preprocess_query()

Every part of the system MUST turn text into terms through this one pipeline,
otherwise query terms would not line up with indexed terms.

Pipeline (follows IIR Ch. 2 - "The term vocabulary and postings lists"):
    1. Tokenize + case-fold : lowercase, split on any non-alphanumeric char.
    2. Remove punctuation   : done implicitly by the split in step 1.
    3. Stop-word removal    : standard English stop list (see STOPWORDS).
    4. Stemming             : Porter stemmer (the algorithm taught in class).

Position policy (documented in the report):
    Positions are the ORIGINAL offsets of a word in the tokenized stream.
    A stop word still consumes a position but is not indexed as a term. So
    phrase / proximity distances reflect the true distance between words in the
    original text (exactly like the "to be or not to be" positional-index
    example in IIR Ch. 2), and a phrase like "cotton shirt" only matches where
    those two words are genuinely adjacent.
"""

import re
from nltk.stem import PorterStemmer

# One shared Porter stemmer instance (Lecture 2 / IIR Ch. 2).
_STEMMER = PorterStemmer()

# Standard English stop-word list (the NLTK English list).
#
# STOP-WORD POLICY (justified in the report):
#   * A standard, well-known English list - not a hand-made one.
#   * Applied CONSISTENTLY to both documents and queries.
#   * We deliberately do NOT add domain words such as "wear", "fabric",
#     "festive" or "winter": they are needed for the Part C phrase / proximity
#     queries, and idf in lnc.ltc already drives collection-wide words to
#     weight 0 for ranking on its own.
STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren aren't as at be
because been before being below between both but by can couldn couldn't d did
didn didn't do does doesn doesn't doing don don't down during each few for from
further had hadn hadn't has hasn hasn't have haven haven't having he he'd he'll
he's her here hers herself him himself his how i i'd i'll i'm i've if in into is
isn isn't it it'd it'll it's its itself just ll m ma me mightn mightn't more most
mustn mustn't my myself needn needn't no nor not now o of off on once only or
other our ours ourselves out over own re s same shan shan't she she'd she'll
she's should should've shouldn shouldn't so some such t than that that'll the
their theirs them themselves then there these they they'd they'll they're they've
this those through to too under until up ve very was wasn wasn't we we'd we'll
we're we've were weren weren't what when where which while who whom why will with
won won't wouldn wouldn't y you you'd you'll you're you've your yours yourself
yourselves
""".split())


def _raw_tokens(text):
    """Step 1 & 2: lower-case and split on any run of non-alphanumeric chars.

    "Men's Checked Cotton Shirt - Black." -> ['men','s','checked','cotton',
    'shirt','black'].  "t-shirt" splits to 't','shirt' (the stray 't' is later
    dropped as a stop word).
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def analyze(text):
    """Full pipeline. Returns a list of (term, position) pairs for content terms.

    `position` is the index of the word in the ORIGINAL token stream (stop words
    are counted but not emitted). This is the single source of truth that both
    the inverted index (term frequencies) and the positional index (positions)
    are built from.
    """
    out = []
    for pos, word in enumerate(_raw_tokens(text)):
        if word in STOPWORDS:          # stop word: consumes a position, not indexed
            continue
        out.append((_STEMMER.stem(word), pos))
    return out


def preprocess_query(query):
    """Apply the identical pipeline to a query -> ordered list of terms.

    Used by the VSM (as a bag of words) and by phrase/proximity search (as an
    ordered sequence of terms).
    """
    return [term for term, _pos in analyze(query)]


# ----------------------------------------------------------------------------
# Corpus reader
# ----------------------------------------------------------------------------
_DOC_RE = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
_FIELD_RE = {
    "docid": re.compile(r"<DOCID>(.*?)</DOCID>", re.DOTALL),
    "category": re.compile(r"<CATEGORY>(.*?)</CATEGORY>", re.DOTALL),
    "title": re.compile(r"<TITLE>(.*?)</TITLE>", re.DOTALL),
    "text": re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL),
}


def load_corpus(path="corpus_100.txt"):
    """Read the pseudo-XML corpus file.

    Returns an insertion-ordered dict: docid -> {category, title, text, terms}
    where `terms` is the analyzed (term, position) list for TITLE + TEXT (the
    two fields we make searchable).
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    docs = {}
    for block in _DOC_RE.findall(raw):
        rec = {}
        for field, rx in _FIELD_RE.items():
            m = rx.search(block)
            rec[field] = m.group(1).strip() if m else ""
        rec["terms"] = analyze(rec["title"] + " " + rec["text"])
        docs[rec["docid"]] = {
            "category": rec["category"],
            "title": rec["title"],
            "text": rec["text"],
            "terms": rec["terms"],
        }
    return docs


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents.")
    d1 = docs["D001"]
    print("D001 category :", d1["category"])
    print("D001 title    :", d1["title"])
    print("D001 first 12 (term, pos):", d1["terms"][:12])
    print("Query 'Cotton Shirt!' ->", preprocess_query("Cotton Shirt!"))
