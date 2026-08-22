# P3 — hybrid retrieval over Russian tax law

Retrieval over the Tax Code of the Russian Federation with **temporal correctness**: not
"what the Code says" but *"what the Code said on date X"*. No generation — the project
boundary is returning the right spans of law.

Specification: [`P3_retrieval_SPEC.md`](P3_retrieval_SPEC.md) (Russian). It cites internal
product-planning documents that are not published here. Full documentation is
in Russian; this file is the English summary. Both report **actual state**, including where
reality diverged from the specification and where a measurement proved me wrong.

```bash
pip install -r requirements.txt && python -m pytest -q     # 301 tests
```

---

## Status

| Milestone | Contents | State |
|---|---|---|
| 1 | ИПС spike, silver set, synthetic layer-1 set | **done** |
| 2 | Parser for both parts of the Code, chunking, reference graph, BM25F, dense, RRF, one-hop, reranker, metrics | **done** |
| 2b | Temporal model and the `as_of` filter | **done** |
| 2c | Negatives and score-separability measurement | **done** |
| 3 | Real entry-into-force dates | not started |

Corpus: **8198 chunks over 815 articles** of both parts, in PostgreSQL and in in-memory
indexes. Deliberately deferred: a three-way encoder comparison, chapter-level references.

---

## Results

Figures come from a **single run over the held-out test split**; weights were tuned on
train only. Tables are generated from the measurement files by `p3 render-results`
rather than typed, so they cannot drift from the data on the next run.

### Silver set, test split

**331 of 337 queries are reachable** (98%). Metrics are computed over the
reachable subset — those whose gold is present in the index — and that share is printed
next to the numbers: filtering silently would be the same flattery, just harder to see.

| Configuration | recall@20 | hit@20 | recall@8 | nDCG@10 | MRR | ms |
|---|---|---|---|---|---|---|
| Dense only | 0.665 | 0.737 | 0.589 | 0.490 | 0.503 | 96 |
| BM25 only | 0.682 | 0.764 | 0.556 | 0.451 | 0.458 | 16 |
| **Hybrid (RRF)** | 0.689 | 0.767 | 0.627 | 0.527 | 0.533 | 24 |
| Hybrid + reranker | 0.596 | 0.680 | 0.596 | 0.536 | 0.563 | 17591 |

The recall@20 ceiling on this corpus is 0.997: the gold is almost entirely
present in the index, so the metric is not bounded by its incompleteness.

### Synthetic layer-1 set (gold at clause level)

| Configuration | recall@20 | hit@20 | recall@8 | nDCG@10 | MRR | ms |
|---|---|---|---|---|---|---|
| Dense only | 0.819 | 0.889 | 0.736 | 0.555 | 0.527 | 64 |
| BM25 only | 0.639 | 0.694 | 0.611 | 0.560 | 0.586 | 10 |
| **Hybrid (RRF)** | 0.819 | 0.889 | 0.778 | 0.666 | 0.666 | 16 |
| Hybrid + reranker | 0.847 | 0.917 | 0.847 | 0.804 | 0.838 | 15304 |

The reranker returns the top 8, so its `recall@20` is capped by eight slots and is not
comparable with configurations returning 50. Compare it on `recall@8`, `nDCG@10`
and `MRR`.

### The headline measurement: what natural language costs

One norm, one gold, two phrasings:

| Phrasing | recall@20 | nDCG@10 |
|---|---|---|
| Excerpt from a demand (citation present) | 0.861 | 0.860 |
| Accountant's question (no citation) | 0.778 | 0.472 |

### Temporal test (`hybrid`)

70 synthetic "edition as of <date>" queries; the gold is a specific version of a norm.

| | |
|---|---|
| Correct edition returned | **64%** |
| Wrong edition of the right norm | **0** |
| Norm not found at all | 25 |
| Filter cost against a single-edition index | **0.0 points** |

The zero on the second row is the figure that matters: whenever the norm is found the
edition is right, so every miss is a retrieval failure rather than a temporal one.


---

## Three things that are usually missing

**Temporal correctness.** Five editions of part one spanning 2019–2026 collapse into 1692
norms and 2894 versions; 821 norms changed. The filter `valid_from <= as_of < valid_to` runs
**before** search in both channels, and costs **0.00 recall points** against a single-edition
baseline while adding the ability to answer as of any date. Whenever the norm is found the
edition is right — zero wrong editions over 70 queries.

**Evaluation without manual annotation.** 1678 query→norm pairs collected automatically from
the tax authority's own complaint decisions and mandatory letters, where the agency labels the
articles itself. Plus a synthetic layer-1 set whose gold is known by construction and verified
against the corpus.

**Negative results are recorded.** Sources tested and rejected, measurement bugs found after
the numbers arrived, and a null result for one-hop expansion that two independent sets agree
on. See `docs/` — all in Russian.

---

## What the measurements say

**Hybrid beats either channel** on 331 queries: recall@20 0.689 against 0.665 for dense and
0.682 for BM25. The reranker trades recall for ordering — on the layer-1 set it wins
everything (recall@8 0.847, MRR 0.838); on the silver set it gives up 3.2 points of recall@8
while gaining MRR.

**Natural language costs 45% of ranking quality.** Same norm, same gold, two phrasings: an
excerpt from a tax demand scores nDCG@10 0.860, an accountant's question 0.472.

**Retrieval scores cannot express "I don't know".** Over 331 positives and 45 negatives —
real questions whose answer is outside the Code, or about a regime abolished and removed from
the current edition:

| Signal | AUC |
|---|---|
| BM25 top-1 score | 0.503 — chance |
| Dense top-1 cosine | 0.704 |
| Reranker top-1 score | 0.775 |

The reranker cleanly rejects questions from another branch of law (median −2.2) but not
questions about norms that are simply absent (median 0.608 against 2.682 for positives): those
read like ordinary tax questions and retrieve neighbouring regimes. A confidence threshold
will not implement refusal in the product; that needs a coverage check, not a relevance score.

**One-hop expansion contributes nothing** on either evaluation set — 0 of 36 and 0 of 331 —
although the mechanism works and adds a median of 11 candidates. The synthetic set could not
have shown otherwise by construction; the silver set could, and did not.

---

## Silent defects found on real data

None of these raise, none appear in logs, and all produce plausible numbers. Each is closed by
a test.

| Defect | What it would have cost |
|---|---|
| Superscript numbers: "Статья 6¹" read as article 61 | **half the articles lost**, including the procedural core — 6.1, 93.1, 101.4, 129.1 |
| `rdk` is not chronological (212 dated 2020, 197 dated 2022) | version intervals built backwards; the filter works and picks the wrong edition |
| ИПС changed its markup around 2019 | the parser returned 0 articles from a 1.5 MB document and said nothing |
| Without `print=1` ИПС returns the opening of a document | 25 articles instead of 142, truncated mid-sentence, while `page` is accepted and ignored |
| A repeated clause number is a table row, not a clause | article 333.45 produced dozens of "clause 2" sharing one chunk id; 1494 chunks collapsed on load |
| Vectors of another corpus silently substituted | 1906 part-one vectors against a 9667-chunk store |
| Dots inside reference tokens | `nk1_st93.1` split into `nk1_st93` and `1` |
| Length normalised per document instead of per field | an exact citation with IDF 6.6 lost to common words in a short document |
| "НК РФ" tokenised as separate words | "рф" occurs in 2 documents of 1906, earns IDF 6.64 and outranks the citation itself |

And three errors in the measurements themselves, found after the numbers arrived: one-hop
reporting 0.0% because expanded chunks fell outside the measured window; the temporal filter
appearing to fail its criterion because it was compared against an index nobody would ship;
and nDCG exceeding 1.0 because one gold article was credited once per matching clause.

---

## Layout

```
src/p3/
  config.py netcheck.py raw_store.py fetch.py   collection: throttling, cache, provenance
  refs/normalize.py      "подпунктом 2 пункта 1 статьи 31" -> nk1/st31/p1/sp2
  sources/               ИПС, complaint decisions, mandatory letters
  nk/parse.py            section -> chapter -> article -> clause
  nk/refs.py             typed reference graph
  nk/chunk.py            chunking, text_dense / text_bm25 / title_bm25
  nk/temporal.py         norm versions and validity intervals
  silver/                silver set, layer-1 set, as_of queries, negatives
  search/                BM25F, dense, RRF, one-hop, reranker, metrics
  report.py              results section generated from measurement files
```

Everything downloaded is stored byte-for-byte before parsing, hashed, and recorded in a
manifest; re-runs read from cache. Numbers in the documentation are **generated** by
`p3 render-results`, never typed.

---

## Licence

Code and documentation are MIT. The texts of the Tax Code, complaint decisions and
tax-authority letters under `data/` are not subject to copyright under article 1259(6) of
the Russian Civil Code, being official documents of state bodies.

---

## What is and is not in this repository

What cannot be reconstructed from it is here: the silver-set labels, the raw-data manifest
with hashes, and the measurement files. **Chunks and vectors are not** — they are a function
of the raw data and the code, weigh 96 MB, and, more importantly, drift out of step with the
corpus without saying so. Rebuild them with the commands below; a mismatch now fails on a
content hash rather than producing wrong numbers.

---

## Limitations

* The silver set aims at a different population than the product serves: layer-1 procedural
  articles account for 52 of 1678 pairs.
* The synthetic set is generated by the same hand that builds the index, so it measures
  structural reachability of a norm rather than the language gap.
* Temporal interval boundaries are approximate — ИПС gives the amending act's date, not the
  date of entry into force, and editions are sampled rather than exhaustive.
* The reranker costs 15–18 seconds per query on a GTX 1650 SUPER, which rules the best
  configuration out for interactive use on this hardware.
* Latency figures compare configurations against each other on this machine; they are not
  product figures.
* Chapter-level references are not parsed, which drops 56 real pairs as having no gold.
