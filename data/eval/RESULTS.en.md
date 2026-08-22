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

