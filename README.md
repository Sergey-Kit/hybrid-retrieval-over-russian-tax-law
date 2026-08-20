# P3 — Hybrid retrieval over Russian tax law

*Русская версия: [README.ru.md](README.ru.md) — она основная.*

Retrieval over the Russian Tax Code with **temporal correctness**: not "what the Code says"
but "what the Code said on date X". No generation — the project boundary is returning the
right spans of law.

Spec: [`P3_retrieval_SPEC.md`](P3_retrieval_SPEC.md) (Russian). This file reports **actual
state**, including where reality diverged from the spec.

## Status: milestone 1 of 3

Machinery complete and tested; live collection blocked on a direct connection (see below).
**92 tests passing.**

```bash
pip install -r requirements.txt && python -m pytest -q
```

## Headline result

The spec calls one question "the highest-leverage unknown in the whole project": are dated
editions of the Tax Code retrievable from ИПС «Законодательство России», the Russian state
legal information system? The answer decided weeks vs months of work — the alternative being
a **consolidator** that reconstructs the Code as of any date from its 1998 original plus
hundreds of amending federal laws, parsing instructions like "in clause 3 replace the words
X with the words Y".

**Answer: they are.** ([`docs/spike-ips.md`](docs/spike-ips.md))

| | |
|---|---|
| Tax Code part 1 | `nd=102054722`, **153 editions**, `rdk` 0…248 |
| Coverage | **1999–2026** |
| Editions since 2021 | **79** — exactly the window the spec requires |
| Edition texts | available; **14 of 14** sampled editions from 2021–2026 |

The consolidator drops off the critical path and becomes a *verifier*.

The spec's premise that the site "does not respond to automated requests" was wrong:
`pravo.gov.ru` drops SYN on port 443 for foreign IPs but serves port 80. That is geofiltering,
not bot protection.

## Second result: the labels already existed

The spec planned to mine norm citations out of regulator letters. Recon found better: the
Federal Tax Service's *complaint decisions* service **labels each decision itself**, carrying
both a natural-language dispute description and the Tax Code articles the agency assigned.
So an entire error class the spec lists as a silent defect — "citations not mapped to
norm_unit_id, so matches are not counted and metrics are silently understated" — cannot
occur for this source.

That also yielded an independent check of the citation normalizer against 263 agency-written
labels: **251/251** Tax Code article labels parsed correctly, **12/12** non-Tax-Code acts
correctly excluded. Known gap: chapter-level labels are not handled.

## Three deliberate departures from the spec

1. **The silver set is split train/dev/test.** The spec tunes BM25F weights and the RRF
   parameter on it, then reports Recall@20 on the same set — measuring on training data.
   Here tuning is restricted to train/dev, reported numbers come once from held-out test,
   and absence of leakage is asserted by a test.
2. **Embedder base is `deepvk/USER-bge-m3`** (359M, 8192 context) rather than the spec's
   FRIDA-vs-BGE-M3 binary. An 8192 context means the spec's truncation trap cannot occur
   and chunking stays encoder-independent, so the three-way comparison runs on identical
   chunks. FRIDA is also `dim=1536`, not 1024 — 1.5× the vector memory and scan cost on a
   CPU without AVX2.
3. **BM25F in-process, not in Postgres.** `ts_rank` is not BM25 and cannot do the field
   weights the spec wants tuned. Index drift is prevented by a shared `corpus_build_id`
   and an assertion, not by "one database".

## Silent defects found in practice

All of these return HTTP 200 and would poison the corpus quietly. Each is closed by a test
or an assertion.

- `rdk` iterated as a range rather than a list: 96 of 249 values do not exist and return a
  "text unavailable" stub — 96 phantom editions.
- Option `value` parsed as a number: it is `rdk,nd`.
- `<option>` matched in raw HTML: `<script>` blocks contain markup that leaks into the
  edition list.
- Collecting through a VPN from a foreign IP: the source filters the address and serves an
  unknown subset, with throttling invisible in the logs.
- A citation with no code named defaults to the Tax Code: "article 5 of Federal Law…"
  becomes a Tax Code norm.

## Limitations

The silver set is 246 pairs against a >=300 target: the published listing of ФНС decisions
ends at page 21, not the 46 the paginator advertises, so a second source is needed. It is
also biased toward disputes, has no negatives and no `as_of`, and its gold is article-level.
`pravo.gov.ru` serves no HTTPS at all — an earlier reading of this as geofiltering of foreign
IPs was disproved from a Russian connection; the exit country is now recorded as provenance
rather than enforced. ИПС terms of use are unconfirmed — a blocker for the commercial
follow-on. Latency figures on this machine (GTX 1650 SUPER, 8 GB RAM, no AVX2) compare
configurations against each other, not against production.
