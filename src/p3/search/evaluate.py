"""Метрики поиска: recall@k, nDCG@k, MRR — с разрезами из ТЗ §7.

Одна вещь здесь важнее самих формул. Индекс сейчас содержит **только часть
первую** НК, а silver set построен из решений и писем, где 87% упоминаний —
статьи части второй. Значит для большинства его запросов правильный ответ
физически отсутствует в корпусе, и общая цифра recall описывала бы полноту
индекса, а не качество поиска.

Поэтому метрики считаются на **достижимом подмножестве** — запросах, чей эталон
есть в индексе, — и доля достижимых выводится рядом с цифрами. Молча
отфильтровать было бы тем же приукрашиванием, только незаметным.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass
class EvalQuery:
    query_id: str
    query: str
    gold_articles: frozenset[str]
    gold_norms: frozenset[str]
    query_type: str
    source: str


def load_queries(rows: Iterable[dict]) -> list[EvalQuery]:
    return [
        EvalQuery(
            query_id=row["query_id"],
            query=row["query"],
            gold_articles=frozenset(row.get("gold_article_ids") or ()),
            gold_norms=frozenset(row.get("gold_norm_ids") or ()),
            query_type=row.get("query_type", "unknown"),
            source=row.get("source", "unknown"),
        )
        for row in rows
    ]


def reachable(queries: list[EvalQuery], indexed_articles: set[str]) -> list[EvalQuery]:
    """Запросы, эталон которых вообще присутствует в индексе."""
    return [q for q in queries if q.gold_articles & indexed_articles]


def recall_at_k(retrieved: list[str], gold: frozenset[str], k: int) -> float:
    if not gold:
        return 0.0
    found = gold & set(retrieved[:k])
    return len(found) / len(gold)


def hit_at_k(retrieved: list[str], gold: frozenset[str], k: int) -> float:
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], gold: frozenset[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: frozenset[str], k: int) -> float:
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else 0.0


@dataclass
class Metrics:
    queries: int
    recall_at_20: float
    hit_at_20: float
    ndcg_at_10: float
    mrr: float

    def as_dict(self) -> dict:
        return {
            "queries": self.queries,
            "recall@20": round(self.recall_at_20, 4),
            "hit@20": round(self.hit_at_20, 4),
            "nDCG@10": round(self.ndcg_at_10, 4),
            "MRR": round(self.mrr, 4),
        }


def evaluate(
    queries: list[EvalQuery],
    run: Callable[[EvalQuery], list[str]],
    *,
    level: str = "article",
) -> Metrics:
    """Прогнать поиск по запросам и посчитать метрики.

    `level` — на каком уровне сравнивать: `article` (эталон ФНС — статьи) или
    `norm` (синтетический набор знает пункт).
    """
    if not queries:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)

    recalls, hits, ndcgs, rrs = [], [], [], []
    for query in queries:
        gold = query.gold_articles if level == "article" else query.gold_norms
        retrieved = run(query)
        recalls.append(recall_at_k(retrieved, gold, 20))
        hits.append(hit_at_k(retrieved, gold, 20))
        ndcgs.append(ndcg_at_k(retrieved, gold, 10))
        rrs.append(reciprocal_rank(retrieved, gold))

    n = len(queries)
    return Metrics(n, sum(recalls) / n, sum(hits) / n, sum(ndcgs) / n, sum(rrs) / n)


def by_slice(
    queries: list[EvalQuery],
    run: Callable[[EvalQuery], list[str]],
    key: Callable[[EvalQuery], str],
    *,
    level: str = "article",
) -> dict[str, dict]:
    """Метрики в разрезе — по типу запроса или по источнику (ТЗ §7)."""
    groups: dict[str, list[EvalQuery]] = {}
    for query in queries:
        groups.setdefault(key(query), []).append(query)
    return {
        name: evaluate(group, run, level=level).as_dict()
        for name, group in sorted(groups.items())
    }
