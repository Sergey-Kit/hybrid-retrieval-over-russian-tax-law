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
import time
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


def matched(retrieved: list[str], gold: frozenset[str], k: int) -> set[str]:
    """Какие элементы эталона покрыты выдачей.

    Сопоставление **уровне-зависимое**: эталон уровня статьи засчитывается, если
    найден любой её пункт. Иначе смешанный эталон вида
    `['nk1/st93/p3', 'nk1/st6.1']` не может совпасть наполовину никогда —
    чанка `nk1/st6.1` не существует, есть только его пункты, — и recall для
    таких запросов ограничен 0.5 независимо от качества поиска.
    """
    window = retrieved[:k]
    found = set(gold & set(window))
    for item in gold - found:
        prefix = item + "/"
        if any(candidate.startswith(prefix) for candidate in window):
            found.add(item)
    return found


def recall_at_k(retrieved: list[str], gold: frozenset[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(matched(retrieved, gold, k)) / len(gold)


def hit_at_k(retrieved: list[str], gold: frozenset[str], k: int) -> float:
    return 1.0 if matched(retrieved, gold, k) else 0.0


def reciprocal_rank(retrieved: list[str], gold: frozenset[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in gold or any(item.startswith(g + "/") for g in gold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: frozenset[str], k: int) -> float:
    """Каждый элемент эталона засчитывается **один раз**, по лучшей позиции.

    Без этого уровне-зависимое сопоставление ломает метрику: эталон «nk1/st6.1»
    удовлетворяется любым её пунктом, и если в выдаче окажутся p1, p2 и p6, все
    три наберут вклад, тогда как знаменатель рассчитан на один релевантный
    элемент. На наборе слоя 1 это давало nDCG@10 = 1.13 — величину, невозможную
    по определению.
    """
    credited: set[str] = set()
    gain = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        for target in gold:
            if target in credited:
                continue
            if item == target or item.startswith(target + "/"):
                gain += 1.0 / math.log2(rank + 1)
                credited.add(target)
                break
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else 0.0


@dataclass
class Metrics:
    queries: int
    recall_at_20: float
    hit_at_20: float
    ndcg_at_10: float
    mrr: float
    #: recall@8 нужен, чтобы сравнение с реранкером было честным: он отдаёт
    #: top-8, поэтому его recall@20 ограничен восемью позициями и не
    #: сопоставим с конфигурациями, возвращающими 50.
    recall_at_8: float = 0.0
    hit_at_8: float = 0.0
    #: Максимум, который в принципе достижим на этом корпусе. Фильтр
    #: достижимости требует, чтобы в индексе была **хоть одна** эталонная
    #: статья, а не все, поэтому у запроса с тремя эталонами, из которых
    #: проиндексирован один, recall не может превысить 1/3. На корпусе одной
    #: части первой средний потолок составлял 0.789: без этой цифры результат
    #: выглядит хуже, чем есть, и непонятно, сколько ещё можно выжать.
    recall_ceiling: float = 1.0
    #: Медиана и 90-й перцентиль времени одного запроса, миллисекунды.
    #: ТЗ §7 требует эту колонку, но числа здесь — **сравнение конфигураций
    #: между собой на этой машине**, а не абсолютные показатели продукта:
    #: целевое железо P5 — 24 ГБ GPU, здесь 4 ГБ и CPU без AVX2.
    latency_ms_median: float = 0.0
    latency_ms_p90: float = 0.0

    def as_dict(self) -> dict:
        return {
            "queries": self.queries,
            "recall@20": round(self.recall_at_20, 4),
            "hit@20": round(self.hit_at_20, 4),
            "recall@8": round(self.recall_at_8, 4),
            "hit@8": round(self.hit_at_8, 4),
            "recall_ceiling": round(self.recall_ceiling, 4),
            "recall_share_of_achievable": (
                round(self.recall_at_20 / self.recall_ceiling, 4) if self.recall_ceiling else 0.0
            ),
            "nDCG@10": round(self.ndcg_at_10, 4),
            "MRR": round(self.mrr, 4),
            "latency_ms_median": round(self.latency_ms_median, 1),
            "latency_ms_p90": round(self.latency_ms_p90, 1),
        }


def recall_ceiling(queries: list[EvalQuery], indexed: set[str] | None, level: str) -> float:
    """Средняя доля эталона, которая вообще присутствует в индексе."""
    if not queries or indexed is None:
        return 1.0
    shares = []
    for query in queries:
        gold = query.gold_articles if level == "article" else query.gold_norms
        if not gold:
            continue
        present = sum(
            1 for item in gold
            if item in indexed or any(x.startswith(item + "/") for x in indexed)
        )
        shares.append(present / len(gold))
    return sum(shares) / len(shares) if shares else 1.0


def evaluate(
    queries: list[EvalQuery],
    run: Callable[[EvalQuery], list[str]],
    *,
    level: str = "article",
    indexed: set[str] | None = None,
) -> Metrics:
    """Прогнать поиск по запросам и посчитать метрики.

    `level` — на каком уровне сравнивать: `article` (эталон ФНС — статьи) или
    `norm` (синтетический набор знает пункт).
    """
    if not queries:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)

    recalls, hits, ndcgs, rrs, timings = [], [], [], [], []
    recalls8, hits8 = [], []
    for query in queries:
        gold = query.gold_articles if level == "article" else query.gold_norms
        started = time.perf_counter()
        retrieved = run(query)
        timings.append((time.perf_counter() - started) * 1000)
        recalls.append(recall_at_k(retrieved, gold, 20))
        hits.append(hit_at_k(retrieved, gold, 20))
        recalls8.append(recall_at_k(retrieved, gold, 8))
        hits8.append(hit_at_k(retrieved, gold, 8))
        ndcgs.append(ndcg_at_k(retrieved, gold, 10))
        rrs.append(reciprocal_rank(retrieved, gold))

    n = len(queries)
    ordered = sorted(timings)
    return Metrics(
        n,
        sum(recalls) / n,
        sum(hits) / n,
        sum(ndcgs) / n,
        sum(rrs) / n,
        recall_at_8=sum(recalls8) / n,
        hit_at_8=sum(hits8) / n,
        recall_ceiling=recall_ceiling(queries, indexed, level),
        latency_ms_median=ordered[n // 2],
        latency_ms_p90=ordered[min(int(n * 0.9), n - 1)],
    )


def by_slice(
    queries: list[EvalQuery],
    run: Callable[[EvalQuery], list[str]],
    key: Callable[[EvalQuery], str],
    *,
    level: str = "article",
    indexed: set[str] | None = None,
) -> dict[str, dict]:
    """Метрики в разрезе — по типу запроса или по источнику (ТЗ §7)."""
    groups: dict[str, list[EvalQuery]] = {}
    for query in queries:
        groups.setdefault(key(query), []).append(query)
    return {
        name: evaluate(group, run, level=level, indexed=indexed).as_dict()
        for name, group in sorted(groups.items())
    }
