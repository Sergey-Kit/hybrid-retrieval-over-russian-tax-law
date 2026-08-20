"""Сборка silver set: пары «запрос → эталонные нормы».

Источник — решения по жалобам ФНС, где ведомство само проставило и описание спора,
и статьи НК. Наша работа сводится к трём вещам, каждая из которых нужна, чтобы
итоговые метрики не врали:

1. **Ярлык статьи → norm_unit_id.** «Статья 93 НК РФ» → `nk1:93`.
2. **Дедупликация.** Формулировки тем повторяются почти дословно; без схлопывания
   один и тот же спор попадёт и в train, и в test, и recall окажется завышенным.
3. **Сплит по кластерам, а не по записям.** Спека принимает `Recall@20 >= 0.9` на
   silver set и на нём же предлагает подбирать веса BM25F и параметр RRF — то есть
   меряет на обучающей выборке. Отчётные числа считаются один раз на test.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from p3.refs.normalize import find_citations
from p3.sources.nalog_ru import Decision

#: Доли train/dev/test. Тюнинг — только на train и dev.
DEFAULT_RATIOS = (0.6, 0.2, 0.2)

QUERY_TYPE_LEXICAL = "lexical_anchor"
QUERY_TYPE_NATURAL = "natural_language"
QUERY_TYPE_MIXED = "mixed"

#: Ниже этого числа слов (без учёта самой ссылки) запрос считается чисто лексическим.
_NATURAL_WORDS_MIN = 5


@dataclass(frozen=True)
class SilverPair:
    query_id: str
    query: str
    #: Эталонные нормы на уровне пункта, если он назван, иначе статьи.
    gold_norm_ids: tuple[str, ...]
    #: Эталон на уровне статьи — основная метрика, ФНС размечает статьями.
    gold_article_ids: tuple[str, ...]
    query_type: str
    source: str
    source_id: str
    #: Ключ дедупликации; сплит идёт по нему, а не по записям.
    cluster_id: str


def normalize_for_clustering(text: str) -> str:
    """Ключ, по которому темы считаются одной и той же.

    Ссылки на нормы из текста убираются: две темы, отличающиеся только номером
    статьи, — это всё-таки один и тот же шаблон формулировки, и разносить их по
    разным сплитам небезопасно.
    """
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    for citation in reversed(find_citations(text, default_code=None)):
        start, end = citation.span
        text = text[:start] + " " + text[end:]
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_query(text: str) -> str:
    """Тип запроса для разрезов §7 спеки.

    Правило намеренно простое и проверяемое глазами: есть ли в самом тексте
    запроса ссылка на норму, и остаётся ли что-то содержательное помимо неё.
    """
    citations = find_citations(text, default_code=None)
    if not citations:
        return QUERY_TYPE_NATURAL
    residual = text
    for citation in reversed(citations):
        start, end = citation.span
        residual = residual[:start] + " " + residual[end:]
    words = [w for w in re.findall(r"\w+", residual) if len(w) > 2]
    return QUERY_TYPE_MIXED if len(words) >= _NATURAL_WORDS_MIN else QUERY_TYPE_LEXICAL


def pair_from_decision(decision: Decision) -> SilverPair | None:
    """Пара из одного решения, либо None, если разметка не про НК."""
    norm_ids: list[str] = []
    article_ids: list[str] = []
    for label in decision.article_labels:
        for citation in find_citations(label, default_code=None):
            if citation.code != "nk":
                continue
            if citation.norm_unit_id not in norm_ids:
                norm_ids.append(citation.norm_unit_id)
            if citation.article_id not in article_ids:
                article_ids.append(citation.article_id)
    if not article_ids:
        return None

    return SilverPair(
        query_id=f"fns-{decision.decision_id}",
        query=decision.topic,
        gold_norm_ids=tuple(norm_ids),
        gold_article_ids=tuple(article_ids),
        query_type=classify_query(decision.topic),
        source="fns_complaint_decision",
        source_id=decision.decision_id,
        cluster_id=normalize_for_clustering(decision.topic),
    )


def dedup(pairs: list[SilverPair]) -> list[SilverPair]:
    """Одна пара на кластер: первая по порядку, эталоны объединяются."""
    by_cluster: dict[str, SilverPair] = {}
    merged_norms: dict[str, list[str]] = defaultdict(list)
    merged_articles: dict[str, list[str]] = defaultdict(list)

    for pair in pairs:
        by_cluster.setdefault(pair.cluster_id, pair)
        for norm_id in pair.gold_norm_ids:
            if norm_id not in merged_norms[pair.cluster_id]:
                merged_norms[pair.cluster_id].append(norm_id)
        for article_id in pair.gold_article_ids:
            if article_id not in merged_articles[pair.cluster_id]:
                merged_articles[pair.cluster_id].append(article_id)

    result = []
    for cluster_id, pair in by_cluster.items():
        result.append(
            SilverPair(
                **{
                    **asdict(pair),
                    "gold_norm_ids": tuple(merged_norms[cluster_id]),
                    "gold_article_ids": tuple(merged_articles[cluster_id]),
                }
            )
        )
    return result


def split(
    pairs: list[SilverPair],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, list[SilverPair]]:
    """Разбиение по кластерам. Один кластер целиком лежит в одном сплите."""
    clusters = sorted({p.cluster_id for p in pairs})
    random.Random(seed).shuffle(clusters)

    train_end = int(len(clusters) * ratios[0])
    dev_end = train_end + int(len(clusters) * ratios[1])
    assignment = {}
    for index, cluster_id in enumerate(clusters):
        if index < train_end:
            assignment[cluster_id] = "train"
        elif index < dev_end:
            assignment[cluster_id] = "dev"
        else:
            assignment[cluster_id] = "test"

    result: dict[str, list[SilverPair]] = {"train": [], "dev": [], "test": []}
    for pair in pairs:
        result[assignment[pair.cluster_id]].append(pair)
    return result


def build(decisions: list[Decision], *, seed: int = 42) -> dict[str, list[SilverPair]]:
    pairs = [p for p in (pair_from_decision(d) for d in decisions) if p is not None]
    return split(dedup(pairs), seed=seed)


def write_jsonl(pairs: list[SilverPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in sorted(pairs, key=lambda p: p.query_id):
            fh.write(json.dumps(asdict(pair), ensure_ascii=False, sort_keys=True) + "\n")


def summarize(splits: dict[str, list[SilverPair]]) -> dict[str, object]:
    all_pairs = [p for pairs in splits.values() for p in pairs]
    by_type: dict[str, int] = defaultdict(int)
    for pair in all_pairs:
        by_type[pair.query_type] += 1
    return {
        "total": len(all_pairs),
        "by_split": {name: len(pairs) for name, pairs in splits.items()},
        "by_query_type": dict(sorted(by_type.items())),
        "distinct_articles": len({a for p in all_pairs for a in p.gold_article_ids}),
    }
