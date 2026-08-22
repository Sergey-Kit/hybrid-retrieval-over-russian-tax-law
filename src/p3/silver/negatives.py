"""Негативы: запросы, ответа на которые в корпусе нет.

ТЗ формулирует прямо: «без негативов метрики измеряют красноречие, а не
правильность». До сих пор у **всех** запросов оценки ответ в корпусе был, и
система ни разу не проверялась на вопросе, отвечать на который не следует.

Негативы не выдуманы, а взяты из тех же источников, что и позитивы:

* **`gold_outside_nk`** — решения и письма, размеченные ведомством на акты вне
  НК: закон 129-ФЗ о госрегистрации, ГК. Настоящие обращения в налоговый орган,
  ответ на которые в Налоговом кодексе отсутствует по существу.
* **`norm_removed_from_edition`** — вопросы про ЕНВД (глава 26.3), отменённый с
  2021 года и изъятый из действующей редакции. Норма когда-то была, в корпусе
  её нет.

## Что именно здесь измеряется

P3 — ретривер, у него нет механизма «не знаю»: он всегда что-то возвращает.
Поэтому мерить recall на негативах бессмысленно. Измеряется **разделимость
оценок**: систематически ли негативы получают более низкий балл, чем позитивы.
Если да — порог уверенности в P5 построить можно, и известно, какой ценой.
Если нет — «не знаю» придётся строить другим механизмом, и знать это лучше
сейчас, чем после интеграции с генерацией.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REASON_OUTSIDE_NK = "gold_outside_nk"
REASON_REMOVED = "norm_removed_from_edition"


@dataclass(frozen=True)
class NegativeQuery:
    query_id: str
    query: str
    reason: str
    source: str
    #: Чем размечен исходный документ — для прослеживаемости.
    original_labels: tuple[str, ...] = ()


def from_agency_items(items, *, source: str, prefix: str) -> list[NegativeQuery]:
    """Записи, размеченные ведомством на акты вне НК."""
    from p3.refs.normalize import find_citations

    negatives: list[NegativeQuery] = []
    for item in items:
        labels = tuple(item.article_labels)
        if not labels:
            continue
        if getattr(item, "is_administrative", False):
            continue
        codes = {
            c.code
            for label in labels
            for c in find_citations(label, default_code=None)
        }
        if not codes or "nk" in codes:
            continue
        query = getattr(item, "topic", None) or getattr(item, "question", "")
        identifier = getattr(item, "decision_id", None) or getattr(item, "letter_id", "")
        if query and identifier:
            negatives.append(NegativeQuery(
                query_id=f"neg-{prefix}-{identifier}",
                query=query,
                reason=REASON_OUTSIDE_NK,
                source=source,
                original_labels=labels,
            ))
    return negatives


def from_unreachable_pairs(rows, indexed_articles: set[str]) -> list[NegativeQuery]:
    """Пары silver set, чей эталон целиком отсутствует в корпусе."""
    negatives: list[NegativeQuery] = []
    for row in rows:
        gold = set(row.get("gold_article_ids") or ())
        if gold and not (gold & indexed_articles):
            negatives.append(NegativeQuery(
                query_id=f"neg-{row['query_id']}",
                query=row["query"],
                reason=REASON_REMOVED,
                source=row.get("source", "unknown"),
                original_labels=tuple(sorted(gold)),
            ))
    return negatives


def write_jsonl(negatives: list[NegativeQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in sorted(negatives, key=lambda n: n.query_id):
            fh.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


def separability(positive_scores: list[float], negative_scores: list[float]) -> dict:
    """Насколько оценки позитивов отделимы от негативов.

    AUC — вероятность того, что случайный позитив получит балл выше случайного
    негатива. 0.5 — неотличимы, 1.0 — разделяются идеально.
    """
    if not positive_scores or not negative_scores:
        return {"auc": 0.0, "positives": len(positive_scores), "negatives": len(negative_scores)}

    wins = ties = 0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    total = len(positive_scores) * len(negative_scores)

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    return {
        "auc": round((wins + 0.5 * ties) / total, 4),
        "positives": len(positive_scores),
        "negatives": len(negative_scores),
        "median_positive": round(median(positive_scores), 4),
        "median_negative": round(median(negative_scores), 4),
    }
