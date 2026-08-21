"""Сборка раздела результатов из файлов замеров.

Цифры в документации не переписываются руками. Любая ручная копия расходится с
данными при первом же перепрогоне, и расхождение это тихое: таблица выглядит
правдоподобно, просто относится к прошлой конфигурации.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_NAMES = {
    "dense": "Чистый dense",
    "bm25": "Чистый BM25",
    "hybrid": "**Гибрид (RRF)**",
    "hybrid+rerank": "Гибрид + реранкер",
}
QUERY_TYPE_NAMES = {
    "lexical_anchor": "Выдержка из требования (якорь есть)",
    "natural_language": "Живой вопрос (якоря нет)",
    "mixed": "Смешанная формулировка",
}


def _row(name: str, m: dict) -> str:
    return "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} |".format(
        name, m["recall@20"], m["hit@20"], m.get("recall@8", 0.0),
        m["nDCG@10"], m["MRR"], m.get("latency_ms_median", 0.0),
    )


def _ceiling_note(configurations: dict) -> list[str]:
    """Потолок recall общий для всех конфигураций набора."""
    ceiling = next(
        (m.get("recall_ceiling") for m in configurations.values()
         if m.get("recall_ceiling") is not None),
        None,
    )
    if ceiling is None or ceiling >= 0.999:
        return []
    best = max(m["recall@20"] for m in configurations.values())
    return [
        "",
        "**Потолок recall@20 на этом корпусе — {:.3f}.** Фильтр достижимости требует, чтобы в".format(ceiling),
        "индексе была хоть одна эталонная статья, а не все, поэтому часть эталона недостижима",
        "физически. Лучший результат {:.3f} — это **{:.0%} достижимого**, а не {:.0%} возможного.".format(
            best, best / ceiling, best),
    ]


def _table(configurations: dict) -> list[str]:
    header = [
        "| Конфигурация | recall@20 | hit@20 | recall@8 | nDCG@10 | MRR | мс |",
        "|---|---|---|---|---|---|---|",
    ]
    return header + [_row(CONFIG_NAMES.get(k, k), v) for k, v in configurations.items()]


def render(eval_path: Path, temporal_path: Path | None = None) -> str:
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    lines: list[str] = [
        "## Результаты",
        "",
        "Цифры посчитаны **один раз на held-out test**; тюнинг весов шёл только на train.",
        "Таблицы собираются из файлов замеров командой `p3 render-results`, а не переписываются",
        "руками — иначе они молча расходятся с данными при следующем прогоне.",
        "",
    ]

    silver = next((k for k in report["sets"] if k.startswith("silver")), None)
    if silver:
        block = report["sets"][silver]
        lines += [
            "### Silver set, test",
            "",
            "Достижимо **{} запросов из {}** ({:.0%}): в индексе только часть первая, а набор".format(
                block["queries_reachable"], block["queries_total"], block["reachable_share"]),
            "построен из решений и писем, где большинство упоминаний — статьи части второй.",
            "Метрики считаются на достижимом подмножестве, доля печатается рядом.",
            "",
        ] + _table(block["configurations"]) + _ceiling_note(block["configurations"]) + [""]

    if "layer1" in report["sets"]:
        block = report["sets"]["layer1"]
        lines += [
            "### Синтетический набор слоя 1 (эталон на уровне пункта)",
            "",
        ] + _table(block["configurations"]) + _ceiling_note(block["configurations"]) + [
            "",
            "Реранкер отдаёт top-8, поэтому его `recall@20` ограничен восемью позициями и по этой",
            "колонке несопоставим с конфигурациями, возвращающими 50. Его следует сравнивать по",
            "`recall@8`, `nDCG@10` и `MRR`.",
            "",
        ]
        if "by_query_type" in block:
            lines += [
                "### Главный замер: цена живого языка",
                "",
                "Одна норма, один эталон, две формулировки:",
                "",
                "| Формулировка | recall@20 | nDCG@10 |",
                "|---|---|---|",
            ]
            for kind, m in block["by_query_type"].items():
                lines.append("| {} | {:.3f} | {:.3f} |".format(
                    QUERY_TYPE_NAMES.get(kind, kind), m["recall@20"], m["nDCG@10"]))
            lines.append("")

    if temporal_path and temporal_path.exists():
        t = json.loads(temporal_path.read_text(encoding="utf-8"))
        accuracy, cost = t["temporal_accuracy"], t["filter_cost"]
        lines += [
            "### Темпоральный тест (`{}`)".format(t["mode"]),
            "",
            "{} синтетических запросов «редакция на дату», эталон — конкретная версия нормы.".format(
                accuracy["cases"]),
            "",
            "| | |",
            "|---|---|",
            "| Возвращена верная редакция | **{:.0%}** |".format(accuracy["correct_version_share"]),
            "| Возвращена чужая редакция нужной нормы | **{}** |".format(
                accuracy["wrong_version_of_right_norm"]),
            "| Норма не найдена вовсе | {} |".format(accuracy["norm_not_found"]),
            "| Цена фильтра относительно индекса одной редакции | **{} п.п.** (порог ТЗ: {}) |".format(
                cost["drop_vs_baseline_pp"], cost["criterion_max_drop_pp"]),
            "",
            "Ноль во второй строке — главная цифра: когда норма найдена, редакция всегда верная,",
            "значит промахи это отказы поиска, а не темпоральной модели. Чужая редакция опаснее",
            "ненайденной нормы, потому что выглядит правильным ответом.",
            "",
            "Цена фильтра меряется относительно индекса **одной редакции**: recall@20 равен",
            "{:.4f} и там, и в темпоральном индексе с фильтром. Фильтр возвращает систему ровно".format(
                cost["recall@20_single_edition_baseline"]),
            "к базовой линии, добавляя при этом возможность отвечать на любую дату.",
            "",
            "Сравнивать с нефильтрованным индексом всех версий нельзя: там одна норма",
            "представлена несколькими версиями, любая засчитывается как попадание, и recall@20",
            "выходит {:.4f} — на {} п.п. выше исключительно за счёт дублей. Такой индекс вдобавок".format(
                cost["recall@20_all_versions_unfiltered"], cost["duplicate_version_bonus_pp"]),
            "выдаёт устаревшие редакции, то есть не является системой, которую можно поставить.",
            "",
        ]
    return "\n".join(lines)
