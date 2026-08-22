"""Сборка раздела результатов из файлов замеров.

Цифры в документации не переписываются руками. Любая ручная копия расходится с
данными при первом же перепрогоне, и расхождение это тихое: таблица выглядит
правдоподобно, просто относится к прошлой конфигурации.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Подписи на двух языках: README есть и русский, и английский, а цифры в них
#: обязаны совпадать. Переписывать таблицы руками нельзя — разойдутся при
#: первом же перепрогоне.
STRINGS = {
    "ru": {
        "title": "## Результаты",
        "intro": [
            "Цифры посчитаны **один раз на held-out test**; тюнинг весов шёл только на train.",
            "Таблицы собираются из файлов замеров командой `p3 render-results`, а не переписываются",
            "руками — иначе они молча расходятся с данными при следующем прогоне.",
        ],
        "silver": "### Silver set, test",
        "reachable": "Достижимо **{n} запросов из {total}** ({share:.0%}). Метрики считаются на достижимом",
        "reachable2": [
            "подмножестве — запросах, чей эталон присутствует в индексе, — и доля печатается",
            "рядом с цифрами: молча отфильтровать было бы приукрашиванием, только незаметным.",
        ],
        "layer1": "### Синтетический набор слоя 1 (эталон на уровне пункта)",
        "header": "| Конфигурация | recall@20 | hit@20 | recall@8 | nDCG@10 | MRR | мс |",
        "rerank_note": [
            "Реранкер отдаёт top-8, поэтому его `recall@20` ограничен восемью позициями и по этой",
            "колонке несопоставим с конфигурациями, возвращающими 50. Его следует сравнивать по",
            "`recall@8`, `nDCG@10` и `MRR`.",
        ],
        "gap": "### Главный замер: цена живого языка",
        "gap_intro": "Одна норма, один эталон, две формулировки:",
        "gap_header": "| Формулировка | recall@20 | nDCG@10 |",
        "temporal": "### Темпоральный тест (`{mode}`)",
    },
    "en": {
        "title": "## Results",
        "intro": [
            "Figures come from a **single run over the held-out test split**; weights were tuned on",
            "train only. Tables are generated from the measurement files by `p3 render-results`",
            "rather than typed, so they cannot drift from the data on the next run.",
        ],
        "silver": "### Silver set, test split",
        "reachable": "**{n} of {total} queries are reachable** ({share:.0%}). Metrics are computed over the",
        "reachable2": [
            "reachable subset — those whose gold is present in the index — and that share is printed",
            "next to the numbers: filtering silently would be the same flattery, just harder to see.",
        ],
        "layer1": "### Synthetic layer-1 set (gold at clause level)",
        "header": "| Configuration | recall@20 | hit@20 | recall@8 | nDCG@10 | MRR | ms |",
        "rerank_note": [
            "The reranker returns the top 8, so its `recall@20` is capped by eight slots and is not",
            "comparable with configurations returning 50. Compare it on `recall@8`, `nDCG@10`",
            "and `MRR`.",
        ],
        "gap": "### The headline measurement: what natural language costs",
        "gap_intro": "One norm, one gold, two phrasings:",
        "gap_header": "| Phrasing | recall@20 | nDCG@10 |",
        "temporal": "### Temporal test (`{mode}`)",
    },
}

CONFIG_NAMES_EN = {
    "dense": "Dense only",
    "bm25": "BM25 only",
    "hybrid": "**Hybrid (RRF)**",
    "hybrid+rerank": "Hybrid + reranker",
}
QUERY_TYPE_NAMES_EN = {
    "lexical_anchor": "Excerpt from a demand (citation present)",
    "natural_language": "Accountant's question (no citation)",
    "mixed": "Mixed phrasing",
}

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


#: Ниже этого потолка оговорка осмысленна. При 0.99+ разница между «достижимым»
#: и «возможным» меньше погрешности, и пояснение превращается в шум.
_CEILING_NOTE_BELOW = 0.95


def _ceiling_note(configurations: dict, lang: str = "ru") -> list[str]:
    """Потолок recall общий для всех конфигураций набора."""
    ceiling = next(
        (m.get("recall_ceiling") for m in configurations.values()
         if m.get("recall_ceiling") is not None),
        None,
    )
    if ceiling is None:
        return []
    if ceiling >= _CEILING_NOTE_BELOW:
        if lang == "ru":
            return ["", "Потолок recall@20 на этом корпусе — {:.3f}: эталон почти полностью".format(ceiling),
                    "присутствует в индексе, и метрика не ограничена его неполнотой."]
        return ["", "The recall@20 ceiling on this corpus is {:.3f}: the gold is almost entirely".format(ceiling),
                "present in the index, so the metric is not bounded by its incompleteness."]
    best = max(m["recall@20"] for m in configurations.values())
    return [
        "",
        "**Потолок recall@20 на этом корпусе — {:.3f}.** Фильтр достижимости требует, чтобы в".format(ceiling),
        "индексе была хоть одна эталонная статья, а не все, поэтому часть эталона недостижима",
        "физически: даже идеальный поиск не превысил бы этой величины. Лучший результат {:.3f} —".format(best),
        "это **{:.0%} достижимого**.".format(best / ceiling),
    ]


def _table(configurations: dict, lang: str = "ru") -> list[str]:
    names = CONFIG_NAMES if lang == "ru" else CONFIG_NAMES_EN
    header = [STRINGS[lang]["header"], "|---|---|---|---|---|---|---|"]
    return header + [_row(names.get(k, k), v) for k, v in configurations.items()]


def render(eval_path: Path, temporal_path: Path | None = None, lang: str = "ru") -> str:
    t = STRINGS[lang]
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    lines: list[str] = [t["title"], ""] + t["intro"] + [""]

    silver = next((k for k in report["sets"] if k.startswith("silver")), None)
    if silver:
        block = report["sets"][silver]
        lines += [
            t["silver"], "",
            t["reachable"].format(
                n=block["queries_reachable"], total=block["queries_total"],
                share=block["reachable_share"]),
        ] + t["reachable2"] + [""] + _table(block["configurations"], lang) \
          + _ceiling_note(block["configurations"], lang) + [""]

    if "layer1" in report["sets"]:
        block = report["sets"]["layer1"]
        lines += [t["layer1"], ""] + _table(block["configurations"], lang) + [""] \
            + t["rerank_note"] + [""]
        if "by_query_type" in block:
            names = QUERY_TYPE_NAMES if lang == "ru" else QUERY_TYPE_NAMES_EN
            lines += [t["gap"], "", t["gap_intro"], "", t["gap_header"], "|---|---|---|"]
            for kind, m in block["by_query_type"].items():
                lines.append("| {} | {:.3f} | {:.3f} |".format(
                    names.get(kind, kind), m["recall@20"], m["nDCG@10"]))
            lines.append("")

    if temporal_path and temporal_path.exists():
        lines += _temporal_section(json.loads(temporal_path.read_text(encoding="utf-8")), lang)
    return "\n".join(lines)


def _temporal_section(t_report: dict, lang: str) -> list[str]:
    accuracy, cost = t_report["temporal_accuracy"], t_report["filter_cost"]
    head = STRINGS[lang]["temporal"].format(mode=t_report["mode"])
    if lang == "ru":
        return [
            head, "",
            "{} синтетических запросов «редакция на дату», эталон — конкретная версия нормы.".format(
                accuracy["cases"]),
            "", "| | |", "|---|---|",
            "| Возвращена верная редакция | **{:.0%}** |".format(accuracy["correct_version_share"]),
            "| Возвращена чужая редакция нужной нормы | **{}** |".format(
                accuracy["wrong_version_of_right_norm"]),
            "| Норма не найдена вовсе | {} |".format(accuracy["norm_not_found"]),
            "| Цена фильтра относительно индекса одной редакции | **{} п.п.** |".format(
                cost["drop_vs_baseline_pp"]),
            "",
            "Ноль во второй строке — главная цифра: когда норма найдена, редакция всегда верная,",
            "значит промахи это отказы поиска, а не темпоральной модели.",
            "",
        ]
    return [
        head, "",
        "{} synthetic \"edition as of <date>\" queries; the gold is a specific version of a norm.".format(
            accuracy["cases"]),
        "", "| | |", "|---|---|",
        "| Correct edition returned | **{:.0%}** |".format(accuracy["correct_version_share"]),
        "| Wrong edition of the right norm | **{}** |".format(
            accuracy["wrong_version_of_right_norm"]),
        "| Norm not found at all | {} |".format(accuracy["norm_not_found"]),
        "| Filter cost against a single-edition index | **{} points** |".format(
            cost["drop_vs_baseline_pp"]),
        "",
        "The zero on the second row is the figure that matters: whenever the norm is found the",
        "edition is right, so every miss is a retrieval failure rather than a temporal one.",
        "",
    ]
