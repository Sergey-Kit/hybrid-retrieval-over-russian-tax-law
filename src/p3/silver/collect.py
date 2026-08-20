"""Сбор решений по жалобам ФНС в raw_store.

Обход идёт по страницам списка, пока страница отдаёт записи. Верхняя граница
намеренно не зашивается константой: пагинатор на сайте показывает 46 страниц,
но записи находились и дальше, а на глубоких номерах список пустеет. Признак
конца — пустая страница, а не число из вёрстки.
"""
from __future__ import annotations

import logging

from p3.config import NALOG
from p3.fetch import PoliteFetcher
from p3.sources.nalog_ru import Decision, list_path, parse_decisions

log = logging.getLogger(__name__)

#: Предохранитель от бесконечного обхода, если сайт начнёт отдавать одно и то же.
MAX_PAGES = 200


def collect_decisions(
    fetcher: PoliteFetcher,
    *,
    max_pages: int = MAX_PAGES,
    stop_after_empty: int = 2,
) -> list[Decision]:
    """Пройти список до конца и вернуть решения без повторов."""
    seen: dict[str, Decision] = {}
    empty_streak = 0

    for page in range(1, max_pages + 1):
        html = fetcher.get(list_path(page), NALOG).text
        decisions = parse_decisions(html)

        if not decisions:
            empty_streak += 1
            log.info("страница %s пуста (%s подряд)", page, empty_streak)
            if empty_streak >= stop_after_empty:
                break
            continue

        empty_streak = 0
        new = 0
        for decision in decisions:
            if decision.decision_id not in seen:
                seen[decision.decision_id] = decision
                new += 1
        log.info("страница %s: записей %s, новых %s", page, len(decisions), new)

        # Сайт начал повторять уже виденное — дальше идти незачем.
        if new == 0:
            break

    return list(seen.values())
