"""Синтетические `as_of`-запросы: проверка темпорального фильтра.

Спека §2 предлагает ровно этот приём: «взять нормы, у которых известны ≥2
редакции, и построить запросы вида „редакция на 2023-06-01". Эталон известен по
построению — это чистая проверка темпорального фильтра без всякой разметки».

Набор строится поверх вопросов слоя 1: берутся те, чья норма менялась, и на
каждую её версию делается запрос с датой внутри интервала действия этой версии.
Эталон — конкретная версия, а не просто норма.

**Что именно проверяется.** Механика фильтра: выбирается ли та версия, которая
действовала на дату. Это **не** юридическое утверждение о том, какая редакция
действовала: границы интервалов приближённые, см. `p3.nk.temporal`.
"""
from __future__ import annotations

from dataclasses import dataclass

from p3.nk.temporal import NormVersion, SENTINEL_VALID_TO
from p3.silver.build import QUERY_TYPE_LEXICAL, QUERY_TYPE_NATURAL, SilverPair
from p3.silver.layer1 import QUESTIONS, Layer1Question

SOURCE_TEMPORAL = "synthetic_as_of"


def _probe_date(version: NormVersion) -> int:
    """Дата внутри интервала действия версии.

    Для последней версии берём её начало плюс запас, иначе — середину.
    Границы намеренно не трогаем: на них попадание зависит от того, включительна
    граница или нет, и тест начал бы проверять соглашение, а не фильтр.
    """
    if version.valid_to >= SENTINEL_VALID_TO:
        return version.valid_from + 1
    start, end = version.valid_from, version.valid_to
    # Даты в формате YYYYMMDD не арифметичны, поэтому берём не среднее, а
    # безопасную точку: следующий день после начала интервала.
    return start + 1 if start + 1 < end else start


def _with_date(text: str, as_of: int, *, formal: bool) -> str:
    day = f"{as_of % 100:02d}.{as_of // 100 % 100:02d}.{as_of // 10000}"
    return f"{text} в редакции на {day}" if formal else f"{text} по состоянию на {day}"


@dataclass(frozen=True)
class TemporalCase:
    pair: SilverPair
    as_of: int
    expected_version_id: str


def build_temporal_cases(
    versions: dict[str, list[NormVersion]],
    questions: tuple[Layer1Question, ...] = QUESTIONS,
) -> list[TemporalCase]:
    """По вопросу слоя 1 и каждой версии его нормы — запрос с датой."""
    cases: list[TemporalCase] = []
    for index, question in enumerate(questions):
        history = versions.get(question.norm_unit_id, [])
        if len(history) < 2:
            continue
        for version in history:
            as_of = _probe_date(version)
            for suffix, source_text, query_type, formal in (
                ("demand", question.demand_excerpt, QUERY_TYPE_LEXICAL, True),
                ("natural", question.natural, QUERY_TYPE_NATURAL, False),
            ):
                pair = SilverPair(
                    query_id=f"as_of-{index:02d}-v{version.version_no}-{suffix}",
                    query=_with_date(source_text, as_of, formal=formal),
                    gold_norm_ids=(version.version_id,),
                    gold_article_ids=(question.article_id,),
                    query_type=query_type,
                    source=SOURCE_TEMPORAL,
                    source_id=f"{question.norm_unit_id}@v{version.version_no}",
                    cluster_id=f"as_of/{question.norm_unit_id}",
                )
                cases.append(TemporalCase(pair, as_of, version.version_id))
    return cases


def temporal_accuracy(cases: list[TemporalCase], retrieved: dict[str, list[str]], top_k: int = 8) -> dict:
    """Доля запросов, где возвращена версия, действовавшая на `as_of`.

    Отдельно считается доля, где вернулась **какая-то другая** версия той же
    нормы: это и есть цена ошибки темпоральной модели — ответ выглядит
    правильным, а редакция не та.
    """
    correct = wrong_version = missing = 0
    for case in cases:
        found = retrieved.get(case.pair.query_id, [])[:top_k]
        unit = case.expected_version_id.split("@")[0]
        same_unit = [v for v in found if v.split("@")[0] == unit]
        if case.expected_version_id in found:
            correct += 1
        elif same_unit:
            wrong_version += 1
        else:
            missing += 1
    total = len(cases) or 1
    return {
        "cases": len(cases),
        "correct_version": correct,
        "correct_version_share": round(correct / total, 4),
        "wrong_version_of_right_norm": wrong_version,
        "norm_not_found": missing,
    }
