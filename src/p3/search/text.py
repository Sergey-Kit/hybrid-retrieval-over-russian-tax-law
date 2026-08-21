"""Подготовка текста для BM25: нормализация ссылок, затем лемматизация.

Порядок здесь — не деталь реализации, а требование спеки §3, и он неочевиден.

**Ссылки нормализуются до лемматизации.** «п. 3 ст. 88», «пункт 3 статьи 88»,
«п.3 ст.88» — это один и тот же, самый информативный токен запроса. Если сначала
лемматизировать, он распадётся на «пункт», «3», «статья», «88» — четыре
высокочастотных токена с низким IDF, и точное совпадение по ссылке перестанет
работать. Спека называет это отдельным тихим дефектом.

**Стоп-слова отключены полностью.** Стандартный русский стоп-лист выбрасывает
«не», «либо», «иной» — в праве это несущие слова, и «не вправе истребовать»
превратилось бы во «вправе истребовать». IDF справится сам.

**Числа, даты и коды не лемматизируются**: морфология там только портит.
"""
from __future__ import annotations

import functools
import re

from p3.refs.normalize import find_citations

_WORD_RE = re.compile(r"[\w§]+", re.UNICODE)
_NUMERIC_RE = re.compile(r"^\d[\d.,/-]*$")


@functools.lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def reference_token(norm_unit_id: str) -> str:
    """`nk1/st93.1/p2` → `nk1_st93_1_p2` — один неделимый токен для BM25.

    Точка заменяется тоже: словоделитель `[\w]+` разорвал бы `nk1_st93.1`
    на `nk1_st93` и `1`, а статьи с точкой (93.1, 129.1, 101.4) — это как раз
    ядро процедурных норм.
    """
    return norm_unit_id.replace("/", "_").replace(".", "_")


def normalize_references(text: str) -> str:
    """Заменить все ссылки на нормы одним токеном каждая.

    Делать **до** лемматизации: иначе самый информативный токен запроса
    распадается на четыре высокочастотных.
    """
    citations = find_citations(text, default_code="nk")
    if not citations:
        return text
    result = text
    for citation in sorted(citations, key=lambda c: c.full_span[0], reverse=True):
        # Берём границу с названием кодекса: «НК РФ» — часть цитаты.
        start, end = citation.full_span
        result = result[:start] + reference_token(citation.norm_unit_id) + result[end:]
    return result


@functools.lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    if _NUMERIC_RE.match(word) or "_" in word:
        return word
    return _morph().parse(word)[0].normal_form


def tokenize(text: str, *, lemmatize: bool = True) -> list[str]:
    """Токены для BM25. Стоп-слова намеренно не удаляются."""
    prepared = normalize_references(text).lower().replace("ё", "е")
    tokens = _WORD_RE.findall(prepared)
    if not lemmatize:
        return tokens
    return [lemma(token) for token in tokens]
