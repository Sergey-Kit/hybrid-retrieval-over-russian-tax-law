"""Поиск ссылок на нормы НК в тексте документа — надстройка над `normalize`.

Перенесено из P4 (2026-10-26, задача C5), чтобы P4 и P5 разбирали ссылки
одним кодом из одного места. `normalize` при этом не меняется: от его
поведения зависят граф ссылок и silver set, а значит опубликованные числа P3.

Что добавляется к `normalize.find_citations`:

1. **Несколько пунктов на одно упоминание статьи.** «В соответствии с пунктом 2,
   абзацем вторым пункта 5 статьи 93.1» — это две нормы. `normalize` обрывает
   цепочку квалификаторов на втором «пункте» и оставляет только п. 5. Для
   требований это подмена основания: п. 1 ст. 93.1 — встречное истребование в
   рамках проверки (5 дней), п. 2 — вне проверки по конкретной сделке (10 дней).
2. **Квалификаторы справа.** «НК ст.93.1 п.2» — сокращённая запись из таблиц.
   В тексте документов порядок обратный, и `normalize` смотрит только влево.
3. **Отсев цитат официального вестника.** «Собрание законодательства …, ст. 1»
   при default_code="nk" становится nk1/st1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import Citation, find_citations, nk_part

GAZETTE = re.compile(r"Собрание законодательства|Российская газета", re.I)

_NUM = r"\d+(?:\.\d+)*"
_CLAUSE = re.compile(rf"(?:п\.\s*|пункт[а-я]*\s+)({_NUM})", re.I)
_SUB = re.compile(rf"(?:пп\.\s*|подпункт[а-я]*\s+)({_NUM})", re.I)
_ART = re.compile(rf"(?:ст\.\s*|стать[а-я]+\s+)({_NUM})", re.I)


@dataclass(frozen=True)
class Ref:
    """Ссылка на норму НК с позицией в исходном тексте."""

    article: str
    point: str | None = None
    subpoint: str | None = None
    paragraph: int | None = None
    span: tuple[int, int] = (0, 0)
    raw: str = ""

    @property
    def unit_id(self) -> str:
        parts = [f"nk{nk_part(self.article)}", f"st{self.article}"]
        if self.point:
            parts.append(f"p{self.point}")
        if self.subpoint:
            parts.append(f"sp{self.subpoint}")
        if self.paragraph:
            parts.append(f"a{self.paragraph}")
        return "/".join(parts)

    @property
    def article_id(self) -> str:
        return f"nk{nk_part(self.article)}/st{self.article}"

    @property
    def is_clause_level(self) -> bool:
        return self.point is not None

    def __str__(self) -> str:
        out = f"НК ст.{self.article}"
        if self.point:
            out += f" п.{self.point}"
        if self.subpoint:
            out += f" пп.{self.subpoint}"
        if self.paragraph:
            out += f" абз.{self.paragraph}"
        return out


def _from_citation(c: Citation) -> Ref:
    return Ref(article=c.article, point=c.point, subpoint=c.subpoint,
               paragraph=c.paragraph, span=c.span, raw=c.raw_text)


def _extra_points(text: str, c: Citation, floor: int) -> list[str]:
    """Пункты, потерянные обрывом цепочки квалификаторов слева.

    P3 включает в span только ту часть цепочки, которую сумел пройти, поэтому
    остальное лежит непосредственно слева. Окно обязано упираться в конец
    предыдущей ссылки (floor): иначе «ст. 129.1» унаследует «пунктом 5» от
    соседней статьи — та же ошибка, от которой защищает сам P3.
    """
    left = text[max(floor, c.span[0] - 60):c.span[0]]
    if re.search(r"[.!?]\s", left):
        left = left[re.search(r"[.!?]\s", left).end():]
    subs = [m.span() for m in _SUB.finditer(left)]
    out = []
    for m in _CLAUSE.finditer(left):
        if any(s <= m.start() < e for s, e in subs):
            continue
        if m.group(1) != c.point:
            out.append(m.group(1))
    return out


def _right_qualifiers(text: str, c: Citation) -> tuple[str | None, str | None]:
    """«ст. 76 п. 3 пп. 2» — сокращённая запись, квалификаторы примыкают справа."""
    right = text[c.span[1]:c.span[1] + 40]
    point = sub = None
    if m := _CLAUSE.match(right.lstrip()):
        point = m.group(1)
        right = right.lstrip()[m.end():]
    if m := _SUB.match(right.lstrip()):
        sub = m.group(1)
    return point, sub


def find(text: str, default_code: str = "nk") -> list[Ref]:
    """Все ссылки на нормы НК в тексте, в порядке появления, без повторов."""
    text = text or ""
    out: list[Ref] = []
    seen: set[str] = set()

    floor = 0
    for c in find_citations(text, default_code=default_code):
        previous_floor, floor = floor, max(floor, c.span[1])
        if c.code != "nk":
            continue
        if GAZETTE.search(text[max(0, c.span[0] - 120):c.span[0]]):
            continue

        candidates = [_from_citation(c)]
        for point in _extra_points(text, c, previous_floor):
            candidates.append(Ref(c.article, point, c.subpoint, None, c.span, c.raw_text))
        if c.point is None and c.subpoint is None:
            point, sub = _right_qualifiers(text, c)
            if point or sub:
                candidates.append(Ref(c.article, point, sub, c.paragraph, c.span, c.raw_text))
                candidates.pop(0)      # уточнённая ссылка заменяет ссылку на статью

        for ref in candidates:
            if ref.unit_id not in seen:
                seen.add(ref.unit_id)
                out.append(ref)
    return out


def parse_cell(value: str | None) -> list[Ref]:
    """Содержимое ячейки таблицы: «НК ст.93.1 п.2; НК ст.126 п.2»."""
    if not value:
        return []
    out: list[Ref] = []
    seen: set[str] = set()
    for chunk in re.split(r"[;\n]+", str(value)):
        if not chunk.strip():
            continue
        for ref in find(chunk):
            if ref.unit_id not in seen:
                seen.add(ref.unit_id)
                out.append(ref)
    return out
