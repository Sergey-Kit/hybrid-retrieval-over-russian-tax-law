"""Абзацы нормы: цитата ведёт к точному месту — «пункт 5, абзац второй».

Чанк хранит границы строк (`para_offsets`) внутри `text_bm25`, но строка — ещё
не абзац в юридическом смысле. Номер абзаца, который назовёт юрист или
инспекция, получается только с учётом трёх вещей, найденных на статьях слоя 1:

1. **Пометки ИПС** — «(Пункт в редакции Федерального закона …)»,
   «(Дополнение пунктом - …)» — отдельные строки, но не абзацы. Их больше
   пятидесяти в одних только статьях слоя 1; посчитай их — и «абзац третий»
   укажет не туда.
2. **Подпункты** лежат отдельными чанками (`…/sp2`), и в каждом повторяется
   вводная строка пункта («Налогоплательщики обязаны:»). Подпункт — не абзац.
3. **Длинный пункт разрезан** на чанки `#0`, `#1`…, смещения в каждом с нуля,
   а нумерация абзацев продолжается. Поэтому структура строится по всем чанкам
   одной версии нормы, а не по одному чанку.

Проверено на реальной ссылке: «абзацем вторым пункта 5 статьи 93.1» — вторая
строка п. 5, про требования по п. 2 ст. 93.1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

#: Строка целиком в скобках со ссылкой на закон — пометка редакции, не абзац.
#: Строка только из скобочных групп со ссылкой на закон — пометка редакции, не
#: абзац. Групп бывает несколько подряд: «(Дополнение пунктом - …) (В редакции
#: …)» — одна группа на всю строку пропускала их, и они считались абзацами.
_GROUP = r"\((?:[^()]|\([^()]*\))*\)"
_ONLY_GROUPS = re.compile(rf"^\s*(?:{_GROUP}\s*)+$")
_LAW = re.compile(r"Федеральн\w+\s+закон|Федеральным\s+законом|утратил\w*\s+силу|"
                  r"в\s+редакции", re.I)


class _Editorial:
    @staticmethod
    def match(line: str) -> bool:
        return bool(_ONLY_GROUPS.match(line) and _LAW.search(line))


EDITORIAL = _Editorial()
SUBCLAUSE = re.compile(r"^\s*(\d+(?:\.\d+)*)\)\s")

ORDINALS = ("первый", "второй", "третий", "четвёртый", "пятый", "шестой", "седьмой",
            "восьмой", "девятый", "десятый", "одиннадцатый", "двенадцатый")


@dataclass(frozen=True)
class Paragraph:
    chunk_id: str
    start: int            # в text_bm25 этого чанка
    end: int
    text: str
    kind: str             # абзац | подпункт | вводная | пометка
    number: int | None    # номер абзаца; для подпункта — None
    label: str            # «абзац второй», «подпункт 3», «вводная часть пункта»


def ordinal(n: int) -> str:
    return ORDINALS[n - 1] if 1 <= n <= len(ORDINALS) else f"{n}-й"


def _offsets(chunk: dict) -> list[list[int]]:
    po = chunk.get("para_offsets") or []
    return json.loads(po) if isinstance(po, str) else po


def _ordinal_of(chunk: dict) -> int:
    tail = str(chunk.get("chunk_id", "")).rsplit("#", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0


def structure(chunks: list[dict]) -> list[Paragraph]:
    """Абзацы одной версии нормы по всем её чанкам, в порядке текста."""
    out: list[Paragraph] = []
    counter = 0
    for chunk in sorted(chunks, key=_ordinal_of):
        body = chunk.get("text_bm25") or ""
        is_subclause = "/sp" in str(chunk.get("norm_unit_id", ""))
        sub: str | None = None          # номер подпункта, внутри которого стоим
        sub_par = 0
        for start, end in _offsets(chunk):
            line = body[start:end]
            cid = chunk.get("chunk_id", "")
            if EDITORIAL.match(line):
                out.append(Paragraph(cid, start, end, line, "пометка", None, "пометка редакции"))
                continue
            m = SUBCLAUSE.match(line)
            if m:
                sub, sub_par = m.group(1), 1
                out.append(Paragraph(cid, start, end, line, "подпункт", None,
                                     f"подпункт {sub}"))
                continue
            if is_subclause and sub is None:
                # Вводная строка пункта, повторённая в чанке подпункта.
                out.append(Paragraph(cid, start, end, line, "вводная", None,
                                     "вводная часть пункта"))
                continue
            if is_subclause:
                sub_par += 1
                out.append(Paragraph(cid, start, end, line, "абзац", sub_par,
                                     f"подпункт {sub}, абзац {ordinal(sub_par)}"))
                continue
            counter += 1
            out.append(Paragraph(cid, start, end, line, "абзац", counter,
                                 f"абзац {ordinal(counter)}"))
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def locate(paragraphs: list[Paragraph], quote: str) -> Paragraph | None:
    """Абзац, содержащий цитату дословно (без учёта пробелов, регистра и «ё»).

    Нечёткого поиска нет намеренно: цитата, не найденная дословно, — повод
    сказать «не нашли», а не привязать её к похожему абзацу.
    """
    q = _norm(quote)
    if not q:
        return None
    hits = [p for p in paragraphs if p.kind != "пометка" and q in _norm(p.text)]
    return hits[0] if len(hits) == 1 else None


def paragraph(paragraphs: list[Paragraph], number: int) -> Paragraph | None:
    """«Абзац второй» — по номеру среди абзацев пункта (не подпунктов)."""
    return next((p for p in paragraphs
                 if p.kind == "абзац" and p.number == number and "подпункт" not in p.label),
                None)
