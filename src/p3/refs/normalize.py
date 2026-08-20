"""Нормализация юридических ссылок в структурный norm_unit_id.

Вход — живой текст письма или решения: «в соответствии с подпунктом 2 пункта 1
статьи 31 Налогового кодекса», «п. 3 ст. 88 НК РФ», «статьями 100 - 101 Кодекса».
Выход — список найденных ссылок с разобранной иерархией.

Зачем это отдельный модуль и почему он строится первым:

* без него silver set не собрать — цитата в письме должна стать идентификатором нормы;
* тот же разбор нужен вехе 2 для графа refs_out и для нормализации перед лемматизацией,
  причём спека называет «ссылки не нормализованы до лемматизации» отдельным тихим
  дефектом: BM25 иначе не находит по самому информативному токену запроса.

Ограничение, которое надо держать в голове: на вехе 1 парсера НК ещё нет, поэтому
norm_unit_id **синтаксический** — он отражает то, что написано в тексте, и не проверен
по реальной иерархии кодекса. Существует ли пункт 7 в статье 88 и не утратил ли он силу,
здесь не известно. Поэтому исходный текст цитаты сохраняется рядом: на вехе 2 разметку
можно будет уточнить, не перекачивая корпус.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Граница между частями НК: часть первая заканчивается статьёй 142.x,
#: часть вторая начинается со статьи 143 (НДС).
NK_PART_TWO_FIRST_ARTICLE = 143

# --- словарь форм ------------------------------------------------------

_ARTICLE = r"ст\.|стат(?:ья|ьи|ье|ью|ьей|ьёй|ей|ями|ьями|ьях|атей)"
_POINT = r"п\.|пункт(?:а|е|у|ом|ы|ов|ам|ами|ах)?"
_SUBPOINT = r"пп\.|п\.\s?п\.|подп\.|подпункт(?:а|е|у|ом|ы|ов|ам|ами|ах)?"
_PART = r"ч\.|част(?:ь|и|ью|ей|ям|ями|ях)"
_ABZAC = r"абз\.|абзац(?:а|е|у|ем|ы|ев|ам|ами|ах)?"

#: Порядковые слова для абзацев: «абзацем вторым».
_ORDINALS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5,
    "шест": 6, "седьм": 7, "восьм": 8, "девят": 9, "десят": 10,
    "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
    "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18,
    "девятнадцат": 19, "двадцат": 20,
}
_ORDINAL_RE = re.compile(
    r"\b(" + "|".join(sorted(_ORDINALS, key=len, reverse=True)) + r")(?:ый|ой|ым|ого|ому|ом|ая|ье|ьим|ьего|ьем|им|его|ем|ий)\b",
    re.IGNORECASE,
)

#: Номер: 88, 93.1, 142.8.
_NUM = r"\d+(?:\.\d+)*"
#: Список номеров: «88, 93 и 93.1», «100 - 101», «3 и 4».
_NUM_LIST = rf"{_NUM}(?:\s*(?:,|и|-|–|—)\s*{_NUM})*"

_UNIT_RE = re.compile(
    rf"(?P<kind>{_SUBPOINT}|{_ABZAC}|{_POINT}|{_PART}|{_ARTICLE})"
    rf"\s*(?P<nums>{_NUM_LIST}|{_ORDINAL_RE.pattern})",
    re.IGNORECASE,
)

#: Маркеры кодексов. Порядок важен: более специфичные раньше, «Кодекс» без
#: уточнения трактуется как НК только потому, что в письмах Минфина и ФНС после
#: первого полного упоминания дальше пишут просто «Кодекса».
_CODE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("koap", re.compile(r"КоАП|Кодекс\w*\s+(?:Российской Федерации\s+)?об административных", re.IGNORECASE)),
    ("gk", re.compile(r"\bГК\b|Гражданск\w*\s+кодекс", re.IGNORECASE)),
    ("uk", re.compile(r"\bУК\b|Уголовн\w*\s+кодекс", re.IGNORECASE)),
    ("apk", re.compile(r"\bАПК\b|Арбитражн\w*\s+процессуальн\w*", re.IGNORECASE)),
    ("gpk", re.compile(r"\bГПК\b|Гражданск\w*\s+процессуальн\w*", re.IGNORECASE)),
    ("tk", re.compile(r"\bТК\b|Трудов\w*\s+кодекс", re.IGNORECASE)),
    ("bk", re.compile(r"\bБК\b|Бюджетн\w*\s+кодекс", re.IGNORECASE)),
    ("nk", re.compile(r"\bНК\b|Налогов\w*\s+кодекс|\bКодекс\w*", re.IGNORECASE)),
    # Не кодекс вовсе: «статьей 5 Федерального закона», «пунктом 2 Приказа».
    # Без этой строки такие ссылки молча уезжали бы в НК по умолчанию и
    # засоряли silver set нормами, которых в нём быть не должно.
    (
        "other",
        re.compile(
            r"Федеральн\w*\s+закон|Закон\w*\s+(?:Российск|РФ|г\.|№|N\s)|"
            r"Приказ\w*|Постановлени\w*|Распоряжени\w*|Указ\w*|Положени\w*|"
            r"Инструкци\w*|Конституци\w*|Соглашени\w*|Конвенци\w*|Регламент\w*|"
            r"Правил\w*\s+|\d+-ФЗ",
            re.IGNORECASE,
        ),
    ),
)

#: Насколько далеко вправо от номера статьи искать название кодекса.
_CODE_LOOKAHEAD = 70


@dataclass(frozen=True)
class Citation:
    """Одна разобранная ссылка."""

    raw_text: str
    code: str
    article: str
    point: str | None = None
    subpoint: str | None = None
    paragraph: int | None = None
    part: int | None = None
    #: Был ли кодекс назван явно, или подставлен по умолчанию.
    code_explicit: bool = True
    span: tuple[int, int] = (0, 0)

    @property
    def norm_unit_id(self) -> str:
        head = f"nk{self.part}" if self.code == "nk" and self.part else self.code
        parts = [head, self.article]
        if self.point:
            parts.append(f"p{self.point}")
        if self.subpoint:
            parts.append(f"sp{self.subpoint}")
        if self.paragraph:
            parts.append(f"a{self.paragraph}")
        return ":".join(parts)

    @property
    def article_id(self) -> str:
        """Идентификатор статьи целиком, без пунктов — для recall на уровне статьи."""
        head = f"nk{self.part}" if self.code == "nk" and self.part else self.code
        return f"{head}:{self.article}"


@dataclass
class _Unit:
    kind: str
    values: list[str]
    start: int
    end: int


def nk_part(article: str) -> int:
    """Часть НК по номеру статьи. Детерминированное правило, а не справочник."""
    major = int(article.split(".")[0])
    return 2 if major >= NK_PART_TWO_FIRST_ARTICLE else 1


def _classify(keyword: str) -> str:
    kw = keyword.lower().replace(" ", "")
    if kw.startswith(("пп", "подп", "п.п")):
        return "subpoint"
    if kw.startswith("абз"):
        return "paragraph"
    if kw.startswith("ч"):
        return "part_of_article"
    if kw.startswith("ст"):
        return "article"
    if kw.startswith("п"):
        return "point"
    return "unknown"


def _expand_numbers(raw: str) -> list[str]:
    """«88, 93 и 93.1» → [88, 93, 93.1]; «100 - 101» → [100, 101]."""
    ordinal = _ORDINAL_RE.search(raw)
    if ordinal:
        return [str(_ORDINALS[ordinal.group(1).lower()])]

    tokens = re.split(r"\s*(,|и|-|–|—)\s*", raw)
    numbers: list[str] = []
    pending_range = False
    for token in tokens:
        if token in {"-", "–", "—"}:
            pending_range = True
            continue
        if token in {",", "и"} or not token.strip():
            continue
        value = token.strip()
        if pending_range and numbers and "." not in numbers[-1] and "." not in value:
            start, end = int(numbers[-1]), int(value)
            if 0 < end - start <= 50:
                numbers.extend(str(n) for n in range(start + 1, end + 1))
                pending_range = False
                continue
        pending_range = False
        numbers.append(value)
    return numbers


def _detect_code(text: str, from_pos: int) -> tuple[str | None, bool]:
    window = text[from_pos : from_pos + _CODE_LOOKAHEAD]
    # Обрываем окно на следующей ссылке, чтобы «ст. 19.4 КоАП» не окрасила предыдущую.
    cut = re.search(rf"(?:{_ARTICLE})\s*{_NUM}", window, re.IGNORECASE)
    if cut:
        window = window[: cut.start()]
    best: tuple[int, str] | None = None
    for code, pattern in _CODE_PATTERNS:
        found = pattern.search(window)
        if found and (best is None or found.start() < best[0]):
            best = (found.start(), code)
    return (best[1], True) if best else (None, False)


def find_citations(text: str, *, default_code: str | None = "nk") -> list[Citation]:
    """Найти и разобрать все ссылки на нормы в тексте.

    default_code подставляется, когда кодекс не назван рядом со статьёй. Для писем
    Минфина и ФНС это оправданно, но результат помечается code_explicit=False,
    чтобы долю таких случаев можно было измерить, а не принять на веру.
    """
    units = [
        _Unit(
            kind=_classify(m.group("kind")),
            values=_expand_numbers(m.group("nums")),
            start=m.start(),
            end=m.end(),
        )
        for m in _UNIT_RE.finditer(text)
    ]

    citations: list[Citation] = []
    for index, unit in enumerate(units):
        if unit.kind != "article":
            continue

        # Квалификаторы стоят слева и вплотную: «подпунктом 2 пункта 1 статьи 31».
        qualifiers: dict[str, _Unit] = {}
        cursor = unit.start
        for previous in reversed(units[:index]):
            gap = text[previous.end : cursor]
            if not re.fullmatch(r"[\s,]*", gap):
                break
            if previous.kind in {"article", "unknown"} or previous.kind in qualifiers:
                break
            qualifiers[previous.kind] = previous
            cursor = previous.start

        code, explicit = _detect_code(text, unit.end)
        if code is None:
            code, explicit = default_code, False
        if code is None:
            continue

        raw_start = min([unit.start] + [q.start for q in qualifiers.values()])
        raw_text = text[raw_start : unit.end].strip()

        points = qualifiers["point"].values if "point" in qualifiers else [None]
        subpoints = qualifiers["subpoint"].values if "subpoint" in qualifiers else [None]
        paragraphs = qualifiers["paragraph"].values if "paragraph" in qualifiers else [None]
        # «часть» в НК не используется, но встречается в КоАП — там она играет роль пункта.
        if "part_of_article" in qualifiers and points == [None]:
            points = qualifiers["part_of_article"].values

        for article in unit.values:
            for point in points:
                for subpoint in subpoints:
                    for paragraph in paragraphs:
                        citations.append(
                            Citation(
                                raw_text=raw_text,
                                code=code,
                                article=article,
                                point=point,
                                subpoint=subpoint,
                                paragraph=int(paragraph) if paragraph else None,
                                part=nk_part(article) if code == "nk" else None,
                                code_explicit=explicit,
                                span=(raw_start, unit.end),
                            )
                        )
    return citations


def norm_unit_ids(text: str, *, default_code: str | None = "nk") -> list[str]:
    """Уникальные norm_unit_id в порядке появления."""
    seen: dict[str, None] = {}
    for citation in find_citations(text, default_code=default_code):
        seen.setdefault(citation.norm_unit_id, None)
    return list(seen)
