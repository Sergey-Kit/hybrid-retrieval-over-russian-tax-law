"""Разбор структуры НК из HTML ИПС: раздел → глава → статья → пункт.

Разметка ИПС (печатная версия документа, `print=1`):

* `<p class="H">` — заголовок: «РАЗДЕЛ …», «Глава N. …» или «Статья N. Название»;
* обычные `<p>` — тело: пункты, абзацы, примечания;
* `<span class="ed">` — текст в действующей редакции;
* `<a class="doclink">` — ссылки на изменяющие акты;
* **`<span class="W9">N</span>` — надстрочный индекс номера.**

Последнее — главная ловушка этого источника. «Статья 6¹» размечена как
`Статья 6<span class="W9">1</span>`, и означает она **статью 6.1**. При наивном
снятии тегов получается «Статья 61» — существующая, но совсем другая норма, а
ссылка «статьи 221¹» превращается в «статьи 2211», которой нет вовсе.

Ошибка тихая вдвойне: половина статей просто не находится. В части первой 275
заголовков статей, но регулярное выражение по «Статья N.» без обработки индексов
видит только 142 — теряются в том числе 6.1 (исчисление сроков), 93.1 (встречная
проверка), 101.4 и 129.1, то есть ядро процедурных норм.

Поэтому индексы разворачиваются **до** любого другого разбора.
"""
from __future__ import annotations

import html as html_module
import re
from dataclasses import dataclass, field

#: Надстрочный индекс: `6<span class="W9">1</span>` → `6.1`.
_SUPERSCRIPT_RE = re.compile(r'(?<=\d)<span class="W9"[^>]*>\s*([\d\-]+)\s*</span>')
_PARAGRAPH_RE = re.compile(r'<p(?P<attrs>[^>]*)>(?P<body>.*?)</p>', re.S)
_CLASS_RE = re.compile(r'class="([^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style).*?</\1>", re.S | re.I)

#: Номер статьи бывает с дефисом: «Статья 105.16-1» — это отдельная статья,
#: а не 105.16 с заголовком, начинающимся на «-1». Таких в части первой семь.
_ARTICLE_HEAD_RE = re.compile(
    r"^Статья\s+(?P<number>\d+(?:\.\d+)*(?:-\d+)?)\.?\s*(?P<title>.*)$", re.S
)
_CHAPTER_HEAD_RE = re.compile(r"^Глава\s+(?P<number>\d+(?:\.\d+)?)\.?\s*(?P<title>.*)$", re.S)
#: «Раздел V. НАЛОГОВАЯ ДЕКЛАРАЦИЯ…», «Раздел V2 …» (надстрочный индекс уже развёрнут
#: только после цифр, поэтому у римских номеров он остаётся приклеенным).
_SECTION_HEAD_RE = re.compile(
    r"^Раздел\s+(?P<number>[IVXL]+(?:\.?\d+)?)\.?\s*(?P<title>.*)$", re.I | re.S
)

#: Начало пункта: «1. », «12. ». Подпункт идёт как «1) ».
_CLAUSE_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)?)\.\s+(?=\S)")
_SUBCLAUSE_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)?)\)\s+(?=\S)")

#: Пометки об утрате силы. Такие нормы **сохраняются** с флагом: старые требования
#: ссылаются на них, и выбросить их значит оставить ссылки в пустоту (спека §10).
#: Формулировки в разметке: «(Утратил силу - …)», «(Пункт утратил силу - …)»,
#: «Абзац. (Утратил силу - …)», «(Исключена - …)». Между скобкой и словом
#: «утратил» может стоять ещё одно слово, поэтому оно допускается явно.
_REPEALED_RE = re.compile(
    r"\(\s*(?:\w+\s+)?утратил\w*\s+силу|\(\s*исключен\w*|признан\w*\s+утративш",
    re.IGNORECASE,
)


def expand_superscripts(html: str) -> str:
    """`6<span class="W9">1</span>` → `6.1`. Делать до любого другого разбора."""
    return _SUPERSCRIPT_RE.sub(lambda m: "." + m.group(1), html)


def _text_of(fragment: str) -> str:
    text = html_module.unescape(_TAG_RE.sub("", fragment))
    return re.sub(r"[\s ]+", " ", text).strip()


@dataclass
class Clause:
    """Пункт статьи. `number is None` — вводный абзац без нумерации."""

    number: str | None
    text: str
    is_repealed: bool = False

    def norm_unit_id(self, article_id: str) -> str:
        return f"{article_id}/p{self.number}" if self.number else article_id


@dataclass
class Article:
    number: str
    title: str
    part: int
    section: str | None = None
    section_title: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    clauses: list[Clause] = field(default_factory=list)
    is_repealed: bool = False

    @property
    def norm_unit_id(self) -> str:
        return f"nk{self.part}/st{self.number}"

    @property
    def text(self) -> str:
        return "\n".join(clause.text for clause in self.clauses)


class UnsupportedMarkupError(RuntimeError):
    """Документ непустой, но ни одной статьи не распознано."""


#: Ниже этого объёма текста пустой результат — не аномалия, а норма (заглушка,
#: фрагмент, тестовая строка). Выше — почти наверняка чужая разметка.
_MIN_TEXT_FOR_GUARD = 50_000


def parse_articles(html: str, *, part: int = 1, strict: bool = True) -> list[Article]:
    """Разобрать печатную версию документа в список статей с пунктами.

    `strict` включает guard на чужую разметку. Редакции ИПС до примерно 2020 года
    свёрстаны иначе: весь документ идёт сплошным текстом в считанных абзацах, и
    заголовки статей ничем не выделены. На таком входе разбор возвращал пустой
    список — документ на полтора мегабайта превращался в ноль статей, и заметить
    это можно было только по счётчику. Теперь такой случай падает.
    """
    html = expand_superscripts(_SCRIPT_RE.sub(" ", html))

    articles: list[Article] = []
    section: str | None = None
    section_title: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    current: Article | None = None

    for match in _PARAGRAPH_RE.finditer(html):
        css = _CLASS_RE.search(match.group("attrs"))
        classes = css.group(1).split() if css else []
        # Заголовки статей и глав идут классом H, разделы — классом T.
        is_header = bool({"H", "T"} & set(classes))
        text = _text_of(match.group("body"))
        if not text:
            continue

        if is_header:
            if (found := _SECTION_HEAD_RE.match(text)):
                section = found.group("number")
                section_title = found.group("title").strip(" .").capitalize() or None
                chapter = chapter_title = current = None
                continue
            if (found := _CHAPTER_HEAD_RE.match(text)):
                chapter = found.group("number")
                chapter_title = found.group("title").strip(" .") or None
                current = None
                continue
            if (found := _ARTICLE_HEAD_RE.match(text)):
                current = Article(
                    number=found.group("number"),
                    title=found.group("title").strip(),
                    part=part,
                    section=section,
                    section_title=section_title,
                    chapter=chapter,
                    chapter_title=chapter_title,
                    is_repealed=bool(_REPEALED_RE.search(found.group("title"))),
                )
                articles.append(current)
                continue
            # Прочие заголовки (например, «(Дополнение главой …)») тело не открывают.
            continue

        if current is None:
            continue

        clause_match = _CLAUSE_RE.match(text)
        # Номера пунктов внутри статьи уникальны по закону. Повтор номера — это
        # не пункт, а строка таблицы: в статьях о ставках и сборах тело пункта
        # бывает перечнем кодов вида «61 25 00; 75 00 00;», и строки начинаются
        # с числа с точкой. Без этой проверки статья 333.45 давала несколько
        # десятков «пунктов 2» с одинаковым идентификатором чанка.
        if clause_match and clause_match.group("number") in {
            c.number for c in current.clauses
        }:
            clause_match = None

        if clause_match:
            number = clause_match.group("number")
            body = text[clause_match.end():]
            current.clauses.append(
                Clause(number=number, text=body, is_repealed=bool(_REPEALED_RE.search(body)))
            )
        elif current.clauses:
            # Продолжение предыдущего пункта: абзац, подпункт, примечание.
            previous = current.clauses[-1]
            previous.text = f"{previous.text}\n{text}"
        else:
            current.clauses.append(
                Clause(number=None, text=text, is_repealed=bool(_REPEALED_RE.search(text)))
            )

    # Статья, утратившая силу целиком, выглядит как пустой заголовок и один абзац
    # «(Статья утратила силу - …)». Такие сохраняются с флагом, а не выбрасываются:
    # требования прошлых лет на них ссылаются, и без них ссылка уходит в пустоту.
    for article in articles:
        if article.clauses and all(clause.is_repealed for clause in article.clauses):
            article.is_repealed = True

    if strict and not articles and len(html) > _MIN_TEXT_FOR_GUARD:
        raise UnsupportedMarkupError(
            f"документ на {len(html)} символов не дал ни одной статьи — "
            "скорее всего это старая вёрстка ИПС, где текст идёт сплошняком "
            "и заголовки статей не размечены"
        )

    return articles
