"""«Письма ФНС России, обязательные для применения налоговыми органами».

Второй источник silver set. Устроен так же, как `sources/nalog_ru.py`, и по той же
причине ценен: **разметку делает ведомство**. У письма есть поле `Теги:` с ярлыками
статей НК, поэтому эталон берётся готовым, а не извлекается эвристикой из текста.

Разведка 2026-08-20:

* список — GET `/rn77/about_fts/about_nalog/<N>.html`, 15 писем на страницу, нумерация с 1;
* письмо — GET `/rn77/about_fts/about_nalog/<id>/`;
* поля размечены как `<strong>Метка: </strong>значение`, теги — ссылками внутри `Теги:`;
* на выборке из 8 писем: теги статей у 6, вопрос по существу у 6.

Почему нужен отсев. Часть писем административные — «Об утверждении приказа…»,
«О направлении…». У них в поле `Вопрос:` стоит не вопрос, а тема распорядительного
документа, и как запрос silver set они бесполезны: эталон есть, а запроса нет.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

LIST_PATH = "/rn77/about_fts/about_nalog/"
RECORDS_PER_PAGE = 15

FIELD_NUMBER = "Номер"
FIELD_LETTER_DATE = "Дата письма"
FIELD_PUBLISHED = "Дата публикации"
FIELD_CATEGORY = "Категория (тематика) письма"
FIELD_TAGS = "Теги"
FIELD_QUESTION = "Вопрос"
FIELD_ANSWER = "Ответ"

#: Начала тем, за которыми стоит распорядительный документ, а не вопрос по существу.
ADMINISTRATIVE_PREFIXES = (
    "об утверждении",
    "о направлении",
    "о внесении изменен",
    "о признании утратив",
    "об отмене",
)

_LETTER_ID_RE = re.compile(r'href="%s(\d{6,})/"' % re.escape(LIST_PATH))


@dataclass(frozen=True)
class Letter:
    """Письмо ФНС, обязательное для применения."""

    letter_id: str
    #: Тема письма из поля «Вопрос:» — это и есть запрос silver set.
    question: str
    #: Ярлыки статей как их написала ФНС: «Статья 431 НК РФ».
    article_labels: tuple[str, ...] = ()
    number: str | None = None
    letter_date: str | None = None
    published: str | None = None
    category: str | None = None
    answer: str | None = None

    @property
    def is_administrative(self) -> bool:
        """Распорядительное письмо: эталон есть, а запроса по существу нет."""
        return is_administrative(self.question)

    @property
    def path(self) -> str:
        return f"{LIST_PATH}{self.letter_id}/"


def is_administrative(question: str) -> bool:
    lowered = question.strip().lower()
    return lowered.startswith(ADMINISTRATIVE_PREFIXES)


def list_path(page: int) -> str:
    if page < 1:
        raise ValueError("страницы нумеруются с 1")
    return f"{LIST_PATH}{page}.html"


def letter_path(letter_id: str) -> str:
    return f"{LIST_PATH}{letter_id}/"


def parse_letter_ids(html: str) -> list[str]:
    """Идентификаторы писем со страницы списка, без повторов, в порядке появления."""
    seen: dict[str, None] = {}
    for letter_id in _LETTER_ID_RE.findall(html):
        seen.setdefault(letter_id, None)
    return list(seen)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fields(html: str) -> dict[str, list]:
    """Поля вида `<strong>Метка: </strong>значение` до следующей метки."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, list] = {}
    for strong in soup.find_all("strong"):
        label = _clean(strong.get_text()).rstrip(":").strip()
        if not label:
            continue
        nodes = []
        for sibling in strong.next_siblings:
            # Следующая метка закрывает предыдущее поле. Проверять надо только у
            # тегов: у NavigableString есть строковый .find, который возвращает -1,
            # а -1 истинно — на этом цикл обрывался на первом же текстовом узле.
            if isinstance(sibling, Tag):
                if sibling.name == "strong" or sibling.find("strong") is not None:
                    break
            elif not isinstance(sibling, NavigableString):
                continue
            nodes.append(sibling)
        result.setdefault(label, nodes)
    return result


def parse_letter(html: str, letter_id: str) -> Letter | None:
    """Разобрать страницу письма. None, если нет темы — как запрос оно бесполезно."""
    fields = _fields(html)

    def text_of(label: str) -> str | None:
        nodes = fields.get(label)
        if nodes is None:
            return None
        joined = " ".join(
            node.get_text() if hasattr(node, "get_text") else str(node) for node in nodes
        )
        return _clean(joined) or None

    question = text_of(FIELD_QUESTION)
    if not question:
        return None

    labels: tuple[str, ...] = ()
    tag_nodes = fields.get(FIELD_TAGS)
    if tag_nodes:
        found = []
        for node in tag_nodes:
            if hasattr(node, "find_all"):
                found.extend(_clean(a.get_text()) for a in node.find_all("a"))
            if getattr(node, "name", None) == "a":
                found.append(_clean(node.get_text()))
        labels = tuple(label for label in found if label)

    return Letter(
        letter_id=letter_id,
        question=question,
        article_labels=labels,
        number=text_of(FIELD_NUMBER),
        letter_date=text_of(FIELD_LETTER_DATE),
        published=text_of(FIELD_PUBLISHED),
        category=text_of(FIELD_CATEGORY),
        answer=text_of(FIELD_ANSWER),
    )
