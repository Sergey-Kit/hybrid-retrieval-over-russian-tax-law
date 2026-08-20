"""Сервис ФНС «Решения по жалобам» (nalog.gov.ru).

Разобрано разведкой 2026-08-20.

Почему этот источник оказался важнее, чем предполагала спека. §2 спеки предлагает
строить silver set, извлекая цитаты норм из текста писем. Здесь этого не нужно:
**ФНС размечает каждое решение сама**. Запись списка несёт одновременно

* «Темы споров» — описание спора на естественном языке, и
* «Статьи НК РФ» — статьи, к которым ФНС это решение отнесла.

То есть пара «запрос → эталонные нормы» берётся готовой, от ведомства, и целый
класс ошибок — «нормы не приведены к norm_unit_id, метрики занижены» из §10 спеки —
для этого источника просто не возникает.

Дополнительно: в фильтре сервиса 263 темы, из них 251 — статьи НК вида
«Статья 101.4 НК РФ». Это независимый набор для проверки нормализатора ссылок:
разметку писал не я.

Устройство:

* список — GET `/rn77/service/complaint_decision/<N>.html`, 15 записей на страницу,
  нумерация с 1; за последней страницей записей просто нет;
* фильтры — POST на `/rn77/service/complaint_decision/` (ASP.NET WebForms,
  требуется `__VIEWSTATE` с той же страницы);
* запись — `div.text_block`, поля внутри — пары `<label>Имя:</label>` / значение;
* «Темы споров» ссылается на страницу самого решения: `<a href='16172699/'>`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

LIST_PATH = "/rn77/service/complaint_decision/"
RECORDS_PER_PAGE = 15

#: Подписи полей в карточке решения.
FIELD_PUBLISHED = "Дата публикации"
FIELD_DECISION = "Дата и номер решения"
FIELD_AUTHORITY = "Налоговый орган, вынесший решение"
FIELD_ARTICLES = "Статьи НК РФ"
FIELD_TAX = "Вид налога"
FIELD_TOPIC = "Темы споров"
FIELD_ACTS = "Нормативно"  # «Нормативно – правовые акты», тире неразрывное


@dataclass(frozen=True)
class Decision:
    """Одна запись из списка решений по жалобам."""

    decision_id: str
    #: Описание спора на естественном языке — это и есть запрос silver set.
    topic: str
    #: Ярлыки статей как их написала ФНС: «Статья 93 НК РФ».
    article_labels: tuple[str, ...] = ()
    published: str | None = None
    decided: str | None = None
    authority: str | None = None
    tax: str | None = None
    acts: str | None = None

    @property
    def detail_path(self) -> str:
        return f"{LIST_PATH}{self.decision_id}/"


def list_path(page: int) -> str:
    if page < 1:
        raise ValueError("страницы нумеруются с 1")
    return f"{LIST_PATH}{page}.html"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_decisions(html: str) -> list[Decision]:
    """Разобрать страницу списка. Записи без темы или без статей пропускаются."""
    soup = BeautifulSoup(html, "lxml")
    decisions: list[Decision] = []

    for block in soup.select("div.text_block"):
        fields: dict[str, object] = {}
        for row in block.select("div.row"):
            label = row.find("label")
            if label is None:
                continue
            name = _clean(label.get_text()).rstrip(":")
            value_cells = [c for c in row.find_all("div", recursive=False) if c is not label.parent]
            cell = value_cells[-1] if value_cells else None
            if cell is None:
                continue
            fields[name] = cell

        topic_cell = fields.get(FIELD_TOPIC)
        articles_cell = fields.get(FIELD_ARTICLES)
        if topic_cell is None or articles_cell is None:
            continue

        topic_link = topic_cell.find("a")
        if topic_link is None:
            continue
        decision_id = topic_link.get("href", "").strip("/")
        topic = _clean(topic_link.get_text())
        if not decision_id.isdigit() or not topic:
            continue

        labels = tuple(_clean(a.get_text()) for a in articles_cell.find_all("a"))
        if not labels:
            text = _clean(articles_cell.get_text())
            labels = (text,) if text else ()
        if not labels:
            continue

        def value(name: str) -> str | None:
            cell = fields.get(name)
            return _clean(cell.get_text()) if cell is not None else None

        acts = next((_clean(c.get_text()) for k, c in fields.items() if k.startswith(FIELD_ACTS)), None)

        decisions.append(
            Decision(
                decision_id=decision_id,
                topic=topic,
                article_labels=labels,
                published=value(FIELD_PUBLISHED),
                decided=value(FIELD_DECISION),
                authority=value(FIELD_AUTHORITY),
                tax=value(FIELD_TAX),
                acts=acts,
            )
        )
    return decisions


def parse_topic_filter(html: str) -> list[tuple[str, str]]:
    """Опции фильтра «Темы споров»: (id, подпись).

    Используется как независимый набор для проверки нормализатора ссылок.
    """
    soup = BeautifulSoup(html, "lxml")
    select = soup.find("select", id=re.compile(r"ddlTopic$"))
    if select is None:
        return []
    options = []
    for option in select.find_all("option"):
        value = (option.get("value") or "").strip()
        label = _clean(option.get_text())
        if value and value != "0" and label and label != "-":
            options.append((value, label))
    return options
