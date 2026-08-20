"""Клиент ИПС «Законодательство России» (pravo.gov.ru).

Разобрано разведкой 2026-08-20; параметры нигде не задокументированы, поэтому
знание о них живёт здесь, а не в вызывающем коде.

Цепочка запросов:

    ?searchres=&<attrs>    -> frameset (сам по себе бесполезен)
    ?searchlist=&<attrs>   -> меню рабочего пространства, здесь же число найденных
    ?list_itself=&<attrs>  -> собственно список документов со ссылками ?docbody=&nd=<id>
    ?docbody=&nd=<id>      -> карточка документа, здесь же список редакций
    ?doc_itself=&nd=<id>&page=<N>&rdk=<R> -> текст редакции R, страница N

Поля формы расширенного поиска (все обязательны, пустые тоже):

    a1  Наименование        a0  Текст               a3  Вид акта
    a6  Принявший орган     a15 Принявший субъект   a8  Номер
    a16 Подписи             a17 Публикации          a4  Состояние
    a23 Ключевые слова      a7type/a7from/a7to/a7date  Дата
    bpas=cd00000 (федеральное законодательство), sort, textpres

Две ловушки, каждая из которых даёт правдоподобно-неверный результат молча:

1. **rdk — не диапазон, а список.** У НК ч.1 максимальный rdk равен 248, но
   существующих редакций 153; 96 значений из промежутка не существуют, и запрос
   такого rdk возвращает страницу «Текст документа отсутствует» с кодом 200.
   Обходить надо перечень из карточки, а не range().
2. **Список редакций в карточке — окно, а не полный перечень.** У ч.2 при
   максимальном rdk 864 карточка перечисляет 26 последних. Считать, что редакций
   столько, сколько в select, нельзя.
"""
from __future__ import annotations

import html as html_module
import re
import urllib.parse
from dataclasses import dataclass

from p3.config import PRAVO, Source
from p3.fetch import PoliteFetcher

IPS_PATH = "/proxy/ips/"

#: Страница отдаётся с кодом 200 и этим текстом вместо содержимого.
TEXT_ABSENT_MARKER = "Текст документа отсутствует"


@dataclass(frozen=True)
class IpsDocument:
    nd: str
    title: str


NK_PART_1 = IpsDocument(nd="102054722", title="Налоговый кодекс Российской Федерации. Часть первая")
NK_PART_2 = IpsDocument(nd="102067058", title="Налоговый кодекс Российской Федерации. Часть вторая")


@dataclass(frozen=True)
class Edition:
    """Редакция документа в терминах ИПС."""

    rdk: int
    label: str
    #: Дата в формате ДД.ММ.ГГГГ. У редакции 0 («Исходная редакция») её нет.
    date: str | None = None
    #: Номер изменяющего акта, если он назван в подписи редакции.
    amending_act: str | None = None

    @property
    def year(self) -> int | None:
        return int(self.date[-4:]) if self.date else None

    @property
    def is_initial(self) -> bool:
        return self.rdk == 0


class EditionTextAbsent(RuntimeError):
    """ИПС вернул страницу-заглушку вместо текста редакции."""


# --- разбор HTML ------------------------------------------------------

#: value опции — это "rdk,nd", а не голое число: <option value='240,102054722'>.
_OPTION_RE = re.compile(
    r"<option[^>]*value=[\"'](?P<rdk>\d+)(?:,(?P<nd>\d+))?[\"'][^>]*>(?P<label>.*?)</option>",
    re.S | re.I,
)
_DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
_ACT_RE = re.compile(r"№\s*([\w\.\-]+)")
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style).*?</\1>", re.S | re.I)


def strip_html(html: str) -> str:
    text = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def parse_editions(html: str) -> list[Edition]:
    """Список редакций из карточки документа, по возрастанию rdk.

    Один и тот же select встречается на странице несколько раз — дубликаты
    схлопываются по rdk.
    """
    # Скрипты вырезаются до поиска опций: в них встречается разметка, которая
    # иначе попадает в перечень редакций как настоящая.
    body = _SCRIPT_RE.sub(" ", html)
    editions: dict[int, Edition] = {}
    for match in _OPTION_RE.finditer(body):
        label = re.sub(r"\s+", " ", html_module.unescape(_TAG_RE.sub("", match.group("label")))).strip()
        if not label:
            continue
        rdk = int(match.group("rdk"))
        date = _DATE_RE.search(label)
        act = _ACT_RE.search(label)
        editions.setdefault(
            rdk,
            Edition(
                rdk=rdk,
                label=label,
                date=date.group(1) if date else None,
                amending_act=act.group(1) if act else None,
            ),
        )
    return [editions[k] for k in sorted(editions)]


def is_text_absent(html: str) -> bool:
    return TEXT_ABSENT_MARKER in html


# --- построение запросов ----------------------------------------------

def search_attrs(
    *,
    title: str = "",
    text: str = "",
    act_kind: str = "",
    number: str = "",
    sort: str = "7",
    text_present: str = "",
) -> dict[str, str]:
    """Полный набор полей формы. Пустые поля обязательны — без них поиск не срабатывает."""
    return {
        "bpas": "cd00000",
        "a3": act_kind, "a3type": "1", "a3value": "",
        "a6": "", "a6type": "1", "a6value": "",
        "a15": "", "a15type": "1", "a15value": "",
        "a7type": "3", "a7from": "", "a7to": "", "a7date": "",
        "a8": number, "a8type": "1",
        "a1": title,
        "a0": text,
        "a16": "", "a16type": "1", "a16value": "",
        "a17": "", "a17type": "1", "a17value": "",
        "a4": "", "a4type": "1", "a4value": "",
        "a23": "", "a23type": "1", "a23value": "",
        "textpres": text_present,
        "sort": sort,
    }


def _query(view: str, attrs: dict[str, str]) -> str:
    # ИПС принимает cp1251 в query string; utf-8 он не понимает.
    return f"{IPS_PATH}?{view}=&" + urllib.parse.urlencode(attrs, encoding="cp1251")


@dataclass(frozen=True)
class SearchHit:
    nd: str
    title: str


_ND_LINK_RE = re.compile(r"href=[\"']([^\"']*nd=(\d+)[^\"']*)[\"'][^>]*>(.{0,250}?)</a>", re.S)


def parse_search_results(html: str) -> list[SearchHit]:
    """Документы из списка результатов, в порядке выдачи, без дубликатов."""
    body = _SCRIPT_RE.sub(" ", html)
    hits: dict[str, SearchHit] = {}
    for _href, nd, raw_label in _ND_LINK_RE.findall(body):
        label = re.sub(r"\s+", " ", html_module.unescape(_TAG_RE.sub("", raw_label))).strip()
        if label and nd not in hits:
            hits[nd] = SearchHit(nd=nd, title=label)
    return list(hits.values())


class IpsClient:
    """Тонкая обёртка: строит URL, кэширует через raw_store, разбирает ответы."""

    def __init__(self, fetcher: PoliteFetcher, source: Source = PRAVO) -> None:
        self.fetcher = fetcher
        self.source = source

    # -- URL --

    def card_path(self, nd: str) -> str:
        return f"{IPS_PATH}?docbody=&nd={nd}"

    def edition_path(self, nd: str, rdk: int, page: int = 1) -> str:
        return f"{IPS_PATH}?doc_itself=&nd={nd}&page={page}&rdk={rdk}"

    def full_text_path(self, nd: str, rdk: int) -> str:
        """Полный текст редакции одним запросом.

        Без `print=1` эндпоинт отдаёт только начало документа и **обрывает его
        на середине предложения**, не сообщая об этом ничем: у НК ч.1 приходит
        142 статьи против 25. Параметры `page`, `pg`, `p` при этом принимаются и
        игнорируются, то есть постраничный обход дал бы один и тот же кусок,
        размноженный сколько угодно раз.
        """
        return f"{IPS_PATH}?doc_itself=&nd={nd}&rdk={rdk}&print=1"

    # -- запросы --

    def card_html(self, nd: str) -> str:
        return self.fetcher.get(self.card_path(nd), self.source).text

    def editions(self, nd: str) -> list[Edition]:
        return parse_editions(self.card_html(nd))

    def edition_html(self, nd: str, rdk: int, page: int = 1) -> str:
        html = self.fetcher.get(self.edition_path(nd, rdk, page), self.source).text
        if is_text_absent(html):
            raise EditionTextAbsent(f"nd={nd} rdk={rdk} page={page}: {TEXT_ABSENT_MARKER}")
        return html

    def edition_text(self, nd: str, rdk: int, page: int = 1) -> str:
        return strip_html(self.edition_html(nd, rdk, page))

    def full_text_html(self, nd: str, rdk: int) -> str:
        html = self.fetcher.get(self.full_text_path(nd, rdk), self.source).text
        if is_text_absent(html):
            raise EditionTextAbsent(f"nd={nd} rdk={rdk}: {TEXT_ABSENT_MARKER}")
        return html

    def search(self, **kwargs) -> list[SearchHit]:
        attrs = search_attrs(**kwargs)
        html = self.fetcher.get(_query("list_itself", attrs), self.source).text
        return parse_search_results(html)
