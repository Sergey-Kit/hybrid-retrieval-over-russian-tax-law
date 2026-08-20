"""Разбор структуры НК. Фикстура — срез реального документа из ИПС."""
from pathlib import Path

import pytest

from p3.nk.parse import Article, Clause, expand_superscripts, parse_articles

FIXTURES = Path(__file__).parent / "fixtures"


def articles():
    return parse_articles((FIXTURES / "nk_part1_slice.html").read_text(encoding="utf-8"), part=1)


# --- надстрочные индексы ----------------------------------------------

def test_superscript_becomes_dotted_number():
    """«Статья 6¹» — это статья 6.1, а не 61.

    Главная ловушка источника: без разворота индексов половина статей просто
    не находится, включая 6.1 (исчисление сроков), 93.1 (встречная проверка),
    101.4 и 129.1 — то есть ядро процедурных норм слоя 1.
    """
    assert expand_superscripts('Статья 6<span class="W9" style="">1</span>.') == "Статья 6.1."


def test_superscript_in_reference_is_expanded():
    """«статьи 221¹» иначе станет «статьи 2211» — нормой, которой не существует."""
    assert expand_superscripts('статьи 221<span class="W9">1</span> Кодекса') == "статьи 221.1 Кодекса"


def test_superscript_only_after_digit():
    """W9 бывает и не индексом номера; разворачивается только после цифры."""
    assert expand_superscripts('текст<span class="W9">1</span>') == 'текст<span class="W9">1</span>'


def test_article_with_superscript_number_is_parsed():
    numbers = [a.number for a in articles()]
    assert "6.1" in numbers


# --- иерархия ---------------------------------------------------------

def test_article_number_and_title_split():
    (article,) = [a for a in articles() if a.number == "6.1"]
    assert article.title.startswith("Порядок исчисления сроков")
    assert article.part == 1
    assert article.norm_unit_id == "nk1/st6.1"


def test_clauses_are_numbered():
    (article,) = [a for a in articles() if a.number == "6.1"]
    numbers = [c.number for c in article.clauses]
    assert numbers[0] == "1"
    assert all(n is None or n[0].isdigit() for n in numbers)


def test_clause_norm_unit_id():
    clause = Clause(number="3", text="…")
    assert clause.norm_unit_id("nk1/st88") == "nk1/st88/p3"
    assert Clause(number=None, text="…").norm_unit_id("nk1/st88") == "nk1/st88"


# --- утратившие силу --------------------------------------------------

def test_wholly_repealed_article_is_kept_and_flagged():
    """Спека §10: выбросить их значит оставить старые ссылки в пустоту."""
    repealed = [a for a in articles() if a.is_repealed]
    assert repealed, "статья, утратившая силу, должна остаться в разборе"
    assert repealed[0].number == "25"


@pytest.mark.parametrize("text", [
    "(Утратил силу - Федеральный закон от 27.07.2006 № 137-ФЗ)",
    "(Пункт утратил силу - Федеральный закон от 26.06.2008 № 103-ФЗ)",
    "Абзац. (Утратил силу - Федеральный закон)",
    "(Исключена - Федеральный закон)",
    "(Статья утратила силу - Федеральный закон)",
])
def test_repeal_markers_recognised(text):
    (article,) = parse_articles(
        f'<p class="H">Статья 999. Тест</p><p>{text}</p>', part=1
    )
    assert article.clauses[0].is_repealed is True


def test_active_clause_not_flagged():
    (article,) = parse_articles(
        '<p class="H">Статья 999. Тест</p><p>1. Обычная действующая норма.</p>', part=1
    )
    assert article.clauses[0].is_repealed is False
    assert article.is_repealed is False


# --- номера с дефисом -------------------------------------------------

def test_hyphenated_article_number():
    """«Статья 105.16-1» — отдельная статья, а не 105.16 с заголовком «-1. …»."""
    (article,) = parse_articles(
        '<p class="H">Статья 105.16-1. Общие положения</p><p>1. Текст.</p>', part=1
    )
    assert article.number == "105.16-1"
    assert article.title == "Общие положения"


# --- устойчивость -----------------------------------------------------

def test_empty_document_yields_nothing():
    assert parse_articles("<html><body></body></html>") == []


def test_paragraphs_before_any_article_are_ignored():
    assert parse_articles("<p>преамбула</p>") == []


def test_unnumbered_lead_paragraph_kept_as_clause():
    (article,) = parse_articles(
        '<p class="H">Статья 14. Региональные налоги</p><p>К региональным налогам относятся:</p>',
        part=1,
    )
    assert article.clauses[0].number is None
    assert article.clauses[0].text.startswith("К региональным")
