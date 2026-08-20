"""Парсер писем ФНС на срезах реальных страниц (разведка 2026-08-20)."""
from pathlib import Path

import pytest

from p3.sources.fns_letters import (
    RECORDS_PER_PAGE,
    is_administrative,
    letter_path,
    list_path,
    parse_letter,
    parse_letter_ids,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def letter():
    return parse_letter(fixture("fns_letter.html"), "16605143")


# --- поля письма ------------------------------------------------------

def test_reads_official_article_tags():
    """Разметку статьями делает ФНС — это эталон, а не результат разбора текста."""
    assert letter().article_labels == ("Статья 431 НК РФ",)


def test_reads_question_as_query():
    assert letter().question.startswith("О контрольных соотношениях")


def test_reads_metadata_fields():
    """Поля идут как <strong>Метка: </strong>значение, значение — текстовый узел."""
    parsed = letter()
    assert parsed.number == "БС-36-11/1367@"
    assert parsed.letter_date == "24.02.2026"
    assert parsed.published == "25.02.2026"
    assert parsed.category == "Страховые взносы"


def test_text_node_values_survive_traversal():
    """Регрессия: у NavigableString есть строковый .find, возвращающий -1.

    -1 истинно, поэтому проверка «а нет ли дальше <strong>» обрывала обход на
    первом же текстовом узле, и все поля, кроме обёрнутых в теги, терялись.
    """
    assert letter().number is not None


def test_multiple_tags_are_all_captured():
    parsed = parse_letter(fixture("fns_letter_admin.html"), "16642349")
    assert parsed.article_labels == ("Статья 213 НК РФ", "Статья 213.1 НК РФ")


def test_letter_without_question_is_dropped():
    assert parse_letter("<html><body><strong>Номер: </strong>X</body></html>", "1") is None


# --- отсев административных -------------------------------------------

@pytest.mark.parametrize("question", [
    "Об утверждении приказа ФНС России от 15.06.2026 № ЕД-1-11/396@",
    "О направлении новых КС к декларации по НДС 2026",
    "О внесении изменений в приложение к письму ФНС России",
    "О признании утратившим силу письма ФНС России",
])
def test_administrative_questions_detected(question):
    assert is_administrative(question) is True


@pytest.mark.parametrize("question", [
    "О контрольных соотношениях показателей формы расчета по страховым взносам",
    "Об определении налоговой базы в отношении систем инженерно-технического обеспечения",
    "По вопросу порядка предоставления инвестиционного налогового вычета",
])
def test_substantive_questions_kept(question):
    assert is_administrative(question) is False


def test_administrative_flag_on_parsed_letter():
    assert parse_letter(fixture("fns_letter_admin.html"), "16642349").is_administrative is True
    assert letter().is_administrative is False


# --- список -----------------------------------------------------------

def test_parse_letter_ids_dedups_and_keeps_order():
    ids = parse_letter_ids(fixture("fns_letters_list.html"))
    assert ids == list(dict.fromkeys(ids))
    assert all(i.isdigit() for i in ids)
    assert "16642349" in ids


def test_empty_page_yields_no_ids():
    assert parse_letter_ids("<html><body>пусто</body></html>") == []


# --- пути -------------------------------------------------------------

def test_paths():
    assert list_path(3) == "/rn77/about_fts/about_nalog/3.html"
    assert letter_path("16605143") == "/rn77/about_fts/about_nalog/16605143/"
    assert RECORDS_PER_PAGE == 15


def test_list_path_rejects_zero():
    with pytest.raises(ValueError):
        list_path(0)
