"""Граф внутренних ссылок НК: разрешение относительных ссылок и типизация."""
import pytest

from p3.nk.refs import (
    TYPE_DEFINITION,
    TYPE_EXCEPTION,
    TYPE_PLAIN,
    TYPE_PROCEDURE,
    TYPE_SANCTION,
    classify,
    extract_refs,
)


def targets(text, article_id="nk1/st88", clause="3"):
    return {(r.target, r.type) for r in extract_refs(text, article_id=article_id, clause=clause)}


# --- относительные ссылки ---------------------------------------------

def test_clause_of_this_article_resolved():
    """«пунктом 8.1 настоящей статьи» — номера статьи в тексте нет вовсе."""
    assert ("nk1/st88/p8.1", TYPE_EXCEPTION) in targets(
        "если иное не предусмотрено пунктом 8.1 настоящей статьи"
    )


def test_subclause_of_this_clause_resolved():
    assert ("nk1/st31/p1/sp2", TYPE_EXCEPTION) in targets(
        "за исключением случаев, предусмотренных подпунктом 2 настоящего пункта",
        article_id="nk1/st31", clause="1",
    )


def test_relative_subclause_without_clause_context_is_skipped():
    """Без номера пункта «подпункт 2 настоящего пункта» адресовать некуда."""
    assert not [
        r for r in extract_refs("подпунктом 2 настоящего пункта", article_id="nk1/st31", clause=None)
    ]


def test_enumerated_relative_refs():
    found = targets("в случаях, установленных пунктами 1 и 3 настоящей статьи")
    assert ("nk1/st88/p1", TYPE_PROCEDURE) in found
    assert ("nk1/st88/p3", TYPE_PROCEDURE) in found


# --- абсолютные ссылки ------------------------------------------------

def test_absolute_reference_to_another_article():
    assert ("nk1/st93.1", TYPE_PROCEDURE) in targets(
        "в порядке, установленном статьей 93.1 настоящего Кодекса"
    )


def test_reference_into_part_two_keeps_its_part():
    assert ("nk2/st221.1/p2", TYPE_PLAIN) in targets("указанного в пункте 2 статьи 221.1 Кодекса")


def test_self_reference_is_dropped():
    assert not targets("в соответствии со статьей 88 настоящего Кодекса", article_id="nk1/st88")


def test_non_nk_reference_is_dropped():
    """Ссылки на ФЗ и приказы в refs_out не место — граф строится по кодексу."""
    assert not targets("в соответствии с Федеральным законом от 08.08.2001 № 129-ФЗ")


# --- типизация --------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("в порядке, установленном статьей 93.1 настоящего Кодекса", TYPE_PROCEDURE),
    ("если иное не предусмотрено статьей 93.1 настоящего Кодекса", TYPE_EXCEPTION),
    ("за исключением случаев, указанных в статье 93.1 настоящего Кодекса", TYPE_EXCEPTION),
    ("влечет ответственность, предусмотренную статьей 129.1 настоящего Кодекса", TYPE_SANCTION),
    ("в значении, определенном статьей 11 настоящего Кодекса", TYPE_DEFINITION),
])
def test_reference_types(text, expected):
    assert expected in {t for _, t in targets(text)}


def test_specific_marker_beats_generic_connector():
    """«предусмотренный» и «установленный» есть во всех конструкциях сразу.

    По близости они всегда побеждали бы смысловую связку, поэтому тип
    определяется приоритетом, а не расстоянием.
    """
    assert classify("влечет ответственность, предусмотренную ", 40) == TYPE_SANCTION
    assert classify("за исключением случаев, предусмотренных ", 40) == TYPE_EXCEPTION


def test_untyped_reference_is_plain():
    assert classify("Согласно ", 9) == TYPE_PLAIN


def test_duplicates_collapse():
    found = extract_refs(
        "в порядке статьи 93.1 настоящего Кодекса и далее статьи 93.1 настоящего Кодекса",
        article_id="nk1/st88", clause="3",
    )
    assert len([r for r in found if r.target == "nk1/st93.1"]) == 1


def test_raw_text_preserved():
    (reference,) = [
        r for r in extract_refs(
            "в порядке, установленном статьей 93.1 настоящего Кодекса",
            article_id="nk1/st88", clause="3",
        ) if r.target == "nk1/st93.1"
    ]
    assert "93.1" in reference.raw
    assert reference.as_payload()["type"] == TYPE_PROCEDURE
