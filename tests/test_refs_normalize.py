"""Golden-набор для нормализатора ссылок.

Случаи взяты из реальных формулировок писем Минфина/ФНС и решений по жалобам.
"""
import pytest

from p3.refs.normalize import Citation, find_citations, nk_part, norm_unit_ids

# (текст, ожидаемые norm_unit_id)
GOLDEN: list[tuple[str, list[str]]] = [
    # Базовые формы
    ("п. 3 ст. 88 НК РФ", ["nk1/st88/p3"]),
    ("пунктом 3 статьи 88 Налогового кодекса", ["nk1/st88/p3"]),
    ("статья 88 Налогового кодекса Российской Федерации", ["nk1/st88"]),
    ("в порядке, установленном статьей 93.1 Кодекса", ["nk1/st93.1"]),
    # Полная иерархия
    ("подпунктом 2 пункта 1 статьи 31 Налогового кодекса", ["nk1/st31/p1/sp2"]),
    ("п.п. 1 п. 1 ст. 31 НК РФ", ["nk1/st31/p1/sp1"]),
    ("абзацем вторым пункта 3 статьи 88 Кодекса", ["nk1/st88/p3/a2"]),
    # Перечисления и диапазоны
    ("статьи 88, 93 и 93.1 НК РФ", ["nk1/st88", "nk1/st93", "nk1/st93.1"]),
    ("статьями 100 - 101 Кодекса", ["nk1/st100", "nk1/st101"]),
    ("пунктах 3 и 4 статьи 88 НК РФ", ["nk1/st88/p3", "nk1/st88/p4"]),
    ("ст. 126 и ст. 129.1 НК РФ", ["nk1/st126", "nk1/st129.1"]),
    # Часть вторая определяется номером статьи
    ("статьей 143 НК РФ", ["nk2/st143"]),
    ("статьи 346.11 НК РФ", ["nk2/st346.11"]),
    ("статьи 142.8 НК РФ", ["nk1/st142.8"]),
    # Другие кодексы не должны попадать в НК
    ("частью 1 статьи 19.4 КоАП РФ", ["koap/st19.4/p1"]),
    ("нарушение п. 1 ст. 126 НК РФ и части 1 статьи 19.4 КоАП РФ",
     ["nk1/st126/p1", "koap/st19.4/p1"]),
    ("статьи 15 Гражданского кодекса", ["gk/st15"]),
    # Не кодексы вовсе
    ("статьей 5 Федерального закона от 27.07.2006 N 149-ФЗ", ["other/st5"]),
    ("в силу статьи 57 Конституции Российской Федерации", ["other/st57"]),
]


@pytest.mark.parametrize("text,expected", GOLDEN, ids=[t for t, _ in GOLDEN])
def test_golden_citations(text, expected):
    assert norm_unit_ids(text) == expected


def test_no_citation_in_plain_text():
    assert find_citations("Общество представило документы 12.03.2021 в полном объёме") == []


def test_date_is_not_parsed_as_article():
    assert norm_unit_ids("письмо от 12.03.2021 N 03-02-07/1") == []


@pytest.mark.parametrize(
    "article,part",
    [("1", 1), ("88", 1), ("142.8", 1), ("143", 2), ("346.11", 2), ("419", 2)],
)
def test_nk_part_boundary(article, part):
    assert nk_part(article) == part


def test_implicit_code_is_flagged():
    """Ссылка без названия кодекса засчитывается, но помечается — долю надо измерять."""
    (citation,) = find_citations("ст. 88")
    assert citation.code == "nk"
    assert citation.code_explicit is False


def test_explicit_code_is_flagged():
    (citation,) = find_citations("ст. 88 НК РФ")
    assert citation.code_explicit is True


def test_default_code_can_be_disabled():
    assert find_citations("ст. 88", default_code=None) == []


def test_raw_text_is_preserved_for_later_remapping():
    """Веха 2 должна уметь переразметить, не перекачивая корпус."""
    (citation,) = find_citations("в соответствии с пунктом 3 статьи 88 Кодекса")
    assert citation.raw_text == "пунктом 3 статьи 88"


def test_article_id_drops_the_point():
    (citation,) = find_citations("подпунктом 2 пункта 1 статьи 31 НК РФ")
    assert citation.norm_unit_id == "nk1/st31/p1/sp2"
    assert citation.article_id == "nk1/st31"


def test_span_points_at_the_citation():
    text = "в соответствии с пунктом 3 статьи 88 Кодекса налоговый орган"
    (citation,) = find_citations(text)
    start, end = citation.span
    assert text[start:end] == "пунктом 3 статьи 88"


def test_ordering_follows_the_text():
    ids = norm_unit_ids("сначала ст. 93.1 НК РФ, затем ст. 88 НК РФ")
    assert ids == ["nk1/st93.1", "nk1/st88"]


def test_duplicates_collapse():
    assert norm_unit_ids("ст. 88 НК РФ и ещё раз ст. 88 НК РФ") == ["nk1/st88"]


def test_citation_is_hashable_for_set_operations():
    """Silver set сравнивает множества норм — Citation должен ложиться в set."""
    assert len({Citation(raw_text="x", code="nk", article="88", part=1)}) == 1


def test_id_format_matches_chunking_spec():
    """Формат обязан совпадать с 02-спекой §5, иначе метрики тихо занизятся.

    P4 забирает silver set импортом по этому ключу, а ETL строит по нему же
    refs_out. Расхождение не упадёт нигде — пересечения просто станут пустыми.
    """
    (citation,) = find_citations("подпунктом 2 пункта 1 статьи 31 НК РФ")
    assert citation.norm_unit_id == "nk1/st31/p1/sp2"
    assert citation.article_id == "nk1/st31"
    # Примеры идентификаторов, выписанные в 02-спеке дословно.
    assert norm_unit_ids("п. 3 ст. 88 НК РФ") == ["nk1/st88/p3"]
    assert norm_unit_ids("статьи 93.1 НК РФ") == ["nk1/st93.1"]
    assert norm_unit_ids("статьи 129.1 НК РФ") == ["nk1/st129.1"]
