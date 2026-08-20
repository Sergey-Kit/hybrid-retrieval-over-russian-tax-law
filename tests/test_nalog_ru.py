"""Парсер сервиса ФНС «Решения по жалобам» на срезе реальной страницы."""
from pathlib import Path

import pytest

from p3.sources.nalog_ru import RECORDS_PER_PAGE, list_path, parse_decisions, parse_topic_filter

FIXTURES = Path(__file__).parent / "fixtures"


def decisions():
    return parse_decisions((FIXTURES / "fns_decisions.html").read_text(encoding="utf-8"))


def test_parses_every_record():
    assert len(decisions()) == 3


def test_reads_topic_and_official_article_label():
    """Разметку статьями делает ФНС — это и есть эталон silver set."""
    first = decisions()[0]
    assert first.decision_id == "16172699"
    assert first.article_labels == ("Статья 93 НК РФ",)
    assert first.topic.startswith("Истребование информации")


def test_reads_metadata_fields():
    first = decisions()[0]
    assert first.published == "10 апреля 2025"
    assert "КЧ-2-9/6961@" in first.decided
    assert first.authority == "Центральный аппарат ФНС России"
    assert first.tax == "Нет"


def test_detail_path_built_from_id():
    assert decisions()[0].detail_path == "/rn77/service/complaint_decision/16172699/"


def test_records_without_topic_or_articles_are_skipped():
    html = '<div class="text_block"><div class="row"><div><label>Вид налога:</label></div><div>Нет</div></div></div>'
    assert parse_decisions(html) == []


def test_empty_page_yields_nothing():
    assert parse_decisions("<html><body></body></html>") == []


def test_topic_filter_options_drop_placeholders():
    options = parse_topic_filter((FIXTURES / "fns_topics.html").read_text(encoding="utf-8"))
    labels = [label for _, label in options]
    assert "Все" not in labels and "-" not in labels
    assert any("НК РФ" in label for label in labels)


@pytest.mark.parametrize("page,expected", [(1, "/rn77/service/complaint_decision/1.html"),
                                           (46, "/rn77/service/complaint_decision/46.html")])
def test_list_path(page, expected):
    assert list_path(page) == expected


def test_list_path_rejects_zero():
    with pytest.raises(ValueError):
        list_path(0)


def test_records_per_page_is_documented():
    assert RECORDS_PER_PAGE == 15
