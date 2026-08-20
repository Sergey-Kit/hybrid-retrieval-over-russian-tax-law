"""Парсеры ИПС на срезах реальных ответов (tests/fixtures/, разведка 2026-08-20)."""
from pathlib import Path

import pytest

from p3.config import PRAVO
from p3.fetch import PoliteFetcher
from p3.raw_store import RawStore
from p3.sources.ips import (
    NK_PART_1,
    NK_PART_2,
    Edition,
    EditionTextAbsent,
    IpsClient,
    is_text_absent,
    parse_editions,
    parse_search_results,
    search_attrs,
    strip_html,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --- редакции ---------------------------------------------------------

def test_parse_editions_reads_rdk_date_and_act():
    editions = parse_editions(fixture("ips_card_nk1.html"))

    assert [e.rdk for e in editions] == [0, 1, 2, 3, 4, 5, 240, 241, 242]
    assert editions[0].is_initial and editions[0].date is None
    assert editions[1] == Edition(
        rdk=1, label="1 - от 09.07.1999 № 154-ФЗ (изм.)", date="09.07.1999", amending_act="154-ФЗ"
    )
    assert editions[-1].year == 2024


def test_option_value_is_rdk_comma_nd_not_a_bare_number():
    """value='240,102054722' — rdk и nd в одном поле; берём только rdk."""
    editions = parse_editions("<select><option value='240,102054722'>240 - от 31.07.2025</option></select>")
    assert [e.rdk for e in editions] == [240]


def test_options_inside_script_are_ignored():
    """В скриптах встречается разметка, которая иначе станет фиктивной редакцией."""
    assert all(e.rdk != 999 for e in parse_editions(fixture("ips_card_nk1.html")))


def test_duplicate_selects_collapse_by_rdk():
    html = "<select><option value='7,1'>7 - от 01.02.2003</option></select>" * 3
    assert len(parse_editions(html)) == 1


def test_edition_without_act_number_still_parses():
    (edition,) = parse_editions("<select><option value='242,1'>242 - от 08.08.2024 (изм.)</option></select>")
    assert edition.date == "08.08.2024"
    assert edition.amending_act is None


def test_empty_card_yields_no_editions():
    assert parse_editions("<html><body>ничего</body></html>") == []


# --- заглушка вместо текста -------------------------------------------

def test_text_absent_is_detected():
    assert is_text_absent(fixture("ips_text_absent.html")) is True


def test_present_text_is_not_flagged():
    assert is_text_absent("<html><body>Статья 1. Законодательство</body></html>") is False


def test_client_raises_on_absent_text(tmp_path):
    """Заглушка приходит с кодом 200 — молча принять её нельзя."""
    store = RawStore(tmp_path)
    client = IpsClient(PoliteFetcher(store, offline=True))
    url = PRAVO.base_url + client.edition_path(NK_PART_1.nd, 100)
    store.put(url, fixture("ips_text_absent.html").encode("cp1251"), source=PRAVO.name, encoding="cp1251")

    with pytest.raises(EditionTextAbsent, match="rdk=100"):
        client.edition_html(NK_PART_1.nd, 100)


# --- поиск ------------------------------------------------------------

def test_parse_search_results_dedups_by_nd():
    hits = parse_search_results(fixture("ips_search_list.html"))
    assert len(hits) == len({h.nd for h in hits})
    assert hits[0].nd == "610824102"
    assert "293-ФЗ" in hits[0].title


def test_search_attrs_include_every_empty_field():
    """ИПС не отвечает, если хоть одно поле формы отсутствует в запросе."""
    attrs = search_attrs(title="Налоговый кодекс")
    for field in ("a0", "a1", "a3", "a4", "a6", "a8", "a15", "a16", "a17", "a23", "bpas", "sort"):
        assert field in attrs
    assert attrs["a1"] == "Налоговый кодекс"
    assert attrs["bpas"] == "cd00000"


# --- URL и утилиты ----------------------------------------------------

def test_edition_path_carries_rdk_and_page():
    client = IpsClient(PoliteFetcher(RawStore(Path("/tmp")), offline=True))
    assert client.edition_path("102054722", 248, page=2) == (
        "/proxy/ips/?doc_itself=&nd=102054722&page=2&rdk=248"
    )


def test_known_document_ids():
    assert NK_PART_1.nd == "102054722"
    assert NK_PART_2.nd == "102067058"


def test_strip_html_drops_scripts_and_collapses_space():
    assert strip_html("<p>Статья  1.</p><script>var x=1</script>") == "Статья 1."
