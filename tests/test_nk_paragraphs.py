"""Абзацы нормы: синтетика на каждое правило + сверка со всеми статьями слоя 1."""
import json
from pathlib import Path

import pytest

from p3.nk import paragraphs as pp


def chunk(cid: str, unit: str, lines: list[str]) -> dict:
    body = "\n".join(lines)
    offsets, pos = [], 0
    for line in lines:
        offsets.append([pos, pos + len(line)])
        pos += len(line) + 1
    return {"chunk_id": cid, "norm_unit_id": unit, "text_bm25": body,
            "para_offsets": json.dumps(offsets)}


def test_editorial_notes_are_not_counted():
    ps = pp.structure([chunk("nk1/st9/p5@r1#0", "nk1/st9/p5", [
        "Первый абзац.", "Второй абзац.",
        "(Абзац дополнительно включен - Федеральный закон от 01.02.2020 № 1-ФЗ)",
        "Третий абзац.",
        "(Пункт в редакции Федерального закона от 03.04.2021 № 2-ФЗ)"])])
    labels = [p.label for p in ps]
    assert labels == ["абзац первый", "абзац второй", "пометка редакции", "абзац третий",
                      "пометка редакции"]


def test_several_notes_on_one_line_are_one_note():
    ps = pp.structure([chunk("nk1/st9/p4@r1#0", "nk1/st9/p4", [
        "Первый.",
        "(Дополнение пунктом - Федеральный закон от 01.02.2020 № 1-ФЗ) "
        "(В редакции Федерального закона от 03.04.2021 № 2-ФЗ)",
        "Второй."])])
    assert [p.label for p in ps] == ["абзац первый", "пометка редакции", "абзац второй"]


def test_parenthesised_text_without_a_law_is_a_paragraph():
    ps = pp.structure([chunk("c#0", "nk1/st9/p4", ["(в случае, указанном выше)"])])
    assert ps[0].kind == "абзац"


def test_numbering_continues_across_split_chunks_in_any_order():
    ps = pp.structure([
        chunk("nk1/st9/p2@r1#1", "nk1/st9/p2", ["Третий.", "Четвёртый."]),
        chunk("nk1/st9/p2@r1#0", "nk1/st9/p2", ["Первый.", "Второй."]),
    ])
    assert [(p.text, p.number) for p in ps] == [
        ("Первый.", 1), ("Второй.", 2), ("Третий.", 3), ("Четвёртый.", 4)]


def test_subclause_chunk_intro_is_not_a_paragraph():
    ps = pp.structure([chunk("nk1/st9/p1/sp2@r1#0", "nk1/st9/p1/sp2", [
        "Налогоплательщики обязаны:", "2) встать на учет;", "продолжение подпункта."])])
    assert [p.label for p in ps] == ["вводная часть пункта", "подпункт 2",
                                     "подпункт 2, абзац второй"]


def test_locate_is_exact_and_unique():
    ps = pp.structure([chunk("c#0", "nk1/st9/p1", ["Срок — десять дней.", "Форма — по ТКС."])])
    assert pp.locate(ps, "форма —  по ткс").label == "абзац второй"
    assert pp.locate(ps, "срок двадцать дней") is None     # похожее — не привязываем


# --- все статьи слоя 1 реального корпуса ------------------------------------

CHUNKS = Path(__file__).resolve().parents[1] / "data" / "chunks"
CORPORA = [CHUNKS / "nk-full.jsonl", CHUNKS / "nk1-temporal.jsonl"]


def _layer1_versions(path: Path) -> dict[str, list[dict]]:
    from p3.silver.build import LAYER1_ARTICLES
    by: dict[str, list[dict]] = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if "/".join(r["norm_unit_id"].split("/")[:2]) in LAYER1_ARTICLES:
            by.setdefault(r["version_id"], []).append(r)
    return by


@pytest.mark.parametrize("path", CORPORA, ids=lambda p: p.name)
def test_every_layer1_line_resolves(path):
    if not path.exists():
        pytest.skip("корпус не собран: чанки производные, в репозитории их нет")
    versions = _layer1_versions(path)
    assert versions
    for vid, chunks in versions.items():
        ps = pp.structure(chunks)
        lines = sum(len(pp._offsets(c)) for c in chunks)
        assert len(ps) == lines, vid                    # у каждой строки есть метка
        nums = [p.number for p in ps if p.kind == "абзац" and "подпункт" not in p.label]
        assert nums == list(range(1, len(nums) + 1)), vid   # абзацы без пропусков
        for p in ps:                                    # цитата находит свой абзац
            if p.kind == "абзац" and len(set(x.text for x in ps if x.text == p.text)) == 1 \
                    and sum(x.text == p.text for x in ps) == 1:
                assert pp.locate(ps, p.text) == p, (vid, p.label)


def test_real_reference_second_paragraph_of_93_1_p5():
    """«…абзацем вторым пункта 5 статьи 93.1» — требования по п. 2, срок 10 дней."""
    path = CORPORA[0]
    if not path.exists():
        pytest.skip("корпус не собран")
    chunks = [json.loads(l) for l in path.open(encoding="utf-8")
              if json.loads(l)["norm_unit_id"] == "nk1/st93.1/p5"]
    second = pp.paragraph(pp.structure(chunks), 2)
    assert "пунктами 2" in second.text and "десяти дней" in second.text
