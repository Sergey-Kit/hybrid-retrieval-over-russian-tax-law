"""Синтетический набор слоя 1 и его самопроверка против корпуса."""
from pathlib import Path

import pytest

from p3.nk.parse import Article, Clause
from p3.silver.build import QUERY_TYPE_LEXICAL, QUERY_TYPE_NATURAL, split
from p3.silver.layer1 import (
    QUESTIONS,
    SOURCE_LAYER1,
    Layer1Question,
    build_layer1_pairs,
    verify,
)

FIXTURES = Path(__file__).parent / "fixtures"


def corpus_for(questions=QUESTIONS) -> list[Article]:
    """Минимальный корпус, в котором сходятся все опорные подстроки."""
    by_article: dict[str, Article] = {}
    for question in questions:
        article = by_article.setdefault(
            question.article, Article(number=question.article, title="Тест", part=1)
        )
        if not any(c.number == question.clause for c in article.clauses):
            article.clauses.append(Clause(number=question.clause, text=question.anchor))
    return list(by_article.values())


# --- самопроверка -----------------------------------------------------

def test_verify_passes_on_matching_corpus():
    assert verify(corpus_for()) == []


def test_verify_reports_missing_article():
    problems = verify([])
    assert len(problems) == len(QUESTIONS)
    assert all("не найдена" in p for p in problems)


def test_verify_reports_wrong_anchor():
    """Если норму переформулировали, набор обязан упасть, а не молчать."""
    corpus = corpus_for()
    corpus[0].clauses[0].text = "совершенно другой текст"
    problems = verify(corpus)
    assert len(problems) == 1
    assert "опорная подстрока не найдена" in problems[0]


def test_build_refuses_to_run_on_mismatched_corpus():
    with pytest.raises(ValueError, match="не сходится"):
        build_layer1_pairs([])


# --- состав набора ----------------------------------------------------

def test_two_phrasings_per_question():
    pairs = build_layer1_pairs(corpus_for())
    assert len(pairs) == 2 * len(QUESTIONS)
    lexical = [p for p in pairs if p.query_type == QUERY_TYPE_LEXICAL]
    natural = [p for p in pairs if p.query_type == QUERY_TYPE_NATURAL]
    assert len(lexical) == len(natural) == len(QUESTIONS)


def test_both_phrasings_share_the_gold():
    """Иначе сравнение «якорь против живого языка» ничего не измеряет."""
    pairs = {p.query_id: p for p in build_layer1_pairs(corpus_for())}
    for index in range(len(QUESTIONS)):
        demand = pairs[f"l1-{index:02d}-demand"]
        natural = pairs[f"l1-{index:02d}-natural"]
        assert demand.gold_norm_ids == natural.gold_norm_ids
        assert demand.cluster_id == natural.cluster_id


def test_both_phrasings_land_in_one_split():
    pairs = build_layer1_pairs(corpus_for())
    splits = split(pairs, seed=42)
    where = {p.query_id: name for name, items in splits.items() for p in items}
    for index in range(len(QUESTIONS)):
        assert where[f"l1-{index:02d}-demand"] == where[f"l1-{index:02d}-natural"]


def test_natural_phrasing_has_no_article_number():
    """Живой вопрос обязан быть без якоря — иначе он не проверяет dense-канал."""
    import re
    for question in QUESTIONS:
        assert not re.search(r"стать[ьие]|ст\.\s*\d|пункт", question.natural, re.IGNORECASE), (
            f"в живой формулировке остался якорь: {question.natural!r}"
        )


def test_demand_phrasing_has_an_anchor():
    for question in QUESTIONS:
        assert "стать" in question.demand_excerpt.lower()


def test_source_is_marked_synthetic():
    assert {p.source for p in build_layer1_pairs(corpus_for())} == {SOURCE_LAYER1}


def test_deadline_questions_pull_in_article_6_1():
    """Любой срок исчисляется по ст. 6.1 — она должна быть в эталоне."""
    pairs = build_layer1_pairs(corpus_for())
    deadline = [p for p in pairs if p.source_id.startswith("deadline/nk1/st93")]
    assert deadline
    assert all("nk1/st6.1" in p.gold_article_ids for p in deadline)


def test_all_five_question_types_present():
    assert len({q.question_type for q in QUESTIONS}) == 5


def test_covers_the_layer1_articles():
    from p3.silver.build import LAYER1_ARTICLES
    covered = {q.article_id for q in QUESTIONS}
    assert covered == LAYER1_ARTICLES


# --- строение записи --------------------------------------------------

def test_norm_unit_id_shape():
    question = Layer1Question("t", "88", "3", "x", "d", "n")
    assert question.norm_unit_id == "nk1/st88/p3"
    assert question.article_id == "nk1/st88"


def test_article_level_question_has_no_clause_suffix():
    question = Layer1Question("t", "88", None, "x", "d", "n")
    assert question.norm_unit_id == "nk1/st88"
