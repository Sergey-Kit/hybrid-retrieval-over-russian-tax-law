"""RRF-слияние, one-hop раскрытие ссылок, приведение к статьям."""
from p3.search.evaluate import (
    EvalQuery, evaluate, load_queries, ndcg_at_k, reachable, recall_at_k, reciprocal_rank,
)
from p3.search.hybrid import (
    EXPANDED_REF_TYPES, RRF_K, ChunkStore, Hit, one_hop_expansion, reciprocal_rank_fusion,
)
from p3.search.pipeline import to_articles, to_norms


def chunk(chunk_id, norm_unit_id, refs=()):
    return {
        "chunk_id": chunk_id,
        "norm_unit_id": norm_unit_id,
        "text_dense": f"текст {norm_unit_id}",
        "text_bm25": f"текст {norm_unit_id}",
        "refs_out": [{"target": t, "type": ty, "raw": ""} for t, ty in refs],
    }


# --- RRF ---------------------------------------------------------------

def test_fusion_ranks_by_rank_not_score():
    """Шкалы dense и BM25 несопоставимы; сложение сырых оценок дало бы
    результат, зависящий от нормировки, а не от релевантности."""
    merged = reciprocal_rank_fusion({
        "bm25": [("a", 100.0), ("b", 99.0)],
        "dense": [("b", 0.9), ("a", 0.1)],
    })
    assert {h.chunk_id for h in merged} == {"a", "b"}
    assert abs(merged[0].score - merged[1].score) < 1e-9


def test_document_in_both_rankings_wins():
    merged = reciprocal_rank_fusion({
        "bm25": [("both", 1.0), ("only_bm25", 0.9)],
        "dense": [("both", 1.0), ("only_dense", 0.9)],
    })
    assert merged[0].chunk_id == "both"


def test_sources_recorded():
    merged = reciprocal_rank_fusion({"bm25": [("a", 1.0)], "dense": [("a", 1.0), ("b", 1.0)]})
    by_id = {h.chunk_id: h for h in merged}
    assert by_id["a"].sources == ("bm25", "dense")
    assert by_id["b"].sources == ("dense",)


def test_rrf_constant_matches_the_spec():
    assert RRF_K == 60


def test_empty_rankings():
    assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []


# --- one-hop -----------------------------------------------------------

def store_with_refs():
    return ChunkStore.from_chunks([
        chunk("c1", "nk1/st88/p3", refs=[("nk1/st93.1", "procedure"), ("nk1/st1", "amendment")]),
        chunk("c2", "nk1/st93.1"),
        chunk("c3", "nk1/st1"),
    ])


def test_expansion_pulls_referenced_chunk():
    expanded = one_hop_expansion([Hit("c1", 1.0)], store_with_refs())
    assert [h.chunk_id for h in expanded] == ["c1", "c2"]
    assert expanded[1].expanded is True


def test_expansion_skips_untyped_kinds():
    """Раскрываются только procedure, exception и definition."""
    expanded = one_hop_expansion([Hit("c1", 1.0)], store_with_refs())
    assert "c3" not in {h.chunk_id for h in expanded}
    assert set(EXPANDED_REF_TYPES) == {"procedure", "exception", "definition"}


def test_expansion_does_not_duplicate():
    expanded = one_hop_expansion([Hit("c1", 1.0), Hit("c2", 0.5)], store_with_refs())
    assert len({h.chunk_id for h in expanded}) == len(expanded)


def test_expanded_hits_go_last():
    expanded = one_hop_expansion([Hit("c1", 1.0)], store_with_refs())
    assert all(not h.expanded for h in expanded[:1])
    assert expanded[-1].expanded


# --- приведение к уровням ----------------------------------------------

def test_chunks_map_to_articles_preserving_order():
    store = ChunkStore.from_chunks([
        chunk("c1", "nk1/st88/p3"), chunk("c2", "nk1/st88/p7"), chunk("c3", "nk1/st93/p1"),
    ])
    hits = [Hit("c1", 1.0), Hit("c2", 0.9), Hit("c3", 0.8)]
    assert to_articles(hits, store) == ["nk1/st88", "nk1/st93"]
    assert to_norms(hits, store) == ["nk1/st88/p3", "nk1/st88/p7", "nk1/st93/p1"]


# --- метрики -----------------------------------------------------------

def test_recall_counts_share_of_gold_found():
    assert recall_at_k(["a", "b"], frozenset({"a", "c"}), 20) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], frozenset({"a"})) == 1 / 3
    assert reciprocal_rank(["x"], frozenset({"a"})) == 0.0


def test_ndcg_rewards_higher_positions():
    high = ndcg_at_k(["a", "x", "y"], frozenset({"a"}), 10)
    low = ndcg_at_k(["x", "y", "a"], frozenset({"a"}), 10)
    assert high > low
    assert high == 1.0


def test_reachable_filters_queries_whose_gold_is_absent():
    """Индекс содержит только часть первую; общая цифра по всему silver set
    описывала бы полноту индекса, а не качество поиска."""
    queries = [
        EvalQuery("q1", "…", frozenset({"nk1/st88"}), frozenset(), "natural_language", "s"),
        EvalQuery("q2", "…", frozenset({"nk2/st220"}), frozenset(), "natural_language", "s"),
    ]
    assert [q.query_id for q in reachable(queries, {"nk1/st88"})] == ["q1"]


def test_evaluate_on_perfect_run():
    queries = [EvalQuery("q1", "…", frozenset({"nk1/st88"}), frozenset(), "t", "s")]
    metrics = evaluate(queries, lambda q: ["nk1/st88"])
    assert metrics.as_dict()["recall@20"] == 1.0
    assert metrics.as_dict()["MRR"] == 1.0


def test_evaluate_on_empty_run():
    queries = [EvalQuery("q1", "…", frozenset({"nk1/st88"}), frozenset(), "t", "s")]
    assert evaluate(queries, lambda q: []).as_dict()["recall@20"] == 0.0


def test_load_queries_from_rows():
    (query,) = load_queries([{
        "query_id": "x", "query": "текст", "gold_article_ids": ["nk1/st88"],
        "gold_norm_ids": ["nk1/st88/p3"], "query_type": "natural_language", "source": "s",
    }])
    assert query.gold_articles == frozenset({"nk1/st88"})
    assert query.gold_norms == frozenset({"nk1/st88/p3"})


def test_latency_is_measured():
    """ТЗ §7 требует колонку латентности."""
    queries = [EvalQuery(f"q{i}", "…", frozenset({"nk1/st88"}), frozenset(), "t", "s")
               for i in range(5)]
    report = evaluate(queries, lambda q: ["nk1/st88"]).as_dict()
    assert report["latency_ms_median"] >= 0.0
    assert report["latency_ms_p90"] >= report["latency_ms_median"]
