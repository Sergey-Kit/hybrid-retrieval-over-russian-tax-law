import json

import pytest
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


def test_expansion_contribution_counts_gold_reached_only_by_expansion():
    """Требование ТЗ §7: вклад one-hop измеряется отдельной цифрой."""
    from p3.search.bm25 import build_index
    from p3.search.pipeline import Retrieval, expansion_contribution

    chunks = [
        chunk("c1", "nk1/st88/p3", refs=[("nk1/st93.1", "exception")]),
        chunk("c2", "nk1/st93.1"),
    ]
    chunks[0]["text_bm25"] = "камеральная проверка пояснения"
    chunks[0]["title_bm25"] = ""
    chunks[1]["text_bm25"] = "совершенно иной предмет регулирования"
    chunks[1]["title_bm25"] = ""
    store = ChunkStore.from_chunks(chunks)
    retrieval = Retrieval(store=store, bm25=build_index(chunks))

    queries = [EvalQuery("q", "камеральная проверка пояснения",
                         frozenset({"nk1/st93.1"}), frozenset(), "t", "s")]
    report = expansion_contribution(retrieval, queries, mode="bm25")
    assert report["gold_reached_only_via_expansion"] == 1
    assert report["share_only_via_expansion"] == 1.0


def test_mismatched_vectors_are_rejected(tmp_path):
    """Раньше при отсутствии файла подставлялись векторы другого корпуса, и
    dense-канал молча возвращал только его часть."""
    import numpy as np
    from p3.search.dense import DenseIndex
    from p3.search.pipeline import Retrieval

    chunks_dir = tmp_path
    (chunks_dir / "small.jsonl").write_text(
        "\n".join(json.dumps(chunk(f"c{i}", f"nk1/st{i}/p1")) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    DenseIndex(doc_ids=["c0"], vectors=np.zeros((1, 4), dtype=np.float32)).save(
        chunks_dir / "dense-small-user-bge-m3.npz"
    )
    with pytest.raises(ValueError, match="не соответствуют корпусу"):
        Retrieval.load(chunks_dir, encoder_name="user-bge-m3", corpus="small.jsonl")


# --- уровне-зависимое сопоставление эталона ----------------------------

def test_article_level_gold_matched_by_any_of_its_clauses():
    """Эталон вида ['nk1/st93/p3', 'nk1/st6.1'] иначе не может совпасть
    наполовину никогда: чанка nk1/st6.1 не существует, есть только пункты."""
    gold = frozenset({"nk1/st93/p3", "nk1/st6.1"})
    assert recall_at_k(["nk1/st93/p3", "nk1/st6.1/p6"], gold, 20) == 1.0


def test_prefix_match_does_not_fire_on_similar_numbers():
    """st93 не должна совпадать с st931."""
    assert recall_at_k(["nk1/st931/p1"], frozenset({"nk1/st93"}), 20) == 0.0


def test_clause_gold_not_satisfied_by_the_article():
    """Обратное направление не работает: пункт конкретнее статьи."""
    assert recall_at_k(["nk1/st93"], frozenset({"nk1/st93/p3"}), 20) == 0.0


def test_hit_and_rr_use_the_same_matching():
    from p3.search.evaluate import hit_at_k
    gold = frozenset({"nk1/st6.1"})
    assert hit_at_k(["nk1/st6.1/p6"], gold, 20) == 1.0
    assert reciprocal_rank(["x", "nk1/st6.1/p6"], gold) == 0.5


# --- потолок recall ----------------------------------------------------

def test_ceiling_reflects_gold_absent_from_the_index():
    """Фильтр достижимости требует хоть одну эталонную статью, а не все:
    у запроса с тремя эталонами при одном проиндексированном recall не может
    превысить 1/3, и без этой цифры результат выглядит хуже, чем есть."""
    queries = [EvalQuery("q", "…", frozenset({"nk1/st88", "nk2/st220", "nk2/st346"}),
                         frozenset(), "t", "s")]
    report = evaluate(queries, lambda q: ["nk1/st88"], indexed={"nk1/st88"}).as_dict()
    assert report["recall_ceiling"] == pytest.approx(1 / 3, abs=1e-3)
    assert report["recall@20"] == pytest.approx(1 / 3, abs=1e-3)
    assert report["recall_share_of_achievable"] == 1.0


def test_ceiling_is_one_when_all_gold_is_indexed():
    queries = [EvalQuery("q", "…", frozenset({"nk1/st88"}), frozenset(), "t", "s")]
    assert evaluate(queries, lambda q: [], indexed={"nk1/st88"}).as_dict()["recall_ceiling"] == 1.0


def test_ceiling_defaults_to_one_without_index_information():
    queries = [EvalQuery("q", "…", frozenset({"nk1/st88"}), frozenset(), "t", "s")]
    assert evaluate(queries, lambda q: []).as_dict()["recall_ceiling"] == 1.0


def test_ndcg_never_exceeds_one_with_level_aware_matching():
    """Эталон уровня статьи удовлетворяется любым её пунктом; если в выдаче
    их несколько, все набирали вклад при знаменателе на один элемент.
    На наборе слоя 1 это дало nDCG@10 = 1.13."""
    gold = frozenset({"nk1/st6.1"})
    assert ndcg_at_k(["nk1/st6.1/p1", "nk1/st6.1/p2", "nk1/st6.1/p6"], gold, 10) == 1.0


def test_ndcg_credits_each_gold_item_once_at_its_best_rank():
    gold = frozenset({"nk1/st1", "nk1/st2"})
    both_first = ndcg_at_k(["nk1/st1/p1", "nk1/st2/p1"], gold, 10)
    one_late = ndcg_at_k(["nk1/st1/p1", "x", "y", "nk1/st2/p1"], gold, 10)
    assert both_first == 1.0
    assert one_late < both_first
