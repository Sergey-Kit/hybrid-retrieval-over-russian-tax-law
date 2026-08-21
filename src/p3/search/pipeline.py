"""Сборка конфигураций поиска для сравнения по ТЗ §7."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from p3.search.bm25 import Bm25fIndex, build_index
from p3.search.dense import DenseEncoder, DenseIndex
from p3.search.hybrid import ChunkStore, Hit, Reranker, one_hop_expansion, reciprocal_rank_fusion

TOP_CANDIDATES = 50
RERANK_INPUT = 30
RERANK_OUTPUT = 8


@dataclass
class Retrieval:
    """Индексы и хранилище, общие для всех конфигураций."""

    store: ChunkStore
    bm25: Bm25fIndex
    dense: DenseIndex | None = None
    encoder: DenseEncoder | None = None
    reranker: Reranker | None = None
    #: Кэш векторов запросов: конфигурации dense/hybrid/hybrid+rerank прогоняют
    #: один и тот же набор запросов, и без кэша каждый кодируется трижды.
    _query_vectors: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        chunks_dir: Path,
        *,
        encoder_name: str | None = None,
        field_weights: dict[str, float] | None = None,
    ) -> "Retrieval":
        chunks = [json.loads(line) for line in (chunks_dir / "nk1.jsonl").open(encoding="utf-8")]
        store = ChunkStore.from_chunks(chunks)
        bm25 = build_index(chunks, field_weights=field_weights)

        dense = encoder = None
        if encoder_name:
            path = chunks_dir / f"dense-{encoder_name}.npz"
            if path.exists():
                dense = DenseIndex.load(path)
                encoder = DenseEncoder(name=encoder_name)
        return cls(store=store, bm25=bm25, dense=dense, encoder=encoder)

    # --- отдельные каналы -------------------------------------------------

    def search_bm25(self, query: str, top_k: int = TOP_CANDIDATES) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=top_k)

    def query_vector(self, query: str):
        cached = self._query_vectors.get(query)
        if cached is None:
            cached = self.encoder.encode_queries([query])[0]
            self._query_vectors[query] = cached
        return cached

    def search_dense(self, query: str, top_k: int = TOP_CANDIDATES) -> list[tuple[str, float]]:
        if self.dense is None or self.encoder is None:
            return []
        return self.dense.search(self.query_vector(query), top_k=top_k)

    # --- конфигурации -----------------------------------------------------

    def run(
        self,
        query: str,
        *,
        mode: str,
        expand: bool = False,
        top_k: int = TOP_CANDIDATES,
    ) -> list[Hit]:
        if mode == "bm25":
            hits = [Hit(d, s, sources=("bm25",)) for d, s in self.search_bm25(query, top_k)]
        elif mode == "dense":
            hits = [Hit(d, s, sources=("dense",)) for d, s in self.search_dense(query, top_k)]
        elif mode in ("hybrid", "hybrid+rerank"):
            hits = reciprocal_rank_fusion({
                "bm25": self.search_bm25(query, top_k),
                "dense": self.search_dense(query, top_k),
            })
        else:
            raise ValueError(f"неизвестная конфигурация {mode!r}")

        if expand:
            hits = one_hop_expansion(hits, self.store)

        if mode == "hybrid+rerank":
            if self.reranker is None:
                self.reranker = Reranker()
            hits = self.reranker.rerank(query, hits[:RERANK_INPUT], self.store, top_k=RERANK_OUTPUT)
        return hits


def to_articles(hits: list[Hit], store: ChunkStore) -> list[str]:
    """Ранжирование чанков → ранжирование статей, порядок сохраняется."""
    seen: dict[str, None] = {}
    for hit in hits:
        article = store.article_of(hit.chunk_id)
        if article:
            seen.setdefault(article, None)
    return list(seen)


def to_norms(hits: list[Hit], store: ChunkStore) -> list[str]:
    seen: dict[str, None] = {}
    for hit in hits:
        chunk = store.by_chunk_id.get(hit.chunk_id)
        if chunk:
            seen.setdefault(chunk["norm_unit_id"], None)
    return list(seen)


def indexed_articles(store: ChunkStore) -> set[str]:
    return {a for a in (store.article_of(c) for c in store.by_chunk_id) if a}


def expansion_contribution(
    retrieval: "Retrieval",
    queries,
    *,
    mode: str = "hybrid",
    level: str = "article",
    top_k: int = 20,
) -> dict:
    """Доля запросов, где эталон достался **только** через раскрытие ссылки.

    Требование ТЗ §7 отдельной строкой. Если доля заметна, это самостоятельный
    результат: детерминированный обход графа закрывает класс ошибок, который
    эмбеддер не закроет никогда — основание не исполнять требование выражено
    отсылкой, а не словами.
    """
    only_via_expansion = 0
    helped_at_all = 0
    for query in queries:
        gold = query.gold_articles if level == "article" else query.gold_norms
        plain = retrieval.run(query.query, mode=mode, expand=False)
        expanded = retrieval.run(query.query, mode=mode, expand=True)

        project = to_articles if level == "article" else to_norms
        before = set(project(plain, retrieval.store)[:top_k])
        after = set(project(expanded, retrieval.store)[:top_k])

        if gold & after and not (gold & before):
            only_via_expansion += 1
        if (gold & after) - (gold & before):
            helped_at_all += 1

    total = len(queries) or 1
    return {
        "queries": len(queries),
        "gold_reached_only_via_expansion": only_via_expansion,
        "share_only_via_expansion": round(only_via_expansion / total, 4),
        "queries_where_expansion_added_gold": helped_at_all,
    }
