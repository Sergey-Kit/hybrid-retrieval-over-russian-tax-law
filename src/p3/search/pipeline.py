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
    #: Кэш множеств чанков, действующих на дату.
    _as_of_cache: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        chunks_dir: Path,
        *,
        encoder_name: str | None = None,
        field_weights: dict[str, float] | None = None,
        corpus: str | None = None,
    ) -> "Retrieval":
        source = chunks_dir / (corpus or "nk1.jsonl")
        chunks = [json.loads(line) for line in source.open(encoding="utf-8")]
        store = ChunkStore.from_chunks(chunks)
        bm25 = build_index(chunks, field_weights=field_weights, cache_dir=chunks_dir / "bm25-cache")

        dense = encoder = None
        if encoder_name:
            stem = (corpus or "nk1.jsonl").replace(".jsonl", "")
            # Историческое имя для корпуса части первой; для остальных — с основой.
            path = chunks_dir / (
                f"dense-{encoder_name}.npz" if stem == "nk1"
                else f"dense-{stem}-{encoder_name}.npz"
            )
            if path.exists():
                dense = DenseIndex.load(path)
                # Запасного варианта здесь быть не должно. Раньше при отсутствии
                # файла подставлялись векторы другого корпуса: 1906 векторов
                # части первой для корпуса из 9667 чанков. Ошибки не возникало,
                # dense-канал просто не мог вернуть ничего за пределами части
                # первой, а метрики выходили правдоподобные.
                missing = set(dense.doc_ids) - set(store.by_chunk_id)
                if missing or len(dense.doc_ids) != len(store.by_chunk_id):
                    raise ValueError(
                        f"векторы не соответствуют корпусу: {len(dense.doc_ids)} векторов "
                        f"против {len(store.by_chunk_id)} чанков"
                        + (f", {len(missing)} идентификаторов не найдено" if missing else "")
                        + f". Пересоберите: p3 build-embeddings --corpus {corpus or 'nk1.jsonl'}"
                    )
                encoder = DenseEncoder(name=encoder_name)
        return cls(store=store, bm25=bm25, dense=dense, encoder=encoder)

    # --- отдельные каналы -------------------------------------------------

    def allowed_at(self, as_of: int | None) -> set[str] | None:
        """Чанки, действующие на дату: `valid_from <= as_of < valid_to`.

        Основной фильтр — `applies_from`, но пока он не извлечён, работаем по
        `valid_from`; разница описана в docstring `p3.nk.temporal`.
        """
        if as_of is None:
            return None
        cached = self._as_of_cache.get(as_of)
        if cached is None:
            cached = {
                chunk_id
                for chunk_id, chunk in self.store.by_chunk_id.items()
                if (chunk.get("valid_from") or 0) <= as_of < (chunk.get("valid_to") or 0)
            }
            self._as_of_cache[as_of] = cached
        return cached

    def search_bm25(
        self, query: str, top_k: int = TOP_CANDIDATES, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=top_k, allowed=allowed)

    def query_vector(self, query: str):
        cached = self._query_vectors.get(query)
        if cached is None:
            cached = self.encoder.encode_queries([query])[0]
            self._query_vectors[query] = cached
        return cached

    def search_dense(
        self, query: str, top_k: int = TOP_CANDIDATES, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self.dense is None or self.encoder is None:
            return []
        return self.dense.search(self.query_vector(query), top_k=top_k, allowed=allowed)

    # --- конфигурации -----------------------------------------------------

    def run(
        self,
        query: str,
        *,
        mode: str,
        expand: bool = False,
        top_k: int = TOP_CANDIDATES,
        as_of: int | None = None,
    ) -> list[Hit]:
        allowed = self.allowed_at(as_of)
        if mode == "bm25":
            hits = [Hit(d, s, sources=("bm25",)) for d, s in self.search_bm25(query, top_k, allowed)]
        elif mode == "dense":
            hits = [Hit(d, s, sources=("dense",)) for d, s in self.search_dense(query, top_k, allowed)]
        elif mode in ("hybrid", "hybrid+rerank"):
            hits = reciprocal_rank_fusion({
                "bm25": self.search_bm25(query, top_k, allowed),
                "dense": self.search_dense(query, top_k, allowed),
            })
        else:
            raise ValueError(f"неизвестная конфигурация {mode!r}")

        if mode == "hybrid+rerank":
            if self.reranker is None:
                self.reranker = Reranker()
            # Кандидаты для реранкера: top-N слияния плюс раскрытые ссылки.
            # Раскрывать после отсечения нельзя: слияние даёт под сотню
            # результатов, раскрытые дописываются в конец и до реранкера просто
            # не доходят — измерение вклада one-hop тогда структурно неспособно
            # что-либо показать и всегда даёт ноль.
            candidates = hits[:RERANK_INPUT]
            if expand:
                candidates = one_hop_expansion(candidates, self.store)
            return self.reranker.rerank(query, candidates, self.store, top_k=RERANK_OUTPUT)

        if expand:
            hits = one_hop_expansion(hits, self.store)
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


def to_versions(hits: list[Hit], store: ChunkStore) -> list[str]:
    """Ранжирование чанков → ранжирование версий норм (`nk1/st88/p3@v2`)."""
    seen: dict[str, None] = {}
    for hit in hits:
        chunk = store.by_chunk_id.get(hit.chunk_id)
        if chunk and chunk.get("version_id"):
            seen.setdefault(chunk["version_id"], None)
    return list(seen)


def indexed_articles(store: ChunkStore) -> set[str]:
    return {a for a in (store.article_of(c) for c in store.by_chunk_id) if a}


def expansion_contribution(
    retrieval: "Retrieval",
    queries,
    *,
    mode: str = "hybrid+rerank",
    level: str = "article",
    top_k: int = 8,
) -> dict:
    """Доля запросов, где эталон достался **только** через раскрытие ссылки.

    Требование ТЗ §7 отдельной строкой. Если доля заметна, это самостоятельный
    результат: детерминированный обход графа закрывает класс ошибок, который
    эмбеддер не закроет никогда — основание не исполнять требование выражено
    отсылкой, а не словами.

    Меряется на **финальном контексте** (top-8 после реранкера), а не на top-20
    слияния. Раскрытые чанки по спеке ставятся в конец, а слияние возвращает под
    сотню результатов — на top-20 они не попадают никогда, и измерение показывало
    бы ноль независимо от того, помогает раскрытие или нет.
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
