"""Гибридный поиск: RRF-слияние, one-hop по графу ссылок, реранкер.

Пайплайн из спеки §6.3. Порядок важен: слияние идёт по **рангам**, а не по
сырым оценкам — шкалы dense и BM25 несопоставимы, и любая попытка сложить их
напрямую даёт результат, зависящий от нормировки, а не от релевантности.

One-hop раскрывает ссылки типов `procedure`, `exception` и `definition`.
`exception` здесь самый ценный: для задачи «отвечать или не отвечать» основания
не исполнять требование живут именно там, и dense-ретривер их не найдёт — в
тексте пункта они выражены отсылкой, а не словами.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Константа RRF из спеки §6.3.
RRF_K = 60

#: Типы ссылок, которые раскрываются на one-hop.
EXPANDED_REF_TYPES = ("procedure", "exception", "definition")


@dataclass
class Hit:
    chunk_id: str
    score: float
    #: Попал в выдачу только через раскрытие ссылки, а не через поиск.
    expanded: bool = False
    sources: tuple[str, ...] = ()


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[Hit]:
    """Слить несколько ранжирований по рангам, а не по оценкам."""
    weights = weights or {}
    scores: dict[str, float] = {}
    origins: dict[str, set[str]] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            origins.setdefault(doc_id, set()).add(name)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(doc_id, score, sources=tuple(sorted(origins[doc_id]))) for doc_id, score in ordered]


@dataclass
class ChunkStore:
    """Доступ к чанкам по идентификатору и по норме."""

    by_chunk_id: dict[str, dict]
    by_norm_unit: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "ChunkStore":
        by_chunk_id = {c["chunk_id"]: c for c in chunks}
        by_norm_unit: dict[str, list[str]] = {}
        for chunk in chunks:
            by_norm_unit.setdefault(chunk["norm_unit_id"], []).append(chunk["chunk_id"])
            article = "/".join(chunk["norm_unit_id"].split("/")[:2])
            if article != chunk["norm_unit_id"]:
                by_norm_unit.setdefault(article, []).append(chunk["chunk_id"])
        return cls(by_chunk_id=by_chunk_id, by_norm_unit=by_norm_unit)

    def article_of(self, chunk_id: str) -> str | None:
        chunk = self.by_chunk_id.get(chunk_id)
        if chunk is None:
            return None
        return "/".join(chunk["norm_unit_id"].split("/")[:2])


def one_hop_expansion(
    hits: list[Hit],
    store: ChunkStore,
    *,
    depth: int = 8,
    ref_types: tuple[str, ...] = EXPANDED_REF_TYPES,
) -> list[Hit]:
    """Дотянуть чанки по refs_out. Раскрытые ставятся в конец, помеченными."""
    seen = {hit.chunk_id for hit in hits}
    expanded: list[Hit] = []
    for hit in hits[:depth]:
        chunk = store.by_chunk_id.get(hit.chunk_id)
        if chunk is None:
            continue
        for reference in chunk.get("refs_out", []):
            if reference.get("type") not in ref_types:
                continue
            for target_id in store.by_norm_unit.get(reference["target"], []):
                if target_id in seen:
                    continue
                seen.add(target_id)
                expanded.append(Hit(target_id, 0.0, expanded=True, sources=("expansion",)))
    return hits + expanded


class Reranker:
    """Кросс-энкодер bge-reranker-v2-m3: единственный компонент, видящий
    запрос и документ совместно."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                 batch_size: int = 8) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def rerank(self, query: str, hits: list[Hit], store: ChunkStore, top_k: int = 8) -> list[Hit]:
        import torch

        if not hits:
            return []
        self._ensure_loaded()
        pairs, kept = [], []
        for hit in hits:
            chunk = store.by_chunk_id.get(hit.chunk_id)
            if chunk is None:
                continue
            pairs.append((query, chunk["text_dense"]))
            kept.append(hit)

        # Батчи формируются из документов близкой длины. Паддинг идёт до самого
        # длинного в батче, а кандидаты различаются от десятков токенов до сотен:
        # при произвольном порядке 52% вычислений уходило в заполнитель.
        # Сортировка не меняет ни одного результата — только состав батчей;
        # проверено сравнением с батчем 1, где паддинга нет вовсе: 5 из 5 запросов
        # дали идентичный порядок выдачи.
        #
        # Замер на реальном пути: 23.1 -> 15.4 с/запрос. Батч больше 8 на 4 ГБ
        # видеопамяти **замедляет** (16 -> 18.2 с, 32 -> 25.4 с) — длинные
        # последовательности начинают давить на память. Микробенчмарк одних лишь
        # прямых проходов показывал обратное, потому что не считал токенизацию
        # тридцати документов на CPU без AVX2.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        scores: list[float] = [0.0] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            batch = [pairs[i] for i in indices]
            encoded = self._tokenizer(
                [q for q, _ in batch], [d for _, d in batch],
                padding=True, truncation=True, max_length=1024, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits.view(-1).float()
            for position, score in zip(indices, logits.cpu().tolist()):
                scores[position] = score

        ranked = sorted(zip(kept, scores), key=lambda item: -item[1])
        return [Hit(h.chunk_id, s, expanded=h.expanded, sources=h.sources) for h, s in ranked[:top_k]]
