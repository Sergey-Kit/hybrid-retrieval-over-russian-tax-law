"""Dense-канал: эмбеддинги чанков и запросов.

Модель и её префиксы задаются **в конфиге по одному разу** — это требование
спеки §4, и оно не формальность. Инструктивные модели (FRIDA) требуют разных
префиксов для запроса и документа, и ошибка здесь тихая: если проиндексировать
корпус с префиксом запроса или забыть префикс вовсе, эмбеддинги посчитаются,
поиск заработает, а recall будет просто хуже, чем мог бы. Ещё коварнее —
одинаковый префикс с обеих сторон.

У `deepvk/USER-bge-m3` префиксов нет, и это записано явно, а не подразумевается:
пустая строка в конфиге — осознанное значение, а не забытое поле.

Guard на обрезку — тоже из спеки: после токенизации проверяется, что вход не
превысил лимит энкодера. Не предупреждение в лог, а исключение. Обрезка не
ломает ничего видимого: хвост текста нормы молча исчезает, остаётся префикс,
метрики выходят правдоподобные.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

#: Профили энкодеров. Сравнение на шаге 9 идёт по этим трём.
ENCODERS = {
    "user-bge-m3": {
        "model": "deepvk/USER-bge-m3",
        "max_tokens": 8192,
        "query_prefix": "",
        "document_prefix": "",
    },
    "bge-m3": {
        "model": "BAAI/bge-m3",
        "max_tokens": 8192,
        "query_prefix": "",
        "document_prefix": "",
    },
    "frida": {
        "model": "ai-forever/FRIDA",
        "max_tokens": 512,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
    },
}

DEFAULT_ENCODER = os.environ.get("P3_ENCODER", "user-bge-m3")


class TruncationError(RuntimeError):
    """Вход длиннее лимита энкодера. Хвост нормы был бы молча отброшен."""


@dataclass
class DenseEncoder:
    name: str = DEFAULT_ENCODER
    device: str | None = None
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.name not in ENCODERS:
            raise ValueError(f"неизвестный энкодер {self.name!r}; известны: {sorted(ENCODERS)}")
        self.profile = ENCODERS[self.name]
        self._model = None
        self._tokenizer = None

    # --- ленивая загрузка ------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.profile["model"])
        self._model = AutoModel.from_pretrained(
            self.profile["model"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    # --- guard на обрезку -------------------------------------------------

    def assert_fits(self, texts: list[str]) -> None:
        """Упасть, если хоть один вход не помещается в энкодер."""
        self._ensure_loaded()
        limit = self.profile["max_tokens"]
        lengths = [len(self._tokenizer.encode(text, truncation=False)) for text in texts]
        too_long = [(i, n) for i, n in enumerate(lengths) if n > limit]
        if too_long:
            worst = max(too_long, key=lambda item: item[1])
            raise TruncationError(
                f"{len(too_long)} из {len(texts)} входов длиннее лимита {limit} токенов "
                f"(максимум {worst[1]}). Обрезка отбросила бы хвост нормы молча."
            )

    # --- кодирование ------------------------------------------------------

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        import torch

        self._ensure_loaded()
        prepared = [prefix + text for text in texts]
        self.assert_fits(prepared)

        vectors: list[np.ndarray] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=False, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = self._model(**encoded)
            # CLS-пулинг — то, чем пользуется сам BGE-M3.
            embeddings = output.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            vectors.append(embeddings.float().cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.profile["document_prefix"])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.profile["query_prefix"])


@dataclass
class DenseIndex:
    """Точный поиск перебором. ANN не нужен: спека §3 — на этом объёме
    brute force и есть эталон, recall ретривера равен 1.0 по построению."""

    doc_ids: list[str]
    vectors: np.ndarray

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def save(self, path) -> None:
        np.savez_compressed(path, doc_ids=np.array(self.doc_ids), vectors=self.vectors)

    @classmethod
    def load(cls, path) -> "DenseIndex":
        data = np.load(path, allow_pickle=False)
        return cls(doc_ids=[str(x) for x in data["doc_ids"]], vectors=data["vectors"])
