"""Хранилище чанков в PostgreSQL с pgvector.

Зачем оно, если при 8198 чанках точный перебор по numpy занимает доли
миллисекунды и СУБД не меняет ни одной измеряемой величины.

Причина продуктовая, а не вычислительная. P5 поставляется on-prem, и одна СУБД
в инсталляции у бухгалтерской фирмы — это прямая экономия на установке против
двух сервисов. Плюс темпоральный фильтр становится обычным `WHERE`, а обход
графа ссылок — рекурсивным CTE в один запрос вместо нескольких round-trip'ов.

Причина инженерная — рассинхрон. Спека называет расхождение векторного и
лексического индексов классическим источником тихих ошибок: чанк обновился в
одном, не обновился в другом, метрики врут. Здесь оба живут в одной транзакции
и несут общий `corpus_build_id`, а расхождение проверяется одним запросом.

Слой намеренно повторяет контракт `DenseIndex`/`Bm25fIndex`
(`search(...) -> [(chunk_id, score)]`), чтобы переезд был заменой реализации,
а не переписыванием пайплайна.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Поля payload, которые кладутся в колонки как есть.
_COLUMNS = (
    "chunk_id", "norm_unit_id", "version_id", "code", "part", "section", "chapter",
    "article", "article_title", "clause", "subclause", "valid_from", "valid_to",
    "applies_from", "published_at", "status", "version_no", "amended_by",
    "source_type", "authority", "text_dense", "text_bm25", "title_bm25",
)
_JSON_COLUMNS = ("refs_out", "para_offsets")


def corpus_build_id(chunks: list[dict]) -> str:
    """Отпечаток сборки корпуса: те же чанки — тот же идентификатор."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        digest.update(chunk["chunk_id"].encode())
        digest.update(chunk.get("text_bm25", "").encode())
    return digest.hexdigest()[:16]


def schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


@dataclass
class PgChunkStore:
    """Тонкая обёртка над соединением. Транзакциями управляет вызывающий."""

    connection: object

    # --- схема и загрузка -------------------------------------------------

    def create_schema(self) -> None:
        with self.connection.cursor() as cur:
            cur.execute(schema_sql())

    def load_chunks(self, chunks: list[dict], build_id: str) -> int:
        rows = []
        for chunk in chunks:
            row = [chunk.get(name) for name in _COLUMNS]
            row += [json.dumps(chunk.get(name, []), ensure_ascii=False) for name in _JSON_COLUMNS]
            row.append(build_id)
            rows.append(tuple(row))
        columns = ", ".join(_COLUMNS + _JSON_COLUMNS + ("corpus_build_id",))
        placeholders = ", ".join(["%s"] * (len(_COLUMNS) + len(_JSON_COLUMNS) + 1))
        with self.connection.cursor() as cur:
            cur.executemany(
                f"INSERT INTO chunks ({columns}) VALUES ({placeholders}) "
                "ON CONFLICT (chunk_id) DO NOTHING",
                rows,
            )
        return len(rows)

    def load_vectors(self, doc_ids: list[str], vectors, encoder: str, build_id: str) -> int:
        rows = [
            (doc_id, encoder, [float(v) for v in vector], build_id)
            for doc_id, vector in zip(doc_ids, vectors)
        ]
        with self.connection.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunk_vectors (chunk_id, encoder, embedding, corpus_build_id) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                rows,
            )
        return len(rows)

    # --- проверка целостности ---------------------------------------------

    def integrity(self, build_id: str) -> dict:
        """Совпадают ли множества чанков и векторов одной сборки.

        Ровно та проверка, отсутствие которой уже стоило двух неверных прогонов:
        подставленные векторы чужого корпуса и схлопнувшиеся дубли
        идентификаторов не вызывали ошибки, а просто давали неверные числа.
        """
        with self.connection.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE corpus_build_id = %s", (build_id,))
            chunks = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM chunk_vectors WHERE corpus_build_id = %s", (build_id,)
            )
            vectors = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM chunks c LEFT JOIN chunk_vectors v USING (chunk_id) "
                "WHERE c.corpus_build_id = %s AND v.chunk_id IS NULL",
                (build_id,),
            )
            missing = cur.fetchone()[0]
        return {
            "chunks": chunks,
            "vectors": vectors,
            "chunks_without_vector": missing,
            "consistent": chunks == vectors and missing == 0,
        }

    # --- поиск -------------------------------------------------------------

    def candidates(self, as_of: int | None = None, status: str | None = "active") -> list[str]:
        """Идентификаторы чанков, прошедших фильтр по метаданным.

        Фильтр применяется **до** поиска, как требует спека §6.3: иначе
        отсечённые кандидаты занимают места в top-k, и потеря незаметна на
        метриках, где темпоральных запросов нет. Здесь это обычный `WHERE`,
        и планировщик сам решает, идти по индексу или сканировать.
        """
        conditions, params = ["TRUE"], []
        if as_of is not None:
            conditions.append("valid_from <= %s AND valid_to > %s")
            params += [as_of, as_of]
        if status is not None:
            conditions.append("status = %s")
            params.append(status)
        with self.connection.cursor() as cur:
            cur.execute(f"SELECT chunk_id FROM chunks WHERE {' AND '.join(conditions)}", params)
            return [row[0] for row in cur.fetchall()]

    def vectors_of(self, chunk_ids: list[str], encoder: str):
        """Эмбеддинги указанных чанков. Сходство считается вызывающим кодом."""
        if not chunk_ids:
            return [], []
        with self.connection.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, embedding FROM chunk_vectors "
                "WHERE encoder = %s AND chunk_id = ANY(%s)",
                (encoder, list(chunk_ids)),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]

    def expand_refs(self, chunk_ids: list[str], ref_types: tuple[str, ...]) -> list[str]:
        """Один шаг по графу ссылок — одним запросом вместо round-trip'ов."""
        if not chunk_ids:
            return []
        with self.connection.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT t.chunk_id "
                "FROM chunks c, LATERAL jsonb_array_elements(c.refs_out) AS r "
                "JOIN chunks t ON t.norm_unit_id = r->>'target' "
                "WHERE c.chunk_id = ANY(%s) AND r->>'type' = ANY(%s) "
                "AND NOT (t.chunk_id = ANY(%s))",
                (list(chunk_ids), list(ref_types), list(chunk_ids)),
            )
            return [row[0] for row in cur.fetchall()]
