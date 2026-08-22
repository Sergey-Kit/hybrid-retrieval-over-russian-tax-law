-- Схема хранилища чанков по спеке чанкинга §5.
--
-- Даты — integer YYYYMMDD, а не timestamp: range-фильтры по integer быстрее,
-- сравнение читаемо в логах, часовых поясов нет.
--
-- valid_to = 99991231 вместо NULL: фильтр по диапазону однороднее проверки на
-- NULL и лучше комбинируется с индексом.

-- pgvector намеренно не используется.
--
-- Причина не идеологическая: на 8198 чанках ANN не нужен — спека §3 прямо
-- говорит, что на таком объёме brute force и есть эталон, recall ретривера
-- равен 1.0 по построению. Расширение понадобится за пределами ~1–2 млн чанков.
--
-- Причина практическая: на целевой машине разработки (i5-2500K, 2011) бинарник
-- pgvector падает с `signal 4: Illegal instruction` — он собран под набор
-- инструкций новее, чем у этого процессора, а заголовков сервера для пересборки
-- в дистрибутиве нет. Это стоит держать в голове и для on-prem-поставки P5:
-- у бухгалтерской фирмы вполне может стоять такое же железо.
--
-- Эмбеддинги хранятся как REAL[]: Postgres остаётся единым источником правды и
-- обслуживает метаданные, темпоральный фильтр и граф ссылок, а сходство
-- считается точным перебором в процессе.

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    norm_unit_id    TEXT NOT NULL,
    version_id      TEXT NOT NULL,

    code            TEXT NOT NULL,
    part            SMALLINT NOT NULL,
    section         TEXT,
    chapter         TEXT,
    article         TEXT NOT NULL,
    article_title   TEXT,
    clause          TEXT,
    subclause       TEXT,

    valid_from      INTEGER,
    valid_to        INTEGER NOT NULL DEFAULT 99991231,
    applies_from    INTEGER,
    published_at    INTEGER,
    status          TEXT NOT NULL DEFAULT 'active',
    version_no      INTEGER,
    amended_by      TEXT,

    source_type     TEXT NOT NULL DEFAULT 'norm',
    authority       REAL NOT NULL DEFAULT 1.0,

    refs_out        JSONB NOT NULL DEFAULT '[]'::jsonb,
    para_offsets    JSONB NOT NULL DEFAULT '[]'::jsonb,

    text_dense      TEXT NOT NULL,
    text_bm25       TEXT NOT NULL,
    title_bm25      TEXT,

    -- Идентификатор сборки корпуса. Рассинхрон векторного и лексического
    -- индексов — классический источник тихих ошибок: чанк обновился в одном,
    -- не обновился в другом, и метрики врут. Общий ключ делает расхождение
    -- проверяемым одним запросом.
    corpus_build_id TEXT NOT NULL
);

-- Индекс статьи: одна норма — много версий, выборка по норме частая.
CREATE INDEX IF NOT EXISTS chunks_norm_unit_idx ON chunks (norm_unit_id);
-- Темпоральный фильтр применяется до поиска, поэтому он на горячем пути.
CREATE INDEX IF NOT EXISTS chunks_temporal_idx ON chunks (valid_from, valid_to);
CREATE INDEX IF NOT EXISTS chunks_article_idx ON chunks (part, article);
CREATE INDEX IF NOT EXISTS chunks_build_idx ON chunks (corpus_build_id);
-- Обход refs_out — рекурсивный CTE по JSONB.
CREATE INDEX IF NOT EXISTS chunks_refs_idx ON chunks USING gin (refs_out);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id        TEXT PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    encoder         TEXT NOT NULL,
    embedding       REAL[] NOT NULL,
    corpus_build_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chunk_vectors_build_idx ON chunk_vectors (corpus_build_id);
