"""BM25F: взвешенная сумма по полям `body` и `title`.

Почему своя реализация, а не Postgres. `ts_rank` — не BM25 и не умеет полевых
весов, которые спека требует тюнить (`body: 1.0, title: 0.3` как стартовая
точка); `pg_search`/ParadeDB поставляется докер-образом, а докера на этой машине
нет. Рассинхрон двух хранилищ гасится не «одной СУБД», а общим `corpus_build_id`
и ассертом на совпадение множеств чанков.

Нормировка длины — **по каждому полю отдельно**, а не по суммарной длине
документа. Разница не косметическая: при общей нормировке длинное тело пункта
«топит» короткое поле с сильным сигналом, и точная ссылка `nk1_st88_p3` с IDF
6.6 проигрывает частотным «нк» и «рф» в коротком документе. Именно это и
происходило на запросе «п. 3 ст. 88 НК РФ», пока формула была упрощённой.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from p3.search.text import reference_token, tokenize

#: Стартовые веса полей: `body` и `title` — из спеки §3, `ref` добавлено.
#: Поле `ref` несёт собственный идентификатор чанка одним токеном. Без него
#: норму нельзя найти по её же цитате: тело пункта ссылается на соседей, но не
#: на себя, а разбор заголовка зависит от того, распознает ли регексп фразу.
#: Идентичность чанка не должна держаться на регулярном выражении.
#: `ref` = 2.0 — единственное значение, подобранное на данных (train, 162
#: достижимых запроса). Свип по 12 конфигурациям показал, что веса полей здесь
#: слабый рычаг: recall@20 гуляет в пределах 1.2 п.п. Но рост веса `ref` даёт
#: устойчивый монотонный прирост nDCG@10 при любом весе заголовка — точно
#: процитированная норма поднимается выше, не меняя состава найденного.
#: Вес заголовка оставлен как в спеке: выбирать «лучший» на разнице 0.4 п.п.
#: значило бы подгонять под шум.
DEFAULT_FIELD_WEIGHTS = {"body": 1.0, "title": 0.3, "ref": 2.0}
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


@dataclass
class Bm25fIndex:
    """Инвертированный индекс по полям. Строится один раз, живёт в памяти."""

    field_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS))
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    doc_ids: list[str] = field(default_factory=list)
    #: term → doc_index → взвешенная и нормированная по длине частота
    postings: dict[str, dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    #: field → список длин по документам; нужен для средних до нормировки.
    _field_lengths: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    #: field → doc_index → Counter токенов, до нормировки.
    _raw: list[dict[str, Counter]] = field(default_factory=list)

    def add(self, doc_id: str, fields: dict[str, str]) -> None:
        self.doc_ids.append(doc_id)
        per_field: dict[str, Counter] = {}
        for name in self.field_weights:
            tokens = tokenize(fields.get(name, "") or "")
            per_field[name] = Counter(tokens)
            self._field_lengths[name].append(len(tokens))
        self._raw.append(per_field)

    def finalize(self) -> None:
        """Посчитать средние длины полей и собрать постинги."""
        averages = {
            name: (sum(lengths) / len(lengths) if lengths else 0.0)
            for name, lengths in self._field_lengths.items()
        }
        for index, per_field in enumerate(self._raw):
            for name, counts in per_field.items():
                weight = self.field_weights.get(name, 0.0)
                if not weight or not counts:
                    continue
                average = averages.get(name) or 1.0
                length = self._field_lengths[name][index]
                # Нормировка длины — внутри поля, до суммирования по полям.
                norm = 1 - self.b + self.b * (length / average)
                for token, count in counts.items():
                    self.postings[token][index] = (
                        self.postings[token].get(index, 0.0) + weight * count / norm
                    )
        self._raw = []

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        # Вариант BM25 с гарантированно положительным IDF.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 50,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.doc_ids:
            return []

        permitted = (
            None if allowed is None
            else {i for i, doc_id in enumerate(self.doc_ids) if doc_id in allowed}
        )
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            for index, weighted_tf in posting.items():
                if permitted is not None and index not in permitted:
                    continue
                # Насыщение применяется к уже нормированной сумме по полям.
                scores[index] += idf * weighted_tf / (self.k1 + weighted_tf)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[i], score) for i, score in ranked[:top_k]]


def _fingerprint(chunks, field_weights: dict[str, float]) -> str:
    """Отпечаток входа: состав чанков и веса полей."""
    digest = hashlib.sha256()
    digest.update(json.dumps(field_weights, sort_keys=True).encode())
    digest.update(str(len(chunks)).encode())
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode())
    return digest.hexdigest()[:16]


def build_index(
    chunks,
    *,
    field_weights: dict[str, float] | None = None,
    cache_dir: "pathlib.Path | None" = None,
) -> Bm25fIndex:
    """Индекс по чанкам. `text_bm25` — тело, `title_bm25` — заголовок.

    С `cache_dir` индекс сохраняется на диск и переиспользуется. Сборка на
    объединённом корпусе (9667 чанков, 1.2 млн слов) занимает десятки минут:
    `tokenize` для каждого поля запускает разбор цитат регулярками, а в части
    второй 32 тысячи ссылок. Без кэша каждый прогон оценки платил бы это заново,
    а впереди сравнение трёх энкодеров — минимум три-четыре прогона.
    """
    weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)
    cache_path = None
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"bm25-{_fingerprint(chunks, weights)}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as fh:
                return pickle.load(fh)

    index = Bm25fIndex(field_weights=weights)
    for chunk in chunks:
        norm_unit_id = chunk["norm_unit_id"]
        article_id = "/".join(norm_unit_id.split("/")[:2])
        # И собственный идентификатор, и идентификатор статьи: запрос «ст. 88»
        # должен доставать все её пункты, «п. 3 ст. 88» — конкретный.
        refs = " ".join(dict.fromkeys([reference_token(norm_unit_id), reference_token(article_id)]))
        index.add(
            chunk["chunk_id"],
            {
                "body": chunk.get("text_bm25", ""),
                "title": chunk.get("title_bm25", ""),
                "ref": refs,
            },
        )
    index.finalize()
    if cache_path is not None:
        with cache_path.open("wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return index
