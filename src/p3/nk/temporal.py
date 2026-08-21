"""Темпоральная модель: версии норм и интервалы их действия.

Спека §5 разводит три уровня: `norm_unit` живёт вечно, `norm_version` — текст в
интервале действия, `chunk` — то, что лежит в индексе. Здесь строится средний
уровень: по нескольким редакциям одного документа определяется, где текст нормы
менялся, и каждой версии присваивается интервал.

## Что здесь честно, а что приближение

**Честно:** факт изменения. Если текст пункта в редакции A и в редакции B
различается, норма менялась — это наблюдение, а не догадка.

**Приближение — границы интервала.** Спека §5 называет четыре несовпадающие даты
и предупреждает, что путать их нельзя. Из ИПС известна только дата изменяющего
акта; дата вступления в силу задаётся самим актом и обычно позже, а
`applies_from` бывает ретроспективной. Поэтому:

* `valid_from` версии = дата редакции, в которой этот текст **впервые встретился
  среди просмотренных**;
* граница смещена вперёд ещё и потому, что редакции берутся выборочно: изменение
  между двумя срезами приписывается дате более позднего.

Из этого следует область применимости: построенные интервалы годятся для
**проверки механики темпорального фильтра** — что фильтр выбирает ту версию,
которую должен, — и не годятся как юридическое утверждение о том, какая редакция
действовала на дату. Второе требует разбора норм о вступлении в силу и относится
к вехе с консолидатором.

## Ловушка нумерации

`rdk` в ИПС **не хронологичен**: редакция 212 датирована 29.12.2020, а 197 —
28.12.2022. Сортировать редакции по номеру нельзя, только по дате.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

#: Sentinel «действует сейчас» из спеки §5.
SENTINEL_VALID_TO = 99991231


def to_int_date(value: date) -> int:
    """Дата как `YYYYMMDD`: range-фильтры по integer быстрее и читаемы в логах."""
    return value.year * 10000 + value.month * 100 + value.day


def parse_ru_date(value: str) -> date:
    day, month, year = value.split(".")
    return date(int(year), int(month), int(day))


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


@dataclass
class NormVersion:
    """Текст нормы в интервале действия."""

    norm_unit_id: str
    text: str
    digest: str
    valid_from: int
    valid_to: int = SENTINEL_VALID_TO
    #: Редакции ИПС, в которых встретился ровно этот текст.
    editions: list[int] = field(default_factory=list)
    version_no: int = 1

    @property
    def version_id(self) -> str:
        return f"{self.norm_unit_id}@v{self.version_no}"

    def covers(self, as_of: int) -> bool:
        return self.valid_from <= as_of < self.valid_to


@dataclass
class EditionSnapshot:
    """Разобранная редакция: норма → текст."""

    rdk: int
    edition_date: date
    texts: dict[str, str]

    @property
    def as_int(self) -> int:
        return to_int_date(self.edition_date)


def build_versions(snapshots: list[EditionSnapshot]) -> dict[str, list[NormVersion]]:
    """Свести срезы в версии с интервалами.

    Срезы упорядочиваются **по дате**, а не по номеру редакции.
    """
    ordered = sorted(snapshots, key=lambda s: s.edition_date)
    versions: dict[str, list[NormVersion]] = {}

    for snapshot in ordered:
        for norm_unit_id, text in snapshot.texts.items():
            digest = text_hash(text)
            history = versions.setdefault(norm_unit_id, [])
            if history and history[-1].digest == digest:
                # Текст не изменился — продолжается та же версия.
                history[-1].editions.append(snapshot.rdk)
                continue
            if history:
                # Предыдущая версия перестаёт действовать с этого среза.
                history[-1].valid_to = snapshot.as_int
            history.append(
                NormVersion(
                    norm_unit_id=norm_unit_id,
                    text=text,
                    digest=digest,
                    valid_from=snapshot.as_int,
                    editions=[snapshot.rdk],
                    version_no=len(history) + 1,
                )
            )
    return versions


def version_at(versions: list[NormVersion], as_of: int) -> NormVersion | None:
    for version in versions:
        if version.covers(as_of):
            return version
    return None


def changed_norms(versions: dict[str, list[NormVersion]]) -> dict[str, list[NormVersion]]:
    """Нормы, у которых больше одной версии — материал для синтетических тестов."""
    return {unit: history for unit, history in versions.items() if len(history) > 1}


def summarize(versions: dict[str, list[NormVersion]]) -> dict:
    histories = list(versions.values())
    changed = changed_norms(versions)
    return {
        "norm_units": len(versions),
        "versions_total": sum(len(h) for h in histories),
        "norm_units_changed": len(changed),
        "max_versions_per_unit": max((len(h) for h in histories), default=0),
    }


def clause_unit_of(chunk: dict) -> str:
    """Идентификатор пункта, к которому относится чанк.

    У чанков-подпунктов `norm_unit_id` длиннее (`…/spN`), а версии считаются на
    уровне пункта — сопоставлять надо по пункту.
    """
    article = f"nk{chunk['part']}/st{chunk['article']}"
    clause = chunk.get("clause")
    return f"{article}/p{clause}" if clause else article


def apply_temporal_fields(
    chunks: list[dict],
    versions: dict[str, list[NormVersion]],
    rdk: int,
) -> list[dict]:
    """Проставить чанкам одной редакции интервалы действия их версии.

    Сопоставление идёт по принадлежности редакции версии, а не по хешу текста:
    длинный пункт дробится на несколько чанков, и хеш чанка не равен хешу пункта.
    """
    for chunk in chunks:
        history = versions.get(clause_unit_of(chunk), [])
        match = next((v for v in history if rdk in v.editions), None)
        if match is None:
            continue
        chunk["valid_from"] = match.valid_from
        chunk["valid_to"] = match.valid_to
        chunk["version_no"] = match.version_no
        chunk["version_id"] = f"{chunk['norm_unit_id']}@v{match.version_no}"
        chunk["chunk_id"] = chunk["version_id"] + "#" + chunk["chunk_id"].rsplit("#", 1)[1]
    return chunks


def deduplicate(chunks: list[dict]) -> list[dict]:
    """Одна запись на (норма, версия, позиция): редакции повторяют неизменный текст."""
    seen: dict[str, dict] = {}
    for chunk in chunks:
        seen.setdefault(chunk["chunk_id"], chunk)
    return list(seen.values())
