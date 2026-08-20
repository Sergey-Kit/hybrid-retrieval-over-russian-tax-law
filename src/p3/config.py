"""Единая точка конфигурации.

Всё, что может отличаться между локальной машиной и арендованной, живёт здесь, а не
в коде этапов: план требует, чтобы перенос прогона на другую машину был настройкой,
а не портированием.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("P3_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SILVER_DIR = DATA_DIR / "silver"

#: Контакт в User-Agent — требование вежливости к государственным источникам.
USER_AGENT = os.environ.get(
    "P3_USER_AGENT",
    "P3-research-crawler/0.1 (+sergey87kitaev@gmail.com)",
)

#: Минимальная пауза между запросами к одному хосту, секунды.
THROTTLE_SECONDS = float(os.environ.get("P3_THROTTLE_SECONDS", "1.0"))

REQUEST_TIMEOUT = float(os.environ.get("P3_REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.environ.get("P3_MAX_RETRIES", "3"))

#: Страна, из которой сбор считается корректным. Источники геофильтруют иностранные
#: адреса, см. docs/network-preflight.md.
EXPECTED_EXIT_COUNTRY = os.environ.get("P3_EXPECTED_EXIT_COUNTRY", "RU")

#: Аварийный выключатель преflight-проверки. Существует только для оффлайн-прогонов
#: из кэша и для тестов; для сбора данных использовать его нельзя.
SKIP_PREFLIGHT = os.environ.get("P3_SKIP_PREFLIGHT", "").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Source:
    """Описание источника: как к нему ходить и что проверять до сбора."""

    name: str
    host: str
    #: Схема, которой пользуемся по умолчанию.
    scheme: str = "https"
    #: Кодировка, если сервер её не объявляет или объявляет неверно.
    force_encoding: str | None = None
    #: Требовать ли доступности 443 в преflight. Для геофильтрованных источников
    #: недоступность 443 означает, что мы смотрим на них с "не того" адреса.
    require_tls: bool = True

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}"


PRAVO = Source(
    name="pravo",
    host="pravo.gov.ru",
    force_encoding="cp1251",
)
PUBLICATION = Source(
    name="publication",
    host="publication.pravo.gov.ru",
)
NALOG = Source(
    name="nalog",
    host="www.nalog.gov.ru",
)

SOURCES: dict[str, Source] = {s.name: s for s in (PRAVO, PUBLICATION, NALOG)}
