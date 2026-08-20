"""Вежливый HTTP-клиент: троттлинг, robots.txt, ретраи, кэш в raw_store.

Правила, которые здесь зашиты и которые не должны зависеть от вызывающего кода:

* один запрос в секунду к хосту (настраивается, но не отключается);
* User-Agent с контактом — источники государственные, аноним там неуместен;
* robots.txt проверяется и соблюдается;
* всё скачанное немедленно уходит в raw_store, до парсинга;
* повторный вызов того же URL берётся из кэша и в сеть не идёт.
"""
from __future__ import annotations

import logging
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

from p3.config import (
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    THROTTLE_SECONDS,
    USER_AGENT,
    Source,
)
from p3.raw_store import RawEntry, RawStore

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


class RobotsDisallowed(FetchError):
    pass


class OfflineMiss(FetchError):
    """Оффлайн-режим, а документа нет в кэше."""


@dataclass
class Fetched:
    entry: RawEntry
    text: str
    from_cache: bool


class PoliteFetcher:
    def __init__(
        self,
        store: RawStore | None = None,
        *,
        offline: bool = False,
        throttle: float = THROTTLE_SECONDS,
        respect_robots: bool = True,
        exit_country: str | None = None,
    ) -> None:
        self.store = store if store is not None else RawStore()
        self.offline = offline
        self.throttle = throttle
        self.respect_robots = respect_robots
        #: Пишется в манифест рядом с каждым документом как происхождение.
        self.exit_country = exit_country
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # --- вежливость -----------------------------------------------------

    def _wait(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            delay = self.throttle - (time.monotonic() - last)
            if delay > 0:
                time.sleep(delay)
        self._last_request[host] = time.monotonic()

    def _robots_for(self, scheme: str, host: str):
        if host in self._robots:
            return self._robots[host]
        parser = urllib.robotparser.RobotFileParser()
        url = f"{scheme}://{host}/robots.txt"
        try:
            self._wait(host)
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
            else:
                # Нет robots.txt — считаем, что ограничений нет. Явно, а не молча.
                parser = None
        except requests.RequestException:
            log.warning("robots.txt недоступен для %s, продолжаем с троттлингом", host)
            parser = None
        self._robots[host] = parser
        return parser

    def _check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return
        parts = urlparse(url)
        parser = self._robots_for(parts.scheme, parts.netloc)
        if parser is not None and not parser.can_fetch(USER_AGENT, url):
            raise RobotsDisallowed(f"robots.txt запрещает {url}")

    # --- загрузка -------------------------------------------------------

    def get(
        self,
        url: str,
        source: Source,
        *,
        params: dict | None = None,
        force_refresh: bool = False,
    ) -> Fetched:
        full_url = url if urlparse(url).scheme else urljoin(source.base_url, url)
        if params:
            full_url = requests.Request("GET", full_url, params=params).prepare().url

        if not force_refresh:
            cached = self.store.lookup(full_url, source.name)
            if cached is not None:
                return Fetched(cached, self.store.get_text(cached), from_cache=True)

        if self.offline:
            raise OfflineMiss(f"нет в кэше и включён оффлайн-режим: {full_url}")

        self._check_robots(full_url)

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait(urlparse(full_url).netloc)
            try:
                resp = self.session.get(full_url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if resp.status_code == 200:
                    encoding = source.force_encoding or resp.encoding or "utf-8"
                    entry = self.store.put(
                        full_url,
                        resp.content,
                        source=source.name,
                        content_type=resp.headers.get("Content-Type"),
                        encoding=encoding,
                        exit_country=self.exit_country,
                    )
                    text = resp.content.decode(encoding, errors="replace")
                    return Fetched(entry, text, from_cache=False)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = FetchError(f"HTTP {resp.status_code} на {full_url}")
                else:
                    raise FetchError(f"HTTP {resp.status_code} на {full_url}")
            backoff = min(2 ** attempt, 30)
            log.warning("попытка %s/%s не удалась (%s), пауза %sс", attempt, MAX_RETRIES, last_error, backoff)
            time.sleep(backoff)

        raise FetchError(f"не удалось загрузить {full_url}: {last_error}")
