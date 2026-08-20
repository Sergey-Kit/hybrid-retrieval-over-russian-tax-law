"""Преflight-проверка сети: падать до сбора, а не портить корпус молча.

Замеры 2026-08-20 с выхода в Финляндии: `pravo.gov.ru:443` и
`publication.pravo.gov.ru:443` не отвечают на SYN, а порт 80 отдаёт HTTP 200.
Источник фильтрует иностранные адреса. Сбор в таком режиме не падает — он даёт
корпус, про полноту которого ничего не известно, и это ровно тот класс тихих
дефектов, против которого написана спека.

Поэтому: перед сбором проверяем доступность TLS и страну выхода, и при
несоответствии останавливаемся.
"""
from __future__ import annotations

import json
import socket
import urllib.request
from dataclasses import dataclass

from p3.config import EXPECTED_EXIT_COUNTRY, SKIP_PREFLIGHT, Source

#: Сервисы, отдающие страну по внешнему IP. Опрашиваются по очереди до первого ответа.
_GEO_ENDPOINTS = (
    "https://ipinfo.io/json",
    "https://ifconfig.co/json",
)


class PreflightError(RuntimeError):
    """Среда не пригодна для сбора данных."""


@dataclass(frozen=True)
class PreflightReport:
    exit_country: str | None
    tls_ok: dict[str, bool]

    def describe(self) -> str:
        tls = ", ".join(f"{h}:443={'ok' if v else 'DROPPED'}" for h, v in self.tls_ok.items())
        return f"страна выхода={self.exit_country or 'неизвестна'}; {tls}"


def tcp_reachable(host: str, port: int, timeout: float = 8.0) -> bool:
    """True, если TCP-соединение устанавливается.

    Отличает фильтрацию (SYN дропается, таймаут) от отказа в обслуживании на
    уровне приложения — для геофильтра важно именно первое.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def exit_country(timeout: float = 10.0) -> str | None:
    """Двухбуквенный код страны внешнего IP, либо None, если определить не удалось.

    Недоступность геосервиса сама по себе не является поводом остановить сбор:
    решение принимает вызывающий код, который знает, что именно он собирает.
    """
    for url in _GEO_ENDPOINTS:
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            continue
        country = payload.get("country") or payload.get("country_iso")
        if isinstance(country, str) and len(country) == 2:
            return country.upper()
    return None


def preflight(sources: list[Source], *, require_country: bool = True) -> PreflightReport:
    """Проверить среду перед сбором. Бросает PreflightError, если она непригодна."""
    tls_ok = {
        s.host: tcp_reachable(s.host, 443) if s.require_tls else True for s in sources
    }
    country = exit_country()
    report = PreflightReport(exit_country=country, tls_ok=tls_ok)

    if SKIP_PREFLIGHT:
        return report

    problems: list[str] = []

    blocked = [h for h, ok in tls_ok.items() if not ok]
    if blocked:
        problems.append(
            "порт 443 недоступен: " + ", ".join(blocked) + ". Признак того, что источник "
            "фильтрует ваш адрес; сбор по открытому 80 даст неизвестно насколько полный корпус"
        )

    if require_country and country is not None and country != EXPECTED_EXIT_COUNTRY:
        problems.append(
            f"внешний IP в стране {country}, ожидается {EXPECTED_EXIT_COUNTRY}. "
            "Похоже, трафик идёт через VPN"
        )

    if problems:
        raise PreflightError(
            "Среда не пригодна для сбора данных:\n  - "
            + "\n  - ".join(problems)
            + "\n\nЧто сделать: пустить домены источников напрямую, мимо VPN-туннеля "
            "(split-tunnel), и повторить. Для оффлайн-прогона из кэша можно выставить "
            "P3_SKIP_PREFLIGHT=1 — но не для сбора."
        )
    return report
