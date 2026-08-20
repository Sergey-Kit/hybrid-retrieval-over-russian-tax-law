"""Преflight-проверка сети.

История этой проверки — сама по себе поучительная, и поэтому записана здесь.

Сначала было замечено, что `pravo.gov.ru:443` не отвечает на SYN, тогда как порт 80
отдаёт HTTP 200, — и из этого сделан вывод, что источник фильтрует иностранные
адреса, а значит сбор через VPN даст неизвестно насколько полный корпус. На этом
выводе была построена жёсткая блокировка сбора.

**Вывод оказался неверным.** Проверка с прямого подключения из РФ показала, что
443 закрыт и оттуда тоже: `pravo.gov.ru` просто не отдаёт HTTPS ни для кого.
Никакой геофильтрации нет, и блокировка сбора по этому признаку запрещала бы
работу навсегда.

Что осталось после разбора:

* требование TLS предъявляется только источникам, которые HTTPS действительно
  отдают (`Source.require_tls`);
* страна выхода **не блокирует** сбор. Гео-сервисы расходятся в показаниях
  (для одного и того же выхода получены FI и GB) и вдобавок недоступны из России
  без VPN — то есть ровно в том режиме, ради которого проверка задумывалась;
* вместо запрета — прослеживаемость: страна выхода пишется в манифест рядом с
  каждым скачанным документом. Если качество корпуса когда-нибудь окажется под
  вопросом, будет видно, откуда он собран.
"""
from __future__ import annotations

import json
import socket
import urllib.request
from dataclasses import dataclass, replace

from p3.config import EXPECTED_EXIT_COUNTRY, SKIP_PREFLIGHT, Source

#: Сервисы, отдающие страну по внешнему IP. Опрашиваются по очереди до первого
#: ответа. Список длинный намеренно: часть из них недоступна из России без VPN,
#: то есть ровно в том режиме, ради которого проверка и делается.
_GEO_ENDPOINTS = (
    "https://api.myip.com",
    "https://ipapi.co/json/",
    "https://ipinfo.io/json",
    "https://ifconfig.co/json",
)


class PreflightError(RuntimeError):
    """Среда не пригодна для сбора данных."""


@dataclass(frozen=True)
class PreflightReport:
    exit_country: str | None
    tls_ok: dict[str, bool]
    #: Доступность 80 — диагностика, не критерий.
    http_ok: dict[str, bool] | None = None
    #: Заполняется, если страна выхода не совпала с ожидаемой. Не блокирует сбор.
    country_warning: str | None = None

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
        for key in ("cc", "country_code", "country_iso", "country"):
            country = payload.get(key)
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
            "порт 443 недоступен у источника, который обязан его отдавать: "
            + ", ".join(blocked)
        )

    if require_country and country is not None and country != EXPECTED_EXIT_COUNTRY:
        # Предупреждение, а не отказ: см. историю в docstring модуля.
        report = replace(report, country_warning=(
            f"внешний IP определён как {country}, ожидается {EXPECTED_EXIT_COUNTRY}. "
            "Происхождение записывается в манифест; если корпус окажется неполным, "
            "это первое, на что смотреть"
        ))

    if problems:
        raise PreflightError(
            "Среда не пригодна для сбора данных:\n  - "
            + "\n  - ".join(problems)
            + "\n\nДля оффлайн-прогона из кэша можно выставить P3_SKIP_PREFLIGHT=1 — "
            "но не для сбора."
        )
    return report
