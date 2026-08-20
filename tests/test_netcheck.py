import socket

import pytest

from p3.config import Source
from p3.netcheck import PreflightError, preflight, tcp_reachable


def test_tcp_reachable_on_listening_socket():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert tcp_reachable("127.0.0.1", port, timeout=2) is True


def test_tcp_reachable_false_on_closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert tcp_reachable("127.0.0.1", port, timeout=2) is False


def test_preflight_fails_when_required_tls_is_unreachable(monkeypatch):
    """443 требуется только с источников, которые HTTPS действительно отдают."""
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: False)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "RU")

    with pytest.raises(PreflightError, match="443"):
        preflight([Source(name="s", host="example.invalid", require_tls=True)])


def test_http_only_source_is_not_blocked_by_closed_443(monkeypatch):
    """pravo.gov.ru не слушает 443 ни для кого — требовать TLS значит запретить сбор."""
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: False)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "RU")

    report = preflight([Source(name="s", host="example.invalid", require_tls=False)])
    assert report.country_warning is None


def test_foreign_exit_warns_but_does_not_block(monkeypatch):
    """Гео-сервисы расходятся и недоступны из РФ без VPN — это не критерий отказа."""
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "FI")

    report = preflight([Source(name="s", host="example.invalid")])
    assert report.exit_country == "FI"
    assert "FI" in report.country_warning


def test_preflight_passes_on_clean_environment(monkeypatch):
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "RU")

    report = preflight([Source(name="s", host="example.invalid")])
    assert report.exit_country == "RU"
    assert report.country_warning is None


def test_unknown_country_does_not_block(monkeypatch):
    """Недоступность геосервиса — не повод считать среду плохой."""
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: None)

    assert preflight([Source(name="s", host="example.invalid")]).exit_country is None
