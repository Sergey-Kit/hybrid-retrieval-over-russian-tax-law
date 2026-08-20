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


def test_preflight_fails_when_tls_port_is_filtered(monkeypatch):
    """Геофильтр 443 обязан останавливать сбор, а не деградировать до http."""
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: False)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "RU")

    with pytest.raises(PreflightError, match="443"):
        preflight([Source(name="s", host="example.invalid")])


def test_preflight_fails_on_foreign_exit(monkeypatch):
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "FI")

    with pytest.raises(PreflightError, match="FI"):
        preflight([Source(name="s", host="example.invalid")])


def test_preflight_passes_on_clean_environment(monkeypatch):
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: "RU")

    report = preflight([Source(name="s", host="example.invalid")])
    assert report.exit_country == "RU"


def test_unknown_country_does_not_block(monkeypatch):
    """Недоступность геосервиса — не повод считать среду плохой."""
    monkeypatch.setattr("p3.netcheck.tcp_reachable", lambda *a, **k: True)
    monkeypatch.setattr("p3.netcheck.exit_country", lambda *a, **k: None)

    assert preflight([Source(name="s", host="example.invalid")]).exit_country is None
