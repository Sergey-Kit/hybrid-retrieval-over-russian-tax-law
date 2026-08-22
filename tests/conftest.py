"""Подготовка окружения до импорта тестовых модулей.

`pgserver` вычисляет путь к файлу межпроцессной блокировки **на импорте**,
через `platformdirs.user_runtime_path`, то есть из `XDG_RUNTIME_DIR`. В этом
окружении переменная указывает на `/run/user/<uid>`, каталог, недоступный на
запись, и импорт падает раньше, чем успевает отработать любая фикстура.

Поэтому переменная подменяется здесь: conftest выполняется до тестовых модулей.
"""
import os
import tempfile
from pathlib import Path


def _usable(path: str | None) -> bool:
    if not path:
        return False
    candidate = Path(path)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".p3-write-probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


if not _usable(os.environ.get("XDG_RUNTIME_DIR")):
    runtime = Path(tempfile.gettempdir()) / f"p3-runtime-{os.getuid()}"
    runtime.mkdir(parents=True, exist_ok=True)
    runtime.chmod(0o700)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime)
