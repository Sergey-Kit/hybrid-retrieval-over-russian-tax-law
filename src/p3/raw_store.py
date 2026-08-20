"""Контентно-адресуемое хранилище скачанного сырья.

Каждый документ сохраняется байт-в-байт **до** любого парсинга, под именем своего
sha256, и получает строку в манифесте. Манифест лежит в git, байты — нет.

Зачем: спека требует хеш на уровне norm_version, но без хеша исходника ни спайк, ни
silver set не воспроизводимы — государственные сайты меняют вёрстку и содержимое, и
через месяц уже нельзя доказать, из чего именно был построен набор. Побочный эффект,
ради которого это и делается на практике: повторный прогон не ходит в сеть.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from p3.config import RAW_DIR

MANIFEST_NAME = "manifest.jsonl"


@dataclass(frozen=True)
class RawEntry:
    """Одна запись манифеста."""

    sha256: str
    url: str
    source: str
    fetched_at: str
    content_type: str | None = None
    encoding: str | None = None
    size: int = 0
    #: Страна выхода в сеть на момент загрузки. Происхождение важнее запретов:
    #: источники могут отдавать разное разным адресам, и если корпус окажется
    #: неполным, это первое, на что нужно смотреть.
    exit_country: str | None = None


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class RawStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else RAW_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME
        self._by_url: dict[tuple[str, str], RawEntry] = {}
        self._load_manifest()

    # --- манифест -------------------------------------------------------

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            self.manifest_path.touch()
            return
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = RawEntry(**json.loads(line))
                # Последняя запись побеждает: сайт мог отдать новое содержимое.
                self._by_url[(entry.source, entry.url)] = entry

    def _append_manifest(self, entry: RawEntry) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")

    # --- пути -----------------------------------------------------------

    def path_for(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    # --- чтение/запись --------------------------------------------------

    def lookup(self, url: str, source: str) -> RawEntry | None:
        """Запись для этого URL, если он уже скачан и байты на месте."""
        entry = self._by_url.get((source, url))
        if entry is None:
            return None
        return entry if self.path_for(entry.sha256).exists() else None

    def get(self, sha: str) -> bytes:
        return self.path_for(sha).read_bytes()

    def get_text(self, entry: RawEntry) -> str:
        """Декодированный текст. Кодировка берётся из записи, а не угадывается заново."""
        return self.get(entry.sha256).decode(entry.encoding or "utf-8", errors="replace")

    def put(
        self,
        url: str,
        content: bytes,
        *,
        source: str,
        content_type: str | None = None,
        encoding: str | None = None,
        exit_country: str | None = None,
    ) -> RawEntry:
        sha = sha256_bytes(content)
        dest = self.path_for(sha)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Запись через временный файл: прерванный краулинг не должен оставить
            # в хранилище файл с правильным именем и обрезанным содержимым.
            fd, tmp = tempfile.mkstemp(dir=str(dest.parent))
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(content)
                os.replace(tmp, dest)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

        entry = RawEntry(
            sha256=sha,
            url=url,
            source=source,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            content_type=content_type,
            encoding=encoding,
            size=len(content),
            exit_country=exit_country,
        )
        previous = self._by_url.get((source, url))
        # Тот же URL с тем же содержимым не плодит строк в манифесте.
        if previous is None or previous.sha256 != sha:
            self._append_manifest(entry)
            self._by_url[(source, url)] = entry
            return entry
        return previous

    def __len__(self) -> int:
        return len(self._by_url)
