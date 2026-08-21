"""Извлечение графа внутренних ссылок НК (`refs_out`).

Спека чанкинга §4 требует размечать ссылки типами и использовать их для
one-hop expansion после гибридного поиска: «дёшево, детерминированно и закрывает
класс ошибок, который dense-ретривер не закроет никогда».

Две вещи, которых не делает общий нормализатор `p3.refs.normalize`:

1. **Разрешение относительных ссылок.** В тексте кодекса норма ссылается на
   соседей словами «пунктом 8.1 **настоящей статьи**», «подпунктом 2 **настоящего
   пункта**». Номера статьи там нет вовсе, и без контекста такая ссылка теряется —
   а именно в ней чаще всего живут исключения.
2. **Типизация.** `exception` — самый ценный тип для задачи «отвечать или не
   отвечать»: основания не исполнять требование живут именно там.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from p3.refs.normalize import find_citations

TYPE_PROCEDURE = "procedure"
TYPE_EXCEPTION = "exception"
TYPE_SANCTION = "sanction"
TYPE_DEFINITION = "definition"
TYPE_AMENDMENT = "amendment"
TYPE_PLAIN = "plain"

#: Насколько далеко влево от ссылки искать связку, задающую тип.
_TYPE_LOOKBEHIND = 90

_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (TYPE_AMENDMENT, re.compile(r"\(\s*(?:В\s+редакции|Дополнение|Изменение)", re.IGNORECASE)),
    (TYPE_EXCEPTION, re.compile(
        r"за\s+исключением|если\s+иное\s+не\s+(?:предусмотрено|установлено)|"
        r"кроме\s+случаев|не\s+применя\w+", re.IGNORECASE)),
    (TYPE_SANCTION, re.compile(
        r"влеч\w+|ответственност\w+|наказыва\w+|штраф\w*", re.IGNORECASE)),
    (TYPE_DEFINITION, re.compile(
        r"в\s+значении|определённ\w+|определенн\w+|понима\w+|признаётся|признается", re.IGNORECASE)),
    # Идёт последним намеренно: «предусмотренный» и «установленный» — универсальные
    # склейки, они присутствуют и в «за исключением случаев, предусмотренных…», и в
    # «влечёт ответственность, предусмотренную…». По близости они всегда побеждали бы
    # смысловую связку, поэтому тип определяется приоритетом, а не расстоянием.
    (TYPE_PROCEDURE, re.compile(
        r"в\s+порядке|в\s+соответствии\s+с|установленн\w+|предусмотренн\w+", re.IGNORECASE)),
)

#: «пунктом 8.1 настоящей статьи», «подпунктом 2 настоящего пункта», «абзацем вторым …»
_RELATIVE_RE = re.compile(
    r"(?P<kind>подпункт\w*|пункт\w*|абзац\w*|пп\.|п\.|абз\.)\s*"
    r"(?P<nums>\d+(?:\.\d+)*(?:\s*(?:,|и)\s*\d+(?:\.\d+)*)*)\s*"
    r"(?P<scope>настоящ\w+\s+(?:статьи|пункта|Кодекса|главы))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Reference:
    """Одна исходящая ссылка."""

    target: str
    type: str
    raw: str

    def as_payload(self) -> dict[str, str]:
        return {"target": self.target, "type": self.type, "raw": self.raw}


def classify(text: str, position: int) -> str:
    """Тип ссылки по связке слева от неё."""
    window = text[max(0, position - _TYPE_LOOKBEHIND) : position]
    for kind, pattern in _TYPE_PATTERNS:
        if pattern.search(window):
            return kind
    return TYPE_PLAIN


def _relative_target(kind: str, number: str, scope: str, article_id: str, clause: str | None) -> str | None:
    kind = kind.lower()
    scope = scope.lower()
    if "стать" in scope:
        base = article_id
        if kind.startswith(("п.", "пункт")):
            return f"{base}/p{number}"
        if kind.startswith(("пп", "подпункт")):
            # Подпункт «настоящей статьи» без пункта адресовать некуда.
            return None
        return None
    if "пункт" in scope:
        if clause is None:
            return None
        base = f"{article_id}/p{clause}"
        if kind.startswith(("пп", "подпункт")):
            return f"{base}/sp{number}"
        if kind.startswith(("абз", "абзац")):
            return f"{base}/a{number}"
    return None


def extract_refs(
    text: str,
    *,
    article_id: str,
    clause: str | None = None,
    part: int = 1,
) -> list[Reference]:
    """Ссылки из текста одного пункта, без повторов, в порядке появления."""
    found: dict[tuple[str, str], Reference] = {}

    # 1. Относительные: «пунктом 8.1 настоящей статьи».
    for match in _RELATIVE_RE.finditer(text):
        numbers = re.split(r"\s*(?:,|и)\s*", match.group("nums"))
        for number in numbers:
            target = _relative_target(
                match.group("kind"), number.strip(), match.group("scope"), article_id, clause
            )
            if target and target != article_id:
                reference = Reference(target, classify(text, match.start()), match.group(0))
                found.setdefault((target, reference.type), reference)

    # 2. Абсолютные: «статьёй 93.1 настоящего Кодекса», «статьи 129.1».
    for citation in find_citations(text, default_code="nk"):
        if citation.code != "nk":
            continue
        target = citation.norm_unit_id
        if target.startswith(article_id) and target != article_id:
            pass  # ссылка внутрь себя — оставляем, это уточнение пункта
        elif target == article_id:
            continue  # ссылка на саму себя смысла не несёт
        reference = Reference(target, classify(text, citation.span[0]), citation.raw_text)
        found.setdefault((target, reference.type), reference)

    return list(found.values())
