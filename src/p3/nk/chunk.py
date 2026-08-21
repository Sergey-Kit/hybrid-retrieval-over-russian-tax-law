"""Чанкинг НК по спеке чанкинга §2–3 и §5.

Единица чанка — **пункт статьи**: она совпадает с единицей цитирования в
требованиях ИФНС («в порядке п. 3 ст. 88 НК РФ») и самодостаточна по смыслу.
Два исключения из спеки: пункт длиннее `MAX_TOKENS` дробится по подпунктам,
статья короче `MIN_TOKENS` или без пунктов идёт одним чанком.

Главное, что здесь легко испортить, — **префикс**:

* в `text_dense` он нужен целиком: даёт эмбеддингу тему («камеральная проверка»),
  которой в теле пункта может не быть ни разу;
* в `text_bm25` его быть не должно: иначе все чанки статьи 88 получают одинаковые
  токены «камеральный», «проверка», «88», IDF по ним падает, и ранжирование
  внутри статьи становится случайным.

Поэтому поля разделены: `text_dense`, `text_bm25` (только тело) и `title_bm25`
(заголовок статьи + ссылка на пункт, вес ~0.3).

## О датах — почему они пустые

Спека §5 называет четыре несовпадающие даты («главная ловушка»): `published_at`,
`valid_from`, `valid_to` и `applies_from`, причём основной фильтр — последняя,
и она бывает ретроспективной. Из ИПС мы знаем только **дату изменяющего акта**
из подписи редакции. Она не равна ни одной из четырёх: вступление в силу задаётся
самим актом и может отстоять на месяцы.

Заполнить `valid_from` датой редакции было бы тем самым тихим дефектом: фильтр
заработает, метрики выйдут правдоподобные, а редакция будет выбираться неверная.
Поэтому даты остаются `None`, а `version_no` и `amended_by` — заполненными.
Извлечение реальных дат — работа вехи 3, вместе с консолидатором.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from p3.nk.parse import Article, Clause
from p3.nk.refs import Reference, extract_refs

#: Пороги из спеки §2.
MAX_TOKENS = 700
MIN_TOKENS = 120

#: Sentinel вместо None: фильтр по диапазону однороднее, чем проверка на null.
SENTINEL_VALID_TO = 99991231

PREFIX_SEPARATOR = "\n———\n"
CODE_NAME = "НК РФ"
PART_NAMES = {1: "часть первая", 2: "часть вторая"}

#: Подпункт в начале абзаца: «1) …», «2.1) …».
_SUBCLAUSE_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\)\s+", re.M)

#: Оценка числа токенов, когда настоящий токенизатор не передан.
#: На юридическом русском XLM-R даёт ~1.85 токена на слово; берём с запасом.
_TOKENS_PER_WORD = 1.9


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKENS_PER_WORD) + 1


@dataclass
class Chunk:
    """Запись индекса. Поля — по схеме payload из спеки §5."""

    chunk_id: str
    norm_unit_id: str
    version_id: str

    code: str
    part: int
    section: str | None
    chapter: str | None
    article: str
    article_title: str
    clause: str | None
    subclause: str | None

    text_dense: str
    text_bm25: str
    title_bm25: str

    refs_out: list[dict] = field(default_factory=list)
    para_offsets: list[list[int]] = field(default_factory=list)

    status: str = "active"
    source_type: str = "norm"
    authority: float = 1.0

    version_no: int | None = None
    amended_by: str | None = None

    #: Даты намеренно не заполняются, см. модульный docstring.
    valid_from: int | None = None
    valid_to: int = SENTINEL_VALID_TO
    applies_from: int | None = None
    published_at: int | None = None

    def as_payload(self) -> dict:
        return asdict(self)


def _hierarchy_prefix(article: Article, clause_label: str | None) -> str:
    lines = [f"Налоговый кодекс Российской Федерации, {PART_NAMES.get(article.part, '')}".strip()]
    if article.section:
        lines.append(f"Раздел {article.section}. {article.section_title or ''}".rstrip(". "))
    if article.chapter:
        lines.append(f"Глава {article.chapter}. {article.chapter_title or ''}".rstrip(". "))
    lines.append(f"Статья {article.number}. {article.title}".rstrip(". "))
    if clause_label:
        lines.append(clause_label)
    return "\n".join(lines)


def _paragraph_offsets(text: str) -> list[list[int]]:
    """Границы абзацев внутри чанка.

    Абзацы в тексте не нумерованы, но нумеруются в ссылках и в требованиях ИФНС,
    поэтому смещения хранятся: по ним восстанавливается «абзац второй пункта 3».
    """
    offsets: list[list[int]] = []
    position = 0
    for line in text.split("\n"):
        end = position + len(line)
        if line.strip():
            offsets.append([position, end])
        position = end + 1
    return offsets


def _split_paragraphs(text: str, token_count: Callable[[str], int]) -> list[str]:
    """Дробление по абзацам — запасной вариант для длинного пункта без подпунктов.

    Режем только по границам абзацев: спека запрещает разрывать предложение,
    перечисление от вводной фразы и условие от последствия. Поэтому одиночный
    абзац длиннее порога остаётся целым — это осознанный выбор в пользу
    целостности нормы. На части первой НК такой случай не встречается.
    """
    pieces: list[str] = []
    current: list[str] = []
    for paragraph in text.split("\n"):
        candidate = current + [paragraph]
        if current and token_count("\n".join(candidate)) > MAX_TOKENS:
            pieces.append("\n".join(current))
            current = [paragraph]
        else:
            current = candidate
    if current:
        pieces.append("\n".join(current))
    return pieces


def _split_subclauses(text: str) -> list[tuple[str | None, str]]:
    """Разбить длинный пункт на вводную часть и подпункты."""
    matches = list(_SUBCLAUSE_RE.finditer(text))
    if not matches:
        return [(None, text)]
    intro = text[: matches[0].start()].strip()
    pieces: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            # Подпункт без вводной фразы пункта — обрывок предложения (спека §2).
            merged = f"{intro}\n{match.group('number')}) {body}" if intro else body
            pieces.append((match.group("number"), merged))
    return pieces or [(None, text)]


def _build(
    article: Article,
    clause: Clause | None,
    subclause: str | None,
    body: str,
    ordinal: int,
    version_no: int | None,
    amended_by: str | None,
    refs: list[Reference],
) -> Chunk:
    clause_number = clause.number if clause else None
    norm_unit_id = clause.norm_unit_id(article.norm_unit_id) if clause else article.norm_unit_id
    if subclause:
        norm_unit_id = f"{norm_unit_id}/sp{subclause}"
    version_id = f"{norm_unit_id}@r{version_no}" if version_no is not None else norm_unit_id

    clause_label = None
    if clause_number:
        clause_label = f"Пункт {clause_number}"
        if subclause:
            clause_label += f", подпункт {subclause}"

    prefix = _hierarchy_prefix(article, clause_label)
    reference = f"Статья {article.number}"
    if clause_number:
        reference += f", пункт {clause_number}"

    repealed = (clause.is_repealed if clause else article.is_repealed) or article.is_repealed

    return Chunk(
        chunk_id=f"{version_id}#{ordinal}",
        norm_unit_id=norm_unit_id,
        version_id=version_id,
        code=CODE_NAME,
        part=article.part,
        section=article.section,
        chapter=article.chapter,
        article=article.number,
        article_title=article.title,
        clause=clause_number,
        subclause=subclause,
        text_dense=prefix + PREFIX_SEPARATOR + body,
        text_bm25=body,
        title_bm25=f"{article.title} {reference}".strip(),
        refs_out=[r.as_payload() for r in refs],
        para_offsets=_paragraph_offsets(body),
        status="repealed" if repealed else "active",
        version_no=version_no,
        amended_by=amended_by,
    )


def chunk_article(
    article: Article,
    *,
    version_no: int | None = None,
    amended_by: str | None = None,
    token_count: Callable[[str], int] = estimate_tokens,
) -> list[Chunk]:
    """Чанки одной статьи по правилам спеки §2."""
    whole = article.text
    # Короткая статья или статья без пунктов идёт целиком.
    if not article.clauses or (
        len(article.clauses) == 1 and article.clauses[0].number is None
    ) or token_count(whole) < MIN_TOKENS:
        refs = extract_refs(whole, article_id=article.norm_unit_id, part=article.part)
        # Даже цельная статья может перерасти порог — тогда режем по абзацам.
        pieces = [whole] if token_count(whole) <= MAX_TOKENS else _split_paragraphs(whole, token_count)
        return [
            _build(article, None, None, piece, ordinal, version_no, amended_by, refs)
            for ordinal, piece in enumerate(pieces)
        ]

    chunks: list[Chunk] = []
    for clause in article.clauses:
        refs = extract_refs(
            clause.text, article_id=article.norm_unit_id, clause=clause.number, part=article.part
        )
        if token_count(clause.text) <= MAX_TOKENS:
            chunks.append(_build(article, clause, None, clause.text, 0, version_no, amended_by, refs))
            continue
        # Длинный пункт дробится по подпунктам, каждый наследует вводную фразу.
        ordinal = 0
        for subclause, body in _split_subclauses(clause.text):
            # Подпунктов может не быть вовсе, либо часть всё равно останется
            # длинной — тогда режем по абзацам, не трогая предложения.
            for piece in (
                [body] if token_count(body) <= MAX_TOKENS else _split_paragraphs(body, token_count)
            ):
                chunks.append(
                    _build(article, clause, subclause, piece, ordinal, version_no, amended_by, refs)
                )
                ordinal += 1
    return chunks


def chunk_articles(
    articles: list[Article],
    *,
    version_no: int | None = None,
    amended_by: str | None = None,
    token_count: Callable[[str], int] = estimate_tokens,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(
            chunk_article(
                article, version_no=version_no, amended_by=amended_by, token_count=token_count
            )
        )
    return chunks
