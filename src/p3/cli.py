"""Точки входа этапов. Каждый этап — вход и выход в файлах, чтобы прогон можно
было перенести на другую машину без переписывания кода."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from p3.config import DATA_DIR, NALOG, PRAVO, PUBLICATION, SILVER_DIR
from p3.fetch import PoliteFetcher
from p3.netcheck import PreflightError, preflight
from p3.raw_store import RawStore
from p3.silver.build import build, summarize, write_jsonl
from p3.silver.collect import collect_decisions, collect_letters
from p3.nk.chunk import chunk_articles
from p3.nk.parse import parse_articles
from p3.silver.layer1 import build_layer1_pairs, verify as verify_layer1
from p3.sources.ips import NK_PART_1, NK_PART_2, IpsClient


def _fetcher(args) -> PoliteFetcher:
    # Страна выхода определяется один раз за прогон и пишется в манифест рядом
    # с каждым документом: происхождение корпуса должно быть прослеживаемым.
    from p3.netcheck import exit_country

    country = None if args.offline else exit_country()
    return PoliteFetcher(RawStore(), offline=args.offline, exit_country=country)


def _require_network(sources) -> None:
    try:
        preflight(sources)
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


def cmd_netcheck(args) -> None:
    """Показать, годится ли среда для сбора, и почему."""
    from p3.netcheck import PreflightError, preflight, tcp_reachable

    sources = [PRAVO, PUBLICATION, NALOG]
    for source in sources:
        tls = tcp_reachable(source.host, 443)
        http = tcp_reachable(source.host, 80)
        required = "нужен" if source.require_tls else "не нужен"
        print(
            f"  {source.host:32} 443={'открыт' if tls else 'закрыт':7} "
            f"80={'открыт' if http else 'закрыт':7} (TLS {required})"
        )

    try:
        report = preflight(sources)
    except PreflightError as exc:
        print(f"\n{exc}")
        raise SystemExit(1)

    print(f"\nстрана выхода: {report.exit_country or 'не определена'}")
    if report.country_warning:
        print(f"  предупреждение: {report.country_warning}")
    print("\nСреда годится для сбора.")


def cmd_spike_ips(args) -> None:
    if not args.offline:
        _require_network([PRAVO])
    client = IpsClient(_fetcher(args))
    editions = client.editions(NK_PART_1.nd)
    recent = [e for e in editions if e.year and e.year >= args.since]
    print(json.dumps({
        "nd": NK_PART_1.nd,
        "title": NK_PART_1.title,
        "editions_listed": len(editions),
        "max_rdk": max((e.rdk for e in editions), default=None),
        "editions_since": {str(args.since): len(recent)},
        "years": sorted({e.year for e in editions if e.year})[:3] + ["..."] +
                 sorted({e.year for e in editions if e.year})[-3:],
    }, ensure_ascii=False, indent=2))


def cmd_collect_fns(args) -> None:
    if not args.offline:
        _require_network([NALOG])
    decisions = collect_decisions(_fetcher(args), max_pages=args.max_pages)
    print(f"собрано решений: {len(decisions)}")


def cmd_collect_all(args) -> None:
    """Весь сетевой сбор одной командой.

    Нужна, потому что окно прямого подключения короткое: пользователь выключает
    VPN, запускает это, включает обратно. Всё дальнейшее считается из кэша.
    Команда идемпотентна — повторный запуск берёт уже скачанное из raw_store.
    """
    _require_network([PRAVO, NALOG])
    fetcher = _fetcher(args)
    problems: list[str] = []

    print("1/2  карточки редакций НК из ИПС...")
    client = IpsClient(fetcher)
    for document in (NK_PART_1, NK_PART_2):
        try:
            editions = client.editions(document.nd)
            years = sorted({e.year for e in editions if e.year})
            span = f"{years[0]}-{years[-1]}" if years else "нет дат"
            print(f"     {document.title}: редакций {len(editions)}, годы {span}")
        except Exception as exc:  # noqa: BLE001 — сообщаем и продолжаем
            problems.append(f"ИПС {document.nd}: {exc}")
            print(f"     ОШИБКА {document.title}: {exc}")

    print("2/2  решения по жалобам ФНС...")
    try:
        decisions = collect_decisions(fetcher, max_pages=args.max_pages)
        print(f"     собрано решений: {len(decisions)}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ФНС: {exc}")
        print(f"     ОШИБКА: {exc}")
        decisions = []

    store = fetcher.store
    print(f"\nв кэше документов: {len(store)}")
    if problems:
        print("\nБЫЛИ ОШИБКИ:")
        for problem in problems:
            print(f"  - {problem}")
        print("Команду можно запустить повторно — уже скачанное берётся из кэша.")
        raise SystemExit(1)
    print("Готово. VPN можно включать обратно.")


def cmd_collect_letters(args) -> None:
    if not args.offline:
        _require_network([NALOG])
    letters = collect_letters(_fetcher(args), max_pages=args.max_pages)
    usable = [x for x in letters if not x.is_administrative and x.article_labels]
    print(f"собрано писем: {len(letters)}; годных для silver set: {len(usable)}")


def cmd_build_silver(args) -> None:
    # Сборка идёт из кэша: сеть здесь не нужна и не должна использоваться.
    fetcher = PoliteFetcher(RawStore(), offline=True)
    decisions = collect_decisions(fetcher, max_pages=args.max_pages)
    try:
        letters = collect_letters(fetcher, max_pages=args.max_pages)
    except Exception as exc:  # noqa: BLE001 — писем может не быть в кэше
        print(f"письма не прочитаны из кэша ({exc}); собираю только по решениям")
        letters = []
    splits = build(decisions, letters, seed=args.seed)

    out = args.out
    for name, pairs in splits.items():
        write_jsonl(pairs, out / f"{name}.jsonl")
    report = summarize(splits)
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_build_chunks(args) -> None:
    """ETL: разобранный НК ч.1 → чанки по спеке чанкинга."""
    client = IpsClient(PoliteFetcher(RawStore(), offline=True))
    editions = client.editions(NK_PART_1.nd)
    rdk = args.rdk if args.rdk is not None else max(e.rdk for e in editions)
    label = next((e.label for e in editions if e.rdk == rdk), None)

    articles = parse_articles(client.full_text_html(NK_PART_1.nd, rdk), part=1)
    chunks = chunk_articles(articles, version_no=rdk, amended_by=label)

    out = args.out / "nk1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.as_payload(), ensure_ascii=False, sort_keys=True) + "\n")

    refs = sum(len(c.refs_out) for c in chunks)
    print(json.dumps({
        "edition": {"rdk": rdk, "label": label},
        "articles": len(articles),
        "chunks": len(chunks),
        "refs_out": refs,
        "repealed_chunks": sum(1 for c in chunks if c.status == "repealed"),
        "out": str(out),
    }, ensure_ascii=False, indent=2))


def cmd_build_embeddings(args) -> None:
    """Векторизовать чанки выбранным энкодером."""
    import time

    from p3.search.dense import DEFAULT_ENCODER, DenseEncoder, DenseIndex

    path = args.chunks / args.corpus
    chunks = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"чанков: {len(chunks)}; энкодер: {args.encoder}", flush=True)

    encoder = DenseEncoder(name=args.encoder, batch_size=args.batch_size)
    texts = [c["text_dense"] for c in chunks]

    started = time.time()
    vectors = encoder.encode_documents(texts, progress_every=max(200, len(texts) // 20))
    elapsed = time.time() - started

    index = DenseIndex(doc_ids=[c["chunk_id"] for c in chunks], vectors=vectors)
    stem = args.corpus.replace(".jsonl", "")
    out = args.chunks / (f"dense-{args.encoder}.npz" if stem == "nk1"
                         else f"dense-{stem}-{args.encoder}.npz")
    index.save(out)
    print(json.dumps({
        "encoder": args.encoder,
        "device": encoder.device,
        "vectors": list(vectors.shape),
        "seconds": round(elapsed, 1),
        "chunks_per_second": round(len(chunks) / elapsed, 1) if elapsed else None,
        "out": str(out),
    }, ensure_ascii=False, indent=2))


def cmd_evaluate(args) -> None:
    """Таблица четырёх конфигураций из ТЗ §7 с разрезами."""
    from p3.search.evaluate import by_slice, evaluate, load_queries, reachable
    from p3.search.pipeline import (
        Retrieval, expansion_contribution, indexed_articles, to_articles, to_norms,
    )

    retrieval = Retrieval.load(args.chunks, encoder_name=args.encoder, corpus=args.corpus)
    available = indexed_articles(retrieval.store)

    def read(path):
        return load_queries(json.loads(line) for line in path.open(encoding="utf-8"))

    sets = {}
    silver = args.silver / f"{args.split}.jsonl"
    if silver.exists():
        sets["silver_" + args.split] = (read(silver), "article")
    layer1 = args.silver / "layer1.jsonl"
    if layer1.exists():
        sets["layer1"] = (read(layer1), "norm")

    report = {"encoder": args.encoder, "indexed_articles": len(available), "sets": {}}

    for name, (queries, level) in sets.items():
        usable = reachable(queries, available)
        entry = {
            "queries_total": len(queries),
            "queries_reachable": len(usable),
            "reachable_share": round(len(usable) / len(queries), 3) if queries else 0.0,
            "configurations": {},
        }
        for mode in ("dense", "bm25", "hybrid", "hybrid+rerank"):
            if mode in ("dense", "hybrid", "hybrid+rerank") and retrieval.dense is None:
                continue

            def run(query, mode=mode, level=level):
                hits = retrieval.run(query.query, mode=mode, expand=args.expand)
                return to_articles(hits, retrieval.store) if level == "article" \
                    else to_norms(hits, retrieval.store)

            indexed = available if level == "article" else {
                c["norm_unit_id"] for c in retrieval.store.by_chunk_id.values()
            }
            entry["configurations"][mode] = evaluate(
                usable, run, level=level, indexed=indexed
            ).as_dict()
            if mode == "hybrid":
                entry["by_query_type"] = by_slice(
                    usable, run, lambda q: q.query_type, level=level, indexed=indexed
                )
                if retrieval.dense is not None:
                    entry["one_hop_expansion"] = expansion_contribution(
                        retrieval, usable, level=level
                    )
        report["sets"][name] = entry

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


#: Редакции ч.1 для темпорального среза. Отобраны по дате (rdk не хронологичен)
#: и только из диапазона, где ИПС отдаёт современную вёрстку — примерно с 2019.
TEMPORAL_EDITIONS = (144, 212, 197, 236, 248)


def cmd_build_temporal(args) -> None:
    """ETL с версиями: несколько редакций ч.1 → чанки с интервалами действия."""
    from p3.nk.temporal import (
        EditionSnapshot, apply_temporal_fields, build_versions, deduplicate,
        parse_ru_date, summarize as summarize_versions,
    )

    client = IpsClient(PoliteFetcher(RawStore(), offline=True))
    editions = {e.rdk: e for e in client.editions(NK_PART_1.nd)}
    rdks = [int(x) for x in args.editions.split(",")] if args.editions else list(TEMPORAL_EDITIONS)

    parsed, snapshots = {}, []
    for rdk in rdks:
        articles = parse_articles(client.full_text_html(NK_PART_1.nd, rdk), part=1)
        parsed[rdk] = articles
        snapshots.append(EditionSnapshot(
            rdk=rdk,
            edition_date=parse_ru_date(editions[rdk].date),
            texts={c.norm_unit_id(a.norm_unit_id): c.text for a in articles for c in a.clauses},
        ))
        print(f"  rdk={rdk} {editions[rdk].date}: статей {len(articles)}")

    versions = build_versions(snapshots)

    all_chunks: list[dict] = []
    for rdk in rdks:
        chunks = [c.as_payload() for c in chunk_articles(
            parsed[rdk], version_no=rdk, amended_by=editions[rdk].label
        )]
        all_chunks.extend(apply_temporal_fields(chunks, versions, rdk))
    unique = deduplicate(all_chunks)

    out = args.out / "nk1-temporal.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in sorted(unique, key=lambda c: c["chunk_id"]):
            fh.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "editions": rdks,
        "versions": summarize_versions(versions),
        "chunks_before_dedup": len(all_chunks),
        "chunks": len(unique),
        "out": str(out),
    }, ensure_ascii=False, indent=2))


def cmd_evaluate_temporal(args) -> None:
    """Синтетический темпоральный тест: точность выбора редакции и цена фильтра."""
    from p3.nk.temporal import (
        EditionSnapshot, build_versions, parse_ru_date, to_int_date,
    )
    from p3.search.evaluate import evaluate, load_queries
    from p3.search.pipeline import Retrieval, to_norms, to_versions
    from p3.silver.temporal_set import build_temporal_cases, temporal_accuracy
    import datetime

    client = IpsClient(PoliteFetcher(RawStore(), offline=True))
    editions = {e.rdk: e for e in client.editions(NK_PART_1.nd)}
    rdks = [int(x) for x in args.editions.split(",")] if args.editions else list(TEMPORAL_EDITIONS)

    snapshots = []
    for rdk in rdks:
        articles = parse_articles(client.full_text_html(NK_PART_1.nd, rdk), part=1)
        snapshots.append(EditionSnapshot(
            rdk=rdk,
            edition_date=parse_ru_date(editions[rdk].date),
            texts={c.norm_unit_id(a.norm_unit_id): c.text for a in articles for c in a.clauses},
        ))
    versions = build_versions(snapshots)

    retrieval = Retrieval.load(
        args.chunks, encoder_name=args.encoder, corpus="nk1-temporal.jsonl"
    )
    if retrieval.dense is None:
        print("нет векторов темпорального корпуса — считаю только по BM25")

    cases = build_temporal_cases(versions)
    print(f"темпоральных запросов: {len(cases)} по {len({c.pair.source_id.split('@')[0] for c in cases})} нормам")

    retrieved: dict[str, list[str]] = {}
    for case in cases:
        hits = retrieval.run(case.pair.query, mode=args.mode, as_of=case.as_of)
        retrieved[case.pair.query_id] = to_versions(hits, retrieval.store)
    accuracy = temporal_accuracy(cases, retrieved)

    # Цена фильтра: recall на слое 1 без даты и с датой «сегодня».
    layer1 = load_queries(
        json.loads(line) for line in (args.silver / "layer1.jsonl").open(encoding="utf-8")
    )
    today = to_int_date(datetime.date.today())

    # Базовая линия — индекс одной редакции. Сравнивать с нефильтрованным
    # индексом всех версий нельзя: там одна норма представлена несколькими
    # версиями, любая засчитывается как попадание, и у поиска просто больше
    # попыток. Такой «эталон» вдобавок выдаёт устаревшие редакции, то есть не
    # является системой, которую кто-то стал бы поставлять.
    baseline_retrieval = Retrieval.load(args.chunks, encoder_name=args.encoder)

    def run_on(source, as_of=None):
        return lambda query: to_norms(
            source.run(query.query, mode=args.mode, as_of=as_of), source.store
        )

    baseline = evaluate(layer1, run_on(baseline_retrieval), level="norm").as_dict()
    unfiltered = evaluate(layer1, run_on(retrieval), level="norm").as_dict()
    with_filter = evaluate(layer1, run_on(retrieval, today), level="norm").as_dict()
    drop = round((baseline["recall@20"] - with_filter["recall@20"]) * 100, 2)
    duplicate_bonus = round((unfiltered["recall@20"] - with_filter["recall@20"]) * 100, 2)

    report = {
        "mode": args.mode,
        "editions": rdks,
        "temporal_accuracy": accuracy,
        "filter_cost": {
            "recall@20_single_edition_baseline": baseline["recall@20"],
            "recall@20_all_versions_with_filter": with_filter["recall@20"],
            "drop_vs_baseline_pp": drop,
            "criterion_max_drop_pp": 5.0,
            "criterion_met": drop <= 5.0,
            # Справочно: насколько нефильтрованный индекс всех версий выглядит
            # лучше за счёт дублей. Базой сравнения это быть не может.
            "recall@20_all_versions_unfiltered": unfiltered["recall@20"],
            "duplicate_version_bonus_pp": duplicate_bonus,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_render_results(args) -> None:
    """Собрать раздел результатов из файлов замеров."""
    from p3.report import render

    text = render(args.eval, args.temporal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_build_layer1(args) -> None:
    """Синтетический набор слоя 1 из разобранного текста НК ч.1."""
    fetcher = PoliteFetcher(RawStore(), offline=True)
    client = IpsClient(fetcher)
    rdk = args.rdk
    if rdk is None:
        rdk = max(e.rdk for e in client.editions(NK_PART_1.nd))
    articles = parse_articles(client.full_text_html(NK_PART_1.nd, rdk), part=1)
    print(f"разобрано статей: {len(articles)} (редакция rdk={rdk})")

    problems = verify_layer1(articles)
    if problems:
        print("набор не сходится с корпусом:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    pairs = build_layer1_pairs(articles)
    write_jsonl(pairs, args.out / "layer1.jsonl")
    by_type: dict[str, int] = {}
    for pair in pairs:
        by_type[pair.query_type] = by_type.get(pair.query_type, 0) + 1
    print(json.dumps({
        "pairs": len(pairs),
        "by_query_type": by_type,
        "articles_covered": sorted({a for p in pairs for a in p.gold_article_ids}),
    }, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="p3")
    parser.add_argument("--offline", action="store_true", help="только из raw_store, без сети")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str):
        """Подкоманда, принимающая -v и после себя, а не только до."""
        child = sub.add_parser(name, help=help_)
        child.add_argument("-v", "--verbose", action="store_true")
        return child

    check = add("netcheck", "годится ли сеть для сбора")
    check.set_defaults(func=cmd_netcheck)

    spike = add("spike-ips", "сводка по редакциям НК ч.1 в ИПС")
    spike.add_argument("--since", type=int, default=2021)
    spike.set_defaults(func=cmd_spike_ips)

    collect = add("collect-fns", "собрать решения по жалобам ФНС")
    collect.add_argument("--max-pages", type=int, default=200)
    collect.set_defaults(func=cmd_collect_fns)

    collect_all = add("collect-all", "весь сетевой сбор одной командой")
    collect_all.add_argument("--max-pages", type=int, default=200)
    collect_all.set_defaults(func=cmd_collect_all)

    letters = add("collect-letters", "собрать письма ФНС, обязательные для применения")
    letters.add_argument("--max-pages", type=int, default=200)
    letters.set_defaults(func=cmd_collect_letters)

    build_cmd = add("build-silver", "собрать silver set из кэша")
    build_cmd.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=SILVER_DIR)
    build_cmd.add_argument("--seed", type=int, default=42)
    build_cmd.add_argument("--max-pages", type=int, default=200)
    build_cmd.set_defaults(func=cmd_build_silver)

    chunks_cmd = add("build-chunks", "ETL: НК ч.1 → чанки")
    chunks_cmd.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "chunks")
    chunks_cmd.add_argument("--rdk", type=int, default=None)
    chunks_cmd.set_defaults(func=cmd_build_chunks)

    emb = add("build-embeddings", "векторизовать чанки")
    emb.add_argument("--chunks", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "chunks")
    emb.add_argument("--encoder", default=None)
    emb.add_argument("--batch-size", type=int, default=8)
    emb.add_argument("--corpus", default="nk1.jsonl")
    emb.set_defaults(func=cmd_build_embeddings)

    ev = add("evaluate", "таблица конфигураций поиска")
    ev.add_argument("--chunks", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "chunks")
    ev.add_argument("--silver", type=lambda p: __import__("pathlib").Path(p), default=SILVER_DIR)
    ev.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ev.add_argument("--encoder", default=None)
    ev.add_argument("--expand", action="store_true", help="включить one-hop раскрытие ссылок")
    ev.add_argument("--corpus", default="nk1.jsonl")
    ev.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "eval" / "report.json")
    ev.set_defaults(func=cmd_evaluate)

    temporal = add("build-temporal", "ETL с версиями: редакции ч.1 → чанки с интервалами")
    temporal.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "chunks")
    temporal.add_argument("--editions", default=None, help="список rdk через запятую")
    temporal.set_defaults(func=cmd_build_temporal)

    tev = add("evaluate-temporal", "синтетический темпоральный тест")
    tev.add_argument("--chunks", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "chunks")
    tev.add_argument("--silver", type=lambda p: __import__("pathlib").Path(p), default=SILVER_DIR)
    tev.add_argument("--encoder", default=None)
    tev.add_argument("--editions", default=None)
    tev.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    tev.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "eval" / "temporal.json")
    tev.set_defaults(func=cmd_evaluate_temporal)

    rr = add("render-results", "собрать раздел результатов из замеров")
    rr.add_argument("--eval", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "eval" / "test.json")
    rr.add_argument("--temporal", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "eval" / "temporal.json")
    rr.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=DATA_DIR / "eval" / "RESULTS.md")
    rr.set_defaults(func=cmd_render_results)

    layer1 = add("build-layer1", "синтетический набор слоя 1 из текста НК")
    layer1.add_argument("--out", type=lambda p: __import__("pathlib").Path(p), default=SILVER_DIR)
    layer1.add_argument("--rdk", type=int, default=None, help="редакция; по умолчанию последняя")
    layer1.set_defaults(func=cmd_build_layer1)

    args = parser.parse_args(argv)
    if getattr(args, "encoder", "sentinel") is None:
        from p3.search.dense import DEFAULT_ENCODER

        args.encoder = DEFAULT_ENCODER
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
