"""Точки входа этапов. Каждый этап — вход и выход в файлах, чтобы прогон можно
было перенести на другую машину без переписывания кода."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from p3.config import NALOG, PRAVO, PUBLICATION, SILVER_DIR
from p3.fetch import PoliteFetcher
from p3.netcheck import PreflightError, preflight
from p3.raw_store import RawStore
from p3.silver.build import build, summarize, write_jsonl
from p3.silver.collect import collect_decisions, collect_letters
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

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
