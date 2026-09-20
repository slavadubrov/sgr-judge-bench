"""Explicit staged CLI. No key presence ever automatically enables model calls."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .core import read_jsonl
from .datasets import prepare, prepare_probes
from .report import report
from .runner import canaries, execute, execute_cascade, fit_and_freeze, load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="config/models.json")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download pinned public datasets; no model calls")
    p.add_argument("--out", default="data")
    p = sub.add_parser(
        "probes", help="Freeze robustness candidates; transformed evidence requires human review"
    )
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data/probes.jsonl")
    for command in ("preflight", "run", "cascade"):
        p = sub.add_parser(command)
        p.add_argument(
            "--live",
            action="store_true",
            required=True,
            help="Explicitly authorize this provider run",
        )
        p.add_argument("--out", required=True)
        p.add_argument("--budget-usd", type=float, required=True)
        p.add_argument(
            "--location",
            required=True,
            help="Actual EU client location/network; do not infer from timezone",
        )
        p.add_argument("--holiday-status", choices=["holiday", "not-holiday"])
        if command != "preflight":
            p.add_argument("--data", default="data")
            p.add_argument("--freeze")
        if command == "run":
            p.add_argument(
                "--phase",
                choices=["development", "calibration", "routing", "test", "repeat", "robustness"],
                required=True,
            )
            p.add_argument("--preflight", required=True)
            p.add_argument("--cache", choices=["off", "reuse"], default="off")
            p.add_argument("--block", type=int, choices=[1, 2, 3])
            p.add_argument("--probe-file", default="data/probes.jsonl")
    p = sub.add_parser("experiment", help="Run all stages under one shared spend ledger")
    p.add_argument("--live", action="store_true", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--preflight", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--prior-spend-usd", type=float, default=0)
    p.add_argument("--location", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--data", default="data")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--probes")
    p.add_argument("--out", default="runs/frozen.json")
    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--freeze")
    p.add_argument("--bootstrap", type=int, default=10000)
    p = sub.add_parser(
        "demo", help="Offline article demo from bundled recorded evidence; no API calls"
    )
    p.add_argument("--out", default="work/article-demo")
    p = sub.add_parser("tabfact-prepare", help="Freeze a fresh TabFact test sample")
    p.add_argument("--upstream", required=True)
    p.add_argument("--exclude", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--simple-per-label", type=int, default=25)
    p.add_argument("--complex-per-label", type=int, default=35)
    p = sub.add_parser("tabfact-run", help="Direct, frozen SGR v2, and native Jev hybrid")
    p.add_argument("--live", action="store_true", required=True)
    p.add_argument("--cases")
    p.add_argument("--canaries", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--planner", default="terra")
    p.add_argument(
        "--include-short-direct",
        "--include-prompt-control",
        dest="include_prompt_control",
        action="store_true",
        help="Also run the original shorter direct prompt as a diagnostic control",
    )
    p.add_argument(
        "--exclude-hybrid", action="store_true", help="Native Jev plus direct/SGR LLM arms only"
    )
    p = sub.add_parser("tabfact-report", help="Offline replay of recorded table results")
    p.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import build

        print(build(args.out))
        return
    if args.command == "tabfact-prepare":
        from .tabfact_data import prepare as prepare_tables

        prepare_tables(
            Path(args.upstream),
            Path(args.out),
            args.exclude,
            args.seed,
            args.simple_per_label,
            args.complex_per_label,
        )
        return
    if args.command == "tabfact-report":
        from .tabfact import report as table_report

        print(json.dumps(table_report(args.run), indent=2))
        return
    if args.command == "tabfact-run":
        from .tabfact import canaries as table_canaries
        from .tabfact import run as table_run

        if bool(args.cases) == args.canaries:
            parser.error("Choose exactly one of --cases and --canaries")
        load_dotenv(Path.cwd() / ".env", override=False)
        cases = table_canaries() if args.canaries else json.loads(Path(args.cases).read_text())
        asyncio.run(
            table_run(
                cases,
                load_models(args.models),
                args.out,
                args.budget_usd,
                planner=args.planner,
                include_hybrid=not args.exclude_hybrid,
                include_prompt_control=args.include_prompt_control,
                phase="preflight" if args.canaries else "test",
            )
        )
        return
    if args.command == "prepare":
        print(json.dumps(prepare(args.out), indent=2))
        return
    if args.command == "probes":
        print(prepare_probes(args.data, args.out))
        return
    if args.command == "report":
        frozen = json.loads(Path(args.freeze).read_text()) if args.freeze else None
        print(report(args.runs, args.out, frozen=frozen, resamples=args.bootstrap))
        return
    models = load_models(args.models)
    if args.command == "freeze":
        result = fit_and_freeze(args.data, models, args.runs, args.out, args.probes)
        print(json.dumps({k: v["status"] for k, v in result["policies"].items()}, indent=2))
        return
    if args.budget_usd <= 0:
        parser.error("budget must be positive")
    load_dotenv(Path.cwd() / ".env", override=False)
    if args.command == "experiment":
        from .experiment import experiment

        print(
            asyncio.run(
                experiment(
                    args.data,
                    models,
                    args.preflight,
                    args.out,
                    args.budget_usd,
                    args.prior_spend_usd,
                    args.location,
                )
            )
        )
        return
    holiday = {None: None, "holiday": True, "not-holiday": False}[args.holiday_status]
    if args.command == "preflight":
        asyncio.run(
            execute(
                canaries(),
                models,
                args.out,
                phase="preflight",
                budget=args.budget_usd,
                location=args.location,
                holiday_status=holiday,
            )
        )
        return
    frozen = json.loads(Path(args.freeze).read_text()) if args.freeze else None
    cases = read_jsonl(Path(args.data) / "cases.jsonl")
    if args.command == "cascade":
        if not frozen:
            parser.error("cascade requires --freeze")
        asyncio.run(
            execute_cascade(
                [c for c in cases if c["split"] == "test"],
                models,
                frozen,
                args.out,
                budget=args.budget_usd,
                location=args.location,
                holiday_status=holiday,
            )
        )
        return
    if args.phase == "repeat" and not args.block:
        parser.error("repeat requires an explicit --block 1, 2 or 3, executed at separate times")
    subset = json.loads((Path(args.data) / "subsets.json").read_text())
    if args.phase == "robustness":
        cases = read_jsonl(args.probe_file)
    else:
        split = "test" if args.phase == "repeat" else args.phase
        cases = [c for c in cases if c["split"] == split]
    admission = json.loads(Path(args.preflight).read_text())
    asyncio.run(
        execute(
            cases,
            models,
            args.out,
            phase=args.phase,
            budget=args.budget_usd,
            location=args.location,
            preflight=admission,
            cache_mode=args.cache,
            block=args.block,
            repeat_ids=subset["repeat_ids"],
            holiday_status=holiday,
            frozen=frozen,
        )
    )


if __name__ == "__main__":
    main()
