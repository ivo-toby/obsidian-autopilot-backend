"""Command line interface for repeatable meeting-summary benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import load_benchmark_config
from .generation import GenerationFilters, generate_candidates
from .judging import judge_generations, judge_pairwise_top_models
from .pi_rpc import PiRpcClient
from .reporting import build_report, write_report
from .storage import RunStore


class CliError(RuntimeError):
    """Raised for an invalid benchmark command or incomplete run."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.meeting_summary",
        description="Run the meeting-summary model benchmark.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="validate configuration inputs"
    )
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=_validate)

    for name, handler in (("generate", _generate), ("all", _all)):
        command = commands.add_parser(
            name, help=handler.__doc__
        )
        _add_config_options(command)
        command.set_defaults(handler=handler)

    judge = commands.add_parser("judge", help="judge a stored benchmark run")
    _add_config_options(judge, include_resume=False)
    judge.add_argument("--run-dir", required=True, type=Path)
    judge.set_defaults(handler=_judge)

    report = commands.add_parser(
        "report", help="write reports for a stored run"
    )
    report.add_argument("--run-dir", required=True, type=Path)
    report.set_defaults(handler=_report)
    return parser


def _add_config_options(
    parser: argparse.ArgumentParser, include_resume: bool = True
) -> None:
    parser.add_argument("--config", required=True, type=Path)
    if include_resume:
        parser.add_argument(
            "--resume", type=Path, help="reuse an existing run directory"
        )
    parser.add_argument(
        "--model", action="append", dest="models", default=[]
    )
    parser.add_argument(
        "--prompt", action="append", dest="prompts", default=[]
    )
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--split", action="append", dest="splits", default=[])
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop a phase after its first failure",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CliError, OSError, ValueError, TypeError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1


def _validate(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    print("Configuration is valid: %s" % config.source)
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    store = _open_or_create_store(config, getattr(args, "resume", None))
    client = _client()
    filters = _filters(args)
    failed = False
    try:
        artifacts = generate_candidates(
            config, store, filters, client, fail_fast=args.fail_fast
        )
        failed = any(artifact.status != "complete" for artifact in artifacts)
    except Exception as error:
        failed = True
        print("generation stopped: %s" % error, file=sys.stderr)
    print("Run directory: %s" % store.run_dir)
    return 1 if failed else 0


def _judge(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    store = _open_store(args.run_dir)
    return _judge_store(config, store, args)


def _report(args: argparse.Namespace) -> int:
    store = _open_store(args.run_dir)
    report = build_report(store)
    write_report(store, report)
    if not _has_complete_judgment(store):
        raise CliError("report requires at least one complete judgment")
    print("Reports written: %s" % store.run_dir)
    return 1 if report.failures else 0


def _all(args: argparse.Namespace) -> int:
    """generate, judge, and report one benchmark run"""
    config = load_benchmark_config(args.config)
    store = _open_or_create_store(config, getattr(args, "resume", None))
    filters = _filters(args)
    client = _client()
    failed = False

    try:
        artifacts = generate_candidates(
            config, store, filters, client, fail_fast=args.fail_fast
        )
        failed = any(artifact.status != "complete" for artifact in artifacts)
    except Exception as error:
        failed = True
        print("generation stopped: %s" % error, file=sys.stderr)

    try:
        _judge_store(config, store, args, print_output=False)
    except Exception as error:
        failed = True
        print("judging stopped: %s" % error, file=sys.stderr)

    try:
        report = build_report(store)
        write_report(store, report)
        failed = (
            failed
            or _has_requested_failures(config, store, args)
            or not _has_complete_judgment(store)
        )
    except Exception as error:
        failed = True
        print("reporting failed: %s" % error, file=sys.stderr)
    print("Run directory: %s" % store.run_dir)
    return 1 if failed else 0


def _judge_store(
    config: Any,
    store: RunStore,
    args: argparse.Namespace,
    print_output: bool = True,
) -> int:
    client = _client()
    failed = False
    try:
        filters = _filters(args)
        judge_generations(
            config, store, client, fail_fast=args.fail_fast, filters=filters
        )
    except Exception as error:
        failed = True
        if args.fail_fast:
            print("absolute judging stopped: %s" % error, file=sys.stderr)
    if not failed or not args.fail_fast:
        try:
            judge_pairwise_top_models(
                config,
                store,
                client,
                fail_fast=args.fail_fast,
                filters=filters,
            )
        except Exception as error:
            failed = True
            if args.fail_fast:
                print("pairwise judging stopped: %s" % error, file=sys.stderr)
    failed = failed or _has_requested_failures(config, store, args)
    if print_output:
        print("Judgments written: %s" % store.run_dir)
    return 1 if failed else 0


def _open_or_create_store(config: Any, resume: Optional[Path]) -> RunStore:
    if resume is not None:
        return _open_store(resume)
    return RunStore.create(config.output_dir, config.source)


def _open_store(path: Path) -> RunStore:
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise CliError("run directory does not exist: %s" % directory)
    return RunStore.open(directory)


def _client() -> PiRpcClient:
    return PiRpcClient(os.environ.get("PI_BENCHMARK_EXECUTABLE", "pi"))


def _filters(args: argparse.Namespace) -> GenerationFilters:
    return GenerationFilters(
        model_ids=set(args.models) or None,
        prompt_ids=set(args.prompts) or None,
        case_ids=set(args.cases) or None,
        splits=set(args.splits) or None,
    )


def _has_complete_judgment(store: RunStore) -> bool:
    root = store.run_dir / "judgments"
    for path in root.rglob("*.json") if root.is_dir() else ():
        try:
            payload = store.read_json(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if (
            payload.get("operation") == "judgment"
            and payload.get("status") == "complete"
            and isinstance(payload.get("result"), dict)
        ):
            return True
    return False


def _has_requested_failures(
    config: Any, store: RunStore, args: argparse.Namespace
) -> bool:
    selected = _filters(args)
    for operation, root_name in (
        ("generation", "generations"),
        ("judgment", "judgments"),
        ("pairwise", "pairwise"),
    ):
        root = store.run_dir / root_name
        for path in root.rglob("*.json") if root.is_dir() else ():
            try:
                payload = store.read_json(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if (
                payload.get("operation") != operation
                or payload.get("status") != "failed"
            ):
                continue
            if _payload_selected(payload, selected, config):
                return True
    return False


def _payload_selected(
    payload: Dict[str, object], filters: GenerationFilters, config: Any
) -> bool:
    split = payload.get("split")
    if split is None:
        case_id = str(payload.get("case_id", ""))
        split = next(
            (
                case.split for case in config.cases if case.id == case_id
            ),
            None,
        )
    values = (
        (filters.model_ids, payload.get("model_id")),
        (filters.prompt_ids, payload.get("prompt_id")),
        (filters.case_ids, payload.get("case_id")),
        (filters.splits, split),
    )
    return all(
        selected is None or str(value) in selected
        for selected, value in values
    )
