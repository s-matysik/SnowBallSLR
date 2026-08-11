"""Command line interface.

Exit codes: 0 ok, 1 verification/validation failure, 2 config error,
3 provider failure, 4 a stopping rule fired during ``step``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from . import __version__
from .config import Config
from .core.run import Run
from .errors import ConfigError, LabelError, ProviderError, StateError, VerificationError
from .report.export import write_bibtex, write_candidates_csv, write_ris

app = typer.Typer(
    add_completion=False,
    help="Deterministic, saturation-aware snowballing for systematic reviews.",
    no_args_is_help=True,
)

EXIT_OK = 0
EXIT_VERIFY = 1
EXIT_CONFIG = 2
EXIT_PROVIDER = 3
EXIT_STOPPED = 4


def _echo(msg: str, *, err: bool = False) -> None:
    typer.echo(msg, err=err)


def _load_config(path: Path | None) -> Config:
    try:
        cfg = Config.from_yaml(path) if path else Config()
        cfg.validate_consistency()
        for warning in cfg.warnings():
            _echo(f"warning: {warning}", err=True)
        return cfg
    except ConfigError as exc:
        _echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc


def _report_filtering(run: Run) -> None:
    """Tell the user what the eligibility filter removed, and on what values.

    A filter that quietly deletes most of the corpus looks exactly like a small
    corpus, so anything above half is reported unprompted.
    """
    report = getattr(run, "last_rejection_report", None)
    if not report or not report.get("n_rejected"):
        return
    share = report["share_rejected"]
    level = "warning" if share >= 0.5 else "note"
    _echo(
        f"{level}: eligibility filter removed {report['n_rejected']} of "
        f"{report['n_fresh']} new records ({share:.0%})",
        err=True,
    )
    for reason, count in sorted(
        report["by_reason"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        _echo(f"    {reason}: {count}", err=True)
    for field, counts in sorted(report.get("observed_values", {}).items()):
        shown = ", ".join(f"{v}={c}" for v, c in list(counts.items())[:8])
        _echo(f"    rejected {field} values seen: {shown}", err=True)
    if share >= 0.5:
        _echo(
            "    if this is unexpected, check that eligibility values match the "
            "vocabulary your providers actually return",
            err=True,
        )


def _load_run(run_dir: Path, graph: Path | None = None) -> Run:
    try:
        return Run.load(run_dir, offline_graph=str(graph) if graph else None)
    except StateError as exc:
        _echo(f"state error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    if version:
        _echo(f"snowballslr {__version__}")
        raise typer.Exit(EXIT_OK)


@app.command()
def init(
    run_dir: Path = typer.Argument(..., help="Directory to create the run in."),
    seeds: Path = typer.Option(..., "--seeds", help="File with one identifier per line."),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration."),
    graph: Path | None = typer.Option(
        None, "--graph", help="Frozen citation graph (offline mode)."
    ),
) -> None:
    """Create a run and resolve the seed set."""
    cfg = _load_config(config)
    identifiers = [
        line.strip()
        for line in Path(seeds).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not identifiers:
        _echo("no seed identifiers found", err=True)
        raise typer.Exit(EXIT_CONFIG)
    try:
        run = Run.init(
            run_dir, identifiers, cfg, offline_graph=str(graph) if graph else None
        )
    except ProviderError as exc:
        _echo(f"provider error: {exc}", err=True)
        raise typer.Exit(EXIT_PROVIDER) from exc
    _echo(
        f"initialised {run_dir} with {len(run.state.seeds)}/{len(identifiers)} "
        f"seeds resolved"
    )
    unresolved = len(identifiers) - len(run.state.seeds)
    if unresolved:
        _echo(f"warning: {unresolved} seed(s) could not be resolved", err=True)
    run.close()


@app.command()
def step(
    run_dir: Path = typer.Argument(...),
    graph: Path | None = typer.Option(None, "--graph"),
) -> None:
    """Run one iteration and export the candidate batch for screening."""
    run = _load_run(run_dir, graph)
    try:
        candidates = run.step()
    except ProviderError as exc:
        _echo(f"provider error: {exc}", err=True)
        raise typer.Exit(EXIT_PROVIDER) from exc

    n = run.state.iteration
    _report_filtering(run)
    if run.stopped:
        _echo(f"stopped after iteration {n}: {run.state.stopped_by}")
        run.close()
        raise typer.Exit(EXIT_STOPPED)
    if not candidates:
        _echo(f"iteration {n}: no new candidates")
        run.close()
        raise typer.Exit(EXIT_OK)

    out = run_dir / "iterations" / f"iter_{n:03d}"
    _echo(f"iteration {n}: {len(candidates)} candidates awaiting screening")
    _echo(f"  candidates: {out / 'candidates.csv'}")
    _echo(f"  labels:     {out / 'labels_template.csv'}")
    run.close()


@app.command()
def label(
    run_dir: Path = typer.Argument(...),
    file: Path = typer.Option(..., "--file", help="CSV with key,decision[,note]."),
    graph: Path | None = typer.Option(None, "--graph"),
) -> None:
    """Ingest screening decisions and evaluate the stopping rules."""
    run = _load_run(run_dir, graph)
    try:
        run.label_from_file(file)
    except (LabelError, StateError) as exc:
        _echo(f"label error: {exc}", err=True)
        raise typer.Exit(EXIT_VERIFY) from exc

    for d in run.state.last_decisions:
        mark = "FIRED" if d["triggered"] else "     "
        _echo(f"  [{mark}] {d['rule']}")
    if run.stopped:
        _echo(f"stopped: {run.state.stopped_by}")
    _echo(f"included so far: {len(run.state.included_keys)}")
    run.close()


@app.command()
def run(
    run_dir: Path = typer.Argument(...),
    oracle: Path = typer.Option(..., "--oracle", help="Gold-standard inclusion list."),
    graph: Path | None = typer.Option(None, "--graph"),
) -> None:
    """Simulation mode: loop to saturation with an oracle answering screening."""
    from .simulation.oracle import Oracle

    r = _load_run(run_dir, graph)
    gold = Oracle.from_file(oracle)
    r.run_to_saturation(gold.as_mapping())
    _echo(
        f"finished after {len(r.state.history)} iterations; "
        f"{len(r.state.included_keys)} included; stopped by {r.state.stopped_by}"
    )
    r.close()


@app.command()
def status(run_dir: Path = typer.Argument(...)) -> None:
    """Print current run state."""
    r = _load_run(run_dir)
    s = r.state
    _echo(f"phase:      {s.phase}")
    _echo(f"iteration:  {s.iteration}")
    _echo(f"seeds:      {len(s.seeds)}")
    _echo(f"discovered: {len(s.works)}")
    _echo(f"screened:   {len(s.screened_keys)}")
    _echo(f"included:   {len(s.included_keys)}")
    _echo(f"frontier:   {len(s.frontier)}")
    _echo(f"pending:    {len(s.pending)}")
    if s.stopped_by:
        _echo(f"stopped by: {s.stopped_by}")
    r.close()


@app.command()
def estimate(
    run_dir: Path = typer.Argument(...),
    method: str = typer.Option("chapman", "--method", help="chapman | loglinear | chao"),
) -> None:
    """Estimate recall by capture-recapture."""
    r = _load_run(run_dir)
    est = r.estimate_recall(method=method)
    _echo(json.dumps(est.to_dict(), indent=2, sort_keys=True))
    if est.warnings:
        _echo("", err=True)
        for w in est.warnings:
            _echo(f"warning: {w}", err=True)
    r.close()


@app.command()
def verify(
    run_dir: Path = typer.Argument(...),
    refresh: bool = typer.Option(
        False, "--refresh", help="Refetch live and classify drift."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Cap entries on refresh."),
) -> None:
    """Verify reproducibility, or quantify database drift since the run."""
    from .determinism.verify import verify_refresh, verify_replay

    if not refresh:
        try:
            report = verify_replay(run_dir, strict=True)
        except VerificationError as exc:
            _echo(f"VERIFY FAILED: {exc}", err=True)
            raise typer.Exit(EXIT_VERIFY) from exc
        report.write(run_dir)
        _echo(f"replay OK: {report.artifacts_checked} artifacts reproduced")
        raise typer.Exit(EXIT_OK)

    r = _load_run(run_dir)
    providers = {p.name: p for p in r.providers}

    def fetcher(entry):
        provider = providers.get(entry.provider)
        if provider is None:
            return entry.body, entry.http_status
        params = {k: v for k, v in entry.params.items() if not k.startswith("_")}
        url = params.pop("_url", None) or getattr(provider, "base_url", "")
        try:
            result = provider._http_get(url, params)
            return result.body, result.status
        except ProviderError:
            return None, 599

    report = verify_refresh(run_dir, fetcher, limit=limit)
    _, md_path = report.write(run_dir)
    for name, count in sorted(report.counts.items()):
        _echo(f"  {name:20s} {count}")
    _echo(f"report: {md_path}")
    r.close()
    raise typer.Exit(EXIT_OK if report.ok else EXIT_VERIFY)


@app.command()
def report(
    run_dir: Path = typer.Argument(...),
    prisma: bool = typer.Option(True, "--prisma/--no-prisma"),
    graph: bool = typer.Option(False, "--graph/--no-graph"),
) -> None:
    """Write PRISMA counts, diagram, network and run summary."""
    r = _load_run(run_dir)
    produced = r.report(prisma=prisma, graph=graph)
    for name, path in sorted(produced.items()):
        _echo(f"  {name:18s} {path}")
    r.close()


@app.command()
def export(
    run_dir: Path = typer.Argument(...),
    fmt: str = typer.Option("csv", "--format", help="csv | ris | bib"),
    what: str = typer.Option("included", "--what", help="candidates | included"),
) -> None:
    """Export candidates or included studies."""
    r = _load_run(run_dir)
    works = r.state.pending_works() if what == "candidates" else r.state.included
    out = run_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "ris":
        path = write_ris(out / f"{what}.ris", works)
    elif fmt == "bib":
        path = write_bibtex(out / f"{what}.bib", works)
    else:
        path = write_candidates_csv(out / f"{what}.csv", works, {})
    _echo(f"wrote {len(works)} records to {path}")
    r.close()


@app.command()
def snapshot(
    run_dir: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out", help="Destination .tar.gz"),
) -> None:
    """Freeze the cache for replication packages."""
    import tarfile

    r = _load_run(run_dir)
    cache_dir = run_dir / r.config.determinism.cache_dir
    if not cache_dir.exists():
        _echo(f"no cache at {cache_dir}", err=True)
        raise typer.Exit(EXIT_CONFIG)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(cache_dir, arcname="cache")
        for name in ("run.json", "config.yaml", "state.json"):
            p = run_dir / name
            if p.exists():
                tar.add(p, arcname=name)
    _echo(f"snapshot written: {out} ({len(r.cache)} cache entries)")
    r.close()


def cli() -> None:  # pragma: no cover - console entry point
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    cli()
