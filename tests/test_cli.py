"""CLI end-to-end on the fixture graph."""

from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from snowballslr.cli import EXIT_CONFIG, EXIT_OK, app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def _seeds_file(tmp_path, oracle, n=3):
    p = tmp_path / "seeds.txt"
    p.write_text("\n".join(sorted(oracle.included)[:n]) + "\n", encoding="utf-8")
    return p


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == EXIT_OK and "snowballslr" in result.stdout


def test_init_step_label_report_cycle(tmp_path, oracle):
    seeds = _seeds_file(tmp_path, oracle)
    run_dir = tmp_path / "run"
    graph = str(FIXTURES / "mini_graph.json")

    r = runner.invoke(app, ["init", str(run_dir), "--seeds", str(seeds), "--graph", graph])
    assert r.exit_code == EXIT_OK, r.stdout
    assert (run_dir / "state.json").exists()

    r = runner.invoke(app, ["step", str(run_dir), "--graph", graph])
    assert r.exit_code == EXIT_OK, r.stdout
    template = run_dir / "iterations" / "iter_001" / "labels_template.csv"
    assert template.exists()

    labels = tmp_path / "labels.csv"
    with open(template, newline="", encoding="utf-8") as fh, open(
        labels, "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(["key", "decision", "note"])
        for row in csv.DictReader(fh):
            writer.writerow([row["key"], oracle.decide(row["key"]), ""])

    r = runner.invoke(app, ["label", str(run_dir), "--file", str(labels), "--graph", graph])
    assert r.exit_code == EXIT_OK, r.stdout

    r = runner.invoke(app, ["status", str(run_dir)])
    assert r.exit_code == EXIT_OK and "included:" in r.stdout

    r = runner.invoke(app, ["report", str(run_dir), "--graph"])
    assert r.exit_code == EXIT_OK
    assert (run_dir / "outputs" / "prisma.svg").exists()


def test_run_simulation_command(tmp_path, oracle):
    seeds = _seeds_file(tmp_path, oracle)
    run_dir = tmp_path / "run"
    graph = str(FIXTURES / "mini_graph.json")
    runner.invoke(app, ["init", str(run_dir), "--seeds", str(seeds), "--graph", graph])
    r = runner.invoke(
        app,
        ["run", str(run_dir), "--oracle", str(FIXTURES / "mini_gold.csv"), "--graph", graph],
    )
    assert r.exit_code == EXIT_OK, r.stdout
    assert "stopped by" in r.stdout


def test_estimate_command_emits_json(tmp_path, oracle):
    seeds = _seeds_file(tmp_path, oracle)
    run_dir = tmp_path / "run"
    graph = str(FIXTURES / "mini_graph.json")
    runner.invoke(app, ["init", str(run_dir), "--seeds", str(seeds), "--graph", graph])
    runner.invoke(
        app,
        ["run", str(run_dir), "--oracle", str(FIXTURES / "mini_gold.csv"), "--graph", graph],
    )
    r = runner.invoke(app, ["estimate", str(run_dir), "--method", "chao"])
    assert r.exit_code == EXIT_OK and '"method"' in r.stdout


def test_verify_replay_command(tmp_path, oracle):
    seeds = _seeds_file(tmp_path, oracle)
    run_dir = tmp_path / "run"
    graph = str(FIXTURES / "mini_graph.json")
    runner.invoke(app, ["init", str(run_dir), "--seeds", str(seeds), "--graph", graph])
    runner.invoke(
        app,
        ["run", str(run_dir), "--oracle", str(FIXTURES / "mini_gold.csv"), "--graph", graph],
    )
    runner.invoke(app, ["report", str(run_dir)])
    r = runner.invoke(app, ["verify", str(run_dir)])
    assert r.exit_code == EXIT_OK and "replay OK" in r.stdout


def test_export_command(tmp_path, oracle):
    seeds = _seeds_file(tmp_path, oracle)
    run_dir = tmp_path / "run"
    graph = str(FIXTURES / "mini_graph.json")
    runner.invoke(app, ["init", str(run_dir), "--seeds", str(seeds), "--graph", graph])
    runner.invoke(
        app,
        ["run", str(run_dir), "--oracle", str(FIXTURES / "mini_gold.csv"), "--graph", graph],
    )
    r = runner.invoke(app, ["export", str(run_dir), "--format", "ris", "--what", "included"])
    assert r.exit_code == EXIT_OK
    assert (run_dir / "outputs" / "included.ris").read_text(encoding="utf-8").startswith("TY  -")


def test_init_with_empty_seed_file_is_a_config_error(tmp_path):
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("# only a comment\n", encoding="utf-8")
    r = runner.invoke(app, ["init", str(tmp_path / "run"), "--seeds", str(seeds)])
    assert r.exit_code == EXIT_CONFIG
