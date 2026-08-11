"""INV-1: identical config plus identical inputs yield byte-identical artifacts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from snowballslr import Config, Run
from snowballslr.determinism.verify import verify_replay

VOLATILE = {"run.json", "state.json"}  # contain wall-clock timestamps


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in VOLATILE
    }


def _full_run(root, seeds, config, graph, oracle):
    run = Run.init(root, seeds, config, offline_graph=graph)
    while not run.stopped:
        candidates = run.step()
        if not candidates:
            break
        run.label({w.key: oracle.decide(w.key) for w in candidates})
    run.report(prisma=True, graph=True)
    return run


def test_two_independent_runs_are_byte_identical(tmp_path, graph, seeds, config, oracle):
    _full_run(tmp_path / "a", seeds, config, graph, oracle)
    _full_run(tmp_path / "b", seeds, config, graph, oracle)
    a, b = _artifact_hashes(tmp_path / "a"), _artifact_hashes(tmp_path / "b")
    assert a.keys() == b.keys()
    assert a == b, [k for k in a if a[k] != b.get(k)]


def test_shuffled_seed_order_does_not_change_results(tmp_path, graph, seeds, config, oracle):
    _full_run(tmp_path / "a", seeds, config, graph, oracle)
    _full_run(tmp_path / "b", list(reversed(seeds)), config, graph, oracle)
    assert _artifact_hashes(tmp_path / "a") == _artifact_hashes(tmp_path / "b")


def test_replay_verification_passes_on_a_clean_run(tmp_path, graph, seeds, config, oracle):
    _full_run(tmp_path / "r", seeds, config, graph, oracle)
    report = verify_replay(tmp_path / "r", strict=True)
    assert report.ok and report.artifacts_checked > 0
    assert not report.artifacts_mismatched


def test_replay_verification_detects_a_tampered_artifact(tmp_path, graph, seeds, config, oracle):
    from snowballslr.errors import VerificationError

    _full_run(tmp_path / "r", seeds, config, graph, oracle)
    target = tmp_path / "r" / "outputs" / "prisma.json"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VerificationError):
        verify_replay(tmp_path / "r", strict=True)


def test_config_hash_ignores_contact_details(tmp_path):
    a = Config.from_dict({"providers": {"openalex": {"mailto": "a@example.org"}}})
    b = Config.from_dict({"providers": {"openalex": {"mailto": "b@example.org"}}})
    assert a.config_hash == b.config_hash


def test_config_hash_reflects_substantive_changes():
    a = Config.from_dict({"dedup": {"jw_threshold": 0.95}})
    b = Config.from_dict({"dedup": {"jw_threshold": 0.90}})
    assert a.config_hash != b.config_hash


@pytest.mark.parametrize("hashseed", ["0", "random"])
def test_results_are_stable_under_python_hash_randomization(tmp_path, hashseed):
    """Any reliance on dict or set iteration order would surface here."""
    repo = Path(__file__).resolve().parents[1]
    fixtures = Path(__file__).parent / "fixtures"
    out_dir = tmp_path / f"run_{hashseed}"

    script = tmp_path / f"driver_{hashseed}.py"
    script.write_text(
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        "from snowballslr import Run, Config\n"
        "from snowballslr.simulation.oracle import Oracle\n"
        f"graph = json.loads(Path({str(fixtures / 'mini_graph.json')!r}).read_text())\n"
        f"oracle = Oracle.from_file({str(fixtures / 'mini_gold.csv')!r})\n"
        "seeds = sorted(oracle.included)[:3]\n"
        "config = Config.from_dict({'run': {'max_iterations': 8}})\n"
        f"run = Run.init({str(out_dir)!r}, seeds, config, offline_graph=graph)\n"
        "while not run.stopped:\n"
        "    batch = run.step()\n"
        "    if not batch:\n"
        "        break\n"
        "    run.label({w.key: oracle.decide(w.key) for w in batch})\n"
        "run.report(prisma=True, graph=True)\n"
        f"print(hashlib.sha256(Path({str(out_dir / 'outputs' / 'prisma.json')!r}).read_bytes()).hexdigest())\n",
        encoding="utf-8",
    )

    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, check=True
    )
    digest = result.stdout.strip().splitlines()[-1]

    expected = tmp_path.parent / "expected_digest.txt"
    if expected.exists():
        assert digest == expected.read_text().strip()
    else:
        expected.write_text(digest)
