"""Human-readable run report.

The stopping-rule rationale is written so it can be pasted into a methods
section unchanged -- that detail, more than any algorithm, is what determines
whether a tool actually gets used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["build_summary", "write_summary"]


def build_summary(run: Any) -> str:
    state = run.state
    est = run.estimate_recall()
    lines: list[str] = []

    lines.append(f"# Snowballing run: {run.config.run.name}")
    lines.append("")
    seed_keys = set(state.seeds)
    screened = [k for k in state.screened_keys if k not in seed_keys]
    included_new = [k for k in state.included_keys if k not in seed_keys]
    # Providers are read from the provenance of the records themselves, not from
    # the currently configured providers: the report must describe the run that
    # happened, not the configuration it is being read under.
    observed = sorted({p for w in state.works.values() for p in w.providers} - {"seed"})

    lines.append(f"- Seed studies: {len(state.seeds)}")
    lines.append(f"- Iterations completed: {len(state.history)}")
    lines.append(f"- Records identified by citation searching: {len(state.works) - len(seed_keys)}")
    lines.append(f"- Records screened: {len(screened)}")
    lines.append(
        f"- Studies included: {len(included_new)} newly identified "
        f"(+{len(seed_keys)} seeds = {len(state.included_keys)} total)"
    )
    lines.append(f"- Directions: {', '.join(run.config.run.directions)}")
    lines.append(f"- Data sources used: {', '.join(observed) if observed else 'none'}")
    lines.append(f"- Status: {state.phase}" + (f" (stopped by: {state.stopped_by})" if state.stopped_by else ""))
    lines.append("")

    lines.append("## Iteration history")
    lines.append("")
    lines.append("| Iter | Expanded | Screened | Included | Cum. screened | Cum. included | Frontier |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for s in state.history:
        lines.append(
            f"| {s.n} | {s.n_expanded} | {s.n_screened} | {s.n_included} | "
            f"{s.cumulative_screened} | {s.cumulative_included} | {s.frontier_size} |"
        )
    lines.append("")

    lines.append("## Recall estimate")
    lines.append("")
    if est.estimable:
        lines.append(f"- Method: `{est.method}`")
        lines.append(f"- Estimated relevant population: {est.n_hat:.1f}")
        if est.n_hat_ci:
            lines.append(f"- 95% CI for N: [{est.n_hat_ci[0]:.1f}, {est.n_hat_ci[1]:.1f}]")
        if est.recall is not None:
            lines.append(f"- Estimated recall: {est.recall:.1%}")
        if est.recall_ci:
            lines.append(
                f"- 95% CI for recall: [{est.recall_ci[0]:.1%}, {est.recall_ci[1]:.1%}]"
            )
    else:
        lines.append(f"- Not estimable: {est.reason}")
    lines.append("")

    if est.warnings:
        lines.append("### Assumption warnings")
        lines.append("")
        for w in est.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Stopping rules")
    lines.append("")
    for d in state.last_decisions:
        mark = "**FIRED**" if d.get("triggered") else "not fired"
        lines.append(f"- `{d.get('rule')}` -- {mark}")
    lines.append("")

    fired = [d for d in state.last_decisions if d.get("triggered")]
    if fired:
        lines.append("### Methods-section text")
        lines.append("")
        for d in fired:
            lines.append(f"> {d.get('rationale')}")
            lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Config hash: `{run.config.config_hash}`")
    lines.append(f"- Cache entries: {len(run.cache)}")
    lines.append(
        "- Re-run `snowballslr verify <run_dir>` to confirm that this run "
        "reproduces byte-for-byte from its cache, or "
        "`snowballslr verify <run_dir> --refresh` to quantify drift in the "
        "underlying bibliographic databases since the run was executed."
    )
    lines.append("")
    lines.append(
        "_Citation searching is a complementary search method: automated citation "
        "searching has not been shown to outperform Boolean database search on "
        "recall. The contribution of this tool is knowing when to stop and how much "
        "of the relevant literature has been found._"
    )
    lines.append("")
    return "\n".join(lines)


def write_summary(path: str | Path, run: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_summary(run), encoding="utf-8")
    return p
