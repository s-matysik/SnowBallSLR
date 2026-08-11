"""PRISMA 2020 counts and flow diagram for the 'other methods' column.

PRISMA 2020 gives citation searching its own identification column; the numbers
required there are produced automatically from the audit trail rather than
tallied by hand.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.state import RunState
from ..types import Decision, Direction

__all__ = ["PrismaCounts", "build_counts", "write_prisma_json", "write_prisma_svg"]


@dataclass
class PrismaCounts:
    seeds: int = 0
    identified_total: int = 0
    identified_by_direction: dict[str, int] = field(default_factory=dict)
    identified_by_provider: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    ineligible_removed: int = 0
    screened: int = 0
    excluded: int = 0
    included: int = 0
    unresolved: int = 0
    per_iteration: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "records_identified_via_citation_searching": self.identified_total,
            "identified_by_direction": dict(sorted(self.identified_by_direction.items())),
            "identified_by_provider": dict(sorted(self.identified_by_provider.items())),
            "duplicates_removed": self.duplicates_removed,
            "records_removed_before_screening": self.ineligible_removed,
            "records_screened": self.screened,
            "records_excluded": self.excluded,
            "studies_included": self.included,
            "records_unresolvable": self.unresolved,
            "per_iteration": self.per_iteration,
        }


def build_counts(state: RunState, audit_events: Sequence[Mapping[str, Any]] = ()) -> PrismaCounts:
    counts = PrismaCounts(seeds=len(state.seeds))
    seed_keys = set(state.seeds)

    by_direction: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    for key, work in sorted(state.works.items()):
        # A seed re-encountered during chasing is not a newly identified record.
        # Counting it here would double count it against the database-search
        # column of the flow diagram.
        if key in seed_keys or Direction.SEED in work.directions:
            continue
        counts.identified_total += 1
        for d in work.directions:
            if d == Direction.SEED:
                continue
            by_direction[str(d)] = by_direction.get(str(d), 0) + 1
        for p in work.providers:
            by_provider[p] = by_provider.get(p, 0) + 1
        if work.unresolved:
            counts.unresolved += 1

    counts.identified_by_direction = by_direction
    counts.identified_by_provider = by_provider

    for event in audit_events:
        if event.get("event") == "dedup_merge":
            counts.duplicates_removed += max(0, len(event.get("members") or []) - 1)
        elif event.get("event") == "ineligible":
            counts.ineligible_removed += int(event.get("n", 0))

    # Seeds entered the review through the database search and are already
    # counted in the main PRISMA column; counting them again here would double
    # count them across the two identification routes.
    counts.screened = len([k for k in state.screened_keys if k not in seed_keys])
    counts.excluded = sum(
        1 for k, v in state.decisions.items() if v == Decision.EXCLUDE and k not in seed_keys
    )
    counts.included = len([k for k in state.included_keys if k not in seed_keys])
    counts.per_iteration = [s.to_dict() for s in state.history]
    return counts


def write_prisma_json(path: str | Path, counts: PrismaCounts) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(counts.to_dict(), sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


_BOX = (
    '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" '
    'stroke="#333333" stroke-width="1.5"/>'
)


def _wrap(text: str, width: int = 34) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def write_prisma_svg(path: str | Path, counts: PrismaCounts) -> Path:
    """Hand-drawn SVG: no external renderer, deterministic output."""
    boxes = [
        ("Seed studies (start set)", f"n = {counts.seeds}"),
        (
            "Records identified via citation searching",
            "\n".join(
                [f"total n = {counts.identified_total}"]
                + [f"{k}: n = {v}" for k, v in sorted(counts.identified_by_direction.items())]
            ),
        ),
        (
            "Records removed before screening",
            f"duplicates n = {counts.duplicates_removed}\n"
            f"ineligible n = {counts.ineligible_removed}\n"
            f"unresolvable n = {counts.unresolved}",
        ),
        ("Records screened", f"n = {counts.screened}"),
        ("Records excluded", f"n = {counts.excluded}"),
        ("Studies included in review", f"n = {counts.included}"),
    ]

    box_w, box_h, gap = 340, 96, 42
    x = 40
    width = box_w + 2 * x + 220
    height = len(boxes) * (box_h + gap) + 40

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    y = 20
    positions: list[int] = []
    for title, body in boxes:
        positions.append(y)
        parts.append(_BOX.format(x=x, y=y, w=box_w, h=box_h))
        parts.append(
            f'<text x="{x + 12}" y="{y + 24}" font-size="13" font-weight="bold" '
            f'fill="#111111">{_escape(title)}</text>'
        )
        line_y = y + 44
        for raw_line in body.split("\n"):
            for line in _wrap(raw_line, 40):
                parts.append(
                    f'<text x="{x + 12}" y="{line_y}" font-size="12" fill="#333333">'
                    f"{_escape(line)}</text>"
                )
                line_y += 16
        y += box_h + gap

    # Vertical connectors between consecutive boxes.
    for i in range(len(positions) - 1):
        y0 = positions[i] + box_h
        y1 = positions[i + 1]
        cx = x + box_w // 2
        parts.append(
            f'<line x1="{cx}" y1="{y0}" x2="{cx}" y2="{y1 - 8}" stroke="#333333" '
            f'stroke-width="1.5"/>'
            f'<polygon points="{cx - 5},{y1 - 8} {cx + 5},{y1 - 8} {cx},{y1}" '
            f'fill="#333333"/>'
        )

    parts.append(
        f'<text x="{x}" y="{height - 8}" font-size="11" fill="#666666">'
        f"Generated by SnowBallSLR - PRISMA 2020, identification via other methods</text>"
    )
    parts.append("</svg>")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
