"""Candidate and result exports: CSV, RIS, BibTeX, labels template."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..types import Work

__all__ = [
    "CANDIDATE_FIELDS",
    "read_labels",
    "write_bibtex",
    "write_candidates_csv",
    "write_labels_template",
    "write_ris",
]

CANDIDATE_FIELDS = [
    "key",
    "rank",
    "score",
    "title",
    "year",
    "doi",
    "venue",
    "type",
    "authors",
    "found_via",
    "providers",
    "parent_keys",
    "unresolved",
]

_TYPE_TO_RIS = {
    "journal-article": "JOUR",
    "proceedings-article": "CPAPER",
    "book-chapter": "CHAP",
    "book": "BOOK",
    "posted-content": "GEN",
}


def _authors_str(work: Work) -> str:
    return "; ".join(
        f"{a.surname}, {a.given}" if a.given else a.surname for a in work.authors
    )


def _row(work: Work, rank: int, score: float) -> dict[str, Any]:
    return {
        "key": work.key,
        "rank": rank,
        "score": f"{score:.6f}",
        "title": work.title,
        "year": work.year if work.year is not None else "",
        "doi": work.doi or "",
        "venue": work.venue or "",
        "type": work.type or "",
        "authors": _authors_str(work),
        "found_via": "|".join(str(d) for d in work.directions),
        "providers": "|".join(work.providers),
        "parent_keys": "|".join(work.parent_keys),
        "unresolved": "1" if work.unresolved else "0",
    }


def write_candidates_csv(
    path: str | Path, works: Sequence[Work], scores: Mapping[str, float]
) -> Path:
    ordered = sorted(works, key=lambda w: (-scores.get(w.key, 0.0), w.key))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for i, w in enumerate(ordered, 1):
            writer.writerow(_row(w, i, scores.get(w.key, 0.0)))
    return p


def write_labels_template(
    path: str | Path, works: Sequence[Work], scores: Mapping[str, float]
) -> Path:
    ordered = sorted(works, key=lambda w: (-scores.get(w.key, 0.0), w.key))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["key", "decision", "note", "title", "year"])
        for w in ordered:
            writer.writerow([w.key, "", "", w.title, w.year or ""])
    return p


def read_labels(path: str | Path) -> dict[str, tuple[str, str]]:
    """Return ``{key: (decision, note)}`` for rows with a non-empty decision."""
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("key") or "").strip()
            decision = (row.get("decision") or "").strip().lower()
            if not key or not decision:
                continue
            out[key] = (decision, (row.get("note") or "").strip())
    return out


def write_ris(path: str | Path, works: Sequence[Work]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for w in sorted(works, key=lambda x: x.key):
        lines.append(f"TY  - {_TYPE_TO_RIS.get(w.type or '', 'JOUR')}")
        lines.append(f"TI  - {w.title}")
        for a in w.authors:
            lines.append(f"AU  - {a.surname}, {a.given}" if a.given else f"AU  - {a.surname}")
        if w.year:
            lines.append(f"PY  - {w.year}")
        if w.venue:
            lines.append(f"JO  - {w.venue}")
        if w.doi:
            lines.append(f"DO  - {w.doi}")
        if w.abstract:
            lines.append(f"AB  - {w.abstract}")
        lines.append(f"ID  - {w.key}")
        lines.append("ER  - ")
        lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _bibkey(work: Work) -> str:
    surname = work.first_author_surname or "anon"
    stem = re.sub(r"[^A-Za-z0-9]", "", surname).lower() or "anon"
    first_word = ""
    for tok in work.title_norm.split():
        if len(tok) > 3:
            first_word = tok
            break
    return f"{stem}{work.year or 'nd'}{first_word}"


def write_bibtex(path: str | Path, works: Sequence[Work]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    seen: dict[str, int] = {}
    for w in sorted(works, key=lambda x: x.key):
        base = _bibkey(w)
        seen[base] = seen.get(base, 0) + 1
        cite = base if seen[base] == 1 else f"{base}{chr(96 + seen[base])}"
        kind = "inproceedings" if (w.type or "").startswith("proceedings") else "article"
        fields = [f"  title = {{{w.title}}}"]
        if w.authors:
            fields.append(
                "  author = {"
                + " and ".join(
                    f"{a.surname}, {a.given}" if a.given else a.surname for a in w.authors
                )
                + "}"
            )
        if w.year:
            fields.append(f"  year = {{{w.year}}}")
        if w.venue:
            fields.append(f"  journal = {{{w.venue}}}")
        if w.doi:
            fields.append(f"  doi = {{{w.doi}}}")
        entries.append("@" + kind + "{" + cite + ",\n" + ",\n".join(fields) + "\n}")
    p.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return p
