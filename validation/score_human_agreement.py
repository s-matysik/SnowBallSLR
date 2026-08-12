"""Score author-versus-ensemble screening agreement on the blind sample.

Usage:
    python validation/score_human_agreement.py

Fill the `decision` column of llm_screening/human_validation_sample.csv with
`include` or `exclude` before running. The ensemble scores live in a separate
file and are NOT in the sample csv, so the screening is blind by construction.

Reports Cohen's kappa with a normal-approximation 95% CI, the confusion matrix,
percent agreement, and prevalence- and bias-adjusted kappa (PABAK) -- the same
statistics ReviQ reports, so the comparison is like for like.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "llm_screening" / "human_validation_sample.csv"
KEY = REPO / "llm_screening" / "human_validation_key.json"
TAU = 7.0


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float, float]:
    """Cohen's kappa with a normal-approximation 95% confidence interval."""
    n = len(a)
    both1 = sum(1 for x, y in zip(a, b, strict=True) if x == 1 and y == 1)
    both0 = sum(1 for x, y in zip(a, b, strict=True) if x == 0 and y == 0)
    po = (both1 + both0) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0, 1.0, 1.0
    k = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return k, k - 1.96 * se, k + 1.96 * se


def main() -> None:
    with open(SAMPLE, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    with open(KEY, encoding="utf-8") as fh:
        key = json.load(fh)
    scores = key["model_scores"]

    labelled = [r for r in rows if (r.get("decision") or "").strip().lower() in {"include", "exclude"}]
    if not labelled:
        raise SystemExit(
            f"No decisions found. Fill the 'decision' column of {SAMPLE.name} "
            "with include/exclude first."
        )
    if len(labelled) < len(rows):
        print(f"WARNING: {len(rows) - len(labelled)} of {len(rows)} rows are unlabelled and skipped.\n")

    human = [1 if r["decision"].strip().lower() == "include" else 0 for r in labelled]
    model = [1 if scores[r["key"]] >= TAU else 0 for r in labelled]

    k, lo, hi = cohen_kappa(human, model)
    n = len(labelled)
    tp = sum(1 for h, m in zip(human, model, strict=True) if h == 1 and m == 1)
    tn = sum(1 for h, m in zip(human, model, strict=True) if h == 0 and m == 0)
    fp = sum(1 for h, m in zip(human, model, strict=True) if h == 0 and m == 1)
    fn = sum(1 for h, m in zip(human, model, strict=True) if h == 1 and m == 0)
    po = (tp + tn) / n
    pabak = 2 * po - 1

    print(f"n = {n} records, blind stratified sample (seed {key['seed']})")
    print(f"\nCohen's kappa = {k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"percent agreement = {po:.3f}")
    print(f"PABAK = {pabak:.3f}")
    print("\nconfusion matrix (rows: author, cols: ensemble)")
    print("                 ensemble include   ensemble exclude")
    print(f"  author include {tp:16d} {fn:18d}")
    print(f"  author exclude {fp:16d} {tn:18d}")
    if tp + fn:
        print(f"\nensemble recall against the author's includes: {tp / (tp + fn):.3f}")
    if tp + fp:
        print(f"ensemble precision against the author's includes: {tp / (tp + fp):.3f}")

    disagreements = [
        (r["key"], scores[r["key"]], r["decision"], r["title"][:70])
        for r, h, m in zip(labelled, human, model, strict=True)
        if h != m
    ]
    print(f"\n{len(disagreements)} disagreements; the 10 largest by model score:")
    for _key, s, dec, title in sorted(disagreements, key=lambda x: -x[1])[:10]:
        print(f"  model {s:5.2f}  author {dec:8s}  {title}")


if __name__ == "__main__":
    main()
