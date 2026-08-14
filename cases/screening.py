"""Ensemble screening for one case, following the protocol validated in the
single-case study (llm_screening/screening_spec.json).

Two stages, because iteration-2 candidate pools run an order of magnitude larger
than iteration 1:
  stage 1  a cheap model scores everything; records below the gate are excluded
  stage 2  the remaining records are scored by the full ensemble

The gate was validated against the four-model reference: it discarded none of
the full ensemble's includes at any cutoff tested. Seed records are scored
unmarked in the same batches as blind positive controls -- they are known
relevant because the reviewer chose them, so the ensemble must separate them
from the candidate pool.

Import and call `screen(records, rq, tag)`; it returns {key: {model: score}}.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

SPEC = json.loads(
    (Path(__file__).resolve().parent.parent / "llm_screening" / "screening_spec.json").read_text()
)
PROMPT = SPEC["prompt_template"]
BATCH = SPEC["batch_size"]
TAU = SPEC["tau"]
STAGE1 = SPEC["stage1_model"]
GATE = SPEC["stage1_gate"]
STAGE2 = SPEC["stage2_models"]


def build_prompt(rq: dict, records: list[dict]) -> str:
    body = "\n\n".join(
        f"[{i}] TITLE: {r['title']}\nYEAR: {r.get('year') or 'n/a'}\n"
        f"ABSTRACT: {(r.get('abstract') or '(no abstract available)')[:1200]}"
        for i, r in enumerate(records)
    )
    # The spec template carries an `{items}` slot for the numbered publications;
    # leaving it unfilled raised KeyError('items') and appended the body outside
    # the template instead. Filled here so the prompt is the spec's prompt.
    head = PROMPT.format(
        rq=rq["research_question"],
        core=rq["core_relation"],
        ins="; ".join(rq["in_scope"]),
        outs="; ".join(rq["out_of_scope"]),
        items=body,
    )
    return (
        f"{head}\n\n"
        f"Return ONLY a JSON object mapping each index to its integer score, "
        f'e.g. {{"0": 7, "1": 3}}. No prose.'
    )


def parse_scores(text: str, n: int) -> dict[int, float]:
    """Parse a batch's scores from either shape the spec prompt admits.

    The spec template asks for a JSON *array* of {id, Relevance_Score, title_only}
    objects while its closing line asks for a flat index->score *object*; models
    return both. Only the object form parsed before, so an array response was
    silently dropped as an unscored batch. Both are accepted here; nothing about
    the prompt or the scoring criteria changes.
    """
    raw = None
    arr = re.search(r"\[\s*\{.*\}\s*\]", text or "", re.S)
    if arr:
        try:
            items = json.loads(arr.group())
        except json.JSONDecodeError:
            items = None
        if isinstance(items, list):
            raw = {}
            for pos, it in enumerate(items):
                if not isinstance(it, dict):
                    continue
                key = it.get("id", it.get("index", pos))
                val = it.get("Relevance_Score", it.get("relevance_score", it.get("score")))
                if val is not None:
                    raw[key] = val
    if raw is None:
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return {}
        try:
            raw = json.loads(m.group())
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def batches(records: list[dict], size: int = BATCH):
    for i in range(0, len(records), size):
        yield i, records[i : i + size]


def spread(scores: dict) -> float:
    """Mean per-record range across models -- the ensemble's own disagreement."""
    rs = [max(v.values()) - min(v.values()) for v in scores.values() if len(v) > 1]
    return sum(rs) / len(rs) if rs else float("nan")


def icc(matrix: list[list[float]]) -> float:
    """ICC(2,1) for the model x record score matrix."""
    n = len(matrix)
    k = len(matrix[0]) if n else 0
    if n < 2 or k < 2:
        return float("nan")
    grand = sum(sum(r) for r in matrix) / (n * k)
    row_m = [sum(r) / k for r in matrix]
    col_m = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ms_r = k * sum((m - grand) ** 2 for m in row_m) / (n - 1)
    ms_c = n * sum((m - grand) ** 2 for m in col_m) / (k - 1)
    ss_e = sum(
        (matrix[i][j] - row_m[i] - col_m[j] + grand) ** 2
        for i in range(n)
        for j in range(k)
    )
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return (ms_r - ms_e) / denom if denom else float("nan")


def auc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUC: can the ensemble separate known-relevant seeds from the pool?"""
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if p > q else 0.5 if p == q else 0.0
        for p in pos
        for q in neg
    )
    return wins / (len(pos) * len(neg))


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def bootstrap_ci(values: list[float], stat, *, n_boot: int = 400, seed: int = 7):
    """Percentile bootstrap CI, so reported statistics carry uncertainty."""
    import random

    rng = random.Random(seed)
    if len(values) < 3:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(stat(sample))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return (lo, hi)


def mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")
