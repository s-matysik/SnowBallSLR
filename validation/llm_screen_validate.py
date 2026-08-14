#!/usr/bin/env python3
"""Cross-vendor validation of an LLM screening ensemble.

Rewrite of `ewaluator.ipynb` as a runnable script. Three things changed, and each
was a finding from the three-case study rather than a matter of taste:

1. **Keys come from the environment, never the source.** The notebook had six live
   API keys pasted into its cells. Set them in your shell (or a .env file this
   script reads) so the file is safe to commit and share.

2. **Vendors are separated, and reported separately.** The three-case study scored
   every record with three Claude models, so its inter-model agreement measured
   within-family consistency, not independent judgement. This script computes
   agreement *within* each vendor family and *between* families, because the
   between-family number is the one that answers a reviewer.

3. **One record's score is written once, on one basis.** In the adverse case a
   record gated out at stage 1 carried a one-model score while a survivor carried
   three, and the AUC computed over that mixture could not be reproduced from the
   saved scores (0.58-0.73 depending on basis, against 0.781 reported). This
   script writes every score with its basis attached and refuses to compute a
   statistic across mixed bases.

Usage
-----
    export OPENAI_API_KEY=...        # any subset; missing vendors are skipped
    export ANTHROPIC_API_KEY=...
    export GOOGLE_API_KEY=...
    export DEEPSEEK_API_KEY=...
    export XAI_API_KEY=...

    # score records and validate
    python validation/llm_screen_validate.py score \
        --records cases/records_mycase.json \
        --question cases/research_question_mycase.json \
        --out cases/scores_mycase.json

    # then, separately, the validations (no API calls; reads the scores file)
    python validation/llm_screen_validate.py validate \
        --scores cases/scores_mycase.json \
        --seeds cases/seed_selection.json --case mycase \
        --out cases/validation_mycase.json

    # the five validation designs, run one at a time
    python validation/llm_screen_validate.py agreement   --scores ... --out ...
    python validation/llm_screen_validate.py controls    --scores ... --seeds ... --case ...
    python validation/llm_screen_validate.py threshold   --scores ... --out ...
    python validation/llm_screen_validate.py robustness  --records ... --question ... --out ...
    python validation/llm_screen_validate.py negative    --records ... --question ... --wrong-question ...

Input formats
-------------
`--records` is a JSON list of objects with at least `key` and `title`, optionally
`year` and `abstract`:

    [{"key": "doi:10.1/abc", "title": "...", "year": 2021, "abstract": "..."}]

`--question` is a JSON object:

    {"research_question": "...", "core_relation": "...",
     "in_scope": ["..."], "out_of_scope": ["..."]}

Dependencies: `pip install openai anthropic google-genai`. Only the vendors whose
keys are present are called; each SDK is imported lazily so a missing package
skips that vendor rather than aborting the run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Vendor registry
# --------------------------------------------------------------------------
# `family` is what makes the cross-vendor comparison possible: agreement within a
# family is a weaker claim than agreement across families, and the two must not be
# pooled into one number.


@dataclass(frozen=True)
class Vendor:
    name: str
    family: str
    env_key: str
    model: str
    base_url: str | None = None
    sdk: str = "openai"          # "openai" | "anthropic" | "google"


VENDORS: tuple[Vendor, ...] = (
    Vendor("openai", "openai", "OPENAI_API_KEY", "gpt-5.1"),
    Vendor("anthropic", "anthropic", "ANTHROPIC_API_KEY",
           "claude-sonnet-4-5-20250929", sdk="anthropic"),
    Vendor("google", "google", "GOOGLE_API_KEY", "gemini-2.5-flash", sdk="google"),
    Vendor("deepseek", "deepseek", "DEEPSEEK_API_KEY", "deepseek-chat",
           base_url="https://api.deepseek.com"),
    Vendor("xai", "xai", "XAI_API_KEY", "grok-4-1-fast-non-reasoning",
           base_url="https://api.x.ai/v1"),
    # Moonshot runs two platforms with separate key namespaces: api.moonshot.cn (mainland)
    # and api.moonshot.ai (international). A key issued for one returns 401 on the other,
    # and the model catalogues differ -- kimi-k2-turbo-preview is a .cn model id and 404s
    # on .ai. If a key 401s here, it is probably a mainland key: switch base_url to
    # https://api.moonshot.cn/v1 and the model to kimi-k2-turbo-preview.
    Vendor("kimi", "moonshot", "KIMI_API_KEY", "kimi-k2.6",
           base_url="https://api.moonshot.ai/v1"),
)


def family_of(rater: str) -> str:
    """Which vendor family a rater belongs to.

    Raters are keyed by vendor name when this script produced them, but a scores
    file written elsewhere may key them by raw model id (`claude-sonnet-5`,
    `gpt-5.1`, `gemini-2.5-flash`). Both must resolve to the same family, or the
    within/between distinction — the whole point of the agreement design —
    silently collapses into "every model is its own family".
    """
    for v in VENDORS:
        if rater == v.name or rater == v.model:
            return v.family
    r = rater.lower()
    for token, fam in (("claude", "anthropic"), ("gpt", "openai"), ("o1", "openai"),
                       ("o3", "openai"), ("gemini", "google"), ("deepseek", "deepseek"),
                       ("grok", "xai"), ("kimi", "moonshot"), ("moonshot", "moonshot"),
                       ("mistral", "mistral"), ("llama", "meta"), ("qwen", "alibaba")):
        if token in r:
            return fam
    return rater


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=value lines into the environment if a .env file is present.

    Values already set in the environment win, so an explicit `export` overrides
    the file.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def available_vendors() -> list[Vendor]:
    return [v for v in VENDORS if os.environ.get(v.env_key)]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

RUBRIC = """Research question against which relevance must be judged: {rq}

Core relation at the heart of the question: {core}
Clearly in scope: {ins}
Adjacent but NOT the core: {outs}

For each publication below, assign a single relevance score from 1 to 10 based on its title and
abstract. Judge each publication on its own -- do not revisit earlier scores and do not compare
publications against each other.

Relevance Score (1-10):
1-2: no material connection to the research problem -- different field or unrelated phenomenon.
3-4: very weak connection -- addresses the broader field in which the problem sits, not the problem.
5-7: connected -- related constructs or partial aspects, but the link is indirect or incomplete.
8-9: directly relevant -- explicitly studies the dependency or phenomenon at the core of the problem.
10:  fully relevant -- a core study of exactly the stated problem, with empirical evidence.

Where only a title is given and no abstract, score what the title supports.

Publications to score (each numbered):
{items}

Output format: return ONLY a JSON array, one object per publication, in the same order as the
numbering. Do not reproduce the publication text:
[{{"id": int, "Relevance_Score": int}}]"""

# An independently worded rubric encoding the SAME decision criteria. Used by the
# `robustness` design: if scores move under this, the screener was sensitive to
# wording rather than to criteria.
RUBRIC_PARAPHRASE = """You are screening publications for a systematic literature review.

The review asks: {rq}
The relationship of interest is: {core}
Treat as relevant work on: {ins}
Treat as NOT relevant work whose focus is: {outs}

Rate how relevant each publication is to this review, from 1 to 10, using only its title and
abstract. Use the whole range. A rating of 1 or 2 means the work belongs to a different problem
entirely. 3 or 4 means it touches the surrounding field but not the review's question. 5, 6 or 7
means it bears on the question partially or indirectly. 8 or 9 means it addresses the question
directly. 10 is reserved for work squarely on the review's central relationship. Judge each
publication independently of the others.

Publications to score (each numbered):
{items}

Output format: return ONLY a JSON array, one object per publication, in the same order as the
numbering:
[{{"id": int, "Relevance_Score": int}}]"""


def render_items(records: list[dict]) -> str:
    out = []
    for i, r in enumerate(records):
        ab = (r.get("abstract") or "").strip()
        out.append(
            f"{i}. TITLE: {r.get('title', '')}\n"
            f"YEAR: {r.get('year') or 'n/a'}\n"
            f"ABSTRACT: {ab[:1200] if ab else '(no abstract available)'}"
        )
    return "\n\n".join(out)


def build_prompt(records: list[dict], rq: dict, template: str = RUBRIC) -> str:
    return template.format(
        rq=rq["research_question"],
        core=rq.get("core_relation", ""),
        ins="; ".join(rq.get("in_scope") or []),
        outs="; ".join(rq.get("out_of_scope") or []),
        items=render_items(records),
    )


def parse_scores(text: str, n: int) -> dict[int, float]:
    """Extract {index: score} from a model reply, tolerating surrounding prose."""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return {}
    try:
        raw = json.loads(m.group())
    except json.JSONDecodeError:
        return {}
    out: dict[int, float] = {}
    for obj in raw if isinstance(raw, list) else []:
        if not isinstance(obj, dict):
            continue
        try:
            idx = int(obj.get("id"))
            score = float(obj.get("Relevance_Score"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n and 1.0 <= score <= 10.0:
            out[idx] = score
    return out


# --------------------------------------------------------------------------
# Vendor calls
# --------------------------------------------------------------------------


def call_vendor(vendor: Vendor, prompt: str, *, max_tokens: int = 3000) -> str:
    """One completion from one vendor. Raises on transport or SDK failure."""
    key = os.environ[vendor.env_key]
    if vendor.sdk == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=vendor.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    if vendor.sdk == "google":
        from google import genai

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=vendor.model, contents=prompt)
        return resp.text or ""

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=vendor.base_url)
    resp = client.chat.completions.create(
        model=vendor.model,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def score_batches(
    records: list[dict],
    rq: dict,
    vendors: list[Vendor],
    *,
    batch_size: int = 25,
    template: str = RUBRIC,
    retries: int = 1,
    pause: float = 0.4,
    log=print,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Score every record with every vendor.

    Returns `(scores, failures)` where `scores[key][vendor] = score`. A batch that
    fails to parse is retried once with a larger token budget; records still
    missing afterwards are reported in `failures` rather than dropped silently --
    a silently dropped record becomes an invisible hole in every later statistic.
    """
    scores: dict[str, dict[str, float]] = {r["key"]: {} for r in records}
    failures: list[dict] = []
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

    for vendor in vendors:
        ok = 0
        for bi, batch in enumerate(batches):
            prompt = build_prompt(batch, rq, template)
            parsed: dict[int, float] = {}
            for attempt in range(retries + 1):
                budget = 3000 if attempt == 0 else 5000
                try:
                    parsed = parse_scores(call_vendor(vendor, prompt, max_tokens=budget), len(batch))
                except Exception as exc:
                    parsed = {}
                    err = f"{type(exc).__name__}: {exc}"
                else:
                    err = "unparseable reply"
                if parsed:
                    break
                if attempt < retries:
                    time.sleep(2.0)
            if parsed:
                for idx, sc in parsed.items():
                    scores[batch[idx]["key"]][vendor.name] = sc
                ok += len(parsed)
            else:
                failures.append({"vendor": vendor.name, "batch": bi,
                                 "keys": [r["key"] for r in batch], "error": err})
            time.sleep(pause)
        log(f"  {vendor.name:10s} {vendor.model:34s} scored {ok}/{len(records)}")
    return scores, failures


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def icc21(matrix: list[list[float]]) -> float:
    """ICC(2,1): agreement between raters treating both raters and records as random."""
    n = len(matrix)
    k = len(matrix[0]) if n else 0
    if n < 2 or k < 2:
        return float("nan")
    grand = sum(sum(r) for r in matrix) / (n * k)
    row_m = [sum(r) / k for r in matrix]
    col_m = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ms_r = k * sum((m - grand) ** 2 for m in row_m) / (n - 1)
    ms_c = n * sum((m - grand) ** 2 for m in col_m) / (k - 1)
    ss_e = sum((matrix[i][j] - row_m[i] - col_m[j] + grand) ** 2
               for i in range(n) for j in range(k))
    ms_e = ss_e / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return (ms_r - ms_e) / den if den else float("nan")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def pabak(a: list[int], b: list[int]) -> float:
    """Prevalence- and bias-adjusted kappa: robust when includes are rare."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    return 2 * po - 1


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def auc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUC: how well scores separate known-relevant from the rest."""
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_ci(fn, *samples, n_boot: int = 2000, seed: int = 7, alpha: float = 0.05):
    """Percentile bootstrap CI, resampling each sample independently."""
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        resampled = [[s[rng.randrange(len(s))] for _ in s] for s in samples]
        try:
            val = fn(*resampled)
        except (ZeroDivisionError, ValueError, statistics.StatisticsError):
            continue
        if not (isinstance(val, float) and math.isnan(val)):
            draws.append(val)
    if len(draws) < 20:
        return (float("nan"), float("nan"))
    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return (round(lo, 4), round(hi, 4))


# --------------------------------------------------------------------------
# Common-basis guard
# --------------------------------------------------------------------------


def common_basis(scores: dict[str, dict[str, float]], raters: list[str]) -> list[str]:
    """Keys scored by EVERY named rater.

    Every statistic below is computed on this subset only. Mixing records scored by
    one rater with records scored by three produces a number nobody can reproduce
    later -- exactly what happened to one case in the three-case study, whose
    reported AUC of 0.781 could not be recovered from its own saved scores (bases
    tried gave 0.58, 0.61 and 0.73). The guard is what makes a figure checkable.
    """
    return sorted(k for k, v in scores.items() if all(r in v for r in raters))


def _report_basis(scores, raters, label, log):
    keys = common_basis(scores, raters)
    log(f"{label}: {len(keys)} of {len(scores)} records scored by all of {raters}")
    return keys


# --------------------------------------------------------------------------
# The five validation designs
# --------------------------------------------------------------------------


def design_agreement(scores, *, log=print) -> dict:
    """1. Agreement, reported WITHIN and BETWEEN vendor families.

    The distinction is the point. Models from one family share training data and
    post-training conventions, so they can agree closely and be wrong together;
    within-family agreement cannot detect that. A reviewer asking "how do you know
    the ensemble is not systematically wrong?" is answered by the between-family
    number and by nothing else.
    """
    present = sorted({r for v in scores.values() for r in v})
    families: dict[str, list[str]] = {}
    for r in present:
        families.setdefault(family_of(r), []).append(r)

    out: dict = {"raters_present": present,
                 "families": families,
                 "n_families": len(families),
                 "within_family": {},
                 "between_family": {}}

    for family, raters in families.items():
        if len(raters) < 2:
            continue
        keys = _report_basis(scores, raters, f"  within {family}", log)
        if len(keys) < 10:
            continue
        matrix = [[scores[k][r] for r in raters] for k in keys]
        val = icc21(matrix)
        out["within_family"][family] = {
            "raters": raters, "n": len(keys), "icc": round(val, 4),
            "icc_ci95": bootstrap_ci(lambda m: icc21(m), matrix),
            "mean_range": round(statistics.mean(max(row) - min(row) for row in matrix), 3),
        }

    fam_names = sorted(families)
    for i, fa in enumerate(fam_names):
        for fb in fam_names[i + 1:]:
            ra, rb = families[fa], families[fb]
            keys = _report_basis(scores, ra + rb, f"  between {fa} and {fb}", log)
            if len(keys) < 10:
                continue
            xa = [statistics.mean(scores[k][r] for r in ra) for k in keys]
            xb = [statistics.mean(scores[k][r] for r in rb) for k in keys]
            out["between_family"][f"{fa}~{fb}"] = {
                "n": len(keys),
                "pearson": round(pearson(xa, xb), 4),
                "icc": round(icc21([[a, b] for a, b in zip(xa, xb, strict=True)]), 4),
                "mae": round(statistics.mean(abs(a - b) for a, b in zip(xa, xb, strict=True)), 3),
            }

    if len(families) < 2:
        out["warning"] = (
            "Only one vendor family is present, so every agreement figure here measures "
            "within-family consistency. This does NOT establish that the ensemble is right; "
            "add a vendor from a different family before reporting agreement as validation."
        )
    return out


def design_controls(scores, seed_keys, *, tau=7.0, log=print) -> dict:
    """2. Blind positive controls.

    Seed records are known relevant, because the reviewer chose them as seeds.
    Scored unmarked in ordinary batches, they should separate from the candidate
    pool. Reported per rater AND on the ensemble mean, on a common basis, so the
    figure is reproducible from the scores file alone.

    Boolean database exports admit off-topic records, so a low-scoring seed may be
    a genuinely off-topic seed rather than a screener failure. Both the raw AUC and
    the AUC excluding the lowest-scoring seeds are reported; inspect the per-seed
    list before concluding either way.
    """
    present = sorted({r for v in scores.values() for r in v})
    keys = common_basis(scores, present)
    log(f"  common basis: {len(keys)} records scored by all {len(present)} raters")
    seeds = [k for k in keys if k in seed_keys]
    cands = [k for k in keys if k not in seed_keys]
    if not seeds or not cands:
        return {"error": f"need seeds and candidates on a common basis; got "
                         f"{len(seeds)} seeds, {len(cands)} candidates",
                "hint": "seeds may have been gated out before the full ensemble scored them; "
                        "score seeds with every rater so they share the candidates' basis"}

    mean = lambda k: statistics.mean(scores[k].values())  # noqa: E731
    pos, neg = [mean(k) for k in seeds], [mean(k) for k in cands]
    per_seed = sorted(({"key": k, "score": round(mean(k), 2),
                        "per_rater": {r: scores[k][r] for r in present}} for k in seeds),
                      key=lambda d: d["score"])
    drop = {d["key"] for d in per_seed[:max(1, len(per_seed) // 10)]}
    pos_trim = [mean(k) for k in seeds if k not in drop]

    return {
        "n_seeds": len(seeds), "n_candidates": len(cands), "basis": present,
        "auc_ensemble_mean": round(auc(pos, neg), 4),
        "auc_ci95": bootstrap_ci(auc, pos, neg),
        "auc_excluding_lowest_seeds": round(auc(pos_trim, neg), 4),
        "n_seeds_excluded": len(drop),
        "auc_per_rater": {r: round(auc([scores[k][r] for k in seeds],
                                       [scores[k][r] for k in cands]), 4) for r in present},
        "seeds_at_or_above_tau": sum(1 for p in pos if p >= tau),
        "candidate_include_rate": round(sum(1 for q in neg if q >= tau) / len(neg), 4),
        "per_seed": per_seed,
        "note": "Inspect per_seed before drawing a conclusion: a Boolean export can admit "
                "off-topic seeds, and a low-scoring seed may be the export's fault, not the "
                "screener's. Report both AUC figures either way.",
    }


def design_threshold(scores, *, taus=(5.0, 6.0, 6.5, 7.0, 7.5, 8.0), log=print) -> dict:
    """3. Threshold sensitivity.

    A relevance score is continuous and the inclusion decision is a cut. In the
    three-case study, rewording the rubric while holding the criteria fixed left
    score correlation at 0.97 but moved the included set by 21%, and 13 of the 14
    affected records scored 6-8. The fragile component is the cut, not the ranking.
    Reporting a band across tau converts that from a weakness a reviewer finds into
    a sensitivity analysis the author bounded.
    """
    present = sorted({r for v in scores.values() for r in v})
    keys = common_basis(scores, present)
    mean = lambda k: statistics.mean(scores[k].values())  # noqa: E731
    means = [mean(k) for k in keys]
    log(f"  common basis: {len(keys)} records")
    rows = {}
    for tau in taus:
        inc = [k for k, m in zip(keys, means, strict=True) if m >= tau]
        per_rater = {r: sum(1 for k in keys if scores[k][r] >= tau) for r in present}
        rows[str(tau)] = {"included": len(inc),
                          "share": round(len(inc) / len(keys), 4) if keys else None,
                          "included_per_rater": per_rater,
                          "rater_disagreement": (max(per_rater.values()) - min(per_rater.values()))
                                                if per_rater else None}
    boundary = sum(1 for m in means if 6.0 <= m <= 8.0)
    return {"n": len(keys), "basis": present, "by_tau": rows,
            "records_in_boundary_zone_6_to_8": boundary,
            "boundary_share": round(boundary / len(keys), 4) if keys else None,
            "note": "Report the band, not a single tau. A large boundary share means the "
                    "included set is sensitive to the cut regardless of screener quality."}


def design_robustness(records, rq, vendors, *, batch_size=25, log=print) -> dict:
    """4. Prompt-wording and record-order robustness.

    Three conditions on the same records: the shipped rubric, an independently
    worded rubric with identical criteria, and the shipped rubric with records
    reversed within each batch. Test-retest separates sampling noise from the
    effect of the manipulation; without it a wording effect and ordinary
    variability are indistinguishable.
    """
    conditions = {}
    log("  condition: shipped rubric")
    base, f1 = score_batches(records, rq, vendors, batch_size=batch_size, log=log)
    log("  condition: paraphrased rubric (same criteria)")
    para, f2 = score_batches(records, rq, vendors, batch_size=batch_size,
                             template=RUBRIC_PARAPHRASE, log=log)
    log("  condition: reversed record order")
    rev_recs = []
    for i in range(0, len(records), batch_size):
        rev_recs.extend(reversed(records[i : i + batch_size]))
    rev, f3 = score_batches(rev_recs, rq, vendors, batch_size=batch_size, log=log)

    present = sorted({r for v in base.values() for r in v})
    keys = [k for k in common_basis(base, present)
            if k in common_basis(para, present) and k in common_basis(rev, present)]
    mean = lambda d, k: statistics.mean(d[k].values())  # noqa: E731
    for label, other in (("paraphrase", para), ("reversed_order", rev)):
        a = [mean(base, k) for k in keys]
        b = [mean(other, k) for k in keys]
        for tau in (6.0, 7.0, 8.0):
            da = [1 if x >= tau else 0 for x in a]
            db = [1 if x >= tau else 0 for x in b]
            conditions.setdefault(label, {})[f"tau_{tau}"] = {
                "kappa": round(cohen_kappa(da, db), 4),
                "included_shipped": sum(da),
                "included_variant": sum(db),
                "flips": sum(1 for x, y in zip(da, db, strict=True) if x != y),
                "flips_in_boundary_zone": sum(
                    1 for x, y, s in zip(da, db, a, strict=True) if x != y and 6.0 <= s <= 8.0),
            }
        conditions[label]["pearson"] = round(pearson(a, b), 4)
        conditions[label]["mae"] = round(statistics.mean(abs(x - y) for x, y in zip(a, b, strict=True)), 3)
    return {"n": len(keys), "basis": present, "conditions": conditions,
            "failures": f1 + f2 + f3,
            "scores": {"shipped": base, "paraphrase": para, "reversed": rev}}


STOPWORDS = frozenset(["a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to", "for", "with", "without", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "their", "his", "her", "our", "your", "my", "we", "you", "they", "he", "she", "i", "not", "no", "nor", "so", "than", "then", "there", "here", "when", "where", "which", "who", "whom", "whose", "what", "how", "why", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "too", "very", "can", "will", "just", "should", "now", "do", "does", "did", "doing", "have", "has", "had", "having", "would", "could", "may", "might", "must", "about", "above", "after", "again", "against", "because", "before", "below", "between", "during", "into", "through", "under", "over", "further", "once", "study", "research", "paper", "article", "analysis", "based", "using", "use", "used", "approach", "method", "results", "new", "toward", "towards", "case", "review", "effect", "effects", "impact", "role", "via", "within", "among", "across", "novel"])


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", (text or "").lower())
            if w not in STOPWORDS]


def _tfidf(docs: list[str]) -> list[dict[str, float]]:
    """L2-normalised tf-idf vectors. Pure stdlib, so this design needs no extra install."""
    from collections import Counter

    df: Counter = Counter()
    counted = []
    for d in docs:
        c = Counter(_tokens(d))
        counted.append(c)
        df.update(c.keys())
    n = len(docs)
    idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
    out = []
    for c in counted:
        v = {w: (1 + math.log(k)) * idf.get(w, 0.0) for w, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({w: x / norm for w, x in v.items()})
    return out


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _centroid(vecs: list[dict[str, float]]) -> dict[str, float]:
    from collections import Counter

    acc: Counter = Counter()
    for v in vecs:
        for w, x in v.items():
            acc[w] += x
    norm = math.sqrt(sum(x * x for x in acc.values())) or 1.0
    return {w: x / norm for w, x in acc.items()}


def design_bibliometric(scores, records, seed_records, *, tau=7.0, log=print) -> dict:
    """6. Convergent validity against non-LLM bibliometric signals.

    The other five designs all interrogate the screener with more model calls. This
    one asks a different question: does an independent signal, computed without any
    model, agree with the screener's decisions?

    The informative result is not a single number but a *pattern*. Two families of
    predictor are computed and reported separately:

      topical  -- tf-idf similarity of a record's title and abstract to the seed
                  set, and whether it appeared in a seed venue
      prestige -- citation count, reference-list length, publication year

    A screener judging topical relevance should track the topical predictors and
    ignore the prestige ones. A screener rewarding well-cited or well-connected
    work — plausible, since abstracts of highly cited papers read differently —
    would show the opposite, and no positive control detects that: a prestige-driven
    screener still scores known-relevant seeds highly, because seeds tend to be
    well-cited papers.

    Measured on the three-case data, the pattern held: topical centroid similarity
    gave AUC 0.65-0.77 across the corpora while citation count gave 0.47, at or
    below chance. Publication year was the exception at 0.68 in the corpus spanning
    1971-2025, where recency genuinely carries topical information.

    Inputs are the records themselves, so any metadata field the records carry can
    be used. `seed_records` are the review's seeds, which define the topic.
    """
    present = sorted({r for v in scores.values() for r in v})
    by_key = {r["key"]: r for r in records}
    # Two bases, and the difference between them matters. The strict common basis is
    # normally gate SURVIVORS, an already topically-enriched subset: restricting to it
    # removes the easy exclusions and a topical predictor's apparent discrimination
    # falls (measured across three corpora, centroid AUC dropped 0.66/0.77/0.65 on the
    # full pool to 0.60/0.71/0.51 on survivors). The full pool answers "does the
    # screener track topic at all"; the survivor subset answers "does it discriminate
    # among borderline records". Both are reported, each with its basis named.
    strict = [k for k in common_basis(scores, present) if k in by_key]
    full = [k for k in scores if scores[k] and k in by_key]
    keys = full if len(full) >= max(30, len(strict)) else strict
    basis_label = "all scored records (mean over whatever raters scored each)" \
        if keys is full else f"common basis, all of {present}"
    log(f"  full pool: {len(full)} records | strict common basis: {len(strict)}")
    log(f"  primary basis: {basis_label}")
    if len(keys) < 30:
        return {"error": f"need at least 30 records with scores and metadata, got {len(keys)}"}

    text = lambda r: f"{r.get('title') or ''} {r.get('abstract') or ''}"  # noqa: E731
    seed_texts = [text(r) for r in seed_records]
    vecs = _tfidf([text(by_key[k]) for k in keys] + seed_texts)
    cand_vecs, seed_vecs = vecs[: len(keys)], vecs[len(keys):]
    cent = _centroid(seed_vecs)
    seed_venues = {(r.get("venue") or "").lower() for r in seed_records} - {""}

    mean = lambda k: statistics.mean(scores[k].values())  # noqa: E731
    rows = {}
    for i, k in enumerate(keys):
        r = by_key[k]
        rows[k] = {
            "topic_centroid": _cosine(cand_vecs[i], cent),
            "topic_max_seed": max((_cosine(cand_vecs[i], sv) for sv in seed_vecs), default=0.0),
            "venue_is_seed_venue": 1 if (r.get("venue") or "").lower() in seed_venues else 0,
            "cited_by": r.get("cited_by_count"),
            "ref_count": r.get("reference_count"),
            "year": r.get("year"),
            "included": 1 if mean(k) >= tau else 0,
        }

    kinds = {"topic_centroid": "topical", "topic_max_seed": "topical",
             "venue_is_seed_venue": "topical", "cited_by": "prestige",
             "ref_count": "prestige", "year": "prestige"}
    per_pred = {}
    for field, kind in kinds.items():
        pos = [v[field] for v in rows.values() if v["included"] and v[field] is not None]
        neg = [v[field] for v in rows.values() if not v["included"] and v[field] is not None]
        if len(pos) < 5 or len(neg) < 5:
            per_pred[field] = {"kind": kind, "auc": None,
                               "reason": f"too few records ({len(pos)} included, {len(neg)} excluded)"}
            continue
        per_pred[field] = {"kind": kind, "auc": round(auc(pos, neg), 4),
                           "auc_ci95": bootstrap_ci(auc, pos, neg, n_boot=500),
                           "n_included": len(pos), "n_excluded": len(neg)}

    topical = [v["auc"] for f, v in per_pred.items()
               if v.get("kind") == "topical" and v.get("auc") is not None]
    prestige = [v["auc"] for f, v in per_pred.items()
                if v.get("kind") == "prestige" and v.get("auc") is not None]

    # Seed-level leave-one-out: does the non-LLM signal agree on which seed is off-topic?
    seed_loo = []
    if len(seed_vecs) >= 3:
        for i, r in enumerate(seed_records):
            others = [v for j, v in enumerate(seed_vecs) if j != i]
            seed_loo.append({"key": r.get("key"), "title": (r.get("title") or "")[:80],
                             "tfidf_loo": round(_cosine(seed_vecs[i], _centroid(others)), 4)})
        seed_loo.sort(key=lambda d: d["tfidf_loo"])

    # Rank agreement at the top of the list, where inclusion decisions are made
    top_n = min(250, max(20, len(keys) // 10))
    top_llm = {k for k in sorted(keys, key=lambda k: -mean(k))[:top_n]}
    top_tf = {k for k in sorted(keys, key=lambda k: -rows[k]["topic_centroid"])[:top_n]}

    # The same topical predictor on the other basis, so range restriction is visible
    # rather than hidden in whichever basis happened to be primary.
    other = strict if keys is full else full
    secondary = None
    if len(other) >= 30 and set(other) != set(keys):
        o_rows = {k: rows[k] for k in other if k in rows}
        if len(o_rows) < 30:
            ov = _tfidf([text(by_key[k]) for k in other] + seed_texts)
            oc = _centroid(ov[len(other):])
            o_rows = {k: {"topic_centroid": _cosine(ov[i], oc),
                          "included": 1 if mean(k) >= tau else 0}
                      for i, k in enumerate(other)}
        pos = [v["topic_centroid"] for v in o_rows.values() if v["included"]]
        neg = [v["topic_centroid"] for v in o_rows.values() if not v["included"]]
        if len(pos) >= 5 and len(neg) >= 5:
            secondary = {"basis": "common basis (gate survivors)" if keys is full
                         else "all scored records",
                         "n": len(o_rows), "topic_centroid_auc": round(auc(pos, neg), 4)}

    return {
        "n": len(keys), "basis": basis_label, "raters": present, "tau": tau,
        "n_full_pool": len(full), "n_strict_common_basis": len(strict),
        "secondary_basis": secondary,
        "per_predictor": per_pred,
        "topical_mean_auc": round(statistics.mean(topical), 4) if topical else None,
        "prestige_mean_auc": round(statistics.mean(prestige), 4) if prestige else None,
        "top_rank_agreement": {"top_n": top_n, "overlap": len(top_llm & top_tf),
                               "share": round(len(top_llm & top_tf) / top_n, 4),
                               "chance": round(top_n / len(keys), 4)},
        "seed_leave_one_out": seed_loo,
        "interpretation": (
            "Read the PATTERN, not one number. Topical predictors above chance with prestige "
            "predictors at chance is evidence the screener judges topical relevance -- the "
            "specific failure a positive control cannot detect, because a prestige-driven "
            "screener also scores seeds highly. Prestige predictors ABOVE the topical ones "
            "would mean the scores track citation counts or venue standing, and the screening "
            "should not be trusted. Substantial-but-partial top-rank agreement is the expected "
            "result: identical rankings would mean the model added nothing over tf-idf."),
    }


def design_negative(records, rq, wrong_rq, vendors, *, tau=7.0, batch_size=25, log=print) -> dict:
    """5. Negative control: a plausible but WRONG research question.

    Positive controls show the screener recognises relevant work. They do not show
    it is responding to the question rather than to generic markers of academic
    quality -- a screener that rewards well-written abstracts would pass a positive
    control. Under a wrong question from a different review, includes should
    collapse. If they do not, the scores are not measuring topical relevance.
    """
    log("  condition: correct research question")
    right, f1 = score_batches(records, rq, vendors, batch_size=batch_size, log=log)
    log("  condition: wrong research question (different review)")
    wrong, f2 = score_batches(records, wrong_rq, vendors, batch_size=batch_size, log=log)

    present = sorted({r for v in right.values() for r in v})
    keys = [k for k in common_basis(right, present) if k in common_basis(wrong, present)]
    mean = lambda d, k: statistics.mean(d[k].values())  # noqa: E731
    a = [mean(right, k) for k in keys]
    b = [mean(wrong, k) for k in keys]
    inc_r = sum(1 for x in a if x >= tau)
    inc_w = sum(1 for x in b if x >= tau)
    return {
        "n": len(keys), "basis": present,
        "correct_question": {"included": inc_r, "mean_score": round(statistics.mean(a), 3)},
        "wrong_question": {"included": inc_w, "mean_score": round(statistics.mean(b), 3)},
        "include_collapse": round(1 - inc_w / inc_r, 4) if inc_r else None,
        "score_drop": round(statistics.mean(a) - statistics.mean(b), 3),
        "pearson_between_questions": round(pearson(a, b), 4),
        "failures": f1 + f2,
        "interpretation": (
            "A large include_collapse and a low pearson_between_questions mean the screener "
            "responds to the question. A small collapse means it is rewarding something else "
            "-- abstract quality, venue prestige, recency -- and the scores do not measure "
            "topical relevance."),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _read_json(path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, obj, log=print) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float),
                          encoding="utf-8")
    log(f"wrote {path}")


def _seed_keys(seeds_path, case) -> set[str]:
    """Seed keys from cases/seed_selection.json, or a plain JSON list of keys/DOIs."""
    data = _read_json(seeds_path)
    if isinstance(data, dict) and case and case in data:
        raw = [s["doi"] if isinstance(s, dict) else s for s in data[case]["seeds"]]
    elif isinstance(data, list):
        raw = [s["key"] if isinstance(s, dict) else s for s in data]
    else:
        raise SystemExit(f"cannot read seeds from {seeds_path}; pass --case for a keyed file")
    out = set()
    for r in raw:
        r = str(r).strip()
        out.add(r)
        out.add(r if r.startswith("doi:") else f"doi:{r.lower().removeprefix('https://doi.org/')}")
    return out


def _pick_vendors(args, log=print) -> list[Vendor]:
    avail = available_vendors()
    if args.vendors:
        wanted = {v.strip() for v in args.vendors.split(",")}
        avail = [v for v in avail if v.name in wanted]
    if not avail:
        raise SystemExit(
            "No vendor API keys found in the environment. Set at least one of: "
            + ", ".join(v.env_key for v in VENDORS)
            + "  (a .env file in the working directory is also read)")
    fams = {v.family for v in avail}
    log(f"vendors: {[v.name for v in avail]}  ({len(fams)} distinct families)")
    if len(fams) < 2:
        log("WARNING: one vendor family only. Agreement figures will measure within-family "
            "consistency, which does not establish that the ensemble is right.")
    return avail


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, *, needs_records=False, needs_scores=False):
        if needs_records:
            p.add_argument("--records", required=True)
            p.add_argument("--question", required=True)
            p.add_argument("--vendors", help="comma-separated subset, e.g. openai,google")
            p.add_argument("--batch-size", type=int, default=25)
            p.add_argument("--limit", type=int, help="score only the first N records")
        if needs_scores:
            p.add_argument("--scores", required=True)
        p.add_argument("--out")
        p.add_argument("--tau", type=float, default=7.0)

    common(sub.add_parser("score", help="score records with every available vendor"),
           needs_records=True)
    common(sub.add_parser("agreement", help="design 1: within- vs between-family agreement"),
           needs_scores=True)
    p = sub.add_parser("controls", help="design 2: blind positive controls")
    common(p, needs_scores=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--case")
    sub.add_parser("check-keys",
                   help="list the vendors whose keys are set, and how many families they span "
                        "(no API calls, no cost)")
    common(sub.add_parser("threshold", help="design 3: threshold sensitivity band"),
           needs_scores=True)
    p = sub.add_parser("bibliometric",
                       help="design 6: convergent validity vs non-LLM signals (no API calls)")
    common(p, needs_scores=True)
    p.add_argument("--records", required=True,
                   help="records with metadata: title, abstract, venue, cited_by_count, "
                        "reference_count, year")
    p.add_argument("--seed-records", required=True,
                   help="the review's seed records, in the same format")
    common(sub.add_parser("robustness", help="design 4: wording and order robustness"),
           needs_records=True)
    p = sub.add_parser("negative", help="design 5: wrong-question negative control")
    common(p, needs_records=True)
    p.add_argument("--wrong-question", required=True)
    p = sub.add_parser("validate", help="designs 1-3 in one pass (no API calls)")
    common(p, needs_scores=True)
    p.add_argument("--seeds")
    p.add_argument("--case")

    args = ap.parse_args(argv)
    log = print

    if args.cmd == "check-keys":
        avail = available_vendors()
        fams = {v.family for v in avail}
        log("vendor           family      model                              key")
        for v in VENDORS:
            has = "set" if os.environ.get(v.env_key) else "-- missing --"
            log(f"{v.name:16s} {v.family:11s} {v.model:34s} {has}")
        log(f"\n{len(avail)} vendor(s) usable, spanning {len(fams)} distinct famil"
            f"{'y' if len(fams) == 1 else 'ies'}: {sorted(fams)}")
        if len(fams) < 2:
            log("\nWARNING: fewer than two families. Agreement measured on this set would be "
                "within-family consistency, which is what the three-case study already reported "
                "and what a reviewer will challenge. Add a vendor from a different family.")
        else:
            log("\nReady: between-family agreement is computable, which is the claim that "
                "within-family agreement cannot make.")
        return

    if args.cmd == "score":
        # Vendors first: a missing API key should say so, not surface as a parse error
        # from whatever file happened to be read before the check.
        vendors = _pick_vendors(args, log)
        records = _read_json(args.records)
        if args.limit:
            records = records[: args.limit]
        rq = _read_json(args.question)
        log(f"scoring {len(records)} records")
        scores, failures = score_batches(records, rq, vendors,
                                         batch_size=args.batch_size, log=log)
        cov = {}
        for v in scores.values():
            cov[len(v)] = cov.get(len(v), 0) + 1
        log(f"coverage by rater count: {dict(sorted(cov.items()))}")
        if failures:
            log(f"WARNING: {len(failures)} batch failures; affected records are recorded "
                f"under 'failures' and are EXCLUDED from every common-basis statistic")
        _write(args.out or "scores.json",
               {"scores": scores, "failures": failures,
                "vendors": [{"name": v.name, "family": v.family, "model": v.model}
                            for v in vendors]}, log)
        return

    if args.cmd in {"agreement", "controls", "threshold", "validate", "bibliometric"}:
        raw = _read_json(args.scores)
        scores = raw["scores"] if isinstance(raw, dict) and "scores" in raw else raw
        scores = {k: v for k, v in scores.items() if isinstance(v, dict) and v}
        log(f"loaded {len(scores)} scored records")

        if args.cmd == "agreement":
            _write(args.out or "agreement.json", design_agreement(scores, log=log), log)
        elif args.cmd == "controls":
            res = design_controls(scores, _seed_keys(args.seeds, args.case),
                                  tau=args.tau, log=log)
            _write(args.out or "controls.json", res, log)
        elif args.cmd == "threshold":
            _write(args.out or "threshold.json", design_threshold(scores, log=log), log)
        elif args.cmd == "bibliometric":
            res = design_bibliometric(scores, _read_json(args.records),
                                      _read_json(args.seed_records), tau=args.tau, log=log)
            _write(args.out or "bibliometric.json", res, log)
        else:
            out = {"agreement": design_agreement(scores, log=log),
                   "threshold": design_threshold(scores, log=log)}
            if args.seeds:
                out["controls"] = design_controls(scores, _seed_keys(args.seeds, args.case),
                                                  tau=args.tau, log=log)
            _write(args.out or "validation.json", out, log)
        return

    vendors = _pick_vendors(args, log)
    records = _read_json(args.records)
    if args.limit:
        records = records[: args.limit]
    rq = _read_json(args.question)

    if args.cmd == "robustness":
        res = design_robustness(records, rq, vendors, batch_size=args.batch_size, log=log)
        _write(args.out or "robustness.json", res, log)
    elif args.cmd == "negative":
        res = design_negative(records, rq, _read_json(args.wrong_question), vendors,
                              tau=args.tau, batch_size=args.batch_size, log=log)
        _write(args.out or "negative_control.json", res, log)


if __name__ == "__main__":
    sys.exit(main())
