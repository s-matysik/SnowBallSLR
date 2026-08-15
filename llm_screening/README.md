# LLM-assisted screening of the citation-search output

This directory records how the 302 iteration-1 candidates and 9,430 iteration-2 candidates of
the `ai-marketing-automation-slr` run were screened, so the decisions in `../run_llm/` are
auditable. It is **not** part of the SnowBallSLR library: the library deliberately does not
screen, and this is the external screener that supplied its labels file.

## Provenance of the method

The protocol is adapted from the author's own `ewaluator.ipynb` (six commercial LLMs scoring
relevance 1-10 in batches of 20 at temperature 0, one CSV per model). Three changes were made.

1. **The research question was derived from the data, not assumed.** The run's config carried a
   name left over from an unrelated project, so the question was inferred from the 15 seed records
   and is recorded verbatim in `research_question.json`. Every score is conditional on it.
2. **Models are reached through the platform, not through embedded keys.** The notebook hard-codes
   live API keys for six providers in its source. Those keys are a standing exposure and the
   notebook is now excluded from version control (`.gitignore`); nothing here reads them.
3. **Scoring is two-stage**, because the iteration-2 pool is 31x larger than iteration 1 and a
   full ensemble over it is wasteful. Stage 1 scores every record with a small model; only records
   at or above the gate are re-scored by two larger models. See the validation below.

## The ensemble

Iteration 1 (302 records) was scored by four models independently -- `claude-opus-5`,
`claude-sonnet-5`, `claude-opus-4-8`, `claude-haiku-4-5` -- giving a full reference matrix.

| Property | Value |
|---|---|
| Inter-model agreement (ICC over 4 models) | 0.923 |
| Pairwise Spearman rho | 0.920 - 0.960 |
| Mean within-record SD | 0.51 points |
| Records where models span >= 4 points | 2 of 302 |

**Blind positive controls.** The 15 seed records -- known relevant, since the reviewer chose them
-- were scored in the same batches without being marked. Mean 7.80 against 5.06 for the candidate
pool (Mann-Whitney p = 1.0e-05, AUC 0.826). The three lowest-scoring seeds (4.25, 5.25, 5.50) are
exactly the three an independent reading identified as peripheral to the question, which is the
behaviour a working screener should show.

**Threshold.** tau = 7.0 on the mean score. It is not arbitrary: the twelve coherent seeds all
score >= 7.25, and tau = 7 reproduces 80% of the seed set while admitting 29% of candidates.

## Two-stage design, validated before use

Measured on the 302 iteration-1 records, for which the full 4-model matrix exists:

| Design | rho vs full | MAE | kappa at tau=7 |
|---|---|---|---|
| haiku alone | 0.985 | 0.37 | 0.830 |
| sonnet-5 + opus-4-8 | 0.994 | 0.26 | 0.935 |
| sonnet-5 + opus-4-8 + haiku | 0.998 | 0.12 | 0.984 |
| two-stage (gate haiku >= 5, then sonnet-5 + opus-4-8) | -- | -- | 0.935 |

The gate is safe in the direction that matters: **at every cutoff from 3 to 6, zero
full-ensemble includes were screened out** (stage-1 recall 1.000). The gate can only cost
precision, which stage 2 then recovers.

## Known measurement bias -- read before using the scores

Records **without an abstract score about one point lower** than records with one
(4.60 vs 5.57, p = 4.6e-04). All 105 unresolved records in iteration 1 fell in that group. This is
a property of the evidence available to the screener, not of the studies' relevance, and it means
the include set is biased toward well-indexed records. The library's own PRISMA output counts
these separately; a manual pass over high-scoring title-only records is the correct remedy and
has not been performed here.

Two further limits. No human-agreement coefficient is reported, and that is a scope decision rather
than a pending measurement: the only screener available was the author, who set the question, seeds
and rubric the models were given, so the coefficient would measure fidelity to that framing rather
than agreement with independent expert judgement - which is why screening guidance requires two
independent reviewers. What the controls do establish is that the ensemble separates known-relevant
seeds from the candidate pool. The tooling for a genuine comparison ships here
(`human_validation_sample.csv` plus `../validation/score_human_agreement.py`) and needs a screener
who did not author the protocol.

Separately, the four models used for the three case runs are all from one vendor, so they are not
independent raters in the sense capture-recapture assumes. That limit has since been tested rather
than only stated: `../validation/llm_screen_validate.py` re-scored 311 records with six models from
six independent vendors, giving cross-family ICC of 0.70 to 0.90 across all fifteen pairs (see
`../cases/xvendor_validation.json`), so the convergence is not an artefact of shared training
provenance.

## Files

- `research_question.json` -- the question, core relation, and in/out-of-scope lists used in every prompt
- `screening_spec.json` -- the exact protocol handed to the ten parallel screening workers
- `ensemble_scores.json` -- per-model scores for iteration 1, plus the blind seed controls
- `iter2_scores.json` -- merged per-record scores for iteration 2
- `../labels_llm_001.csv`, `../labels_llm_002.csv` -- the labels files ingested by the run
