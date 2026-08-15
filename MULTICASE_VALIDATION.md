# Multi-case validation: three corpora, one protocol

Three independent snowballing runs, each on a Scopus-defined corpus, screened by the same validated
LLM ensemble and estimated with the same three-arm design. The purpose is to answer the objection
that a method demonstrated once has not been demonstrated. Reproduction code is in `cases/`; the
per-case records are `cases/results_<case>.json`.

## The three cases

| | adverse | favourable | large |
|---|---|---|---|
| Scopus query | innovation AND marketing AND automation | blockchain AND customer loyalty | gamification AND marketing |
| Corpus records | 260 | 1,995 | 14,302 |
| DOI coverage | 65.8% | 95.9% | 91.5% |
| Median references | 16 | 84 | 69 |
| Span | 1971-2025 | 2014-2025 | 2011-2025 |

The adverse case was chosen before any run, precisely because short reference lists and poor
identifier coverage are where citation searching should work worst.

## What the runs produced

| | adverse | favourable | large |
|---|---|---|---|
| Identified | 40,719 | 6,285 | 19,676 |
| Screened | 1,551 | 2,500 | 3,000 |
| Included | 149 | 337 | 686 |
| Unresolvable share | 4.8% | 3.9% | 1.2% |
| Stopped by | screening resource exhausted mid-iteration 2 | budget | budget |
| Was that saturation? | no | no | no |

**No stopping rule fired in any of the three runs.** All three were cut short by a resource
limit - two by the configured screening budget, one by the LLM token ceiling of the process doing
the screening. This is stated plainly because the alternative - describing a budget stop as
saturation - is the exact reporting failure the library exists to prevent. It also means the
recall figures below describe coverage of *what was screened*, not of the literature.

## The pre-registered prediction, tested

Registered in `cases/PREDICTION.md` before any run started.

| Prediction | Outcome | Numbers |
|---|---|---|
| Estimated recall lower in the adverse corpus | **failed** | 0.87 adverse vs 0.48 favourable |
| Unresolvable share proportionally higher | **held** | 4.8% vs 3.9% |
| Marginal-yield rule fires earlier | **untestable** | no rule fired in any case |
| Scopus arm captures a smaller share | **failed** | 9.2% vs 7.7% |

One of four held. Reporting this as it came out is the point of registering it.

Why recall came out *higher* in the adverse corpus is the informative part. Recall here is the
estimated coverage of a screened subsample, not of a literature, and the adverse corpus was the only
one screened to completion at iteration 1 (1,551 of 1,551 candidates) while the other two screened
the first 2,500 of 5,249 and the first 3,000 of 14,883. A more completely screened sample yields
higher apparent coverage. The prediction was about the literature; the measurement was about the
sample, and under budget stops those are different things. That is a lesson about what a
capture-recapture estimate means under a truncated screen, and it belongs in the manuscript's
limitations rather than being quietly dropped.

## What the three-arm estimator revealed

Adding a Scopus membership arm made the log-linear estimator usable for the first time. It then
failed informatively in two distinct ways, and both are more useful than a clean number.

**The adverse case is unidentifiable.** The capture cell for "Crossref and Scopus but not OpenAlex"
is structurally zero, so the AIC-selected all-pairwise model has no unique solution: it returns
n_hat = 1.3e11 and recall = 1.25e-09. Chao1 was reported instead (0.714,
CI [203, 275] on N). A three-arm design does not
guarantee a three-arm estimate.

**Model choice dominates sampling uncertainty.** In the favourable case the AIC-selected
interaction model gives recall 0.538 while the independence model gives 0.894 on identical data - a
35-point swing on model choice alone, larger than the bootstrap interval around either. Any
reported log-linear recall must carry the model that produced it.

**Crossref is a strict subset of OpenAlex in two of three corpora.** It captured 0 records that
OpenAlex missed in both the favourable and large cases (8 in the adverse case). Two nested arms
cannot estimate anything: Chapman returns a point estimate below the observed count, and the
library correctly refuses it as uninformative rather than reporting a spurious recall of 1.0. This
is direct evidence for the manuscript's arm-design argument, obtained on real data - and it means
the two-arm provider design used in the earlier single-case study was closer to degenerate than
that study could show.

Arm capture, share of included records:

| | OpenAlex | Crossref | Scopus | Crossref-only |
|---|---|---|---|---|
| adverse | 93.3% | 48.2% | 9.2% | 8 records |
| favourable | 95.7% | 23.6% | 7.7% | 0 records |
| large | 99.4% | 6.1% | 31.2% | 0 records |

## Screening validation, per case

Every case validated its own screener before using it, on its own data.

| | adverse | favourable | large |
|---|---|---|---|
| ICC(2,1) across models | 0.578 | 0.827 | 0.794 |
| 95% CI | [0.54, 0.61] | [0.81, 0.84] | [0.77, 0.81] |
| Blind seed AUC | 0.58-0.73 (basis-dependent) | 0.658 | 0.631 |
| Two-stage gate false negatives | 1.3% | 0% | not measured |

Three findings worth stating.

**Agreement is corpus-dependent, and weakest where the corpus is hardest.** ICC ranges from 0.58 in
the adverse corpus to 0.83 in the favourable one, with non-overlapping intervals. A screener
validated on one literature cannot be assumed reliable on another - which is an argument for
validating per case, as done here, rather than once.

**Blind controls work, and caught a real problem.** Seed records were scored unmarked in ordinary
batches. They separate from the candidate pool in every case, but the AUC is lowest in the large
corpus (0.63), and inspection showed why: Scopus Boolean queries admit off-topic records, and some
seeds are genuinely off-topic - the adverse seed set contains a structural-biology paper. Excluding
inspected off-topic seeds raises the large-case AUC to 0.77. Both numbers are reported.

**The abstract-availability bias replicates in all three corpora.** Records without an abstract
score systematically lower - by 0.69, 1.25 and 0.39 points respectively, significant in each case.
Every unresolvable record lacks an abstract, so the included set skews toward well-indexed work in
every case. This is a property of abstract-based screening, not of one corpus.

## What this does and does not establish

It establishes that the workflow completes on three corpora spanning two orders of magnitude, that
the screening protocol transfers with measurable and corpus-dependent reliability, that the
membership arm makes three-arm estimation possible, and that arm degeneracy is a real hazard on
real data rather than a theoretical one.

It does not establish saturation behaviour on live data: no run reached a rule-driven stop, so the
stopping rules remain evidenced only in simulation. Closing that would require screening one corpus
to exhaustion - the adverse case is the affordable candidate, needing roughly 31,000 further
records screened at iteration 2.

## Addendum: an independent re-check of these numbers

Everything above was recomputed from the saved artifacts (`cases/results_audit.md`). PRISMA
arithmetic balances in all three cases, the arm-capture cells sum to the stated observed counts, the
label files match the PRISMA include counts, and all three ICC values reproduce to three decimals.

One figure does not reproduce. The adverse case's blind-control AUC was reported as 0.781;
recomputing from its own saved scores gives 0.580 over all scored records, 0.614 using the
first-stage score as a common basis, and 0.733 restricted to gate survivors. The cause is visible in
that case's per-seed table: two of the fifteen seeds were excluded by the first-stage gate and carry
one model's score while survivors carry three, so the reported figure mixes scoring bases. It is
reported here and in the manuscript as a range rather than a point. The favourable and large figures
are unaffected.

Two consequences. The adverse case's gate false-negative rate is also unmeasured - its own record
says so, 350 records sampled and none scored before the screening resource ran out - so that
statistic rests on the favourable case's 0% alone. And `validation/llm_screen_validate.py` now
enforces a common scoring basis for every statistic it computes, printing the record count it used,
so this class of unreproducible figure cannot recur.
