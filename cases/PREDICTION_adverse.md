# Pre-registered prediction — adverse case

**Case:** `adverse` — Scopus query `"innovation" AND "marketing" AND "automation"`
**Registered:** 2026-08-13T01:17:50Z (UTC, machine clock), before any `label`, `step` or
`report` action was invoked on `run_adverse/`. The run's `init` had already been performed by
the study harness at 2026-08-13T01:08:16Z; no screening decision existed at registration time
(`state.json`: `phase = awaiting_labels`, `decisions = 15` — the 15 seeds only, `pending = 1551`).
Repo HEAD at registration: `770f4f8`.

This file is the case-level instance of the study-wide prediction in `PREDICTION.md`. It is
recorded separately so that the adverse case's numbers can be checked against a claim that
demonstrably predates them.

## Corpus properties fixed in advance

| Property | Value |
|---|---|
| Scopus export records | 260 |
| DOI coverage in export | 65.8% |
| Median references per record | 16 |
| Publication span | 1971–2025 |
| Seeds | 15 (DOI-resolved, all 15 resolved) |
| Iteration-1 candidates | 1,551 |
| Screening budget (cumulative screened) | 4,000 |
| Inclusion threshold | tau = 7.0 (mean ensemble score) |
| Max iterations | 4 |

The budget is set deliberately above the iteration-1 pool so that the *budget* rule is not the
first rule with an opportunity to fire. Whether it nonetheless fires is one of the things being
measured.

## Predictions (directional, not numeric)

Relative to the favourable case (`mid`: blockchain x customer loyalty; 1,995 records, 95.9% DOI
coverage, median 84 references):

1. **Estimated recall will be lower.** A median of 16 references per record means fewer edges per
   node, so a larger share of the relevant population is unreachable by chasing citations at all.
2. **The unresolvable share of identified records will be proportionally higher.** 65.8% DOI
   coverage means more retrieved references carry no persistent identifier; unresolvable records
   are retained and screened but are terminal nodes and are never expanded.
3. **The marginal-yield rule will fire earlier** — in fewer iterations — because the frontier
   exhausts sooner. (Exhaustion firing before marginal yield would be consistent with the same
   underlying mechanism; budget firing first would not.)
4. **The Scopus membership arm will capture a smaller share of included records**, bounded above
   by the export's 65.8% DOI coverage: an arm keyed on DOI cannot see a record that has none.

## Screening-validation expectations (registered, weaker confidence)

5. Records lacking an abstract will score **lower** on average than records with one. 626 of the
   1,551 iteration-1 candidates (40.4%) carry no abstract, so if this bias exists it will push the
   included set toward well-indexed work. Two-sided test, alpha = 0.05.
6. Blind seed AUC will be **lower than in the other two cases**. The seed set is known to contain
   at least one record that the Boolean query admitted off-topic ("Structural Biology and Drug
   Discovery", 2006), and two further seeds sit far from any marketing construct ("Progress toward
   the factory of the future", 1984; "Investigating Factors Affecting the Uptake of Automated
   Assessment Technology", 2011). A depressed AUC here is therefore predicted to reflect *seed*
   off-topicness rather than screener failure, and the control analysis is expected to identify
   which seeds are responsible. This distinction is the point of scoring the seeds blind; it will
   be reported per-seed, and no seed will be dropped from the analysis.
7. The two-stage gate (haiku >= 5.0) will lose **fewer than 5%** of the records the full stage-2
   ensemble would have included, measured on >= 300 randomly sampled gate-excluded records.

## Deviations from the harness defaults, declared in advance

- `run_adverse/config.yaml` was written by `init` with `stopping.budget.max_screened = 1200`,
  which predates the case budget of 4,000 recorded in `cases/seed_selection.json`
  (`driver.py` rebuilds the case config but `Run.load` reads the run directory's own
  `config.yaml`). With 1,551 iteration-1 candidates, leaving 1,200 in place would guarantee a
  budget stop at iteration 1 and would destroy the test of prediction 3. The value is corrected
  to 4,000 in `run_adverse/config.yaml` before the first `label`, and only that field is changed.
  This is recorded here rather than applied silently, and the resulting `config_hash` change is
  expected.

## What would refute the design rather than the prediction

If the three-arm log-linear estimator is not estimable in this case at all — for example because
the Scopus arm captures too few included records to fit an interaction term — then the adverse
case cannot speak to arm dependence, and that is a limitation of the estimator's applicability
rather than evidence about recall. It will be reported as such.
