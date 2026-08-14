# Pre-registered prediction

Registered 2026-08-13T00:51:22+00:00, before any of the three
runs was started. Recorded here so the comparison in the report is a test of a prior claim rather
than a description of an outcome.

## Corpora, and why these three

| Case | Records | DOI coverage | Median references | Role |
|---|---|---|---|---|
| innovation x marketing x automation | 260 | 65.8% | 16 | adverse |
| blockchain x customer loyalty | 1,995 | 95.9% | 84 | favourable |
| gamification x marketing | 14,302 | 91.5% | 69 | large, budget-bound |

The adverse case was chosen deliberately: a literature with short reference lists and incomplete
identifier coverage is where citation searching should work worst. A method demonstrated only where
it works well is not demonstrated.

## The prediction

In the adverse corpus, relative to the favourable one:

1. **Estimated recall will be lower.** Short reference lists mean fewer edges per record, so a
   larger share of the relevant population is unreachable by chasing at all.
2. **The unresolvable share will be proportionally higher.** 65.8% DOI coverage against 95.9%
   means more records cannot be resolved to a persistent identifier, and unresolved records are
   never expanded.
3. **The marginal-yield rule will fire earlier**, in fewer iterations, because the frontier
   exhausts sooner.
4. **The Scopus arm will capture a smaller share of included records**, bounded above by that
   corpus's 65.8% DOI coverage: an arm keyed on DOI cannot see a record that has none.

A directional prediction, not a numeric one. If any of the four fails, the report says so.

## What would refute the design rather than the prediction

If the three-arm log-linear estimator selects an independence model in every case, arm dependence
is not the concern the manuscript treats it as. If it selects a dependence model everywhere, the
two-arm estimates reported previously are biased and must be reported as upper bounds.
