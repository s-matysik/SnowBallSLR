# Empirical validation of SnowBallSLR

Validation study run 2026-08-09 against SnowBallSLR v1.0.0. Every number below was
computed in this study; the code that produced them is saved alongside this report
(`validation_generator.py`, `validation_experiments.py`). All experiments are
offline and deterministic - no experiment touches a live API.

---

## 1. What was tested, and against what ground truth

The library's headline contributions are (i) quantitative stopping rules and (ii)
capture-recapture recall estimation. Neither can be validated on a real review,
because a real review has no ground truth: you cannot measure the recall of a
search whose target set is what you were trying to find. The study therefore uses
two complementary sources of evidence.

**A synthetic citation-graph benchmark with known ground truth.** A generator
produces temporally-ordered citation DAGs in which the relevant population is known
by construction. Three properties are modelled because each one drives a result
reported below:

- **Temporal acyclicity.** A work cites only works published no later than itself.
  This is why backward and forward expansion sample near-disjoint strata (§3).
- **Topic homophily.** Relevant works cite relevant works with elevated
  probability; reference-list *size* is sampled independently of the mix, so
  navigability and reference count are controlled separately.
- **A reachability ceiling.** A tunable share of relevant works has no citation
  path to the rest. Real reviews have this ceiling, and a benchmark without one
  flatters every stopping rule. Across the runs reported here the mean achievable
  recall was **0.913**, not 1.0 - every rule is scored against that ceiling,
  never against 1.0.

Design: 3 graph regimes (dense / sparse / diffuse) × 6 graphs per regime × 3 seed
strategies (random / most-cited / oldest) × 3 seed sizes (3 / 5 / 10), giving
**270 trials**, each run to exhaustion so that every rule can be scored at the
iteration it would have fired.

**The author's own real run.** 493 records from 15 seeds over OpenAlex and
Crossref, used for the arm-structure analysis (§3), the deduplication audit (§5)
and the verification audit (§6). This run is real data with no ground truth, so it
is used to establish *structure*, never accuracy.

---

## 2. Stopping rules: the flagship rule stopped at 45% recall

Each rule is scored at the iteration it fires: the true recall achieved at that
point, the shortfall against the achievable ceiling, and the number of relevant
studies left unfound.

| Rule | Fires | Mean true recall at stop | Mean shortfall vs ceiling | Mean studies missed |
|---|---|---|---|---|
| estimated_recall, unguarded (before fix) | 270/270 | 0.450 | 0.463 | 73.7 |
| estimated_recall, closure guard (after fix) | 127/270 | 0.906 | 0.000 | 0.0 |
| marginal_yield (eps=0.01, k=2) | 17/270 | 0.921 | 0.000 | 0.0 |
| asymptotic_coverage (tau=0.95) | 3/270 | 0.924 | 0.000 | 0.0 |
| budget (max_screened=2500) | 0/0 | never fired | - | - |
| exhaustion | 270/270 | 0.913 | 0.000 | 0.0 |

**The unguarded `estimated_recall` rule fired in every one of the 270 trials at a
mean true recall of 0.450** - a 46-percentage-point shortfall, leaving a mean of
73.7 relevant studies unfound. It was the worst rule in the suite while being the
one the README leads with.

The cause is a **closure violation**, not an implementation error.
Capture-recapture assumes a closed population. Snowballing enlarges the reachable
population every time an included record enters the frontier, so an estimate at
iteration *i* describes the population reachable *so far*. Early in a run the arms
agree almost perfectly on a small reachable set, N̂ collapses onto the observed
count, and estimated recall approaches 1.0 while true recall is still low. The
lower confidence bound does not protect against this: the interval is narrow
precisely because the arms agree.

### Mitigations tested

Four candidate guards were compared on a common subset of trials:

| Guard | Mean true recall at stop | Shortfall | Studies missed | Mean records screened |
|---|---|---|---|---|
| M0 baseline | 0.403 | 0.511 | 81.6 | 221 |
| M1 min_iter>=3 | 0.880 | 0.034 | 6.1 | 1306 |
| M2 N_hat stable | 0.913 | 0.001 | 0.1 | 1421 |
| M3 frontier gate | 0.914 | 0.000 | 0.0 | 1423 |

`M0` is the shipped behaviour. `M1` (a minimum iteration count) is crude but
recovers most of the loss. **`M2` - requiring N̂ to be stable within 5% for two
consecutive iterations - removes the shortfall almost entirely (0.001) while still
stopping earlier than exhaustion**, and it is the one adopted, because it is stated
in the estimator's own terms rather than as an arbitrary iteration count. `M3`
(frontier-growth gate) performs marginally better but converges on exhaustion,
which defeats the purpose of having a rule.

**After the fix**, on the full benchmark, `estimated_recall` fires in 127/270
trials at a mean true recall of **0.906**, shortfall **0.000**, studies missed
**0.0**. It abstains in the remaining trials rather than firing wrongly.

---

## 3. Arm design: the shipped default could not produce an estimate at all

Capture-recapture requires two arms that can both capture the same record. The
shipped default paired the **backward** and **forward** directions. On the author's
own real run:

| Arm design | Arm 1 | Arm 2 | Overlap m | Estimable? |
|---|---|---|---|---|
| direction (backward / forward) - shipped default | 362 | 116 | **0** | No |
| provider (OpenAlex / Crossref) | 338 | 312 | **172** | Yes |

**The flagship feature was not estimable on the author's own data.** This is
structural, not accidental: a citation graph is temporally acyclic, so a work older
than the seed set is normally reachable only backward and a newer one only forward.
The real run bears this out - median seed year 2023, median backward-discovered
year 2013, median forward-discovered year 2023.

Direction arms can only overlap once a *later* iteration reaches a work from the
opposite direction. In simulation, mean overlap under direction arms rises from
13.5 at iteration 1 to 97.8 at iteration 2; under provider arms it is 61.4 at
iteration 1. Since the author's run stopped at iteration 1 (§7), overlap could not
have accrued under any circumstances.

**Fix:** the default is now provider arms, in `snowballslr/config.py`,
`config.yaml` and `examples/config.yaml`, and `Config.warnings()` now names the
degenerate design and the remedy explicitly rather than letting the estimate fail
silently downstream.

---

## 4. Estimator calibration: bias is driven by iteration depth, not dependence

Chapman estimates were compared against known true recall across arm designs and
iteration depths.

| Arm design | Iteration | n | Mean estimated recall | Mean true recall | Mean bias |
|---|---|---|---|---|---|
| provider | 1 | 24 | 0.998 | 0.318 | +0.680 |
| provider | 2 | 24 | 1.000 | 0.874 | +0.125 |
| provider | 3 | 24 | 1.000 | 0.920 | +0.080 |
| direction | 1 | 8 | 0.663 | 0.420 | +0.243 |
| direction | 2 | 24 | 0.960 | 0.874 | +0.085 |
| direction | 3 | 24 | 0.999 | 0.920 | +0.079 |

Bias is large and positive early and shrinks as the run proceeds - the closure
signature again. By contrast, varying the simulated dependence between provider
arms from independent to strongly dependent barely moves it:

| Arm dependence | n | Mean bias |
|---|---|---|
| 0.0 | 24 | +0.237 |
| 0.3 | 24 | +0.237 |
| 0.6 | 24 | +0.238 |
| 0.9 | 24 | +0.253 |

This is a result worth stating plainly in the manuscript: **the dominant bias in
capture-recapture applied to snowballing is the closure violation, not the
dependence violation** that the methods literature usually emphasises. Dependence
still biases N̂ downward and estimated recall upward - so estimated recall remains
an upper bound, and the library says so - but at realistic iteration depths the
closure effect is the larger of the two.

Chao1 was separately found to have no degeneracy guard: on the real run's capture
histories, where every record was captured exactly once, it returned
`estimable=True` with N̂ = **114,481** and implied recall **0.4%**. That is
precisely the authoritative-looking-but-empty number the README promises to refuse.
It now refuses when `f2 == 0 and f1 == S_obs`, while continuing to estimate
whenever doubletons or higher-order captures exist.

---

## 5. Identity layer: 12 true duplicates recovered on the real corpus

The deduplication cascade was audited against the 493-record real corpus.
Twenty-seven pairs shared an identical normalised title but were never merged. Two
independent causes, both fixed:

- **Blocking defeated the tolerance it was meant to accelerate.** The blocking key
  bucketed on `year // 2` while tiers T2/T3 allow ±1 year, so 1995 and 1996 landed
  in different buckets and were never compared, though 1996 and 1997 shared one.
  On the real corpus this separated **6 of 21** title-identical pairs whose years
  were within tolerance. Records now emit two blocking keys.
- **Surname folding was too literal.** Crossref unstructured references yield
  `"AM Cook"` and `"G.Amudha"` where OpenAlex yields `"Cook"` and `"Amudha"`; after
  folding to bare letters these compare unequal, so genuine duplicates survived
  every tier. Matching now tolerates a leading-initials prefix, with a length floor
  and a maximum dropped prefix of three characters so that `"Berg"` does not merge
  into `"Vandenberg"`.

**Result on the real corpus: 493 → 481 records, 12 additional true duplicates
merged (2.4%)**, all of them resolved-DOI versus unresolved-signature twins of the
same work. Determinism was re-verified after the change: order-independence over 8
permutations, idempotence, and non-growth all hold.

---

## 6. Robustness and determinism audit

All checks below were run against the real 493-record corpus.

| Check | Result |
|---|---|
| Dedup order-independence (8 random permutations) | PASS |
| Dedup idempotence | PASS |
| Dedup never grows the input set | PASS |
| Provenance union preserved through merge (INV-6) | PASS (665 entries in, 665 out) |
| Runaway-merge guard on 12 records titled "Editorial" | PASS (kept apart, 1 suspicious cluster reported) |
| `candidates.csv` byte-stability across repeated writes | PASS |
| Full suite under `PYTHONHASHSEED=random` | PASS |

**`verify` was checking only half of what it claimed.** It counted cache entries
but validated none, so an edited cache body passed with `ok=True` - the cache being
exactly the substrate a replay would be recomputed from. Every entry is
content-addressed twice (its filename is the request key; it carries the hash of
its own body), so both are now re-checked. Verified against the real run: the clean
run passes over 70 entries; a tampered body and a tampered request key are each
detected and named.

---

## 7. Two defects that silently ended the author's own run

Both were found by reading `run/state.json` rather than by simulation, and both are
reported here because they explain a real result the author already has.

**The silent-exhaustion trap.** `run/state.json` records `phase: stopped`,
`stopped_by: exhaustion` at iteration 1, with 15 decisions - all of them seeds -
and 308 candidates never screened. `labels_001.csv` has all 308 decision cells
empty. Because nothing was included, nothing entered the frontier, so the run
terminated by `exhaustion` and reported a *completed* snowballing run that had
screened zero records. The PRISMA counts looked superficially valid. Labelling a
batch in which no record carries a decision is now refused with an actionable
error, unless `allow_empty=True` is passed deliberately.

**Unresolved records were expanded.** The README stated that references which
cannot be resolved to a persistent identifier are "never expanded forward". The
core loop did not enforce it: in a controlled test, 45 of 45 included unresolved
records entered the frontier. Expanding such a record means issuing a title query
whose match cannot be verified, which grafts an unverifiable neighbourhood onto the
citation graph. The guard now exists in `Run._expand` and emits an audit event; the
records are still retained, screened and counted in PRISMA as before.

---

## 8. What this study does and does not establish

**Establishes.** That the unguarded `estimated_recall` rule stops far too early and
that a closure guard fixes it; that direction-based arms cannot support
capture-recapture on a citation graph; that Chao1 and `verify` had guards their
documentation promised but did not implement; that the identity layer missed a
measurable share of true duplicates; and that the determinism properties the
library claims do hold, on real data, after all changes.

**Does not establish.** The stopping-rule benchmark runs on **simulated** citation
graphs, not on published systematic reviews with known inclusion lists. Simulation
is what makes ground truth available at all, and it is the same design Rajit et al.
(2025) used, but the generator's homophily and coverage parameters are chosen, not
estimated from a corpus. The direction of the closure effect is structural and
would survive different parameters; the exact numbers (0.450, 0.906) would not.
Replication on published reviews with known inclusion lists is the natural next
step and is the single strongest addition the manuscript could still gain.

Two further limits should be stated in the paper. Capture-recapture arms in
citation searching are positively dependent, so N̂ is biased downward and estimated
recall should be read as an **upper** bound; with exactly two arms this is not
testable, and a third arm is needed to make it so. And no stopping rule can exceed
the reachable-population ceiling: in this benchmark a mean of 8.7% of the
relevant population had no citation path to the seed set and was unreachable by any
citation-searching strategy whatsoever.
