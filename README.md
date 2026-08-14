# SnowBallSLR

**Deterministic, saturation-aware snowballing for systematic literature reviews.**

Automated bidirectional citation searching (backward and forward) that iterates to an explicitly defined saturation point, estimates how much of the relevant literature has been found, and replays byte-for-byte from its own cache.

To our knowledge, and based on a survey of citation-searching tools indexed on PyPI, CRAN and GitHub as of August 2026, no existing tool combines all three of: configurable quantitative stopping rules evaluated and logged at every iteration; capture–recapture recall estimation with explicit assumption diagnostics; and a hash-verified reproducibility layer that replays from a content-addressed response cache and classifies database drift on refetch. It is *not* the first tool to perform bidirectional citation searching — citationchaser and paperfetcher (both 2022) precede it, and both are cited below.

```bash
git clone https://github.com/s-matysik/SnowBallSLR.git
pip install ./SnowBallSLR
```

A PyPI release is pending; until it lands, install from a clone as above. What was tested: a
clean-virtual-environment install from a local checkout, after which `import snowballslr` and the
`snowballslr` console script both work outside the source tree. `pip install
git+https://github.com/s-matysik/SnowBallSLR.git` should be equivalent — the repository carries the
same `pyproject.toml`, and it declares the build backend and console script — but it has not been
exercised here, so the two-step form above is the one to trust.

---

## Why this exists

Citation searching is a recommended component of systematic reviews ([TARCiS](https://doi.org/10.1136/bmj-2023-078384)), and PRISMA 2020 gives it its own identification column. In practice it is still run by hand: one pass over the references, one pass over the citing works, and a stop when "nothing new seems to be coming up."

Three problems follow, and SnowBallSLR addresses each:

**1. The stopping decision is left to judgement.** "Nothing new seems to be coming up" is rarely pre-specified and rarely reported. Formal stopping criteria exist on the screening side — Bron et al. (2025) use Chao's estimator as a stopping criterion for technology-assisted review, and Callaghan & Müller-Hansen (2020) provide hypothesis-test criteria in `buscar` — but to our knowledge no citation-searching tool exposes them as configurable rules over chase iterations. SnowBallSLR ships five. Every rule you configure is evaluated at every iteration and every decision is recorded, firing or not, so you can report the rule you pre-specified and inspect what the others would have done.

**2. How much was missed is rarely quantified.** Where the configured source arms overlap sufficiently, SnowBallSLR estimates the size of the relevant population by capture–recapture and reports achieved recall with a confidence interval — plus a blunt statement of which assumptions are violated and in which direction the bias runs. Where the overlap is too small, it refuses to estimate and says why.

**3. Live APIs are not reproducible.** OpenAlex and Semantic Scholar change daily. The same seeds run in March and in June give different sets. SnowBallSLR caches every raw response by content hash, records artifact hashes in a run manifest, and ships `verify` — which re-checks every emitted artifact against the manifest and every cache entry against its own two content addresses, or, with `--refresh`, refetches against the live API and classifies precisely what drifted: new citations, changed metadata, retractions, deindexed records, merged identifiers.

## What it does not do

**It does not screen.** Include/exclude decisions come from outside as a labels file. Ranking exists solely to order the screening queue, never to decide inclusion — this is enforced structurally, not just by convention.

It also does not do Boolean database search, full-text retrieval, data extraction, or synthesis. And it trains nothing: no active learning, no learned thresholds, no stochastic component anywhere in the decision path.

## Quickstart

```bash
snowballslr init ./run --seeds seeds.txt --config config.yaml
snowballslr step ./run                       # → iterations/iter_001/candidates.csv
#   ... screen the batch, fill in the decision column ...
snowballslr label ./run --file labels.csv    # → evaluates every stopping rule
snowballslr estimate ./run                   # → capture-recapture recall estimate
snowballslr report ./run --graph             # → PRISMA counts, diagram, network
snowballslr verify ./run                     # → re-checks artifacts and cache against their hashes
```

Or from Python:

```python
from snowballslr import Run, Config

run = Run.init("./run", seeds=["10.1145/2601248.2601268"], config=Config.from_yaml("config.yaml"))

while not run.stopped:
    candidates = run.step()          # ranked, deduplicated, eligibility-filtered
    if not candidates:
        break
    run.label({w.key: my_screening_function(w) for w in candidates})

estimate = run.estimate_recall()
print(estimate.recall, estimate.warnings)
run.report(prisma=True, graph=True)
```

## Stopping rules

| Rule | Fires when |
|---|---|
| `marginal_yield` | New includes per record screened stay below ε for k consecutive iterations |
| `asymptotic_coverage` | Observed inclusions reach τ of a fitted accumulation-curve asymptote |
| `estimated_recall` | Capture–recapture recall (lower CI bound by default) reaches τ **and** the estimated population has stopped moving |
| `budget` | A screening cap is hit, or yield per 100 screened drops below a floor |
| `exhaustion` | The frontier is empty — always active, and the upper-bound baseline |

**Why `estimated_recall` waits.** Capture–recapture assumes a *closed* population, and snowballing violates closure by construction: every included record enlarges the reachable population. Early in a run the arms agree almost perfectly on a small reachable set, so N̂ collapses onto the observed count and estimated recall approaches 1.0 while true recall is still low. Benchmarked on 270 simulated reviews with known ground truth, the unguarded rule fired in every trial at a mean true recall of **0.45**, missing ~74 studies per review. Requiring N̂ to be stable within 5% for two consecutive iterations raised mean true recall at stop to **0.91** against an achievable ceiling of 0.91. The guard is on by default; set `min_stable_iterations: 0` to disable it.

Each returns a `rationale` written to be pasted into a methods section unchanged:

> Citation searching was stopped after iteration 4: capture-recapture across the backward and forward arms estimates a relevant population of 61.3 studies, of which 58 were identified; the lower bound of the 95% confidence interval of estimated recall is 95.2%, meeting the pre-specified stopping threshold of 95%.

## Recall estimation, honestly

Two arms: Chapman-corrected Lincoln–Petersen with Seber variance. Three or more: log-linear models over the observable capture-history cells, selected by AIC. Also available: Chao1 as a distribution-free lower bound.

Every estimate carries diagnostics, and the library refuses to bluff:

- **With exactly two arms, independence is not testable** — the 2×2 table has zero residual degrees of freedom. SnowBallSLR says so rather than estimating the missing cell from N̂ and then "testing" it, which would be circular. In citation searching the arms are almost certainly positively dependent, so N̂ is biased downward and **estimated recall should be read as an upper bound**. Configure a third arm and the assumption becomes testable.
- **Heterogeneous catchability** is reported across publication year, language and venue type.
- **Closure violations** are detected from cache timestamps.

Estimates with fewer than three overlapping records are refused outright, and Chao1 is refused when every record was captured exactly once. A number that looks authoritative in a manuscript but rests on an overlap of one is worse than no number.

**Choose arms that can capture the same record.** A citation graph is temporally acyclic, so backward and forward expansion sample near-disjoint strata: a work older than your seeds is normally reachable only backward, a newer one only forward. On this project's own reference run the two direction arms shared **0 of 478** records, and recall was correctly reported as not estimable; the same run under OpenAlex/Crossref arms overlaps on **172**. Arms are therefore defined by *provider* by default. Overlap also accrues only from the second iteration onward, which is why the `estimated_recall` rule will not fire while the estimated population is still moving (see below).

## Related tools

SnowBallSLR is not the first tool to do bidirectional citation searching, and does not claim to be. Marks reflect what is present in source or documented as of **2026-08-09**, with the version or commit audited given so that every mark is falsifiable. "n/d" means not documented and not verified by us — deliberately not marked ✗.

| Tool | Form | Audited | Both directions | Iterates | Quantitative stopping rule | Recall estimate | Hash-verified reproducibility |
|---|---|---|:-:|:-:|:-:|:-:|:-:|
| [citationchaser](https://doi.org/10.1002/jrsm.1563) | R package + Shiny | CRAN 0.0.4 | ✅ | ✗ (single pass) | ✗ | ✗ | ✗ |
| [paperfetcher](https://doi.org/10.1002/jrsm.1604) | Python library | PyPI 1.2.2 | ✅ (forward via OpenCitations) | ✗ (single pass) | ✗ | ✗ | ✗ |
| [rjglasse/snowball](https://github.com/rjglasse/snowball) | Python library + TUI/CLI | commit 4130d38 | ✅ | ✅ (while the last round yielded an include) | ✗ (continue-condition, not a reportable criterion) | ✗ | ✗ |
| [BibliZap](https://doi.org/10.1002/jrsm.70018) | R/Shiny | published 2026 | ✅ (multi-level) | ✅ (to a fixed depth) | ✗ (depth limit) | ✗ | ✗ |
| [LitBall](https://arxiv.org/abs/2402.08339) | Kotlin/JVM desktop | repo @ 2025-09-29 | ✅ | ✅ (user-driven rounds) | ✗ (stop is the curator's judgement) | ✗ | partial (persists state; no artifact manifest) |
| SpiderCite (SR-Accelerator) | Web app | not inspected | ✅ (documented) | n/d | n/d | n/d | n/d |
| [SYMBALS](https://doi.org/10.3389/frma.2021.685591) | Methodology | published article | ✗ (backward only) | ✅ (with active learning) | ✗ | ✗ | ✗ |
| [ReviQ](https://doi.org/10.1016/j.softx.2026.102814) | Review workbench (Docker) | repo @ 2026-08-10 | ✗ (the chase is done externally and imported as BibTeX; no citation retrieval in the codebase) | ✗ (iterations are recorded, not performed) | ✗ (saturation is a reviewer-set flag with an undo endpoint) | ✗ (relative recall per database; the source notes the true population is unknown) | ✗ (state export; no response hashing or cache) |
| **SnowBallSLR** | Python library + CLI | this repo, v1.0.0 | ✅ | ✅ (automatic) | ✅ (five rules, all logged every iteration) | ✅ (Chapman / log-linear / Chao1, with diagnostics) | ✅ (response cache, artifact manifest, drift classification) |

A word on **ReviQ**, because the ✗ marks above understate it: it is a complete review workbench covering all eight Kitchenham phases, with two-reviewer screening, Cohen's κ and PABAK, quality assessment and data extraction — everything this library deliberately leaves out. The marks record only that it manages citation searching rather than performing it. The two compose: ReviQ can run the review, SnowBallSLR the citation-searching stage. Marks verified against its source, see `audit/reviq_code_audit.md`.

Two tools outside citation searching hold single differentiators and should be read alongside this table: [searchAnalyzeR](https://cran.r-project.org/package=searchAnalyzeR) (CRAN, 2025) provides manifests, checksums, audit trails and search re-execution for Boolean *database* searching, and [buscar/buscarpy](https://doi.org/10.1186/s13643-020-01521-4) provides hypothesis-test stopping criteria over a machine-prioritised *screening* ranking. Neither operates on the citation graph, but neither is the reproducibility nor the stopping-rule idea novel in evidence synthesis as a whole.

Closely related work worth reading before you use this: [Rajit et al. (2025)](https://doi.org/10.1017/rsm.2024.15) simulated automated citation searching across 27 reviews and found it beats standard strategies on precision and F1 but **not** on recall; and [Bron et al. (2025)](https://doi.org/10.1145/3724116) used Chao's estimator as a stopping criterion for technology-assisted review — on the screening side rather than the citation graph.

**Citation searching is a complementary search method.** It has not been shown to outperform Boolean database search on recall, and this library does not change that. What it contributes is knowing when to stop, and how much you have found.

## Coverage caveat

OpenAlex reference coverage is roughly 83% for recent DOI-bearing literature and materially lower for older, non-English and non-STEM work ([Culbert et al. 2025](https://doi.org/10.1007/s11192-025-05293-3)). That is why the GROBID provider exists: it extracts bibliographies from PDFs you already hold and resolves them through Crossref, with match confidence recorded. References that cannot be resolved are kept, flagged `unresolved`, never expanded forward, and reported separately in the PRISMA counts.

## Outputs

```
run/
  config.yaml            state.json             run.json          audit.jsonl
  cache/                 content-addressed raw provider responses
  iterations/iter_NNN/   candidates.csv  candidates.ris  labels_template.csv
  outputs/               prisma.json  prisma.svg  network.graphml
                         included.csv  report.md  verify_report.md
```

## Development

```bash
pip install -e ".[dev,loglinear]"
pytest --cov=snowballslr
PYTHONHASHSEED=random pytest tests/test_determinism.py
ruff check snowballslr tests
```

The test suite never touches the network — an autouse fixture raises on any HTTP call. Property-based tests cover the identity layer: deduplication is idempotent, order-independent, and never grows the input set.

## Citation

See `CITATION.cff`, or cite the SoftwareX article once published.

## Licence

MIT.
