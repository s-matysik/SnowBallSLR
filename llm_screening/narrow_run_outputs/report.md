# Snowballing run: narrow-rule-driven

- Seed studies: 6
- Iterations completed: 10
- Records identified by citation searching: 11994
- Records screened: 10876
- Studies included: 585 newly identified (+6 seeds = 591 total)
- Directions: backward
- Data sources used: crossref, openalex
- Status: stopped (stopped by: exhaustion)

## Iteration history

| Iter | Expanded | Screened | Included | Cum. screened | Cum. included | Frontier |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 115 | 6 | 115 | 12 | 6 |
| 2 | 12 | 331 | 37 | 446 | 49 | 37 |
| 3 | 49 | 1634 | 85 | 2080 | 134 | 85 |
| 4 | 134 | 2596 | 146 | 4676 | 280 | 146 |
| 5 | 280 | 3305 | 216 | 7981 | 496 | 216 |
| 6 | 495 | 1843 | 83 | 9824 | 578 | 83 |
| 7 | 578 | 988 | 8 | 10812 | 586 | 8 |
| 8 | 586 | 34 | 4 | 10846 | 590 | 4 |
| 9 | 590 | 32 | 1 | 10878 | 591 | 1 |
| 10 | 1 | 0 | 0 | 10878 | 591 | 0 |

## Recall estimate

- Method: `chapman`
- Estimated relevant population: 643.3
- 95% CI for N: [620.0, 666.6]
- Estimated recall: 90.9%
- 95% CI for recall: [87.8%, 94.3%]

### Assumption warnings

- Independence of the two source arms is assumed but NOT testable with two arms (zero residual degrees of freedom). In citation searching the arms are typically positively dependent, because backward and forward chasing both concentrate on well-connected literature; under positive dependence N_hat is biased downward and the estimated recall must be read as an UPPER BOUND. Configure a third source arm to test this assumption.
- Catchability is heterogeneous across language: 'unknown' records are captured by 1.00 arms on average versus 1.80 for 'en'. N_hat is biased downward.
- Catchability is heterogeneous across venue_type: 'conference-paper' records are captured by 0.00 arms on average versus 1.81 for 'article'. N_hat is biased downward.
- 256 discovered records could not be resolved to a persistent identifier and are excluded from the capture table.

## Stopping rules

- `marginal_yield` -- not fired
- `estimated_recall` -- not fired
- `budget` -- not fired
- `exhaustion` -- **FIRED**

### Methods-section text

> Citation searching was continued until the frontier was exhausted after iteration 10: no newly included study remained whose references and citations had not already been retrieved.

## Reproducibility

- Config hash: `sha256:c4eff3a4d258332b7d2260a6a8ff0468dcc5220f89597f461eaf8784d04c8038`
- **Configuration changed after this run was written.** The manifest records `sha256:c4eff3a4d258332b7d2260a6a8ff0468dcc5220f89597f461eaf8784d04c8038`, which is the configuration the run executed under; the configuration now on disk hashes to `sha256:034847c30854826336dcc71f8ce1a2833606d208e4f2e4499414c37c828f9bdc`. The counts above describe the run as executed. Re-running from the current configuration would not reproduce them.
- Cache entries: 1064
- Re-run `snowballslr verify <run_dir>` to confirm that this run reproduces byte-for-byte from its cache, or `snowballslr verify <run_dir> --refresh` to quantify drift in the underlying bibliographic databases since the run was executed.

_Citation searching is a complementary search method: automated citation searching has not been shown to outperform Boolean database search on recall. The contribution of this tool is knowing when to stop and how much of the relevant literature has been found._
