# Snowballing run: ai-marketing-automation-slr

- Seed studies: 15
- Iterations completed: 2
- Records identified by citation searching: 12287
- Records screened: 9732
- Studies included: 2772 newly identified (+15 seeds = 2787 total)
- Directions: backward, forward
- Data sources used: crossref, openalex
- Status: stopped (stopped by: budget)

## Iteration history

| Iter | Expanded | Screened | Included | Cum. screened | Cum. included | Frontier |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 302 | 90 | 302 | 105 | 90 |
| 2 | 105 | 9430 | 2682 | 9732 | 2787 | 2682 |

## Recall estimate

- Method: `chapman`
- Estimated relevant population: 3834.6
- 95% CI for N: [3622.4, 4046.9]
- Estimated recall: 72.6%
- 95% CI for recall: [68.8%, 76.8%]

### Assumption warnings

- Independence of the two source arms is assumed but NOT testable with two arms (zero residual degrees of freedom). In citation searching the arms are typically positively dependent, because backward and forward chasing both concentrate on well-connected literature; under positive dependence N_hat is biased downward and the estimated recall must be read as an UPPER BOUND. Configure a third source arm to test this assumption.
- Catchability is heterogeneous across venue_type: 'unknown' records are captured by 1.00 arms on average versus 2.00 for 'book'. N_hat is biased downward.
- 158 discovered records could not be resolved to a persistent identifier and are excluded from the capture table.

## Stopping rules

- `estimated_recall` -- not fired
- `marginal_yield` -- not fired
- `budget` -- **FIRED**
- `exhaustion` -- not fired

### Methods-section text

> Citation searching was stopped after 9732 records had been screened, reaching the pre-specified screening budget of 2500.

## Reproducibility

- Config hash: `sha256:debd3c7c5b5bc40d453e6a9e226511af82c7f3b9c151505c4189aa4abcbdf1ba`
- Cache entries: 472
- Re-run `snowballslr verify <run_dir>` to confirm that this run reproduces byte-for-byte from its cache, or `snowballslr verify <run_dir> --refresh` to quantify drift in the underlying bibliographic databases since the run was executed.

_Citation searching is a complementary search method: automated citation searching has not been shown to outperform Boolean database search on recall. The contribution of this tool is knowing when to stop and how much of the relevant literature has been found._
