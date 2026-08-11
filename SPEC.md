# SnowSLR — Implementation Specification v1.0

**Deterministic, saturation-aware snowballing for systematic literature reviews.**

Target: Python ≥3.11 · MIT · PyPI `snowballslr` · Zenodo DOI · SoftwareX submission.

---

## 0. Scope

### 0.1 What SnowSLR does

Automates iterative bidirectional citation searching (snowballing) for SLRs, with three properties no existing tool combines:

1. **Automatic iteration to an explicitly defined saturation point**, governed by formal, quantitative stopping rules.
2. **Recall estimation via capture–recapture**, with explicit reporting of assumption violations.
3. **Bit-for-bit reproducibility** of runs against live, mutating bibliographic APIs, plus a `verify` command that classifies drift between runs.

### 0.2 What SnowSLR does NOT do

- **It does not screen.** Include/exclude decisions enter from outside as a labels file. Ranking exists only to order the screening queue.
- It does not do Boolean database search. Seeds come from elsewhere.
- It does not do full-text retrieval, data extraction, or synthesis.
- It does not train models. No active learning, no fine-tuning, no stochastic components anywhere in the decision path.

### 0.3 Design invariants (non-negotiable)

| ID | Invariant |
|----|-----------|
| INV-1 | Given identical config + identical cache, two runs produce byte-identical outputs. |
| INV-2 | Every emitted record carries complete provenance: provider, direction, parent, iteration, timestamp, response hash. |
| INV-3 | No unordered iteration reaches output. Every collection is sorted by an explicit total order before serialization. |
| INV-4 | No network access in the test suite. |
| INV-5 | Ranking scores never influence inclusion, only queue order. Enforced by type separation: rankers cannot write to `included`. |
| INV-6 | Provenance is append-only. Re-discovery of a known work adds a provenance entry; it never overwrites one. |

---

## 1. Terminology

| Term | Meaning |
|------|---------|
| **Seed set** | Initial included works supplied by the user. |
| **Frontier** | Works whose neighbourhoods have not yet been expanded. Initially = seed set. |
| **Backward** | References of a work (its bibliography). |
| **Forward** | Works citing a work. |
| **Candidate** | Newly discovered, deduplicated, eligible work awaiting a screening decision. |
| **Iteration** | One full cycle: expand frontier → normalize → dedup → filter → rank → export → ingest labels → evaluate stopping rules. |
| **Included / Excluded** | Externally supplied screening decisions. |
| **Saturation** | State in which a configured stopping rule fires. |
| **Oracle** | In simulation mode, a complete ground-truth label set used to answer screening decisions automatically. |

---

## 2. Package layout

```
snowballslr/
  __init__.py
  types.py                 # Work, Author, Provenance, Direction, Decision
  config.py                # pydantic Settings, YAML load/dump, config hashing
  errors.py

  core/
    run.py                 # Run: orchestration, state machine
    state.py               # RunState: frontier, included, excluded, screened, history
    frontier.py            # expansion planning, deduplication of expansion targets
    iteration.py           # IterationRecord, IterationHistory

  providers/
    base.py                # Provider protocol, RateLimiter, RetryPolicy
    openalex.py
    crossref.py
    semanticscholar.py
    grobid.py
    offline.py             # RIS/BibTeX/JSON fixture provider; replay-only
    registry.py

  identity/
    normalize.py           # DOI, title, author, year normalization
    keys.py                # canonical key derivation
    dedup.py               # deterministic cascade + cluster merge

  ranking/
    base.py                # Ranker protocol
    lexical.py             # BM25
    network.py             # co-citation, bibliographic coupling, parent count
    embedding.py           # optional extra
    fusion.py              # reciprocal rank fusion

  stopping/
    base.py                # StoppingRule protocol, StopDecision
    yield_rule.py          # MarginalYield
    asymptote.py           # AsymptoticCoverage
    recall_rule.py         # EstimatedRecall
    budget.py              # Budget
    exhaustion.py          # Exhaustion
    compose.py             # AnyOf, AllOf

  estimate/
    capture_recapture.py   # Chapman, Seber variance, CI
    loglinear.py           # 3+ sources, AIC/BIC model selection
    chao.py                # Chao1 lower bound (frequency-based)
    assumptions.py         # dependence / heterogeneity / closure diagnostics

  determinism/
    cache.py               # content-addressed HTTP cache
    manifest.py            # RunManifest read/write/hash
    verify.py              # drift classification

  report/
    prisma.py              # PRISMA 2020 counts + SVG flow diagram
    audit.py               # JSONL audit trail
    export.py              # CSV, RIS, BibTeX
    graph.py               # GraphML citation network
    summary.py             # human-readable run report

  simulation/
    oracle.py              # label oracle from gold standard
    harness.py             # multi-review, multi-seed experiment driver
    metrics.py             # recall@iter, screening burden, over/undershoot, regret

  cli.py                   # typer app
```

---

## 3. Data model

```python
from enum import StrEnum
from datetime import datetime
from typing import Mapping, Sequence

class Direction(StrEnum):
    SEED = "seed"
    BACKWARD = "backward"
    FORWARD = "forward"

class Decision(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNSCREENED = "unscreened"

@dataclass(frozen=True, slots=True, order=True)
class Author:
    surname: str
    given: str | None = None
    orcid: str | None = None

@dataclass(frozen=True, slots=True)
class Provenance:
    provider: str          # "openalex" | "crossref" | "semanticscholar" | "grobid" | "seed"
    direction: Direction
    parent_key: str | None # canonical key of the work this was found from
    iteration: int
    retrieved_at: datetime # UTC, from cache entry, NOT wall clock at read time
    response_hash: str     # sha256 of the raw provider response that yielded this record

@dataclass(frozen=True, slots=True)
class Work:
    key: str                          # canonical, see §5.2
    doi: str | None                   # normalized, lowercase, bare (10.xxxx/yyy)
    title: str
    title_norm: str                   # normalization output, cached
    abstract: str | None
    year: int | None
    authors: tuple[Author, ...]
    venue: str | None
    type: str | None                  # journal-article, proceedings-article, ...
    language: str | None
    is_retracted: bool
    source_ids: Mapping[str, str]     # {"openalex": "W123", "s2": "...", "pmid": "..."}
    reference_count: int | None
    cited_by_count: int | None
    provenance: tuple[Provenance, ...]

    def merged_with(self, other: "Work") -> "Work": ...
    # Field precedence resolved by config.provider_precedence; conflicts logged.
    # provenance = tuple(sorted(set(self.provenance) | set(other.provenance)))
```

**Sort key for all outputs:** `(-score, key)` where score defaults to 0.0. Where no score exists, `key` alone. Never rely on insertion order.

---

## 4. Providers

### 4.1 Protocol

```python
class Provider(Protocol):
    name: str
    supports: frozenset[Direction]

    def resolve(self, ident: str) -> Work | None: ...
    def references(self, work: Work) -> list[Work]: ...   # backward
    def citations(self, work: Work) -> list[Work]: ...    # forward
```

All provider methods go through the cache layer (§8). A provider never touches the network if a cache entry exists, regardless of age, unless `--refresh` is set.

### 4.2 Implementations

| Provider | Backward | Forward | Notes |
|----------|----------|---------|-------|
| `openalex` | ✅ `referenced_works` | ✅ `filter=cites:Wxxx` | Polite pool via `mailto`. Cursor pagination, `per_page=200`. Primary source. |
| `crossref` | ✅ `reference` array | ❌ | Polite pool via `mailto` in UA. Reference arrays often incomplete for older/non-STEM. |
| `semanticscholar` | ✅ `/references` | ✅ `/citations` | Optional API key. Without key: 1 req/s hard limit. |
| `grobid` | ✅ from local PDFs | ❌ | `processReferences` endpoint. DOI resolution of extracted refs via Crossref `query.bibliographic`, storing match confidence. |
| `offline` | ✅ | ✅ | Replay from fixtures/cache only. Used by tests and by frozen-snapshot experiments. |

### 4.3 GROBID pipeline

1. User supplies `pdf_dir` mapping canonical keys → PDF paths (`pdfs.csv`: `key,path`).
2. POST to `processReferences`, parse TEI.
3. For each extracted reference: attempt DOI match via Crossref `query.bibliographic` with title + first author + year.
4. Accept match if Crossref score ≥ `grobid.min_match_score` (default 60) AND normalized-title Jaro-Winkler ≥ 0.90 AND year within ±1.
5. Unmatched references are retained as `Work` with `doi=None` and a synthetic key; flagged `unresolved=True`; excluded from forward expansion; reported separately in PRISMA as "identified but unresolvable".
6. Match confidence stored in provenance metadata.

### 4.4 Rate limiting and retries

- Token-bucket limiter per provider, configurable rps.
- Retry on 429/500/502/503/504: exponential backoff, base 1s, factor 2, jitter **disabled** (determinism — use fixed backoff sequence), max 5 attempts.
- 404 → `None`, cached as a negative result.
- Any unrecoverable error aborts the iteration and writes a resumable checkpoint. No partial iteration is ever committed to state.

---

## 5. Identity and deduplication

### 5.1 Normalization (`identity/normalize.py`)

**DOI:** strip scheme/host prefixes (`https://doi.org/`, `doi:`, `DOI:`), lowercase, strip trailing punctuation and whitespace, validate `^10\.\d{4,9}/\S+$`.

**Title:** Unicode NFKD → strip combining marks → casefold → replace all non-alphanumeric with single space → collapse whitespace → strip. Additionally strip a trailing period and common prefixes ("erratum:", "correction:", "corrigendum:") into a flag.

**Author surname:** NFKD → strip marks → casefold → keep alphabetic only.

**Year:** integer; if provider gives a range or partial date, take the earliest full year.

### 5.2 Canonical key

Precedence:
1. `doi:<normalized-doi>` if DOI present and valid.
2. `oa:<openalex-id>` if OpenAlex ID present.
3. `s2:<corpus-id>` if S2 corpus ID present.
4. `sig:<sha1(title_norm|year|first_author_surname_norm)[:16]>`.

Keys are stable across runs and across providers. A key never changes once assigned within a run; if a later provider supplies a DOI for a work first keyed by signature, the merge records an alias in `state.aliases` and rewrites references to the alias in a single, logged pass at end of iteration.

### 5.3 Dedup cascade (deterministic, no ML)

Applied in order; first match wins:

| Tier | Rule |
|------|------|
| T1 | Equal normalized DOI. |
| T2 | Equal `title_norm` AND `abs(year_a - year_b) ≤ 1` AND equal first-author surname. |
| T3 | Jaro-Winkler(`title_norm`) ≥ `dedup.jw_threshold` (default 0.95) AND `abs(year_a - year_b) ≤ 1` AND equal first-author surname. |
| T4 | Equal `title_norm` AND equal year (author missing on one side). |

Blocking key for T2–T4: first 12 chars of `title_norm` + year bucket, to keep comparison O(n) in practice.

Clusters merged transitively. Merge order is deterministic: cluster representative is the member with the lexicographically smallest key.

**Guard:** if a cluster exceeds `dedup.max_cluster_size` (default 8), the merge is aborted, the cluster is logged as suspicious, and members are kept separate. Prevents runaway merges on generic titles ("Editorial", "Introduction").

---

## 6. Ranking

### 6.1 Protocol

```python
class Ranker(Protocol):
    name: str
    def fit(self, included: Sequence[Work]) -> None: ...
    def score(self, candidates: Sequence[Work]) -> Mapping[str, float]: ...
```

Rankers receive `included` read-only and return scores. They have no access to `RunState.include()`.

### 6.2 Implementations

**`LexicalRanker`** — BM25 (k1=1.2, b=0.75) over `title + " " + abstract`, query = concatenation of included titles+abstracts. Own implementation (≈60 LoC) rather than a dependency, to guarantee deterministic tokenization and IDF computation. Tokenizer: `title_norm` pipeline + split on whitespace + drop tokens of length 1 + fixed English stopword list shipped in-package.

**`NetworkRanker`** — feature vector per candidate, all computed from the citation graph only (works when abstracts are missing, which is common):
- `n_parents`: number of distinct included works that led to this candidate
- `cocitation`: number of included works co-cited with the candidate
- `coupling`: size of shared reference set with included works
- `direction_bonus`: candidates found in both directions rank higher

Score = weighted sum of min-max normalized features; weights in config, defaults `(0.4, 0.25, 0.25, 0.10)`.

**`EmbeddingRanker`** *(extra `[embeddings]`)* — cosine similarity of candidate embedding to centroid of included embeddings. Model name **and revision hash** pinned in config and recorded in the manifest. `model.eval()`, `torch.set_grad_enabled(False)`, fixed dtype float32, deterministic pooling. If the resolved model revision differs from the manifest, the run aborts.

**`RRFFusion`** — default. `score(d) = Σ_r 1/(k + rank_r(d))`, k=60. Deterministic, no training, trivially defensible in review.

Ties broken by canonical key ascending.

---

## 7. Stopping rules

### 7.1 Protocol

```python
@dataclass(frozen=True)
class StopDecision:
    rule: str
    triggered: bool
    value: float | None
    threshold: float | None
    rationale: str          # methods-section-ready sentence
    detail: dict            # rule-specific diagnostics

class StoppingRule(Protocol):
    name: str
    def evaluate(self, history: IterationHistory) -> StopDecision: ...
```

Every rule is evaluated every iteration and all decisions are recorded, even non-firing ones. This yields the comparison data for the methodological paper at zero extra cost.

### 7.2 Rules

**`MarginalYield(eps=0.01, k=2)`**
Fires when `new_includes_i / screened_i < eps` for `k` consecutive iterations. Guard: never fires before iteration `k+1`.

**`AsymptoticCoverage(tau=0.95, min_points=4)`**
Fits `N(i) = N_inf * (1 - exp(-lambda * i))` to the cumulative-includes curve by nonlinear least squares (`scipy.optimize.curve_fit`, fixed initial guess `N_inf=2*N_obs, lambda=0.5`, `maxfev` fixed). Fires when `N_obs / N_inf_hat ≥ tau`. Returns `triggered=False` with `detail.fit_failed=True` if the fit does not converge or `N_inf_hat < N_obs`.

**`EstimatedRecall(tau=0.95, method="chapman", use_lower_ci=True)`**
Fires when estimated recall ≥ tau. With `use_lower_ci=True` (default, conservative) uses the lower bound of the 95% CI. Requires at least two independent-ish source arms configured (§8 of config).

**`Budget(max_screened=None, min_yield_per_100=None)`**
Fires when cumulative screened ≥ `max_screened`, or when includes per 100 screened in the last iteration < `min_yield_per_100`.

**`Exhaustion()`**
Fires when the frontier is empty. Always active; serves as the upper-bound baseline in experiments.

### 7.3 Composition

```yaml
stopping:
  mode: any_of        # any_of | all_of
  rules:
    - {type: estimated_recall, tau: 0.95, use_lower_ci: true}
    - {type: marginal_yield, eps: 0.01, k: 2}
    - {type: budget, max_screened: 3000}
    - {type: exhaustion}
```

---

## 8. Capture–recapture estimation

### 8.1 Source arms

An "arm" is a labelled subset of the discovery process. Configurable:

```yaml
estimate:
  arms:
    - {name: backward, filter: {direction: backward}}
    - {name: forward,  filter: {direction: forward}}
  # or
    - {name: openalex, filter: {provider: openalex}}
    - {name: s2,       filter: {provider: semanticscholar}}
    - {name: grobid,   filter: {provider: grobid}}
```

Arms are computed over **included** works only (relevant population), never over all candidates.

### 8.2 Estimators

**Two arms — Chapman (`capture_recapture.py`):**

```
N_hat = ((n1 + 1) * (n2 + 1)) / (m + 1) - 1
var   = ((n1+1)(n2+1)(n1-m)(n2-m)) / ((m+1)^2 * (m+2))
CI95  = N_hat ± 1.96 * sqrt(var)            # also report log-transformed CI
recall_hat = n_union / N_hat
```

Guard: if `m < 3`, return `estimable=False` with an explicit reason. Never emit an estimate from near-zero overlap.

**Three or more arms — log-linear (`loglinear.py`):**
Fit Poisson GLM on the 2^k − 1 observable capture-history cells (`statsmodels.GLM`), candidate models = independence + all sets of pairwise interactions; select by AIC (report BIC too); estimate the missing cell. Report the selected model formula explicitly.

**Frequency-based lower bound — Chao1 (`chao.py`):**
`N_hat = S_obs + f1²/(2·f2)` where `f1`/`f2` = number of included works discovered by exactly one / exactly two arms. Bias-corrected variant when `f2 = 0`. Serves as a distribution-free lower bound and as the direct point of comparison with Bron et al. (2025).

### 8.3 Assumption diagnostics (`assumptions.py`)

Report, never silently ignore:

| Diagnostic | Test | Consequence stated in output |
|---|---|---|
| Positive dependence between arms | Odds ratio of the 2×2 capture table; χ² test | `N_hat` biased downward → **recall overestimated**. Estimate is an upper bound on recall. |
| Heterogeneous catchability | Compare capture rate across strata: publication year quartile, language (EN/non-EN), venue type | Reports which strata are under-captured; `N_hat` biased downward. |
| Closure violation | Count of works added to the underlying APIs during the run window (from cache timestamps) | Population not closed; report magnitude. |

Output object carries a `warnings: list[str]` that propagates into the run report and into `EstimatedRecall.detail`. The manuscript must state that estimated recall is, under positive dependence, an upper bound.

---

## 9. Determinism layer

### 9.1 Cache (`determinism/cache.py`)

Content-addressed, on disk under `<run_dir>/cache/`.

```
cache_key = sha256(f"{provider}|{method}|{canonical_json(params)}").hexdigest()
path      = cache/<key[:2]>/<key>.json
```

Entry:
```json
{
  "cache_key": "...",
  "provider": "openalex",
  "method": "citations",
  "params": {...},
  "retrieved_at": "2026-08-09T12:00:00Z",
  "http_status": 200,
  "response_hash": "sha256:...",
  "provider_version": {"updated_date": "2026-08-01"},
  "body": {...}
}
```

`canonical_json` = sorted keys, no whitespace, UTF-8, `ensure_ascii=False`. Bodies stored raw and unmodified — parsing happens on read, so a parser bugfix never requires refetching.

### 9.2 Run manifest (`run.json`)

```json
{
  "snowballslr_version": "1.0.0",
  "python_version": "3.12.4",
  "config_hash": "sha256:...",
  "config": {...},
  "providers": [{"name": "openalex", "base_url": "...", "polite_pool": true}],
  "ranker": {"type": "rrf", "components": ["bm25", "network"]},
  "embedding_model": null,
  "started_at": "...", "finished_at": "...",
  "iterations": [
    {"n": 1, "frontier_hash": "...", "candidates_hash": "...",
     "labels_hash": "...", "n_expanded": 12, "n_raw": 604, "n_after_dedup": 511,
     "n_eligible": 498, "n_screened": 498, "n_included": 23,
     "stop_decisions": [...]}
  ],
  "outputs": {"prisma.json": "sha256:...", "candidates_iter_001.csv": "sha256:..."},
  "cache_entries": 1843,
  "stopped_by": "estimated_recall"
}
```

### 9.3 `verify` (`determinism/verify.py`)

Two modes:

**`snowballslr verify <run_dir>`** — replay from cache. Recomputes every artifact and compares hashes against the manifest. Any mismatch is a bug. Exit code 1 on mismatch. This is INV-1's enforcement.

**`snowballslr verify <run_dir> --refresh`** — refetch every cached request against the live API and classify the diff:

| Class | Definition |
|---|---|
| `unchanged` | Response hash identical. |
| `new_citing` | Forward-citation set grew; lists added keys. |
| `lost_citing` | Forward-citation set shrank. |
| `metadata_changed` | Same record, differing fields; lists changed field names. |
| `retracted` | Record now flagged retracted. |
| `deindexed` | Record no longer resolvable (404). |
| `merged` | Provider now redirects the ID to another record. |

Emits `verify_report.md` + `verify_report.json`: counts per class, per provider, plus the elapsed interval since the original run. **This report is Figure 2 of the paper.** A refresh run never mutates the original cache; it writes to `cache_refresh/`.

---

## 10. Reporting

### 10.1 PRISMA 2020 (`report/prisma.py`)

Emits `prisma.json` with the counts needed for the "Identification of studies via other methods" column, broken down by iteration and direction, plus `prisma.svg` (hand-drawn SVG, no external renderer). Fields: records identified by citation searching (per source), duplicates removed, records screened, records excluded, reports assessed, studies included.

### 10.2 Audit trail (`report/audit.py`)

`audit.jsonl`, one event per line, append-only, deterministic ordering:

```json
{"ts":"...","iteration":1,"event":"expand","key":"doi:10.1000/x","provider":"openalex","direction":"forward","n_returned":47,"cache_hit":true}
{"ts":"...","iteration":1,"event":"dedup_merge","cluster":["doi:10.1000/x","sig:ab12"],"tier":"T3","jw":0.971}
{"ts":"...","iteration":1,"event":"stop_eval","rule":"marginal_yield","triggered":false,"value":0.046,"threshold":0.01}
```

### 10.3 Exports (`report/export.py`)

`candidates_iter_NNN.csv` (with `key,title,year,doi,venue,score,rank,found_via,parent_keys`), `.ris`, `.bib`; `included.csv`; `network.graphml` (nodes = works with decision attribute, edges = citation links with direction); `report.md` — human-readable run summary including the methods-ready stopping-rule rationale and the CMR estimate with its warnings.

### 10.4 Labels file contract

Input `labels_iter_NNN.csv`:

```csv
key,decision,note
doi:10.1000/abc,include,
doi:10.1000/def,exclude,wrong population
```

Validation: unknown keys → error listing them; missing keys → error unless `--allow-partial` (then unlabelled candidates are treated as `unscreened` and remain in the pool for the next iteration); `decision` values outside the enum → error. Labels file hash recorded in manifest.

---

## 11. CLI

```
snowballslr init RUN_DIR --seeds seeds.txt [--config config.yaml]
snowballslr step RUN_DIR                    # one iteration → exports candidates, then halts
snowballslr label RUN_DIR --file labels.csv # ingest decisions, evaluate stopping rules
snowballslr run RUN_DIR --oracle gold.csv   # simulation: loop until a rule fires
snowballslr status RUN_DIR
snowballslr estimate RUN_DIR [--method chapman|loglinear|chao]
snowballslr verify RUN_DIR [--refresh]
snowballslr report RUN_DIR [--prisma] [--graph]
snowballslr export RUN_DIR --format ris|bib|csv --what candidates|included
snowballslr snapshot RUN_DIR --out snapshot.tar.zst   # freeze cache for replication
```

`seeds.txt`: one identifier per line — DOI, OpenAlex ID, S2 ID, or PMID. Mixed types allowed.

Exit codes: `0` ok · `1` verification/validation failure · `2` config error · `3` provider unrecoverable error · `4` stopping rule fired (in `step`, informational).

---

## 12. Python API

```python
from snowballslr import Run, Config

cfg = Config.from_yaml("config.yaml")
run = Run.init("./run", seeds=["10.1000/abc", "10.1000/def"], config=cfg)

while not run.stopped:
    candidates = run.step()               # list[Work], ranked
    if not candidates:
        break
    decisions = my_screening_function(candidates)   # external
    run.label(decisions)

print(run.estimate_recall())              # RecallEstimate with .value, .ci, .warnings
run.report(prisma=True, graph=True)
```

`Run.step()` is idempotent within an iteration: calling it twice without an intervening `label()` returns the same candidate list from state, does not refetch.

---

## 13. Configuration

```yaml
run:
  name: "uav-delivery-slr"
  max_iterations: 10
  directions: [backward, forward]

providers:
  order: [openalex, crossref, semanticscholar, grobid]
  precedence: [openalex, crossref, semanticscholar, grobid]   # metadata conflict resolution
  openalex:
    mailto: "user@example.org"
    rps: 8
  crossref:
    mailto: "user@example.org"
    rps: 8
  semanticscholar:
    api_key_env: "S2_API_KEY"
    rps: 1
  grobid:
    url: "http://localhost:8070"
    pdf_map: "pdfs.csv"
    min_match_score: 60

eligibility:
  year_min: 2010
  year_max: null
  types: [journal-article, proceedings-article, book-chapter]
  languages: null            # null = no filter
  exclude_retracted: true
  require_title: true

dedup:
  jw_threshold: 0.95
  max_cluster_size: 8

ranking:
  type: rrf
  components: [bm25, network]
  network_weights: [0.4, 0.25, 0.25, 0.10]
  top_k: null                # null = export all eligible candidates

stopping:
  mode: any_of
  rules:
    - {type: estimated_recall, tau: 0.95, use_lower_ci: true}
    - {type: marginal_yield, eps: 0.01, k: 2}
    - {type: budget, max_screened: 3000}
    - {type: exhaustion}

estimate:
  arms:
    - {name: backward, filter: {direction: backward}}
    - {name: forward, filter: {direction: forward}}
  method: chapman

determinism:
  cache_dir: "cache"
  offline: false             # true = fail rather than hit the network
  refresh: false
```

`config_hash` = sha256 of canonical JSON of the resolved config **excluding** secrets and `mailto`.

---

## 14. Testing

| Layer | Content |
|---|---|
| **Property (Hypothesis)** | Dedup idempotent: `dedup(dedup(x)) == dedup(x)`. Dedup order-independent: `dedup(x) == dedup(shuffle(x))`. Key canonicalization stable across DOI spelling variants. Ranking is a total order. Merge is associative and commutative under the configured precedence. Normalization is idempotent. |
| **Golden replay** | Fixture cache for a 3-review mini-corpus committed to the repo. Full runs execute offline; outputs compared byte-for-byte against committed goldens. |
| **Determinism** | Two runs on the same cache → identical hashes for every artifact in the manifest. Run under `PYTHONHASHSEED=0` and under a random seed; both must pass. |
| **Estimator** | Chapman/log-linear/Chao validated against synthetic populations with known N across dependence levels; assert bias direction matches the documented claim. |
| **Stopping rules** | Synthetic accumulation curves with known optimal stop; assert each rule's over/undershoot falls in the documented range. |
| **Provider contract** | Recorded HTTP fixtures per provider; parser tests for malformed/partial responses, missing abstracts, missing references, redirected IDs. |
| **CLI** | `typer.testing.CliRunner` end-to-end on the fixture corpus. |

Targets: line coverage ≥ 90%, branch coverage ≥ 80%. CI on Linux + macOS (arm64), Python 3.11/3.12/3.13. **No network in CI** (INV-4) — enforced by a `pytest` fixture that patches `httpx` transport to raise.

---

## 15. Dependencies

**Core:** `httpx`, `pydantic>=2`, `rapidfuzz`, `numpy`, `scipy`, `networkx`, `statsmodels`, `typer`, `pyyaml`, `orjson`.
**Extras:** `[embeddings]` → `sentence-transformers`; `[pdf]` → `lxml` (TEI parsing); `[viz]` → `matplotlib`.
**Dev:** `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy --strict`, `respx`.

Packaging: `pyproject.toml`, hatchling. Type hints everywhere, `py.typed` shipped.

---

## 16. Validation harness (`simulation/`)

This module produces the paper's results and ships as part of the package.

```
snowballslr-bench run \
  --reviews benchmarks/synergy_subset.yaml \
  --seed-strategies random,most_cited,oldest,boolean \
  --seed-sizes 1,3,5,10 \
  --repeats 50 \
  --snapshot snapshots/frozen_2026-08.tar.zst \
  --out results/
```

**Corpora:** SYNERGY (26 reviews, OpenAlex IDs), the 27 reviews from Rajit et al. 2025, plus 5–8 management/IS reviews with published inclusion lists. Minimum 25 reviews across ≥3 domains.

**Frozen snapshot:** build the full citation neighbourhood once, freeze it, run every simulation offline. Resolves API rate limits, cuts the experiment from weeks to hours, makes the study fully replicable, and dogfoods the determinism layer. The snapshot is deposited on Zenodo alongside the code.

**Metrics (`simulation/metrics.py`):**
- `recall@iteration`, precision, screening burden to reach 80/90/95% recall
- Per stopping rule: actual recall at stop, over/undershoot vs oracle-optimal stop, regret (excess screening or missed studies)
- CMR calibration: estimated vs true recall — calibration plot, bias, 95% CI coverage rate
- Ablations: backward/forward/both; OpenAlex/+Crossref/+S2/+GROBID; RRF/BM25/network/random

**Baselines:** exhaustion · single iteration (= what citationchaser and paperfetcher do) · paperfetcher itself where runnable · random candidate order · Boolean search alone where the original query is available.

**Statistics:** unit of analysis is the review, not the record. Mixed-effects models with review as random effect; Friedman + post-hoc with multiple-comparison correction across stopping rules; effect sizes with CIs; results disaggregated by domain.

---

## 17. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 | Skeleton, types, config, CI | `mypy --strict` clean, CI green |
| M1 | Core loop with `offline` provider | Fixture run completes, state persists and resumes |
| M2 | OpenAlex + Crossref + S2 | Live smoke test on 5 seeds, cache populated |
| M3 | Identity + dedup | All property tests pass |
| M4 | Ranking (BM25, network, RRF) | Deterministic ordering test passes |
| M5 | Stopping rules | Synthetic-curve tests pass |
| M6 | Capture–recapture + diagnostics | Synthetic-population bias tests pass |
| M7 | Determinism: cache, manifest, verify | INV-1 test green; `verify --refresh` produces classified diff |
| M8 | Reporting: PRISMA, audit, exports, graph | Valid SVG; GraphML opens in VOSviewer/Gephi |
| M9 | GROBID provider | End-to-end on 20 local PDFs, match rate reported |
| M10 | Simulation harness + benchmarks | Full run on ≥25 reviews, results tables generated |
| M11 | Docs, README, examples, Zenodo, PyPI | `pip install snowballslr` works from clean env |

M0–M8 are the SoftwareX submission. M9–M11 complete the package; M10 additionally feeds the separate methodological paper.

---

## 18. Acceptance criteria for v1.0.0

Status measured 2026-08-09 against this repository; a criterion is only marked met
where it was actually run.

1. INV-1 through INV-6 hold, each with a dedicated passing test. — **met**: INV-1 and INV-4 in `tests/test_determinism.py` and the autouse fixture in `tests/conftest.py`; INV-2 and INV-3 in `tests/test_invariants.py`; INV-5 in `tests/test_ranking.py`; INV-6 in `tests/test_types.py` and `tests/test_invariants.py`.
2. `pip install snowballslr` in a clean venv; the quickstart in the README runs end-to-end offline on shipped fixtures. — **not met**: the package is not yet published to PyPI. Blocker for submission.
3. Coverage ≥ 90% line / ≥ 80% branch. — **line met** at 91% (3543 statements, 314 missed); branch coverage not measured.
4. `verify` detects and correctly classifies injected drift in a synthetic scenario covering all seven diff classes.
5. Benchmark run over ≥25 reviews completes from the frozen snapshot in under 2 hours on a laptop.
6. `report.md` contains a stopping-rule rationale sentence directly usable in a methods section without editing.
7. Zenodo DOI minted; `CITATION.cff` present; SoftwareX code-metadata table complete.

---

## 19. Positioning constraints (carry into code, docs, and paper)

- **Never claim** "first Python library for snowballing." `paperfetcher` (Pallath & Zhang 2023) exists and does bidirectional citation searching in beta. The README must cite it in a "Related tools" section.
- **Claim instead:** first library to combine automatic iteration to an explicitly defined saturation point, capture–recapture recall estimation, and bit-for-bit reproducibility against mutating APIs, in a programmatic pipeline.
- **Cite and delineate** in both README and paper: `citationchaser` (R/Shiny, single pass), `paperfetcher` (Python, single beta layer, Crossref/COCI), `LitBall` (Kotlin desktop, iterative, no recall estimation, no reproducibility layer), Rajit et al. 2025 (simulation study with replication scripts, not a library), Bron et al. 2025 (Chao's estimator as a stopping criterion — for TAR screening, not snowballing).
- The documentation must state plainly that snowballing is a **complementary** method: automated citation searching has been shown not to beat Boolean search on recall. SnowSLR's value proposition is knowing when to stop and how much has been found — not replacing database search.
