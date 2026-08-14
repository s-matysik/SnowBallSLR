# Running SnowBallSLR in Google Colab

Everything below was run end to end on a clean Python 3.12 environment — the version
Colab ships — with no editable install and no repository on the path. Where a step's
output is quoted, it is the output that step actually produced.

Colab gives you a throwaway machine with network access, which suits citation searching:
the crawl is I/O-bound, not compute-bound, so the free tier is enough. What it does *not*
give you is a persistent filesystem. Runs are reproducible from their own cache, so it is
worth keeping the run directory — the last section covers that.

## 1. Install

```python
!pip install -q git+https://github.com/s-matysik/SnowBallSLR.git
```

Verify, in a new cell:

```python
import snowballslr
print(snowballslr.__version__)      # 1.1.0
```

The command-line interface is installed too, if you prefer it:

```python
!snowballslr --help
```

**Optional extras.** Log-linear estimation with three or more arms needs `statsmodels`;
embedding-based ranking needs `sentence-transformers`, which is a large download:

```python
!pip install -q "snowballslr[loglinear] @ git+https://github.com/s-matysik/SnowBallSLR.git"
```

## 2. A live run in a notebook

`pip install` gives you the library, not the demonstration data — the offline fixtures live
in the repository, not in the wheel. So the quickest real thing to do in Colab is a live
crawl. This is the shape of it: one seed DOI, backward only, a screening budget so it
cannot run away from you.

```python
from pathlib import Path
from snowballslr.config import Config
from snowballslr.core.run import Run

config = Config.from_dict({
    "run": {"name": "colab-demo", "max_iterations": 2, "directions": ["backward"]},
    "providers": {
        "order": ["openalex", "crossref"],
        # Both APIs ask for a contact address and give higher rate limits with one.
        "openalex": {"enabled": True, "rps": 4, "mailto": "you@example.org"},
        "crossref": {"enabled": True, "rps": 4, "mailto": "you@example.org"},
    },
    "eligibility": {"types": ["article", "book-chapter"], "require_title": True},
    "stopping": {"mode": "any_of", "rules": [
        {"type": "budget", "max_screened": 60},
        {"type": "exhaustion"},
    ]},
    "estimate": {
        # Arms must be able to capture the SAME record. Backward and forward expansion
        # sample near-disjoint layers of a temporally acyclic graph, so direction arms
        # overlap on nothing and yield no estimate. Define arms by source instead.
        "arms": [
            {"name": "openalex", "filter": {"provider": "openalex"}},
            {"name": "crossref", "filter": {"provider": "crossref"}},
        ],
        "method": "chapman",
    },
})

for warning in config.warnings():
    print("config warning:", warning)

run = Run.init(Path("run_demo"), seeds=["10.1145/2601248.2601268"], config=config)
print("seeds resolved:", len(run.state.works))

candidates = run.step()
print("candidates to screen:", len(candidates))
for work in candidates[:5]:
    print(" ", work.year, (work.title or "")[:70])
```

Running exactly this produced `seeds resolved: 1` and `candidates: 18`.

Read the config warnings rather than skipping them. They are the library's way of
catching a design that cannot work — an arm on a disabled provider, an outdated type
vocabulary, a set of arms that cannot overlap.

## 3. Screening, which is the part that needs you

`step()` hands back candidates; nothing decides them for you. Screen them, then feed the
decisions back:

```python
decisions = {}
for work in candidates:
    print(f"\n{work.year}  {work.title}")
    print((work.abstract or "[no abstract]")[:400])
    answer = input("include / exclude / skip? ").strip().lower()
    decisions[work.key] = {"i": "include", "e": "exclude"}.get(answer[:1], "unscreened")

run.label(decisions)
print("phase:", run.state.phase, "| stopped by:", run.state.stopped_by)
```

For anything beyond a few dozen records, export a spreadsheet, screen it outside the
notebook, and read it back:

```python
run.label_from_file("labels_001.csv")   # columns: key, decision[, note]
```

Then call `run.step()` again for the next iteration, and repeat until `run.state.phase`
is `stopped`. Whichever rule fired is in `run.state.stopped_by`, and its reasoning —
written to be pasted into a methods section — is in `run.state.last_decisions`.

## 4. Outputs

```python
run.report()
!ls run_demo/outputs/
```

That writes `prisma.json` and `prisma.svg` (PRISMA 2020 flow counts and diagram),
`included.csv`, and `report.md` containing the run's own methods text. To show the
diagram inline:

```python
from IPython.display import SVG, display
display(SVG("run_demo/outputs/prisma.svg"))
```

## 5. Keeping the run when the runtime disappears

A Colab runtime is deleted when you close it. A completed run replays from its own cache
without contacting any API, so archiving the run directory preserves the review exactly:

```python
!snowballslr verify run_demo          # confirms the run replays from its cache

from google.colab import drive
drive.mount("/content/drive")
!cp -r run_demo /content/drive/MyDrive/
```

Bringing it back in a later session:

```python
from snowballslr.core.run import Run
run = Run.load("/content/drive/MyDrive/run_demo")
print(run.state.phase, run.state.iteration)
```

`verify` also checks that the configuration on disk still matches the one the run
executed under, so editing a threshold to continue a stopped run is reported rather than
silently changing what the outputs describe.

## 6. Reproducing the paper's numbers

The validation harness is offline and deterministic, but it lives in the repository
rather than the package, so clone rather than install:

```python
!git clone -q https://github.com/s-matysik/SnowBallSLR.git
%cd SnowBallSLR
!pip install -q -e .
!python validation/reproduce.py --quick
```

`--quick` uses fewer graph seeds and takes a couple of minutes; dropping it runs the full
270-cell grid. Each regenerated figure is printed beside the published one, and one
documented difference is stated rather than tuned away.

## Things worth knowing before you start

**Screening is yours.** The library automates the crawl, the deduplication, the stopping
decision and the recall estimate. It does not decide relevance. If you screen with a
language model, validate it: `validation/llm_screen_validate.py` runs six designs,
including blind positive controls and cross-vendor agreement, and refuses to compute a
figure over records scored by different numbers of raters.

**Ask for a small budget first.** A second iteration is typically five to twenty times
the first. On the reviews in this project iteration 1 ranged from 115 to 14,883
candidates; the growth is a property of your corpus, not of a setting, so measure it on
one iteration before committing to a screening effort.

**Both APIs are free and neither needs a key.** Supplying `mailto` is courteous and
raises your rate limit. Semantic Scholar is supported and does take an optional key via
`api_key_env`.

**Colab's `input()` blocks the cell.** Fine for a handful of records, painful for
hundreds — use the CSV route instead.
