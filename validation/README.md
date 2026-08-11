# Validation study

Code reproducing the results in `../VALIDATION_REPORT.md`. Everything here is
offline and deterministic; no module touches a live API.

- `validation_generator.py` — synthetic citation-graph generator. Temporally
  acyclic, topic-homophilous, with per-provider edge visibility driven by a shared
  latent indexability (so provider arms are positively dependent, as in reality)
  and a tunable share of relevant works with no citation path to the rest.
- `validation_arms.py` — frontier expansion that records, for each discovered
  work, which arm found it under each candidate arm design. Backs §3 and §4.
- `validation_stopping.py` — runs a review to exhaustion and scores each stopping
  rule against the oracle-optimal stop. Backs §2.

Reproduce with a fixed seed, e.g.:

```python
from validation_generator import GraphSpec, generate_graph
from validation_stopping import run_to_exhaustion, rule_marginal_yield

graph = generate_graph(GraphSpec(seed=1))
history = run_to_exhaustion(graph, sorted(graph["gold"])[:5], set(graph["gold"]))
```

`GraphSpec.seed` fully determines the graph; every trial in the report is keyed by
`(regime, graph seed, seed strategy, seed size, repeat)`.
