"""Citation network export (GraphML), openable in VOSviewer or Gephi."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from ..core.state import RunState

__all__ = ["build_graph", "write_graphml"]


def build_graph(state: RunState) -> nx.DiGraph:
    g = nx.DiGraph()
    for key in sorted(state.works):
        w = state.works[key]
        g.add_node(
            key,
            label=w.title[:120],
            title=w.title,
            year=w.year if w.year is not None else 0,
            doi=w.doi or "",
            venue=w.venue or "",
            decision=str(state.decision_for(key)),
            is_seed=key in set(state.seeds),
            providers="|".join(w.providers),
            unresolved=bool(w.unresolved),
        )
    for key in sorted(state.works):
        w = state.works[key]
        for prov in sorted(w.provenance):
            if not prov.parent_key:
                continue
            parent = state.resolve_alias(prov.parent_key)
            if parent not in g:
                continue
            # Backward: parent cites child. Forward: child cites parent.
            if str(prov.direction) == "backward":
                g.add_edge(parent, key, direction="backward", iteration=prov.iteration)
            else:
                g.add_edge(key, parent, direction="forward", iteration=prov.iteration)
    return g


def write_graphml(path: str | Path, state: RunState) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(build_graph(state), p, named_key_ids=True)
    return p
