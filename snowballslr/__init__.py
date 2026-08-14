"""SnowBallSLR -- deterministic, saturation-aware snowballing for systematic reviews.

Automated bidirectional citation searching with three properties no existing
tool combines: automatic iteration to an explicitly defined saturation point,
recall estimation by capture-recapture, and bit-for-bit reproducibility against
mutating bibliographic APIs.

Screening stays external by design: the library retrieves, deduplicates, ranks
and decides when to stop. It never decides what to include.

Quickstart::

    from snowballslr import Run, Config

    run = Run.init("./run", seeds=["10.1000/abc"], config=Config())
    while not run.stopped:
        candidates = run.step()
        if not candidates:
            break
        run.label({w.key: my_screen(w) for w in candidates})
    print(run.estimate_recall())
    run.report(prisma=True, graph=True)
"""

from __future__ import annotations

__version__ = "1.1.0"

from .config import Config
from .core.run import Run
from .core.state import Phase, RunState
from .errors import (
    ConfigError,
    EstimationError,
    LabelError,
    OfflineError,
    ProviderError,
    SnowballError,
    StateError,
    VerificationError,
)
from .estimate import RecallEstimate, estimate_recall
from .types import Author, Decision, Direction, Provenance, Work

__all__ = [
    "Author",
    "Config",
    "ConfigError",
    "Decision",
    "Direction",
    "EstimationError",
    "LabelError",
    "OfflineError",
    "Phase",
    "Provenance",
    "ProviderError",
    "RecallEstimate",
    "Run",
    "RunState",
    "SnowballError",
    "StateError",
    "VerificationError",
    "Work",
    "__version__",
    "estimate_recall",
]
