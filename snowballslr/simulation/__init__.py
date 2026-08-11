"""Simulation harness for validating stopping rules against gold standards."""

from .harness import ReviewSpec, SeedStrategy, TrialResult, run_benchmark, run_trial
from .metrics import (
    RecallCurve,
    StoppingOutcome,
    calibration,
    evaluate_stopping_rules,
    recall_curve,
    screening_burden,
)
from .oracle import Oracle, load_gold_standard

__all__ = [
    "Oracle",
    "RecallCurve",
    "ReviewSpec",
    "SeedStrategy",
    "StoppingOutcome",
    "TrialResult",
    "calibration",
    "evaluate_stopping_rules",
    "load_gold_standard",
    "recall_curve",
    "run_benchmark",
    "run_trial",
    "screening_burden",
]
