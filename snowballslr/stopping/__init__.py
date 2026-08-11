"""Formal stopping rules for iterative citation searching."""

from .asymptote import AsymptoticCoverage
from .base import StopDecision, StoppingRule
from .budget import Budget
from .compose import RuleSet, build_rules
from .exhaustion import Exhaustion
from .recall_rule import EstimatedRecall
from .yield_rule import MarginalYield

__all__ = [
    "AsymptoticCoverage",
    "Budget",
    "EstimatedRecall",
    "Exhaustion",
    "MarginalYield",
    "RuleSet",
    "StopDecision",
    "StoppingRule",
    "build_rules",
]
