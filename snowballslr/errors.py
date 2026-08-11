"""Exception hierarchy. Exit codes are assigned in :mod:`snowballslr.cli`."""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "ConfigWarning",
    "EstimationError",
    "LabelError",
    "OfflineError",
    "ProviderError",
    "RateLimitError",
    "SnowballError",
    "StateError",
    "VerificationError",
]


class SnowballError(Exception):
    """Base class for all library errors."""


class ConfigError(SnowballError):
    """Invalid or inconsistent configuration."""


class ConfigWarning(UserWarning):
    """Configuration that is valid but will silently distort the run."""


class ProviderError(SnowballError):
    """Unrecoverable provider/transport failure."""


class RateLimitError(ProviderError):
    """Provider refused the request after the retry budget was exhausted."""


class OfflineError(ProviderError):
    """A network fetch was required while running in offline mode."""


class LabelError(SnowballError):
    """Malformed or inconsistent labels file."""


class VerificationError(SnowballError):
    """Replay produced artifacts that differ from the manifest (INV-1 violated)."""


class EstimationError(SnowballError):
    """Capture-recapture estimate is not computable from the given data."""


class StateError(SnowballError):
    """Run state is missing, corrupt, or in the wrong phase for the operation."""
