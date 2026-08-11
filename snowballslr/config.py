"""Configuration model.

The resolved configuration is hashed into the run manifest. Secrets and contact
addresses are excluded from the hash so that two researchers using different
polite-pool mailtos still produce comparable ``config_hash`` values.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .errors import ConfigError, ConfigWarning

__all__ = [
    "LEGACY_OPENALEX_TYPES",
    "Config",
    "DedupSettings",
    "DeterminismSettings",
    "EligibilitySettings",
    "EstimateSettings",
    "ProvidersSettings",
    "RankingSettings",
    "RunSettings",
    "StoppingSettings",
    "canonical_json",
]

_SECRET_FIELDS = {"mailto", "api_key", "api_key_env", "url", "pdf_map"}

# OpenAlex replaced its Crossref-derived type vocabulary in July 2023:
# journal-article, proceedings-article and posted-content were all merged into
# the single type "article". A config written against the old vocabulary does
# not error -- it silently discards every journal and conference paper.
LEGACY_OPENALEX_TYPES = {
    "journal-article": "article",
    "proceedings-article": "article",
    "posted-content": "preprint",
}


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class OpenAlexSettings(BaseModel):
    enabled: bool = True
    mailto: str | None = None
    rps: float = 8.0
    base_url: str = "https://api.openalex.org"
    per_page: int = 200


class CrossrefSettings(BaseModel):
    enabled: bool = True
    mailto: str | None = None
    rps: float = 8.0
    base_url: str = "https://api.crossref.org"


class SemanticScholarSettings(BaseModel):
    enabled: bool = False
    api_key_env: str = "S2_API_KEY"
    rps: float = 1.0
    base_url: str = "https://api.semanticscholar.org/graph/v1"


class GrobidSettings(BaseModel):
    enabled: bool = False
    url: str = "http://localhost:8070"
    pdf_map: str | None = None
    min_match_score: float = 60.0
    min_title_similarity: float = 0.90


class ProvidersSettings(BaseModel):
    order: list[str] = Field(default_factory=lambda: ["openalex", "crossref"])
    precedence: list[str] = Field(
        default_factory=lambda: ["openalex", "crossref", "semanticscholar", "grobid"]
    )
    openalex: OpenAlexSettings = Field(default_factory=OpenAlexSettings)
    crossref: CrossrefSettings = Field(default_factory=CrossrefSettings)
    semanticscholar: SemanticScholarSettings = Field(
        default_factory=SemanticScholarSettings
    )
    grobid: GrobidSettings = Field(default_factory=GrobidSettings)


class RunSettings(BaseModel):
    name: str = "snowballslr-run"
    max_iterations: int = 10
    directions: list[Literal["backward", "forward"]] = Field(
        default_factory=lambda: ["backward", "forward"]
    )

    @field_validator("directions")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one direction must be enabled")
        return v


class EligibilitySettings(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    types: list[str] | None = None
    languages: list[str] | None = None
    exclude_retracted: bool = True
    require_title: bool = True
    require_abstract: bool = False
    # Records whose year, type or language is simply absent from the provider
    # are kept by default and left for the screener to judge. Treating missing
    # metadata as failing a criterion silently discards exactly the records with
    # the worst indexing -- older, non-English and non-STEM work -- which is the
    # population citation searching exists to recover. Set to false for a strict
    # run; either way the counts are reported separately.
    keep_unknown_metadata: bool = True


class DedupSettings(BaseModel):
    jw_threshold: float = 0.95
    max_cluster_size: int = 8


class RankingSettings(BaseModel):
    type: Literal["rrf", "bm25", "network", "embedding", "none"] = "rrf"
    components: list[str] = Field(default_factory=lambda: ["bm25", "network"])
    network_weights: list[float] = Field(default_factory=lambda: [0.4, 0.25, 0.25, 0.10])
    rrf_k: int = 60
    top_k: int | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None


class StoppingRuleSettings(BaseModel):
    type: str
    model_config = {"extra": "allow"}


class StoppingSettings(BaseModel):
    mode: Literal["any_of", "all_of"] = "any_of"
    rules: list[StoppingRuleSettings] = Field(
        default_factory=lambda: [
            StoppingRuleSettings(type="marginal_yield", eps=0.01, k=2),
            StoppingRuleSettings(type="exhaustion"),
        ]
    )


class ArmSettings(BaseModel):
    name: str
    filter: dict[str, str] = Field(default_factory=dict)


class EstimateSettings(BaseModel):
    # Provider arms, not direction arms. Backward and forward expansion sample
    # near-disjoint strata of a temporally acyclic citation graph, so direction
    # arms typically yield zero overlap and no estimate at all; two providers can
    # both return the same record. See Config._arm_design_warnings.
    arms: list[ArmSettings] = Field(
        default_factory=lambda: [
            ArmSettings(name="openalex", filter={"provider": "openalex"}),
            ArmSettings(name="crossref", filter={"provider": "crossref"}),
        ]
    )
    method: Literal["chapman", "loglinear", "chao"] = "chapman"


class DeterminismSettings(BaseModel):
    cache_dir: str = "cache"
    offline: bool = False
    refresh: bool = False
    max_retries: int = 5
    backoff_base: float = 1.0
    backoff_factor: float = 2.0


class Config(BaseModel):
    model_config = {"extra": "ignore"}

    run: RunSettings = Field(default_factory=RunSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    eligibility: EligibilitySettings = Field(default_factory=EligibilitySettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    ranking: RankingSettings = Field(default_factory=RankingSettings)
    stopping: StoppingSettings = Field(default_factory=StoppingSettings)
    estimate: EstimateSettings = Field(default_factory=EstimateSettings)
    determinism: DeterminismSettings = Field(default_factory=DeterminismSettings)

    # -- io --------------------------------------------------------------

    @staticmethod
    def from_yaml(path: str | Path) -> Config:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return Config.model_validate(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Config:
        return Config.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    # -- hashing ---------------------------------------------------------

    def hashable(self) -> dict[str, Any]:
        """Config with secrets and contact details stripped."""
        return _strip_secrets(self.model_dump(mode="json"))

    @property
    def config_hash(self) -> str:
        return "sha256:" + hashlib.sha256(
            canonical_json(self.hashable()).encode("utf-8")
        ).hexdigest()

    # -- validation ------------------------------------------------------

    def validate_consistency(self) -> None:
        enabled = {
            name
            for name in ("openalex", "crossref", "semanticscholar", "grobid")
            if getattr(self.providers, name).enabled
        }
        unknown = [p for p in self.providers.order if p not in enabled | {"offline"}]
        if unknown:
            raise ConfigError(
                f"providers.order references disabled or unknown providers: {unknown}"
            )
        if not self.providers.order:
            raise ConfigError("providers.order is empty")
        if "forward" in self.run.directions and self.providers.order == ["crossref"]:
            raise ConfigError(
                "forward direction requested but crossref is the only provider "
                "(crossref exposes references only)"
            )
        if self.ranking.type == "embedding" and not self.ranking.embedding_model:
            raise ConfigError("ranking.type=embedding requires ranking.embedding_model")

        if not getattr(self, "_warned", False):
            object.__setattr__(self, "_warned", True)
            for warning in self.warnings():
                warnings.warn(warning, ConfigWarning, stacklevel=2)

    def warnings(self) -> list[str]:
        """Non-fatal configuration problems that would silently distort a run."""
        out: list[str] = []
        if self.providers.openalex.enabled and self.eligibility.types:
            legacy = sorted(set(self.eligibility.types) & set(LEGACY_OPENALEX_TYPES))
            if legacy:
                suggested = sorted(
                    {LEGACY_OPENALEX_TYPES[t] for t in legacy}
                    | (set(self.eligibility.types) - set(legacy))
                )
                out.append(
                    "eligibility.types uses OpenAlex's pre-2023 vocabulary "
                    f"({', '.join(legacy)}). Since July 2023 OpenAlex reports all "
                    "journal, conference and preprint items under different type "
                    "values, so these entries match nothing and every article is "
                    f"silently discarded. Suggested replacement: {suggested}"
                )

        out.extend(self._arm_design_warnings())
        return out

    def _arm_design_warnings(self) -> list[str]:
        """Arm configurations that cannot yield an interpretable recall estimate.

        Capture-recapture needs arms that can both capture the same record. Two
        arm designs in common use cannot, and both fail silently -- the run
        completes, the estimate is simply refused for "insufficient overlap",
        and the user is left thinking their corpus was unlucky rather than their
        design impossible.
        """
        out: list[str] = []
        filters = [dict(a.filter or {}) for a in self.estimate.arms]
        directions = {f.get("direction") for f in filters if f.get("direction")}
        providers = {f.get("provider") for f in filters if f.get("provider")}

        if len(self.estimate.arms) >= 2 and directions >= {"backward", "forward"}:
            out.append(
                "estimate.arms uses backward and forward direction as the two "
                "capture arms. A citation graph is temporally acyclic, so a work "
                "older than the seed set can normally only be reached backward and "
                "a newer one only forward: the arms sample near-disjoint strata, "
                "the overlap m tends to 0, and the estimate is refused as "
                "uninterpretable. Overlap only accumulates once a *later* "
                "iteration reaches a work from the opposite direction, so this "
                "design also cannot produce an estimate at iteration 1. Prefer "
                "provider arms -- e.g. {name: openalex, filter: {provider: "
                "openalex}} and {name: crossref, filter: {provider: crossref}} -- "
                "which can both capture the same record, and add a third provider "
                "to make the independence assumption testable."
            )

        if len(providers) == 1 and len(self.estimate.arms) >= 2:
            out.append(
                "every arm in estimate.arms filters on the same provider; the arms "
                "cannot capture disjoint evidence and the estimate will be "
                "degenerate."
            )

        enabled_providers = {
            name
            for name in ("openalex", "crossref", "semanticscholar", "grobid")
            if getattr(self.providers, name).enabled
        }
        missing = providers - enabled_providers
        if missing:
            out.append(
                f"estimate.arms references provider(s) {sorted(missing)} that are "
                "not enabled in providers.order, so those arms will capture "
                "nothing."
            )

        if len(self.estimate.arms) == 2 and self.method_needs_three_arms():
            out.append(
                f"estimate.method='{self.estimate.method}' is configured with only "
                "two arms; log-linear model selection requires three or more."
            )
        return out

    def method_needs_three_arms(self) -> bool:
        return self.estimate.method == "loglinear"


def _strip_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if k in _SECRET_FIELDS else _strip_secrets(v))
            for k, v in sorted(obj.items())
        }
    if isinstance(obj, list):
        return [_strip_secrets(v) for v in obj]
    return obj
