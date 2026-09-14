"""Explicit marginal and innovation priors for models with shared states.

Initial-state priors belong to the component declarations. This object controls
process standard deviations and each channel's observation parameters; it does
not silently introduce selection indicators or residual dependence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from .process import (
    FixedSD, HalfStudentTSD, PCSD, SDPrior, ShapePrior, TruncatedNormalPrior,
    sd_prior_from_dict, shape_prior_from_dict,
)


@dataclass(frozen=True)
class JointPriors:
    process: Mapping[str, SDPrior]
    observation_sd: Mapping[str, SDPrior]
    shape: Mapping[str, ShapePrior] = field(default_factory=dict)
    profile: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process": {key: prior.to_dict() for key, prior in self.process.items()},
            "observation_sd": {key: prior.to_dict() for key, prior in self.observation_sd.items()},
            "shape": {key: prior.to_dict() for key, prior in self.shape.items()},
            "profile": self.profile,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JointPriors":
        return cls(
            process={key: sd_prior_from_dict(item) for key, item in value["process"].items()},
            observation_sd={key: sd_prior_from_dict(item) for key, item in value["observation_sd"].items()},
            shape={key: shape_prior_from_dict(item) for key, item in value.get("shape", {}).items()},
            profile=value.get("profile", "custom"),
            metadata=value.get("metadata", {}),
        )


def default_joint_priors(compiled: Any) -> JointPriors:
    """Recorded data-adaptive convenience priors; use explicit priors for studies.

    The reference scale never uses the sample size. Process names ending in
    ``slope`` are divided by ten seasonal cycles as a heuristic for their
    integration into the level; this is not a prior on total horizon change.
    """
    period = int(getattr(compiled.model, "period", None) or 1)
    upper = {}
    for name in compiled.noise_names:
        reference = float(compiled.process_reference_scale.get(name, compiled.difference_scale))
        reference = max(reference, 0.05 * float(compiled.observation_scale), 1e-8)
        value = 0.1 * reference
        if name.split(".")[-1] == "slope":
            value /= 10 * period
        if name.startswith("departure."):
            value *= 0.5
        upper[name] = value
    shapes = {}
    for channel in compiled.model.channels:
        if channel.family == "gev":
            lower, higher = channel.observation.xi_bounds
            shapes[channel.name] = TruncatedNormalPrior(0.0, 0.2, lower, higher)
    return JointPriors(
        process={key: PCSD(value, alpha=0.05) for key, value in upper.items()},
        observation_sd={
            name: HalfStudentTSD(4.0, max(float(compiled.channel_observation_scale[name]), 1e-8))
            for name in compiled.channel_names
        },
        shape=shapes,
        profile="data_adaptive_regularized",
        metadata={
            "upper_by_process": upper,
            "period": period,
            "calibration": "PC P(sd > upper)=0.05; upper=0.1 robust local-change reference, slope divided by ten cycles, departures additionally halved.",
            "data_adaptive": True,
            "publication_note": "Specify scientific priors explicitly and assess sensitivity; these defaults use the observed series for calibration.",
        },
    )


def resolve_joint_priors(compiled: Any, priors: JointPriors | None) -> JointPriors:
    resolved = default_joint_priors(compiled) if priors is None else priors
    if not isinstance(resolved, JointPriors):
        raise TypeError("Shared-state models require JointPriors or None; univariate SSVS and hierarchy priors do not apply.")
    expected = {
        "process": set(compiled.noise_names),
        "observation_sd": set(compiled.channel_names),
        "shape": {channel.name for channel in compiled.model.channels if channel.family == "gev"},
    }
    for field_name, keys in expected.items():
        actual = set(getattr(resolved, field_name))
        if actual != keys:
            raise ValueError(f"JointPriors.{field_name} keys do not match the model; missing={sorted(keys - actual)}, extra={sorted(actual - keys)}.")
    for name, prior in {**resolved.process, **{f"sigma.{k}": p for k, p in resolved.observation_sd.items()}}.items():
        for method in ("initial", "logpdf", "sample", "to_dict"):
            if not callable(getattr(prior, method, None)):
                raise TypeError(f"Prior {name!r} must implement the SD-prior protocol ({method}).")
        start = float(prior.initial())
        if not np.isfinite(start) or start < 0.0 or not np.isfinite(prior.logpdf(start)):
            raise ValueError(f"Prior {name!r} must have a finite supported initial SD.")
        if start == 0.0 and not isinstance(prior, FixedSD):
            raise ValueError(f"A nonfixed SD prior must initialize strictly above zero: {name!r}.")
    for channel in compiled.model.channels:
        prior = resolved.observation_sd[channel.name]
        if float(prior.initial()) <= 0.0:
            raise ValueError(f"Observation scale for {channel.name} must be strictly positive.")
        if channel.family == "gev":
            shape = resolved.shape[channel.name]
            lower, upper = channel.observation.xi_bounds
            if shape.lower < lower or shape.upper > upper:
                raise ValueError(f"Shape prior for {channel.name} must lie within observation.xi_bounds.")
            if not np.isfinite(shape.logpdf(shape.initial())):
                raise ValueError(f"Shape prior for {channel.name} must have a finite supported initial value.")
    return replace(resolved, metadata={
        **dict(resolved.metadata),
        "resolved_initial_mean": compiled.initial_mean.tolist(),
        "resolved_initial_sd": np.sqrt(np.diag(compiled.initial_cov)).tolist(),
        "innovation_prior_note": "Process mappings refer to unique SD parameters; a departure SD may be reused by multiple contrast innovations.",
    })


__all__ = ["JointPriors", "default_joint_priors", "resolve_joint_priors"]
