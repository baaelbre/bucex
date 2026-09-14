"""Resolve and validate model, engine and parameterization choices."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..components import DummySeasonal, LocalLevel, LocalLinearTrend, Regression
from ..models.compiler import CompiledModel


@dataclass(frozen=True)
class InferencePlan:
    family: str
    engine: str
    parameterization: str
    asis: bool
    state_update: str
    targets_exact_posterior: bool
    approximation: str | None
    proposal: str | None = None
    interweaves_with: str | None = None
    backend: str = "state_space"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InferencePlan":
        payload = dict(value)
        payload["warnings"] = tuple(payload.get("warnings", ()))
        payload.setdefault("proposal", None)
        payload.setdefault("interweaves_with", None)
        payload.setdefault("backend", "state_space")
        return cls(**payload)


def supports_fs(compiled: CompiledModel) -> bool:
    """Whether the exact FS augmented layout exists for this model."""

    if hasattr(compiled, "channel_names"):
        return bool(
            getattr(compiled.model, "supports_fs_parameterization", False)
        )
    if not isinstance(compiled, CompiledModel):
        return False

    trend = [component for component in compiled.model.components if isinstance(component, LocalLinearTrend)]
    seasonal = [component for component in compiled.model.components if isinstance(component, DummySeasonal)]
    regression = [component for component in compiled.model.components if isinstance(component, Regression)]
    return bool(
        len(trend) == 1
        and trend[0].level_mode == "dynamic"
        and trend[0].trend_mode in {"dynamic", "off"}
        and len(seasonal) <= 1
        and all(component.mode in {"dynamic", "off"} for component in seasonal)
        and not regression
    )


def normalize_parameterization(value: str, *, fs_supported: bool) -> str:
    key = str(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "centred": "centered",
        "fs": "fruehwirth_schnatter",
        "fruhwirth_schnatter": "fruehwirth_schnatter",
        "noncentered": "fruehwirth_schnatter",
        "noncentred": "fruehwirth_schnatter",
        "ncp": "fruehwirth_schnatter",
        "scaled_disturbances": "disturbance",
        "scaled_disturbance": "disturbance",
        "innovation": "disturbance",
    }
    if key == "auto":
        return "fruehwirth_schnatter" if fs_supported else "disturbance"
    resolved = aliases.get(key, key)
    if resolved not in {"centered", "fruehwirth_schnatter", "disturbance"}:
        raise ValueError(
            "parameterization must be centered, fruehwirth_schnatter, "
            "disturbance, or auto."
        )
    if resolved == "fruehwirth_schnatter" and not fs_supported:
        raise ValueError(
            "The FS augmented parameterization requires either one univariate "
            "local-linear trend with optional dummy seasonality, or a "
            "MultiSeriesModel whose channels have that structural layout. Use "
            "parameterization='disturbance' for other model graphs."
        )
    return resolved


def inference_plan(
    compiled: Any,
    *,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
) -> InferencePlan:
    family = compiled.family
    if bool(getattr(compiled, "is_shared", False)):
        return _shared_inference_plan(
            compiled, engine=engine, parameterization=parameterization, asis=asis
        )
    is_multiseries = hasattr(compiled, "channel_names")
    all_gaussian = bool(getattr(compiled, "all_gaussian", family == "gaussian"))
    requested_engine = str(engine).lower().replace("-", "_")
    if requested_engine in {"pgas", "particle"}:
        raise ValueError(
            "PGAS is retired in BUCEX 1.6. Use engine='laplace_mh' for exact "
            "non-Gaussian inference, or engine='ffbs' for Gaussian models."
        )
    resolved_engine = (
        "ffbs" if all_gaussian else "laplace_mh"
    ) if requested_engine == "auto" else requested_engine
    allowed = {"ffbs"} if all_gaussian else {"laplace", "laplace_mh"}
    if resolved_engine not in allowed:
        raise ValueError(
            f"engine='{resolved_engine}' is incompatible with family='{family}'; "
            f"choose {sorted(allowed)}."
        )

    resolved_parameterization = normalize_parameterization(
        parameterization,
        fs_supported=supports_fs(compiled),
    )
    if is_multiseries and resolved_parameterization != "fruehwirth_schnatter":
        raise ValueError(
            "MultiSeriesModel uses joint hierarchical inference and therefore "
            "requires parameterization='fruehwirth_schnatter' (alias 'fs')."
        )
    if is_multiseries and asis:
        raise ValueError(
            "ASIS is not combined with the joint hierarchical sampler; use asis=False."
        )
    exact = resolved_engine in {"ffbs", "laplace_mh"}
    approximation = None if exact else "iterated_laplace"
    proposal = "iterated_laplace_smoother" if resolved_engine == "laplace_mh" else None
    if asis:
        interweaves_with = (
            "centered"
            if resolved_parameterization in {"fruehwirth_schnatter", "disturbance"}
            else "disturbance"
        )
    else:
        interweaves_with = None

    warnings: list[str] = []
    if is_multiseries:
        warnings.append(
            "Channels share structural-selection probabilities and slab scales, "
            "while retaining separate latent paths. Residual/copula dependence "
            "is not modeled."
        )
        if resolved_engine == "laplace":
            warnings.append(
                "Hierarchical Laplace SSVS is an exploratory approximation. Use "
                "it for screening or initialization, and rerun exact Laplace-MH "
                "for final non-Gaussian inference."
            )
    state_update = {
        "ffbs": "exact Gaussian FFBS",
        "laplace": "iterated Laplace FFBS approximation",
        "laplace_mh": "Laplace independence Metropolis-Hastings",
    }[resolved_engine]
    return InferencePlan(
        family=family,
        engine=resolved_engine,
        parameterization=resolved_parameterization,
        asis=bool(asis),
        interweaves_with=interweaves_with,
        backend=(
            "hierarchical_state_space"
            if is_multiseries
            else (
                "fruehwirth_schnatter"
                if resolved_parameterization == "fruehwirth_schnatter"
                else "state_space"
            )
        ),
        state_update=state_update,
        targets_exact_posterior=exact,
        approximation=approximation,
        proposal=proposal,
        warnings=tuple(warnings),
    )


def _shared_inference_plan(
    compiled: Any,
    *,
    engine: str,
    parameterization: str,
    asis: bool,
) -> InferencePlan:
    """Explicit support contract for graphs with states shared across channels."""

    requested = str(engine).lower().replace("-", "_")
    if requested in {"pgas", "particle"}:
        raise ValueError(
            "PGAS is retired in BUCEX 1.6; use engine='laplace_mh'."
        )
    all_gaussian = bool(compiled.all_gaussian) and compiled.model.copula is None
    resolved = ("ffbs" if all_gaussian else "laplace_mh") if requested == "auto" else requested
    allowed = {"ffbs"} if all_gaussian else {"laplace_mh"}
    if resolved not in allowed:
        raise ValueError(
            f"engine='{resolved}' is incompatible with shared family='{compiled.family}'; "
            f"choose {sorted(allowed)}."
        )
    parameterization_key = str(parameterization).lower().replace("-", "_")
    if parameterization_key not in {"auto", "centered", "centred"}:
        raise ValueError(
            "Shared-state models currently require parameterization='centered' "
            "(or 'auto'); FS structural selection is available for univariate "
            "and independent-path hierarchical models."
        )
    if asis:
        raise ValueError("ASIS is not supported for shared-state models; use asis=False.")
    warnings = ([
        "A Gaussian copula models contemporaneous residual dependence in original-response "
        "orientation. The full observation Hessian defines a joint Laplace proposal, corrected against the exact joint likelihood. "
        "Gaussian copulas do not impose deterministic ordering of summaries."
    ] if compiled.model.copula is not None else [
        "Shared temporal states induce dependence across channels; observation "
        "errors remain conditionally independent. Residual/copula dependence is not modeled."
    ])
    from ..models.shared import Departures

    has_departure_trend = any(
        isinstance(item, Departures)
        and isinstance(item.component, (LocalLevel, LocalLinearTrend))
        for item in compiled.model.shared
    )
    has_shared_trend = any(
        not isinstance(item, Departures)
        and isinstance(item.component, (LocalLevel, LocalLinearTrend))
        for item in compiled.model.shared
    )
    private_trends = [
        channel.name
        for channel in compiled.model.channels
        if any(
            (isinstance(component, LocalLevel) and component.mode == "dynamic")
            or (
                isinstance(component, LocalLinearTrend)
                and (
                    component.level_mode == "dynamic"
                    or component.trend_mode != "off"
                )
            )
            for component in channel.components
        )
    ]
    if has_shared_trend and has_departure_trend and private_trends:
        warnings.append(
            f"Channels {private_trends} also have private evolving levels or slopes. "
            "Those private changes are outside the weighted zero-sum departure "
            "constraint, so the shared trend is not the weighted mean of channel "
            "warming. Use fixed private intercepts for that interpretation."
        )
    return InferencePlan(
        family=compiled.family,
        engine=resolved,
        parameterization="centered",
        asis=False,
        state_update=(
            "exact joint Gaussian FFBS"
            if all_gaussian else "joint Laplace independence Metropolis-Hastings"
        ),
        targets_exact_posterior=True,
        approximation=None,
        proposal=(None if all_gaussian else "copula_joint_laplace" if compiled.model.copula is not None else "iterated_laplace_smoother"),
        backend="copula_state_space" if compiled.model.copula is not None else "shared_state_space",
        warnings=tuple(warnings),
    )
