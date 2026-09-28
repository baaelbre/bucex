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
        and trend[0].level_mode in {"dynamic", "static"}
        and trend[0].trend_mode in {"dynamic", "static", "off"}
        and len(seasonal) <= 1
        and all(component.mode in {"dynamic", "static", "off"} for component in seasonal)
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
    is_multiseries = hasattr(compiled, "channel_names")
    seasonal_scale = (not hasattr(compiled, "channel_names")
                      and getattr(compiled.model.observation, "scale", None) is not None)
    if is_multiseries or seasonal_scale:
        from .fit.marginal import marginal_plan
        return marginal_plan(compiled, engine=engine, parameterization=parameterization, asis=asis)
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

    fs_available = supports_fs(compiled)
    # Preserve the established automatic generic route for static scalar
    # models. The explicit FS route now supports those reductions as well.
    if not is_multiseries and str(parameterization).lower() == "auto" and any(
            getattr(c,"level_mode",None) == "static" or getattr(c,"trend_mode",None) == "static" or
            getattr(c,"mode",None) == "static" for c in compiled.model.components):
        fs_available = False
    resolved_parameterization = normalize_parameterization(
        parameterization,
        fs_supported=fs_available,
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
        backend=("fruehwirth_schnatter" if resolved_parameterization == "fruehwirth_schnatter" else "state_space"),
        state_update=state_update,
        targets_exact_posterior=exact,
        approximation=approximation,
        proposal=proposal,
        warnings=tuple(warnings),
    )
