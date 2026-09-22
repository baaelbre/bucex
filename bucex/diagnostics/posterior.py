"""Multi-chain and engine-specific diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


Array = np.ndarray


def _finite_mean(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_median(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _split_chains(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (chains, draws).")
    half = values.shape[1] // 2
    if half < 2:
        return values
    return np.concatenate([values[:, :half], values[:, -half:]], axis=0)


def _rank_normalize(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1)
    ranks = rankdata(flat, method="average")
    return norm.ppf((ranks - 0.375) / (flat.size + 0.25)).reshape(values.shape)


def _basic_rhat(values: Array) -> float:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.nan
    n = values.shape[1]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(n * np.var(np.mean(values, axis=1), ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else np.inf
    variance = (n - 1.0) / n * within + between / n
    return float(np.sqrt(variance / within))


def rhat(values: Array) -> float:
    """Rank-normalized split R-hat with the folded-tail diagnostic."""

    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == values.reshape(-1)[0]):
        # R-hat compares within- and between-chain variation.  It is not one
        # when both are zero; it is mathematically undefined.
        return np.nan
    split = _split_chains(values)
    if split.shape[0] < 2 or split.shape[1] < 2:
        return np.nan
    bulk = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - float(np.median(split)))
    tail = _basic_rhat(_rank_normalize(folded))
    return float(max(bulk, tail))


def _autocovariance(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    n = values.size
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:n]
    return covariance / np.arange(n, 0, -1)


def ess_bulk(values: Array) -> float:
    """Rank-normalized split-chain bulk effective sample size."""

    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == values.reshape(-1)[0]):
        # A constant sampled indicator may represent posterior structural
        # certainty, but it supplies no autocorrelation information.
        return np.nan
    values = _rank_normalize(_split_chains(values))
    chains, draws = values.shape
    if draws < 3:
        return float(chains * draws)
    autocov = np.vstack([_autocovariance(chain) for chain in values])
    within = float(np.mean(autocov[:, 0]))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1)) if chains > 1 else 0.0
    variance_plus = (draws - 1.0) / draws * within + between / draws
    if variance_plus <= 0.0:
        return float(chains * draws)
    rho = np.ones(draws)
    for lag in range(1, draws):
        rho[lag] = 1.0 - (within - np.mean(autocov[:, lag])) / variance_plus
    pair_sums: list[float] = []
    for lag in range(1, draws - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0.0:
            break
        pair_sums.append(float(pair))
    for index in range(1, len(pair_sums)):
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    positive_sum = float(np.sum(pair_sums))
    return float(
        min(
            chains * draws,
            chains * draws / max(1.0 + 2.0 * positive_sum, 1e-12),
        )
    )


def ess_tail(values: Array) -> float:
    """Minimum ESS of the 5% and 95% empirical-quantile indicators."""
    values = np.asarray(values,float)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.all(values == values.flat[0]):
        return np.nan
    lower,upper = np.quantile(values,[.05,.95])
    return float(min(ess_bulk((values <= lower).astype(float)),ess_bulk((values <= upper).astype(float))))


def posterior_pit(fit) -> Array:
    """In-sample posterior PIT values.

    These values reuse observations to estimate the latent state and should be
    treated as a posterior-predictive check, not as forecast calibration.
    Prefer ``bx.leave_future_out(...).pit_diagnostics()`` for validation.
    """
    if getattr(fit, "is_multiseries_model", False):
        eta = fit.eta_draws(original_scale=False)
        output = np.full((fit.n_time, len(fit.channel_names)), np.nan)
        for index, channel in enumerate(fit.model.channels):
            sigma = fit.sigma_draws(channel=channel.name)
            xi = (
                fit.parameter(f"xi.{channel.name}")[:, None]
                if channel.family == "gev"
                else None
            )
            cdf = channel.observation.cdf(
                fit.y[None, :, index],
                eta[:, :, index],
                sigma=sigma,
                xi=xi,
            )
            if float(fit.transform_sign[index]) < 0.0:
                cdf = 1.0 - cdf
            output[:, index] = np.nanmean(cdf, axis=0)
        return output
    eta = fit.eta_draws(original_scale=False)
    sigma = fit.sigma_draws()
    xi = fit.parameter("xi")[:, None] if fit.family == "gev" else None
    cdf = fit.model.observation.cdf(fit.y[None, :], eta, sigma=sigma, xi=xi)
    if float(fit.transform_sign) < 0.0:
        cdf = 1.0 - cdf
    return np.nanmean(cdf, axis=0)


def fit_diagnostics(fit):
    acceptance = fit.sampler_diagnostics.get("acceptance", {})
    update_methods = fit.sampler_diagnostics.get("update_methods", {})
    rows = []
    scalar_draws = dict(fit.parameter_draws)
    for name, values in fit.parameter_draws.items():
        if values.ndim == 3 and name.startswith(("scale.seasonal", "initial.")):
            scalar_draws.update({f"{name}[{j+1:02d}]": values[..., j] for j in range(values.shape[-1])})
    for name, values in scalar_draws.items():
        if values.ndim != 2:
            continue
        constant = bool(
            values.size > 0 and np.all(values == values.reshape(-1)[0])
        )
        row = {
            "parameter": name,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "rhat": rhat(values),
            "ess_bulk": ess_bulk(values),
            "ess_tail": ess_tail(values),
            "acceptance": _finite_mean(acceptance[name]) if name in acceptance else np.nan,
            "update": update_methods.get(name, "derived_or_fixed"),
            "constant": constant,
            "diagnostic": (
                "constant draw; R-hat and ESS undefined"
                if constant
                else "sampled"
            ),
        }
        rows.append(row)
    try:
        import pandas as pd

        table = pd.DataFrame(rows).set_index("parameter")
    except ImportError:
        table = rows
    metrics = fit.sampler_diagnostics.get("draw_metrics", {})
    engine: dict[str, float] = {}
    if fit.plan.engine in {"laplace", "laplace_mh"}:
        engine = {
            "convergence_rate": _finite_mean(metrics.get("laplace_converged", [])),
            "median_iterations": _finite_median(metrics.get("laplace_iterations", [])),
            "median_relative_change": _finite_median(
                metrics.get("laplace_relative_change", [])
            ),
            "mean_support_rejections": _finite_mean(
                metrics.get("laplace_support_rejections", [])
            ),
        }
        support_repair_rate = _finite_mean(
            metrics.get(
                "laplace_initial_support_repaired", np.asarray([])
            )
        )
        if np.isfinite(support_repair_rate):
            engine["initial_support_repair_rate"] = support_repair_rate
        if fit.plan.engine == "laplace_mh":
            engine.update(
                # Retained compatibility name: this measures only the joint
                # Laplace-MH proposal, not the separate elliptical-slice steps.
                state_acceptance=_finite_mean(
                    metrics.get("laplace_mh_acceptance", np.asarray([]))
                ),
                mean_log_acceptance_ratio=_finite_mean(
                    metrics.get(
                        "laplace_mh_mean_log_acceptance_ratio", np.asarray([])
                    )
                ),
                mean_proposal_support_rejections=_finite_mean(
                    metrics.get(
                        "laplace_mh_support_rejections", np.asarray([])
                    )
                ),
            )
            key = "conditional_state_mh_acceptance" if fit.metadata.get("marginal_fs") else "joint_state_mh_acceptance"
            engine[key] = engine["state_acceptance"]
            if "state_ess_evaluations" in metrics:
                for label, metric in (
                    ("elliptical_slice_mean_likelihood_evaluations", "state_ess_evaluations"),
                    ("elliptical_slice_mean_blocks_moved", "state_ess_blocks_moved"),
                    ("elliptical_slice_mean_stochastic_blocks", "state_ess_stochastic_blocks"),
                    ("elliptical_slice_mean_absolute_angle", "state_ess_mean_absolute_angle"),
                ):
                    engine[label] = _finite_mean(metrics.get(metric, []))
    for name in ('coefficient_slice_evaluations', 'coefficient_reference_converged',
                 'coefficient_reference_iterations', 'coefficient_reference_fallback',
                 'coefficient_reference_support_repaired', 'coefficient_reference_regularized'):
        if name in metrics:
            engine[name] = _finite_mean(metrics[name])
    warnings = list(fit.plan.warnings)
    if fit.n_chains < 2:
        warnings.append("A single chain cannot assess between-chain convergence; use multiple chains for scientific inference.")
    if hasattr(table, "columns"):
        if "rhat" in table and np.any(table["rhat"].to_numpy() > 1.01):
            warnings.append("Some parameter R-hat values exceed 1.01; inspect traces and scientific-target diagnostics.")
        if "ess_bulk" in table and np.any(table["ess_bulk"].to_numpy() < 400):
            warnings.append("Some bulk effective sample sizes are below 400; quantify Monte Carlo error before reporting results.")
    return {
        "parameters": table,
        "engine": engine,
        "pit": posterior_pit(fit),
        "warnings": warnings,
    }
