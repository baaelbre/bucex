"""Posterior summaries and convergence diagnostics for scientific contrasts."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .posterior import ess_bulk, ess_tail, rhat


def summarize_draws(draws: Mapping[str, np.ndarray], *, credible_interval: float = 0.90):
    """Summarize named scalar posterior quantities while preserving chains.

    Every value must have shape ``(chains, draws)``. For example, derive a
    warming change from ``fit.shared_draws(combine_chains=False)`` by taking
    the difference between its final and initial time columns. The returned
    table includes pointwise posterior intervals, rank-normalized split R-hat
    and bulk ESS. These diagnostics describe the supplied scientific quantity,
    whose mixing can differ from the model's static parameters.

    Constant or nonfinite quantities receive undefined convergence diagnostics
    and an explicit status. Infinite GEV endpoints are not silently discarded.
    No Monte Carlo standard error is inferred from rank-normalized bulk ESS.
    """
    if not isinstance(draws, Mapping) or not draws:
        raise ValueError("draws must be a nonempty mapping of names to (chains, draws) arrays.")
    if not 0.0 < float(credible_interval) < 1.0:
        raise ValueError("credible_interval must lie in (0, 1).")
    alpha = (1.0 - float(credible_interval)) / 2.0
    rows = []
    for name, draw in draws.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every quantity must have a nonempty string name.")
        values = np.asarray(draw, dtype=float)
        if values.ndim != 2 or not all(values.shape):
            raise ValueError(f"{name!r} must have shape (chains, draws); keep chains separate when deriving the quantity.")
        finite = bool(np.all(np.isfinite(values)))
        constant = bool(finite and np.all(values == values.flat[0]))
        row = {"quantity": name, "chains": values.shape[0], "draws_per_chain": values.shape[1],
               "finite_fraction": float(np.mean(np.isfinite(values))), "constant": constant}
        if finite:
            lower, median, upper = np.quantile(values, [alpha, 0.5, 1.0 - alpha])
            row.update(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                       lower=float(lower), median=float(median), upper=float(upper),
                       rhat=rhat(values), ess_bulk=ess_bulk(values), ess_tail=ess_tail(values))
            row["diagnostic"] = ("constant draw; R-hat and ESS undefined" if constant else
                                 "insufficient draws for split R-hat" if values.shape[1] < 4 else
                                 "single original chain; use independently initialized chains" if values.shape[0] < 2 else "sampled")
        else:
            row.update({key: np.nan for key in ("mean", "sd", "lower", "median", "upper", "rhat", "ess_bulk", "ess_tail")})
            row["diagnostic"] = "nonfinite draws; summarize finite-endpoint probability separately"
        rows.append(row)
    import pandas as pd

    return pd.DataFrame(rows).set_index("quantity")


__all__ = ["summarize_draws"]
