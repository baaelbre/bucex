"""Small, chain-preserving exports for regenerating diagnostic figures."""
from __future__ import annotations

import numpy as np
import pandas as pd


def parameter_trace_draws(fit, *, channel=None):
    """Physical innovation magnitudes, initial level/slope, scale and shape.

    FS sign symmetries are not mixed with physical-SD diagnostics. Univariate
    aliases are omitted. Arrays retain (chain, draw) dimensions without thinning.
    """
    if fit.is_multiseries_model:
        if channel not in fit.channel_names:
            raise ValueError("Select a fitted channel for parameter traces.")
        prefix, suffix = f"channel.{channel}.", f".{channel}"
    else:
        prefix, suffix = "", ""
    names = [f"initial.{prefix}level", f"initial.{prefix}slope"]
    names += [f"sd.{prefix}{c}" for c in ("level", "slope", "seasonal")]
    names += [key+suffix for key in ("sigma", "xi", "scale_slope", "scale_rw_sd")]
    result = {name: fit.parameter(name, combine_chains=False)
              for name in names if name in fit.parameter_draws
              and np.asarray(fit.parameter_draws[name]).ndim == 2}
    key = "scale.seasonal" + suffix
    if key in fit.parameter_draws:
        values = fit.parameter(key, combine_chains=False)
        if values.ndim == 3:
            result.update({f"{key}[{j+1:02d}]": values[..., j] for j in range(values.shape[-1])})
    return result


def trace_frame(draws):
    """Wide table with explicit one-based chain and retained-draw indices."""
    if not draws:
        raise ValueError("Provide at least one trace.")
    shapes = {np.asarray(v).shape for v in draws.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("All traces must share a (chain, retained draw) shape.")
    chains, n = next(iter(shapes))
    if not chains or not n:
        raise ValueError("Trace arrays must be non-empty.")
    return pd.DataFrame({"chain": np.repeat(np.arange(1, chains+1), n),
                         "draw": np.tile(np.arange(1, n+1), chains),
                         **{key: np.asarray(value).ravel() for key, value in draws.items()}})


def traces_from_frame(frame):
    """Reconstruct aligned chains, rejecting duplicates or missing iterations."""
    if not {"chain", "draw"} <= set(frame) or frame.empty:
        raise ValueError("Trace tables require chain and draw columns and non-empty rows.")
    if frame.duplicated(["chain", "draw"]).any():
        raise ValueError("Duplicate chain/draw indices in trace table.")
    ordered = frame.sort_values(["chain", "draw"])
    groups = list(ordered.groupby("chain", sort=True))
    index = groups[0][1].draw.to_numpy()
    if any(not np.array_equal(g.draw.to_numpy(), index) for _, g in groups):
        raise ValueError("Chains must contain identical retained-draw indices.")
    if not np.array_equal(index, np.arange(index[0], index[0]+len(index))):
        raise ValueError("Trace iterations must be consecutive; ACF needs their actual spacing.")
    keys = [key for key in frame if key not in {"chain", "draw"}]
    if not keys:
        raise ValueError("No parameter columns in trace table.")
    return {key: np.stack([g[key].to_numpy(dtype=float) for _, g in groups]) for key in keys}


__all__ = ["parameter_trace_draws", "trace_frame", "traces_from_frame"]
