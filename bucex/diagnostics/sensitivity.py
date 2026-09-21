"""Descriptive prior updating and paired predictive comparisons.

No quantity here measures an optimal amount of 'learning', selects a prior,
or treats a positive continuous-SD interval as evidence for a nonzero process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comparison import paired_block_comparison


def innovation_prior_diagnostics(comparison):
    """Compare interval widths and medians in compare_innovation_priors output.

    Optional ``variant`` and ``channel`` labels are preserved. A width ratio
    below one describes contraction, not proof of identification or adequacy.
    A shift can coexist with strong prior sensitivity; there is no target value.
    """
    table = pd.DataFrame(comparison).copy()
    keys = [k for k in ("variant", "channel", "component", "scale") if k in table]
    required = {"component", "scale", "distribution", "lower", "median", "upper", "credible_interval"}
    if not required <= set(table) or table.empty:
        raise ValueError("Supply a nonempty compare_innovation_priors table.")
    if set(table.distribution) != {"prior", "posterior"}:
        raise ValueError("Both prior and posterior rows are required.")
    left = table[table.distribution == "prior"]
    right = table[table.distribution == "posterior"]
    paired = left.merge(right, on=keys, suffixes=("_prior", "_posterior"),
                        validate="one_to_one", how="outer", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Prior and posterior components must match exactly.")
    if not np.array_equal(paired.credible_interval_prior, paired.credible_interval_posterior):
        raise ValueError("Prior and posterior intervals must use the same probability.")
    result = paired[keys].copy()
    for distribution in ("prior", "posterior"):
        for stat in ("lower", "median", "upper"):
            result[f"{distribution}_{stat}"] = paired[f"{stat}_{distribution}"]
    prior_width = paired.upper_prior - paired.lower_prior
    posterior_width = paired.upper_posterior - paired.lower_posterior
    result["posterior_to_prior_width"] = posterior_width / prior_width.where(prior_width > 0)
    result["posterior_to_prior_median"] = paired.median_posterior / paired.median_prior.where(paired.median_prior != 0)
    result["median_shift_in_prior_widths"] = (paired.median_posterior-paired.median_prior) / prior_width.where(prior_width > 0)
    result["credible_interval"] = paired.credible_interval_prior
    result["interpretation"] = "Descriptive updating only; assess predictive calibration and sensitivity."
    return result


def compare_predictive_scores(scores, *, baseline, block="origin", min_blocks=6,
                             draws=2000, level=.95, seed=None):
    """Compare named candidates on exactly matching held-out forecast cases.

    Supply the tidy output of ``Forecast.score(aggregate=False)`` with
    ``variant, channel, origin, time, horizon`` labels. All BUCEX scores are
    losses: positive improvement = baseline loss minus candidate loss.
    Whole forecast origins are the default resampling units. With fewer than
    ``min_blocks`` blocks, only descriptive differences are reported. Even with
    more blocks, bootstrap intervals assume approximately independent blocks;
    they are neither posterior intervals nor a correction for prior tuning.
    """
    table = pd.DataFrame(scores).copy()
    keys = ["channel", "origin", "time", "horizon", "score", "setting"]
    if not set(keys + ["variant", "value", block]) <= set(table) or table.empty:
        raise ValueError("Scores require variant, channel, origin, time, horizon, score, setting and value.")
    if baseline not in set(table.variant):
        raise ValueError("The declared baseline is absent from the score table.")
    if min_blocks < 2:
        raise ValueError("min_blocks must be at least two.")
    table["time"] = pd.to_datetime(table.time)
    table["horizon"] = pd.to_numeric(table.horizon, errors="raise")
    if table.duplicated(["variant", *keys]).any():
        raise ValueError("Duplicate forecast cases; supply one run per candidate.")
    if table.horizon.isna().any() or (table.horizon < 1).any():
        raise ValueError("Forecast horizons must be positive.")
    left = table[table.variant == baseline]
    rows = []
    for candidate in table.variant.drop_duplicates():
        if candidate == baseline:
            continue
        right = table[table.variant == candidate]
        paired = left.merge(right, on=keys, how="outer", suffixes=("_baseline", "_candidate"),
                            indicator=True, validate="one_to_one")
        if not paired._merge.eq("both").all():
            raise ValueError(f"{candidate}: candidates must contain exactly the same forecast cases.")
        block_key = block if block in keys else block + "_baseline"
        if block not in keys and not np.array_equal(paired[block_key], paired[block+"_candidate"]):
            raise ValueError("Baseline and candidate block labels differ.")
        selections = {"all": np.ones(len(paired), dtype=bool),
                      "horizons_1_12": paired.horizon <= 12,
                      "horizons_13_plus": paired.horizon > 12}
        for band, mask in selections.items():
            for labels, group in paired[mask].groupby(["channel", "score", "setting"], dropna=False, sort=False):
                a, b = group.value_baseline.to_numpy(float), group.value_candidate.to_numpy(float)
                finite = np.isfinite(a) & np.isfinite(b)
                blocks = group[block_key].to_numpy()
                result = dict(improvement=np.nan, lower=np.nan, upper=np.nan,
                              n_cases=len(group), n_blocks=len(pd.unique(blocks)), level=level)
                if not finite.all():
                    result["status"] = "nonfinite scores: investigate predictive support; no cases dropped"
                else:
                    result["improvement"] = float(np.mean(a-b))
                    if result["n_blocks"] < min_blocks:
                        result["status"] = "few forecast blocks: descriptive comparison only"
                    else:
                        result = paired_block_comparison(a, b, blocks, draws=draws, level=level, seed=seed)
                rows.append(dict(zip(("channel", "score", "setting"), labels)) | result |
                            dict(baseline=baseline, candidate=candidate, horizon_band=band,
                                 baseline_mean=float(np.mean(a)), candidate_mean=float(np.mean(b)),
                                 nonfinite_pairs=int((~finite).sum()), block=block))
    return pd.DataFrame(rows)


__all__ = ["innovation_prior_diagnostics", "compare_predictive_scores"]
