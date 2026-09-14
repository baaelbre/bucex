"""Scientific contrasts retain chains and flag unassessable quantities."""
from types import SimpleNamespace

import numpy as np
import pytest

import bucex as bx


def test_contrast_table_uses_original_chains_for_convergence():
    rng = np.random.default_rng(14)
    stable = rng.normal(size=(4, 512))
    separated = stable + np.arange(4)[:, None] * 3.0
    table = bx.summarize_draws({"warming": stable, "poorly_mixed_departure": separated})
    assert table.loc["warming", "rhat"] == bx.rhat(stable)
    assert table.loc["warming", "ess_bulk"] == bx.ess_bulk(stable)
    assert table.loc["warming", "rhat"] < 1.02
    assert table.loc["poorly_mixed_departure", "rhat"] > 1.1
    assert table.loc["poorly_mixed_departure", "ess_bulk"] < table.loc["warming", "ess_bulk"]
    assert table.loc["warming", "lower"] == pytest.approx(np.quantile(stable, 0.05))


def test_constant_and_infinite_endpoints_do_not_look_converged():
    values = np.ones((2, 8))
    endpoints = values.copy()
    endpoints[0] = np.inf
    table = bx.summarize_draws({"fixed": values, "endpoint": endpoints})
    assert np.isnan(table.loc["fixed", "rhat"])
    assert np.isnan(table.loc["fixed", "ess_bulk"])
    assert table.loc["fixed", "constant"]
    assert table.loc["endpoint", "finite_fraction"] == 0.5
    assert np.isnan(table.loc["endpoint", "median"])
    assert "nonfinite" in table.loc["endpoint", "diagnostic"]


def test_contrast_validation_rejects_flattened_and_mismatched_draws():
    with pytest.raises(ValueError, match="chains, draws"):
        bx.summarize_draws({"warming": np.arange(16)})
    with pytest.raises(ValueError, match="preserve"):
        bx.FitResult.contrast_diagnostics(SimpleNamespace(n_chains=2, draws_per_chain=8),
                                         {"warming": np.ones((1, 16))})
    with pytest.raises(ValueError, match="credible_interval"):
        bx.summarize_draws({"warming": np.ones((2, 8))}, credible_interval=1.0)
