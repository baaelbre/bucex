"""Real pre-release fixtures preserve scientific results across schema changes."""
from pathlib import Path
import json
import zipfile

import numpy as np
import pytest

import bucex as bx


@pytest.mark.parametrize("family", ["gaussian", "gev"])
def test_real_v1_5_2_archive_loads_forecasts_and_resaves(tmp_path, family):
    archive = Path(__file__).with_name("fixtures") / f"v1_5_2_{family}.bucex"
    with zipfile.ZipFile(archive) as source:
        metadata = json.loads(source.read("metadata.json"))
    assert metadata["schema_version"] == "2.7.0"
    old = bx.FitResult.load(archive)
    assert old.family == family
    assert old.n_time == 12
    assert np.all(np.isfinite(old.eta_draws()))
    before = old.forecast(4, draws=5, seed=16250)
    path = tmp_path / "migrated.bucex"
    old.save(path)
    migrated = bx.FitResult.load(path)
    np.testing.assert_array_equal(old.eta_draws(), migrated.eta_draws())
    after = migrated.forecast(4, draws=5, seed=16250)
    np.testing.assert_array_equal(before.eta, after.eta)
    np.testing.assert_array_equal(before.observations, after.observations)
