from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import bucex as bx


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return bx.load_config(ROOT / relative)


def test_primary_mean_temperature_configs_are_parallel_and_readable():
    txm = _load("examples/config/uccle_gaussian/01_txm.json")
    tnm = _load("examples/config/uccle_gaussian/02_tnm.json")

    for config, series in ((txm, "TXm"), (tnm, "TNm")):
        assert config["data"]["series"] == [series]
        assert config["model"]["observation_family"] == "gaussian"
        assert config["model"]["time_varying_parameter"] == "mean"
        assert config["inference"]["engine"] == "ffbs"
        assert config["priors"]["innovation_slab_sd"] == {
            "level": 0.02,
            "trend": 0.00005,
            "season": 0.02,
        }
        assert config["mcmc"]["draws"] == 1000
        assert config["mcmc"]["warmup"] == 1000
        assert config["mcmc"]["chains"] == 4

    comparable_txm = deepcopy(txm)
    comparable_tnm = deepcopy(tnm)
    for config in (comparable_txm, comparable_tnm):
        config.pop("_comment", None)
        config["data"].pop("_comment", None)
        config["mcmc"].pop("seed")
    comparable_txm["data"]["series"] = ["mean"]
    comparable_tnm["data"]["series"] = ["mean"]
    assert comparable_txm == comparable_tnm


def test_phi_sensitivity_uses_the_primary_location_slabs():
    paths = sorted((ROOT / "examples/config/phi/uccle").glob("*.json"))
    assert len(paths) == 16
    for path in paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["priors"]["innovation_slab_sd"] == {
            "level": 0.02,
            "trend": 0.00005,
            "season": 0.02,
        }


def test_txm_and_tnm_are_bundled_gaussian_series():
    assert bx.UCCLE_INFO["TXm"]["family"] == "gaussian"
    assert bx.UCCLE_INFO["TNm"]["family"] == "gaussian"
    for name in ("TXm", "TNm"):
        values = bx.load_uccle_series(name, start="2020-01-01", end="2020-12-31")
        assert len(values) == 12
        assert values.notna().all()
