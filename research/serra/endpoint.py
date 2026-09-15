"""Paired Laplace/Laplace-MH checks around the actual July 2019 Uccle record."""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.serra.experiment import fit_case, save_case
from research.serra.report import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = bx.load_config(parser.parse_args().config)
    data = bx.load_uccle_multiseries(**config["data"])
    event = config["event"]
    indices = np.flatnonzero(pd.to_datetime(data.index).to_period("M") == pd.Period(event["date"], "M"))
    if indices.size != 1:
        raise ValueError("The configured data window must contain the event month exactly once.")
    index = int(indices[0])
    if not np.isclose(data[event["series"]].iloc[index], event["value"], atol=.051):
        raise ValueError("The observed event value does not match the stated record; inspect the data.")
    directory = new_run(config["output"], "endpoint_2019")
    bx.save_config(config, directory / "config.json")
    results = []
    for variant in config["variants"]:
        for engine in config["engines"]:
            print(f"{variant['name']}: {engine}", flush=True)
            local = copy.deepcopy(config)
            local["variant"] = variant
            fit, prior = fit_case(data, event["series"], local, variant, engine=engine)
            target = directory / variant["name"] / engine
            summary = save_case(fit, prior, target, local, threshold=event["value"], event_index=index)
            results.append(summary.reset_index().assign(variant=variant["name"], engine=engine))
            # Quantities for every neighboring month reveal propagated approximation differences.
            window = np.arange(max(0, index-12), min(fit.n_time, index+13))
            location = fit.eta_draws(original_scale=True)[:, window]
            quantiles = np.quantile(location, [.05, .5, .95], axis=0)
            pd.DataFrame(dict(time=fit.time[window], lower=quantiles[0], median=quantiles[1], upper=quantiles[2])).to_csv(target / "event_window.csv", index=False)
            pd.concat(results).to_csv(directory / "endpoint_comparison.csv", index=False)
            # Train strictly before July 2019; no event information enters this fit.
            prefit, _ = fit_case(data.iloc[:index], event["series"], local, variant, engine=engine)
            prediction = prefit.forecast(1, dates=data.index[index:index+1], seed=config["seed"])
            probability = 1-prediction.conditional_cdf(np.asarray([event["value"]]))[:,0]
            pd.DataFrame({"mean_probability": [probability.mean()],
                          "lower_probability": [np.quantile(probability,.05)],
                          "upper_probability": [np.quantile(probability,.95)],
                          "log_score": prediction.log_score(np.asarray([event["value"]]))}).to_csv(
                              target / "pre_event_forecast.csv", index=False)
            prefit.diagnostics()["parameters"].to_csv(target / "pre_event_mcmc.csv")
    (directory / "interpretation.txt").write_text(
        "Full-record tables describe smoothing with the event included. pre_event_forecast.csv uses only earlier months; this selected-event case is not an overall calibration test.\n"
        "Compare identical-prior engines only after satisfactory mixing. Positive-shape draws have infinite upper endpoints.\n")
    print(directory)


if __name__ == "__main__":
    main()
