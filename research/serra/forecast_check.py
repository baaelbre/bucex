"""Latent versus observed forecast uncertainty and annual aggregation checks."""
import argparse
from pathlib import Path
import pandas as pd
import bucex as bx
from research.serra.experiment import fit_case, save_case
from research.serra.report import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit", type=Path, help="Inspect an existing univariate fit instead of refitting.")
    args = parser.parse_args()
    config = bx.load_config(args.config)
    directory = new_run(config["output"], "forecast_uncertainty")
    bx.save_config(config, directory / "config.json")
    if args.fit:
        analyses = [(config["data"]["series"][0], bx.FitResult.load(args.fit), None)]
    else:
        data = bx.load_uccle_multiseries(**config["data"])
        analyses = ((name, *fit_case(data, name, config, {})) for name in data)
    for name, fit, prior in analyses:
        target = directory / name
        target.mkdir()
        if prior is not None:
            save_case(fit, prior, target, config, threshold=config["risks"][name])
        forecast = fit.forecast(config["forecast_horizon"], seed=config["seed"])
        uncertainty = bx.forecast_uncertainty(forecast)
        uncertainty.to_csv(target / "forecast_uncertainty.csv", index=False)
        bx.annual_aggregation_check(forecast, config["risks"][name]).to_csv(target / "annual_aggregation.csv", index=False)
        if config.get("figures", True):
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(8, 3))
            for component in ("level", "location", "observation"):
                rows = uncertainty[(uncertainty.target == component) & (uncertainty.nominal == .90)]
                axis.plot(rows.horizon, rows.interval_width, label=component)
            axis.set(xlabel="forecast horizon / months", ylabel="90% interval width / °C")
            axis.tick_params(labelsize=11); axis.xaxis.label.set_size(12); axis.yaxis.label.set_size(12)
            axis.legend(); figure.tight_layout(); figure.savefig(target / "forecast_widths.pdf"); plt.close(figure)
    print(directory)


if __name__ == "__main__":
    main()
