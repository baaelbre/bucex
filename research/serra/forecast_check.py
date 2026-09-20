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
        fit = bx.load_fit(args.fit)
        if fit.is_multiseries_model:
            parser.error('--fit currently requires a univariate archive; use report for joint forecasts.')
        name = fit.series_name or config['data']['series'][0]
        if name not in config['risks']:
            parser.error('No risk threshold is configured for the saved fit series.')
        analyses = [(name, fit, None)]
    else:
        data = bx.load_uccle_multiseries(**config["data"])
        analyses = ((name, *fit_case(data, name, config, {})) for name in data)
    for name, fit, prior in analyses:
        target = directory / name
        target.mkdir()
        if prior is not None:
            save_case(fit, prior, target, config, threshold=config["risks"][name])
        level = config.get('credible_interval', .95)
        forecast = fit.forecast(config["forecast_horizon"], draws=config.get('forecast_draws'), seed=config["seed"])
        uncertainty = bx.forecast_uncertainty(forecast, levels=tuple(sorted({.90, .95, .99, level})))
        uncertainty.to_csv(target / "forecast_uncertainty.csv", index=False)
        annual = forecast.aggregate()  # day-weighted means; maxima/minima of extreme blocks
        annual.summary(level=level).to_csv(target / "annual_forecast.csv", index=False)
        if fit.family == "gev":
            bx.annual_aggregation_check(forecast, config["risks"][name]).to_csv(target / "annual_aggregation.csv", index=False)
        else:
            # This threshold concerns the annual mean, not at least one hot month.
            annual.risk_summary(config["risks"][name], level=level).to_csv(target / "annual_mean_risk.csv", index=False)
        if config.get("figures", True):
            import matplotlib.pyplot as plt
            with bx.publication_style(style=config.get('figure_style','manuscript')):
                figure, axis = plt.subplots(figsize=(8, 3))
                for component in ("level", "location", "observation"):
                    rows = uncertainty[(uncertainty.target == component) & (uncertainty.nominal == level)]
                    axis.plot(rows.horizon, rows.interval_width, label=component)
                axis.set(xlabel="Forecast horizon / months", ylabel=f"{level:.0%} interval width / °C")
                axis.legend(); figure.tight_layout()
                bx.save_figure(figure,target/'forecast_widths',formats=(config.get('figure_format','png'),),
                               dpi=config.get('figure_dpi',180),close=True)
    print(directory)


if __name__ == "__main__":
    main()
