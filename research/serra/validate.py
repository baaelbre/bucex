"""Expanding-window forecasts with calibration at 90%, 95% and 99%."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.serra.run import arguments
from research.serra.report import convergence_parameters, new_run, scientific_targets
from research.serra.models import channel, marginal_prior, joint_model, fit_options


def calibration(forecast, observed, *, channel=None):
    """Keep every held-out interval and quantile event, not only averages."""
    index = forecast.channel_names.index(channel) if forecast.is_multiseries_forecast else None
    samples = forecast.observations[:, :, index] if index is not None else forecast.observations
    observed = np.asarray(observed)
    base = dict(time=forecast.dates, horizon=np.arange(1, forecast.horizon + 1), observed=observed)
    rows = []
    for level in (0.90, 0.95, 0.99):
        lower, upper = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
        rows.append(pd.DataFrame({**base, "kind": "central_interval", "nominal": level,
            "lower": lower, "upper": upper, "width": upper - lower,
            "covered": (observed >= lower) & (observed <= upper)}))
    for probability in (0.005, 0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99, 0.995):
        quantile = np.quantile(samples, probability, axis=0)
        rows.append(pd.DataFrame({**base, "kind": "cdf_quantile", "nominal": probability,
            "quantile": quantile, "covered": observed <= quantile}))
    return pd.concat(rows, ignore_index=True)


def validate(config, *, directory=None):
    data = bx.load_uccle_multiseries(**config["data"])
    settings = config["validation"]
    initial = data.iloc[:settings["initial"]]
    directory = new_run(config["output"], f"validation_{config['analysis']}") if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bx.save_config(config, directory / "config.json")
    analyses = []
    # Freeze all declared priors before the first held-out fold.
    if config["analysis"] == "independent":
        for name in data:
            item = channel(name, initial, config)
            analyses.append((name, data[[name]], bx.Model(item.observation, item.components),
                             marginal_prior(item, initial, config), item.tail))
    else:
        model, prior = joint_model(initial, config)
        analyses.append(("joint", data, model, prior, None))
    splits = tuple(bx.rolling_origin_splits(len(data), **{k: settings[k] for k in ("initial", "horizon", "step")}))
    if not splits:
        raise ValueError("No complete held-out fold fits inside the supplied record.")
    for label, values, model, prior, tail in analyses:
        target = directory / label
        target.mkdir(exist_ok=True)
        scores, pits, coverages, joint_scores, compound, predictions, folds = [], [], [], [], [], [], []
        for fold, (train, test) in enumerate(splits):
            print(f"{label}: fold {fold + 1}/{len(splits)}, training {train.stop} blocks", flush=True)
            training = values.iloc[:train.stop]
            if label != "joint":
                training = training.iloc[:, 0]
            fit = bx.fit(training, model, priors=prior,
                         **fit_options(config, family=model.family, tail=tail))
            forecast = fit.forecast(len(test), dates=values.index[list(test)],
                draws=settings.get("draws", config.get("forecast_draws")), seed=config["seed"] + fold)
            diagnostics = fit.diagnostics()["parameters"].assign(chains=fit.n_chains)
            diagnostics.to_csv(target / f"mcmc_{train.stop}.csv")
            targets = fit.contrast_diagnostics(scientific_targets(fit))
            targets.to_csv(target / f"targets_{train.stop}.csv")
            assessment = bx.convergence_assessment({'parameters': convergence_parameters(diagnostics, fit.n_chains), 'scientific_targets': targets},
                **config.get('diagnostic_thresholds', {}))
            bx.save_config(assessment, target / f"convergence_{train.stop}.json")
            bx.save_config(fit.sampler_diagnostics.get('execution', {}), target / f"execution_{train.stop}.json")
            folds.append(dict(origin=train.stop, training_start=str(training.index[0]),
                training_end=str(training.index[-1]), forecast_start=str(values.index[test.start]),
                forecast_end=str(values.index[test.stop-1]), horizon=len(test),
                numerical_status=assessment['status']))
            pd.DataFrame(folds).to_csv(target/'folds.csv', index=False)
            if settings.get("save_fits", False):
                fit.save(target / f"fit_{train.stop}.bucex")
            for name in values:
                observed = values[name].iloc[list(test)].to_numpy()
                kwargs = {"channel": name} if label == "joint" else {}
                score = forecast.score(observed, thresholds=[config["risks"][name]],
                                       quantiles=(0.90, 0.95, 0.99), aggregate=False, **kwargs)
                scores.append(score.assign(origin=train.stop, channel=name,
                    horizon=score['time_index'].to_numpy(dtype=int)+1,
                    time=forecast.dates[score["time_index"].to_numpy(dtype=int)]))
                observations = forecast.observations
                if label == 'joint':
                    observations = observations[:, :, forecast.channel_names.index(name)]
                interval = config.get('credible_interval', .95)
                low, median, high = np.quantile(observations, [(1-interval)/2, .5, (1+interval)/2], axis=0)
                predictions.append(pd.DataFrame(dict(origin=train.stop, channel=name,
                    time=forecast.dates, horizon=np.arange(1,len(test)+1), observed=observed,
                    lower=low, median=median, upper=high, credible_interval=interval)))
                pits.append(pd.DataFrame({"origin": train.stop, "channel": name,
                    "time": forecast.dates, "horizon": np.arange(1, len(test) + 1),
                    "pit": forecast.pit(observed, **kwargs)}))
                coverages.append(calibration(forecast, observed, **kwargs).assign(origin=train.stop, channel=name))
            if label == "joint":
                observed = values.iloc[list(test)].to_numpy()
                joint_scores.append(pd.DataFrame({"origin": train.stop, "time": forecast.dates,
                    "score": forecast.joint_log_score(observed)}))
                if {'TXx', 'TNx'} <= set(values):
                    events = {'TXx': ('>', config['risks']['TXx']), 'TNx': ('>', config['risks']['TNx'])}
                    probability = forecast.compound_probability_draws(events).mean(axis=0)
                    event = ((values['TXx'].iloc[list(test)].to_numpy() > config['risks']['TXx']) &
                             (values['TNx'].iloc[list(test)].to_numpy() > config['risks']['TNx']))
                    compound.append(pd.DataFrame({'origin': train.stop, 'time': forecast.dates,
                        'probability': probability, 'observed_event': event,
                        'brier': (probability-event)**2}))
                constraints = [pair for pair in bx.UCCLE_ORDER_CONSTRAINTS if set(pair) <= set(values)]
                if constraints:
                    forecast.ordering_diagnostics(constraints, observed=observed).by_time.to_csv(
                        target / f"ordering_{train.stop}.csv", index=False)
            # Save paired case scores at each origin, not just after a long grid.
            pd.concat(scores).to_csv(target / "scores.csv", index=False)
            pd.concat(pits).to_csv(target / "held_out_pit.csv", index=False)
            pd.concat(coverages).to_csv(target / "coverage_by_case.csv", index=False)
            pd.concat(predictions).to_csv(target/'predictions.csv', index=False)
            if joint_scores:
                pd.concat(joint_scores).to_csv(target / "joint_log_scores.csv", index=False)
            if compound:
                pd.concat(compound).to_csv(target / "compound_heat_scores.csv", index=False)
            del fit, forecast  # Retain compact scores, not every full state posterior.
            gc.collect()
        score_table, coverage_table = pd.concat(scores), pd.concat(coverages)
        score_table.to_csv(target / "scores.csv", index=False)
        pd.concat(pits).to_csv(target / "held_out_pit.csv", index=False)
        coverage_table.to_csv(target / "coverage_by_case.csv", index=False)
        bx.coverage_by_month(coverage_table).to_csv(target / 'coverage_by_month.csv',index=False)
        monthly_pits = []
        for name, group in pd.concat(pits).groupby('channel'):
            monthly_pits.append(bx.pit_by_month(group.pit,group.time).assign(channel=name))
        pd.concat(monthly_pits,ignore_index=True).to_csv(target/'held_out_pit_by_month.csv',index=False)
        score_table.groupby(["channel", "score", "setting"], dropna=False)["value"].agg(
            mean="mean", n="size").to_csv(target / "score_summary.csv")
        coverage_table.groupby(["channel", "kind", "nominal"])["covered"].agg(
            empirical="mean", n="size").to_csv(target / "coverage_summary.csv")
        if compound:
            pd.concat(compound).to_csv(target / 'compound_heat_scores.csv', index=False)
        if joint_scores:
            pd.concat(joint_scores).to_csv(target / "joint_log_scores.csv", index=False)
    return directory


if __name__ == "__main__":
    print(validate(arguments(__doc__)))
