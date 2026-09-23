"""Paired approximate/exact GEV recovery over a declared generating shape grid."""
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
    directory = new_run(config["output"], "shape_recovery")
    bx.save_config(config, directory / "config.json")
    truth_config = config["simulation"]
    model = bx.Model(bx.GEV(xi_bounds=config["priors"].get("xi_bounds")),
        (bx.LocalLinearTrend(), bx.DummySeasonal(period=config["model"]["period"])))
    metrics, status = [], []
    for replicate in range(truth_config["replicates"]):
        # Same innovations/uniforms across xi within replicate; independent replicate seeds.
        seed = config["seed"] + 10000 * replicate
        for xi in truth_config["xi_grid"]:
            params = {**truth_config["parameters"], "xi": xi}
            simulation = bx.simulate(model, truth_config["n_time"], params,
                initial_state=truth_config["initial_state"], seed=seed)
            if truth_config.get("endpoint_probability") is not None and xi < 0:
                index = int(truth_config.get("event_index", -1))
                simulation.y[index] = model.observation.ppf(truth_config["endpoint_probability"],
                    simulation.eta[index], sigma=params["sigma"], xi=xi)
            data = pd.DataFrame({"TXx": simulation.y})
            label = f"replicate_{replicate:03d}_xi_{xi:+.2f}"
            for engine in config["engines"]:
                print(f"{label}: {engine}", flush=True)
                try:
                    local = copy.deepcopy(config)
                    local["mcmc"]["seed"] = config["mcmc"]["seed"] + 10000 * replicate
                    local["figures"] = config.get("case_figures", False)
                    local["generating_xi"], local["replicate"] = xi, replicate
                    fit, prior = fit_case(data, "TXx", local, {}, engine=engine)
                    target = directory / label / engine
                    save_case(fit, prior, target, local, threshold=config["risks"]["TXx"])
                    truth = pd.DataFrame({"time": np.arange(simulation.y.size), "y": simulation.y, "location": simulation.eta})
                    truth.to_csv(target / "truth.csv", index=False)
                    rows = bx.recovery_metrics(fit, simulation, config["risks"]["TXx"])
                    metrics.append(rows.assign(replicate=replicate, xi=xi, engine=engine,
                        sampling_regime="endpoint_stress" if truth_config.get("endpoint_probability") else "unconditional"))
                    status.append(dict(replicate=replicate, xi=xi, engine=engine, status="completed"))
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    status.append(dict(replicate=replicate, xi=xi, engine=engine, status="failed", error=str(error)))
                    if not config.get("continue_on_error", False):
                        pd.DataFrame(status).to_csv(directory / "status.csv", index=False)
                        raise
                pd.DataFrame(status).to_csv(directory / "status.csv", index=False)
                if metrics:
                    pd.concat(metrics).to_csv(directory / "recovery.csv", index=False)
    if metrics and config.get("figures", True):
        import matplotlib.pyplot as plt
        table = pd.concat(metrics)
        table = table[(table.nominal == .90) & (table.target == "location")]
        figure, axes = plt.subplots(1, 2, figsize=(9, 3))
        for engine, rows in table.groupby("engine"):
            grouped = rows.groupby("xi")
            for axis, metric in zip(axes, ("rmse", "pointwise_coverage")):
                average = grouped[metric].mean()
                axis.plot(average.index, average.values, marker="o", label=engine)
                axis.set_xlabel("generating shape ξ", fontsize=12)
                axis.set_ylabel(metric.replace("_", " "), fontsize=12)
                axis.tick_params(labelsize=11)
        axes[1].axhline(.90, color="grey", linestyle=":")
        axes[0].legend(); figure.tight_layout()
        figure.savefig(directory / "recovery.pdf"); plt.close(figure)
    print(directory)


if __name__ == "__main__":
    main()
