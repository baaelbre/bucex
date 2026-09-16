"""Fit any Uccle summary independently, or an optional joint model."""
import argparse
from copy import deepcopy
from pathlib import Path
import bucex as bx
from research.serra.report import new_run, write_report
from research.serra.models import channel, marginal_prior, joint_model, fit_options


def selected_config(config, *, series=None, phi=None, copula=False):
    """CLI overrides are explicit and saved alongside each result."""
    config = deepcopy(config)
    if series:
        config["data"]["series"] = list(series)
    if phi is not None:
        config["model"]["scale_mode"] = "constant" if phi == "stationary" else phi
    if copula:
        modes = {"joint": "copula"}
        if config["analysis"] not in modes:
            raise ValueError("--copula requires a joint configuration.")
        config["analysis"] = modes[config["analysis"]]
    return config


def run(config, *, series=None):
    config = selected_config(config, series=series)
    data = bx.load_uccle_multiseries(**config["data"])
    directory = new_run(config["output"], f"uccle_{config['analysis']}")
    report = dict(config=config, risks=config["risks"], horizon=config["forecast_horizon"],
                  level=config["credible_interval"])
    if config["analysis"] == "independent":
        for name in data:
            item = channel(name, data, config)
            model = bx.Model(item.observation, item.components)
            fit = bx.fit(data[name], model=model, priors=marginal_prior(item, data, config),
                         **fit_options(config, family=item.family, tail=item.tail))
            write_report(fit, directory / name, **{**report, "risks": {"series": config["risks"][name]}})
            del fit  # Six marginal jobs need not retain six posterior state arrays.
    else:
        model, prior = joint_model(data, config)
        fit = bx.fit(data, model=model, priors=prior, **fit_options(config, family=model.family))
        write_report(fit, directory, **report)
    return directory


def arguments(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--series", nargs="+", choices=tuple(bx.UCCLE_INFO))
    parser.add_argument("--phi", choices=("stationary", "linear", "rw"))
    parser.add_argument("--copula", action="store_true", help="Add residual Gaussian dependence to a private joint config.")
    args = parser.parse_args()
    return selected_config(bx.load_config(args.config), series=args.series, phi=args.phi, copula=args.copula)


if __name__ == "__main__":
    print(run(arguments()))
