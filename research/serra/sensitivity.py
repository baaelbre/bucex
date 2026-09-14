"""Prior and shape-bound sensitivity, retaining a six-series univariate route."""
import argparse
import copy
from pathlib import Path
import pandas as pd
import bucex as bx
from research.serra.experiment import fit_case, save_case
from research.serra.report import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--series", nargs="+")
    args = parser.parse_args()
    config = bx.load_config(args.config)
    if args.series:
        config["data"]["series"] = args.series
    data = bx.load_uccle_multiseries(**config["data"])
    directory = new_run(config["output"], "sensitivity")
    bx.save_config(config, directory / "config.json")
    results, status = [], []
    for variant in config["variants"]:
        for name in data:
            local = copy.deepcopy(config)
            local["variant"] = variant
            label = variant["name"]
            if bx.UCCLE_INFO[name]["family"] == "gaussian" and set(variant) <= {"name", "xi_bounds"} and "xi_bounds" in variant:
                status.append(dict(variant=label, series=name, status="not_applicable", reason="Gaussian observations have no GEV shape."))
                continue
            print(f"{label}: {name}", flush=True)
            try:
                fit, prior = fit_case(data, name, local, variant)
                target = save_case(fit, prior, directory / label / name, local,
                                  threshold=config["risks"][name])
                results.append(target.reset_index().assign(variant=label, series=name))
                status.append(dict(variant=label, series=name, status="completed"))
            except (ValueError, RuntimeError, FloatingPointError) as error:
                status.append(dict(variant=label, series=name, status="failed", error=str(error)))
                if not config.get("continue_on_error", False):
                    pd.DataFrame(status).to_csv(directory / "status.csv", index=False)
                    raise
            pd.DataFrame(status).to_csv(directory / "status.csv", index=False)
            if results:
                pd.concat(results).to_csv(directory / "sensitivity.csv", index=False)
    print(directory)


if __name__ == "__main__":
    main()
