"""Generate manuscript Figures 1 and 2 directly from the monthly observations."""
import argparse
from pathlib import Path

import bucex as bx
from .report import new_run


def run(config, *, output=None):
    data = bx.load_uccle_multiseries(**config["data"])
    exploration = bx.explore_monthly(data, periods=config["periods"], eras=config["eras"],
                                    min_count=config.get("min_count", 3))
    directory = new_run(output or config["output"], "exploration")
    bx.save_config(config, directory/"config.json")
    exploration.save(directory, **config.get("figures", {}))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("research/serra/config/revision/exploration.json"))
    parser.add_argument("--data-dir", type=Path, help="Folder containing the monthly date/series CSVs.")
    parser.add_argument("--series", nargs="+", choices=tuple(bx.UCCLE_INFO))
    parser.add_argument("--output", type=Path, help="Root for a new timestamped exploration directory.")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"))
    args = parser.parse_args()
    config = bx.load_config(args.config)
    if args.data_dir:
        config["data"]["data_dir"] = str(args.data_dir.resolve())
    if args.series:
        config["data"]["series"] = args.series
    if args.formats:
        config.setdefault("figures", {})["formats"] = args.formats
    if args.output:
        config["output"] = str(args.output)
    directory = run(config)
    print(f"Exploration directory: {directory.resolve()}")
    print("Figures 1 and 2 and their numerical tables are ready; no posterior fitting was required.")


if __name__ == "__main__":
    main()
