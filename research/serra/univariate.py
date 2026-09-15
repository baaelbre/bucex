"""Fit any of the six private FS/SSVS temperature trajectories."""
import argparse
from pathlib import Path
import bucex as bx
from .run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("research/serra/config/independent_smoke.json"))
    parser.add_argument("--series", nargs="+", choices=tuple(bx.UCCLE_INFO))
    args = parser.parse_args()
    config = bx.load_config(args.config)
    config["analysis"] = "independent"
    print(run(config, series=args.series))


if __name__ == "__main__":
    main()
