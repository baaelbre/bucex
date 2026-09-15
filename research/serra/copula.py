"""Fit private FS/SSVS trajectories with joint residual dependence."""
import argparse
from pathlib import Path
import bucex as bx
from .run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("research/serra/config/copula_smoke.json"))
    parser.add_argument("--independence", action="store_true", help="Fix R=I for a matched baseline.")
    parser.add_argument("--eta", type=float, help="LKJ concentration; use 1, 2 and 4 for sensitivity.")
    parser.add_argument("--series", nargs="+", choices=tuple(bx.UCCLE_INFO))
    args = parser.parse_args()
    config = bx.load_config(args.config)
    config["analysis"] = "joint" if args.independence else "copula"
    if args.eta is not None:
        config["copula"] = {"eta": args.eta}
    print(run(config, series=args.series))


if __name__ == "__main__":
    main()
