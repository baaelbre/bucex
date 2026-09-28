"""Fit private continuous FS trajectories with joint residual dependence."""
import argparse
from pathlib import Path
import bucex as bx
from .run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("research/monthly/config/main.json"))
    parser.add_argument("--independence", action="store_true", help="Fix R=I for a matched baseline.")
    parser.add_argument("--eta", type=float, help="LKJ concentration; use 1, 2 and 4 for sensitivity.")
    parser.add_argument("--series", nargs="+", choices=tuple(bx.UCCLE_INFO))
    parser.add_argument("--prior", choices=("normal", "lasso", "triple_gamma"))
    parser.add_argument("--scale", choices=("constant", "seasonal"),
                        help="Marginal observation scales; independent of copula --structure.")
    parser.add_argument("--structure", choices=("constant", "harmonic", "seasons", "monthly"))
    args = parser.parse_args()
    config = bx.load_config(args.config)
    config["analysis"] = "joint" if args.independence else "copula"
    if args.eta is not None:
        config.setdefault("copula", {})["eta"] = args.eta
    if args.prior:
        config['priors']['innovation'] = args.prior
    if args.scale:
        config['model']['scale_mode'] = 'constant'
        config['model']['seasonal_scale'] = args.scale == 'seasonal'
    if args.structure:
        config.setdefault('copula', {})['structure'] = args.structure
    print(run(config, series=args.series))


if __name__ == "__main__":
    main()
