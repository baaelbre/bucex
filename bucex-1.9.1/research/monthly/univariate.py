"""Fit any of the six private continuous FS temperature trajectories."""
import argparse
from pathlib import Path
import bucex as bx
from .run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("research/monthly/config/independent.json"))
    parser.add_argument("--series", nargs="+", choices=tuple(bx.UCCLE_INFO))
    parser.add_argument("--prior", choices=("normal", "lasso", "triple_gamma"))
    parser.add_argument("--scale", choices=("constant", "seasonal"),
                        help="Constant scale or repeating monthly scale (supplement).")
    args = parser.parse_args()
    config = bx.load_config(args.config)
    config["analysis"] = "independent"
    if args.prior:
        config['priors']['innovation'] = args.prior
    if args.scale:
        config['model']['scale_mode'] = 'constant'
        config['model']['seasonal_scale'] = args.scale == 'seasonal'
    print(run(config, series=args.series))


if __name__ == "__main__":
    main()
