"""Rebuild the Uccle summaries through the package's existing data API."""
import argparse
from pathlib import Path

import bucex as bx


def prepare(config):
    monthly = bx.derive_uccle_monthly(
        config["daily_file"],
        end=config.get("end"),
        output_dir=config["output_dir"],
    )
    print(f"Saved {len(monthly)} months ({monthly.index[0]:%Y-%m} to "
          f"{monthly.index[-1]:%Y-%m}) in {config['output_dir']}.")
    return monthly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("research/serra/config/prepare_uccle.json"))
    prepare(bx.load_config(parser.parse_args().config))
