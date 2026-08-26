"""Add selected-year seasonal-pattern figures to completed Uccle runs.

No model is refitted.  The script reads ``fits/<series>/combined.bucex`` from
each completed run and writes ``seasonal_patterns.png`` and
``seasonal_patterns.pdf`` beside its existing figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


REQUESTED_RUN_NAMES = (
    "20260825_222926_03-tnx_27536126_combined__y1892-latest_d1000w1000c4",
    "20260825_222924_01-txx_27536124_combined__y1892-latest_d1000w1000c4",
)


def resolve_run(specification: str, results_root: Path) -> Path:
    supplied = Path(specification).expanduser()
    if supplied.is_dir():
        return supplied.resolve()
    matches = sorted(
        path.resolve()
        for path in results_root.expanduser().rglob(supplied.name)
        if path.is_dir()
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find run {specification!r} below {results_root}. "
            "Pass its complete directory path if it is stored elsewhere."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Run name {specification!r} is ambiguous:\n  "
            + "\n  ".join(str(path) for path in matches)
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot complete seasonal patterns without refitting."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        default=REQUESTED_RUN_NAMES,
        help="Run directories or names to find recursively below --results-root.",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--years", nargs="+", type=int, default=(1892, 2022))
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--interval", type=float, default=0.90)
    parser.add_argument("--no-interval", action="store_true")
    arguments = parser.parse_args()

    for specification in arguments.runs:
        run_dir = resolve_run(specification, arguments.results_root)
        fit_paths = sorted(run_dir.glob("fits/*/combined.bucex"))
        if not fit_paths:
            raise FileNotFoundError(
                f"{run_dir} has no fits/<series>/combined.bucex. The lightweight "
                "download containing only figures.zip cannot be replotted; run this "
                "script against the complete result directory on the HPC."
            )
        for fit_path in fit_paths:
            fit = bx.FitResult.load(fit_path)
            series = fit.series_name or fit_path.parent.name
            figure_dir = run_dir / "figures" / series
            figure_dir.mkdir(parents=True, exist_ok=True)
            figure, _ = fit.plot(
                "seasonal_patterns",
                years=arguments.years,
                credible_interval=arguments.interval,
                show_interval=not arguments.no_interval,
            )
            for extension in arguments.formats:
                output_path = figure_dir / f"seasonal_patterns.{extension}"
                figure.savefig(
                    output_path,
                    dpi=arguments.dpi,
                    bbox_inches="tight",
                )
                print(output_path)
            plt.close(figure)


if __name__ == "__main__":
    main()
