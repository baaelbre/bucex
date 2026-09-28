"""Gate a seasonal manuscript run before its numbers enter the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

import bucex as bx


SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
EXPECTED_DAILY_SHA256 = "a08cefa73732dc3ed7e40b3bdf79c22090b16931b6ae71c502de642d842f6e38"


def assess(run: Path, expected_config: Path) -> dict:
    """Return a strict, machine-readable final-run assessment."""
    run = run.resolve()
    expected = bx.load_config(expected_config)
    issues: list[str] = []
    required = [
        "config.json", "data_window.json", "run.json", "convergence.json",
        "mcmc.csv", "scientific_targets.csv", "period_contrasts.csv",
        "shared_shrinkage.csv", "initial_slope_prior_posterior.csv",
    ]
    if expected.get('analysis') == 'copula':
        required.append('copula_correlations.csv')
    if expected.get('report_joint_risks', True):
        required.append('compound_heat_conditional_risk.csv')
    for name in SERIES:
        required.extend([
            f"{name}_level.csv", f"{name}_slope_C_per_decade.csv",
            f"{name}_scale_by_season.csv", f"{name}_smoothed_pit.csv",
            f"{name}_risk.csv", f"{name}_forecast_season_risk.csv",
            f"{name}_forecast_uncertainty.csv",
        ])
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        issues.append("missing report files: " + ", ".join(missing))

    stored = bx.load_config(run / "config.json") if (run / "config.json").is_file() else None
    if stored is not None and stored != expected:
        issues.append(f"saved config does not exactly match {expected_config}")

    convergence = (bx.load_config(run / "convergence.json")
                   if (run / "convergence.json").is_file() else {})
    if convergence.get("status") != "passed_numerical_checks":
        details = [item.get("quantity", "unknown") for item in convergence.get("issues", [])]
        issues.append("convergence gate is not passed" + (": " + ", ".join(details) if details else ""))

    window = bx.load_config(run / "data_window.json") if (run / "data_window.json").is_file() else {}
    if window.get("n_blocks") != 538:
        issues.append(f"expected 538 complete seasons, found {window.get('n_blocks')!r}")
    if window.get("last_included_day") != "2026-08-31":
        issues.append(f"expected last included day 2026-08-31, found {window.get('last_included_day')!r}")
    if window.get("daily_sha256") != EXPECTED_DAILY_SHA256:
        issues.append("daily-source checksum differs from the declared Uccle source")

    metadata = bx.load_config(run / "run.json") if (run / "run.json").is_file() else {}
    if metadata.get("bucex_version") != bx.__version__:
        issues.append(f"report version is {metadata.get('bucex_version')!r}, expected {bx.__version__!r}")
    if stored is not None:
        if stored.get('analysis') != 'joint':
            issues.append('1.9.1 final analysis must use identity dependence')
        p=stored.get('priors', {})
        if p.get('initial_slope_sd') != .0125 or p.get('shared_shrinkage', {}).get('pool_initial_slope', True):
            issues.append('initial rates must have separate Normal priors with SD 0.5 C/decade')
        mcmc = stored.get("mcmc", {})
        if (mcmc.get("chains"), mcmc.get("warmup"), mcmc.get("draws")) != (4, 3000, 8000):
            issues.append("final MCMC budget must be 4 chains, 3000 warm-up and 8000 retained draws")
        if (stored.get('forecast_horizon'), stored.get('forecast_draws'),
                stored.get('credible_interval')) != (120, 12000, .95):
            issues.append('final forecast must cover 120 seasons using 12000 draws and 95% intervals')
    for name in SERIES:
        file = run/f'{name}_forecast_uncertainty.csv'
        if file.exists():
            frame = pd.read_csv(file)
            if (not set(('horizon','nominal','target','lower','upper')) <= set(frame) or
                    len(frame) != 360 or frame.horizon.max() != 120 or
                    not frame.nominal.eq(.95).all()):
                issues.append(f'{name}: invalid 30-year forecast uncertainty or interval level')

    return {
        "bucex_version": bx.__version__,
        "run": str(run),
        "expected_config": str(expected_config.resolve()),
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "convergence": convergence,
        "data": {
            "n_blocks": window.get("n_blocks"),
            "last_included_day": window.get("last_included_day"),
            "daily_sha256": window.get("daily_sha256"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="The timestamped joint report printed by research.seasonal.fit.")
    parser.add_argument("--config", type=Path,
                        default=Path("research/seasonal/config/final.json"))
    args = parser.parse_args()
    report = assess(args.run, args.config)
    target = args.run / "final_check.json"
    try:
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        # Read-only archived reports can still be audited. A newly completed
        # final run is expected to be writable and will retain the check.
        print(f"warning: could not write {target}: {error}", file=sys.stderr)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
