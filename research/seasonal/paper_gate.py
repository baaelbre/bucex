"""Reject missing or flagged numerical checks in supporting manuscript runs."""
import argparse
import json
from pathlib import Path


def assess(paths):
    issues = []
    counts = {}
    for root in paths:
        files = sorted(Path(root).rglob('convergence*.json'))
        counts[str(root)] = len(files)
        if not files:
            issues.append(f'{root}: no convergence reports')
        for path in files:
            report = json.loads(path.read_text())
            if report.get('status') not in ('passed', 'passed_numerical_checks'):
                issues.append(f'{path}: {report.get("status", "missing status")}')
    return dict(status='passed' if not issues else 'failed', counts=counts, issues=issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    result = assess(args.runs)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 2)
