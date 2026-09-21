"""Fit two new prior settings and compare them with completed assessments."""
import argparse
import json
from pathlib import Path

import bucex as bx
from .prior_assessment import run, study_plan


def saved_runs(directory, config):
    """Map compact report directories to BUCEX's general comparison API."""
    return {
        key: {
            variant['name']: {
                name: directory / stage / variant['name'] / name
                for name in config['data']['series']
            }
            for variant in config['variants']
        }
        for stage, key in (('sensitivity', 'posterior_runs'),
                           ('predictive', 'predictive_runs'))
    }


def completed_config(directory):
    manifest = bx.load_config(directory / 'assessment.json')
    if any(manifest.get(stage) != 'completed' for stage in ('sensitivity', 'predictive')):
        raise ValueError(f'{directory}: both assessment stages must be completed.')
    return bx.load_config(directory / 'config.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True,
                        help='Completed assessment containing the two existing settings.')
    parser.add_argument('--config', type=Path,
                        default=Path('research/serra/config/priors/refine2.json'))
    parser.add_argument('--stage', choices=('plan', 'all', 'report'), default='plan')
    parser.add_argument('--run', type=Path,
                        help='Completed new assessment, for --stage report only.')
    args = parser.parse_args()
    if (args.stage == 'report') != (args.run is not None):
        parser.error('Use --run only with --stage report, supplying the completed new directory.')

    previous = args.previous.expanduser().resolve()
    old = completed_config(previous)
    config = completed_config(args.run) if args.stage == 'report' else bx.load_config(args.config)
    shared = ('analysis', 'data', 'model', 'priors', 'inference', 'validation',
              'contrasts', 'risks', 'credible_interval')
    changed = [key for key in shared if config.get(key) != old.get(key)]
    if changed:
        raise ValueError(f'Reuse requires matching scientific settings; different: {changed}')
    old_names = {v['name'] for v in old['variants']}
    new_names = {v['name'] for v in config['variants']}
    if old_names & new_names:
        raise ValueError('New variants must have distinct names; existing runs are reused.')
    baseline = old['assessment']['baseline']
    if baseline not in old_names:
        raise ValueError('The previous assessment must include its declared baseline.')
    mappings = saved_runs(previous, old)
    # Check that the compact old exports can be read before launching any fits.
    bx.SensitivityReport(**mappings, baseline=baseline).tables()

    if args.stage == 'plan':
        plan = study_plan(config)
        plan.update(reused_variants=sorted(old_names), comparison_baseline=baseline)
        print(json.dumps(plan, indent=2))
        return

    directory = args.run.resolve() if args.stage == 'report' else run(config, stage='all')
    for key, values in saved_runs(directory, config).items():
        mappings[key].update(values)
    output = bx.SensitivityReport(**mappings, baseline=baseline).save(
        directory / 'combined_comparison', figures=config.get('figures', True),
        style=config.get('figure_style', 'manuscript'), dpi=config.get('figure_dpi', 180))
    bx.save_config({'previous': str(previous), 'new': str(directory),
                    'baseline': baseline}, output / 'assessment_sources.json')
    print(f'All four settings: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
