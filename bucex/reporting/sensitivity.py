"""Compact prior-sensitivity reports from explicitly selected run directories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..__about__ import __version__
from ..diagnostics.calendar import coverage_by_month, pit_by_month
from ..diagnostics.sensitivity import innovation_prior_diagnostics, compare_predictive_scores
from ..plotting.style import publication_style, save_figure


class SensitivityReport:
    """Compare saved runs without loading state archives or refitting models.

    ``posterior_runs`` and ``predictive_runs`` map candidate names to mappings
    of response names to report directories. Candidate labels are never paired
    by row position. Forecast comparisons require identical held-out cases.

    Example::

        report = SensitivityReport(
            posterior_runs={"reference": {"TNm": "results/reference/TNm"},
                            "level_half": {"TNm": "results/level_half/TNm"}},
            baseline="reference")
        report.save("results/comparison")
    """

    def __init__(self, *, posterior_runs=None, predictive_runs=None, baseline="normal_reference", block_frequency="monthly"):
        self.posterior_runs = posterior_runs or {}
        self.predictive_runs = predictive_runs or {}
        self.baseline = baseline
        if block_frequency not in {'monthly','seasonal'}:
            raise ValueError('block_frequency must be monthly or seasonal.')
        self.block_frequency=block_frequency
        if not self.posterior_runs and not self.predictive_runs:
            raise ValueError("Supply posterior or predictive report directories.")
        self.sources = []

    def _read(self, path):
        path = Path(path).resolve()
        data = pd.read_csv(path)
        self.sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return data

    def tables(self):
        """Return tidy comparison tables, retaining numerical warnings."""
        self.sources = []
        collected = {}
        checks = []
        intervals = set()
        events = {}
        windows = {}
        joint_seen = set()

        def add(label, data, variant, channel):
            if 'channel' in data and not data.channel.eq(channel).all():
                raise ValueError(f"{variant}/{channel}: report channel labels do not match.")
            collected.setdefault(label, []).append(data.assign(variant=variant, channel=channel))

        for variant, channels in self.posterior_runs.items():
            for channel, directory in channels.items():
                directory = Path(directory)
                config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
                joint = config.get('analysis') in {'joint', 'copula'}
                intervals.add(config['credible_interval'])
                event = config.get('risks', {}).get(channel)
                if channel in events and events[channel] != event:
                    raise ValueError("Risk comparisons require identical thresholds for each response.")
                events[channel] = event
                for label, filename in (("prior_posterior", f"{channel}_prior_posterior.csv" if joint else "prior_posterior.csv"),
                                        ("scientific_targets", "period_and_endpoint_targets.csv"),
                                        ("mcmc", "mcmc.csv")):
                    data = self._read(directory/filename)
                    if label != 'prior_posterior' and 'quantity' not in data:
                        data = data.rename(columns={data.columns[0]: 'quantity'})
                    if joint and label == 'scientific_targets':
                        data = data[data.quantity.str.startswith(channel+'.') |
                                    data.quantity.isin([channel+'_level_change', channel+'_end_slope_C_per_decade'])]
                    elif joint and label == 'mcmc':
                        scalar_names = data.quantity.str.split('[', regex=False).str[0]
                        data = data[data.quantity.str.contains('channel.'+channel+'.', regex=False) |
                                    scalar_names.str.endswith('.'+channel)]
                    add(label, data, variant, channel)
                for quantity in ('level', 'slope', 'risk'):
                    stem = 'slope_C_per_decade' if joint and quantity == 'slope' else quantity
                    path = directory/((channel+'_' if joint else '')+stem+'.csv')
                    if not path.exists():
                        continue
                    data = self._read(path)
                    dates = tuple(pd.to_datetime(data.time))
                    if channel in windows and windows[channel] != dates:
                        raise ValueError("Posterior comparisons require exactly matching fitted dates.")
                    windows[channel] = dates
                    add('paths', data.assign(quantity=quantity), variant, channel)
                assessment = json.loads((directory/'convergence.json').read_text(encoding='utf-8'))
                checks.append(dict(stage='posterior', variant=variant, channel=channel,
                    origin='full_record', numerical_status=assessment['status'],
                    issues=len(assessment.get('issues', []))))
                unique = ('posterior', variant, str(directory.resolve()))
                if joint and unique not in joint_seen:
                    joint_seen.add(unique)
                    for label, filename in (('shared_shrinkage','shared_shrinkage.csv'),
                                            ('joint_scientific_targets','period_and_endpoint_targets.csv'),
                                            ('joint_mcmc','mcmc.csv')):
                        path = directory/filename
                        if path.exists():
                            data = self._read(path)
                            if label != 'shared_shrinkage' and 'quantity' not in data:
                                data = data.rename(columns={data.columns[0]:'quantity'})
                            collected.setdefault(label, []).append(data.assign(variant=variant))
        if len(intervals) > 1:
            raise ValueError("Posterior runs use different credible interval levels.")
        for variant, channels in self.predictive_runs.items():
            for channel, directory in channels.items():
                directory = Path(directory)
                for label, filename in (("scores", "scores.csv"), ("coverage", "coverage_by_case.csv"),
                                        ("pit", "held_out_pit.csv"), ("predictions", "predictions.csv")):
                    data = self._read(directory/filename)
                    if 'channel' in data:
                        data = data[data.channel == channel]
                        if data.empty:
                            raise ValueError(f'{variant}/{channel}: no matching held-out cases.')
                    add(label, data, variant, channel)
                folds = self._read(directory/'folds.csv')
                for fold in folds.to_dict('records'):
                    assessment = json.loads((directory/f"convergence_{fold['origin']}.json").read_text(encoding='utf-8'))
                    checks.append(dict(stage='predictive', variant=variant, channel=channel,
                        origin=fold['origin'], numerical_status=assessment['status'],
                        issues=len(assessment.get('issues', [])), **{k: v for k, v in fold.items() if k not in {'origin', 'numerical_status'}}))
                unique = ('predictive', variant, str(directory.resolve()))
                if unique not in joint_seen:
                    joint_seen.add(unique)
                    for label, filename in (('joint_log_scores','joint_log_scores.csv'),
                                            ('compound_heat_scores','compound_heat_scores.csv'),
                                            ('event_counts','event_counts.csv')):
                        if (directory/filename).exists():
                            collected.setdefault(label, []).append(self._read(directory/filename).assign(variant=variant))
                    for fold in folds.to_dict('records'):
                        path = directory/f"shared_shrinkage_{fold['origin']}.csv"
                        if path.exists():
                            collected.setdefault('shared_shrinkage_forecasts', []).append(
                                self._read(path).assign(variant=variant, origin=fold['origin']))
        result = {key: pd.concat(values, ignore_index=True) for key, values in collected.items()}
        result['convergence'] = pd.DataFrame(checks)
        if 'prior_posterior' in result:
            result['prior_updates'] = innovation_prior_diagnostics(result['prior_posterior'])
        if 'shared_shrinkage' in result:
            result['shared_shrinkage_updates'] = innovation_prior_diagnostics(result['shared_shrinkage'])
        for label, metric, value in (('joint_log_scores', 'joint_log', 'score'),
                                     ('compound_heat_scores', 'compound_brier', 'brier')):
            if label in result:
                data = result[label].copy()
                data['horizon'] = data.groupby(['variant','origin'], sort=False).cumcount()+1
                data = data.assign(channel='joint', setting=np.nan, value=data[value], score=metric)
                if self.baseline in set(data.variant):
                    result[label+'_comparison'] = compare_predictive_scores(data, baseline=self.baseline, seed=173)
                result[label+'_by_origin'] = data.groupby(['variant','origin'], sort=False).value.agg(
                    mean=lambda x: float(np.mean(x.to_numpy())), n='size').reset_index()
        if 'scores' in result:
            scores = result['scores'].copy()
            result['predictive_comparison'] = compare_predictive_scores(scores, baseline=self.baseline, seed=173)
            first_year=4 if self.block_frequency=='seasonal' else 12
            scores['horizon_band'] = np.where(scores.horizon <= first_year,
                f'horizons_1_{first_year}',f'horizons_{first_year+1}_plus')
            scores = pd.concat([scores, scores.assign(horizon_band='all')], ignore_index=True)
            grouping = ['variant','channel','origin','score','setting','horizon_band']
            result['scores_by_origin'] = scores.groupby(grouping, dropna=False, sort=False).value.agg(
                mean=lambda x: float(np.mean(x.to_numpy())), n='size').reset_index()
            grouping.remove('origin')
            result['score_summary'] = scores.groupby(grouping, dropna=False, sort=False).value.agg(
                mean=lambda x: float(np.mean(x.to_numpy())), n='size').reset_index()
            coverage = result['coverage']
            result['coverage_summary'] = coverage.groupby(['variant','channel','kind','nominal'], dropna=False).covered.agg(
                empirical='mean', n='size').reset_index()
            result['coverage_by_month'] = pd.concat([
                coverage_by_month(group).assign(variant=variant)
                for variant, group in coverage.groupby('variant', sort=False)], ignore_index=True)
            result['pit_by_month'] = pd.concat([
                pit_by_month(group.pit, group.time).assign(variant=variant, channel=channel)
                for (variant, channel), group in result['pit'].groupby(['variant','channel'], sort=False)], ignore_index=True)
            if self.block_frequency=='seasonal':
                for kind in ['coverage','pit']:
                    table=result.pop(kind+'_by_month').rename(columns={'month':'block_start_month'})
                    table['season']=table.block_start_month.map({12:'DJF',3:'MAM',6:'JJA',9:'SON'})
                    table['phase']=(table.block_start_month%12)//3+1
                    result[kind+'_by_season']=table
        return result

    def save(self, directory, *, figures=True, style='manuscript', dpi=180):
        """Write CSVs and PNGs; no automatic winner or convergence waiver."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tables = self.tables()
        for name, data in tables.items():
            data.to_csv(directory/(name+'.csv'), index=False)
        images = []
        if figures:
            from .sensitivity_plots import save_sensitivity_plots
            with publication_style(style=style, dpi=dpi):
                images = save_sensitivity_plots(tables, directory, dpi=dpi)
        manifest = dict(bucex_version=__version__, baseline=self.baseline, block_frequency=self.block_frequency,
            sources=self.sources, tables=list(tables), figures=images,
            interpretation=[
                'All forecast scores are losses: smaller is better; paired improvement is baseline minus candidate.',
                'Few forecast origins support descriptive screening, not precise model-ranking uncertainty.',
                'Shared hyperpriors are integrated in marginal prior intervals; the six responses are not six independent datasets.',
                'Prior/posterior displacement and contraction are descriptive, not objectives to maximize.',
                'Continuous SD intervals above zero are not posterior selection probabilities.',
                'Smooth paths do not by themselves establish acceleration; inspect period slope contrasts and their sensitivity.',
                'Runs with numerical warnings remain visible. No automatic winner is selected.',
                'A chosen prior is calibrated using these data; final intervals condition on that choice.'
            ])
        (directory/'report.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        (directory/'README.md').write_text(
            '# Prior and predictive assessment\n\n'
            'Start with `convergence.csv`; short runs commonly need more draws. '
            'Then inspect `prior_updates.csv`, `scientific_targets.csv` and the level/slope/risk overlays. '
            'For held-out forecasts read `scores_by_origin.csv`, `predictive_comparison.csv`, ' +
            ('`coverage_by_season.csv` and `pit_by_season.csv`.\n\n' if self.block_frequency=='seasonal' else
             '`coverage_by_month.csv` and `pit_by_month.csv`.\n\n') +
            '\n'.join('- '+item for item in manifest['interpretation'])+'\n', encoding='utf-8')
        return directory


__all__ = ['SensitivityReport']
