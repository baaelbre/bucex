"""Known-truth mixed-margin recovery with private continuous FS and a residual copula."""
import argparse
from dataclasses import replace
import numpy as np
import pandas as pd
import bucex as bx
from research.serra.models import joint_model, fit_options
from research.serra.report import new_run


def run(config, replicate=None):
    settings = config['simulation']
    n, horizon = settings['n_time'], config['forecast_horizon']
    dates = pd.date_range('1900-01-01', periods=n + horizon, freq='MS')
    names = config['data']['series']
    template = pd.DataFrame(0., index=dates[:n], columns=names)
    base, _ = joint_model(template, {**config, 'analysis': 'copula'})
    directory = new_run(config['output'], 'joint_recovery')
    bx.save_config(config, directory / 'config.json')
    replicates = range(settings['replicates']) if replicate is None else [replicate]
    if any(r < 0 or r >= settings['replicates'] for r in replicates):
        raise ValueError('replicate is a zero-based index within the configured budget.')
    recovery, correlations, structures, scores = [], [], [], []
    for rep in replicates:
        for ri, rho in enumerate(settings['rho_grid']):
            for xi_i, xi in enumerate(settings['xi_grid']):
                seed = config['seed'] + 10000 * rep + 100 * ri + xi_i
                R = rho ** np.abs(np.arange(len(names))[:, None] - np.arange(len(names)))
                generator = replace(base, copula=bx.GaussianCopula(correlation=R))
                compiled = bx.compile_model(generator, np.zeros((n + horizon, len(names))))
                initial, params = np.zeros(compiled.state_dim), {}
                for item, block in zip(base.channels, compiled.blocks):
                    name, sign = item.name, item.transform_sign
                    # Statistical recovery experiment, not a daily-summary ordering model.
                    initial[block.state_slice.start] = sign * 10.
                    initial[block.state_slice.start + 1] = sign * .003
                    period = config['model']['period']
                    initial[block.state_slice.start + 2:block.state_slice.stop] = 2 * np.cos(2*np.pi*np.arange(period-1)/period)
                    params[f'sigma.{name}'] = 1.5
                    scale_period = item.observation.scale.period if item.observation.scale else 1
                    params[f'scale.seasonal.{name}'] = (settings['log_scale_amplitude'] * np.cos(2*np.pi*np.arange(scale_period)/scale_period) if scale_period > 1 else np.zeros(1))
                    if item.family == 'gev':
                        params[f'xi.{name}'] = xi
                    for process, legacy in [('level', 'level'), ('slope', 'trend'), ('seasonal', 'season')]:
                        params[f'sd.channel.{name}.{process}'] = config['priors']['innovation_median'][legacy] * settings.get('innovation_multiplier', 1.)
                truth = bx.simulate(generator, n_time=n+horizon, params=params,
                                    initial_state=initial, dates=dates, seed=seed)
                training = pd.DataFrame(truth.y[:n], index=dates[:n], columns=names)
                for mode in ('joint', 'copula'):
                    print(f'replicate {rep}, rho={rho}, xi={xi}, {mode}', flush=True)
                    current = {**config, 'analysis': mode, 'mcmc': {**config['mcmc'], 'seed': seed+1}}
                    model, prior = joint_model(training, current)
                    fit = bx.fit(training, model, priors=prior, **fit_options(current, family='mixed'))
                    label = dict(replicate=rep, rho=rho, xi=xi, analysis=mode)
                    prefix = f'{rep}_{ri}_{xi_i}_{mode}'
                    if config.get('save_fits', False):
                        fit.save(directory / (prefix+'.bucex'))
                    fit.diagnostics()['parameters'].to_csv(directory / f'mcmc_{prefix}.csv')
                    for j, name in enumerate(names):
                        estimate = fit.channel_eta_draws(name, original_scale=True)
                        target = truth.eta[:n, j]
                        lo, hi = np.quantile(estimate, [.025, .975], axis=0)
                        recovery.append({**label, 'channel': name,
                            'location_rmse': np.sqrt(np.mean((estimate.mean(axis=0)-target)**2)),
                            'location_coverage_95': np.mean((lo <= target) & (target <= hi)),
                            'scale_rmse': np.sqrt(np.mean((fit.sigma_draws(channel=name).mean(axis=0)-np.asarray(truth.params.get(f'sigma_path.{name}', np.full(n+horizon, 1.5)))[:n])**2))})
                    structures.append(fit.contrast_diagnostics({key:value for name in names for key,value in
                        ((name+'.'+k,v) for k,v in fit.innovation_effect_draws(120,channel=name,combine_chains=False).items())}).reset_index().assign(**label))
                    table = fit.copula_summary(credible_interval=.95).reset_index()
                    table['truth'] = [R[names.index(key.split('.')[1]), names.index(key.split('.')[2])] for key in table.iloc[:, 0]]
                    correlations.append(table.assign(**label))
                    forecast = fit.forecast(horizon, dates=dates[n:], seed=seed+2)
                    scores.append({**label, 'mean_joint_log_score': np.mean(forecast.joint_log_score(truth.y[n:]))})
                    # Checkpoint each completed case so long grids remain auditable.
                    pd.DataFrame(recovery).to_csv(directory / 'recovery.csv', index=False)
                    pd.concat(correlations).to_csv(directory / 'correlations.csv', index=False)
                    pd.concat(structures).to_csv(directory / 'innovation_effects.csv', index=False)
                    pd.DataFrame(scores).to_csv(directory / 'scores.csv', index=False)
    pd.DataFrame(recovery).to_csv(directory / 'recovery.csv', index=False)
    pd.concat(correlations).to_csv(directory / 'correlations.csv', index=False)
    pd.concat(structures).to_csv(directory / 'innovation_effects.csv', index=False)
    pd.DataFrame(scores).to_csv(directory / 'scores.csv', index=False)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='research/serra/config/joint_recovery_smoke.json')
    parser.add_argument('--replicate', type=int)
    args = parser.parse_args()
    print(run(bx.load_config(args.config), args.replicate))
