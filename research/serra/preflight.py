"""Inspect the resolved scientific model and resource budget without sampling."""
import argparse
import json
from pathlib import Path
import numpy as np
import bucex as bx
from .models import channel, joint_model, fit_options


def inspect(config):
    data = bx.load_uccle_multiseries(**config['data'])
    declarations = [channel(name, data, config) for name in data]
    periods = config.get('contrasts')
    if periods:
        for key in ('reference', 'comparison'):
            bx.period_average(np.zeros(len(data)), data.index, periods[key])
    mcmc = bx.MCMC(**config['mcmc'])
    dimensions = [sum(c.spec.state_dim for c in item.components) for item in declarations]
    joint = config['analysis'] != 'independent'
    if joint:
        model, prior = joint_model(data, config)
        bx.plan(model, data, parameterization='fs', asis=config['inference']['asis'])
    bytes_per_state = mcmc.chains*mcmc.draws*(len(data)+1)*8
    return dict(version=bx.__version__, analysis=config['analysis'],
        start=str(data.index[0].date()), end=str(data.index[-1].date()), n_months=len(data),
        channels=[dict(name=c.name,family=c.family,tail=c.tail,model=c.to_dict()) for c in declarations],
        inference=fit_options(config, family='mixed' if any(c.family=='gev' for c in declarations) else 'gaussian')['engine'],
        asis=config['inference']['asis'], priors=config['priors'],
        copula=config.get('copula') if config['analysis']=='copula' else 'independence',
        mcmc=config['mcmc'], contrasts=periods,
        chain_execution=dict(workers=min(mcmc.chain_workers,mcmc.chains),
            requested_workers=mcmc.chain_workers, start_method='spawn' if min(mcmc.chain_workers,mcmc.chains)>1 else 'serial',
            numerical_threads_per_chain=1),
        centered_state_storage_GB=bytes_per_state*sum(dimensions)/1e9,
        peak_single_fit_state_storage_GB=bytes_per_state*(sum(dimensions) if joint else max(dimensions))/1e9,
        note='State arrays only; process workers and result assembly need additional RAM, including transfer copies. This check performs no inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(bx.load_config(args.config)),indent=2))
