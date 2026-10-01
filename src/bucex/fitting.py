"""Fit models, independently in each chain, with reproducible process parallelism."""
from __future__ import annotations
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import platform
import time
import numpy as np
import pandas as pd
from .models import Model, Channel, MultiSeriesModel
from .inference import MCMC, Laplace
from .inference._sampler import run_chain


def _data(y, model, dates):
    if isinstance(model, Model):
        name = str(y.name) if isinstance(y, pd.Series) and y.name is not None else 'y'
        model = MultiSeriesModel((Channel(name, model),))
    if not isinstance(model, MultiSeriesModel):
        raise TypeError('model must be Model or MultiSeriesModel.')
    names = [c.name for c in model.channels]
    if dates is None and isinstance(y, (pd.Series, pd.DataFrame)) and isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        dates = y.index.to_timestamp() if isinstance(y.index, pd.PeriodIndex) else y.index
    if isinstance(y, pd.DataFrame):
        if set(y.columns) != set(names) or not y.columns.is_unique:
            raise ValueError('DataFrame columns must match channel names exactly.')
        data = {name: y[name].to_numpy(dtype=float) for name in names}
    elif isinstance(y, dict):
        if set(y) != set(names):
            raise ValueError('Data keys must match channel names exactly.')
        data = {k: np.asarray(y[k], dtype=float) for k in names}
    else:
        array = np.asarray(y, dtype=float)
        if array.ndim == 1 and len(names) == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != len(names):
            raise ValueError('Use a vector for one response or a (time, channel) array.')
        data = {name: array[:, i] for i, name in enumerate(names)}
    lengths = {len(a) for a in data.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 3:
        raise ValueError('Channels need equal lengths and at least three observations.')
    if any(a.ndim != 1 or not np.all(np.isfinite(a)) for a in data.values()):
        raise ValueError('Observations must be finite vectors; missing blocks must not be silently removed.')
    index = np.arange(next(iter(lengths)))
    if dates is not None:
        index = pd.DatetimeIndex(dates)
        if len(index) != next(iter(lengths)) or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('dates must be unique, increasing, complete and match the data length.')
        if index.tz is not None:
            raise ValueError('Use timezone-naive block dates.')
        if pd.infer_freq(index) is None:
            raise ValueError('dates must form a regular calendar with no missing blocks.')
    return model, data, index


def target_fingerprint(model, data, index, steps_per_year):
    from .serialization import encode
    h = hashlib.sha256(json.dumps(encode(model), sort_keys=True, allow_nan=False).encode())
    h.update(str(float(steps_per_year)).encode())
    h.update(np.asarray(index).astype(str).tobytes())
    for name, value in data.items():
        h.update(name.encode()); h.update(np.ascontiguousarray(value, dtype='<f8').tobytes())
    return h.hexdigest()


def fit(y, *, model, priors=None, dates=None, steps_per_year=None, mcmc=None, laplace=None):
    """Fit one or several series. GEV path proposals always receive MH correction.

    Use a main guard in scripts when workers > 1. No thread or cluster globals
    are changed. Shared scales are updated inside each independent chain.
    """
    if priors is not None:
        if not isinstance(model, Model):
            raise ValueError('Set each channel Model.priors for a multiseries model.')
        model = replace(model, priors=priors)
    model, data, index = _data(y, model, dates)
    mcmc = MCMC() if mcmc is None else mcmc
    laplace = Laplace() if laplace is None else laplace
    if not isinstance(mcmc, MCMC) or not isinstance(laplace, Laplace):
        raise TypeError('mcmc and laplace must be MCMC and Laplace objects.')
    if steps_per_year is None and len({c.model.period for c in model.channels}) != 1:
        raise ValueError('Specify steps_per_year when channels have different seasonal structures.')
    steps_per_year = model.channels[0].model.period if steps_per_year is None else steps_per_year
    if not np.isfinite(steps_per_year) or steps_per_year <= 0:
        raise ValueError('steps_per_year must be positive and finite.')
    jobs = [(model, data, mcmc, laplace, i) for i in mcmc.chain_ids]
    start = time.perf_counter()
    if mcmc.workers > 1 and mcmc.chains > 1:
        with ProcessPoolExecutor(max_workers=min(mcmc.workers, mcmc.chains),
                mp_context=multiprocessing.get_context('spawn')) as pool:
            output = list(pool.map(run_chain, jobs))
    else:
        output = [run_chain(job) for job in jobs]
    from .results import FitResult, ChannelResult
    channels = {}
    for c in model.channels:
        groups = [row[0][c.name] for row in output]
        channels[c.name] = ChannelResult(c.name, c.model, np.array(data[c.name], copy=True), index,
            np.stack([g['states'] for g in groups]),
            {k: np.stack([g['parameters'][k] for g in groups]) for k in groups[0]['parameters']},
            {k: np.stack([g['metrics'][k] for g in groups]) for k in groups[0]['metrics']}, float(steps_per_year))
    shared = {k: np.stack([row[1][k] for row in output]) for k in output[0][1]}
    meta = dict(version='1.0.0', mcmc=asdict(mcmc), laplace=asdict(laplace),
        chain_ids=list(mcmc.chain_ids), chain_seconds=[r[2] for r in output],
        elapsed_seconds=time.perf_counter()-start, python=platform.python_version(),
        platform=platform.platform(), processor=platform.processor(), exact_target=True,
        target_fingerprint=target_fingerprint(model, data, index, steps_per_year))
    return FitResult(model, channels, shared, meta)


def combine_fits(*fits):
    """Combine independently run chains only when model, data and settings agree."""
    from .results import FitResult, ChannelResult
    if not fits:
        raise ValueError('Provide at least one fit.')
    first = fits[0]
    ids, identities = [], []
    for f in fits:
        if f.metadata['target_fingerprint'] != first.metadata['target_fingerprint']:
            raise ValueError('Cannot combine different data, calendars, models or priors.')
        if f.draws != first.draws:
            raise ValueError('All chains must have the same retained draw count.')
        if f.metadata['laplace'] != first.metadata['laplace'] or f.metadata['mcmc']['warmup'] != first.metadata['mcmc']['warmup']:
            raise ValueError('Use the same sampler settings and warmup for chain combination.')
        streams = f.metadata.get('random_streams', [[f.metadata['mcmc']['seed'], i] for i in f.metadata['chain_ids']])
        identities.extend(tuple(s) for s in streams)
        ids.extend(f.metadata['chain_ids'])
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate random streams: choose distinct chain_ids or seeds.')
    channels = {}
    for name, c in first.channels.items():
        channels[name] = ChannelResult(name, c.model, c.y.copy(), c.index,
            np.concatenate([f[name].states for f in fits]),
            {k: np.concatenate([f[name].parameters[k] for f in fits]) for k in c.parameters},
            {k: np.concatenate([f[name].metrics[k] for f in fits]) for k in c.metrics}, c.steps_per_year)
    metadata = dict(first.metadata)
    metadata.update(chain_ids=ids, random_streams=[list(i) for i in identities],
        chain_seconds=[s for f in fits for s in f.metadata['chain_seconds']],
        elapsed_seconds=None, combined=True)
    metadata['mcmc'] = dict(metadata['mcmc'], chains=len(ids), chain_ids=ids)
    return FitResult(first.model, channels,
        {k: np.concatenate([f.shared_scales[k] for f in fits]) for k in first.shared_scales}, metadata)
