"""Versioned JSON + NPZ archives. No pickle, arbitrary imports or code execution."""
from dataclasses import fields, is_dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pandas as pd
from . import priors, components, distributions, models, parameters

REGISTRY = {c.__name__: c for c in (priors.Normal, priors.HalfNormal, priors.InverseGamma,
    priors.Fixed, priors.Priors, priors.Pooling, components.LocalLevel, components.LocalLinearTrend,
    components.DummySeasonal, components.Regression, components.Cycle, parameters.Constant, parameters.Latent, distributions.Gaussian, distributions.GEV, distributions.SeasonalScale,
    models.Model, models.Channel, models.MultiSeriesModel)}


def register_type(cls, *, name=None):
    """Opt in a trusted dataclass for configuration/fit round trips.

    Extensions must register on both writing and reading processes. Loading
    never imports modules named in an archive or executes archive code.
    """
    if not is_dataclass(cls):
        raise TypeError('Only dataclass configurations can be registered.')
    name = cls.__name__ if name is None else name
    if not isinstance(name, str) or not name or (name in REGISTRY and REGISTRY[name] is not cls):
        raise ValueError('A distinct nonempty serialization name is required.')
    REGISTRY[name] = cls
    return cls


def encode(value):
    if is_dataclass(value):
        names = [name for name, cls in REGISTRY.items() if cls is type(value)]
        if not names:
            raise TypeError(f'Register configuration type {type(value).__name__} with register_type first.')
        return {'type': names[0], **{f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return tuple(decode(v) for v in value)
    if isinstance(value, dict):
        if 'type' not in value:
            return {k: decode(v) for k, v in value.items()}
        if value['type'] not in REGISTRY:
            raise ValueError('Unknown configuration type; explicitly register trusted extensions before loading.')
        cls = REGISTRY[value['type']]
        valid = {f.name for f in fields(cls)}
        if set(value)-{'type'}-valid:
            raise ValueError(f'Unknown fields for {cls.__name__}: {set(value)-valid-{"type"}}')
        return cls(**{k: decode(v) for k, v in value.items() if k != 'type'})
    return value


def save(fit, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, channel_info = {}, []
    for i, (name, c) in enumerate(fit.channels.items()):
        prefix = f'channel{i}'
        arrays[prefix+'.y'], arrays[prefix+'.states'] = c.y, c.states
        if c.exog is not None:
            arrays[prefix+'.exog'] = c.exog.to_numpy(dtype=float)
        for group in ('parameters', 'metrics'):
            for j, (key, value) in enumerate(getattr(c, group).items()):
                arrays[f'{prefix}.{group}.{j}'] = value
        datetime = isinstance(c.index, pd.DatetimeIndex)
        arrays[prefix+'.index'] = c.index.to_numpy(dtype='datetime64[ns]').astype('int64') if datetime else np.asarray(c.index)
        channel_info.append(dict(name=name, prefix=prefix, datetime=datetime, steps_per_year=c.steps_per_year,
            parameters=list(c.parameters), metrics=list(c.metrics), exog_columns=list(c.exog.columns) if c.exog is not None else None))
    for i, value in enumerate(fit.shared_scales.values()):
        arrays[f'shared.{i}'] = value
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    metadata = dict(format='bucex-fit', schema=2, version='1.0.0', model=encode(fit.model),
        channels=channel_info, shared=list(fit.shared_scales), metadata=fit.metadata,
        sha256=hashlib.sha256(payload).hexdigest())
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('metadata.json', json.dumps(metadata, allow_nan=False, indent=2))
            archive.writestr('arrays.npz', payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load(path):
    from .results import FitResult, ChannelResult
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read('metadata.json'))
        if meta.get('format') != 'bucex-fit' or meta.get('schema') != 2:
            raise ValueError('Unsupported archive. Draft 1.0 and development 1.9 archives require their original reader; regenerate fits with this release.')
        payload = archive.read('arrays.npz')
    if hashlib.sha256(payload).hexdigest() != meta['sha256']:
        raise ValueError('Archive checksum mismatch.')
    model = decode(meta['model'])
    channels = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        for channel, info in zip(model.channels, meta['channels']):
            prefix = info['prefix']
            index = arrays[prefix+'.index']
            if info['datetime']:
                index = pd.DatetimeIndex(index.astype('datetime64[ns]'))
            channels[channel.name] = ChannelResult(channel.name, channel.model, arrays[prefix+'.y'], index,
                arrays[prefix+'.states'],
                {k: arrays[f'{prefix}.parameters.{i}'] for i, k in enumerate(info['parameters'])},
                {k: arrays[f'{prefix}.metrics.{i}'] for i, k in enumerate(info['metrics'])}, info['steps_per_year'],
                pd.DataFrame(arrays[prefix+'.exog'], columns=info['exog_columns'], index=index) if info.get('exog_columns') is not None else None)
        shared = {k: arrays[f'shared.{i}'] for i, k in enumerate(meta['shared'])}
    fit = FitResult(model, channels, shared, meta['metadata'])
    # Check that the configuration and observed data have not been mixed.
    from .fitting import target_fingerprint
    first = next(iter(channels.values()))
    fingerprint = target_fingerprint(model, {k: c.y for k, c in channels.items()}, first.index, first.steps_per_year, {k: c.exog for k, c in channels.items()})
    if fingerprint != fit.metadata['target_fingerprint']:
        raise ValueError('Stored model/data fingerprint mismatch.')
    return fit
