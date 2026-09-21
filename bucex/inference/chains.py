"""Execution of independent chains, separate from all statistical kernels.

Spawned processes receive the same SeedSequence children as serial execution.
Results are assembled in chain order, never completion order. Only this module
owns process creation, numerical thread limits and cross-chain bookkeeping.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import asdict, replace
from functools import wraps
from importlib import import_module
import multiprocessing
import os

import numpy as np
from threadpoolctl import threadpool_limits


_seeds = ContextVar("bucex_chain_seeds", default=None)
_position = ContextVar("bucex_chain_position", default=None)


def chain_seeds(seed, chains, *, integer=False):
    """Retain each backend's historical seed convention, including one chain."""
    supplied = _seeds.get()
    if supplied is not None:
        if len(supplied) != chains:
            raise RuntimeError("Nested chain execution has an inconsistent chain count.")
        return list(supplied)
    if integer and chains == 1 and seed is not None:
        return [int(seed)]
    children = np.random.SeedSequence(seed).spawn(int(chains))
    return [int(s.generate_state(1)[0]) for s in children] if integer else children


def chain_position(chain, chains):
    """One-based progress labels shared by serial and process workers."""
    return _position.get() or (chain, chains)


def _worker(module, name, args, kwargs, sequence, index, total):
    seed_token = _seeds.set((sequence,))
    position_token = _position.set((index + 1, total))
    try:
        with threadpool_limits(limits=1):
            result = getattr(import_module(module), name)(*args, **kwargs)
        return result, os.getpid()
    finally:
        _seeds.reset(seed_token)
        _position.reset(position_token)


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(left, right, equal_nan=True)
        except TypeError:
            return np.array_equal(left, right)
    return left == right


def _join_arrays(mappings):
    keys = mappings[0].keys()
    if any(item.keys() != keys for item in mappings):
        raise RuntimeError("Chains produced different parameter or diagnostic blocks.")
    return {key: np.concatenate([np.asarray(item[key]) for item in mappings], axis=0)
            for key in keys}


def _merge(results, mcmc):
    first = results[0]
    for result in results:
        if result.n_chains != 1 or result.draws_per_chain != mcmc.draws:
            raise RuntimeError("A chain returned an unexpected number of retained draws.")
        if result.state_draws.shape[2:] != first.state_draws.shape[2:]:
            raise RuntimeError("Chain state dimensions differ.")
        if result.plan != first.plan or not _same(result.y, first.y):
            raise RuntimeError("Cannot assemble chains with different models or observations.")
    diagnostics = dict(first.sampler_diagnostics)
    array_groups = {"draw_metrics", "acceptance", "final_proposal_steps", "sign_switch_counts"}
    list_keys = {"chain_seeds", "restored_fraction_by_chain", "sign_switch_counts_by_chain",
                 "restored_iterations_by_chain", "laplace_initialization_by_chain",
                 "final_proposal_widths"}
    for key in diagnostics:
        values = [r.sampler_diagnostics[key] for r in results]
        if key in array_groups:
            diagnostics[key] = _join_arrays(values)
        elif key in list_keys:
            diagnostics[key] = [entry for value in values for entry in value]
        elif key != "mcmc" and not all(_same(values[0], v) for v in values[1:]):
            raise RuntimeError(f"Unrecognized chain-varying diagnostic: {key}.")
    diagnostics["mcmc"] = asdict(mcmc)
    initial = dict(first.initial_values)
    for key in initial:
        values = [r.initial_values[key] for r in results]
        if key == "chains" or key.endswith("_by_chain"):
            initial[key] = [entry for value in values for entry in value]
        elif not all(_same(values[0], v) for v in values[1:]):
            raise RuntimeError(f"Unrecognized chain-varying initial value: {key}.")
    metadata = dict(first.metadata)
    for key in metadata:
        values = [r.metadata[key] for r in results]
        if key.endswith("_by_chain"):
            metadata[key] = [entry for value in values for entry in value]
        elif key in {"attempt_failure_counts", "restore_failure_counts"}:
            counts = Counter()
            for value in values:
                counts.update(value)
            metadata[key] = dict(counts)
        elif key == "restored_iterations":
            metadata[key] = sum(values)
        elif key == "restored_fraction":
            metadata[key] = float(np.mean(values))
        elif not all(_same(values[0], v) for v in values[1:]):
            raise RuntimeError(f"Unrecognized chain-varying metadata: {key}.")
    return replace(first,
        state_draws=np.concatenate([r.state_draws for r in results], axis=0),
        parameter_draws=_join_arrays([r.parameter_draws for r in results]),
        log_posterior=np.concatenate([r.log_posterior for r in results], axis=0),
        auxiliary_draws=_join_arrays([r.auxiliary_draws for r in results]),
        sampler_diagnostics=diagnostics, initial_values=initial, metadata=metadata)


def independent_chains(*, integer_seeds=False):
    """Decorate a FitResult-producing backend with the common execution policy.

    Serial execution retains its existing multi-chain loop. Parallel execution
    calls that same backend once per chain with an explicit seed context, so no
    sampler algorithm or public initialization convention is duplicated here.
    """
    def decorate(function):
        @wraps(function)
        def execute(*args, **kwargs):
            if _seeds.get() is not None:
                return function(*args, **kwargs)
            mcmc = kwargs["mcmc"]
            seeds = chain_seeds(mcmc.seed, mcmc.chains, integer=integer_seeds)
            workers = min(mcmc.chain_workers, mcmc.chains)
            pids = [os.getpid()] * mcmc.chains
            if workers == 1:
                token = _seeds.set(tuple(seeds))
                try:
                    with threadpool_limits(limits=1):
                        result = function(*args, **kwargs)
                finally:
                    _seeds.reset(token)
            else:
                initial = kwargs.get("initial_parameters")
                starts = initial.get("chains") if isinstance(initial, dict) else None
                if starts is not None and len(starts) != mcmc.chains:
                    raise ValueError("Initial chains must contain one entry per requested chain.")
                if mcmc.progress:
                    print(f"BUCEX: running {mcmc.chains} chains with {workers} processes", flush=True)
                results = [None] * mcmc.chains
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                    jobs = {}
                    for index, seed in enumerate(seeds):
                        local = {**kwargs, "mcmc": replace(mcmc, chains=1, chain_workers=1)}
                        if starts is not None:
                            local["initial_parameters"] = {**initial, "chains": [starts[index]]}
                        job = executor.submit(_worker, function.__module__, function.__name__,
                            args, local, seed, index, mcmc.chains)
                        jobs[job] = index
                    for job in as_completed(jobs):
                        index = jobs[job]
                        try:
                            results[index], pids[index] = job.result()
                        except Exception as error:
                            for pending in jobs:
                                pending.cancel()
                            raise RuntimeError(
                                f"MCMC chain {index + 1}/{mcmc.chains} failed; no partial fit returned. "
                                "Run scripts under an if __name__ == '__main__' guard when using "
                                f"chain_workers > 1. Original error: {error}") from error
                        if mcmc.progress:
                            print(f"BUCEX: chain {index + 1}/{mcmc.chains} completed", flush=True)
                result = _merge(results, mcmc)
            # Store entropy and spawn keys, not only a lossy integer fingerprint.
            seed_states = [({"integer": int(s)} if isinstance(s, (int, np.integer)) else
                            {"entropy": s.entropy, "spawn_key": list(s.spawn_key), "pool_size": s.pool_size})
                           for s in seeds]
            result.sampler_diagnostics["execution"] = {
                "chain_workers_requested": mcmc.chain_workers, "chain_workers": workers,
                "start_method": "spawn" if workers > 1 else "serial",
                "numerical_threads_per_chain": 1, "worker_pids": pids,
                "seed_states": seed_states, "chain_order": list(range(mcmc.chains)),
            }
            return result
        return execute
    return decorate
