"""Compile additive linear state blocks to a generic non-centred system."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag
from ..priors import Fixed
from ..parameters import Latent


@dataclass
class CompiledInnovation:
    name: str
    prior: object
    coefficient: int
    state_slice: slice
    projection: np.ndarray
    loading: np.ndarray


@dataclass
class CompiledModel:
    transition: np.ndarray
    design: np.ndarray
    initial: np.ndarray
    initial_design: np.ndarray
    priors: tuple
    coefficient_names: tuple
    state_names: tuple
    groups: tuple
    ncp_transition: np.ndarray
    ncp_covariance: np.ndarray
    outputs: dict
    initial_outputs: dict

    @property
    def state_dim(self):
        return len(self.transition)

    @property
    def ncp_dim(self):
        return len(self.ncp_transition)

    @property
    def initial_count(self):
        return self.initial.shape[1]

    def coefficient_design(self, z):
        columns = [self.initial_design]
        for g in self.groups:
            projected = z[1:, g.state_slice] @ g.projection.T
            columns.append(np.sum(self.design*projected, axis=1)[:, None])
        return np.column_stack(columns)

    def path_design(self, theta):
        H = np.zeros((len(self.design), self.ncp_dim))
        for g in self.groups:
            H[:, g.state_slice] = theta[g.coefficient] * (self.design @ g.projection)
        return H

    def offset(self, theta):
        return self.initial_design @ theta[:self.initial_count]

    def physical_path(self, theta, z):
        initial = self.initial @ theta[:self.initial_count]
        path = np.empty((len(self.design)+1, self.state_dim)); path[0] = initial
        for t in range(1, len(path)):
            path[t] = self.transition @ path[t-1]
        for g in self.groups:
            path += theta[g.coefficient] * (z[:, g.state_slice] @ g.projection.T)
        return path

    def paths(self, physical):
        result = {name: np.einsum('...td,td->...t', physical, row)
                  for name, row in self.outputs.items()}
        result['location'] = np.einsum('...td,td->...t', physical, self.design)
        return result

    def parameter_values(self, theta):
        result = dict(zip(self.coefficient_names, theta))
        for g in self.groups:
            result['amplitude.'+g.name] = theta[g.coefficient]
            result['variance.'+g.name] = theta[g.coefficient]**2
        for name, row in self.initial_outputs.items():
            result[name] = float(row @ theta[:self.initial_count])
        return result


def placeholder_exog(model, steps):
    import pandas as pd
    names = tuple(dict.fromkeys(f for c in model.components for f in getattr(c, 'features', ())))
    return pd.DataFrame(np.zeros((steps, len(names))), columns=names) if names else None


def compile_model(model, steps, exog=None):
    """Compile the location predictor; sampler capabilities are checked separately."""
    return _compile_components(model.components, model.priors, steps, exog)


def compile_parameter(model, name, steps, exog=None):
    """Compile a named latent *linear predictor*, including future parameter blocks.

    A log link is applied by the observation/update backend, not to F or H.
    Compilation does not claim that the 1.0 sampler supports that parameter.
    """
    if name == 'location':
        return compile_model(model, steps, exog)
    parameter = model.parameters.get(name)
    if not isinstance(parameter, Latent):
        raise ValueError(f'{name!r} has no latent predictor.')
    return _compile_components(parameter.components, model.priors, steps, exog)


def _compile_components(components, priors, steps, exog):
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer.')
    blocks = [c.build(steps, priors, exog).validate(steps) for c in components
              if getattr(c, 'mode', None) != 'off']
    if not blocks:
        raise ValueError('At least one active component is required.')
    F = block_diag(*(b.transition for b in blocks))
    B = block_diag(*(b.initial for b in blocks))
    H = np.column_stack([b.design for b in blocks])
    priors = [p for b in blocks for p in b.priors]
    names = [n for b in blocks for n in b.coefficient_names]
    state_names = tuple(n for b in blocks for n in b.state_names)
    if len(set(names)) != len(names) or len(set(state_names)) != len(state_names):
        raise ValueError('Component coefficient and state names must be unique.')
    d, n0 = B.shape
    D = np.empty((steps, n0)); propagated = B.copy()
    for t in range(steps):
        propagated = F @ propagated
        D[t] = H[t] @ propagated
    outputs, initial_outputs, groups, nFs, nQs = {}, {}, [], [], []
    pos, ipos, zpos = 0, 0, 0
    for b in blocks:
        bd, bp = b.initial.shape
        rows = {b.name: b.design, **b.outputs}
        for name, row in rows.items():
            if name in outputs:
                raise ValueError(f'Duplicate component output {name}.')
            full = np.zeros((steps, d)); full[:, pos:pos+bd] = row
            outputs[name] = full
        for name, row in b.initial_outputs.items():
            full = np.zeros(n0); full[ipos:ipos+bp] = row
            initial_outputs[name] = full
        for g in b.innovations:
            if isinstance(g.prior, Fixed) and g.prior.value == 0:
                continue
            if any(other.name == g.name for other in groups):
                raise ValueError(f'Duplicate innovation group {g.name}.')
            R = np.asarray(g.loading, dtype=float)
            # Remove exactly unreachable coordinates, retaining singular transitions.
            reach = R.copy(); term = R.copy()
            for _ in range(1, bd):
                term = b.transition @ term
                reach = np.column_stack((reach, term))
            active = np.flatnonzero(np.any(reach != 0, axis=1))
            if not len(active):
                raise ValueError(f'{g.name}: innovation does not affect any state.')
            projection = np.zeros((d, len(active))); projection[pos+active, np.arange(len(active))] = 1
            loading = np.zeros((d, R.shape[1])); loading[pos:pos+bd] = R
            zslice = slice(zpos, zpos+len(active))
            groups.append(CompiledInnovation(g.name, g.prior, len(priors), zslice, projection, loading))
            priors.append(g.prior); names.append('amplitude.'+g.name)
            nFs.append(b.transition[np.ix_(active, active)])
            nQs.append(R[active] @ R[active].T)
            zpos += len(active)
        pos += bd; ipos += bp
    # Add physical coordinate outputs only when the component has not named them.
    for i, name in enumerate(state_names):
        if name not in outputs:
            outputs[name] = np.tile(np.eye(d)[i], (steps, 1))
    nF = block_diag(*nFs) if nFs else np.zeros((0, 0))
    nQ = block_diag(*nQs) if nQs else np.zeros((0, 0))
    return CompiledModel(F, H, B, D, tuple(priors), tuple(names), state_names,
                         tuple(groups), nF, nQ, outputs, initial_outputs)
