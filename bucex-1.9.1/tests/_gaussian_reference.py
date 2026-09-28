"""Independent dense Gaussian conditioning, used only in numerical validation."""
import numpy as np


def dense_gaussian_predictor_posterior(compiled, values, parameters):
    """Condition an independently constructed dense Gaussian joint distribution."""
    n_time, n_channel = values.shape
    dim = compiled.state_dim
    transition = compiled.transition
    process = compiled.transition_cov(parameters)
    means = [compiled.initial_mean]
    cov = np.zeros(((n_time + 1) * dim, (n_time + 1) * dim))
    cov[:dim, :dim] = compiled.initial_cov
    for time in range(1, n_time + 1):
        current = slice(time * dim, (time + 1) * dim)
        previous = slice((time - 1) * dim, time * dim)
        means.append(transition @ means[-1])
        cov[current, :time * dim] = transition @ cov[previous, :time * dim]
        cov[:time * dim, current] = cov[current, :time * dim].T
        cov[current, current] = transition @ cov[previous, previous] @ transition.T + process
    design = compiled.design(params=parameters)
    observation = np.zeros((n_time * n_channel, cov.shape[0]))
    for time in range(n_time):
        observation[time * n_channel:(time + 1) * n_channel,
                    (time + 1) * dim:(time + 2) * dim] = design[time]
    variance = np.tile([parameters[f"sigma.{name}"] ** 2
                        for name in compiled.channel_names], n_time)
    mean = np.concatenate(means)
    cross = cov @ observation.T
    observed_cov = observation @ cross + np.diag(variance)
    gain = np.linalg.solve(observed_cov, cross.T).T
    posterior_mean = mean + gain @ (values.ravel() - observation @ mean)
    posterior_cov = cov - gain @ cross.T
    return observation @ posterior_mean, observation @ posterior_cov @ observation.T

