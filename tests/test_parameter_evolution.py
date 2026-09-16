"""Independent target checks and lifecycle tests for ancillary FS evolution."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm, genextreme
import bucex as bx
from bucex.inference.fit.evolution import GaussianEvolutionState, forecast_evolution


def test_named_declarations_and_explicit_unsupported_parameters():
    m = bx.Model(bx.GEV(), parameters={'mu':bx.Latent([bx.LocalLinearTrend()]),
                                     'sigma':bx.Constant(), 'xi':bx.Constant()})
    assert m.observation.scale.mode == 'constant'
    assert bx.Model.from_dict(m.to_dict()).to_dict() == m.to_dict()
    with pytest.raises(ValueError, match='not both'):
        bx.Model(bx.Gaussian(), [bx.LocalLevel()], parameters={'mu':bx.Constant()})
    with pytest.raises(NotImplementedError, match='latent shape'):
        bx.Model(bx.GEV(), parameters={'mu':bx.Constant(),'xi':bx.Latent([bx.LocalLevel()])})
    with pytest.raises(ValueError, match='log link'):
        bx.Model(bx.Gaussian(), parameters={'mu':bx.Constant(),'sigma':bx.Latent([bx.LocalLevel()],link='identity')})


@pytest.mark.parametrize('family', ['gaussian', 'gev'])
def test_scale_slope_conditional_matches_independent_quadrature(family):
    specification = bx.StructuralScale([bx.LocalLinearTrend(level_mode='static',trend_mode='static')],
                                      bx.EvolutionPriors(innovation='normal',initial_slope_sd=.15))
    state = GaussianEvolutionState(specification, 6)
    y = np.array([.3,-.2,.7,-.4,1.1,.8])
    def likelihood(offset):
        if family == 'gaussian':
            return np.sum(norm.logpdf(y,scale=np.exp(offset)))
        return np.sum(genextreme.logpdf(y,c=.2,scale=np.exp(offset)))
    grid = np.linspace(-1.5,1.5,15001)
    density = np.exp(np.array([likelihood(b*np.arange(1,7)) for b in grid])+norm.logpdf(grid,scale=.15))
    density /= trapezoid(density,grid)
    expected = trapezoid(grid*density,grid)
    expected_second = trapezoid(grid**2*density,grid)
    rng = np.random.default_rng(17001)
    draws = []
    for i in range(2600):
        state.step(likelihood,rng)
        if i >= 300:
            draws.append(state.params_state['beta0'])
    assert abs(np.mean(draws)-expected) < .009
    assert abs(np.mean(np.square(draws))-expected_second) < .003


def test_structural_scale_forecast_propagates_level_and_slope_noise():
    specification = bx.StructuralScale([bx.LocalLinearTrend()],bx.EvolutionPriors(innovation='normal'))
    state = GaussianEvolutionState(specification, 2)
    # Stored coefficients: initial slope, signed level SD, signed slope SD.
    assert state.names == ['beta0','s_level','s_trend']
    class Saved:
        n_time = 2
        def parameter(self,key):
            if key == 'evolution.theta':
                return np.array([[.1,.2,.05]])
            return np.array([[[0.,.1],[.1,.1],[.2,.1]]])
    samples = forecast_evolution(specification,Saved(),np.zeros(6000,dtype=int),6,np.random.default_rng(6))
    h = np.arange(1,7)
    variance = .2**2*h + .05**2*h*(h-1)*(2*h-1)/6
    np.testing.assert_allclose(samples.mean(axis=0),.2+.1*h,atol=.025)
    np.testing.assert_allclose(samples.var(axis=0),variance,rtol=.065)


@pytest.mark.parametrize('innovation', ['normal','lasso','triple_gamma'])
def test_mixed_copula_scale_evolution_roundtrip_and_components(tmp_path, innovation):
    dates = pd.date_range('2000-03-01',periods=24,freq='MS')
    data = pd.DataFrame(np.random.default_rng(2).normal(size=(24,2)),index=dates,columns=['a','b'])
    def channel(name, observation, tail=None):
        return bx.Channel(name,observation,tail=tail,parameters={
            'mu':bx.Latent([bx.LocalLinearTrend(),bx.DummySeasonal(4)]),
            'sigma':bx.Latent([bx.LocalLinearTrend(),bx.DummySeasonal(4)],
                              priors=bx.EvolutionPriors(innovation=innovation))})
    model = bx.MultiSeriesModel([channel('a',bx.Gaussian()),channel('b',bx.GEV(),tail='lower')],
                                copula=bx.GaussianCopula(eta=1.))
    priors = bx.MarginalPriors({c.name:bx.fs_priors(c.family,period=4,innovation=innovation) for c in model.channels})
    options = dict(priors=priors,parameterization='fs',asis=False,
                   mcmc=bx.MCMC(draws=3,warmup=2,chains=1,seed=10,progress=False))
    fit = bx.fit(data,model,**options)
    for name in data:
        assert np.all(np.isfinite(fit.parameter_path('sigma',channel=name)))
        total = sum(fit.parameter_component_draws('sigma',component,channel=name)
                    for component in ('level','seasonal'))
        np.testing.assert_allclose(total,fit.parameter_path('sigma',channel=name,link_scale=True))
    assert fit.parameter_path('xi',channel='b').shape == (3,24)
    with pytest.raises(ValueError):
        fit.parameter_path('xi',channel='a')
    fit.save(tmp_path/'evolution.bucex')
    loaded = bx.load_fit(tmp_path/'evolution.bucex')
    assert loaded.model.to_dict() == model.to_dict()
    future = fit.forecast(12,seed=12)
    restored_future = loaded.forecast(12,seed=12)
    np.testing.assert_array_equal(future.observations,restored_future.observations)
    assert np.all(np.isfinite(future.joint_log_score(np.zeros((12,2)))))
    restarted = bx.fit(data,loaded.model,init=loaded,**options)
    assert restarted.sigma_draws(channel='b').shape == (3,24)


def test_structural_scale_simulation_rejects_missing_process_parameters():
    model = bx.Model(bx.Gaussian(scale=bx.StructuralScale([bx.LocalLevel()])),[bx.LocalLinearTrend()])
    params = {'sigma':1.,'sd.level':.1,'sd.slope':.01}
    with pytest.raises(ValueError,match='scale.sd.level'):
        bx.simulate(model,12,params,seed=10)
    simulated = bx.simulate(model,12,{**params,'scale.sd.level':.15},seed=10)
    assert np.ptp(simulated.sigma) > 0
    assert np.all(simulated.sigma > 0)
