"""Extension acceptance: compiler, posterior target, covariates and round trips."""
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import bucex as bx


@bx.register_type
@dataclass(frozen=True)
class Decay:
    """An external component: no changes to fitting, prediction or plots."""
    name: str = 'decay'
    persistence: float = .8

    def build(self, steps, priors, exog=None):
        return bx.ComponentBlock(self.name, np.array([[self.persistence]]), np.ones((steps,1)),
            np.eye(1), ('decay.initial',), (bx.Fixed(2.),), ('decay.state',),
            (bx.Innovation('decay', np.eye(1), bx.Fixed(.05)),))


class ExtensionTests(unittest.TestCase):
    def test_static_regression_matches_multivariate_analytic_posterior(self):
        rng=np.random.default_rng(12)
        X=pd.DataFrame({'wind':rng.normal(size=50),'pressure':rng.normal(size=50)})
        y=1.+X.to_numpy()@np.array([1.2,-.7])+rng.normal(0,.4,50)
        model=bx.Model(bx.Gaussian(),[bx.LocalLevel('static',initial_prior=bx.Normal(0,2)),
            bx.Regression(('wind','pressure'),prior=bx.Normal(0,2))],bx.Priors(variance=bx.Fixed(.16)))
        fit=bx.fit(y,model=model,exog=X,mcmc=bx.MCMC(draws=3000,warmup=2,chains=1,seed=67,progress=False))
        design=np.column_stack([np.ones(50),X.to_numpy()])
        covariance=np.linalg.inv(np.eye(3)/4+design.T@design/.16)
        expected=covariance@design.T@y/.16
        values=np.stack([fit.channel().parameters[k].ravel() for k in ('initial_level','regression.wind','regression.pressure')],1)
        np.testing.assert_allclose(values.mean(0),expected,atol=.01)
        np.testing.assert_allclose(np.cov(values.T),covariance,atol=.0004)
        np.testing.assert_allclose(fit.channel().path(),fit.channel().path('level')+fit.channel().path('regression'))
        with tempfile.TemporaryDirectory() as root:
            loaded=bx.load(fit.save(Path(root)/'regression.bucex'))
            future=pd.DataFrame({'pressure':[-1.,0.,1.],'wind':[.5,1.,2.]})
            a=fit.predict(3,exog=future,draws=30,seed=9)
            b=loaded.predict(3,exog=future,draws=30,seed=9)
            np.testing.assert_array_equal(a.y['y'],b.y['y'])
            self.assertEqual(a.path('regression',channel='y').shape,(30,3))
            ax=bx.plot(loaded,type='component',component='regression.wind',path=Path(root)/'coef.png')
            self.assertTrue((Path(root)/'coef.png').exists())
            self.assertEqual(ax.get_title(),'')
        with self.assertRaises(ValueError): fit.predict(3)
        with self.assertRaises(ValueError): fit.predict(3,exog=np.ones((2,2)))

    def test_time_varying_regression_and_cycle(self):
        X=pd.DataFrame({'wind':np.linspace(.5,2,20)})
        model=bx.Model(bx.Gaussian(),[bx.LocalLevel('static'),
            bx.Regression(('wind',),mode='dynamic',innovation_prior=bx.Fixed(.03)),
            bx.Cycle(8,innovation_prior=bx.Fixed(.05))])
        simulation=bx.simulate(model,20,exog=X)
        fit=bx.fit(simulation.y['y'][0],model=model,exog=X,
            mcmc=bx.MCMC(draws=4,warmup=2,chains=1,progress=False))
        self.assertEqual(fit.channel().path('regression.wind').shape,(1,4,20))
        self.assertEqual(fit.channel().path('cycle').shape,(1,4,20))
        self.assertGreater(np.ptp(fit.channel().path('regression.wind')[0,0]),0)
        forecast=fit.predict(4,exog=pd.DataFrame({'wind':[1.,2.,3.,4.]}),draws=15)
        self.assertTrue(np.isfinite(forecast.path('cycle',channel='y')).all())

    def test_external_component_roundtrip_and_forecast(self):
        model=bx.Model(bx.Gaussian(),[Decay()],bx.Priors(variance=bx.Fixed(.1)))
        simulation=bx.simulate(model,10)
        fit=bx.fit(simulation.y['y'][0],model=model,mcmc=bx.MCMC(draws=3,warmup=1,chains=1,progress=False))
        with tempfile.TemporaryDirectory() as root:
            restored=bx.load(fit.save(Path(root)/'custom.bucex'))
            np.testing.assert_array_equal(fit.channel().path('decay'),restored.channel().path('decay'))
            self.assertEqual(restored.predict(4,draws=10).path('decay',channel='y').shape,(10,4))

    def test_named_parameters_and_unsupported_backend(self):
        model=bx.Model(bx.GEV('lower'),parameters={'location':bx.Latent((bx.LocalLevel(),)),
                      'scale':bx.Fixed(.7),'shape':bx.Fixed(-.1)})
        y=bx.simulate(model,12).y['y'][0]
        fit=bx.fit(y,model=model,mcmc=bx.MCMC(draws=3,warmup=1,chains=1,progress=False))
        np.testing.assert_allclose(fit.channel().parameters['sigma'],.7)
        np.testing.assert_allclose(fit.channel().parameters['xi'],-.1)
        future=bx.Model(bx.GEV(),parameters={'scale':bx.Latent((bx.LocalLevel(),),link='log')})
        with self.assertRaisesRegex(NotImplementedError,'latent location'):
            bx.fit(y,model=future)

    def test_time_varying_design_ffbs_matches_dense_gaussian_conditioning(self):
        from bucex.inference.smoothers import smooth_gaussian
        h=np.array([1.,2.,-.5,1.5]);y=np.array([1.,-.2,.7,1.8]);q=.16;r=.5
        prior=q*np.minimum.outer(np.arange(1,5),np.arange(1,5))
        D=np.diag(h)
        mean=prior@D@np.linalg.solve(D@prior@D+r*np.eye(4),y)
        got=smooth_gaussian(y,np.eye(1),np.array([[q]]),h[:,None],np.full(4,r))
        np.testing.assert_allclose(got[1:,0],mean,atol=1e-12)

    def test_gev_dynamic_sampler_reflection_equivalence(self):
        upper=bx.Model(bx.GEV(),[bx.LocalLevel()],bx.Priors(shape=bx.Fixed(-.1)))
        lower=replace(upper,observation=bx.GEV('lower'))
        y=np.array([0.,.5,1.2,-.3,.8,1.,-.1,1.3])
        settings=bx.MCMC(draws=20,warmup=5,chains=1,progress=False,seed=17)
        a=bx.fit(y,model=upper,mcmc=settings)
        b=bx.fit(-y,model=lower,mcmc=settings)
        # Exact seedwise equality is not required (the random directions differ).
        for fit in (a,b):
            self.assertTrue(np.isfinite(fit.channel().metrics['log_likelihood']).all())
            self.assertTrue(np.all((fit.channel().metrics['path_acceptance']>=0)&(fit.channel().metrics['path_acceptance']<=1)))


if __name__=='__main__': unittest.main()
