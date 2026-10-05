"""Statistical invariants, not merely implementation-shaped assertions."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
import bucex as bx
from bucex.inference.paths import laplace_mh
from bucex.inference.smoothers import sample_gaussian as ffbs_gaussian_1d_tvR, smooth_gaussian as gaussian_smoother_mean_1d_tvR
from bucex.inference._shared import _half_normal_gig_log_multiplier


class InferenceTests(unittest.TestCase):
    def test_gev_static_location_matches_likelihood_quadrature(self):
        obs=bx.GEV()
        y=np.array([-.4, .2, 1.5, .7, -.1, 1.2, .3, .6])
        p={'sigma':1.,'xi':-.15}
        def density(mu):
            return np.exp(obs.logpdf(y,mu,p).sum()+norm.logpdf(mu,scale=2))
        lo,hi=-4.,8.
        mass=quad(density,lo,hi,epsabs=1e-12)[0]
        expected=quad(lambda mu:mu*density(mu),lo,hi,epsabs=1e-12)[0]/mass
        model=bx.Model(obs,[bx.LocalLevel('static')],bx.Priors(
            initial_level=bx.Normal(0,2),variance=bx.Fixed(1),shape=bx.Fixed(-.15)))
        fit=bx.fit(y,model=model,mcmc=bx.MCMC(draws=1800,warmup=50,chains=1,progress=False,seed=49))
        values=fit.channel().parameters['initial_level'].ravel()
        from bucex.diagnostics import ess_bulk
        mcse=values.std()/np.sqrt(ess_bulk(values[None,:]))
        self.assertLess(abs(values.mean()-expected),max(.035,5*mcse))

    def test_known_gev_scale_support_safe_static_initialization(self):
        model=bx.Model(bx.GEV(),[bx.LocalLevel('static')],bx.Priors(
            initial_level=bx.Normal(0,10),variance=bx.Fixed(.01),shape=bx.Fixed(-.2)))
        fit=bx.fit(np.array([1.,2.,3.,4.,8.]),model=model,
                   mcmc=bx.MCMC(draws=2,warmup=1,chains=1,progress=False))
        self.assertTrue(np.isfinite(fit.channel().metrics['log_likelihood']).all())

    def test_gaussian_static_level_has_analytic_posterior(self):
        y = np.array([1., 2., 1.5, .5, 3., 2.])
        m = bx.Model(bx.Gaussian(), [bx.LocalLevel('static')],
                     bx.Priors(initial_level=bx.Normal(0, 2), variance=bx.Fixed(1)))
        f = bx.fit(y, model=m, mcmc=bx.MCMC(draws=1600, warmup=5, chains=1, progress=False, seed=81))
        values = f.channel().parameters['initial_level'].ravel()
        variance = 1/(len(y)+.25)
        expected = variance*y.sum()
        self.assertLess(abs(values.mean()-expected), 4*np.sqrt(variance/len(values)))
        self.assertLess(abs(values.var()-variance), .025)
        np.testing.assert_array_equal(f.channel().path('level'), np.broadcast_to(values[None,:,None], (1,len(values),len(y))))
        np.testing.assert_array_equal(f.channel().parameters['variance.level'], 0)

    def test_gaussian_smoother_matches_dense_conditioning(self):
        y=np.array([1., -.2, .7, 1.8]); q=.16; r=.5
        prior=q*np.minimum.outer(np.arange(1,5),np.arange(1,5))
        mean=prior@np.linalg.solve(prior+r*np.eye(4),y)
        got=gaussian_smoother_mean_1d_tvR(y,np.ones((1,1)),np.array([[q]]),np.ones(1),np.full(4,r))
        np.testing.assert_allclose(got[1:,0],mean,atol=1e-12)
        covariance=prior-prior@np.linalg.solve(prior+r*np.eye(4),prior)
        rng=np.random.default_rng(4)
        values=np.stack([ffbs_gaussian_1d_tvR(y,np.ones((1,1)),np.array([[q]]),np.ones(1),np.full(4,r),rng=rng)[1:,0] for _ in range(2000)])
        np.testing.assert_allclose(values.mean(0),mean,atol=.04)
        np.testing.assert_allclose(np.cov(values.T),covariance,atol=.025)

    def test_laplace_correction_constant_for_gaussian_likelihood(self):
        model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend(),bx.DummySeasonal(4)])
        compiled=bx.compile_model(model, 20)
        theta=np.zeros(len(compiled.priors))
        for g in compiled.groups:
            theta[g.coefficient]={'level':.1, 'slope':.01, 'seasonal':.05}[g.name]
        y=np.random.default_rng(3).normal(size=20)
        current=np.zeros((21,compiled.ncp_dim))
        F,Q=compiled.ncp_transition,compiled.ncp_covariance
        path, metric=laplace_mh(y,model.observation,{'sigma':np.full(20,.4)},F,Q,
            compiled.path_design(theta),compiled.offset(theta),current,bx.Laplace(mh_steps=6),np.random.default_rng(8))
        self.assertEqual(metric['path_acceptance'],1.)
        residual=path[1:]-path[:-1]@F.T
        np.testing.assert_array_equal(residual[:,np.diag(Q)==0],0)

    def test_gev_path_mh_matches_independent_quadrature(self):
        obs=bx.GEV(); y=np.array([2.]); q=.5
        params={'sigma':np.ones(1),'xi':-.2}
        def density(x):
            return np.exp(obs.logpdf(y[0],x,params).item()+norm.logpdf(x,scale=np.sqrt(q)))
        mass=quad(density,-3,8,epsabs=1e-12)[0]
        expected=quad(lambda x:x*density(x),-3,8,epsabs=1e-12)[0]/mass
        expected2=quad(lambda x:x*x*density(x),-3,8,epsabs=1e-12)[0]/mass
        rng=np.random.default_rng(887); current=np.zeros((2,1)); values=[]
        for i in range(1600):
            current,metrics=laplace_mh(y,obs,params,np.eye(1),np.array([[q]]),
                np.ones((1,1)),np.zeros(1),current,bx.Laplace(),rng)
            if i>=100: values.append(current[-1,0])
        values=np.asarray(values)
        self.assertLess(abs(values.mean()-expected),.065)
        self.assertLess(abs(values.var()-(expected2-expected**2)),.055)

    def test_gev_derivatives_and_reflection(self):
        y=np.array([-.2,.5,1.4]);eta=np.array([.1,.2,.3]);sigma=np.array([.8,1.1,1.3]);h=1e-4
        for tail in ['upper','lower']:
            obs=bx.GEV(tail)
            for xi in [0.,1e-9,-1e-9,.15,-.2]:
                p={'sigma':sigma,'xi':xi}
                minus=obs.logpdf(y,eta-h,p);at=obs.logpdf(y,eta,p);plus=obs.logpdf(y,eta+h,p)
                np.testing.assert_allclose(obs.grad_eta(y,eta,p),(plus-minus)/(2*h),atol=1e-7)
                np.testing.assert_allclose(obs.hess_eta(y,eta,p),(plus-2*at+minus)/h**2,atol=1e-6)
                np.testing.assert_allclose(obs.cdf(obs.ppf([.1,.5,.9],eta,p),eta,p),[.1,.5,.9],atol=1e-12)
                np.testing.assert_allclose(obs.cdf(y,eta,p)+obs.sf(y,eta,p),1,atol=1e-14)
        self.assertEqual(bx.GEV().logpdf(10,0,{'sigma':1,'xi':-.2}),-np.inf)
        self.assertEqual(bx.GEV('lower').logpdf(-10,0,{'sigma':1,'xi':-.2}),-np.inf)

    def test_gig_agrees_with_independent_quadrature(self):
        for count,relative in [(1,.03),(6,1.),(6,1e-6)]:
            anchor=.1;values=anchor*relative*np.linspace(.4,1.2,count)
            r=np.linalg.norm(values/anchor)
            mode=.5*np.log(2*r*r/(np.hypot(count-1,2*r)+count-1))
            def logp(u):
                tau=anchor*np.exp(u)
                return norm.logpdf(values,scale=tau).sum()+norm.logpdf(tau,scale=anchor)+u
            center=logp(mode);density=lambda u:np.exp(logp(u)-center)
            lower,upper=mode-20,mode+20
            mass=quad(density,lower,upper)[0]
            median=brentq(lambda u:quad(density,lower,u)[0]/mass-.5,lower,upper)
            rng=np.random.default_rng(55+count)
            samples=np.array([_half_normal_gig_log_multiplier(values,anchor=anchor,rng=rng) for _ in range(1500)])
            self.assertLess(abs(np.mean(samples<=median)-.5),.05)
        tiny=1e-190*np.arange(1.,7.)
        draw=_half_normal_gig_log_multiplier(tiny,anchor=.1,rng=np.random.default_rng(1))
        self.assertTrue(np.isfinite(draw))
        with self.assertRaises(ValueError):
            _half_normal_gig_log_multiplier(np.zeros(6),anchor=.1,rng=np.random.default_rng(1))

    def test_orthonormal_initial_cycle_prior(self):
        model=bx.Model(bx.Gaussian(),[bx.LocalLevel('static'),bx.DummySeasonal(4,'static')],bx.Priors(initial_seasonal=bx.Normal(0,2)))
        samples=bx.prior_samples(model,draws=30000,seed=2)
        effects=np.stack([samples[f'y.initial_seasonal[{i}]'] for i in range(4)],1)
        np.testing.assert_allclose(effects.sum(1),0,atol=1e-13)
        np.testing.assert_allclose(np.cov(effects.T),4*(np.eye(4)-np.ones((4,4))/4),atol=.06)

    def test_static_and_fixed_components_remain_fixed(self):
        p=bx.Priors(initial_level=bx.Fixed(4), initial_slope=bx.Fixed(.2),initial_seasonal=bx.Fixed(0),variance=bx.Fixed(1))
        model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend('static','static'),bx.DummySeasonal(4,'static')],p)
        f=bx.fit(np.linspace(4,6,12),model=model,mcmc=bx.MCMC(draws=2,warmup=0,chains=1,progress=False))
        np.testing.assert_allclose(f.channel().path('level')[0,0],4+.2*np.arange(1,13))
        pred=f.predict(4,draws=8)
        np.testing.assert_allclose(pred.level['y'],np.broadcast_to(4+.2*np.arange(13,17),(8,4)))
        np.testing.assert_array_equal(pred.seasonal['y'],0)

    def test_forecast_innovation_variance_matches_local_trend_formula(self):
        priors=bx.Priors(initial_level=bx.Fixed(0),initial_slope=bx.Fixed(.01),
            level=bx.Fixed(.1),slope=bx.Fixed(.02),variance=bx.Fixed(1))
        model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend()],priors)
        fit=bx.fit(np.linspace(0,1,10),model=model,
            mcmc=bx.MCMC(draws=1,warmup=0,chains=1,progress=False,seed=84))
        prediction=fit.predict(12,draws=12000,seed=98)
        last=fit.channel().states[0,0,-1]
        expected_mean=last[0]+12*last[1]
        expected_variance=12*.1**2+sum(i*i for i in range(12))*.02**2
        values=prediction.level['y'][:,-1]
        self.assertLess(abs(values.mean()-expected_mean),.025)
        self.assertLess(abs(values.var()/expected_variance-1),.04)

    def test_prior_physical_horizon_formula(self):
        p=bx.Priors(initial_level=bx.Fixed(0),initial_slope=bx.Fixed(0), level=bx.Fixed(.1),slope=bx.Fixed(.02), variance=bx.Fixed(1))
        model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend()],p)
        sim=bx.prior_predictive(model,12,draws=20000,seed=4)
        exact=12*.1**2+sum(i*i for i in range(12))*.02**2
        self.assertLess(abs(sim.level['y'][:,-1].var()/exact-1),.04)


if __name__=='__main__':
    unittest.main()
