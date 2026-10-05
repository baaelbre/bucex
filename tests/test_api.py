import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import bucex as bx
from bucex.diagnostics import rhat,ess_bulk,posterior_checks


class APITests(unittest.TestCase):
    def setUp(self):
        self.y=np.random.default_rng(7).normal(1, .5, 16)
        self.model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend(),bx.DummySeasonal(4,'static')])
        self.config=bx.MCMC(draws=4,warmup=2,chains=2,progress=False,seed=19)

    def test_public_exports_and_python310_syntax(self):
        self.assertEqual(bx.__version__,'1.0.0')
        self.assertFalse(hasattr(bx,'GaussianCopula'))
        root=Path(__file__).resolve().parents[1]
        for directory in ['src','research','examples','tests']:
            for path in (root/directory).rglob('*.py'):
                ast.parse(path.read_text(),filename=str(path),feature_version=(3,10))

    def test_multiseries_pooling_and_roundtrip(self):
        channels=[bx.Channel('a',self.model),bx.Channel('b',bx.Model(bx.GEV('lower'),self.model.components))]
        model=bx.MultiSeriesModel(channels,bx.Pooling(level=bx.HalfNormal(.1),slope=bx.HalfNormal(.002)))
        dates=pd.date_range('2000-02-29',periods=16,freq=pd.offsets.QuarterEnd(startingMonth=2))
        y=pd.DataFrame({'a':self.y,'b':-self.y},index=dates)
        f=bx.fit(y,model=model,mcmc=self.config)
        self.assertEqual(set(f.shared_scales),{'level','slope'})
        self.assertNotIn('initial_slope',f.shared_scales)
        self.assertTrue(np.isfinite(f['b'].states).all())
        with tempfile.TemporaryDirectory() as d:
            path=f.save(Path(d)/'result.bucex'); restored=bx.load(path)
            np.testing.assert_array_equal(f['b'].states,restored['b'].states)
            np.testing.assert_array_equal(f.shared_scales['level'],restored.shared_scales['level'])
            self.assertTrue(restored['a'].index.equals(dates))
        pred=f.predict(4,draws=15)
        self.assertEqual(str(pred.index[0].date()),'2004-02-29')
        check=posterior_checks(f,draws=10)
        self.assertTrue(check.ks_bayesian_p.between(0,1).all())

    def test_parallel_matches_serial_and_separate_jobs(self):
        serial=bx.fit(self.y,model=self.model,mcmc=self.config)
        parallel=bx.fit(self.y,model=self.model,mcmc=replace(self.config,workers=2))
        np.testing.assert_array_equal(serial.channel().states,parallel.channel().states)
        singles=[bx.fit(self.y,model=self.model,mcmc=replace(self.config,chains=1,chain_ids=(i,))) for i in range(2)]
        combined=bx.combine_fits(*singles)
        np.testing.assert_array_equal(serial.channel().states,combined.channel().states)
        with self.assertRaises(ValueError): bx.combine_fits(singles[0],singles[0])
        changed=bx.fit(self.y+1,model=self.model,mcmc=replace(self.config,chains=1,chain_ids=(2,)))
        with self.assertRaises(ValueError): bx.combine_fits(singles[0],changed)

    def test_plotting_accepts_native_objects(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        f=bx.fit(self.y,model=self.model,mcmc=self.config)
        for kind in ['level','slope','seasonal','cycle','normal_qq','pit','acf','risk','return_level','prior_posterior','trace']:
            ax=bx.plot(f,type=kind,threshold=2,phase=1,parameter='variance.level')
            self.assertEqual(ax.get_title(),'')
            plt.close(ax.figure)
        p=f.predict(4,draws=20)
        ax=bx.plot(p,type='forecast',history=f)
        self.assertGreaterEqual(len(ax.lines),3)
        plt.close('all')

    def test_reject_bad_data_and_unsupported_priors(self):
        with self.assertRaises(ValueError): bx.fit([1,np.nan,3],model=self.model,mcmc=self.config)
        with self.assertRaises(ValueError): bx.compile_model(bx.Model(bx.Gaussian(),[bx.LocalLevel(),bx.LocalLinearTrend()]), 5)
        with self.assertRaises(ValueError): bx.Priors(level=bx.Normal(1,1))
        with self.assertRaises(ValueError): bx.Priors(variance=bx.Fixed(0))
        with self.assertRaises(TypeError): bx.Pooling(level=bx.Normal())
        dates=pd.to_datetime(['2001-01-01','2001-02-01','2001-04-01'])
        with self.assertRaises(ValueError): bx.fit([1,2,3],model=self.model,dates=dates,mcmc=self.config)

    def test_different_channel_structures(self):
        model=bx.MultiSeriesModel([bx.Channel('a',bx.Model(bx.Gaussian(),[bx.LocalLevel()])),
                                   bx.Channel('b',self.model)],bx.Pooling(level=bx.HalfNormal(.1)))
        data={'a':self.y,'b':self.y+1}
        with self.assertRaises(ValueError): bx.fit(data,model=model,mcmc=self.config)
        result=bx.fit(data,model=model,steps_per_year=4,mcmc=self.config)
        self.assertEqual(result['a'].states.shape[-1],1)
        self.assertEqual(result['b'].states.shape[-1],5)
        self.assertEqual(result.predict(4,draws=5).y['b'].shape,(5,4))

    def test_diagnostics_identify_chain_shift(self):
        x=np.random.default_rng(41).normal(size=(4,1500))
        self.assertLess(rhat(x),1.01)
        self.assertGreater(ess_bulk(x),3500)
        x[0]+=3
        self.assertGreater(rhat(x),1.1)
        self.assertLess(ess_bulk(x),100)
        self.assertTrue(np.isnan(rhat(np.ones((4,20)))))

    def test_window_risk_and_extreme_aggregation(self):
        f=bx.fit(self.y,model=self.model,mcmc=self.config)
        p=f.predict(8,draws=30)
        risks=p.risk(2)
        np.testing.assert_allclose(bx.window_risk(p,2),1-np.prod(1-risks,axis=1))
        np.testing.assert_allclose(bx.block_extremes(p,blocks=4),p.y['y'].reshape(30,2,4).max(2))
        with self.assertRaises(ValueError): bx.block_extremes(p,blocks=3)
        table=bx.coverage(p,np.zeros(8))
        self.assertEqual(len(table),3)


if __name__=='__main__': unittest.main()
