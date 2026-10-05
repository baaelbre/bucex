import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import bucex as bx
from research.data import load_summaries
from research.configuration import CONFIG,load_config,build_model,read_config
from research.sensitivity import plan


class ResearchTests(unittest.TestCase):
    def test_complete_blocks_and_current_reference(self):
        seasonal=load_summaries()
        monthly=load_summaries(frequency='monthly')
        self.assertEqual(len(seasonal),538)
        self.assertEqual(len(monthly),1616)
        self.assertEqual(str(seasonal.index[0].date()),'1892-05-31')
        self.assertEqual(str(seasonal.index[-1].date()),'2026-08-31')
        self.assertEqual(seasonal.loc['2019-08-31','TXx'],39.7)
        c=load_config(); m=build_model(c)
        self.assertEqual(m.pooling.slope.sd,.002)
        self.assertEqual(m.channels[0].model.priors.initial_slope.sd,.01)
        self.assertEqual(m.channels[0].model.priors.initial_seasonal.sd,10)

    def test_five_analysis_models_fit(self):
        for name in ['main','monthly','private','constant_dispersion','monthly_constant_dispersion']:
            c=load_config(CONFIG/(name+'.json'))
            data=load_summaries(frequency=c['data']['frequency']).iloc[:24]
            f=bx.fit(data,model=build_model(c),mcmc=bx.MCMC(draws=2,warmup=1,chains=1,progress=False))
            self.assertEqual(len(f.channels),6)
            self.assertTrue(all(np.isfinite(v.states).all() for v in f.channels.values()))
            self.assertEqual(bool(f.shared_scales),c['pooling']=='shared')
            self.assertEqual('sigma[0]' in f['TXx'].parameters,c['model']['seasonal_scale'])

    def test_monthly_calibration_matches_30_year_variance(self):
        a=load_config();b=load_config(CONFIG/'monthly.json')
        for c in [a,b]:
            h=30*c['model']['steps_per_year']
            c['variance_level']=h*c['priors']['level_sd']**2
            c['variance_slope']=h*(h-1)*(2*h-1)/6*c['priors']['slope_sd']**2
        self.assertAlmostEqual(a['variance_level'],b['variance_level'])
        self.assertAlmostEqual(a['variance_slope'],b['variance_slope'])

    def test_sensitivity_plan_and_config_failures(self):
        variants=plan()
        self.assertEqual(len(variants),23)
        self.assertEqual(sum(c['name'].startswith('level_') for c in variants),9)
        for c in variants: build_model(c)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps({'extends':'bad.json'}))
            with self.assertRaises(ValueError): read_config(p)
            p.write_text(json.dumps({'extends':str(CONFIG/'main.json'),'priros':{}}))
            with self.assertRaises(ValueError): load_config(p)


if __name__=='__main__': unittest.main()
