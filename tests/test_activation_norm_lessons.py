import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'scripts')]
from foundation_core import Parameter,Module
from _foundation_checks import evaluate_layer
from _assessment import evaluate_special

class ActivationNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
    def scope(self,key,code=None):
        s=dict(np=np,Parameter=Parameter,Module=Module);exec(code or self.b['lessons'][key]['reference'],s);return s
    def test_reference_all_cases(self):
        for k in ['layer_gelu','layer_rmsnorm']:
            rows=evaluate_layer(self.scope(k),self.b['lessons'][k]['cases'])
            self.assertEqual(len(rows),9);self.assertTrue(all(x['passed'] for x in rows),rows)
    def test_constructor_rejects_wrong_eps_and_initial_value(self):
        k='layer_rmsnorm';m=self.b['lessons'][k];case=m['cases'][-1]
        for wrong in [m['reference'].replace('self.eps = eps','self.eps = 1e-5'),m['reference'].replace('np.ones(C)','np.zeros(C)')]:
            self.assertFalse(evaluate_special(self.scope(k,wrong),case)['passed'])
    def test_actual_gradient_paths_and_whole_batch(self):
        x=np.array([[[1.,2.,3.],[-2.,1.,0.]],[[3.,-1.,2.],[0.,0.,0.]]]);g=np.arange(12.).reshape(x.shape)/7
        for key in ['layer_gelu','layer_rmsnorm']:
            scope=self.scope(key);obj=scope['GELU']() if key=='layer_gelu' else scope['RMSNorm'](3,.01)
            obj.forward(x);dx=obj.backward(g)
            for idx in np.ndindex(x.shape):
                a=x.copy();b=x.copy();a[idx]+=1e-5;b[idx]-=1e-5
                numeric=np.sum((obj.forward(a)-obj.forward(b))*g)/2e-5
                self.assertAlmostEqual(dx[idx],numeric,places=5)
    def test_source_idempotent_and_complete_scaffold(self):
        from activation_norm_lessons import apply_activation_norm_lessons
        from course_math import math,aligned
        b=copy.deepcopy(self.b);apply_activation_norm_lessons(b,math,aligned);self.assertEqual(b,self.b)
        for key in ['layer_gelu','layer_rmsnorm']:
            self.assertIn('run_local',b['lessons'][key]['scaffold_notes']['starter'])
            self.assertEqual(b['lessons'][key]['derivation'],'')
if __name__=='__main__':unittest.main()
