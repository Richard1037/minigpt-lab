"""初始化接口升级：新旧接口、真实个人依赖及两条执行路径。"""
import json,sys,unittest,subprocess,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'exercises'),str(ROOT/'webapp')]
from foundation_core import Parameter,Module,initialize
from _transformer_checks import evaluate_project
from pilot_server import RUNNER,standalone

class UpgradeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'));cls.meta=cls.book['lessons']['t_init_upgrade']
 def scope(self,source=None,personal=False):
  s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize,__name__='student')
  for key in self.meta['dependencies']:
   m=self.book['lessons'][key];exec((ROOT/'exercises'/m['file']).read_text(encoding='utf-8') if personal else m['reference'],s)
  if not personal:exec(source or self.meta['reference'],s)
  return s
 def test_all_upgrade_cases_and_unchanged_numeric_samples(self):
  rows=evaluate_project(self.scope(),self.meta['cases']);self.assertTrue(all(r['passed'] for r in rows),rows)
  old=json.loads((ROOT/'.check-tmp/init-upgrade-before.json').read_text(encoding='utf-8'))['lessons']['t_gpt']['cases']
  new=self.book['lessons']['t_gpt']['cases'];self.assertEqual(old,new[:len(old)])
 def test_missing_passthrough_wrong_exit_duplicate_draw_rejected(self):
  src=self.meta['reference']
  for a,b in [('self.fc2 = Linear(hidden, C, rng=rng, init_std=init_std, init_scale=init_scale)','self.fc2 = Linear(hidden, C, rng=rng, init_std=init_std)'),('self.qkv = Linear(C, 3 * C, rng=rng, init_std=init_std)','self.qkv = Linear(C, 3 * C, rng=rng, init_std=init_std, init_scale=init_scale)'),('self.W = Parameter(values * init_scale)','rng.normal(0, 1, (C,D)) if rng is not None else None\n        self.W = Parameter(values * init_scale)')]:
   bad=src.replace(a,b);self.assertNotEqual(bad,src)
   self.assertTrue(any(not r['passed'] for r in evaluate_project(self.scope(bad),self.meta['cases'])),a)
 def test_personal_modules_not_filled_by_check_driver(self):
  driver=(ROOT/'exercises/t_init_upgrade.py').read_text(encoding='utf-8');self.assertNotIn('class ',driver)
  s=self.scope(personal=True);exec(driver,s)
  # 个人文件尚未升级时必须报告接口缺口，不能用参考补全使其通过。
  if 'init_std' not in __import__('inspect').signature(s['Linear']).parameters:
   rows=evaluate_project(s,self.meta['cases']);self.assertFalse(rows[1]['passed']);self.assertIn('Linear',str(rows[1]))
 def test_web_reference_and_real_local_reference_imports(self):
  m=self.meta;p=dict(exercises=str(ROOT/'exercises'),kind=m['kind'],code=m['reference'],dependencies=[[d,self.book['lessons'][d]['reference']] for d in m['dependencies']],cases=m['cases'])
  proc=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(p),text=True,encoding='utf-8',capture_output=True,timeout=20)
  result=json.loads(proc.stdout.strip().splitlines()[-1]);self.assertTrue(all(r['passed'] for r in result['rows']),result)
  with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
   folder=Path(folder)
   for k in [*m['dependencies'],'t_init_upgrade']:
    meta=self.book['lessons'][k];(folder/meta['file']).write_text(standalone(meta,meta['reference']),encoding='utf-8')
   (folder/'learner_core.py').write_text('from layer_infrastructure import Parameter,Module\nfrom initialize_weights import initialize\n',encoding='utf-8')
   boot='import sys,runpy;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
   proc=subprocess.run([sys.executable,'-I','-c',boot,str(folder/m['file']),str(folder),str(ROOT/'exercises')],text=True,encoding='utf-8',capture_output=True,timeout=20)
   self.assertEqual(proc.returncode,0,proc.stdout[-1400:]+proc.stderr);self.assertIn('5/5',proc.stdout)

if __name__=='__main__':unittest.main()
