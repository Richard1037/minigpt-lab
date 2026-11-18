"""Part08精修：保留两题，验证状态边界、错误拒绝与网页/本地两路。"""
import copy,json,sys,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'exercises'),str(ROOT/'webapp')]
from foundation_core import Parameter,Module,initialize
from _transformer_checks import evaluate_project
from pilot_server import RUNNER,standalone,Handler
from part08_lessons import apply_part08
from course_math import math,aligned


class Part08Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
 def scope(self,key,code=None):
  s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize,__name__='student')
  for dep in self.book['lessons'][key]['dependencies']:exec(self.book['lessons'][dep]['reference'],s)
  exec(code or self.book['lessons'][key]['reference'],s);return s
 def test_cases_and_lesson_contract(self):
  self.assertEqual(next(s for s in self.book['stages'] if s['title'].startswith('第八'))['modules'][0]['tasks'],['t_optim','t_train'])
  for key in ('t_optim','t_train'):
   m=self.book['lessons'][key];rows=evaluate_project(self.scope(key),m['cases'])
   self.assertTrue(all(r['passed'] for r in rows),[r for r in rows if not r['passed']]);self.assertEqual(len(rows),6)
   for title in ('题目描述','输入格式','输出格式','状态变化','动手任务清单','关键公式速查'):self.assertIn(title,m['learning_page']['statement'])
   self.assertIn('核心总结',m['body']);compile(m['scaffold_notes']['code'],key,'exec')
  b=copy.deepcopy(self.book);apply_part08(b,math,aligned);self.assertEqual(b,self.book)
 def test_wrong_optimizer_states_and_shared_updates_rejected(self):
  m=self.book['lessons']['t_optim'];ref=m['reference']
  for old,new in [('self.t += 1','self.t = 1'),('self.params = parameters','self.params = parameters[1:]'),('if p.data.ndim >= 2:','if True:'),('p.data -=','p.data = p.data -')]:
   bad=ref.replace(old,new);self.assertNotEqual(bad,ref)
   rows=evaluate_project(self.scope('t_optim',bad),m['cases']);self.assertTrue(any(not r['passed'] for r in rows),old)
 def test_wrong_train_order_and_reset_rejected(self):
  m=self.book['lessons']['t_train'];ref=m['reference']
  for old,new in [('model.zero_grad()','pass'),('clip_grad(model.parameters(),1.)','pass'),('opt.step()','opt = AdamW(model.parameters(),lr=lr,decay=.01)\n        opt.step()')]:
   # Source spacing is part of neither the teaching contract nor student grading.
   bad=ref.replace(old,new);self.assertNotEqual(bad,ref)
   rows=evaluate_project(self.scope('t_train',bad),m['cases']);self.assertTrue(any(not r['passed'] for r in rows),old)
 def test_web_and_real_local_imports(self):
  for key in ('t_optim','t_train'):
   m=self.book['lessons'][key];payload=dict(exercises=str(ROOT/'exercises'),kind=m['kind'],code=m['reference'],dependencies=[[d,self.book['lessons'][d]['reference']] for d in m['dependencies']],cases=m['cases'])
   run=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=40)
   data=json.loads(run.stdout.strip().splitlines()[-1]);self.assertTrue(all(r['passed'] for r in data['rows']),data)
  with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
   folder=Path(folder)
   for key in [*self.book['lessons']['t_train']['dependencies'],'t_train']:
    m=self.book['lessons'][key];(folder/m['file']).write_text(standalone(m,m['reference']),encoding='utf-8')
   (folder/'learner_core.py').write_text('from layer_infrastructure import Parameter,Module\nfrom initialize_weights import initialize\n',encoding='utf-8')
   boot='import sys,runpy;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
   for key in ('t_optim','t_train'):
    run=subprocess.run([sys.executable,'-I','-c',boot,str(folder/self.book['lessons'][key]['file']),str(folder),str(ROOT/'exercises')],text=True,encoding='utf-8',capture_output=True,timeout=40)
    self.assertEqual(run.returncode,0,run.stdout[-2200:]+run.stderr);self.assertIn('6/6',run.stdout)
 def test_formula_stylesheet_delivery(self):
  handler=Handler.__new__(Handler);handler.path='/course-katex.css';sent=[]
  handler.send=lambda data,status=200,kind='application/json':sent.append((data,status,kind))
  handler.do_GET();css,status,kind=sent[0]
  self.assertEqual(status,200);self.assertEqual(kind,'text/css; charset=utf-8')
  self.assertIn('.katex-mathml',css);self.assertIn('url(/vendor/katex/fonts/',css)
  self.assertNotIn('url(fonts/',css)
  self.assertIn('href="/course-katex.css"',(ROOT/'webapp/pilot.html').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
