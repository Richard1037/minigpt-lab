"""Part07完整交付、等价个人Block、初始化和两路共享梯度回归。"""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'exercises'),str(ROOT/'webapp')]
from foundation_core import Parameter,Module,initialize
from _transformer_checks import evaluate_project
from pilot_server import RUNNER,standalone
from part07_lessons import apply_part07
from course_math import math,aligned


class Part07Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
 def scope(self,key,block=None,code=None):
  s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize,__name__='student')
  for dep in self.book['lessons'][key]['dependencies']:
   source=self.book['lessons'][dep]['reference'];source=source.replace(self.book['lessons'][dep]['reference'][self.book['lessons'][dep]['reference'].index('class Block('):],block) if dep=='t_init_upgrade' and block else source;exec(source,s)
  exec(code or self.book['lessons'][key]['reference'],s);return s
 def test_all_cases_and_persistent_generation(self):
  for key in ['t_data','t_gpt']:
   m=self.book['lessons'][key];rows=evaluate_project(self.scope(key),m['cases'])
   self.assertTrue(all(r['passed'] for r in rows),[r for r in rows if not r['passed']])
   for title in ['题目描述','输入格式','输出格式','状态变化','动手任务清单','关键公式速查']:self.assertIn(title,m['learning_page']['statement'])
   self.assertIn('核心总结',m['body']);self.assertIn('__main__',m['scaffold_notes']['starter'])
  b=copy.deepcopy(self.book);apply_part07(b,math,aligned);self.assertEqual(b,self.book)
 def test_public_branch_and_fully_renamed_block_preserve_seed_and_gradients(self):
  code=self.book['lessons']['t_init_upgrade']['reference'].split('class Block(')[1];code='class Block('+code.replace('        self.norm2 = self._ffn.norm\n','').replace('        self.mlp = self._ffn.mlp\n','').replace('self._ffn','self.ffn')
  for block in [code,code.replace('self.ffn','self.feature_branch').replace('self.attn','self.reader').replace('self.norm1','self.input_norm')]:
   rows=evaluate_project(self.scope('t_gpt',block),self.book['lessons']['t_gpt']['cases'])
   self.assertTrue(all(r['passed'] for r in rows),[r for r in rows if not r['passed']])
 def test_wrong_initialization_and_shared_paths_are_rejected(self):
  m=self.book['lessons']['t_gpt']
  for old,new in [('scale = 1 / np.sqrt(2 * layers)','scale = 1 / np.sqrt(layers)'),('self.token.weight.grad +=','self.token.weight.grad ='),('self.position.backward(dx.sum(axis=0))','self.position.backward(dx[0])'),('for block in reversed(self.blocks):','for block in self.blocks:')]:
   bad=m['reference'].replace(old,new);self.assertNotEqual(bad,m['reference'])
   rows=evaluate_project(self.scope('t_gpt',code=bad),m['cases']);self.assertTrue(any(not r['passed'] for r in rows),old)
  bad=self.book['lessons']['t_data']['reference'].replace('ids[s+1:s+T+1]','ids[s:s+T]')
  self.assertTrue(any(not r['passed'] for r in evaluate_project(self.scope('t_data',code=bad),self.book['lessons']['t_data']['cases'])))
 def test_web_and_real_local_module_imports(self):
  for key in ['t_data','t_gpt']:
   m=self.book['lessons'][key];payload=dict(exercises=str(ROOT/'exercises'),kind=m['kind'],code=m['reference'],dependencies=[[d,self.book['lessons'][d]['reference']] for d in m['dependencies']],cases=m['cases'])
   run=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=30)
   data=json.loads(run.stdout.strip().splitlines()[-1]);self.assertTrue(all(r['passed'] for r in data['rows']),data)
  with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
   folder=Path(folder)
   for key in [*self.book['lessons']['t_gpt']['dependencies'],'t_gpt','t_data']:
    m=self.book['lessons'][key];(folder/m['file']).write_text(standalone(m,m['reference']),encoding='utf-8')
   (folder/'learner_core.py').write_text('from layer_infrastructure import Parameter,Module\nfrom initialize_weights import initialize\n',encoding='utf-8')
   boot='import sys,runpy;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
   for key,total in [('t_data',6),('t_gpt',9)]:
    run=subprocess.run([sys.executable,'-I','-c',boot,str(folder/self.book['lessons'][key]['file']),str(folder),str(ROOT/'exercises')],text=True,encoding='utf-8',capture_output=True,timeout=30)
    self.assertEqual(run.returncode,0,run.stdout[-1800:]+run.stderr);self.assertIn(f'{total}/{total}',run.stdout)

if __name__=='__main__':unittest.main()
