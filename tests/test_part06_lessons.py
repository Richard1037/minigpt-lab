import ast,copy,json,sys,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'exercises'),str(ROOT/'webapp')]
from foundation_core import Parameter,Module,initialize
from _checks import evaluate
from _transformer_checks import evaluate_project
from _assessment import evaluate_special
from pilot_server import RUNNER,standalone
from part06_lessons import IDS,apply_part06,sync_part06_imports
from course_math import math,aligned
from attention_structure import PRACTICE


class Part06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
    def scope(self,key,replace=None):
        s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize,__name__='student')
        m=self.book['lessons'][key]
        for dep in m.get('dependencies',[]):exec(self.book['lessons'][dep]['reference'],s)
        exec(replace or m['reference'],s);return s
    def test_all_public_references_and_original_expectations(self):
        for key in IDS:
            m=self.book['lessons'][key];s=self.scope(key)
            rows=evaluate(s['solve'],m['cases']) if m['kind']=='array' else evaluate_project(s,m['cases'])
            self.assertTrue(all(v['passed'] for v in rows),(key,[v for v in rows if not v['passed']]))
    def test_single_block_optional_boundaries_and_personal_reuse(self):
        main=[t for s in self.book['stages'] if not s.get('optional') for mod in s['modules'] for t in mod['tasks']]
        self.assertIn('t_block',main)
        for key in ['rebuild_block','t_dropout','t_layernorm',*PRACTICE]:self.assertNotIn(key,main)
        sixth=next(s for s in self.book['stages'] if s['title'].startswith('第六'))
        self.assertEqual([t for mod in sixth['modules'] for t in mod['tasks']],['t_attention','t_mha','t_block'])
        for key in main:
            self.assertFalse(set(PRACTICE).intersection(self.book['lessons'][key].get('dependencies',[])))
        s=self.scope('t_block');obj=s['Block'](6,3,9)
        self.assertIs(type(obj._ffn),s['MLPBranch']);self.assertIs(obj.norm2,obj._ffn.norm)
        self.assertIs(type(obj.attn.qkv),s['Linear']);self.assertIs(type(obj.mlp.fc2),s['Linear'])
        self.assertEqual(len(obj.parameters()),10)
    def test_reject_missing_residual_branch_bypass_and_cache_alias(self):
        changes=[('t_block','g1 = g + self._ffn.backward(g)','g1 = self._ffn.backward(g)'),('t_block','self._ffn(x1)','self.mlp(self.norm2(x1))'),('t_attention','q.copy(), k.copy(), v.copy()','q, k, v')]
        for key,old,new in changes:
            m=self.book['lessons'][key];bad=m['reference'].replace(old,new);self.assertNotEqual(bad,m['reference'])
            rows=evaluate_project(self.scope(key,bad),m['cases']);self.assertTrue(any(not v['passed'] for v in rows),(key,old))
    def test_block_encapsulation_names_do_not_change_gradient_contract(self):
        m=self.book['lessons']['t_block']
        public=m['reference'].replace('        self.norm2 = self._ffn.norm\n','').replace('        self.mlp = self._ffn.mlp\n','').replace('self._ffn','self.ffn')
        renamed=public.replace('self.ffn','self.feature_branch').replace('self.norm1','self.input_norm').replace('self.attn','self.context_reader')
        for code in [public,renamed]:
            rows=evaluate_project(self.scope('t_block',code),m['cases'])
            self.assertTrue(all(r['passed'] for r in rows),[r for r in rows if not r['passed']])
        bad=public.replace('g1 = g + self.ffn.backward(g)','g1 = self.ffn.backward(g)')
        rows=evaluate_project(self.scope('t_block',bad),m['cases'])
        self.assertTrue(any(not r['passed'] for r in rows))
        self.assertFalse(any('缺少此项' in e or '多余返回项' in e for r in rows for e in r.get('diagnostics',[])))
    def test_complete_teaching_and_persistent_generation(self):
        b=copy.deepcopy(self.book);apply_part06(b,math,aligned);self.assertEqual(b,self.book)
        for key in IDS[:-1]:
            m=self.book['lessons'][key]
            for title in ['题目描述','输入格式','输出格式','状态变化','动手任务清单','关键公式速查']:self.assertIn(title,m['learning_page']['statement'])
            ast.parse(m['scaffold_notes']['starter']);self.assertIn('__main__',m['scaffold_notes']['starter'])
            self.assertEqual(m['derivation'],'')
    def test_web_runner_and_local_imports(self):
        for key in ['t_attention','t_mha','t_block','t_dropout','t_layernorm']:
            m=self.book['lessons'][key]
            payload=dict(exercises=str(ROOT/'exercises'),kind=m['kind'],code=m['reference'],dependencies=[[d,self.book['lessons'][d]['reference']] for d in m['dependencies']],cases=m['cases'])
            run=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=30)
            data=json.loads(run.stdout.strip().splitlines()[-1]);self.assertTrue(all(r['passed'] for r in data['rows']),data)
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
            folder=Path(folder)
            for key in [*self.book['lessons']['t_block']['dependencies'],'t_block']:
                m=self.book['lessons'][key];source=standalone(m,m['reference'])
                # 主模块与所有真实依赖在隔离文件中使用参考代码，不触碰个人文件。
                (folder/m['file']).write_text(source,encoding='utf-8')
            (folder/'learner_core.py').write_text('from layer_infrastructure import Parameter,Module\nfrom initialize_weights import initialize\n',encoding='utf-8')
            boot='import sys,runpy;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
            run=subprocess.run([sys.executable,'-I','-c',boot,str(folder/'t_block.py'),str(folder),str(ROOT/'exercises')],text=True,encoding='utf-8',capture_output=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout[-1200:]+run.stderr);self.assertIn('8/8',run.stdout)
            public=self.book['lessons']['t_block']['reference'].replace('        self.norm2 = self._ffn.norm\n','').replace('        self.mlp = self._ffn.mlp\n','').replace('self._ffn','self.ffn')
            imports='import numpy as np\nfrom learner_core import Parameter,Module,initialize\nfrom layer_rmsnorm import RMSNorm\nfrom t_mha import MultiHeadAttention\nfrom foundation_capstone import MLPBranch\n'
            tail='\nif __name__ == "__main__":\n from _transformer_checks import run_local\n run_local("t_block",globals())\n'
            (folder/'t_block.py').write_text(imports+public+tail,encoding='utf-8')
            run=subprocess.run([sys.executable,'-I','-c',boot,str(folder/'t_block.py'),str(folder),str(ROOT/'exercises')],text=True,encoding='utf-8',capture_output=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout[-1600:]+run.stderr);self.assertIn('8/8',run.stdout)
    def test_future_gpt_compatibility(self):
        # 下游正文未精修；当前 Block 的公开路径仍应通过原 GPT 数值期望。
        for key in ['t_gpt','t_project','rebuild_gpt']:
            m=self.book['lessons'][key];rows=evaluate_project(self.scope(key),m['cases'])
            self.assertTrue(all(v['passed'] for v in rows),(key,[v for v in rows if not v['passed']]))
    def test_original_step_cases_are_checked_inside_complete_classes(self):
        for source in PRACTICE:
            target='t_mha' if source=='att_merge' else 't_attention'
            moved=[c for c in self.book['lessons'][target]['cases'] if c.get('assessment')=='attention_component' and c['input']['source']==source]
            original=self.book['lessons'][source]['cases'];self.assertEqual(len(moved),len(original))
            for a,b in zip(moved,original):self.assertEqual(a['input']['args'],b['args']);self.assertEqual(a['expected'],b['expected'])
        for key,old,new in [('t_attention',' / self._scale',''),('t_attention','return self._a @ v','return np.zeros_like(self._a @ v)'),('t_mha','y.transpose(0,2,1,3).reshape(B,T,C)','y.reshape(B,T,C)')]:
            m=self.book['lessons'][key];bad=m['reference'].replace(old,new);self.assertNotEqual(bad,m['reference'])
            cases=[c for c in m['cases'] if c.get('assessment')=='attention_component']
            rows=evaluate_project(self.scope(key,bad),cases);self.assertTrue(any(not r['passed'] for r in rows),(key,old))
    def test_import_migration_preserves_all_personal_methods(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
            folder=Path(folder);p=folder/'rebuild_gpt.py'
            code='from rebuild_block import Block\nclass GPT:\n    def forward(self,x):\n        # personal work\n        return x+3\n'
            p.write_text(code,encoding='utf-8');b=copy.deepcopy(self.book)
            self.assertEqual(sync_part06_imports(b,folder),['rebuild_gpt'])
            self.assertEqual(p.read_text(encoding='utf-8'),code.replace('from rebuild_block','from t_block'))
            self.assertEqual(next((folder/'backups').iterdir()).read_text(encoding='utf-8'),code)
            self.assertEqual(sync_part06_imports(b,folder),[])

if __name__=='__main__':unittest.main()
