import ast,copy,json,sys,unittest,subprocess,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'scripts'),str(ROOT/'webapp')]
from foundation_core import Parameter,Module,initialize
from _foundation_checks import evaluate_layer
from _assessment import evaluate_special
from pilot_server import RUNNER,standalone

class IndependentLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'));cls.meta=cls.b['lessons']['rebuild_linear']
    def scope(self,code=None):
        s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize)
        exec(self.b['lessons']['initialization']['reference'],s);exec(code or self.meta['reference'],s);return s
    def test_all_ten_checks(self):
        rows=evaluate_layer(self.scope(),self.meta['cases']);self.assertEqual(len(rows),10);self.assertTrue(all(r['passed'] for r in rows),rows)
    def test_current_initialize_is_really_called(self):
        wrong=self.meta['reference'].replace('initialize((C, D))','np.random.default_rng(0).normal(0,np.sqrt(2/C),(C,D))')
        self.assertNotEqual(wrong,self.meta['reference'])
        self.assertFalse(evaluate_special(self.scope(wrong),self.meta['cases'][-2])['passed'])
    def test_optional_bias_and_independent_instance_caches(self):
        ref=self.meta['reference']
        wrong=ref.replace(' if use_bias else None','')
        self.assertFalse(evaluate_special(self.scope(wrong),self.meta['cases'][-2])['passed'])
        wrong=ref.replace('self._x2','type(self)._x2')
        self.assertFalse(evaluate_special(self.scope(wrong),self.meta['cases'][-1])['passed'])
    def test_no_bias_four_dimensional_gradient_and_accumulation(self):
        s=self.scope();obj=s['Linear'](3,2,False);rng=np.random.default_rng(193);x=rng.normal(size=(2,2,3,3));g=rng.normal(size=(2,2,3,2));before=obj.W.data.copy()
        y=obj(x);dx=obj.backward(g);self.assertEqual(y.shape,g.shape);self.assertEqual(dx.shape,x.shape)
        expected=x.reshape(-1,3).T@g.reshape(-1,2);np.testing.assert_allclose(obj.W.grad,expected)
        obj(x);obj.backward(g);np.testing.assert_allclose(obj.W.grad,2*expected);obj.zero_grad();np.testing.assert_array_equal(obj.W.grad,0);np.testing.assert_array_equal(obj.W.data,before)
    def test_full_scaffold_and_generator_are_consistent(self):
        from independent_linear_lesson import apply_independent_linear_lesson
        from course_math import math,aligned
        b=copy.deepcopy(self.b);apply_independent_linear_lesson(b,math,aligned);self.assertEqual(b,self.b)
        full=self.meta['scaffold_notes']['starter'];ast.parse(full)
        cls=next(n for n in ast.parse(full).body if isinstance(n,ast.ClassDef))
        self.assertEqual([n.name for n in cls.body if isinstance(n,ast.FunctionDef)],['__init__','forward','backward'])
        self.assertTrue(all(any(isinstance(x,ast.Name) and x.id=='NotImplementedError' for x in ast.walk(n)) for n in cls.body))
        self.assertIn('动手任务清单',self.meta['learning_page']['statement']);self.assertIn('关键公式速查',self.meta['learning_page']['statement']);self.assertEqual(self.meta['task_brief']['next_use']['mode'],'contrast')
    def test_code_and_math_are_not_plain_prose(self):
        from html.parser import HTMLParser
        import re
        class Prose(HTMLParser):
            def __init__(self):super().__init__();self.skip=[];self.text=[]
            def handle_starttag(self,tag,attrs):
                if tag in ('code','pre') or any(k=='class' and 'math-' in (v or '') for k,v in attrs):self.skip.append(tag)
            def handle_endtag(self,tag):
                if self.skip and self.skip[-1]==tag:self.skip.pop()
            def handle_data(self,text):
                if not self.skip:self.text.append(text)
        parser=Prose();parser.feed(self.meta['body']+self.meta['learning_page']['statement'])
        self.assertFalse(re.search(r'(?<![A-Za-z0-9_])(initialize|Parameter|Module|forward|backward|reshape|W|G|C|D)(?![A-Za-z0-9_])',' '.join(parser.text)))
    def test_web_runner_and_local_file_use_same_ten_checks(self):
        deps=[[k,self.b['lessons'][k]['reference']] for k in self.meta['dependencies']]
        payload=dict(exercises=str(ROOT/'exercises'),kind='layer',code=self.meta['reference'],cases=self.meta['cases'],dependencies=deps)
        run=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=15)
        result=json.loads(run.stdout.strip().splitlines()[-1]);self.assertEqual(len(result['rows']),10);self.assertTrue(all(r['passed'] for r in result['rows']),result)
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
            folder=Path(folder);(folder/'learner_core.py').write_text('from foundation_core import Parameter,Module\nimport numpy as np\n'+self.b['lessons']['initialization']['reference'],encoding='utf-8')
            program=folder/'linear_program.py';program.write_text(standalone(self.meta,self.meta['reference']),encoding='utf-8')
            command='import runpy,sys;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
            run=subprocess.run([sys.executable,'-I','-c',command,str(program),str(folder),str(ROOT/'exercises')],capture_output=True,text=True,encoding='utf-8',timeout=15)
            self.assertEqual(run.returncode,0,run.stderr+run.stdout);self.assertIn('10/10',run.stdout)
if __name__=='__main__':unittest.main()
