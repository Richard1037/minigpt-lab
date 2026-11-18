import copy,json,sys,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'scripts'),str(ROOT/'webapp')]
from foundation_core import Parameter,Module,initialize
from _foundation_checks import evaluate_layer
from _transformer_checks import evaluate_project
from _assessment import evaluate_special
from pilot_server import RUNNER,standalone


class Part45Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
    def scope(self,key,code=None):
        s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize,__name__='student')
        for dep in self.book['lessons'][key]['dependencies']:exec(self.book['lessons'][dep]['reference'],s)
        exec(code or self.book['lessons'][key]['reference'],s);return s
    def test_references_all_current_cases(self):
        for key,count in [('layer_linear',10),('foundation_capstone',12),('t_softmax',9),('t_loss',8)]:
            meta=self.book['lessons'][key];rows=(evaluate_layer if meta['kind']=='layer' else evaluate_project)(self.scope(key),meta['cases'])
            self.assertEqual(len(rows),count);self.assertTrue(all(r['passed'] for r in rows),rows)
    def test_stage_reuses_single_linear_and_retains_old_history(self):
        stage=next(s for s in self.book['stages'] if s['title'].startswith('第四'))
        tasks=[t for mod in stage['modules'] for t in mod['tasks']]
        self.assertEqual(tasks[-1],'foundation_capstone');self.assertNotIn('rebuild_linear',tasks)
        self.assertTrue(any('rebuild_linear' in mod['tasks'] for s in self.book['stages'] if s.get('optional') for mod in s['modules']))
        self.assertIn('layer_linear',self.book['lessons']['foundation_capstone']['dependencies'])
        self.assertNotIn('rebuild_linear',self.book['lessons']['foundation_capstone']['dependencies'])
    def test_branch_must_use_current_submodules_in_correct_order(self):
        m=self.book['lessons']['foundation_capstone'];case=next(c for c in m['cases'] if c.get('assessment')=='branch_composition')
        for old,new in [('self.mlp(self.norm(x))','self.mlp(x)'),('self.norm.backward(self.mlp.backward(g))','self.mlp.backward(self.norm.backward(g))')]:
            bad=m['reference'].replace(old,new);self.assertNotEqual(bad,m['reference'])
            self.assertFalse(evaluate_special(self.scope('foundation_capstone',bad),case)['passed'])
    def test_gate_really_runs_personal_embedding(self):
        s=self.scope('foundation_capstone');s['Embedding'].forward=lambda self,x:np.zeros((*x.shape,self.weight.data.shape[-1]))
        case=next(c for c in self.book['lessons']['foundation_capstone']['cases'] if c.get('assessment')=='foundation_gate' and c['input']['lesson']=='layer_embedding')
        self.assertFalse(evaluate_special(s,case)['passed'])
    def test_probability_alias_bugs_are_rejected(self):
        for key,old,new in [('t_softmax','return self._p.copy()','return self._p'),('t_loss','self._p.copy().reshape','self._p.reshape'),('t_loss','targets.copy()','targets'),('t_loss','z = logits - logits.max(axis=-1, keepdims=True)','logits -= logits.max(axis=-1, keepdims=True)\n        z = logits')]:
            m=self.book['lessons'][key];bad=m['reference'].replace(old,new);self.assertNotEqual(bad,m['reference'])
            case=next(c for c in m['cases'] if c.get('assessment')=='probability_contract')
            self.assertFalse(evaluate_special(self.scope(key,bad),case)['passed'])
    def test_lesson_contracts_and_generator(self):
        from part45_lessons import apply_part45
        from course_math import math,aligned
        b=copy.deepcopy(self.book);apply_part45(b,math,aligned);self.assertEqual(b,self.book)
        for key in ['foundation_capstone','t_softmax','t_loss','layer_linear']:
            m=self.book['lessons'][key]
            for name in ['输入格式','输出格式','状态变化','动手任务清单','关键公式速查']:self.assertIn(name,m['learning_page']['statement'])
            self.assertIn('__main__',m['scaffold_notes']['starter'])
        self.assertIn('targets[..., None]',self.book['lessons']['t_loss']['body'])
        self.assertIn('公共分母',self.book['lessons']['t_softmax']['body'])
        import re,html
        sample=self.book['lessons']['layer_linear']['learning_page']['sample_note']
        for tex in re.findall(r'data-tex="([^"]+)"',sample):self.assertFalse(re.search(r'\\\\[A-Za-z]',html.unescape(tex)))
    def test_actual_web_runner_and_local_imports(self):
        for key in ['foundation_capstone','t_softmax','t_loss']:
            m=self.book['lessons'][key];deps=[[d,self.book['lessons'][d]['reference']] for d in m['dependencies']]
            payload=dict(exercises=str(ROOT/'exercises'),kind=m['kind'],code=m['reference'],dependencies=deps,cases=m['cases'])
            run=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=25)
            result=json.loads(run.stdout.strip().splitlines()[-1]);self.assertTrue(all(r['passed'] for r in result['rows']),result)
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
            folder=Path(folder)
            for key in self.book['lessons']['foundation_capstone']['dependencies']:
                meta=self.book['lessons'][key];imports='import numpy as np\n'
                if key not in ['infrastructure','initialization']:imports+='from learner_core import Parameter,Module,initialize\n'
                for d in meta.get('dependencies',[]):
                    if d in ['infrastructure','initialization']:continue
                    dm=self.book['lessons'][d];imports+='from '+dm['file'][:-3]+' import '+', '.join(dm.get('exports',[dm.get('class','')]))+'\n'
                (folder/meta['file']).write_text(imports+meta['reference'],encoding='utf-8')
            (folder/'learner_core.py').write_text('from layer_infrastructure import Parameter,Module\nfrom initialize_weights import initialize\n',encoding='utf-8')
            m=self.book['lessons']['foundation_capstone'];program=folder/'entry.py';program.write_text(standalone(m,m['reference']),encoding='utf-8')
            boot='import runpy,sys;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
            run=subprocess.run([sys.executable,'-I','-c',boot,str(program),str(folder),str(ROOT/'exercises')],capture_output=True,text=True,encoding='utf-8',timeout=25)
            self.assertEqual(run.returncode,0,run.stderr+run.stdout);self.assertIn('12/12',run.stdout)
    def test_code_identifiers_are_not_plain_prose(self):
        import re
        from html.parser import HTMLParser
        class Prose(HTMLParser):
            def __init__(self):super().__init__();self.skip=[];self.text=[]
            def handle_starttag(self,tag,attrs):
                if tag in ('pre','code') or any(k=='class' and 'math-' in (v or '') for k,v in attrs):self.skip.append(tag)
            def handle_endtag(self,tag):
                if self.skip and self.skip[-1]==tag:self.skip.pop()
            def handle_data(self,value):
                if not self.skip:self.text.append(value)
        for key in ['foundation_capstone','t_softmax','t_loss']:
            m=self.book['lessons'][key];parser=Prose();parser.feed(m['body']+m['learning_page']['statement'])
            self.assertFalse(re.search(r'(?<![A-Za-z0-9_])(forward|backward|targets|logits|logp|self\._p|zero_grad|take_along_axis)(?![A-Za-z0-9_])',' '.join(parser.text)),key)


if __name__=='__main__':unittest.main()
