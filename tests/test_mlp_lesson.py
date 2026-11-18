import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'scripts')]
from foundation_core import Parameter,Module,initialize
from _foundation_checks import evaluate_layer
from _assessment import evaluate_special

class MLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'));cls.meta=cls.b['lessons']['layer_mlp']
    def scope(self,code=None):
        s=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize)
        for key in ['initialization','layer_linear','layer_gelu']:exec(self.b['lessons'][key]['reference'],s)
        exec(code or self.meta['reference'],s);return s
    def test_all_ten_checks(self):
        rows=evaluate_layer(self.scope(),self.meta['cases']);self.assertEqual(len(rows),10);self.assertTrue(all(r['passed'] for r in rows),rows)
    def test_constructor_and_wrong_backward_are_rejected(self):
        ref=self.meta['reference'];case=self.meta['cases'][-1]
        bad=ref.replace('Linear(hidden, C)','Linear(C, C)')
        self.assertFalse(evaluate_special(self.scope(bad),case)['passed'])
        bad=ref.replace('self.act.backward(self.fc2.backward(g))','self.fc2.backward(self.act.backward(g))')
        self.assertFalse(evaluate_special(self.scope(bad),case)['passed'])
    def test_recreating_submodule_is_rejected(self):
        bad=self.meta['reference'].replace('return self.fc2(self.act(self.fc1(x)))','self.fc1 = Linear(x.shape[-1], self.fc1.W.data.shape[1])\n        return self.fc2(self.act(self.fc1(x)))')
        self.assertFalse(evaluate_special(self.scope(bad),self.meta['cases'][-1])['passed'])
    def test_persistent_source_and_empty_constructor_scaffold(self):
        from mlp_lesson import apply_mlp_lesson
        from course_math import math,aligned
        b=copy.deepcopy(self.b);apply_mlp_lesson(b,math,aligned);self.assertEqual(b,self.b)
        self.assertIn('请实现子层构造',self.meta['starter']);self.assertIn('动手任务清单',self.meta['learning_page']['statement']);self.assertIn('关键公式速查',self.meta['learning_page']['statement'])
    def test_identifiers_are_not_left_in_plain_prose(self):
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
        self.assertFalse(re.search(r'(?<![A-Za-z0-9_])(fc1|fc2|tanh|forward|backward)(?![A-Za-z0-9_])',' '.join(parser.text)))
if __name__=='__main__':unittest.main()
