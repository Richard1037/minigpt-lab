"""初始化实际实现、依赖接线和旧选择题草稿的迁移检查。"""
import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'webapp')]
from _initialization_checks import evaluate_initialize
from pilot_server import RUNNER,standalone


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        cls.meta=cls.book['lessons']['initialization']

    def execute(self,code,meta=None,dependencies=None):
        meta=meta or self.meta
        payload=dict(exercises=str(ROOT/'exercises'),code=code,kind=meta['kind'],cases=meta['cases'],dependencies=dependencies or [])
        proc=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),capture_output=True,text=True,encoding='utf-8',timeout=15)
        self.assertEqual(proc.returncode,0,proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_reference_passes_all_cases_and_starter_is_unfinished(self):
        good=self.execute(self.meta['reference']);self.assertTrue(all(r['passed'] for r in good['rows']),good)
        bad=self.execute(self.meta['starter']);self.assertFalse(any(r['passed'] for r in bad['rows']))
        self.assertTrue(all('NotImplementedError' in r['errorText'] for r in bad['rows']))

    def test_wrong_width_rng_dtype_and_return_type_are_rejected(self):
        wrong=[
            lambda shape,rng=None: (rng or np.random.default_rng(0)).standard_normal(shape),
            lambda shape,rng=None: (rng or np.random.default_rng(0)).standard_normal(shape)*np.sqrt(2/shape[1]),
            lambda shape,rng=None: np.random.default_rng(0).standard_normal(shape)*np.sqrt(2/shape[0]),
            lambda shape,rng=None: ((rng or np.random.default_rng(0)).standard_normal(shape)*np.sqrt(2/shape[0])).astype(np.float32),
            lambda shape,rng=None: ((rng or np.random.default_rng(0)).standard_normal(shape)*np.sqrt(2/shape[0])).tolist(),
        ]
        for fn in wrong:
            with self.subTest(fn=fn):self.assertFalse(all(r['passed'] for r in evaluate_initialize(fn,self.meta['cases'])))

    def test_rng_state_check_rejects_clone_and_extra_draw(self):
        def clone(shape,rng=None):
            rng=copy.deepcopy(rng) if rng is not None else np.random.default_rng(0)
            return rng.standard_normal(shape)*np.sqrt(2/shape[0])
        def extra(shape,rng=None):
            rng=rng if rng is not None else np.random.default_rng(0)
            out=rng.standard_normal(shape)*np.sqrt(2/shape[0]);rng.standard_normal();return out
        case=self.meta['cases'][0]
        for fn in (clone,extra):
            rows=evaluate_initialize(fn,[case]);self.assertFalse(rows[0]['passed']);self.assertIn('状态不符',' '.join(rows[0]['diagnostics']))

    def test_empty_submission_cannot_use_injected_support_initializer(self):
        for source in ('pass','from foundation_core import initialize'):
            result=self.execute(source);self.assertIn('error',result)

    def test_linear_uses_the_initializer_dependency_even_with_full_imports(self):
        meta=self.book['lessons']['layer_linear']
        deps=[(d,standalone(self.book['lessons'][d],self.book['lessons'][d]['reference'])) for d in meta['dependencies']]
        good=self.execute(standalone(meta,meta['reference']),meta,deps);self.assertTrue(all(r['passed'] for r in good['rows']),good)
        bad_deps=[(d,self.meta['starter'] if d=='initialization' else code) for d,code in deps]
        bad=self.execute(standalone(meta,meta['reference']),meta,bad_deps)
        self.assertFalse(any(r['passed'] for r in bad['rows']));self.assertIn('NotImplementedError',bad['rows'][0]['errorText'])

    def test_local_wrapper_and_learner_core_use_personal_function(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as tmp:
            folder=Path(tmp)
            (folder/'initialize_weights.py').write_text(standalone(self.meta,self.meta['reference']),encoding='utf-8')
            (folder/'layer_infrastructure.py').write_text(self.book['lessons']['infrastructure']['reference'].replace('class Parameter:', 'import numpy as np\nclass Parameter:',1),encoding='utf-8')
            code='import sys;sys.path[:0]='+repr([str(folder),str(ROOT/'exercises')])+";import learner_core,initialize_weights;assert learner_core.initialize is initialize_weights.initialize;from _initialization_checks import run_local;run_local(learner_core.initialize)"
            proc=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,encoding='utf-8',timeout=15)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)

    def test_old_choices_are_preserved_until_explicit_user_actions(self):
        script=r'''const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const src=fs.readFileSync('webapp/app.js','utf8'),book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
const storage=new Map();const ctx={book,JSON,localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)}};
vm.createContext(ctx);vm.runInContext(src.slice(src.indexOf('function get('),src.indexOf('let book,current')),ctx);
vm.runInContext(src.slice(src.indexOf('function isLegacyAnswers('),src.indexOf('function explainCall(')),ctx);
const old={version:JSON.stringify(book.lessons.initialization.legacy_assessment.questions),code:'[1,2]',data:{rows:[{passed:true},{passed:true}]},when:'old'};
storage.set('pilot-initialization','[1,2]');storage.set('pilot-result-initialization',JSON.stringify(old));const before=[...storage];
assert(ctx.draft('initialization').includes('def initialize'));assert.deepEqual([...storage],before);
ctx.preserveLegacy('initialization');assert.equal(storage.get('pilot-initialization'),'[1,2]');assert.equal(storage.get('pilot-result-initialization'),JSON.stringify(old));
const key='pilot-history-initialization-concept-v1',archive=storage.get(key);assert.equal(JSON.parse(archive).result.when,'old');
const nodes={editorStatus:{},completion:{},results:{}};
Object.assign(ctx,{current:'initialization',m:()=>book.lessons.initialization,navigation:()=>{},status:()=>({kind:'warn',text:'旧选择题记录保留'}),running:new Set(),$:id=>nodes[id]});
vm.runInContext(src.slice(src.indexOf('function refresh(){'),src.indexOf('function paintCases(){')),ctx);ctx.refresh();
assert(nodes.results.textContent.includes('6 项'));assert(!nodes.results.textContent.includes('2/6'));assert(nodes.completion.hidden);
storage.set('pilot-initialization','def initialize(shape, rng=None): pass');assert.equal(ctx.draft('initialization'),'def initialize(shape, rng=None): pass');ctx.preserveLegacy('initialization');assert.equal(storage.get(key),archive);
'''
        proc=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
