"""Embedding 教学、实际构造交付与个人代码/成绩保护。"""
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
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'webapp'),str(ROOT/'scripts')]
from foundation_core import Parameter,Module
from _foundation_checks import evaluate_layer
from _assessment import evaluate_special
from pilot_server import RUNNER,standalone


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        cls.meta=cls.book['lessons']['layer_embedding']

    def scope(self, source=None):
        scope=dict(np=np,Parameter=Parameter,Module=Module)
        exec(source or self.meta['reference'],scope)
        return scope

    def test_all_checks_and_repeated_row_sources(self):
        scope=self.scope()
        rows=evaluate_layer(scope,self.meta['cases'])
        self.assertEqual(len(rows),9)
        self.assertTrue(all(r['passed'] for r in rows),rows)
        case=self.meta['cases'][0];obj=scope['Embedding'](*case['init'])
        obj.weight.data[...] = case['parameters']['weight']
        ids=np.array(case['X']);g=np.array(case['G'])
        y=obj.forward(ids);self.assertEqual(y.shape,(2,3,3))
        np.testing.assert_array_equal(y[0,0],y[0,2]);np.testing.assert_array_equal(y[0,0],y[1,1])
        self.assertIsNone(obj.backward(g))
        np.testing.assert_array_equal(obj.weight.grad[1],g[0,0]+g[0,2]+g[1,1])
        np.testing.assert_array_equal(obj.weight.grad[1],[18,21,24])

    def test_new_construction_check_covers_previously_unchecked_small_vocab(self):
        wrong=self.meta['reference'].replace('np.random.default_rng(0).normal(0, .02, (V,C))','np.random.default_rng(0).normal(0, .02 if V >= 3 else .04, (V,C))')
        self.assertNotEqual(wrong,self.meta['reference'])
        rows=evaluate_layer(self.scope(wrong),self.meta['cases'])
        self.assertTrue(all(r['passed'] for r in rows[:-1]),rows)
        self.assertFalse(rows[-1]['passed'])
        self.assertIn('初值',rows[-1]['diagnostics'][0])

    def test_duplicate_index_and_cache_alias_bugs_are_rejected(self):
        ref=self.meta['reference']
        wrong=ref.replace('np.add.at(self.weight.grad, self._ids.reshape(-1), g.reshape(-1, g.shape[-1]))','self.weight.grad[self._ids] += g')
        rows=evaluate_layer(self.scope(wrong),self.meta['cases'][:1])
        self.assertFalse(rows[0]['passed'])
        cache=next(c for c in self.meta['cases'] if c.get('assessment')=='cache')
        self.assertFalse(evaluate_special(self.scope(ref.replace('np.asarray(ids).copy()','np.asarray(ids)')),cache)['passed'])

    def test_three_methods_are_todo_and_standalone_scaffold_executes(self):
        cls=next(n for n in ast.parse(self.meta['starter']).body if isinstance(n,ast.ClassDef))
        for method in cls.body:
            if isinstance(method,ast.FunctionDef):
                self.assertTrue(any(isinstance(n,ast.Name) and n.id=='NotImplementedError' for n in ast.walk(method)),method.name)
        deps=[['infrastructure',self.book['lessons']['infrastructure']['reference']]]
        payload=dict(exercises=str(ROOT/'exercises'),kind='layer',cases=self.meta['cases'],dependencies=deps,
                     code=self.meta['scaffold_notes']['starter'])
        p=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=12)
        out=json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(len(out['rows']),9,out)
        self.assertFalse(any(row['passed'] for row in out['rows']))

    def test_custom_local_comments_are_not_mistaken_for_untouched_template(self):
        from embedding_lesson import apply_embedding_lesson
        from course_math import math,aligned
        book=copy.deepcopy(self.book)
        meta=book['lessons']['layer_embedding']
        meta['local_code']+='\n# 我的本地学习笔记\n'
        apply_embedding_lesson(book,math,aligned)
        self.assertEqual(meta['scaffold_notes']['source'],'')
        self.assertTrue(meta['local_code'].endswith('# 我的本地学习笔记\n'))

    def test_local_complete_file_shares_same_checks(self):
        # 临时目录提供已完成的个人接口，既有 exercises 文件不写入。
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as folder:
            folder=Path(folder)
            (folder/'learner_core.py').write_text('from foundation_core import Parameter,Module,initialize\n',encoding='utf-8')
            source=standalone(self.meta,self.meta['reference'])
            path=folder/'embedding.py';path.write_text(source,encoding='utf-8')
            script='import sys,runpy;sys.path[:0]=sys.argv[2:];runpy.run_path(sys.argv[1],run_name="__main__")'
            p=subprocess.run([sys.executable,'-I','-c',script,str(path),str(folder),str(ROOT/'exercises')],text=True,encoding='utf-8',capture_output=True,timeout=12)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertIn('9/9 项通过',p.stdout)

    def test_default_scaffold_and_revision_archive_preserve_personal_data(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const src=fs.readFileSync('webapp/app.js','utf8'),book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
const data=new Map(),context={localStorage:{getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v)},bookSeed:book};
vm.createContext(context);vm.runInContext(src.slice(0,src.indexOf('function go(')),context);
vm.runInContext('book=bookSeed;current="layer_embedding"',context);
const meta=book.lessons.layer_embedding;
assert.equal(context.draft('layer_embedding'),meta.local_code);
const personal='MY PERSONAL IMPLEMENTATION';data.set('pilot-layer_embedding',personal);
assert.equal(context.draft('layer_embedding'),personal);
const prior={code:personal,version:JSON.stringify([2,meta.cases.slice(0,8)]),data:{rows:Array(8).fill({passed:true})},when:'previous submission'};
data.set('pilot-result-layer_embedding',JSON.stringify(prior));const before=new Map(data);
assert.equal(JSON.stringify(context.priorAssessmentRecord('layer_embedding')),JSON.stringify(prior));assert.deepEqual(data,before);
context.preservePriorAssessment('layer_embedding');
assert.equal(data.get('pilot-result-layer_embedding'),JSON.stringify(prior));
assert.equal(data.get('pilot-layer_embedding'),personal);
assert.equal(data.get('pilot-history-layer_embedding-revision-2'),JSON.stringify(prior));
data.set('pilot-result-layer_embedding',JSON.stringify({version:JSON.stringify([3,meta.cases]),data:{rows:[]}}));
assert.equal(context.priorAssessmentRecord('layer_embedding').code,personal);
const first={...prior,version:JSON.stringify([1,meta.cases.slice(0,2)])};
data.set('pilot-result-layer_embedding',JSON.stringify(first));context.preservePriorAssessment('layer_embedding');
assert.equal(data.get('pilot-history-layer_embedding-revision-1'),JSON.stringify(first));
assert.equal(data.get('pilot-history-layer_embedding-revision-2'),JSON.stringify(prior));
const size=data.size;context.preservePriorAssessment('linear');assert.equal(data.size,size);
console.log('drafts and old scores preserved');
'''
        p=subprocess.run(['node','-e',script],cwd=ROOT,text=True,encoding='utf-8',capture_output=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_actual_gpt_uses_personal_embedding(self):
        meta=self.book['lessons']['t_gpt'];deps=[]
        for id in meta['dependencies']:
            source=self.book['lessons'][id]['reference']
            if id=='layer_embedding':
                source=source.replace('self._ids = np.asarray(ids).copy()', 'raise RuntimeError("PERSONAL_EMBEDDING_USED")')
            deps.append([id,standalone(self.book['lessons'][id],source)])
        self.assertTrue(any(id=='layer_embedding' for id,_ in deps))
        payload=dict(exercises=str(ROOT/'exercises'),code=standalone(meta,meta['reference']),kind=meta['kind'],cases=meta['cases'][:1],dependencies=deps)
        p=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=15)
        out=json.loads(p.stdout.strip().splitlines()[-1])
        self.assertIn('PERSONAL_EMBEDDING_USED',json.dumps(out))
        self.assertFalse(any(row['passed'] for row in out.get('rows',[])))


if __name__=='__main__':unittest.main()
