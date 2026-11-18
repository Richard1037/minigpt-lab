"""第一、二、四部分的端到端测评验证，不写入个人草稿。"""
import json
import subprocess
import sys
import unittest
import threading
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import urllib.request
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'exercises'))
sys.path.insert(0,str(ROOT/'webapp'))
import pilot_server
from pilot_server import RUNNER, standalone


class CourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))

    def check(self,id,code,deps=None):
        meta=self.book['lessons'][id]
        payload=dict(exercises=str(ROOT/'exercises'),code=code,kind=meta['kind'],cases=meta['cases'],dependencies=list((deps or {}).items()))
        proc=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=12)
        self.assertEqual(proc.returncode,0,proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def references(self):
        return {id:meta['reference'] for id,meta in self.book['lessons'].items() if meta.get('reference')}

    def test_all_reference_lessons(self):
        refs=self.references()
        for id,meta in self.book['lessons'].items():
            if meta['kind']=='concept':continue
            with self.subTest(id=id):
                result=self.check(id,refs[id],{dep:refs[dep] for dep in meta.get('dependencies',[])})
                self.assertIn('rows',result,result)
                self.assertTrue(all(row['passed'] for row in result['rows']),result)

    def test_all_skeletons_remain_unfinished(self):
        refs=self.references()
        for id,meta in self.book['lessons'].items():
            if meta['kind']=='concept':continue
            with self.subTest(id=id):
                result=self.check(id,meta['starter'],{dep:refs[dep] for dep in meta.get('dependencies',[])})
                self.assertFalse(all(row['passed'] for row in result.get('rows',[])) if result.get('rows') else False)

    def test_missing_gradient_accumulation_is_rejected(self):
        source=self.references()['layer_linear'].replace('self.W.grad +=','self.W.grad[...] =').replace('self.b.grad +=','self.b.grad[...] =')
        result=self.check('layer_linear',source)
        self.assertFalse(result['rows'][1]['passed'])

    def test_repeated_embedding_index_bug_is_rejected(self):
        source=self.references()['layer_embedding']
        source=source.replace('np.add.at(self.weight.grad, self._ids.reshape(-1), g.reshape(-1, g.shape[-1]))','self.weight.grad[self._ids] += g')
        result=self.check('layer_embedding',source)
        self.assertFalse(result['rows'][0]['passed'])

    def test_reshape_that_loses_token_axis_is_rejected(self):
        source=self.references()['layer_linear'].replace('return y2.reshape(*x.shape[:-1], self.W.data.shape[1])','return y2')
        self.assertFalse(self.check('layer_linear',source)['rows'][0]['passed'])

    def test_mlp_uses_student_dependencies(self):
        refs=self.references()
        result=self.check('layer_mlp',refs['layer_mlp'],{'layer_linear':self.book['lessons']['layer_linear']['starter'],'layer_gelu':refs['layer_gelu']})
        self.assertFalse(any(row['passed'] for row in result['rows']))
        self.assertIn('NotImplementedError',result['rows'][0]['errorText'])

    def test_local_wrappers_without_overwriting_exercises(self):
        refs=self.references()
        # Complete the prerequisite only in an isolated fixture, never in the learner folder.
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as tmp:
            folder=Path(tmp)
            for id,source in refs.items():
                if not self.book['lessons'][id].get('file'):continue
                (folder/self.book['lessons'][id]['file']).write_text(standalone(self.book['lessons'][id],source),encoding='utf-8')
            for id in ['shape','layer_scale','layer_linear','layer_embedding','layer_gelu','layer_rmsnorm','infrastructure']:
                with self.subTest(id=id):
                    code=(folder/self.book['lessons'][id]['file']).read_text(encoding='utf-8')
                    bootstrap='import sys;sys.path[:0]='+repr([str(folder),str(ROOT/'exercises')])+';'+code
                    result=subprocess.run([sys.executable,'-c',bootstrap],capture_output=True,text=True,encoding='utf-8',timeout=12)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_catalog_boundaries(self):
        self.assertEqual(len(self.book['lessons']),56)
        self.assertTrue(self.book['stages'][2]['paused'])
        self.assertIn('链式法则',self.book['lessons']['layer_gelu']['derivation'])
        self.assertTrue(all('extension' in l for l in self.book['lessons'].values()))

    def test_high_dimensional_default_samples(self):
        import numpy as np
        for id in ['linear','reshape','transpose','layer_scale','layer_linear','layer_gelu','layer_rmsnorm','layer_mlp']:
            meta=self.book['lessons'][id]
            case=meta['cases'][meta.get('sample_index',0)]
            shape=np.asarray(case['args'][0] if meta['kind']=='array' else case['X']).shape
            self.assertEqual(len(shape),3,id)
            self.assertTrue(all(size>1 for size in shape),(id,shape))
        case=self.book['lessons']['heads']['cases'][0]
        expected=np.asarray(case['expected'])
        self.assertEqual(expected.shape,(2,2,3,4))
        restored=expected.transpose(0,2,1,3).reshape(2,3,8)
        self.assertTrue(np.array_equal(restored,np.asarray(case['args'][0])))
        embedding=self.book['lessons']['layer_embedding']['cases'][0]
        self.assertGreater(len(embedding['X']),1)
        self.assertTrue(set(embedding['X'][0])&set(embedding['X'][1]))

    def test_http_grading_and_save_backup(self):
        # Exercise the actual routes with disposable files, never learner work.
        with tempfile.TemporaryDirectory(dir=ROOT/".check-tmp") as tmp:
            directory=Path(tmp)
            (directory/'numpy_shape.py').write_text('original learner draft',encoding='utf-8')
            server=ThreadingHTTPServer(('127.0.0.1',0),pilot_server.Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            def post(path,body):
                request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
                return json.load(urllib.request.urlopen(request))
            try:
                with patch.object(pilot_server,'EXERCISES',directory):
                    result=post('/api/save',dict(lesson='shape',code=self.references()['shape']))
                    self.assertEqual(result['path'],'exercises/numpy_shape.py')
                    saved=(directory/'numpy_shape.py').read_text(encoding='utf-8')
                    self.assertIn('run_local("shape", solve)',saved)
                    backups=list((directory/'backups').iterdir())
                    self.assertEqual(len(backups),1)
                    self.assertEqual(backups[0].read_text(encoding='utf-8'),'original learner draft')
                result=post('/api/check',dict(lesson='map',answers=[0,2]))
                self.assertTrue(all(row['passed'] for row in result['rows']))
                result=post('/api/check',dict(lesson='map',answers=[-1,0]))
                self.assertFalse(any(row['passed'] for row in result['rows']))
                result=post('/api/check',dict(lesson='initialization',code=self.references()['initialization']))
                self.assertTrue(all(row['passed'] for row in result['rows']),result)
                result=post('/api/check',dict(lesson='layer_linear',code=self.references()['layer_linear'],dependencies={'infrastructure':self.references()['infrastructure'],'initialization':self.references()['initialization']}))
                self.assertTrue(all(row['passed'] for row in result['rows']),result)
                result=post('/api/check',dict(lesson='axes',code='def solve(X): return np.full(X.shape[0], np.nan)'))
                self.assertFalse(any(row['passed'] for row in result['rows']))
                self.assertEqual(result['rows'][0]['actual'][0],'nan')
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':
    unittest.main(verbosity=2)
