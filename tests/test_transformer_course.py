"""独立差分、因果性、真实依赖及训练闭环验证。"""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'solutions'),str(ROOT/'exercises'),str(ROOT/'webapp')]
import transformer_layers as r
from _transformer_lab import experiment

class TransformerTests(unittest.TestCase):
    def difference(self,array,fn,expected):
        actual=np.empty_like(array);h=1e-5
        for index in np.ndindex(array.shape):
            old=array[index];array[index]=old+h;a=fn();array[index]=old-h;b=fn();array[index]=old
            actual[index]=(a-b)/(2*h)
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-4)

    def test_attention_three_gradients(self):
        rng=np.random.default_rng(3)
        for causal,Tq in [(True,3),(False,2)]:
            obj=r.AttentionCore(causal)
            q=rng.normal(size=(2,2,Tq,2));k=rng.normal(size=(2,2,3,2));v=rng.normal(size=(2,2,3,2));g=rng.normal(size=q.shape)
            obj(q,k,v);grads=obj.backward(g)
            for x,grad in zip([q,k,v],grads):self.difference(x,lambda:float(np.sum(obj(q,k,v)*g)),grad)

    def test_composed_layers_and_shared_gpt_gradients(self):
        rng=np.random.default_rng(5)
        for obj in [r.MultiHeadAttention(4,2),r.Block(4,2,6),r.LayerNorm(4)]:
            x=rng.normal(size=(2,2,4));g=rng.normal(size=x.shape)
            obj(x);dx=obj.backward(g);grads=[p.grad.copy() for p in obj.parameters()]
            self.difference(x,lambda:float(np.sum(obj(x)*g)),dx)
            for p,grad in zip(obj.parameters(),grads):self.difference(p.data,lambda:float(np.sum(obj(x)*g)),grad)
        obj=r.GPT(5,4,2,6,1,4,7);ids=np.array([[1,2],[2,1]]);g=rng.normal(size=(2,2,5))
        obj(ids);obj.backward(g);grads=[p.grad.copy() for p in obj.parameters()]
        for p,grad in zip(obj.parameters(),grads):self.difference(p.data,lambda:float(np.sum(obj(ids)*g)),grad)
        self.assertEqual(len({id(p) for p in obj.parameters()}),len(obj.parameters()))

    def test_loss_derivative_and_shift(self):
        obj=r.CrossEntropy();x=np.random.default_rng(2).normal(size=(2,3,5));target=np.array([[0,1,2],[3,4,0]])
        loss=obj(x,target);g=obj.backward();self.difference(x,lambda:float(obj(x,target)),g)
        np.testing.assert_allclose(obj(x+10000,target),loss,atol=1e-10)
        np.testing.assert_allclose(g.sum(-1),0,atol=1e-14)

    def test_no_future_leakage(self):
        model=r.GPT(6,4,2,8,2,5,9)
        a=np.array([[0,1,2,3],[1,2,3,4]]);b=a.copy();b[:,-1]=5
        ya=model(a);yb=model(b)
        np.testing.assert_allclose(ya[:,:-1],yb[:,:-1],atol=1e-12)
        g=np.zeros_like(ya);g[:,0,:]=1;model.zero_grad();model(a);model.backward(g)
        np.testing.assert_allclose(model.position.weight.grad[1:],0,atol=1e-12)

    def test_training_decreases_fixed_batch_loss(self):
        model=r.GPT(5,8,2,16,1,4,3);x=np.array([[0,1,2],[1,2,3]]);y=(x+1)%5
        losses=r.train(model,x,y,60,.02)
        self.assertLess(losses[-1],losses[0]*.4)

    def test_lab_runs_with_reference_scope(self):
        out=experiment(vars(r),dict(text='春天来了。小猫喜欢鱼。小狗喜欢跑。'*20,steps=8,tokens=4))
        self.assertEqual(len(out['losses']),8);self.assertEqual(len(out['generated']),2)
        self.assertTrue(np.isfinite(out['validation_after']))

    def test_public_specs_dependency_order_and_local_imports(self):
        book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        for id,meta in book['lessons'].items():
            if meta['kind']!='project':continue
            scope=dict(np=np,__name__='student')
            from foundation_core import Module,Parameter,initialize
            scope.update(Module=Module,Parameter=Parameter,initialize=initialize)
            for dep in meta['dependencies']:exec(book['lessons'][dep]['reference'],scope)
            exec(meta['reference'],scope)
            from _transformer_checks import evaluate_project
            self.assertTrue(all(row['passed'] for row in evaluate_project(scope,meta['cases'])),id)
            if not meta.get('repair') and not meta.get('integration_only'):
                self.assertIn('NotImplementedError',meta['starter'])
        # 缺少个人 AttentionCore 不允许用参考层悄悄替代。
        meta=book['lessons']['t_mha'];scope=dict(np=np,Module=Module,Parameter=Parameter,initialize=initialize,__name__='student')
        for dep in meta['dependencies']:exec(book['lessons'][dep]['starter' if dep=='t_attention' else 'reference'],scope)
        exec(meta['reference'],scope)
        self.assertFalse(any(row['passed'] for row in evaluate_project(scope,meta['cases'])))

    def test_local_files_and_training_cli_in_isolation(self):
        import tempfile, subprocess
        from pilot_server import standalone
        book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as tmp:
            folder=Path(tmp)
            for meta in book['lessons'].values():
                if meta.get('reference') and meta.get('file'):
                    (folder/meta['file']).write_text(standalone(meta,meta['reference']),encoding='utf-8')
            bootstrap='import sys;sys.path[:0]='+repr([str(folder),str(ROOT/'exercises')])+';'
            for id,meta in book['lessons'].items():
                if meta['kind']!='project':continue
                cmd=bootstrap+'exec(compile('+repr((folder/meta['file']).read_text(encoding='utf-8'))+',"<isolated exercise>","exec"))'
                result=subprocess.run([sys.executable,'-c',cmd],text=True,capture_output=True,encoding='utf-8',timeout=15)
                self.assertEqual(result.returncode,0,(id,result.stdout[-1200:],result.stderr))
            corpus=folder/'corpus.txt';corpus.write_text('春天来了。小猫喜欢鱼。小狗喜欢跑。'*30,encoding='utf-8')
            script=(ROOT/'exercises/run_gpt.py').read_text(encoding='utf-8-sig')
            for extra in [[],['--resume'],['--generate-only'],['--independent'],['--resume']]:
                args=['run_gpt.py',str(corpus),'--out',str(folder/'run'),'--steps','2','--C','4','--hidden','8','--T','4','--tokens','3']+extra
                cmd=bootstrap+'sys.argv='+repr(args)+';exec(compile('+repr(script)+',"<training cli>","exec"))'
                result=subprocess.run([sys.executable,'-c',cmd],text=True,capture_output=True,encoding='utf-8',timeout=15)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue((folder/'run/checkpoint.npz').exists())

    def test_visualization_math_matches_full_tensor_reference(self):
        import subprocess
        rgen=np.random.default_rng(31)
        x=rgen.normal(size=(2,3,4));w=rgen.normal(size=(4,5));g=rgen.normal(size=(2,3,5))
        q,k,v,ag=[rgen.normal(size=(2,2,3,2)) for _ in range(4)]
        payload=dict(X=x.tolist(),W=w.tolist(),G=g.tolist(),Q=q.tolist(),K=k.tolist(),V=v.tolist(),AG=ag.tolist())
        js="const m=require('./webapp/understanding.js');let text='';process.stdin.on('data',d=>text+=d);process.stdin.on('end',()=>{const p=JSON.parse(text);process.stdout.write(JSON.stringify({linear:m.linearBackward(p.X,p.W,p.G),attention:m.attentionBackward(p.Q,p.K,p.V,p.AG)}));});"
        proc=subprocess.run(['node','-e',js],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT,timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr);out=json.loads(proc.stdout)
        np.testing.assert_allclose(out['linear']['dX'],g@w.T,atol=1e-12)
        np.testing.assert_allclose(out['linear']['dW'],x.reshape(-1,4).T@g.reshape(-1,5),atol=1e-12)
        np.testing.assert_allclose(out['linear']['db'],g.sum((0,1)),atol=1e-12)
        np.testing.assert_allclose(out['attention']['S'],q@k.swapaxes(-1,-2)/np.sqrt(q.shape[-1]),atol=1e-12)
        core=r.AttentionCore();y=core(q,k,v);dq,dk,dv=core.backward(ag)
        for key,target in [('Y',y),('dQ',dq),('dK',dk),('dV',dv),('A',core._a)]:np.testing.assert_allclose(out['attention'][key],target,atol=1e-12)

    def test_optimizer_visualization_matches_every_public_case(self):
        import subprocess
        b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        cases=[c for c in b['lessons']['t_optim']['cases'] if c.get('mode')=='optimizer']
        js="const m=require('./webapp/understanding.js');let s='';process.stdin.on('data',d=>s+=d);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(c=>m.optimizerTrace(c.input)))));"
        proc=subprocess.run(['node','-e',js],input=json.dumps(cases),text=True,capture_output=True,cwd=ROOT,timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr)
        for case,result in zip(cases,json.loads(proc.stdout)):
            expected=case['expected']
            for i in range(len(case['input']['weights'])):
                for key,field in [('p','weights'),('m','m'),('v','v')]:
                    np.testing.assert_allclose(result[field][i],expected[f'{key}{i}'],rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose([f['norm'] for f in result['frames']],expected['norms'],rtol=1e-11,atol=1e-12)
            self.assertEqual(len(result['frames']),expected['t'])

    def test_expanded_gradient_and_behavior_checks_reject_bugs(self):
        from _assessment import evaluate_special
        b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        class WrongSoftmax(r.Softmax):
            def backward(self,g):return g*self._p*(1-self._p)
        case=next(c for c in b['lessons']['t_softmax']['cases'] if c.get('assessment')=='gradient')
        row=evaluate_special(dict(Softmax=WrongSoftmax),case)
        self.assertFalse(row['passed']);self.assertIn('dInput',row['diagnostics'][0])
        class WrongGPT(r.GPT):
            def backward(self,g):
                super().backward(g)
                self.token.weight.grad[-1]=0
        case=next(c for c in b['lessons']['t_gpt']['cases'] if c.get('assessment')=='shared')
        self.assertFalse(evaluate_special(dict(GPT=WrongGPT),case)['passed'])
        from _assessment import diagnose
        errors=diagnose({'dX':np.zeros((2,3,4))},{'dX':np.ones((2,3,4)).tolist()})
        self.assertIn('dX',errors[0]);self.assertIn('(0, 0, 0)',errors[0])

if __name__=='__main__':unittest.main(verbosity=2)
