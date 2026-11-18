"""后续章节网页/本地统一测评。probe 明确调用学习者接口，无参考实现注入。"""
import json
import tempfile
import traceback
from pathlib import Path
import numpy as np
from foundation_core import Parameter, Module
from _foundation_checks import compare, pack

def probe(scope, case):
    P,M=scope.get('Parameter',Parameter),scope.get('Module',Module)
    if case.get('assessment'):
        from _assessment import behavior
        return behavior(scope,case)
    p = case['input']
    a = lambda name: np.array(p[name],dtype=np.int64 if name in ['ids','targets','starts'] else np.float64)
    mode = case['mode']
    if mode in ['module','loss','attention']:
        obj = scope[p['class']](*p.get('init',[]))
        if mode == 'loss':
            out = obj.forward(a('X'),a('targets')); dx = obj.backward()
            return dict(loss=out,dLogits=dx)
        if mode == 'attention':
            out = obj.forward(a('Q'),a('K'),a('V'))
            dq,dk,dv = obj.backward(a('G'))
            return dict(Y=out,dQ=dq,dK=dk,dV=dv,A=obj._a)
        for key,value in p.get('parameters',{}).items():
            getattr(obj,key).data[...] = value
        x = a('ids') if p['class']=='GPT' else a('X')
        before = x.copy()
        weights = [v.data.copy() for v in obj.parameters()]
        out = obj.forward(x,**p.get('kwargs',{}))
        dx = obj.backward(a('G'))
        assert np.array_equal(x,before), '不得修改输入'
        assert all(np.array_equal(v.data,w) for v,w in zip(obj.parameters(),weights)), 'backward 不得更新参数'
        grads = {}
        seen=set()
        def collect(value,path=''):
            if isinstance(value,P):
                if id(value) not in seen:
                    seen.add(id(value));grads[path+'.grad']=value.grad.copy()
            elif isinstance(value,M):
                if type(value) is scope.get('Block'):
                    from _block_inspection import block_parts
                    n1,attn,branch,n2,mlp=block_parts(value,scope,require_branch=False)
                    for name,child in [('norm1',n1),('norm2',n2),('attn',attn),('mlp',mlp)]:collect(child,(path+'.' if path else '')+name)
                    return
                for name,child in vars(value).items():
                    if not name.startswith('_'):collect(child,(path+'.' if path else '')+name)
            elif isinstance(value,(list,tuple,dict)):
                for i,child in (value.items() if isinstance(value,dict) else enumerate(value)):collect(child,path+f'.{i}')
        if p['class']=='Block':
            from _block_inspection import block_parts
            norm1,attn,branch,norm2,mlp=block_parts(obj,scope,require_branch=False)
            for name,value in [('norm1',norm1),('norm2',norm2),('attn',attn),('mlp',mlp)]:collect(value,name)
            assert seen==set(map(id,obj.parameters())), 'Block全部分支参数须可由 parameters() 找到，且不包含额外参数；公开分支或其子层即可'
        else:collect(obj)
        # 同一缓存的两次 backward 应累加参数梯度，且不更新权重。
        first=[v.grad.copy() for v in obj.parameters()]
        obj.backward(a('G'))
        assert all(np.allclose(v.grad,2*g,atol=1e-10) for v,g in zip(obj.parameters(),first)), '参数梯度必须累加'
        obj.zero_grad()
        assert all(np.all(v.grad==0) for v in obj.parameters())
        return dict(Y=out,dX=dx,**grads)
    if mode == 'data':
        vocab, ids = scope['tokenize'](p['text'])
        x,y = scope['make_batch'](ids,a('starts'),p['T'])
        return dict(vocab=''.join(vocab),ids=ids,X=x,targets=y)
    if mode == 'optimizer':
        params = [P(v) for v in p['weights']]
        opt = scope['AdamW'](params,**p['config'])
        norms=[]
        for gs in p['gradients']:
            for param,g in zip(params,gs): param.grad[...] = g
            norms.append(float(scope['clip_grad'](params,p['limit'])))
            opt.step()
        return {**{f'p{i}':v.data for i,v in enumerate(params)},'norms':np.array(norms),
                **{f'm{i}':v for i,v in enumerate(opt.m)},**{f'v{i}':v for i,v in enumerate(opt.v)},'t':np.array(opt.t)}
    if mode == 'sample':
        result = scope['sample'](a('X'),p['temperature'],p['top_k'],np.random.default_rng(p['seed']))
        return dict(ids=result)
    if mode in ['train','checkpoint','generate']:
        model = scope['GPT'](*p['init'])
        x,y = a('ids'),a('targets')
        if mode == 'generate':
            return dict(ids=scope['generate'](model,x,p['steps'],p['temperature'],p['top_k'],p['seed']))
        if mode == 'train':
            losses=scope['train'](model,x,y,p['steps'],p['lr'])
            return dict(losses=losses,logits=model(x))
        opt=scope['AdamW'](model.parameters(),lr=.01)
        rng=np.random.default_rng(42)
        ce=scope['CrossEntropy']()
        def update():
            model.zero_grad();ce(model(x),y);model.backward(ce.backward());opt.step()
        update()
        temp_root=Path(__file__).resolve().parents[1]/'.check-tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path=str(Path(tmp)/'checkpoint.npz')
            scope['save_checkpoint'](path,model,opt,rng)
            update(); expected=model(x).copy(); random_expected=rng.random(4)
            scope['load_checkpoint'](path,model,opt,rng)
            assert all(np.all(v.grad==0) for v in model.parameters()), '加载后梯度应清零'
            update(); actual=model(x)
            assert np.allclose(actual,expected,atol=1e-12), '恢复后下一次更新不一致'
            return dict(logits=actual,rng=rng.random(4),t=np.array(opt.t),continuation=np.array(np.max(np.abs(actual-expected))))
    raise ValueError(mode)

def evaluate_project(scope,cases):
    rows=[]
    for c in cases:
        if c.get('assessment'):
            from _assessment import evaluate_special
            rows.append(evaluate_special(scope,c));continue
        try:
            out=probe(scope,c)
            def equal(a,b):
                if isinstance(b,str): return a==b
                if isinstance(b,dict): return isinstance(a,dict) and set(a)==set(b) and all(equal(a[k],v) for k,v in b.items())
                return compare(a,b)
            from _assessment import diagnose
            rows.append(dict(label=c['label'],passed=bool(equal(out,c['expected'])),diagnostics=diagnose(out,c['expected']),actual=pack(out) if not any(isinstance(v,str) for v in out.values()) else {k:v if isinstance(v,str) else pack(v) for k,v in out.items()}))
        except Exception:
            rows.append(dict(label=c['label'],passed=False,errorText=traceback.format_exc(limit=5)))
    return rows

def run_local(lesson,scope):
    book=json.loads((Path(__file__).resolve().parents[1]/'webapp/foundations.json').read_text(encoding='utf-8'))
    cases=book['lessons'][lesson]['cases']
    rows=evaluate_project(scope,cases)
    for c,r in zip(cases,rows):
        print(('PASS' if r['passed'] else 'FAIL')+' | '+r['label'])
        print('输入:',json.dumps(c['input'],ensure_ascii=False))
        print('预期:',json.dumps(c['expected'],ensure_ascii=False))
        print('实际:',r.get('actual',r.get('errorText')))
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} 项通过")
    raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
