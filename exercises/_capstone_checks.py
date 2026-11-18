"""基础层综合：调用当前个人类，数值对照由独立数组公式计算。"""
import json
from pathlib import Path
import numpy as np


def oracle(x, g, weights, eps):
    w, a, b, c, d = weights
    r = 1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)
    u = x*r*w
    z = u@a+b
    k = np.sqrt(2/np.pi); t = np.tanh(k*(z+.044715*z**3))
    v = .5*z*(1+t); y = v@c+d
    dv = g@c.T
    dz = dv*(.5*(1+t)+.5*z*(1-t*t)*k*(1+3*.044715*z*z))
    du = dz@a.T; q = du*w
    dx = q*r-x*r**3*np.mean(q*x, axis=-1, keepdims=True)
    axes = tuple(range(x.ndim-1))
    return y, dx, [(du*x*r).sum(axes), u.reshape(-1,u.shape[-1]).T@dz.reshape(-1,z.shape[-1]),
                   dz.sum(axes), v.reshape(-1,v.shape[-1]).T@g.reshape(-1,g.shape[-1]), g.sum(axes)]


def behavior(scope, case):
    kind=case['assessment'];p=case['input']
    if kind=='foundation_gate':
        from _foundation_checks import evaluate_layer,evaluate_infrastructure
        from _initialization_checks import evaluate_initialize
        book=json.loads((Path(__file__).resolve().parents[1]/'webapp/foundations.json').read_text(encoding='utf-8'))
        meta=book['lessons'][p['lesson']]
        if p['lesson']=='infrastructure':rows=evaluate_infrastructure(scope,meta['cases'])
        elif p['lesson']=='initialization':rows=evaluate_initialize(scope['initialize'],meta['cases'])
        else:rows=evaluate_layer(scope,meta['cases'])
        failures=[r['label'] for r in rows if not r['passed']]
        assert not failures,p['lesson']+' 的个人实现未通过：'+'；'.join(failures)
        return {'复用个人模块':p['lesson'],'通过检查数':len(rows)}
    cls=scope['MLPBranch'];rng=np.random.default_rng(p.get('seed',632))
    C,H=p['C'],p['hidden'];eps=p.get('eps',1e-5)
    obj=cls(C,H,eps)
    assert isinstance(obj,scope['Module'])
    assert isinstance(obj.norm,scope['RMSNorm']) and isinstance(obj.mlp,scope['MLP']),'使用当前个人RMSNorm和MLP'
    assert obj.norm.eps==eps,'eps须传给子层'
    assert isinstance(obj.mlp.fc1,scope['Linear']) and isinstance(obj.mlp.act,scope['GELU'])
    params=obj.parameters()
    expected=[obj.norm.w,obj.mlp.fc1.W,obj.mlp.fc1.b,obj.mlp.fc2.W,obj.mlp.fc2.b]
    assert len(params)==5 and all(a is b for a,b in zip(params,expected)),'五个参数须由Module按对象身份收集'
    for param in params:param.data[:]=rng.normal(size=param.data.shape)*.3
    x=rng.normal(size=p['shape']);g=rng.normal(size=x.shape);xb=x.copy();gb=g.copy()
    weights=[param.data.copy() for param in params]
    y=obj(x);dx=obj.backward(g);ey,edx,eg=oracle(x,g,weights,eps)
    for a,b in [(y,ey),(dx,edx),*zip([v.grad for v in params],eg)]:
        assert a.shape==b.shape
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8)
    if kind=='branch_gradient':
        h=1e-5
        for source,grad in [(x,dx),*zip([v.data for v in params],[v.grad.copy() for v in params])]:
            for flat in np.linspace(0,source.size-1,min(6,source.size),dtype=int):
                co=np.unravel_index(flat,source.shape);old=source[co]
                source[co]=old+h;plus=np.sum(obj(x)*g)
                source[co]=old-h;minus=np.sum(obj(x)*g);source[co]=old
                np.testing.assert_allclose(grad[co],(plus-minus)/(2*h),rtol=2e-4,atol=3e-6)
    if kind=='branch_state':
        obj(x);x[:]+=7;obj.backward(g)
        for param,grad in zip(params,eg):np.testing.assert_allclose(param.grad,2*grad,rtol=1e-6,atol=1e-8)
        saved=[v.grad for v in params];obj.zero_grad()
        assert all(v.grad is old and np.all(v.grad==0) for v,old in zip(params,saved))
        for v,w in zip(params,weights):np.testing.assert_array_equal(v.data,w)
        np.testing.assert_array_equal(g,gb)
    if kind=='branch_composition':
        # 使用可追踪子层验证真实调用与返回值传递，禁止内联另一份实现绕开组合。
        trace=[]
        namespace=cls.__init__.__globals__
        base_norm,base_mlp=namespace['RMSNorm'],namespace['MLP']
        class Norm(base_norm):
            def forward(self,a):trace.append('norm.forward');return super().forward(a)
            def backward(self,a):trace.append('norm.backward');return super().backward(a)
        class Mlp(base_mlp):
            def forward(self,a):trace.append('mlp.forward');return super().forward(a)
            def backward(self,a):trace.append('mlp.backward');return super().backward(a)
        try:
            namespace['RMSNorm'],namespace['MLP']=Norm,Mlp
            tracked=cls(C,H,eps);tracked(xb);tracked.backward(gb)
            assert isinstance(tracked.norm,Norm) and isinstance(tracked.mlp,Mlp),'构造时调用当前个人类'
            assert trace==['norm.forward','mlp.forward','mlp.backward','norm.backward'],'子层调用顺序/次数不正确'
        finally:namespace['RMSNorm'],namespace['MLP']=base_norm,base_mlp
    if kind!='branch_state':
        np.testing.assert_array_equal(x,xb);np.testing.assert_array_equal(g,gb)
    return {'综合分支检查':True}
