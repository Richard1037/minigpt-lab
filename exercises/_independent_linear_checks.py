"""独立Linear构造与当前个人类的实际组合验收。"""
import numpy as np
from foundation_core import Parameter,Module

def check_build(scope,case):
    P,M=scope.get('Parameter',Parameter),scope.get('Module',Module)
    for C,D,bias in case['input']['configs']:
        obj=scope['Linear'](C,D,bias);other=scope['Linear'](C,D,bias)
        assert isinstance(obj,M) and isinstance(obj.W,P),'权重需使用当前Parameter，层继承当前Module'
        expected=np.random.default_rng(0).normal(0,np.sqrt(2/C),(C,D))
        assert obj.W.data.shape==(C,D) and obj.W.data.dtype==np.float64,'权重形状或dtype不符合契约'
        np.testing.assert_allclose(obj.W.data,expected,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(obj.W.grad,np.zeros((C,D)))
        params=[obj.W]
        if bias:
            assert isinstance(obj.b,P),'偏置应为Parameter'
            np.testing.assert_array_equal(obj.b.data,np.zeros(D));np.testing.assert_array_equal(obj.b.grad,np.zeros(D));params.append(obj.b)
        else:assert obj.b is None,'关闭偏置时self.b为None'
        found=obj.parameters();assert len(found)==len(params) and all(x is y for x,y in zip(found,params)),'参数收集的对象/顺序不正确'
        for x,y in zip(found,other.parameters()):
            assert not np.shares_memory(x.data,y.data) and not np.shares_memory(x.grad,y.grad),'不同实例的参数存储应独立'
    namespace=scope['Linear'].__init__.__globals__
    original=namespace['initialize'];calls=[]
    def supplied(shape, rng=None):
        calls.append(tuple(shape))
        return np.arange(np.prod(shape),dtype=float).reshape(shape)/13+.27
    try:
        namespace['initialize']=supplied
        obj=scope['Linear'](3,4);weight=obj.W;weight.data[:]+=.3;weight.grad[:]=.7
        data=weight.data.copy();x=np.arange(18,dtype=float).reshape(2,3,3)/9
        for _ in range(2):
            np.testing.assert_allclose(obj.forward(x),x@data,rtol=0,atol=1e-12)
            assert obj.W is weight,'前向需复用构造时的参数对象'
            np.testing.assert_array_equal(weight.data,data);np.testing.assert_array_equal(weight.grad,np.full((3,4),.7))
        assert calls==[(3,4)],'构造调用当前initialize一次；前向复用当前参数'
        np.testing.assert_allclose(data,np.arange(12).reshape(3,4)/13+.57,rtol=0,atol=1e-12)
    finally:namespace['initialize']=original
    return {'独立构造与初始化接入':True}

def check_composition(scope,case):
    cfg=case['input'];rng=np.random.default_rng(cfg['seed']);C,H,D=cfg['C'],cfg['hidden'],cfg['D']
    first=scope['Linear'](C,H);second=scope['Linear'](H,D)
    assert first is not second and isinstance(first,scope['Linear']) and isinstance(second,scope['Linear'])
    for obj in [first,second]:
        for param in obj.parameters():param.data[:]=rng.normal(size=param.data.shape)*.2
    w1,b1,w2,b2=first.W.data.copy(),first.b.data.copy(),second.W.data.copy(),second.b.data.copy()
    x=rng.normal(size=cfg['shape']);g=rng.normal(size=(*cfg['shape'][:-1],D))
    z=first(x);y=second(z);dz=second.backward(g);dx=first.backward(dz)
    expected_z=x@w1+b1;expected_dz=g@w2.T
    for actual,expected in [(z,expected_z),(y,expected_z@w2+b2),(dz,expected_dz),(dx,expected_dz@w1.T),
                            (first.W.grad,x.reshape(-1,C).T@expected_dz.reshape(-1,H)),(first.b.grad,expected_dz.reshape(-1,H).sum(0)),
                            (second.W.grad,expected_z.reshape(-1,H).T@g.reshape(-1,D)),(second.b.grad,g.reshape(-1,D).sum(0))]:
        assert actual.shape==expected.shape,'串联后应保留全部前置维度'
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
    for a in first.parameters():
        for b in second.parameters():assert not np.shares_memory(a.data,b.data),'两个层的参数应独立'
    return {'两个个人Linear已串联':True,'对象和缓存独立':True}
