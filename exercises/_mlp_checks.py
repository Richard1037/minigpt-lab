"""MLP 组合行为验收，网页与本地共用。"""
import numpy as np
from foundation_core import Module,Parameter

def check_composition(scope,case):
    M=scope.get('Module',Module);P=scope.get('Parameter',Parameter)
    for C,H in case['input']['configs']:
        obj=scope['MLP'](C,H);other=scope['MLP'](C,H)
        assert isinstance(obj,M) and isinstance(obj.fc1,scope['Linear']) and isinstance(obj.fc2,scope['Linear']) and isinstance(obj.act,scope['GELU']),'创建规定的三个子层'
        assert obj.fc1.W.data.shape==(C,H) and obj.fc2.W.data.shape==(H,C),'两个Linear的输入输出宽度错误'
        assert obj.fc1.b.data.shape==(H,) and obj.fc2.b.data.shape==(C,),'偏置宽度错误'
        params=obj.parameters();expected=[obj.fc1.W,obj.fc1.b,obj.fc2.W,obj.fc2.b]
        assert len(params)==4 and all(a is b and isinstance(a,P) for a,b in zip(params,expected)),'递归收集四个独立参数张量'
        for a,b in zip(params,other.parameters()):
            assert not np.shares_memory(a.data,b.data) and not np.shares_memory(a.grad,b.grad),'实例不能共用参数存储'
    # 替身只验证组合协议；数值与梯度由原九项真实子层测评覆盖。
    events=[];created=[]
    class TraceLinear(M):
        def __init__(self,cin,cout,use_bias=True,*,rng=None,init_std=None,init_scale=1.0):
            self.cin,self.cout=cin,cout
            self.tag='fc1' if not created else 'fc2'
            created.append(self)
        def forward(self,x):
            assert x.shape[-1]==self.cin
            events.append(self.tag+'.forward');return np.repeat(x.sum(-1,keepdims=True),self.cout,axis=-1)+1
        def backward(self,g):
            assert g.shape[-1]==self.cout
            events.append(self.tag+'.backward');return np.repeat(g.sum(-1,keepdims=True),self.cin,axis=-1)+2
    class TraceGELU(M):
        def forward(self,x):events.append('act.forward');return x*3
        def backward(self,g):events.append('act.backward');return g*3
    namespace=scope['MLP'].__init__.__globals__
    old_linear,old_gelu=namespace['Linear'],namespace['GELU']
    try:
        namespace['Linear'],namespace['GELU']=TraceLinear,TraceGELU
        obj=scope['MLP'](3,5);children=(obj.fc1,obj.act,obj.fc2)
        assert len(created)==2
        x=np.arange(18,dtype=float).reshape(2,3,3)/10;g=np.full_like(x,.7)
        z=np.repeat(x.sum(-1,keepdims=True),5,axis=-1)+1
        y=np.repeat((z*3).sum(-1,keepdims=True),3,axis=-1)+1
        da=np.repeat(g.sum(-1,keepdims=True),5,axis=-1)+2
        dx=np.repeat((da*3).sum(-1,keepdims=True),3,axis=-1)+2
        for _ in range(2):
            events.clear();np.testing.assert_allclose(obj.forward(x),y);np.testing.assert_allclose(obj.backward(g),dx)
            assert events==['fc1.forward','act.forward','fc2.forward','fc2.backward','act.backward','fc1.backward'],'子层调用顺序或次数不符合串联协议'
            assert children==(obj.fc1,obj.act,obj.fc2) and len(created)==2,'前后向复用构造的子层'
    finally:namespace['Linear'],namespace['GELU']=old_linear,old_gelu
    return {'构造与串联':True}
