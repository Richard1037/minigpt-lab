"""只检查本次导入的个人类；不加载参考或修改个人对象。"""
import numpy as np
from _block_inspection import block_parts


class TrackedRNG(np.random.Generator):
    def __init__(self,seed):
        super().__init__(np.random.PCG64(seed));self.draws=[]
    def normal(self,*args,**kwargs):
        out=super().normal(*args,**kwargs);self.draws.append(np.array(out,copy=True));return out


def behavior(scope,case):
    def check(ok,msg):
        if not ok:raise AssertionError(msg)
    def close(a,b,msg):
        a,b=np.asarray(a),np.asarray(b)
        check(a.shape==b.shape and np.allclose(a,b,atol=1e-11,rtol=1e-9),msg)
    def build(name,*args,**kwargs):
        try:return scope[name](*args,**kwargs)
        except TypeError as e:raise AssertionError(name+' 的原文件尚未接通可选初始化参数：'+str(e)) from e
    variant=case['input']['variant']
    if variant=='gpt':
        from unittest.mock import patch
        streams=[]
        def create(seed=None):
            rng=TrackedRNG(seed);streams.append(rng);return rng
        with patch.object(np.random,'default_rng',create):
            build('GPT',5,4,2,7,2,6,seed=17)
        check(len(streams)==1,'GPT只创建一个随机数生成器并向下传递')
        check([v.shape for v in streams[0].draws]==[(5,4),(6,4)]+[(4,12),(4,4),(4,7),(7,4)]*2,'整网权重在构造时各采样一次，不进行外层重置')
        return dict(一个生成器=True,十张矩阵各采样一次=True)
    if variant=='legacy':
        layer=build('Linear',3,4);close(layer.W.data,scope['initialize']((3,4)),'Linear旧默认fan-in保持')
        x=np.arange(6.).reshape(2,3);close(layer(x),x@layer.W.data+layer.b.data,'旧前向接口保持')
        no_bias=build('Linear',3,2,False);check(no_bias.b is None,'原use_bias位置参数保持')
        embedding=build('Embedding',4,3);close(embedding.weight.data,np.random.default_rng(0).normal(0,.02,(4,3)),'Embedding默认种子/标准差保持')
        for name,args in [('MLP',(4,7)),('MLPBranch',(4,7)),('MultiHeadAttention',(4,2)),('Block',(4,2,7))]:
            obj=build(name,*args);check(obj(np.ones((2,3,4))).shape==(2,3,4),name+'旧调用保持')
        return dict(旧构造与前向=True,旧初始化默认值=True)
    if variant=='linear':
        rng=TrackedRNG(11);a=build('Linear',3,4,rng=rng,init_std=.07,init_scale=.5)
        close(a.W.data,np.random.default_rng(11).normal(0,.07,(3,4))*.5,'直接指定标准差并仅缩放一次')
        check(len(rng.draws)==1,'Linear每张权重只采样一次，不先fan-in再重采样')
        close(a.b.data,np.zeros(4),'偏置保留零');close(a.W.grad,np.zeros((3,4)),'初始梯度为零')
        second=build('Linear',3,4,rng=rng,init_std=.07)
        check(not np.array_equal(a.W.data*2,second.W.data),'同形状参数使用流的不同片段')
        close(second.W.data,rng.draws[1],'第二个Linear使用同一生成器')
        return dict(一次采样=True,可选标准差和系数=True,同一随机流=True)
    if variant=='embedding':
        rng=TrackedRNG(23);a=build('Embedding',4,3,rng=rng);b=build('Embedding',4,3,rng=rng,init_std=.03)
        expected=np.random.default_rng(23)
        close(a.weight.data,expected.normal(0,.02,(4,3)),'首表使用传入生成器')
        close(b.weight.data,expected.normal(0,.03,(4,3)),'后表连续取数并接受标准差')
        check(len(rng.draws)==2 and a.weight is not b.weight,'两个独立参数、各采样一次')
        return dict(生成器连续使用=True,参数独立=True)
    rng=TrackedRNG(31);E=build('Embedding',5,4,rng=rng);P=build('Embedding',6,4,rng=rng)
    blocks=[build('Block',4,2,7,rng=rng,init_std=.02,init_scale=.5) for _ in range(2)]
    matrices=[E.weight,P.weight];exits=[];norms=[]
    for block in blocks:
        n1,attn,branch,n2,mlp=block_parts(block,scope)
        check(type(attn) is scope['MultiHeadAttention'] and type(branch) is scope['MLPBranch'],'复用当前个人分支')
        check(type(mlp) is scope['MLP'],'分支复用当前个人MLP')
        check(all(type(v) is scope['Linear'] for v in [attn.qkv,attn.proj,mlp.fc1,mlp.fc2]),'投影真实复用个人Linear')
        matrices.extend([attn.qkv.W,attn.proj.W,mlp.fc1.W,mlp.fc2.W]);exits.extend([attn.proj.W,mlp.fc2.W]);norms.extend([n1,n2])
    check(len(rng.draws)==len(matrices),'全链路每张随机矩阵仅采样一次，共10次')
    expected=np.random.default_rng(31)
    for w in matrices:
        value=expected.normal(0,.02,w.data.shape)
        if any(w is e for e in exits):value*=.5
        close(w.data,value,'按字符表、位置表及各层注意力/MLP顺序，仅两出口缩放')
    for n in norms:close(n.w.data,np.ones(4),'归一化不缩放且不消耗随机数')
    close(rng.normal(),expected.normal(),'构造后的随机状态完全对应')
    if variant=='chain':return dict(个人子层复用=True,十次采样=True,只缩放出口=True,随机顺序=True)
    obj=blocks[0];params=obj.parameters();ids=[(id(p),id(p.data),id(p.grad)) for p in params]
    values=[p.data.copy() for p in params];x=np.arange(24.).reshape(2,3,4)/20;before=x.copy();g=np.ones_like(x)
    y=obj(x);dx=obj.backward(g);grads=[p.grad.copy() for p in params]
    close(x,before,'前向不改输入');check(y.shape==dx.shape==x.shape,'前后向形状保持')
    obj.backward(g)
    for p,v,g1 in zip(params,values,grads):close(p.data,v,'前后向不重新初始化');close(p.grad,2*g1,'重复反向累计')
    obj(x);check(ids==[(id(p),id(p.data),id(p.grad)) for p in obj.parameters()],'参数和数组身份保持')
    obj.zero_grad()
    for p in params:close(p.grad,np.zeros_like(p.grad),'沿用个人清零')
    return dict(前后向保持=True,参数身份保持=True,梯度累计和清零=True)
