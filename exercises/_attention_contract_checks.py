"""Part06 的构造、实际复用和缓存检查；不导入参考模块。"""
import numpy as np


def behavior(scope, case):
    key=case['input']['lesson'];variant=case['input']['variant']
    cls={'t_attention':'AttentionCore','t_mha':'MultiHeadAttention','t_block':'Block','t_dropout':'Dropout','t_layernorm':'LayerNorm'}[key]
    args={'t_attention':(True,),'t_mha':(6,3),'t_block':(6,3,9),'t_dropout':(.3,5),'t_layernorm':(6,1e-4)}[key]
    def check(ok,msg):
        if not ok:raise AssertionError(msg)
    def close(a,b,label):
        aa,bb=np.asarray(a),np.asarray(b)
        check(aa.shape==bb.shape and np.allclose(aa,bb,atol=2e-7,rtol=2e-6),label)
    make=lambda:scope[cls](*args)
    obj,other=make(),make();rng=np.random.default_rng(64)
    if key=='t_attention':x=[rng.normal(size=(2,3,4,2)) for _ in range(3)]
    else:x=[rng.normal(size=(2,4,6))]
    if variant=='build':
        check(isinstance(obj,scope['Module']),'必须继承当前个人 Module')
        if key=='t_attention':
            check(type(obj.softmax) is scope['Softmax'] and obj.softmax is not other.softmax,'创建独立的当前个人 Softmax')
            check(obj.causal is True and obj.parameters()==[],'核心模式或空参数错误')
            children=[(obj.softmax,'softmax')]
            try:obj.forward(np.ones((2,2,2,3)),np.ones((2,2,3,3)),np.ones((2,2,3,4)))
            except ValueError:pass
            else:raise AssertionError('因果自注意力必须拒绝 Q/K 长度不一致')
        elif key=='t_mha':
            check(type(obj.qkv) is scope['Linear'] and type(obj.proj) is scope['Linear'],'投影须复用当前个人 Linear')
            check(type(obj.core) is scope['AttentionCore'] and obj.core.causal,'复用当前因果核心')
            check(obj.qkv.W.data.shape==(6,18) and obj.proj.W.data.shape==(6,6),'构造投影宽度错误')
            check(len(obj.parameters())==4,'应收集两层投影的四个参数')
            try:scope[cls](5,2)
            except ValueError:pass
            else:raise AssertionError('不能整除的头数必须拒绝')
            children=[(obj.qkv,'qkv'),(obj.core,'core'),(obj.proj,'proj')]
        elif key=='t_block':
            from _block_inspection import block_parts,learner_type
            norm1,attn,branch,norm2,mlp=block_parts(obj,scope)
            check(type(norm2) is learner_type(scope,'RMSNorm') and type(mlp) is learner_type(scope,'MLP'),'分支复用当前个人归一化与 MLP')
            check(norm1 is not norm2,'两次归一化实例必须独立')
            check(type(mlp.fc1) is learner_type(scope,'Linear') and type(attn.qkv) is learner_type(scope,'Linear'),'两条分支均复用同一个个人 Linear 类')
            check(len(obj.parameters())==10,'完整 Block 应收集十个不重复参数张量')
            children=[(norm1,'norm1'),(attn,'attn'),(branch,'ffn')]
        elif key=='t_dropout':
            check(obj.p==.3 and obj.parameters()==[],'概率与空参数列表错误')
            for v in [-.1,1.]:
                try:scope[cls](v)
                except ValueError:pass
                else:raise AssertionError('非法丢弃概率须拒绝')
            # 评估不应消耗随机状态；下次训练须与未评估的同种子实例相同。
            obj.forward(x[0],training=False)
            close(obj.forward(x[0]),other.forward(x[0]),'评估推进了 RNG 或同种子不复现')
            return dict(合法概率=True,评估不推进随机状态=True,无可训练参数=True)
        else:
            check(obj.eps==1e-4 and len(obj.parameters())==2,'构造 eps 或参数注册错误')
            close(obj.w.data,np.ones(6),'缩放初值');close(obj.b.data,np.zeros(6),'偏置初值')
            check(type(obj.w) is scope['Parameter'] and type(obj.b) is scope['Parameter'],'使用当前个人 Parameter')
            children=[]
        params=obj.parameters();op=other.parameters()
        check(len(set(map(id,params)))==len(params),'参数须按身份去重')
        for a,b in zip(params,op):
            check(not np.shares_memory(a.data,b.data) and not np.shares_memory(a.grad,b.grad),'实例参数存储必须独立')
            close(a.grad,np.zeros_like(a.data),'初始梯度应为零')
        events=[]
        # 在实例上追踪调用，既支持网页共享作用域，也支持本地分模块导入。
        for child,name in children:
            for method in ['forward','backward']:
                original=getattr(child,method)
                def wrapped(*a,_fn=original,_tag=name+'.'+method,**kw):
                    events.append(_tag);return _fn(*a,**kw)
                setattr(child,method,wrapped)
        y=obj.forward(*x);obj.backward(np.ones_like(y))
        if key=='t_attention':expected=['softmax.forward','softmax.backward']
        elif key=='t_mha':expected=['qkv.forward','core.forward','proj.forward','proj.backward','core.backward','qkv.backward']
        elif key=='t_block':expected=['norm1.forward','attn.forward','ffn.forward','ffn.backward','attn.backward','norm1.backward']
        else:expected=[]
        check(events==expected,'应调用构造时的持久子层一次并按逆序返回，实际：'+str(events))
        return dict(个人类身份=True,持久子层调用顺序=events,参数张量数=len(params),实例隔离=True)

    y=obj.forward(*x);baseline=other.forward(*[v.copy() for v in x]);close(y,baseline,'同配置前向')
    g=rng.normal(size=y.shape);saved=g.copy()
    for value in x:value[...] = 91
    y[...] = -83
    got=obj.backward(g);want=other.backward(saved)
    a=list(got) if isinstance(got,tuple) else [got];b=list(want) if isinstance(want,tuple) else [want]
    for aa,bb in zip(a,b):close(aa,bb,'原输入或返回值修改破坏反向缓存')
    close(g,saved,'反向修改了调用方上游')
    grads=[v.grad.copy() for v in obj.parameters()];data=[v.data.copy() for v in obj.parameters()]
    twice=obj.backward(g)
    for aa,bb in zip(list(twice) if isinstance(twice,tuple) else [twice],b):close(aa,bb,'反复反向污染缓存')
    for par,grad,weight in zip(obj.parameters(),grads,data):
        close(par.grad,2*grad,'参数梯度未累计');close(par.data,weight,'反向修改权重')
    buffers=[id(v.grad) for v in obj.parameters()];obj.zero_grad()
    for par,ident in zip(obj.parameters(),buffers):
        check(id(par.grad)==ident,'清零须保留梯度缓冲身份');close(par.grad,np.zeros_like(par.grad),'清零失败')
    return dict(完整多批次形状=list(baseline.shape),输入返回值缓存隔离=True,上游保持=True,重复反向累计=True,原地清零=True)
