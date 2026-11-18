"""Part08 输入/状态行为检查；所有计算均调用当前提交的个人实现。"""
import numpy as np


def behavior(scope, case):
    key=case['input']['lesson'];variant=case['input']['variant']
    def check(ok,message):
        if not ok:raise AssertionError(message)
    def close(actual,expected,label):
        a,b=np.asarray(actual),np.asarray(expected)
        check(a.shape==b.shape and np.allclose(a,b,rtol=2e-7,atol=2e-9),label)
    if key=='t_optim':
        P=scope['Parameter'];clip=scope['clip_grad'];Adam=scope['AdamW']
        if variant=='empty':
            norm=clip([],1.0)
            check(isinstance(norm,np.ndarray) and norm.shape==(),'返回零维 NumPy 范数数组')
            close(norm,0.,'空参数范数为零');Adam([]).step()
            ps=[P(np.ones((2,2))),P(np.ones(2))]
            ids=[(id(p.data),id(p.grad)) for p in ps]
            close(clip(ps,0.),0.,'全零梯度不产生NaN')
            check(Adam(ps,lr=.1,decay=.2).step() is None,'step 返回 None')
            close(ps[0].data,np.full((2,2),.98),'二维权重解耦衰减')
            close(ps[1].data,np.ones(2),'一维参数不衰减')
            check(ids==[(id(p.data),id(p.grad)) for p in ps],'保留 data/grad 数组身份')
            return dict(空参数=True,零梯度=True,衰减范围=True,数组身份=True)
        ps=[P([[1.,-2.],[.5,3.]]),P([.2,-.4])]
        ids=[(id(p.data),id(p.grad)) for p in ps]
        model=scope['Module']()
        model.weight,model.bias,model.shared_weight=ps[0],ps[1],ps[0]
        params=model.parameters()
        check(list(map(id,params))==list(map(id,ps)),'Module.parameters() 按对象身份收集唯一参数')
        opt=Adam(params,lr=.03,beta1=.8,beta2=.7,eps=1e-6,decay=.2)
        check(isinstance(opt.params,list) and list(map(id,opt.params))==list(map(id,params)),
              'AdamW 按传入顺序保存原参数引用')
        theta=[p.data.copy() for p in ps];ms=[np.zeros_like(p.data) for p in ps];vs=[v.copy() for v in ms]
        for t in (1,2,3):
            gs=[np.array([[3.,-4.],[2.,1.]])*(-1 if t==2 else 1),np.array([2.,-1.])*t]
            if t==3:gs=[np.zeros_like(g) for g in gs]
            for p,g in zip(ps,gs):p.grad[...] = g
            old=[p.data.copy() for p in ps];norm=np.sqrt(sum(np.sum(g*g) for g in gs));a=min(1.,2./(norm+1e-12))
            close(clip(model.parameters(),2.),norm,'模型参数列表的全参数共同范数')
            for j,p in enumerate(ps):
                g=gs[j]*a;close(p.grad,g,'统一缩放每个参数梯度');close(p.data,old[j],'裁剪不更新权重')
                ms[j]=.8*ms[j]+.2*g;vs[j]=.7*vs[j]+.3*g*g
                theta[j]=theta[j]*(1-.03*.2 if p.data.ndim>=2 else 1)-.03*(ms[j]/(1-.8**t))/(np.sqrt(vs[j]/(1-.7**t))+1e-6)
            opt.step()
            for p,w,g in zip(ps,theta,gs):
                close(p.data,w,'跨步历史、偏差校正与解耦衰减')
                close(p.grad,g*a,'step 不清零或改写梯度')
        check(ids==[(id(p.data),id(p.grad)) for p in ps],'更新保留数组身份')
        return dict(连续三步=True,混合形状=True,共享仅更新一次=True,零梯度历史保留=True)

    model=scope['GPT'](5,4,2,7,1,5)
    x=np.array([[0,1,0],[2,1,2]],dtype=np.int64);y=np.array([[1,0,2],[1,2,0]],dtype=np.int64)
    original=(x.copy(),y.copy());ps=model.parameters();before=[p.data.copy() for p in ps]
    train=scope['train']
    if variant=='zero':
        for p in ps:p.grad[...] = .3
        result=train(model,x,y,0,.01)
        check(isinstance(result,np.ndarray) and result.shape==(0,),'零步返回一维空 NumPy 数组')
        for p,w in zip(ps,before):close(p.data,w,'零步保留权重');close(p.grad,np.full_like(p.grad,.3),'零步保留梯度')
    else:
        # 用函数自己的全局作用域，兼容网页同域与真实分模块导入。
        env=train.__globals__;names=('CrossEntropy','AdamW','clip_grad');saved={n:env[n] for n in names}
        events=[];losses=[];optimizers=[]
        def wrap(obj,name,event,after=None):
            fn=getattr(obj,name)
            def call(*a,**kw):
                events.append(event);r=fn(*a,**kw)
                if after:after(r)
                return r
            setattr(obj,name,call)
        for method in ('zero_grad','forward','backward'):wrap(model,method,'model.'+method)
        def loss_factory(*a,**kw):
            events.append('loss.init');obj=saved['CrossEntropy'](*a,**kw)
            wrap(obj,'forward','loss.forward',lambda r:losses.append(float(r)));wrap(obj,'backward','loss.backward');return obj
        def optim_factory(*a,**kw):
            events.append('optim.init');obj=saved['AdamW'](*a,**kw);optimizers.append(obj);wrap(obj,'step','optim.step');return obj
        def clipping(*a,**kw):events.append('clip');return saved['clip_grad'](*a,**kw)
        try:
            env.update(CrossEntropy=loss_factory,AdamW=optim_factory,clip_grad=clipping)
            result=train(model,x,y,2,.01)
        finally:env.update(saved)
        check(len(optimizers)==1 and events[:2] in (['loss.init','optim.init'],['optim.init','loss.init']),'循环外各创建一个个人损失和优化器')
        check(events[2:]==['model.zero_grad','model.forward','loss.forward','loss.backward','model.backward','clip','optim.step']*2,'每步按清零/前向/损失/反向/裁剪/更新执行')
        check(isinstance(result,np.ndarray) and result.shape==(2,),'返回长度等于 steps 的一维 NumPy 数组')
        close(result,losses,'记录每次更新前损失')
        check(any(not np.array_equal(p.data,w) for p,w in zip(ps,before)),'个人模型参数实际更新')
    check(np.array_equal(x,original[0]) and np.array_equal(y,original[1]),'训练不修改输入/目标编号')
    return dict(个人组件调用=True,连续优化器状态=True,返回与输入保持=True)
