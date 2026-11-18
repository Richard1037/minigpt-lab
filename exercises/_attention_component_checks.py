"""在提交的完整模块中检查原小题步骤；只追踪边界，不执行小题或参考模块。"""
import numpy as np


def behavior(scope,c):
    source=c['input']['source'];args=[np.asarray(a,dtype=np.float64) for a in c['input']['args']]
    expected=np.asarray(c['expected'],dtype=np.float64);captured=[]
    def require(ok,msg):
        if not ok:raise AssertionError(msg)
    def eye(prefix,n):return np.broadcast_to(np.eye(n),(*prefix,n,n)).copy()
    if source=='att_merge':
        x=args[0];B,H,T,d=x.shape;obj=scope['MultiHeadAttention'](H*d,H)
        obj.core.forward=lambda q,k,v:x.copy()
        original=obj.proj.forward
        def capture(a):captured.append(a.copy());return original(a)
        obj.proj.forward=capture;obj.forward(np.zeros((B,T,H*d)))
        require(len(captured)==1,'合头结果必须进入持久输出投影一次')
        actual=captured[0]
    else:
        obj=scope['AttentionCore'](source=='att_probs');original=obj.softmax.forward
        if source=='att_scores':
            q,k=args;v=np.zeros((*k.shape[:-1],2))
            def capture(s):captured.append(s.copy());return original(s)
            obj.softmax.forward=capture;obj.forward(q.copy(),k.copy(),v)
            require(len(captured)==1,'完整前向应将打分结果交给持久 Softmax 一次')
            actual=captured[0]
        elif source=='att_probs':
            s=args[0];prefix=s.shape[:-2];n=s.shape[-1]
            obj.forward(s.copy()*np.sqrt(n),eye(prefix,n),np.zeros((*prefix,n,2)))
            actual=obj._a
        else:
            a=args[0];prefix=a.shape[:-2];tq,tk=a.shape[-2:]
            # 在Softmax边界给定真实概率，隔离要检查的读取/回流；缓存仍由个人Softmax产生。
            def fixed_probabilities(s):
                with np.errstate(divide='ignore'):logits=np.log(a)
                return original(logits)
            obj.softmax.forward=fixed_probabilities
            if source=='att_weighted':
                v=args[1];actual=obj.forward(np.zeros((*prefix,tq,2)),np.zeros((*prefix,tk,2)),v.copy())
            elif source=='att_value_backward':
                g=args[1];obj.forward(np.zeros((*prefix,tq,2)),np.zeros((*prefix,tk,2)),np.zeros((*prefix,tk,g.shape[-1])))
                actual=obj.backward(g.copy())[2]
            elif source=='att_score_backward':
                da=args[1];identity=eye(prefix,tk)
                obj.forward(np.zeros((*prefix,tq,tk)),identity,identity.copy())
                # K为单位矩阵，dQ=dS/sqrt(d)，从真实返回值还原分数梯度。
                actual=obj.backward(da.copy())[0]*np.sqrt(tk)
            else:raise ValueError(source)
    from _assessment import diagnose
    require(isinstance(actual,np.ndarray),'该步骤必须产生 NumPy 数组')
    errors=diagnose(actual,c['expected'],path=c['label'],atol=1e-10,rtol=0)
    require(not errors,'；'.join(errors))
    return dict(完整类=c['input']['lesson'],检查步骤=source,产物形状=list(actual.shape),预期来源='保留的原公开样例',全部数值一致=True)
