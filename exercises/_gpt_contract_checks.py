"""第七部分行为检查；仅检查当前提交/个人子层，不代写计算。"""
import numpy as np
from _block_inspection import block_parts, learner_type


def behavior(scope, case):
    key=case['input']['lesson'];variant=case['input']['variant']
    def check(ok,msg):
        if not ok:raise AssertionError(msg)
    def close(a,b,label):
        a,b=np.asarray(a),np.asarray(b)
        check(a.shape==b.shape and np.allclose(a,b,atol=2e-9,rtol=2e-7),label)
    if key=='t_data':
        text='b a\nb a';vocab,ids=scope['tokenize'](text)
        check(vocab==['\n',' ','a','b'],'按字符排序去重，保留空格与换行')
        check(isinstance(ids,np.ndarray) and ids.dtype==np.int64,'编号必须为int64数组')
        close(ids,[3,1,2,0,3,1,2],'按原文顺序编码')
        check(''.join(vocab[int(i)] for i in ids)==text,'同一词表能还原原文')
        if variant=='build':return dict(字符顺序=True,空格换行保留=True,编解码一致=True)
        for T,starts in [(1,[5,0,5]),(3,[3,0,3])]:
            original=ids.copy();ss=np.array(starts,dtype=np.int64);saved=ss.copy()
            x,y=scope['make_batch'](ids,ss,T)
            check(isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and x.dtype==y.dtype==np.int64,'批次为int64数组')
            close(x,[[original[s+t] for t in range(T)] for s in starts],'输入窗口顺序与边界')
            close(y,[[original[s+t+1] for t in range(T)] for s in starts],'目标逐位置右移一位')
            check(np.array_equal(ids,original) and np.array_equal(ss,saved),'不修改输入编号与起点')
            x[...] = -1;y[...] = -2
            check(np.array_equal(ids,original),'返回值不能与原编号共用可写存储')
        return dict(最后合法起点=True,顺序与重复起点=True,输入保持=True)

    GPT=scope['GPT'];args=(5,4,2,7,2,5)
    obj=GPT(*args,seed=19)
    check(type(obj.token) is scope['Embedding'] and type(obj.position) is scope['Embedding'],'两表须复用当前个人Embedding')
    check(type(obj.norm) is scope['RMSNorm'],'末尾复用当前个人RMSNorm')
    check(len(obj.blocks)==2 and len({id(v) for v in obj.blocks})==2,'创建两个独立个人Block')
    matrices=[obj.token.weight,obj.position.weight];exits=[];norms=[obj.norm];children=[]
    P=scope['Parameter'];Block=scope['Block']
    for i,block in enumerate(obj.blocks):
        check(type(block) is Block,'实际复用当前个人Block')
        # 本地GPT作用域不必直接导入Block的全部内部类型。
        block_scope=dict(scope)
        block_scope.setdefault('MLPBranch',getattr(Block.__init__,'__globals__',{}).get('MLPBranch'))
        block_scope.setdefault('MultiHeadAttention',getattr(Block.__init__,'__globals__',{}).get('MultiHeadAttention'))
        n1,attn,branch,n2,mlp=block_parts(block,block_scope)
        matrices.extend([attn.qkv.W,attn.proj.W,mlp.fc1.W,mlp.fc2.W]);exits.extend([attn.proj.W,mlp.fc2.W]);norms.extend([n1,n2]);children.append((block,'block'+str(i)))
    check(len({id(p) for p in matrices})==len(matrices),'二维参数不得意外跨层共享')
    params=obj.parameters();check(all(type(p) is P for p in params),'使用个人Parameter')
    check(set(map(id,matrices))=={id(p) for p in params if p.data.ndim==2},'所有二维参数恰好初始化一次')
    rng=np.random.default_rng(19)
    for param in matrices:
        expected=rng.normal(0,.02,param.data.shape)
        if any(param is v for v in exits):expected/=np.sqrt(4)
        close(param.data,expected,'单一随机流、语义顺序及两个出口缩放')
    for norm in norms:close(norm.w.data,np.ones(4),'归一化缩放保留全一')
    for p in params:
        if p.data.ndim==1 and not any(p is n.w for n in norms):close(p.data,np.zeros_like(p.data),'偏置保留零')
        close(p.grad,np.zeros_like(p.grad),'初始梯度为零')
    if variant=='build':
        other=GPT(*args,seed=20)
        check(not np.array_equal(obj.token.weight.data,other.token.weight.data),'不同seed须改变随机样本')
        for bad in [(5,4,2,7,0,5),(5,5,2,7,2,5)]:
            try:GPT(*bad)
            except ValueError:pass
            else:raise AssertionError('拒绝零层数与不可整除头宽')
        return dict(个人子层复用=True,独立层=True,单一随机流与出口=True,二维参数张量数=len(matrices))
    events=[]
    for child,name in children+[(obj.norm,'norm')]:
        for method in ['forward','backward']:
            fn=getattr(child,method)
            def wrapped(*a,_fn=fn,_tag=name+'.'+method,**kw):events.append(_tag);return _fn(*a,**kw)
            setattr(child,method,wrapped)
    hidden={};norm_forward=obj.norm.forward
    def remember(x):
        y=norm_forward(x);hidden['h']=y.copy();return y
    obj.norm.forward=remember
    ids=np.array([[0,1,0],[2,1,2]],dtype=np.int64);original=ids.copy();weights=[p.data.copy() for p in params]
    buffers=[id(p.grad) for p in params];y=obj.forward(ids)
    g=np.arange(y.size,dtype=np.float64).reshape(y.shape)/19;saved=g.copy();h=hidden['h']
    dx=obj.backward(g);check(dx is None,'整数编号的反向须返回None')
    check(events==['block0.forward','block1.forward','norm.forward','norm.backward','block1.backward','block0.backward'],'按实际前向顺序与逆序复用持久层')
    check(np.array_equal(ids,original) and np.array_equal(g,saved),'输入/上游保持原值')
    head=g.reshape(-1,5).T@h.reshape(-1,4)
    # 缺席的词表行只收到输出头贡献；查表行的两路由原差分用例完整核对。
    close(obj.token.weight.grad[4],head[4],'未查表行收到输出头贡献')
    first=[p.grad.copy() for p in params];obj.backward(g)
    for p,grad,w in zip(params,first,weights):close(p.grad,2*grad,'重复反向累加');close(p.data,w,'反向不更新参数')
    for _ in range(2):obj.forward(ids)
    for p,w,grad in zip(params,weights,first):close(p.data,w,'前向不能重复初始化或缩放');close(p.grad,2*grad,'前向不清梯度')
    obj.zero_grad()
    check([id(p.grad) for p in params]==buffers,'清零保留缓冲身份')
    for p in params:close(p.grad,np.zeros_like(p.grad),'清零全部个人参数')
    return dict(完整批次形状=list(y.shape),持久层逆序=True,未查表行输出梯度=True,前向不更新=True,累计及原地清零=True)
