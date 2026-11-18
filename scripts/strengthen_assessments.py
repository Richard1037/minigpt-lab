"""分层验收与最终项目：由课程构建调用，不覆盖个人练习。"""
import copy
import inspect
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'solutions')]
import foundation_layers as base
import transformer_layers as ref
from foundation_core import Parameter,Module,initialize
from _assessment import CATEGORIES
from _transformer_checks import probe
book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
for m in book['lessons'].values():m['cases']=[c for c in m.get('cases',[]) if not c.get('expanded')]
def add(id,c,category='数值与形状'):
    c.update(expanded=True,category=category);book['lessons'][id]['cases'].append(c)
def special(id,kind,label,**p):
    operations={
      'gradient':'构造当前个人模块；以 L=Σ(Y⊙G) 为标量目标，对输入及每个参数张量选定坐标作中心差分，h=1e-5；与 backward 的解析梯度比较。每个参数张量最多检查 coordinates 个坐标，不是逐元素全量检查。',
      'cache':'构造两份相同参数的当前个人模块。第一份 forward 后把调用方输入加 37（整数 ids 改为下一个编号）再 backward；第二份用保留的原始输入 forward/backward。两者输入梯度和全部参数梯度必须一致，验证缓存未受调用方修改影响。',
      'accumulation':'同一 X 分别以 G、2G 反向得到两份梯度，再连续累加并检查等于两份之和；zero_grad 后 grad=0，data 保持不变。',
      'softmax':'输入 shape=(2,3,5)、seed=41 的正态数组乘 3；比较 X 与 X+10000 的概率、每行概率和，以及全一上游的零输入梯度。',
      'causal':'分别改变最后一个 token 或最后一个 K/V；比较所有之前位置的输出。然后仅从首个 query 反向，未来位置的梯度必须为零。',
      'shared':'输入 ids=[[0,1],[1,0]]，反向后检查共享参数只出现一次；对未查表的最后一行词表参数中心差分，验证输出头仍为其提供梯度。',
      'overfit':'ids=[[0,1,2],[1,2,3]]，targets=(ids+1)%V；调用 train(model,ids,targets,steps,0.02)，最终损失必须小于初始损失的一半。',
      'evaluation':'所有参数梯度预置为 0.7；仅 forward(ids)，记录 mean(logits²)，不执行 backward/step。所有 data/grad 必须逐元素不变。',
      'generation':'调用 generate(model,[[0,1,2],[1,2,3]],6,0.8,1,7)，预期 shape=(2,9)、整数编号、前缀保留；重复运行须复现，权重和梯度不变。',
      'checkpoint':'对随机梯度更新一次，保存参数/优化器/RNG；原地扰乱参数、m/v、t、学习率与随机状态后加载，单独检查 focus 指定的对象。',
      'dropout':'Dropout(0.3,5)；X=ones((2,3,8))，连续两次训练 forward 的掩码应变化；backward(ones) 复用第二次掩码；评估 forward/backward 均为恒等。',
      'infrastructure':'Parameter 复制数值并建立 float64 同形零梯度；Module 递归列表/字典、去重共享对象、跳过下划线缓存，zero_grad 不改 data。',
      'project_step':'ids=[[0,1,2],[1,2,3]]，targets=(ids+1)%V，旧梯度预置 13；AdamW(lr=0.01)。project_step 返回更新前零维平均损失，参数应等于清零→交叉熵→反向→裁剪1→更新一次的结果。'
    }
    p['operation']=operations[kind]
    if kind in ['causal','overfit','evaluation','generation','project_step'] and p.get('class')=='GPT':p['ids']=[[0,1,2,3],[1,2,3,4]] if kind=='causal' else [[0,1,2],[1,2,3]]
    if kind=='gradient':
        r=np.random.default_rng(p.get('seed',41));cls=p['class'];shape=tuple(p['shape'])
        if cls=='AttentionCore':arr=[r.normal(size=shape) for _ in range(3)];names=['Q','K','V']
        elif cls in ['GPT','Embedding']:arr=[r.integers(0,p['init'][0],size=shape)];names=['ids']
        else:arr=[r.normal(size=shape)];names=['logits' if cls=='CrossEntropy' else 'X']
        obj=getattr(ref,cls,getattr(base,cls,None))(*p['init'])
        if cls=='CrossEntropy':targets=r.integers(0,shape[-1],size=shape[:-1]);y=obj.forward(arr[0],targets);p['targets']=targets.tolist()
        else:y=obj.forward(*arr)
        p['sample']={name:a.tolist() for name,a in zip(names,arr)};p['sample']['G']=r.normal(size=np.asarray(y).shape).tolist()
    add(id,dict(assessment=kind,label=label,input=p,expected={'验收目标':label,'判断方式':operations[kind]}),CATEGORIES[kind])

def numeric(cls,init,x,g,label,params=None,repeat=False):
    obj=getattr(base,cls)(*init)
    for name,v in (params or {}).items():getattr(obj,name).data[:]=v
    x=np.array(x,dtype=np.int64 if cls=='Embedding' else float);g=np.array(g)
    y=obj(x).copy();expected={'Y':y.tolist()}
    def diff(a):
        out=np.zeros_like(a,dtype=float)
        for ix in np.ndindex(a.shape):
            old=a[ix];h=1e-5
            a[ix]=old+h;plus=np.sum(obj(x)*g);a[ix]=old-h;minus=np.sum(obj(x)*g);a[ix]=old
            out[ix]=(plus-minus)/(2*h)
        return out
    expected['dX']=None if cls=='Embedding' else diff(x).tolist()
    from _assessment import named_parameters
    for name,p in named_parameters(obj):expected[name+'.grad']=(diff(p.data)*(2 if repeat else 1)).tolist()
    return dict(label=label,**{'class':cls},init=init,X=x.tolist(),G=g.tolist(),parameters=params or {},expected=expected,repeat=repeat,check_zero=True)
rng=np.random.default_rng(20261107)
# 数组练习：尺寸/轴/索引结构分别变化，而非只替换数值。
inputs={
 'shape':[[rng.normal(size=(2,3,4)).tolist()],[rng.normal(size=(2,2,3,4)).tolist()]],
 'axes':[[rng.normal(size=(3,5)).tolist()],[np.zeros((4,2)).tolist()]],
 'broadcast':[[rng.normal(size=(3,4)).tolist(),[-2,0,1,3]],[rng.normal(size=(2,5)).tolist(),[1,2,3,4,5]]],
 'matmul':[[rng.normal(size=(3,4)).tolist(),rng.normal(size=(4,2)).tolist()],[rng.normal(size=(2,3)).tolist(),np.zeros((3,5)).tolist()]],
 'linear':[[rng.normal(size=(3,2,5)).tolist(),rng.normal(size=(5,2)).tolist(),[1,-1]],[rng.normal(size=(2,2,3)).tolist(),np.zeros((3,4)).tolist(),[1,2,3,4]]],
 'index':[[rng.normal(size=(5,4)).tolist(),[4,0,4,2],3],[rng.normal(size=(3,5)).tolist(),[2,1,0],4]],
 'reduce':[[rng.normal(size=(3,5)).tolist()],[np.full((2,4),7).tolist()]],
 'reshape':[[rng.normal(size=(3,2,5)).tolist()],[rng.normal(size=(2,4,3)).tolist()]],
 'transpose':[[rng.normal(size=(3,2,5)).tolist()],[rng.normal(size=(2,4,3)).tolist()]],
 'heads':[[rng.normal(size=(3,2,12)).tolist(),4],[rng.normal(size=(2,4,6)).tolist(),3]],
 'copy':[[rng.normal(size=(3,5)).tolist()],[np.zeros((2,4)).tolist()]],
 'scatter':[[2,0,2,2],[[1,2,3],[4,5,6],[-1,-2,-3],[2,1,0]],np.zeros((4,3)).tolist()],
 'random':[[19,[2,3,4]],[3,[3,5]]]
}
labels=['改变非方形尺寸，禁止硬编码原样例','零值/不同轴长/重复编号的结构检查']
for id,batches in inputs.items():
    if id=='scatter':batches=[batches,[[1,0,1],[[1,2],[3,4],[5,6]],np.ones((3,2)).tolist()]]
    meta=book['lessons'][id];scope={'np':np};exec(meta['reference'],scope)
    for n,args in enumerate(batches):
        dtypes=(['int64','float64','float64'] if id=='scatter' else ['float64','int64','int64'] if id=='index' else ['float64','int64'] if id=='heads' else ['int64','int64'] if id=='random' else ['float64']*len(args))
        converted=[np.array(a,dtype=t) for a,t in zip(args,dtypes)]
        add(id,dict(label=labels[n],args=args,dtypes=dtypes,expected=scope['solve'](*converted).tolist()))
# 完整基础层：不同形状 + 零上游 + 独立差分 + 状态。
for id,cls,init in [('layer_scale','Scale',[1.7]),('layer_linear','Linear',[3,5]),('layer_embedding','Embedding',[5,3]),('layer_gelu','GELU',[]),('layer_rmsnorm','RMSNorm',[3]),('layer_mlp','MLP',[3,5])]:
    shape=(2,3) if cls=='Embedding' else (2,2,3)
    x=rng.integers(0,5,size=shape) if cls=='Embedding' else rng.normal(size=shape)
    outshape=(*shape,3) if cls=='Embedding' else (*shape[:-1],5) if cls=='Linear' else shape
    add(id,numeric(cls,init,x,rng.normal(size=outshape),'不同特征数与非单位上游：核对所有梯度'),'前向与反向')
    add(id,numeric(cls,init,x,np.zeros(outshape),'零上游：所有输入与参数梯度应为零'),'反向传播')
    shape4=(2,2,2,3) if cls not in ['Embedding'] else (2,2,2)
    x4=rng.integers(0,5,size=shape4) if cls=='Embedding' else rng.normal(size=shape4)
    out4=(*shape4,3) if cls=='Embedding' else (*shape4[:-1],5) if cls=='Linear' else shape4
    add(id,numeric(cls,init,x4,rng.normal(size=out4),'更高维：保留所有前置轴，不展平后丢失形状'),'形状与组合')
    special(id,'gradient','独立中心差分：输入与每个参数张量的选定坐标',**{'class':cls},init=init,shape=list(shape),seed=81,coordinates=4)
    special(id,'cache','前向后修改调用方输入：反向仍使用本次原始缓存',**{'class':cls},init=init,shape=list(shape),seed=87)
    if cls not in ['GELU']:special(id,'accumulation','两种上游梯度累加；清零不得改变权重',**{'class':cls},init=init,shape=list(shape))
# 为无参数 GELU 添加分段范围，避免差分只落在正区间。
add('layer_gelu',numeric('GELU',[],np.array([[[-8.,-2.,0.],[.5,2.,8.]],[[1.,-1.,3.],[-3.,.1,-.1]]]),np.ones((2,2,3)),'负饱和区、零点、正区间的完整导数'),'边界与稳定性')
special('infrastructure','infrastructure','输入复制、嵌套字典/列表、共享去重、跳过缓存与清零')
# 后续模块多尺寸和零上游。
for id,cls,init,shape in [('t_softmax','Softmax',[],[2,2,5]),('t_mha','MultiHeadAttention',[6,3],[2,3,6]),('t_block','Block',[4,2,7],[2,2,4]),('t_layernorm','LayerNorm',[5],[2,2,5])]:
    for zero in [False,True]:
        x=rng.normal(size=shape);g=np.zeros(shape) if zero else rng.normal(size=shape)
        c=dict(mode='module',label='零上游：梯度全零' if zero else '改变序列长度与特征数：核对前向与反向',input=dict(**{'class':cls},init=init,X=x.tolist(),G=g.tolist()))
        out=probe(vars(ref),c)
        from _foundation_checks import pack
        c['expected']=pack(out);add(id,c,'反向传播' if zero else '形状与组合')
    special(id,'gradient','独立差分：输入及所有参数张量的采样坐标',**{'class':cls},init=init,shape=shape,coordinates=4,seed=51)
    if cls!='Softmax':special(id,'accumulation','不同上游累加与 zero_grad 状态检查',**{'class':cls},init=init,shape=shape)
special('t_softmax','softmax','平移不变、概率行和、全一上游梯度为零')
for causal in [True,False]:special('t_attention','gradient','因果注意力三路梯度差分' if causal else '非因果注意力三路梯度差分',**{'class':'AttentionCore'},init=[causal],shape=[2,2,3,2],coordinates=8,seed=42)
special('t_attention','causal','未来 K/V 改变不影响过去；未来 dK/dV 为零',**{'class':'AttentionCore'},init=[True])
# loss 梯度及不同词表/标签。
special('t_loss','gradient','平均交叉熵：随机上游标量与 logits 差分',**{'class':'CrossEntropy'},init=[],shape=[2,3,5],coordinates=12,seed=12)
for shape in [(3,2,7),(2,4,3)]:
    c=dict(mode='loss',label=f'批次/词表变化 {shape}：每个位置只平均一次',input=dict(**{'class':'CrossEntropy'},X=rng.normal(size=shape).tolist(),targets=rng.integers(0,shape[-1],size=shape[:-1]).tolist()))
    c['expected']={k:v.tolist() for k,v in probe(vars(ref),c).items()};add('t_loss',c,'数值与形状')
special('t_dropout','dropout','随机状态推进、最近掩码复用、评估恒等')
for train in [True,False]:
    shape=[2,2,6];c=dict(mode='module',label='不同概率/种子的训练掩码' if train else '不同尺寸的评估分支',input=dict(**{'class':'Dropout'},init=[.5,31],X=rng.normal(size=shape).tolist(),G=rng.normal(size=shape).tolist(),kwargs={'training':train}))
    c['expected']={k:v.tolist() for k,v in probe(vars(ref),c).items()};add('t_dropout',c,'训练与评估')
gptinit=[5,4,2,7,2,5,11]
for kind,label in [('gradient','所有参数张量差分：包含位置表与共享词表'),('causal','未来 token 隔离与未来位置梯度为零'),('shared','共享词表只收集一次，未查表行仍收到输出头梯度'),('evaluation','验证前向不更新权重或已有梯度')]:
    special('t_gpt',kind,label,**{'class':'GPT'},init=gptinit,shape=[2,3],seed=71,coordinates=3)
special('t_train','overfit','固定双批次：45 步后损失至少下降一半',**{'class':'GPT'},init=[5,8,2,16,1,5,3],steps=45)
special('t_train','evaluation','验证过程只能前向，保留现有权重和梯度',**{'class':'GPT'},init=gptinit)
special('t_generate','generation','两个提示超过窗口：前缀保留、可复现、无参数更新',**{'class':'GPT'},init=gptinit)
# 数据、优化器、采样、续训：各自增加结构性用例。
extras={
 't_data':[dict(mode='data',label='不同起点顺序与相同起点重复',input=dict(text='abcdeabcdeabcde',starts=[4,0,4],T=4)),dict(mode='data',label='含换行/空格的字符词表与最后合法起点',input=dict(text='a b\na b\na b\n',starts=[0,7],T=4))],
 't_optim':[dict(mode='optimizer',label='未触发裁剪时方向和范数保持',input=dict(weights=[[[1.,-1.],[2.,-2.]],[.1,.2]],gradients=[[[[.01,-.02],[.03,-.04]],[.02,.01]]],config={'lr':.01,'decay':0.},limit=10.)),dict(mode='optimizer',label='三次相反梯度：必须保留 m/v 与时间步',input=dict(weights=[[[1.,2.],[3.,4.]]],gradients=[[[[1.,-1.],[2.,-2.]]],[[[-1.,1.],[-2.,2.]]],[[[.1,.2],[.3,.4]]]],config={'lr':.01,'beta1':.8,'beta2':.95,'decay':.01},limit=2.))],
 't_sample':[dict(mode='sample',label='不截断全部词表：两个分布归一化后抽样',input=dict(X=[[1000,999,998,997],[0,-10,2,1]],temperature=1.3,top_k=0,seed=27)),dict(mode='sample',label='k=V 与低温度的批量抽样',input=dict(X=[[2,0,1,-1],[1,3,0,2]],temperature=.1,top_k=4,seed=2))],
 't_checkpoint':[dict(mode='checkpoint',label='不同模型宽度和种子下的下一步恢复一致',input=dict(init=[5,6,3,9,1,5,21],ids=[[0,1,2],[1,2,3]],targets=[[1,2,3],[2,3,4]],steps=1,lr=.01))]
}
for id,cases in extras.items():
    for c in cases:
        out=probe(vars(ref),c);c['expected']={k:v if isinstance(v,str) else np.asarray(v).tolist() for k,v in out.items()};add(id,c,'状态与边界')
for kind,label in [('project_step','完整训练入口：清零、损失、反向、裁剪、一次更新'),('gradient','完整模型：每个参数张量独立差分'),('causal','未来信息隔离与未来位置零梯度'),('shared','共享权重去重和两路梯度'),('overfit','固定批次过拟合：损失下降至少一半'),('evaluation','独立评估保持权重和梯度'),('generation','批量生成跨过窗口，保留提示与参数状态')]:
    special('t_project',kind,label,**{'class':'GPT'},init=[5,8,2,16,1,5,3],shape=[2,3],steps=45,coordinates=3,seed=19)
c=dict(mode='checkpoint',label='检查点：恢复后下一步更新与随机状态一致',input=dict(init=[5,4,2,8,1,5,3],ids=[[0,1,2],[1,2,3]],targets=[[1,2,3],[2,3,4]],steps=1,lr=.01))
c['expected']={k:np.asarray(v).tolist() for k,v in probe(vars(ref),c).items()};add('t_project',c,'恢复一致性')
# 注意力核心前的阶梯练习：每次只改变一个关键操作，最后组合到标准层。
def primitive(id,title,args,source,body,derivation,inputs):
    meta=dict(id=id,title=title,kind='array',args=args,goal=body,body='<h3>输入与输出</h3><p>'+body+'</p>',derivation='<p>'+derivation+'</p>',
        starter='def solve('+', '.join(args)+'):\n    raise NotImplementedError("TODO：实现本节计算")',reference=source,file=id+'.py',cases=[],steps=[body,'保留全部 batch/head；完成本节后在 AttentionCore 中使用相同计算。'],extension='<p>在完整 AttentionCore 任务中继续检查反向传播与因果性。</p>')
    ns={'np':np};exec(source,ns)
    for n,values in enumerate(inputs):
        arr=[np.asarray(v,dtype=float) for v in values]
        meta['cases'].append(dict(label=['完整多批次多头样例','非方形/不同轴长度','零值或边界尺寸'][n],args=[a.tolist() for a in arr],expected=ns['solve'](*arr).tolist(),category='形状与计算'))
    code='import numpy as np\n\n'+meta['starter']+f'\n\nif __name__ == "__main__":\n    from _checks import run_local\n    run_local("{id}", solve)\n'
    path=ROOT/'exercises'/meta['file']
    if not path.exists():path.write_text(code,encoding='utf-8')
    meta['local_code']=path.read_text(encoding='utf-8');book['lessons'][id]=meta
q=rng.normal(size=(2,2,3,2));k=rng.normal(size=(2,2,3,2));a=ref.Softmax()(q@k.swapaxes(-1,-2)/np.sqrt(2));v=rng.normal(size=(2,2,3,2))
primitive('att_scores','先实现 QKᵀ 缩放打分',['Q','K'],'def solve(Q,K):\n    return Q @ K.swapaxes(-1,-2) / np.sqrt(Q.shape[-1])',
'Q（B,H,Tq,d）、K（B,H,Tk,d），输出 S（B,H,Tq,Tk）。最后两轴是 query/key token，不是特征。',
'S[b,h,t,s]=Σᵢ Q[b,h,t,i]K[b,h,s,i]/√d。只转置 K 最后两轴；Tq 与 Tk 可不同。',[[q,k],[rng.normal(size=(2,3,2,4)),rng.normal(size=(2,3,5,4))],[np.zeros((2,2,3,2)),k]])
primitive('att_probs','再实现因果掩码与逐行概率',['S'],'def solve(S):\n    mask=np.triu(np.ones(S.shape[-2:],dtype=bool),1)\n    z=np.where(mask,-np.inf,S)\n    z=z-z.max(axis=-1,keepdims=True)\n    e=np.exp(z)\n    return e/e.sum(axis=-1,keepdims=True)',
'S（B,H,T,T）是未掩码分数；输出同形 A。未来位置概率为零，每个 query 行概率和为 1。',
'先屏蔽 s>t，再 Softmax；不能先 Softmax 再直接清零，否则每行概率和改变。此练习只接受方形因果自注意力分数。',[[q@k.swapaxes(-1,-2)/np.sqrt(2)],[rng.normal(size=(3,2,4,4))],[np.zeros((2,2,3,3))]])
primitive('att_weighted','将注意力概率变成输出向量',['A','V'],'def solve(A,V):\n    return A @ V',
'A（B,H,Tq,Tk）、V（B,H,Tk,dv），输出 Y（B,H,Tq,dv）。同一 query 的概率对各个 key 的内容向量加权。',
'Y[b,h,t,j]=Σₛ A[b,h,t,s]V[b,h,s,j]。value 特征数 dv 不必与打分特征数相同。',[[a,v],[ref.Softmax()(rng.normal(size=(2,3,2,5))),rng.normal(size=(2,3,5,4))],[a,np.zeros_like(v)]])
primitive('att_merge','将多头输出恢复为 token 向量',['X'],'def solve(X):\n    B,H,T,d=X.shape\n    return X.transpose(0,2,1,3).reshape(B,T,H*d)',
'X（B,H,T,d）是所有 head 的输出；返回（B,T,H×d），每个 token 的各 head 特征按 head 顺序拼接。',
'先把轴顺序还原为 B,T,H,d，再合并 H,d。直接 reshape(B,T,H*d) 会打乱 token/head 的对应。',[[rng.normal(size=(2,2,3,4))],[rng.normal(size=(3,3,2,2))],[np.arange(48).reshape(2,2,3,4)]])
for stage in book['stages']:
    if stage['title'].startswith('第六'):
        stage['modules'][0]['tasks']=['att_scores','att_probs','att_weighted','t_attention','att_merge','t_mha']
book['lessons']['att_scores']['output_axes']=['batch b','head h','query t','key s']
book['lessons']['att_probs']['output_axes']=['batch b','head h','query t','key s']
book['lessons']['att_probs']['input_axes']={'S':['batch b','head h','query t','key s']}
book['lessons']['att_weighted']['input_axes']={'A':['batch b','head h','query t','key s']}
# 残差展示与当前公开输入/上游一致，所有 batch/token 完整保留。
meta=book['lessons']['t_block'];p=meta['cases'][0]['input'];obj=ref.Block(*p['init']);x=np.array(p['X']);g=np.array(p['G']);obj(x)
branch2=obj.norm2.backward(obj.mlp.backward(g));g1=g+branch2
branch1=obj.norm1.backward(obj.attn.backward(g1))
meta['residual_trace']=[dict(text='第二条残差：G 直接回到 x₁，同时经过 MLP 和 norm2 后回到 x₁。两路合并得到 G₁。',direct=g.tolist(),branch=branch2.tolist(),sum=g1.tolist()),dict(text='第一条残差：G₁ 直接回到 x，同时经过 Attention 和 norm1 后回到 x。两路合并得到 dX。',direct=g1.tolist(),branch=branch1.tolist(),sum=(g1+branch1).tolist())]
for focus,label in [('weights','加载权重并清除旧梯度'),('optimizer','单独检查 AdamW 的 m、v、t、学习率恢复'),('rng','单独检查 RNG 下一段随机序列恢复')]:
    special('t_checkpoint','checkpoint',label,**{'class':'GPT'},init=[5,4,2,8,1,5,6],focus=focus,seed=37)
for focus,label in [('copy','Parameter：复制输入、float64、同形零梯度'),('empty','空模块与下划线缓存不产生可训练参数')]:special('infrastructure','infrastructure',label,focus=focus)
for name,shape,Tk,dv,zero in [('非方形交叉注意力且 value 特征数不同',(2,2,2,3),4,5,False),('零 Q/K：每行允许位置概率均匀',(2,2,3,2),3,2,True),('三个 head 与更长序列',(2,3,4,3),4,3,False)]:
    q=np.zeros(shape) if zero else rng.normal(size=shape);k=np.zeros((shape[0],shape[1],Tk,shape[-1])) if zero else rng.normal(size=(shape[0],shape[1],Tk,shape[-1]));v=rng.normal(size=(shape[0],shape[1],Tk,dv));g=rng.normal(size=(*shape[:-1],dv));causal=shape[-2]==Tk
    c=dict(mode='attention',label=name,input=dict(**{'class':'AttentionCore'},init=[causal],Q=q.tolist(),K=k.tolist(),V=v.tolist(),G=g.tolist()));c['expected']={k:np.asarray(v).tolist() for k,v in probe(vars(ref),c).items()};add('t_attention',c,'形状与边界')
# 为失败用例定位保存明确的参考初始化和残差分支数值。
for c in book['lessons']['layer_linear']['cases']:
    if c.get('assessment'):continue
    obj=base.Linear(*c['init'])
    for name,value in c.get('parameters',{}).items():getattr(obj,name).data[...] = value
    c['visual_parameters']={'W':obj.W.data.tolist()}
    if obj.b is not None:c['visual_parameters']['b']=obj.b.data.tolist()
for c in book['lessons']['t_block']['cases']:
    if c.get('assessment'):continue
    p=c['input'];obj=ref.Block(*p['init']);x=np.asarray(p['X']);g=np.asarray(p['G']);obj(x)
    branch2=obj.norm2.backward(obj.mlp.backward(g));g1=g+branch2;branch1=obj.norm1.backward(obj.attn.backward(g1))
    c['residual_trace']=[dict(text='本测评第二条残差：G 加上 MLP/Norm2 分支回流。',direct=g.tolist(),branch=branch2.tolist(),sum=g1.tolist()),dict(text='本测评第一条残差：G₁ 加上 Attention/Norm1 分支回流。',direct=g1.tolist(),branch=branch1.tolist(),sum=(g1+branch1).tolist())]
# 各项公开输入/输出与分类写回；学习阶段成果也进入网页。
for id,m in book['lessons'].items():
    if m['kind']=='concept':m.pop('cases',None);continue
    for c in m['cases']:c.setdefault('category','前向与反向' if m['kind']=='layer' else '数值与形状')
    m['assessment_revision']=2
    m['outcome']='完成本任务后，将此接口接入后续模块；任何当前代码或前置代码变化都需要重新验收。'
goals=['能区分 token 编号、向量、logits、损失，以及训练与生成的数据流。','能操作真实多批次、多 token、多 head 张量，正确归约、广播、换轴与散射。','暂缓。','能独立实现基础层前向、缓存、输入梯度、参数梯度与清零。','能将 logits 转为稳定概率与平均交叉熵，并把梯度传回 logits。','能从打分、掩码、加权输出逐步组装完整注意力和 Transformer Block，并手写反向。','能组装多层 GPT，正确处理位置向量、共享词表与全部梯度。','能构造训练批次，完成裁剪、AdamW 更新与独立验证。','能生成文本、保存状态、重新载入，并验证续训一致性。']
for stage,goal in zip(book['stages'],goals):stage['outcome']=goal
book['version']=4
(ROOT/'webapp/foundations.json').write_text(json.dumps(book,ensure_ascii=False,indent=2),encoding='utf-8')
arrays=json.loads((ROOT/'exercises/cases.json').read_text(encoding='utf-8'))
for id,m in book['lessons'].items():
    if m['kind']=='array':arrays[id]=m['cases']
(ROOT/'exercises/cases.json').write_text(json.dumps(arrays,ensure_ascii=False,indent=2),encoding='utf-8')
print('Expanded checks:',sum(len(m.get('cases',m.get('questions',[]))) for m in book['lessons'].values()))

import runpy
runpy.run_path(str(ROOT/'scripts/build_independence.py'),run_name='__main__')
