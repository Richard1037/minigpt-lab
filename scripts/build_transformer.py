"""追加第五至九部分，不覆盖已有个人练习；可独立重建后续章节。"""
import ast
import inspect
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'solutions')]
import transformer_layers as ref
from _transformer_checks import probe
from _foundation_checks import pack
book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
# 保留前四部分，只重建后续课程元数据。
book['stages']=[s for s in book['stages'] if not s['title'].startswith(('第五','第六','第七','第八','第九'))]
new={}
scope=vars(ref)
def sec(title,text): return f'<h3>{title}</h3><p>{text}</p>'
def hollow(source,names):
    class Holes(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name in names:
                node.body=ast.parse('raise NotImplementedError("TODO：按正文公式实现 '+node.name+'")').body
            return node
    tree=Holes().visit(ast.parse(source));ast.fix_missing_locations(tree)
    return ast.unparse(tree)
def case(mode,label,**p):
    if mode in ['loss','attention']: p.setdefault('class', 'CrossEntropy' if mode=='loss' else 'AttentionCore')
    c=dict(mode=mode,label=label,input=p)
    out=probe(scope,c)
    c['expected']={k:v if isinstance(v,str) else pack(v) for k,v in out.items()}
    return c
def add(id,title,exports,methods,deps,body,deriv,cases,steps,trace=None):
    source='\n\n'.join(inspect.getsource(getattr(ref,k)) for k in exports)
    ordered=[]
    def visit(dep):
        for child in new.get(dep,book['lessons'].get(dep,{})).get('dependencies',[]): visit(child)
        if dep not in ordered: ordered.append(dep)
    for d in deps: visit(d)
    imports='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n'
    if ordered:
        imports+='if __name__ != "student":\n'
        for dep in ordered:
            m=new.get(dep,book['lessons'].get(dep))
            names=m.get('exports',[m.get('class','')])
            imports+=f"    from {Path(m['file']).stem} import {', '.join(names)}\n"
    starter=hollow(source,methods)
    code=imports+'\n'+starter+f'\n\nif __name__ == "__main__":\n    from _transformer_checks import run_local\n    run_local("{id}", globals())\n'
    path=ROOT/'exercises'/f'{id}.py'
    if not path.exists():path.write_text(code,encoding='utf-8')
    new[id]=dict(id=id,title=title,goal=steps[0],kind='project',exports=exports,dependencies=ordered,
        body=body,derivation=deriv,cases=cases,steps=steps,starter=imports+'\n'+starter,local_code=path.read_text(encoding='utf-8'),file=path.name,reference=source,
        extension=sec('数值检查（拓展）','对任意输出构造 L=Σ(Y⊙G)，中心差分逐项检查解析梯度。复杂层先用小尺寸 float64 核对，再测试多个 batch/head。参数共享时同一参数的所有使用路径都必须累加。固定公开样例不能替代对新数据的正确实现。'),trace=trace or [])
X=np.linspace(-1,1,24).reshape(2,3,4);G=np.linspace(.2,1.1,24).reshape(2,3,4)
module=lambda cls,init=[],label='B=2、T=3、C=4：前向、输入梯度与全部参数梯度',**kw:case('module',label,**{'class':cls},init=init,X=X.tolist(),G=G.tolist(),**kw)
add('t_softmax','稳定 Softmax 与向量雅可比积',['Softmax'],{'forward','backward'},[],
sec('输入与输出','X 是 logits 数组（B,T,V），每个 token 有 V 个类别分数。forward 返回同形概率 P；最后一轴求和为 1。backward(G) 接受概率上的上游梯度，返回同形 dX。无参数，不修改 X。')+sec('动手','沿最后一轴减最大值，再指数化并归一化。缓存 P。这里不是把整个三维张量归一化成一个分布。'),
sec('平移与稳定性','令 m=maxⱼxⱼ。exp(xᵢ−m)/Σⱼexp(xⱼ−m)=exp(xᵢ)/Σⱼexp(xⱼ)。减去 m 避免大正数指数溢出。')+sec('完整反向','∂pᵢ/∂xⱼ=pᵢ(δᵢⱼ−pⱼ)。dXⱼ=ΣᵢGᵢpᵢ(δᵢⱼ−pⱼ)=pⱼ(Gⱼ−ΣᵢGᵢpᵢ)。用 keepdims 保留求和轴，不构造 V×V 雅可比。'),
[module('Softmax'),case('module','大 logits：1000 量级仍有限',**{'class':'Softmax'},init=[],X=(X+1000).tolist(),G=G.tolist())],
['实现最后一轴的稳定 Softmax 和 backward。','缓存概率，计算逐 token 的 sum(G*P)。'],
[dict(label='原始 logits X',value=X.tolist(),text='六个 token 各有四个类别分数，逐 token 独立计算。'),dict(label='减去各行最大值',value=(X-X.max(-1,keepdims=True)).tolist(),text='最大值变成 0；其他分数非正，相对差不变。'),dict(label='概率 P',value=ref.Softmax()(X).tolist(),text='每个 token 的四个概率之和为 1。')])
target=np.array([[0,2,1],[3,1,0]])
add('t_loss','LogSumExp 交叉熵与平均梯度',['CrossEntropy'],{'forward','backward'},[],
sec('接口','forward(logits,targets)：logits（B,T,V），targets（B,T）为整数正确类别。返回零维 NumPy 数组，值为 B×T 个位置的平均负对数概率。backward(g=1) 返回（B,T,V），不对整数 targets 求导。')+sec('索引','targets[b,t] 指定该位置正确类别，不是 one-hot。先稳定计算 logp，再 take_along_axis 取正确类；不要先 softmax 再 log，极小概率会下溢。'),
sec('损失','N=B×T，logpᵢ=zᵢ−logΣⱼexp(zⱼ)，z=x−max(x)。L=−(1/N)Σₙlogpₙ,yₙ。')+sec('反向推导','单个位置 ℓ=logΣⱼexp(xⱼ)−x_y，所以 ∂ℓ/∂xⱼ=pⱼ−1[j=y]。所有位置平均后 dlogits=(P−one_hot(targets))/N；再乘标量上游 g。N 只除一次，Linear 不再次平均。'),
[case('loss','六个位置平均；类别各不相同',X=X.tolist(),targets=target.tolist()),case('loss','正确类别概率极小仍输出有限 loss',X=(X*1000).tolist(),targets=target.tolist())],
['实现返回平均标量损失及 logits 梯度的 CrossEntropy。','前向使用 logsumexp；反向每个正确类减 1，再除 B×T。'])
q=np.linspace(-.7,.8,24).reshape(2,2,3,2);k=np.cos(np.arange(24)).reshape(2,2,3,2);v=np.sin(np.arange(24)).reshape(2,2,3,2);ag=np.linspace(-1,1,24).reshape(2,2,3,2)
ac=case('attention','B=2、H=2、T=3、d=2：未来概率为零',Q=q.tolist(),K=k.tolist(),V=v.tolist(),G=ag.tolist(),init=[True])
core=ref.AttentionCore();out=core(q,k,v)
scores=q@k.swapaxes(-1,-2)/np.sqrt(2)
add('t_attention','缩放点积、因果掩码与完整反向',['AttentionCore'],{'forward','backward'},['t_softmax'],
sec('Q、K、V 是什么','Q 表示当前位置要寻找的特征，K 表示每个位置可被匹配的特征，V 表示匹配后传递的内容。本节它们已由调用方给定；下一节由同一个 X 经三套投影得到。')+sec('输入输出','Q/K/V 为（B,H,T,d），每个 batch、head 独立计算。forward(Q,K,V) 返回 Y（B,H,T,d）。backward(G) 返回三元组 dQ,dK,dV。缓存 Q/K/V、概率 A 和 mask；无参数。causal=False 时允许 Q 的 Tq 与 K/V 的 Tk 不同。')+sec('掩码','S=QKᵀ/√d，S 的最后两轴是查询 token、被读取 token。j>i 的位置在 softmax 前设 −∞。每行至少保留自己，避免全 −∞。不要用零替代被屏蔽的分数。'),
sec('前向','A=softmax(mask(S))，Y=AV。点积由 d 项相加；在分量方差约为 1 且独立的假设下，方差随 d 增长，除 √d 控制分数尺度。')+sec('从输出逆序推导','dV=AᵀG；dA=GVᵀ。Softmax 的 dS=A⊙(dA−sum(dA⊙A,axis=−1,keepdims=True))。将掩码位置 dS 清零。')+sec('回到 Q/K','dQ=dS K/√d；dK=dSᵀQ/√d。转置只交换最后两轴。所有公式均保留 B/H，不取第一个 batch 代替完整计算。')+sec('来源','<a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need，§3.2</a>。'),
[ac,case('attention','交叉注意力：Tq=2、Tk=3',Q=q[:,:,:2].tolist(),K=k.tolist(),V=v.tolist(),G=ag[:,:,:2].tolist(),init=[False])],
['实现因果/非因果 AttentionCore 及 Q、K、V 三路反向。','分别计算打分、掩码、Softmax、加权求和；缓存反向中间量。','检查所有 batch/head，以及 Tq≠Tk 的非因果情况。'],
[dict(label='原始分数 S',value=scores.tolist(),text='每张 3×3 表固定一个 batch/head：行是 query，列是 key。'),dict(label='允许读取的位置',value=np.broadcast_to(np.tril(np.ones((3,3))),scores.shape).tolist(),text='1 表示允许；0 表示未来。掩码作用于 Softmax 之前。'),dict(label='注意力概率 A',value=core._a.tolist(),text='未来位置概率为 0，每个 query 行求和为 1。'),dict(label='加权后的 Y=AV',value=out.tolist(),text='每个输出 token 用同一行概率对各 token 的 V 做加权和。')])
add('t_mha','多头注意力：QKV 投影到输出投影',['MultiHeadAttention'],{'forward','backward'},['layer_linear','t_attention'],
sec('接口与参数','MultiHeadAttention(C,H)，C 必须被 H 整除。X/G/Y/dX 均为（B,T,C）。qkv 是 Linear(C,3C)，proj 是 Linear(C,C)，core 是 AttentionCore(True)。所有可训练参数来自两个 Linear。')+sec('形状链','X → qkv（B,T,3C）→ 沿最后一轴 split 为 Q/K/V（B,T,C）→ reshape（B,T,H,d）→ transpose（B,H,T,d）→ core → 逆转置与 reshape（B,T,C）→ proj。'),
sec('反向逐步还原','先 proj.backward；再把 dY 拆为（B,H,T,d）；core.backward 得三路梯度；各路先转为（B,T,H,d）再合并为（B,T,C）；最后 concatenate(dQ,dK,dV,axis=−1)，交给 qkv.backward。')+sec('为什么不能相加','qkv 输出的三个切片对应不同输出列，反向应拼回原列。只有三条投影都直接以同一 X 为输入且分开计算时，才把三个 dX 相加。'),
[module('MultiHeadAttention',[4,2]),module('MultiHeadAttention',[4,1],label='单头边界：接口与梯度保持一致')],
['组合个人 Linear 与 AttentionCore，完成多头注意力。','反向严格逆序执行每次 reshape/transpose/split。'])
add('t_block','Pre-Norm Block 与残差梯度合流',['Block'],{'forward','backward'},['layer_rmsnorm','layer_mlp','t_mha'],
sec('模块','Block(C,H,hidden) 含 norm1、attn、norm2、mlp。输入输出（B,T,C）。x₁=x+attn(norm1(x))；y=x₁+mlp(norm2(x₁))。缓存由各子层保存，不能用同一归一化实例连续两次覆盖缓存。'),
sec('第二条残差','设上游为 G。x₁ 一路直接到 y，一路经过 norm2/MLP，所以 G₁=G+norm2.backward(mlp.backward(G))。')+sec('第一条残差','x 同样有直接与注意力路径：dX=G₁+norm1.backward(attn.backward(G₁))。残差相加的每条分支都接收完整上游梯度。参数梯度由各子层累加。'),
[module('Block',[4,2,8]),case('module','常量输入仍能走通残差反向',**{'class':'Block'},init=[4,2,6],X=np.ones((2,2,4)).tolist(),G=np.full((2,2,4),.2).tolist())],
['组合 RMSNorm、Attention、MLP，实现完整 Block。','反向在两个残差分叉处分别相加，不能覆盖直连梯度。'])
add('t_dropout','Dropout：训练、评估与缓存掩码',['Dropout'],{'forward','backward'},[],
sec('接口','Dropout(p,seed)。forward(X,training=True) 使用生成器采样与 X 同形掩码；评估时直接返回同值数组。backward(G) 乘本次前向缓存的同一掩码。不含参数。默认 GPT 主线不启用 Dropout，先保证确定性训练；本节可独立验证训练/评估差异。'),
sec('倒置 Dropout','mᵢ=Bernoulli(1−p)/(1−p)，Y=X⊙m，故 E[Y|X]=X。dX=G⊙m。评估 m=1，不需要再次缩放。随机状态只在训练前向推进；反向不能重采样。'),
[module('Dropout',[.25,7],kwargs={'training':True}),module('Dropout',[.25,7],label='评估模式为恒等映射',kwargs={'training':False}),module('Dropout',[0.,7],label='p=0 边界')],
['实现可复现的训练掩码、评估分支和 backward。','随机生成器只在构造时创建；验证 p=0 与评估模式。'])
add('t_layernorm','LayerNorm 与原始 Transformer 对照',['LayerNorm'],{'forward','backward'},[],
sec('接口','LayerNorm(C,eps) 包含 w、b 两个（C，）参数。逐 token 对最后一轴减均值再除标准差。X/Y/G/dX 均为（B,T,C）。RMSNorm 不减均值；两者不可直接替换公式。')+sec('原始结构对照','原始 encoder/decoder 使用 LayerNorm 与残差，并以正弦位置编码为例；本项目主线采用 Pre-Norm、RMSNorm、可学习位置向量。交叉注意力 Q 来自 decoder，K/V 来自 encoder，可直接使用上一节 causal=False 的 core。'),
sec('前向','μ=mean(x)，r=(mean((x−μ)²)+ε)⁻¹ᐟ²，h=(x−μ)r，y=h⊙w+b。')+sec('反向','令 u=G⊙w。经过中心化与方差两条链式路径合并：dX=r[u−mean(u)−h mean(uh)]。dw=Σ位置(Gh)，db=Σ位置G。所有 mean 只沿特征轴，参数梯度沿 B/T 归约。')+sec('位置公式对照','PE(pos,2i)=sin(pos/10000^(2i/C))，PE(pos,2i+1)=cos(pos/10000^(2i/C))。它没有可训练参数；本项目的位置表按位置编号查表，并把所有 batch 的位置梯度相加。'),
[module('LayerNorm',[4]),case('module','相同特征向量：均值消除、epsilon 保持有限',**{'class':'LayerNorm'},init=[4],X=np.ones((2,3,4)).tolist(),G=G.tolist())],
['手写 LayerNorm 的完整前向与反向，对照 RMSNorm。','检查两个参数梯度与零方差输入。'])
config=[5,4,2,8,2,6,7]
gids=np.array([[0,1,2],[2,1,3]])
add('t_gpt','完整 GPT：位置嵌入、堆叠与权重共享',['GPT'],{'forward','backward'},['layer_embedding','t_block'],
sec('构造参数','GPT(V,C,H,hidden,layers,context,seed)。token 表（V,C），position 表（context,C），layers 个独立 Block，末尾 RMSNorm。输出头直接使用 token.weight 的转置；不另建一份权重。构造器给定，完成 forward/backward。')+sec('输入输出','输入 ids（B,T）整数，T≤context；输出 logits（B,T,V）。输入不是向量 X，由 token(ids)+position(arange(T))[None,:,:] 得到（B,T,C）。backward(G) 返回 None，同时累加全部参数梯度。')+sec('缓存与初始化','每个 Block 使用独立实例，forward 缓存服务紧接的一次 backward。二维权重由同一随机生成器依次初始化；两个残差出口权重除 √(2×layers)，只在构造时执行一次。'),
sec('共享头反向','末尾激活 H（B,T,C），E（V,C），logits=HEᵀ。展平位置后 dE_head=G2ᵀH2；dH=GE。经过 norm、反序 Blocks 后的 dX 再散射到 E.grad。同一 E.grad 最终等于输出头与输入查表两路贡献之和。')+sec('位置反向','位置向量广播到 B 条序列，故先对 batch 求和：dPosition=Σ_b dX；再按位置编号交给 position.backward。整数 ids 不求导。共享参数只在 parameters() 中出现一次，避免优化器更新两次。'),
[case('module','双 Block、双头、两个 batch：共享权重完整梯度',**{'class':'GPT'},init=config,ids=gids.tolist(),G=np.linspace(-.2,.3,30).reshape(2,3,5).tolist()),case('module','不同上下文长度与重复 token',**{'class':'GPT'},init=[5,4,2,8,1,6,9],ids=[[1,1],[2,1]],G=np.ones((2,2,5)).tolist())],
['将个人模块组合成能手写反向的 GPT。','先累加共享输出头梯度，再把梯度反传到输入 token 表。','位置梯度对 batch 求和，Blocks 按逆序反向。'])
add('t_data','字符词表、移位标签与批次',['tokenize','make_batch'],{'tokenize','make_batch'},[],
sec('接口','tokenize(text) 返回 (vocab,ids)：vocab 是按字符排序的 Python 列表，ids 是（L，）int64。make_batch(ids,starts,T) 接受起点数组（B，），返回 X、targets，均为（B,T）整数数组。每个起点满足 0≤s 且 s+T<L。')+sec('对应关系','第 b 行 X=ids[s:s+T]，targets=ids[s+1:s+T+1]。同一位置预测下一个字符。样例语料“猫喜欢鱼猫喜欢鱼”得到 ids=[2,0,1,3,2,0,1,3]；T=3、起点 0 时，X=[2,0,1]，targets=[0,1,3]；起点 3 时，X=[3,2,0]，targets=[2,0,1]。训练/验证应先划分，再分别采样窗口。'),
sec('目标','L=−mean log p(targets[b,t]|X[b,0:t+1])。训练时所有位置并行提供监督；因果掩码负责避免未来泄漏，目标移位本身不能替代掩码。'),
[case('data','两个起点：相邻字符目标精确移位',text='猫喜欢鱼猫喜欢鱼',starts=[0,3],T=3),case('data','重复字符词表去重',text='abacabadabac',starts=[1,5],T=3)],
['实现字符词表与批次构造，返回正确 dtype 和形状。','显示每行输入与目标的逐位置对应。'])
add('t_optim','全局梯度裁剪与 AdamW',['clip_grad','AdamW'],{'clip_grad','step'},[],
sec('接口与状态','clip_grad(parameters,limit) 原地缩放所有参数的 grad，返回裁剪前零维 NumPy 范数。AdamW(parameters,lr,beta1,beta2,eps,decay) 保存参数列表、同形 m/v 和步数 t。step 读取 grad 并原地更新 data，不清零 grad。')+sec('衰减分组','本课程约定二维及更高维权重参与衰减，一维 bias/norm scale 不衰减。共享参数只能传入一次。裁剪先于 AdamW。'),
sec('裁剪','n=√Σ参数Σ元素g²，s=min(1,limit/(n+10⁻¹²))，每个 g←sg。不能每个张量分别裁剪，否则方向改变。')+sec('AdamW','t←t+1；m←β₁m+(1−β₁)g；v←β₂v+(1−β₂)g²。m̂=m/(1−β₁ᵗ)，v̂=v/(1−β₂ᵗ)。参与衰减的 θ←(1−ηλ)θ；再 θ←θ−ηm̂/(√v̂+ε)。衰减与损失梯度分离。')+sec('来源','<a href="https://arxiv.org/abs/1711.05101">Decoupled Weight Decay Regularization</a>。'),
[case('optimizer','两步更新：全局裁剪、偏差校正、衰减分组',weights=[[[1.,2.],[3.,4.]],[1.,2.]],gradients=[[[[1.,-2.],[3.,-4.]],[2.,-1.]],[[[.5,.3],[-.2,.1]],[.2,.4]]],config={'lr':.01,'decay':.1},limit=1.),case('optimizer','零梯度：仅权重衰减',weights=[[[1.,2.],[3.,4.]],[1.,2.]],gradients=[[[[0.,0.],[0.,0.]],[0.,0.]]],config={'lr':.1,'decay':.2},limit=1.)],
['实现全局范数裁剪与带偏差校正的 AdamW。','核对两步状态、零梯度、权重衰减分组。'])
traincase=dict(init=config,ids=gids.tolist(),targets=[[1,2,3],[1,3,4]],steps=12,lr=.02)
add('t_train','训练闭环：从损失到参数更新',['train'],{'train'},['t_loss','t_gpt','t_optim','t_data'],
sec('接口','train(model,x,y,steps,lr) 接受已构造 GPT、输入和目标（B,T），在固定小批次上训练 steps 步。返回（steps，）float64 损失数组；模型参数在函数内更新。使用 CrossEntropy、AdamW(decay=.01) 和 clip_grad(limit=1)。')+sec('按顺序执行','每步 model.zero_grad → logits=model(x) → loss_fn(logits,y) → model.backward(loss_fn.backward()) → clip_grad → opt.step。记录更新前 loss。本任务先过拟合固定批次；本页下方实验室可检验独立验证集。')+sec('训练与评估','评估只 forward 和计算 loss，不能 backward/step。缓存会被评估 forward 覆盖，必须在训练 backward 结束后再评估。若使用 Dropout，评估关闭随机掩码。多 micro-batch 累加时先清零一次，各批 loss 按总 token 数加权，再更新一次。'),
sec('梯度下降与困惑度','θₜ₊₁=Optimizer(θₜ,∂L/∂θₜ)。较小固定批次 loss 应能够下降；下降不能单独证明无因果泄漏。困惑度 exp(L) 仅在同样 tokenizer、数据与平均方式下可比较。')+sec('学习率扩展','warmup 阶段 η=ηmax×step/warmup；随后余弦 η=ηmin+½(ηmax−ηmin)(1+cos(πu))，u 是衰减阶段完成比例。先用固定学习率建立基线，再加入调度。'),
[case('train','12 步双批次训练：逐步损失和最终 logits',**traincase),case('train','不同随机种子和单步更新',**{**traincase,'init':[5,4,2,8,1,6,3],'steps':1})],
['实现完整训练循环，返回每步更新前的损失。','检查清零时机、反向、裁剪与更新次序。'])
add('t_sample','温度、top-k 与批量采样',['sample'],{'sample'},['t_softmax'],
sec('接口','sample(logits,temperature,top_k,rng)：logits（B,V），temperature>0；top_k=0 保留全部，其他取 1…V。返回（B，）int64 编号。不得修改 logits。rng 是调用方创建的 Generator，本函数不能重置种子。')+sec('规则','先除温度，再保留每行最大的 k 个分数，其他置 −∞，Softmax 后按概率抽样。并列按稳定 argsort 保留末尾 k 项，固定此规则保证公开样例可复现。'),
sec('概率','pᵢ=exp(zᵢ/τ)/Σⱼexp(zⱼ/τ)。τ 小时分布尖锐，τ 大时平缓。top-k 是重新归一化后的随机采样；k=1 才退化为选定最大类别。'),
[case('sample','两个分布独立采样，固定生成器种子',X=[[1.,3.,2.,0.],[2.,-1.,1.,3.]],temperature=.8,top_k=2,seed=8),case('sample','top-k=1 与并列规则',X=[[1.,1.,1.,1.],[2.,-1.,1.,3.]],temperature=1.,top_k=1,seed=2)],
['实现按温度和 top-k 调整概率后采样。','每条 batch 返回一个编号，保留调用方随机状态。'])
add('t_generate','自回归生成与上下文窗口',['generate'],{'generate'},['t_gpt','t_sample'],
sec('接口','generate(model,ids,steps,temperature=1,top_k=0,seed=0) 返回（B,T+steps）整数数组。ids 是提示编号（B,T）。每步仅将最近 context 个 token 送入模型，取最后位置 logits 采样，再追加。输入不修改，不求梯度。')+sec('位置与缓存','当前完整重算窗口，位置编号在窗口内从 0 开始。KV cache 是可选优化：缓存每层历史 K/V 并处理新 token 的绝对位置及偏移掩码，不能直接复用训练缓存。先核对完整重算，再扩展 cache；本章不以 cache 作为必修捷径。'),
sec('条件概率','p(x₁…xₙ)=∏ₜp(xₜ|x＜ₜ)。训练用真实前缀一次并行预测所有位置；生成每次使用已生成的前缀。只有最后位置的分布用于追加一个新 token。'),
[case('generate','两个提示，生成后超过 context，自动滑窗',**{**traincase,'steps':5,'temperature':.8,'top_k':3,'seed':6}),case('generate','零步生成保持原提示',**{**traincase,'steps':0,'temperature':1.,'top_k':0,'seed':0})],
['实现完整重算的批量自回归生成。','追加新 token 时保留 batch 轴；整个生成过程共享一个 RNG。'])
add('t_checkpoint','检查点与精确续训',['save_checkpoint','load_checkpoint'],{'save_checkpoint','load_checkpoint'},['t_gpt','t_loss','t_optim'],
sec('接口','save_checkpoint(path,model,optimizer,rng) 写入本地 npz；load_checkpoint 同参数，原地恢复并返回 None。只读写显式路径。保存 p0…、m0…、v0… 和 JSON meta；meta 包含 t、rng 状态、学习率、beta1/2、eps、decay。模型结构由调用方按相同配置构造；实际实验同时保存配置及词表。')+sec('验收','先更新一次并保存，再更新一次得到参照；恢复后重复同一更新，logits 与 RNG 下一次采样必须一致。只保存权重不足以恢复 Adam 的下一步。加载使用 allow_pickle=False，检查参数形状，清零旧梯度。'),
sec('状态关系','权重决定当前预测；m/v/t 决定下一次更新；数据 RNG 决定下一批采样。恢复前须用相同模型配置和参数顺序构造模型与优化器。本练习的文件保存参数、优化器配置和 RNG；数据划分、词表、模型配置与数据读取进度需在训练程序中另行记录。精确续训时，这些信息需与检查点对应同一个训练步。'),
[case('checkpoint','恢复后下一步更新和 RNG 完全相同',**traincase)],
['实现包含优化器与 RNG 状态的检查点保存和恢复。','运行续训一致性测评；文件在临时目录中生成并自动清理。'])
add('t_project','miniGPT 项目综合验收',['project_step'],{'project_step'},['t_train','t_generate','t_checkpoint'],
sec('最终交付','提交完整 project_step(model,optimizer,ids,targets) 训练入口；实际模型来自你此前的所有个人模块。输入 ids/targets 均为（B,T）整数数组；返回更新前平均损失的零维 NumPy 数组，同时完成一次参数更新。')+sec('执行顺序','清空旧梯度 → GPT 前向 → CrossEntropy → 完整反向 → 全局裁剪到 1 → optimizer.step。下面还会独立检查模型因果性、全部参数张量梯度、共享权重、固定批次过拟合、评估状态、滑窗生成和精确续训。')+sec('本地交付','完成所有个人模块后，用 exercises/run_gpt.py 读取自己的 UTF-8 文本，训练并保存，再重新载入生成。项目验收只接受当前代码和当前依赖对应的结果。'),
sec('完整链路','L → logits → 共享词表输出头 → 最终 RMSNorm → 反序 Blocks → token 与 position 表。输出头和输入查表对同一词表参数的贡献相加。优化器收集共享参数一次。'),[],
['实现一次完整、可重复调用的 miniGPT 训练更新。','通过下面八类综合检查，再用自己的文本完成本地训练与恢复。'])
new['t_train']['lab']=True
new['t_train']['experiment_dependencies']=list(dict.fromkeys(new['t_train']['dependencies']+['t_train']+new['t_generate']['dependencies']+['t_generate']))
book['lessons'].update(new)
for title,mods in [
 ('第五部分 · 概率与损失',[('从分数到平均损失',['t_softmax','t_loss'])]),
 ('第六部分 · 注意力与 Transformer',[('注意力核心与多头',['t_attention','t_mha']),('残差与结构对照',['t_block','t_dropout','t_layernorm'])]),
 ('第七部分 · 组装 GPT',[('位置与共享输出头',['t_gpt']),('文本到训练批次',['t_data'])]),
 ('第八部分 · 优化与训练',[('优化器与完整训练',['t_optim','t_train'])]),
 ('第九部分 · 生成与保存',[('采样和自回归',['t_sample','t_generate']),('保存与续训',['t_checkpoint'])])]:
    book['stages'].append(dict(title=title,modules=[dict(name=n,tasks=ids) for n,ids in mods]))
book['stages'][-1]['modules'].append(dict(name='miniGPT 项目综合验收',tasks=['t_project']))
book['version']=3
(ROOT/'webapp/foundations.json').write_text(json.dumps(book,ensure_ascii=False,indent=2),encoding='utf-8')
print('Added',len(new),'lessons;',len(book['lessons']),'total; checks',sum(len(m.get('cases',m.get('questions',[]))) for m in book['lessons'].values()))

import runpy
runpy.run_path(str(ROOT/'scripts/strengthen_assessments.py'),run_name='__main__')
