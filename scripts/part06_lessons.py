"""Part06：注意力构件、持久模块、多头与残差综合；对照与旧题独立保留。"""
import ast
import copy
import re
from datetime import datetime
import shutil
from html.parser import HTMLParser
from part45_lessons import scaffold

IDS=['att_scores','att_probs','att_weighted','att_merge','att_value_backward','att_score_backward','t_attention','t_mha','t_block','t_dropout','t_layernorm','rebuild_block']
ALIASES={'att_scores':'attention_scores','att_weighted':'weighted_values','att_merge':'merge_heads','att_value_backward':'value_gradient'}
BLOCK='''class Block(Module):
    def __init__(self, C, H, hidden):
        self.norm1 = RMSNorm(C)
        self._ffn = MLPBranch(C, hidden)
        # 公开别名让原有参数路径和后续 GPT 初始化继续找到同一组对象。
        self.norm2 = self._ffn.norm
        self.attn = MultiHeadAttention(C, H)
        self.mlp = self._ffn.mlp

    def forward(self, x):
        x1 = x + self.attn(self.norm1(x))
        return x1 + self._ffn(x1)

    def backward(self, g):
        g1 = g + self._ffn.backward(g)
        return g1 + self.norm1.backward(self.attn.backward(g1))
'''


def empty_methods(code):
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in ('__init__','forward','backward'):
            node.body=ast.parse('raise NotImplementedError("请完成 '+node.name+'")').body
    return ast.unparse(tree)+'\n'


def sync_part06_imports(book, exercises):
    """上游Block合一后的必要接线；只改已知导入，不改后续Part的方法。"""
    path=exercises/'rebuild_gpt.py'
    if not path.exists():return []
    old=path.read_text(encoding='utf-8');needle='from rebuild_block import Block'
    if needle not in old:return []
    new=old.replace(needle,'from t_block import Block')
    methods=lambda s:[ast.dump(v) for v in ast.walk(ast.parse(s)) if isinstance(v,ast.FunctionDef)]
    if methods(old)!=methods(new):raise ValueError('导入迁移不得改变任何方法')
    if path.read_text(encoding='utf-8')!=old:return []
    backups=exercises/'backups';backups.mkdir(exist_ok=True)
    shutil.copy2(path,backups/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-block-import-'+path.name))
    path.write_text(new,encoding='utf-8');book['lessons']['rebuild_gpt']['local_code']=new
    return ['rebuild_gpt']


def apply_part06(book,math,aligned):
    lessons=book['lessons'];m=lambda t:math(t,inline=True)
    p=lambda t:'<p>'+t+'</p>'
    h=lambda t,s:'<h3>'+t+'</h3>'+s
    ul=lambda a:'<ul>'+''.join('<li>'+v+'</li>' for v in a)+'</ul>'
    intro=p('注意力让每个位置读取其他位置的信息。'+m('Q')+' 是提问向量，'+m('K')+' 是供匹配的索引向量，'+m('V')+' 是被读取的内容。多头模块会用个人 <code>Linear</code> 从输入产生这三组向量；前面的构件题由调用者提供它们。')
    axes=p(m(r'Q\in\mathbb R^{B\times H\times T_q\times d},\quad K\in\mathbb R^{B\times H\times T_k\times d},\quad V\in\mathbb R^{B\times H\times T_k\times d_v}')+'。'+m('B')+' 是批次数，'+m('H')+' 是头数，'+m('T_q,T_k')+' 是提问/被读取位置数，'+m('d,d_v')+' 是匹配/内容宽度。每个批次、每个头独立计算。')
    score=math(r'S_{b,h,i,j}=\frac1{\sqrt d}\sum_cQ_{b,h,i,c}K_{b,h,j,c}')
    mask=math(r'\widetilde S_{b,h,i,j}=\begin{cases}S_{b,h,i,j}&j\le i\\-\infty&j>i\end{cases},\qquad A=\operatorname{softmax}_{j}(\widetilde S)')
    weighted=math(r'Y_{b,h,i,c}=\sum_j A_{b,h,i,j}V_{b,h,j,c}')
    dv=aligned(r'\frac{\partial Y_{i,c}}{\partial V_{j,c}}&=A_{i,j}',r'\mathrm dV_{j,c}&=\sum_i A_{i,j}G_{i,c}',r'\mathrm dV&=A^{\mathsf T}G')
    ds=aligned(r'\frac{\partial A_{i,k}}{\partial S_{i,j}}&=A_{i,k}(\delta_{kj}-A_{i,j})',r'\mathrm dS_{i,j}&=\sum_k\mathrm dA_{i,k}A_{i,k}(\delta_{kj}-A_{i,j})',r'&=A_{i,j}\left(\mathrm dA_{i,j}-\sum_kA_{i,k}\mathrm dA_{i,k}\right)')
    merges=aligned(r'X&:(B,H,T,d)',r'X^{\rm reordered}&:(B,T,H,d)',r'Y_{b,t,hd+c}&=X_{b,h,t,c},\qquad Y:(B,T,Hd)')

    def install(key,body,interface,inputs,outputs,state,actions,formula,purpose,example,extension=''):
        meta=lessons[key]
        meta['body']=body;meta['derivation']='';meta['extension']=extension
        meta['definition']='';meta['steps']=[];meta['hint']='先按题面末尾清单核对输入、缓存与返回值。'
        meta['task_brief']=dict(scope=interface,variables=[dict(name='输入',meaning=purpose,shape='见输入格式')],operation=purpose,output=outputs,conditions=[],next_use=None)
        meta['learning_page']=dict(role='miniGPT 组成' if key not in ['t_dropout','t_layernorm'] else '结构对照应用',problem_title=meta['title'],purpose=purpose,statement=h('题目描述',p('实现 '+interface+'。')+p(purpose))+h('输入格式',p(inputs))+h('输出格式',p(outputs))+h('状态变化',p(state))+h('动手任务清单',ul(actions))+h('关键公式速查',formula),sample_input='',sample_note=p(example))

    install('att_scores',intro+axes+h('1. 每个提问要与全部索引匹配',p('固定一个批次和头，'+m('Q')+' 的每一行分别与 '+m('K')+' 的每一行做点积。结果的行是 query，列是 key；最后的特征轴被求和掉。')+score)+h('2. 一次算出整张匹配表',p('<code>K.swapaxes(-1, -2)</code> 只交换最后两轴，得到 '+m('(B,H,d,T_k)')+'。用批量矩阵乘法得到 '+m('(B,H,T_q,T_k)')+'，再除以 '+m(r'\sqrt d')+'。从输入的最后一轴读取宽度。')), '<code>solve(Q, K)</code>','两组实数数组，形状如上；批次、头数和匹配宽度一致，长度均为正。',m(r'S:(B,H,T_q,T_k)')+'，每个格子是对应 query/key 的缩放匹配分数。','保留输入原值；没有持久缓存。',['交换索引的最后两轴。','批量点积并按匹配宽度缩放。'],score,'产生注意力的整张匹配分数表。','本页第一项公开样例含两个批次、两个头、三个 query。图示用同一组向量继续产生掩码、概率和读取结果。',h('为什么缩放',p('若分量独立且方差约为一，点积方差随宽度增长；除以 '+m(r'\sqrt d')+' 控制分数尺度。')))
    install('att_probs',h('1. 把分数变成可读取的权重',p('输入 '+m(r'S:(B,H,T_q,T_k)')+' 是上一种匹配表。每行对应一个 query，最后轴列出它可读取的 key。因果规则允许 '+m(r'j\le i')+'；未来 key 必须完全排除。')+mask)+h('2. 先排除，再逐行归一化',p('<code>np.triu(np.ones(S.shape[-2:], dtype=bool), 1)</code> 产生最后两轴的严格上三角布尔掩码。<code>True</code> 表示未来，用 <code>np.where</code> 把这些分数改为 '+m(r'-\infty')+'。同一二维规则广播到全部批次和头。')+p('然后沿 <code>axis=-1</code> 减本行最大值、指数化、除以本行指数和，归约保留轴。因为 '+m(r'e^{-\infty}=0')+'，未来权重严格为零，允许的 key 权重之和为一。分数改为零仍会产生正指数，不能实现屏蔽。')),'<code>solve(S)</code>',m(r'S:(B,H,T_q,T_k)')+'；位置采用相同起点，至少有一个 key，每个 query 至少允许一个 key。',m('A')+' 与分数同形，沿 key 轴归一化。','输入保持原值。',['构造严格上三角未来掩码。','屏蔽后执行稳定的逐行 Softmax。'],mask,'实现因果屏蔽与稳定概率化，供理解完整核心的前向。','样例中第一行只允许第零列，第二行允许前两列；图中灰色格子在概率化之前就被排除。')
    install('att_weighted',h('1. 一行权重读取整张内容表',axes+p(m('A')+' 的形状为 '+m('(B,H,T_q,T_k)')+'，每一行决定一个 query 怎样组合全部内容行。权重乘对应的整行内容，再对 key 求和，得到这个 query 的输出向量。')+weighted)+h('2. 全部 query 一起读取',p('批量矩阵乘法 <code>A @ V</code> 正好沿 key 轴汇总。同一内容行可以被很多 query 读取；内容宽度 '+m('d_v')+' 可以与匹配宽度 '+m('d')+' 不同。')),'<code>solve(A, V)</code>',m(r'A:(B,H,T_q,T_k),\quad V:(B,H,T_k,d_v)')+'；两个数组的 key 长度一致。',m(r'Y:(B,H,T_q,d_v)')+'。','输入保持原值。',['按权重读取对应内容行并沿 key 轴求和。'],weighted,'将注意力权重变成真正的内容输出。','同一组图中的概率行对应完整内容表；点一行输出可看到该 query 对各 key 的读取关系。')
    install('att_merge',h('1. 把各头结果送回特征轴',p('每个头已经产生自己的内容向量。合头保留每个批次和 token，把同一 token 的各头向量按头编号拼接。输入 '+m('X')+' 的轴顺序是批次、头、token、头内特征。')+merges)+h('2. 为什么要先交换轴',p('先用 <code>transpose(0, 2, 1, 3)</code> 让同一 token 的各头相邻，再将末两轴合并。直接 <code>reshape</code> 会把不同 token 的内容交错进输出。反向把这些步骤倒过来：先拆末轴，再交换头与 token。')),'<code>solve(X)</code>',m('X:(B,H,T,d)')+'，各维长度为正。',m('Y:(B,T,Hd)')+'，特征顺序先头编号、后头内特征。','输入保持原值。',['读取四个轴长度。','交换头/token，再合并头和特征。'],merges,'合并多头输出，随后交给输出投影。','用不同颜色表示不同头。合头后的同一行保留所有头的完整片段，不改变 token 的先后。')
    install('att_value_backward',h('1. 从前向读取关系找回流',p('前向 '+m('Y=AV')+' 中，同一个内容行 '+m('V_j')+' 被所有 query 共享。现在输出收到上游 '+m(r'G=\partial L/\partial Y')+'。每个 query 用过这行内容，就应把自己的上游乘读取权重送回来。')+weighted)+h('2. 同一内容的所有用途相加',p('暂略批次和头下标。固定内容位置 '+m('j')+' 与特征 '+m('c')+'，它影响每个 '+m('Y_{i,c}')+'，局部系数就是 '+m('A_{i,j}')+'。因此沿 query 求和：')+dv+p('转置后，原来属于这个 key 的整列权重变成一行，正好收集全部 query 的上游。批次和头保留，最后两个轴互换。')),'<code>solve(A, G)</code>',m(r'A:(B,H,T_q,T_k),\quad G:(B,H,T_q,d_v)')+'；来自同一次前向。',m(r'\mathrm dV:(B,H,T_k,d_v)')+'。','输入保持原值；返回本次梯度。',['沿 query 收集上游，返回内容梯度。'],dv,'实现注意力回到内容表的梯度，用于完整核心。','图中前向从 key 读取到多行 query，反向沿相同权重回到 key；主讲解展示整张内容梯度的归约关系。')
    install('att_score_backward',h('1. 上游已经到达概率表',p(m(r'\mathrm dA=\partial L/\partial A')+' 与概率 '+m('A')+' 同形。第五部分已经实现的 <code>Softmax.backward</code> 正是在做这件事；本题把它应用到每行 query 的 key 分布。')+p('一项分数影响本行所有概率，因为它们共享分母；所以不能只乘本格概率。'))+h('2. 收齐同一行所有路径',p('固定 query '+m('i')+'，'+m(r'\delta_{kj}')+' 在两个 key 编号相同时为一，否则为零。由已学 Softmax 局部导数：')+ds+p('先沿最后轴计算 '+m(r'R=\sum_k A_{i,k}\mathrm dA_{i,k}')+'，保留长度为一的 key 轴，再广播回本行。不同 query 不共享分母。被屏蔽的位置概率为零，因而其分数梯度也为零。')),'<code>solve(A, dA)</code>',m(r'A,\mathrm dA:(B,H,T_q,T_k)')+'；每行概率归一化，可含被掩码产生的零。',m(r'\mathrm dS')+' 与概率同形。','输入及上游保持原值。',['求本行概率加权上游和。','广播该和，逐元素计算分数梯度。'],math(r'\mathrm dS=A\odot(\mathrm dA-\operatorname{sum}_{k}(A\odot\mathrm dA))'),'理解并应用完整核心使用的 Softmax 反向。','图示突出行内耦合：单个分数影响同一行所有概率；反向各行独立，未来格无回流。')

    core=lessons['t_attention']
    for key,alias in ALIASES.items():
        meta=lessons[key]
        if ('\n'+alias+' = solve') not in meta['reference']:meta['reference']=meta['reference'].rstrip()+'\n\n'+alias+' = solve\n'
        meta['exports']=[alias]
        raw=meta['starter'].split('\n\n# 具名导出')[0].split('\n\n'+alias+' = solve')[0].rstrip()+'\n\n# 具名导出供后续模块复用；仍用 solve 接收本题测评。\n'+alias+' = solve\n'
        meta['starter']=raw
    core['reference']=core['reference'].replace('q @ k.swapaxes(-1,-2) / self._scale','attention_scores(q, k)').replace('return self._a @ v','return weighted_values(self._a, v)').replace('dv = self._a.swapaxes(-1,-2) @ g','dv = value_gradient(self._a, g)')
    core['dependencies']=['infrastructure','t_softmax','att_scores','att_weighted','att_value_backward']
    cq=aligned(r'\mathrm dA_{i,j}&=\sum_cG_{i,c}V_{j,c},\qquad\mathrm dA=GV^{\mathsf T}',r'\frac{\partial S_{i,j}}{\partial Q_{i,c}}&=K_{j,c}/\sqrt d,\qquad\frac{\partial S_{i,j}}{\partial K_{j,c}}=Q_{i,c}/\sqrt d',r'\mathrm dQ_{i,c}&=\sum_j\mathrm dS_{i,j}K_{j,c}/\sqrt d',r'\mathrm dK_{j,c}&=\sum_i\mathrm dS_{i,j}Q_{i,c}/\sqrt d')
    summary=aligned(r'S&=QK^{\mathsf T}/\sqrt d,\quad A=\operatorname{softmax}(\widetilde S),\quad Y=AV',r'\mathrm dV&=A^{\mathsf T}G,\quad\mathrm dA=GV^{\mathsf T}',r'\mathrm dS&=A\odot(\mathrm dA-\operatorname{sum}_{k}(A\odot\mathrm dA))',r'\mathrm dQ&=\mathrm dS K/\sqrt d,\quad\mathrm dK=\mathrm dS^{\mathsf T}Q/\sqrt d')
    install('t_attention',intro+axes+h('1. 创建一次，复用已有计算',p('<code>AttentionCore(causal=True)</code> 保存模式并创建一个个人 <code>Softmax</code>。复用前三题具名导出的 <code>attention_scores</code>、<code>weighted_values</code>、<code>value_gradient</code>；骨架已给出导入。概率构件题的规则由持久 <code>Softmax</code> 和掩码组合，反向直接调用它的 <code>backward</code>。'))+h('2. 完整前向',score+mask+weighted+p('先复制保存 <code>q</code>、<code>k</code>、<code>v</code> 到私有缓存，保存匹配宽度的平方根，再计算分数。因果模式要求两个序列长度相等，按上三角屏蔽；非因果模式保留全部 key，允许交叉注意力的两种序列长度不同。')+p('概率存入 <code>self._a</code>，它的每行对应一个 query。所有向量按同一批次和头对应。'))+h('3. 输出先把梯度分给内容和权重',p('上游 '+m('G')+' 与输出同形。前向乘法的两个因子都影响输出：内容乘权重回流，权重乘内容回流。')+dv+aligned(r'\mathrm dA&=GV^{\mathsf T}')+p('将 '+m(r'\mathrm dA')+' 交给持久 <code>softmax.backward</code> 得到 '+m(r'\mathrm dS')+'；因果模式再明确把未来位置置零。'))+h('4. 分数回到提问与索引',p('每个 query 与所有 key 匹配；同一个 key 又被所有 query 使用。两侧分别收齐自己的所有用途：')+cq+p('返回三元组顺序固定为 '+m(r'(\mathrm dQ,\mathrm dK,\mathrm dV)')+'。只有最后两轴交换；缩放是前向的局部系数，所以两条匹配路径都保留它。')),'<code>AttentionCore.__init__(causal=True)</code>、<code>forward(q, k, v)</code>、<code>backward(g)</code>',axes+'上游 '+m('G')+' 与输出同形；因果模式为等长自注意力。','前向返回 '+m('(B,H,T_q,d_v)')+'；反向返回分别与三个输入同形的三元组。','保存输入副本、缩放、概率和因果掩码；本层无可训练参数。每个实例拥有独立 Softmax。',['创建模式和持久 Softmax。','复用打分/读取，缓存并应用因果掩码。','逆序算内容、概率、分数、提问和索引梯度。'],summary,'把已有注意力构件组装成完整可反向的核心。','主图保持两个批次、两个头，展示全部 query/key；可切换整体前向、内容回流、匹配回流，单格仅作对应核对。',h('缩放和稳定性',p('缩放控制点积尺度，减最大值保持 Softmax 概率不变。它们的详细原因可回看打分题与第五部分拓展。')))

    mha=lessons['t_mha'];mha['dependencies']=['infrastructure','initialization','layer_linear',*core['dependencies'][1:],'t_attention','att_merge']
    mha['reference']=mha['reference'].replace('y.transpose(0,2,1,3).reshape(B,T,C)','merge_heads(y)')
    mf=aligned(r'Z&=\operatorname{qkv}(X):(B,T,3C)',r'(Q_0,K_0,V_0)&=\operatorname{split}(Z):\text{各 }(B,T,C)',r'(Q,K,V)&:\text{各 }(B,H,T,d),\quad d=C/H',r'O&=\operatorname{core}(Q,K,V):(B,H,T,d)',r'M&=\operatorname{merge}(O):(B,T,C),\quad Y=\operatorname{proj}(M)')
    mb=aligned(r'\mathrm dM&=\operatorname{proj.backward}(G)',r'\mathrm dO&=\operatorname{splitHeads}(\mathrm dM)',r'(\mathrm dQ,\mathrm dK,\mathrm dV)&=\operatorname{core.backward}(\mathrm dO)',r'\mathrm dZ&=\operatorname{concat}(\operatorname{merge}(\mathrm dQ),\operatorname{merge}(\mathrm dK),\operatorname{merge}(\mathrm dV))',r'\mathrm dX&=\operatorname{qkv.backward}(\mathrm dZ)')
    install('t_mha',h('1. 给核心装上可训练投影',p('<code>MultiHeadAttention(C, H)</code> 接收总特征宽度与头数，要求 '+m(r'C\bmod H=0')+'。创建三个持久子层：<code>qkv = Linear(C, 3*C)</code> 一次产生三组向量，<code>core = AttentionCore(True)</code> 做因果读取，<code>proj = Linear(C, C)</code> 将合头结果重新混合。全部调用个人实现。'))+h('2. 前向：投影、拆头、读取、合头、投影',mf+p('<code>np.split(..., 3, axis=-1)</code> 按顺序取三组。每组先从 '+m('(B,T,C)')+' 变为 '+m('(B,T,H,d)')+'，再用 <code>transpose(0, 2, 1, 3)</code> 拆头。合头直接复用 <code>merge_heads</code>。保存原形状供反向使用。'))+h('3. 反向把相同布局倒回去',mb+p('拼接必须仍为提问、索引、内容顺序，与前向分割相对应。输入同时影响这三组投影输出，<code>qkv.backward</code> 会把三条用途汇到同一输入，并跨全部批次/token 累计共享投影参数梯度。两个投影层各有自己的输入缓存，不在前向反复创建。')),'<code>MultiHeadAttention.__init__(C, H)</code>、<code>forward(x)</code>、<code>backward(g)</code>',m('X,G:(B,T,C)')+'；均为实数数组，头数与总宽度为正且整除。','前向与反向分别返回 '+m(r'Y,\mathrm dX:(B,T,C)')+'。','四个投影参数由个人 Linear 管理，反向累加，继承 zero_grad 清零；缓存属于最近一次前向。',['创建两层 Linear 与一个 AttentionCore。','正确分割、拆头、合头和输出投影。','逆序传递、拼接梯度，复用持久子层。'],mf+mb,'为 GPT 提供完整可训练多头因果自注意力。','图中展示同一 token 的三组投影与每头片段，反向按原布局回传；所有头都保留，不用单头假装完整模块。')

    block=lessons['t_block'];block['reference']=BLOCK
    block['dependencies']=list(dict.fromkeys(['infrastructure','initialization','layer_rmsnorm','layer_linear','layer_gelu','layer_mlp','foundation_capstone',*mha['dependencies'][3:],'t_mha']))
    # capstone 的公开检查依赖不是 Block 的运行依赖；只补它实际使用的类。
    bf=aligned(r'U&=\operatorname{norm1}(X),\quad F_1=\operatorname{attn}(U)',r'X_1&=X+F_1',r'F_2&=\operatorname{MLPBranch}(X_1)=\operatorname{mlp}(\operatorname{norm2}(X_1))',r'Y&=X_1+F_2')
    bb=aligned(r'G_1&=\underbrace{G}_{\text{第二次直连}}+\underbrace{\operatorname{MLPBranch.backward}(G)}_{\text{第二条分支}}',r'\mathrm dX&=\underbrace{G_1}_{\text{第一次直连}}+\underbrace{\operatorname{norm1.backward}(\operatorname{attn.backward}(G_1))}_{\text{第一条分支}}')
    install('t_block',h('1. 复用前面写好的两种分支',p('<code>Block(C, H, hidden)</code> 是本章综合任务：注意力分支负责不同 token 间读取信息，第四部分的 <code>MLPBranch</code> 负责每个 token 内的特征变换。两条分支各先做自己的归一化。'))+h('2. 前向的两次加法',bf+p('每次把分支结果加回它的输入，输入输出始终为 '+m('(B,T,C)')+'。先完成第一条，再把相加后的 '+m('X_1')+' 同时交给第二条分支与直连。'))+h('3. 为什么反向需要两次相加',p('如果 '+m('Y=X_1+F_2(X_1)')+'，提高 '+m('X_1')+' 会沿两条路径影响输出：加法直连的局部导数是一，分支的局部导数由已有 <code>backward</code> 计算。所以两条路径各接收完整上游，回到同一对象时相加。第一条残差遵守相同规则。')+bb+p('不要只传分支梯度，也不要把上游先除以二。分支内部已经逆序调用 MLP/RMSNorm；Block 只负责把直连与分支返回值合流。'))+h('4. 构造与旧参数路径',p('创建独立 <code>norm1</code>、<code>attn</code> 和 <code>_ffn = MLPBranch(C, hidden)</code>。用 <code>norm2 = _ffn.norm</code>、<code>mlp = _ffn.mlp</code> 作为同一对象的公开别名，保留后续 GPT 已使用的参数路径。私有 <code>_ffn</code> 是调用入口；参数通过这两个公开别名被收集一次。两个 Norm 必须是不同实例。')),'<code>Block.__init__(C, H, hidden)</code>、<code>forward(x)</code>、<code>backward(g)</code>',m('X,G:(B,T,C)')+'；构造宽度/头数合法，隐藏宽度为正。','返回 '+m(r'Y,\mathrm dX:(B,T,C)')+'。','子层保存各自缓存、累加各自参数梯度；Block 不更新权重。参数去重、累计与清零沿用 Module。',['创建并复用已有的归一化层 <code>RMSNorm</code>、多头注意力层 <code>MultiHeadAttention</code> 和特征变换分支 <code>MLPBranch</code>，注册公开别名。','顺序完成两条分支和残差加法。','从第二次加法开始逆序传递并两次合流。'],bf+bb,'组合全部注意力层与已有基础分支，形成 GPT 将堆叠的唯一 Block。','本页检查完整双批次前后向、所有参数梯度、个人子层真实身份与调用，以及两条残差的零/非零分支。')

    drop=lessons['t_dropout'];ln=lessons['t_layernorm']
    df=aligned(r'M_i&=\frac{\mathbf1[u_i\ge p]}{1-p},\quad u_i\sim U[0,1)',r'Y&=X\odot M,\qquad\mathrm dX=G\odot M',r'\text{评估： }M&=1,\quad Y=X,\quad\mathrm dX=G')
    install('t_dropout',h('1. 训练和评估是两种调用模式',p('本项目主线 GPT 使用前面完成的确定性模块；本题作为训练正则化的对照组件。<code>Dropout(p, seed)</code> 在训练时独立屏蔽一部分元素，评估时完整保留输入。'+m('p')+' 是丢弃概率。'))+h('2. 前向保存本次掩码',df+p('持久 <code>default_rng(seed)</code> 每次训练调用推进随机状态。用 <code>random(x.shape)</code> 取同形随机数，与概率比较得到保留位置；保留值除以 '+m('1-p')+'，因此训练输出的期望与输入相同。把缩放后的掩码保存在 <code>self._mask</code>。'))+h('3. 反向复用相同选择',p('前向被屏蔽的元素对输出没有影响，反向为零；保留位置的局部系数仍为 '+m('1/(1-p)')+'。直接乘最近一次前向掩码。评估前向保存全一掩码，且不消耗随机数。')),'<code>Dropout.__init__(p=.2, seed=0)</code>、<code>forward(x, training=True)</code>、<code>backward(g)</code>','任意同形实数输入与上游，'+m(r'0\le p<1')+'；训练/评估由布尔参数决定。','同形输出和输入梯度，无可训练参数。','每次训练推进 RNG；评估不推进；反向复用当前掩码并保持输入与上游。',['创建 RNG 并检查概率。','训练随机生成/评估生成全一掩码。','前后向使用同一缩放掩码。'],df,'理解可选 Dropout 的训练/评估状态约定。','展示同一掩码在前后向的保留/阻断；此图是确定的关系示例，不冒充浏览器复现 NumPy 随机数。')
    lf=aligned(r'\mu&=\operatorname{mean}_{c}(X),\quad Z=X-\mu',r'R&=(\operatorname{mean}_{c}(Z^2)+\epsilon)^{-1/2},\quad H=ZR',r'Y&=H\odot w+b')
    lb=aligned(r'U&=G\odot w',r'\mathrm dX&=R\left(U-\operatorname{mean}_{c}(U)-H\operatorname{mean}_{c}(UH)\right)',r'\mathrm dw_c&=\sum_{b,t}G_{b,t,c}H_{b,t,c},\qquad\mathrm db_c=\sum_{b,t}G_{b,t,c}')
    deriv=aligned(r'\mathrm dZ&=R\left(U-H\operatorname{mean}_{c}(UH)\right)',r'\mathrm dX&=\mathrm dZ-\operatorname{mean}_{c}(\mathrm dZ)',r'\operatorname{mean}_{c}(H)&=0')
    install('t_layernorm',h('1. 在归一化前增加去均值',p('本题是 RMSNorm 的结构对照。对同一 token 的整行特征先算均值并减去它，再按中心化后的均方缩放，最后用共享缩放 '+m('w')+' 和偏置 '+m('b')+' 调整。先前的均值、去均值、RMSNorm 在这里组合。')+lf+p('所有均值只沿最后特征轴，保留该轴为一；批次和 token 完整保留。参数 '+m('w,b:(C,)')+' 跨全部位置共享，分别初始化为全一和全零。'))+h('2. 梯度先回到归一化特征',p(m(r'G=\partial L/\partial Y')+' 是外部给定上游。仿射变换先产生 '+m(r'U=G\odot w')+'，参数梯度则沿所有前置位置求和。'))+h('3. 先逆均方缩放，再逆去均值',p('把中心化特征 '+m('Z')+' 当作已学 RMSNorm 输入：先回直接乘法与均方统计两条路径，再通过去均值返回原输入。')+deriv+p('由于中心化特征均值为零，代入并整理得到：')+lb+p('缓存 <code>self._r</code> 与 <code>self._h</code> 足以计算反向。参数梯度用加法累计，输入梯度直接返回。')),'<code>LayerNorm.__init__(C, eps=1e-5)</code>、<code>forward(x)</code>、<code>backward(g)</code>',m(r'X,G:(\ldots,C)')+'，至少二维；'+m(r'\epsilon>0')+'。','同形输出与输入梯度；两个 '+m('(C,)')+' 参数。','保存倒标准差与归一化特征；累计参数梯度；继承清零，不修改输入/权重。',['创建全一缩放和零偏置。','逐行去均值、均方缩放并缓存。','逆序两条统计关系，累计共享参数梯度。'],lf+lb,'实现经典 LayerNorm 对照，辨别它与主线 RMSNorm 的差异。','主图把每行均值、中心化与缩放放在同一条线上；参数梯度按同一特征跨全部位置聚合。')

    # 保留原用例，只增加本次新增的构造/复用/状态验收。
    for key in ['t_attention','t_mha','t_block','t_dropout','t_layernorm']:
        meta=lessons[key]
        meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='part06_contract']
        for variant,label in [('build','构造、当前个人子层与实例隔离'),('state','完整张量缓存、重复反向与状态')]:
            meta['cases'].append(dict(label=label,assessment='part06_contract',input=dict(lesson=key,variant=variant,operation='按本题接口检查构造、个人子层身份/调用及完整多批次状态；具体检查见题面与展开结果。'),expected=dict(要求='当前个人类、持久实例与约定状态全部满足')))
        meta['assessment_revision']=3;meta['assessment_migration']=dict(previous_revision=2,reason='保留原用例，新增学习者构造、当前个人模块复用与状态验收。')
        meta['starter']=empty_methods(meta['reference'])

    # 只改必要的依赖接线；后续Part正文和测评保持原值。
    def expand(deps):
        out=[]
        def add(d):
            if d=='rebuild_block':d='t_block'
            if d in out:return
            if d in ['t_attention','t_mha','t_block']:
                for child in lessons[d]['dependencies']:add(child)
            out.append(d)
        for d in deps:add(d)
        return out
    for meta in lessons.values():
        for field in ['dependencies','experiment_dependencies']:
            if field in meta:meta[field]=expand(meta[field])

    for key in IDS:
        if key=='rebuild_block':continue
        meta=lessons[key];imports='import numpy as np\n'
        if meta['kind']!='array':imports+='from learner_core import Parameter, Module, initialize\n'
        extra=[]
        if key=='t_attention':extra=['from t_softmax import Softmax','from att_scores import attention_scores','from att_weighted import weighted_values','from numpy_att_value_backward import value_gradient']
        if key=='t_mha':extra=['from layer_linear import Linear','from t_attention import AttentionCore','from att_merge import merge_heads']
        if key=='t_block':extra=['from layer_rmsnorm import RMSNorm','from t_mha import MultiHeadAttention','from foundation_capstone import MLPBranch']
        if extra:imports+='if __name__ != "student":\n'+''.join('    '+v+'\n' for v in extra)
        scaffold(meta,meta['starter'],imports,'_checks' if meta['kind']=='array' else '_transformer_checks')
        if meta['kind']=='array':
            for name in ['code','starter']:meta['scaffold_notes'][name]=meta['scaffold_notes'][name].replace('globals())','solve)')

    # 自然语言中的代码标识同样需要排版；跳过现有代码与数学节点。
    class CodeProse(HTMLParser):
        def __init__(self):super().__init__(convert_charrefs=False);self.out=[];self.skip=[]
        def handle_starttag(self,tag,attrs):
            self.out.append(self.get_starttag_text())
            if tag in ['code','pre'] or any(k=='class' and 'math-' in (v or '') for k,v in attrs):self.skip.append(tag)
        def handle_endtag(self,tag):
            self.out.append('</'+tag+'>')
            if self.skip and self.skip[-1]==tag:self.skip.pop()
        def handle_data(self,data):
            self.out.append(data if self.skip else re.sub(r'(?<![\w.])(Linear|Softmax|RMSNorm|MLPBranch|MLP|GELU|Module|Parameter|AttentionCore|MultiHeadAttention|MHA|Block|Dropout|LayerNorm|zero_grad)(?!\w)',r'<code>\1</code>',data))
        def handle_entityref(self,name):self.out.append('&'+name+';')
        def handle_charref(self,name):self.out.append('&#'+name+';')
    for key in IDS[:-1]:
        meta=lessons[key]
        for field in ['body','extension']:
            parser=CodeProse();parser.feed(meta[field]);meta[field]=''.join(parser.out)
        for field in ['statement','sample_note']:
            parser=CodeProse();parser.feed(meta['learning_page'][field]);meta['learning_page'][field]=''.join(parser.out)

    sixth=next(s for s in book['stages'] if s.get('id')=='part-06' or s.get('title','').startswith('第六'))
    oldmodule=copy.deepcopy(sixth['modules'][-1])
    sixth['modules']=[v for v in sixth['modules'] if 'rebuild_block' not in v['tasks'] and 't_dropout' not in v['tasks']]
    for mod in sixth['modules']:
        if 't_block' in mod['tasks']:mod['name']='综合组装唯一 Block'
    # 原Block所在模块可能包含对照层，被过滤后需要恢复主线收尾。
    if not any('t_block' in mod['tasks'] for mod in sixth['modules']):
        oldmodule.update(name='综合组装唯一 Block',tasks=['t_block']);sixth['modules'].append(oldmodule)
    sixth['outcome']='组合个人基础分支与多头因果注意力，完整实现并验收唯一 Block。'
    history=next(s for s in book['stages'] if s.get('historical'))
    history['modules']=[v for v in history['modules'] if 'rebuild_block' not in v['tasks']]
    oldmodule=copy.deepcopy(oldmodule);oldmodule.update(name='旧独立 Block（可选历史）',tasks=['rebuild_block']);history['modules'].append(oldmodule)
    old=lessons['rebuild_block'];old['superseded_by']='t_block'
    if not any(s.get('id')=='optional-architecture' for s in book['stages']):
        book['stages'].append(dict(id='optional-architecture',title='选修 · 结构与训练对照',optional=True,outcome='理解 Dropout 与 LayerNorm 的用途，不计主线完成条件。',modules=[dict(name='Dropout 与 LayerNorm',tasks=['t_dropout','t_layernorm'])]))
    from attention_structure import apply_attention_structure
    return apply_attention_structure(book,math,aligned,empty_methods)
