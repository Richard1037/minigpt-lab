"""注意力按三个完整模块组织；页内分步教学、分项测评，可选小练习保留历史成果。"""
import copy
from part45_lessons import scaffold

PRACTICE=['att_scores','att_probs','att_weighted','att_merge','att_value_backward','att_score_backward']
CORE='''class AttentionCore(Module):
    def __init__(self, causal=True):
        self.causal = causal
        self.softmax = Softmax()

    def forward(self, q, k, v):
        self._q, self._k, self._v = q.copy(), k.copy(), v.copy()
        self._scale = np.sqrt(q.shape[-1])
        scores = q @ k.swapaxes(-1, -2) / self._scale
        if self.causal:
            if q.shape[-2] != k.shape[-2]:
                raise ValueError('因果自注意力要求 Q/K 序列长度相同')
            self._mask = np.triu(np.ones(scores.shape[-2:], dtype=bool), 1)
            scores = np.where(self._mask, -np.inf, scores)
        self._a = self.softmax(scores)
        return self._a @ v

    def backward(self, g):
        dv = self._a.swapaxes(-1, -2) @ g
        da = g @ self._v.swapaxes(-1, -2)
        ds = self.softmax.backward(da)
        if self.causal:
            ds = np.where(self._mask, 0., ds)
        dq = ds @ self._k / self._scale
        dk = ds.swapaxes(-1, -2) @ self._q / self._scale
        return dq, dk, dv
'''


def apply_attention_structure(book, math, aligned, empty_methods):
    lessons=book['lessons'];m=lambda t:math(t,inline=True)
    p=lambda t:'<p>'+t+'</p>';h=lambda t,s:'<h3>'+t+'</h3>'+s
    core=lessons['t_attention'];mha=lessons['t_mha']
    core.update(title='注意力核心：完整前向与反向',reference=CORE,dependencies=['infrastructure','t_softmax'])
    core['body']=h('1. 这个模块把什么变成什么',p('每个 query 位置提出一次信息请求：先用提问向量 '+m('Q')+' 与所有索引向量 '+m('K')+' 匹配，得到读取各位置的权重，再按这些权重组合内容向量 '+m('V')+'。输出 '+m('Y')+' 是每个 query 读回的信息。三组输入将在下一节由个人 <code>Linear</code> 从 token 特征投影产生。')+p(m(r'Q:(B,H,T_q,d),\quad K:(B,H,T_k,d),\quad V:(B,H,T_k,d_v)')+'。'+m('B')+' 是批次数，'+m('H')+' 是头数，'+m('T_q,T_k')+' 分别是 query/key 位置数，'+m('d')+' 是匹配宽度，'+m('d_v')+' 是内容宽度。输出为 '+m('(B,H,T_q,d_v)')+'；每个批次和头独立计算。'))
    core['body']+=h('2. 先看完整前向，再看每一步',p('以下四步都写在同一个 <code>forward(q, k, v)</code> 中。每步产物就是下一步输入。')+aligned(r'(Q,K)&\xrightarrow{\text{点积并缩放}}S\xrightarrow{\text{排除未来}}\widetilde S\xrightarrow{\text{按key归一化}}A',r'(A,V)&\xrightarrow{\text{按权重读取内容}}Y'))
    core['body']+=h('① 匹配：生成整张 query/key 分数表',math(r'S_{b,h,i,j}=\frac1{\sqrt d}\sum_c Q_{b,h,i,c}K_{b,h,j,c}')+p('提问的每一行与索引的每一行点积。'+m('i')+' 是 query，'+m('j')+' 是 key，特征轴 '+m('c')+' 被求和，得到 '+m('S:(B,H,T_q,T_k)')+'。<code>k.swapaxes(-1, -2)</code> 只交换最后两轴，因此批量乘法一次产生全部匹配，再除以从输入最后轴读出的 '+m(r'\sqrt d')+'。'))
    core['body']+=h('② 排除未来：哪些信息可以读',math(r'\widetilde S_{b,h,i,j}=\begin{cases}S_{b,h,i,j}&j\le i\\-\infty&j>i\end{cases}')+p('<code>causal=True</code> 表示因果自注意力，要求两个序列长度相等。用 <code>np.triu(np.ones(scores.shape[-2:], dtype=bool), 1)</code> 标出未来区域；<code>True</code> 表示屏蔽，<code>np.where</code> 将其改为 '+m(r'-\infty')+'。二维规则广播到全部批次和头。非因果模式保留所有 key，允许 '+m(r'T_q\ne T_k')+' 的交叉注意力。'))
    core['body']+=h('③ 概率化：每个 query 在允许的 key 间分配权重',math(r'A_{b,h,i,j}=\frac{\exp(\widetilde S_{b,h,i,j})}{\sum_s\exp(\widetilde S_{b,h,i,s})}')+p('直接调用第五部分已写好的个人 <code>Softmax</code>，沿最后 key 轴稳定归一化。每行和为一，'+m(r'e^{-\infty}=0')+' 使未来权重严格为零。屏蔽分数不能改成零，因为它仍会产生正指数。'))
    core['body']+=h('④ 读取：权重组合整行内容',math(r'Y_{b,h,i,c}=\sum_j A_{b,h,i,j}V_{b,h,j,c},\qquad Y=AV')+p('一行权重依次乘所有 key 的内容行，再沿 key 相加，得到一个完整输出向量。所有 query 一起用批量矩阵乘法完成；同一内容行会被不同 query 重复读取。内容宽度 '+m('d_v')+' 可以与匹配宽度不同。'))
    core['body']+=h('3. 对象与缓存怎样准备',p('<code>__init__</code> 保存模式并创建一个持久 <code>Softmax</code>。前向保存三组输入的副本、缩放系数及概率 <code>self._a</code>；因果模式还保存掩码。输入副本让调用者之后修改输入时，不破坏本次反向。个人 <code>Softmax</code> 已负责自己的概率缓存。本层没有可训练参数。'))
    core['body']+=h('4. 反向沿同一条前向链返回',p('输出收到上游 '+m(r'G=\partial L/\partial Y')+'，与输出同形。先从读取结果回到内容与权重，再通过概率回到分数，最后回到两组匹配向量。')+aligned(r'G&\longrightarrow(\mathrm dV,\mathrm dA)',r'\mathrm dA&\xrightarrow{\rm Softmax.backward}\mathrm dS\longrightarrow(\mathrm dQ,\mathrm dK)'))
    core['body']+=h('① 读取的两侧都影响输出',p('暂略批次和头下标。内容 '+m('V_{j,c}')+' 被每个 query 使用，其局部系数是读取权重；权重 '+m('A_{i,j}')+' 又影响输出向量的全部特征，其局部系数是内容。所以分别收齐这两组用途：')+aligned(r'\mathrm dV_{j,c}&=\sum_iG_{i,c}\frac{\partial Y_{i,c}}{\partial V_{j,c}}=\sum_i A_{i,j}G_{i,c}',r'\mathrm dA_{i,j}&=\sum_cG_{i,c}\frac{\partial Y_{i,c}}{\partial A_{i,j}}=\sum_cG_{i,c}V_{j,c}',r'\mathrm dV&=A^{\mathsf T}G,\qquad\mathrm dA=GV^{\mathsf T}')+p('内容梯度沿 query 汇总；权重梯度沿内容特征汇总。转置来自需要收集的用途，仍只交换最后两轴。'))
    core['body']+=h('② 同一行概率的所有路径一起回到分数',p('第五部分已推导 Softmax 的局部导数。'+m(r'\delta_{kj}')+' 在两个 key 编号相同时为一，否则为零。每个分数影响同一 query 行的所有概率：')+aligned(r'\frac{\partial A_{i,k}}{\partial S_{i,j}}&=A_{i,k}(\delta_{kj}-A_{i,j})',r'\mathrm dS_{i,j}&=\sum_k\mathrm dA_{i,k}A_{i,k}(\delta_{kj}-A_{i,j})',r'&=A_{i,j}\left(\mathrm dA_{i,j}-\sum_kA_{i,k}\mathrm dA_{i,k}\right)')+p('调用持久 <code>softmax.backward(da)</code> 完成行内归约，因果模式再把掩码位置明确置零。不同 query 不共享概率分母。'))
    core['body']+=h('③ 匹配的两侧分别收齐所有用途',p('前向一个分数由一行提问和一行索引相乘得到。固定某个输入特征，其局部系数就是另一侧特征和缩放：')+aligned(r'\frac{\partial S_{i,j}}{\partial Q_{i,c}}&=K_{j,c}/\sqrt d,\qquad\frac{\partial S_{i,j}}{\partial K_{j,c}}=Q_{i,c}/\sqrt d',r'\mathrm dQ_{i,c}&=\sum_j\mathrm dS_{i,j}K_{j,c}/\sqrt d,\qquad\mathrm dQ=\mathrm dS K/\sqrt d',r'\mathrm dK_{j,c}&=\sum_i\mathrm dS_{i,j}Q_{i,c}/\sqrt d,\qquad\mathrm dK=\mathrm dS^{\mathsf T}Q/\sqrt d')+p('提问收齐与它匹配的全部 key，索引收齐用过它的全部 query。返回顺序固定为 '+m(r'(\mathrm dQ,\mathrm dK,\mathrm dV)')+'，分别与三个原输入同形。'))
    core['learning_page']['problem_title']='实现一个完整的 AttentionCore'
    core['learning_page']['purpose']='实现从三组向量到读取结果、再回到三组输入梯度的完整注意力核心。'
    statement=core['learning_page']['statement']
    statement=statement.replace('把已有注意力构件组装成完整可反向的核心。','完成一份可供多头模块调用的完整注意力核心。')
    statement=statement.replace('复用打分/读取，缓存并应用因果掩码。','在同一前向方法中完成打分、掩码、个人 <code>Softmax</code> 和内容读取，并保存独立缓存。')
    core['learning_page']['statement']=statement
    core['hint']='按完整前向、缓存、逆序反向检查同一份类；概率化复用个人 Softmax。'

    # 合头属于多头模块的布局步骤，不再要求导入独立的一行练习函数。
    mha['title']='多头注意力：投影、拆合头与完整反向'
    mha['reference']=mha['reference'].replace('merge_heads(y)','y.transpose(0,2,1,3).reshape(B,T,C)')
    mha['dependencies']=['infrastructure','initialization','layer_linear','t_softmax','t_attention']
    mha['body']=mha['body'].replace('合头直接复用 <code>merge_heads</code>。','合头先交换头与 token 两轴，再把头和头内特征合为最后轴。')
    extra=h('同一个 token 的各头怎样合回去',aligned(r'O&:(B,H,T,d)\xrightarrow{\operatorname{transpose}(0,2,1,3)}(B,T,H,d)',r'M_{b,t,hd+c}&=O_{b,h,t,c},\qquad M:(B,T,Hd)')+p('同一 token 的各头片段相邻后再合并，顺序先头编号、后头内特征；直接变形会把不同 token 交错。反向先把最后轴拆成头与头内特征，再交换头与 token。三组投影的梯度分别还原后，按原顺序拼接给 <code>qkv.backward</code>。'))
    marker='<h3>3. 反向把相同布局倒回去</h3>'
    mha['body']=mha['body'].replace(marker,extra+marker)
    mha['learning_page']['problem_title']='实现一个完整的 MultiHeadAttention'

    # 原20个小题样例进入真实类的分项检查，原始小题和成绩仍保留为选修。
    for meta in [core,mha]:meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='attention_component']
    labels={'att_scores':'前向·缩放打分','att_probs':'前向·因果概率','att_weighted':'前向·内容读取','att_value_backward':'反向·内容梯度','att_score_backward':'反向·行内分数梯度','att_merge':'布局·完整合头'}
    for source in PRACTICE:
        destination='t_mha' if source=='att_merge' else 't_attention'
        for i,c in enumerate(lessons[source]['cases']):
            lessons[destination]['cases'].append(dict(label=labels[source]+' · '+c['label'],assessment='attention_component',input=dict(lesson=destination,source=source,args=copy.deepcopy(c['args']),operation='构造本题提交的完整类，追踪子层边界来定位该步骤，并与保留的小题独立预期对照；不调用小题solve或参考实现。'),expected=copy.deepcopy(c['expected'])))
        lessons[source]['practice_for']=destination
    for meta in [core,mha]:
        meta['assessment_revision']=4
        meta['assessment_migration']=dict(previous_revision=3,reason='主线改为完整模块，原小题样例并入本题分项检查；旧代码和成绩保留。')
        meta['starter']=empty_methods(meta['reference'])
        imports='import numpy as np\nfrom learner_core import Parameter, Module, initialize\nif __name__ != "student":\n'
        imports+='    from t_softmax import Softmax\n' if meta is core else '    from layer_linear import Linear\n    from t_attention import AttentionCore\n'
        scaffold(meta,meta['starter'],imports,'_transformer_checks')

    # 去掉主线与下游的可选小题依赖；三个真正复用的模块仍按原接口接线。
    for meta in lessons.values():
        for field in ['dependencies','experiment_dependencies']:
            if field in meta:meta[field]=[d for d in meta[field] if d not in PRACTICE]
    stage=next(s for s in book['stages'] if s['title'].startswith('第六'))
    stage['modules']=[dict(name='从注意力到完整 Block',tasks=['t_attention','t_mha','t_block'])]
    stage['outcome']='完成注意力核心、多头注意力与双残差 Block，逐层组合成可用于 GPT 的完整模块。'
    practice=next((s for s in book['stages'] if s.get('id')=='optional-attention-practice'),None)
    if practice is None:
        practice=dict(id='optional-attention-practice',title='选修 · 注意力分步练习',optional=True,modules=[]);book['stages'].append(practice)
    practice['modules']=[dict(name='按需练习单个操作',tasks=PRACTICE.copy())]
    practice['outcome']='需要单独练习某一步时使用；完整实现与评测在第六部分的三个主线模块中完成。'

    # 反向段落就地重列前向，明确数学梯度与代码变量名的对应关系。
    notation=p('设最终标量损失为 '+m('L')+'，本章用 '+m(r'\nabla X')+' 表示损失对 '+m('X')+' 的梯度，与 '+m('X')+' 同形。')+aligned(r'\nabla X&=\frac{\partial L}{\partial X}',r'G&=\nabla Y=\frac{\partial L}{\partial Y}')
    core['body']=core['body'].replace('<h3>4. 反向沿同一条前向链返回</h3>',notation+'<h3>4. 反向沿同一条前向链返回</h3>')
    recalls={
        '① 读取的两侧都影响输出':aligned(r'\text{对应前向：}\quad Y&=AV',r'Y_{i,c}&=\sum_j A_{i,j}V_{j,c}'),
        '② 同一行概率的所有路径一起回到分数':aligned(r'\text{对应前向：}\quad A&=\operatorname{softmax}_{j}(\widetilde S)',r'A_{i,k}&=\frac{\exp(\widetilde S_{i,k})}{\sum_s\exp(\widetilde S_{i,s})}'),
        '③ 匹配的两侧分别收齐所有用途':aligned(r'\text{对应前向：}\quad S&=QK^{\mathsf T}/\sqrt d',r'S_{i,j}&=\frac1{\sqrt d}\sum_cQ_{i,c}K_{j,c}')}
    for title,forward in recalls.items():core['body']=core['body'].replace('<h3>'+title+'</h3>','<h3>'+title+'</h3>'+forward)
    core['body']=core['body'].replace('因果模式再把掩码位置明确置零。','因果模式再把掩码位置明确置零。掩码前向将未来分数替换为常量，所以该位置对原分数的局部导数为零；允许位置为一。')
    for key,title,forward in [
        ('t_mha','3. 反向把相同布局倒回去',aligned(r'\text{对应前向：}\quad Z&=\operatorname{qkv}(X)',r'(Q,K,V)&=\operatorname{splitHeads}(\operatorname{split}(Z))',r'O&=\operatorname{core}(Q,K,V)',r'M&=\operatorname{merge}(O),\quad Y=\operatorname{proj}(M)')),
        ('t_block','3. 为什么反向需要两次相加',aligned(r'\text{对应前向：}\quad X_1&=X+F_1(X)',r'Y&=X_1+F_2(X_1)',r'F_1&=\operatorname{attn}\circ\operatorname{norm1}',r'F_2&=\operatorname{MLPBranch}=\operatorname{mlp}\circ\operatorname{norm2}'))]:
        lessons[key]['body']=lessons[key]['body'].replace('<h3>'+title+'</h3>',notation+'<h3>'+title+'</h3>'+forward)
    intent=h('先理解：注意力为什么这样安排',p('注意力的核心目的，是让每个位置根据当前内容，从其他位置读取有用的信息。例如读到“他”时，上文的人名可能很重要。此前的 <code>Linear</code> 主要变换每个 token 自己的特征；注意力让不同 token 之间交换信息。')+'<ul><li>'+m('Q')+'：当前位置用什么条件寻找信息。</li><li>'+m('K')+'：其他位置提供什么特征供匹配。</li><li>'+m('V')+'：匹配到这个位置后，实际取回什么内容。</li></ul>'+p('把匹配特征和传递内容分开，模型可以分别学习“怎样找到相关位置”和“从那里带回什么”。这些用途通过训练形成。'))
    intent+=h('两种“权重”：保存的计算规则与本次读取比例','<div class="contract-table"><table><thead><tr><th>名称</th><th>是什么</th><th>什么时候变化</th></tr></thead><tbody><tr><td>'+m('W_Q,W_K,W_V')+'</td><td>模型保存的可训练投影矩阵</td><td>训练更新参数时变化</td></tr><tr><td>'+m('A')+'</td><td>当前输入算出的读取比例</td><td>输入变化时重新计算</td></tr></tbody></table></div>'+p('参数规定计算规则，计算结果还取决于输入：')+aligned(r'Q&=XW_Q,\quad K=XW_K,\quad V=XW_V',r'S&=QK^{\mathsf T}/\sqrt d,\quad A=\operatorname{softmax}(\widetilde S)')+p('这里 '+m(r'\widetilde S')+' 是应用因果掩码后的分数。在一次推理中，投影参数保持不变。换一句话，输入 '+m('X')+' 变化，算出的三组向量与读取比例也可能变化。')+'<blockquote>小明在跑步，他……<br>小刚在跑步，他……</blockquote>'+p('即使“他”的提问向量相同，前面“小明”和“小刚”提供的索引、内容也不同，因此“他”可以读回不同的上下文信息。训练调整投影参数，使这套计算能找到对预测有帮助的信息。模型保存计算规则，每次输入重新产生读取比例。')+p('实现中，三个投影矩阵合并存放在 <code>qkv.W</code> 中；按输出列分成三段就对应上述三个矩阵。注意力核心接收的是已经算好的三组向量。'))
    intent+=h('每一步都服务于读取上下文','<div class="contract-table"><table><thead><tr><th>操作</th><th>整个机制中的作用</th></tr></thead><tbody><tr><td>'+m(r'QK^{\mathsf T}')+'</td><td>给当前位置与其他位置的相关程度打分</td></tr><tr><td>除以 '+m(r'\sqrt d')+'</td><td>控制分数尺度，避免概率过早过于集中</td></tr><tr><td>因果掩码</td><td>预测下一个 token 时，只读取已经出现的信息</td></tr><tr><td><code>Softmax</code></td><td>将分数变成各位置的相对读取权重</td></tr><tr><td>'+m('Y=AV')+'</td><td>组合内容，得到融合上下文的新特征</td></tr></tbody></table></div>')
    intent+=h('多头、输出投影、MLP 和残差接着做什么',p('每个头使用不同的投影参数，所以同一份输入可以产生不同的匹配分数和读取结果。几个头同时从不同角度处理同一句话，训练决定各头具体学到什么用途。')+p('输出投影先把各头读回的特征拼在一起，再交给已学的 <code>Linear</code>，学习把不同头的信息组合为后续模块需要的表示：')+aligned(r'M&=\operatorname{concat}(Y^{(1)},\ldots,Y^{(H)})',r'Z&=MW_O+b_O')+p('注意力负责从其他 token 读取信息；<code>MLP</code> 对每个 token 已拥有的整组特征继续加工。残差让原有表示也参与下一步：')+aligned(r'R&=X+\operatorname{MHA}(\operatorname{Norm}_1(X))',r'Y&=R+\operatorname{MLP}(\operatorname{Norm}_2(R))')+p('这里的两个归一化层就是已经学过的 <code>RMSNorm</code>。因此 <code>MLP</code> 接收的表示里，已经包含原来的信息和注意力读回的信息。整个 <code>Block</code> 做的是：读取上下文，加入当前表示，再加工这份表示。接下来三个主线任务逐层实现这些完整模块。'))
    core['body']=intent+core['body']
    core['learning_page']['sample_note']=p('整体图说明完整流程，来源分项默认只展示一个 batch/head 切片；可切换切片核对同一规则。实际输入仍保留完整 '+m('B=2,H=2')+'，测评继续检查全部位置。')
    from attention_wrapper_lessons import apply_wrapper_explanations
    return apply_wrapper_explanations(book, math, aligned)
