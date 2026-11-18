"""数据准备到完整GPT：保持两道完整任务，学习者实现所有计算与组合。"""
from part45_lessons import scaffold
from part06_lessons import empty_methods

GPT = '''class GPT(Module):
    def __init__(self, V, C, H, hidden, layers, context, seed=0):
        if min(V, C, H, hidden, layers, context) <= 0 or C % H:
            raise ValueError('尺寸须为正且 C 能被 H 整除')
        self.context = context
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(2 * layers)
        self.token = Embedding(V, C, rng=rng)
        self.position = Embedding(context, C, rng=rng)
        self.blocks = [Block(C, H, hidden, rng=rng, init_std=.02, init_scale=scale) for _ in range(layers)]
        self.norm = RMSNorm(C)

    def forward(self, ids):
        if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('ids 须为二维整数数组')
        if not 0 < ids.shape[1] <= self.context:
            raise ValueError('序列长度须在 context 内')
        x = self.token(ids) + self.position(np.arange(ids.shape[1]))[None, :, :]
        for block in self.blocks:
            x = block(x)
        self._h = self.norm(x)
        return self._h @ self.token.weight.data.T

    def backward(self, g):
        C = self._h.shape[-1]
        self.token.weight.grad += g.reshape(-1, g.shape[-1]).T @ self._h.reshape(-1, C)
        dx = self.norm.backward(g @ self.token.weight.data)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.position.backward(dx.sum(axis=0))
        self.token.backward(dx)
        return None
'''


def apply_part07(book, math, aligned):
    m = lambda s: math(s, inline=True)
    p = lambda s: '<p>'+s+'</p>'
    h = lambda title, content: '<h3>'+title+'</h3>'+content
    ul = lambda a: '<ul>'+''.join('<li>'+v+'</li>' for v in a)+'</ul>'
    fold = lambda title, content: '<details class="block-detail"><summary>'+title+'</summary><div class="block-detail-body">'+content+'</div></details>'
    lessons=book['lessons'];data=lessons['t_data'];gpt=lessons['t_gpt']
    def page(meta, purpose, interface, inputs, outputs, state, actions, formula, sample):
        meta.update(definition='',derivation='',steps=[],hint='先看核心公式与题末清单，再展开对应步骤核对。')
        meta['task_brief']=dict(scope=interface,variables=[],operation=purpose,output=outputs,conditions=[],next_use=None)
        meta['learning_page']=dict(role='miniGPT 组成',problem_title=meta['title'],purpose=purpose,
            statement=h('题目描述',p('实现 '+interface+'。')+p(purpose))+h('输入格式',p(inputs))+h('输出格式',p(outputs))+h('状态变化',p(state))+h('动手任务清单',actions if isinstance(actions,str) else ul(actions))+h('关键公式速查',formula),sample_input='',sample_note=p(sample))

    data['title']='文本到训练批次：字符编码与下一字符目标'
    df=aligned(r'\mathrm{ids}_i&=\operatorname{stoi}(\mathrm{text}_i)',r'X_{b,t}&=\mathrm{ids}_{s_b+t},\qquad \mathrm{targets}_{b,t}=\mathrm{ids}_{s_b+t+1}')
    data['body']=h('先看整体：模型怎样得到输入和正确答案',p('GPT 接收整数编号。你要把一段文本变成编号序列，再取出几段连续窗口：窗口里的每个字符是输入，它后面的一个字符就是训练目标。模型输出每个位置对下一字符的分数，下一部分训练时再用交叉熵比较这些分数与目标。')+p('因此本题交付两个函数：<code>tokenize</code> 建立字符词表并编码，<code>make_batch</code> 取输入及右移一位的目标。')+df)
    encode=p('Python 字符串可以按字符迭代。<code>set(text)</code> 去掉重复字符，<code>sorted</code> 按字符的 Unicode 顺序排序；<code>enumerate(vocab)</code> 给排序结果依次编号，建立 <code>stoi</code> 字典。再按原文顺序逐字符查字典，输出 <code>np.int64</code> 数组。空格与换行也各是一个字符，不能先去掉。')+p('例如文本“猫喜欢鱼猫喜欢鱼”，排序词表为“喜、欢、猫、鱼”，对应编号 '+m('0,1,2,3')+'。按原文编码得到：')+math(r'\mathrm{ids}=\begin{bmatrix}2&0&1&3&2&0&1&3\end{bmatrix}')+p('返回的 <code>vocab</code> 是字符列表。已有编号可以用 <code>vocab[i]</code> 查回字符；将字符拼接就能解码。生成阶段继续使用这份词表，不能重新排序另一段文本来改变编号含义。')
    windows=p(m('B')+' 是窗口数，'+m('T')+' 是每个窗口的输入长度，'+m('N')+' 是编号序列长度，'+m('s_b')+' 是 <code>starts[b]</code>。Python 切片右端不包含在结果中，因此取输入用 <code>ids[s:s+T]</code>，目标用 <code>ids[s+1:s+T+1]</code>；每个窗口需要 '+m('T+1')+' 个连续编号。合法起点满足 '+m(r'0\le s_b\le N-T-1')+'。')+df+p('用同一份八个编号、'+m('T=3')+'：起点 '+m('0')+' 得到第一行，最后合法起点 '+m('4')+' 得到第二行：')+aligned(r'X&=\begin{bmatrix}2&0&1\\2&0&1\end{bmatrix},\qquad \mathrm{targets}=\begin{bmatrix}0&1&3\\0&1&3\end{bmatrix}')+p('这里两行相同源于原文恰好重复，不是复制第一行。<code>starts</code> 的顺序决定批次行顺序，重复起点也要保留；用 <code>np.stack</code> 将各窗口堆成 '+m('(B,T)')+'。输入和目标不会覆盖原始编号。')
    data['body']+=fold('01 建立词表与编码：字符怎样对应整数',encode)+fold('02 取输入和目标：同一窗口怎样右移一位',windows)
    data['body']+=p('再核对公开样例的起点 3：从原序列位置 '+m('3')+' 开始取输入，从位置 '+m('4')+' 开始取目标；两者各取 '+m('T')+' 个编号，目标的最后一项来自输入窗口之后的那个位置。')
    data['body']+=h('这份数据交给谁',p('<code>GPT.forward(X)</code> 生成 '+m('(B,T,V)')+' 分数，'+m('V')+' 是词表大小；<code>CrossEntropy</code> 接收这些分数和 '+m('(B,T)')+' 的 <code>targets</code>。训练/验证先确定数据边界，再分别取窗口，避免目标跨到另一份数据。词表的建立范围由语料约定决定；本题只使用传入的 <code>text</code>，后续各阶段保存并复用其编号规则。'))
    data['body']+=h('核心总结：字符编号化，再让每个位置预测下一个字符',df+p('编码保留原文顺序；输入与目标来自同一原序列，相差一个位置。本题处理数据，没有梯度，也不训练参数。'))
    page(data,'把文本转换为 GPT 输入与下一字符训练目标。','<code>tokenize(text)</code>、<code>make_batch(ids, starts, T)</code>',
        '<code>text</code> 为非空字符串；<code>ids</code> 为一维整数数组；<code>starts</code> 为非空一维整数起点数组；'+m('T>0')+'，每个起点均合法。',
        '<code>tokenize</code> 返回排序的字符列表及一维 '+m('(N,)')+' 编号；<code>make_batch</code> 返回 '+m(r'X,\mathrm{targets}:(B,T)')+'，均为 <code>np.int64</code> 数组。',
        '输入数组和起点保持原值；没有参数或反向缓存。',
        ['排序去重建立 <code>vocab</code>，用字典将原文编码成整数数组。','按每个起点分别取输入窗口及右移目标，保留顺序和重复起点。','堆叠成两个二维数组，不修改传入的编号或起点。'],df,
        '第一项公开样例使用起点 '+m('0,3')+'：分别从原文的第零与第三个位置取窗口。目标从各自起点后一个位置开始，不是把输入末尾循环挪回开头。')
    data['extension']=h('词表与分词器的边界',p('本项目按 Python 字符编码，适合观察完整链路。实际语言模型也会使用子词分词。两者的共同要求是：模型、训练数据、检查点和解码使用相同编号规则。'))
    data['task_brief']['conditions']=['0≤s≤N−T−1；每个窗口需要 T+1 个编号。','输入与目标均为 int64，不修改原编号数组。']
    data['task_brief']['variables']=[dict(name='ids',meaning='按原文顺序排列的字符编号',shape='(N,)'),dict(name='starts',meaning='窗口起点，保留顺序和重复项',shape='(B,)')]

    gpt['title']='完整 GPT：从编号到下一 token 分数'
    gf=aligned(r'X_0&=E[\mathrm{ids}]+P[0:T]',r'X_{\ell+1}&=\operatorname{Block}_{\ell}(X_\ell),\quad \ell=0,\ldots,L-1',r'Z&=\operatorname{RMSNorm}(X_L)',r'S&=ZE^{\mathsf T}')
    init=p('上一题已经让各模块在构造时初始化自己的权重。GPT 创建一个 <code>np.random.default_rng(seed)</code>，计算出口系数 '+m(r'1/\sqrt{2L}')+'，将同一生成器交给两张表和各个 <code>Block</code>。Block 接收 <code>init_std=0.02</code> 与出口 <code>init_scale</code>，内部组件各自处理参数。')+p('不再收集权重或寻找出口；参数在创建时只采样一次。构造完成后，前向只使用已有权重。公开对象仍为 <code>token</code>、<code>position</code>、<code>blocks</code>、<code>norm</code>、<code>context</code>。')
    from part07_gpt_explanation import explanation, implementation_checklist
    gpt['body']=explanation(math, aligned, init)
    actions,formulas=implementation_checklist(math,aligned,gf)
    page(gpt,'亲手组装并初始化完整 GPT，得到每个位置的下一 token 分数与全部参数梯度。','<code>GPT.__init__(V, C, H, hidden, layers, context, seed=0)</code>、<code>forward(ids)</code>、<code>backward(g)</code>',
        '各构造尺寸为正整数，'+m(r'C\bmod H=0')+'；'+m(r'L=\mathrm{layers}')+'。前向是 '+m(r'\mathrm{ids}:(B,T),\ 1\le T\le\mathrm{context}')+' 的整数编号，范围 '+m(r'0\le\mathrm{ids}<V')+'；反向是 '+m('G:(B,T,V)')+' 实数上游。',
        '前向返回 '+m('S:(B,T,V)')+' 实数分数数组；反向返回 <code>None</code>，各个人参数梯度累加。',
        '构造时由各模块按同一随机流初始化自己的参数，仅两个出口缩放一次；归一化缩放与偏置保留默认值。前向缓存最近一次隐藏表示，各子层持有各自缓存；前后向不更新权重，不修改输入或上游。',
        actions,formulas,
        '第一项公开样例包含两个 batch、三个 token、两个 Block。每个位置对完整词表给分，全部位置共同参与上游回传；共享表只注册一次。构造与状态检查另核对当前个人子层、随机顺序、原地初始化及重复前向权重不变。')
    # 标识排版只处理已知普通文案，不改变公式与代码节点。
    gpt['task_brief']['variables']=[dict(name='ids',meaning='输入 token 编号；整数不求导',shape='(B,T)'),dict(name='g',meaning='输出分数的上游梯度',shape='(B,T,V)')]
    for name in ['Embedding','RMSNorm','Block','RNG','arange','None']:
        for field in ['statement']:
            # actions在上面是普通文字；显式替换这些确切短语即可。
            for old,new in [(f'个人 {name}',f'个人 <code>{name}</code>'),(f'最终 {name}',f'最终 <code>{name}</code>'),(f'返回 {name}',f'返回 <code>{name}</code>'),(f'用 {name} ',f'用 <code>{name}</code> ')]:
                gpt['learning_page'][field]=gpt['learning_page'][field].replace(old,new)
    gpt['extension']=h('为什么采用这个初始化配置',p('本项目选择二维权重标准差 '+m('0.02')+' 与残差出口 '+m(r'1/\sqrt{2L}')+' 缩放，是一套具体模型配置。其目标是控制初始表示与多层残差补充的尺度；不同架构可以采用其他规则。这里考查准确实现选定配置，不把它当作所有网络的通用数学定律。'))
    gpt['reference']=GPT;gpt['starter']=empty_methods(GPT)
    data['starter']='def tokenize(text):\n    raise NotImplementedError("请建立词表并编码")\n\ndef make_batch(ids, starts, T):\n    raise NotImplementedError("请取窗口与右移目标")\n'
    imports='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n'
    scaffold(data,data['starter'],imports,'_transformer_checks')
    imports+='if __name__ != "student":\n'+''.join('    '+v+'\n' for v in ['from layer_embedding import Embedding','from layer_rmsnorm import RMSNorm','from t_mha import MultiHeadAttention','from foundation_capstone import MLPBranch','from t_block import Block'])
    scaffold(gpt,gpt['starter'],imports,'_transformer_checks')
    for key in ['t_data','t_gpt']:
        meta=lessons[key];meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='part07_contract']
        for variant,label in [('build','构造与语义初始化' if key=='t_gpt' else '词表、空白字符与编解码规则'),('state','共享两路、缓存与重复反向' if key=='t_gpt' else '窗口顺序、最后起点与输入保持')]:
            meta['cases'].append(dict(label=label,assessment='part07_contract',input=dict(lesson=key,variant=variant),expected=dict(要求='当前个人实现满足讲解中的构造、输入输出与状态约定')))
        meta['assessment_revision']=3;meta['assessment_migration']=dict(previous_revision=2,reason='第七部分补充学习者GPT构造/初始化与完整状态、数据语法行为检查；原数值用例保留。')
    from initialization_upgrade import apply_upgrade
    apply_upgrade(book,math,aligned)
    gpt['dependencies']=list(dict.fromkeys(gpt['dependencies']+['t_init_upgrade']))
    gpt['assessment_revision']=4
    gpt['cases']=[c for c in gpt['cases'] if c.get('assessment')!='initialization_upgrade']
    gpt['cases'].append(dict(label='构造时一次初始化，GPT不重置子层权重',assessment='initialization_upgrade',input=dict(variant='gpt'),expected=dict(要求='同一生成器、十张矩阵各采样一次'),category='初始化封装'))
    gpt['assessment_migration']=dict(previous_revision=3,reason='新增模块构造时初始化与接口升级先修；原数值样例保持，历史成绩保留。')
    stage=next(s for s in book['stages'] if s['title'].startswith('第七'))
    stage['modules']=[dict(name='从文本到完整预测器',tasks=['t_data','t_init_upgrade','t_gpt'])]
    stage['outcome']='先准备输入与下一字符目标，再亲手构造、初始化并反向传播完整的个人 GPT。'
    # 下游只补齐GPT新增先修的接线，不改后续章节正文或验收标准。
    for key,meta in lessons.items():
        for field in ['dependencies','experiment_dependencies']:
            original=meta.get(field,[])
            if key not in ('t_gpt','t_init_upgrade') and 't_gpt' not in original:continue
            ordered=[];visiting=set()
            def add(dep):
                if dep in ordered:return
                if dep in visiting:raise ValueError('初始化升级依赖存在循环：'+dep)
                visiting.add(dep)
                for child in lessons[dep].get('dependencies',[]):add(child)
                visiting.remove(dep);ordered.append(dep)
            for dep in original:add(dep)
            if field in meta:meta[field]=ordered
    return book
