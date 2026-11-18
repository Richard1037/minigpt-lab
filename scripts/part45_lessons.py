"""单一基础层实现、章末组合验收与概率损失教学的持久生成源。"""
import copy
import html
import re
import numpy as np

BRANCH_REFERENCE='''class MLPBranch(Module):
    def __init__(self, C, hidden, eps=1e-5):
        self.norm = RMSNorm(C, eps)
        self.mlp = MLP(C, hidden)

    def forward(self, x):
        return self.mlp(self.norm(x))

    def backward(self, g):
        return self.norm.backward(self.mlp.backward(g))
'''
BRANCH_STARTER='''class MLPBranch(Module):
    def __init__(self, C, hidden, eps=1e-5):
        raise NotImplementedError("请创建两个已有子层")

    def forward(self, x):
        raise NotImplementedError("请串联子层返回值")

    def backward(self, g):
        raise NotImplementedError("请逆序传递梯度")
'''
DEPS=['infrastructure','initialization','layer_scale','layer_linear','layer_embedding','layer_gelu','layer_rmsnorm','layer_mlp']


def scaffold(meta, starter, imports, checker):
    old=meta.get('local_code','')
    full=imports+'\n'+starter+'\n\n# 直接运行执行本节公开测评；作为模块导入时跳过。\nif __name__ == "__main__":\n    from '+checker+' import run_local\n    run_local("'+meta['id']+'", globals())\n'
    notes=meta.get('scaffold_notes',{})
    meta.update(starter=starter,sync_empty_scaffold=True,scaffold_notes=dict(source=notes.get('source',old),code=full,starter=full))


def apply_part45(book,math,aligned):
    m=lambda s:math(s,inline=True)
    p=lambda s:'<p>'+s+'</p>'
    h=lambda title,s:'<h3>'+title+'</h3>'+s
    lessons=book['lessons']
    # 构造责任回到唯一主线类，不再要求另一文件重写相同前后向。
    linear=lessons['layer_linear']
    linear['starter']=linear['starter'].replace('        self.W = Parameter(initialize((C, D)))\n        self.b = Parameter(np.zeros(D)) if use_bias else None','        raise NotImplementedError("请创建权重和可选偏置")')
    linear['body']=linear['body'].replace('给定构造器 <code>Linear(C, D, use_bias=True)</code>','你要实现的构造器 <code>Linear(C, D, use_bias=True)</code>').replace('这里补写前向和反向；后续独立 Linear 题会让你补写构造器。','在同一份类中完成构造、前向和反向，后续组合直接复用它。')
    linear['learning_page']['purpose']='在同一份个人基础层中完成参数构造、缓存、前向与反向，供后续模块直接复用。'
    linear['learning_page']['statement']=linear['learning_page']['statement'].replace('<code>Linear.forward(x)</code>、<code>Linear.backward(g)</code>；构造器 <code>Linear(C, D, use_bias=True)</code> 已给出。','<code>Linear.__init__(C, D, use_bias=True)</code>、<code>forward(x)</code>、<code>backward(g)</code>。')
    if '动手任务清单' not in linear['learning_page']['statement']:
        linear['learning_page']['statement']+=h('动手任务清单','<ol><li>调用个人 <code>initialize</code>，创建 <code>W</code> 和可选零偏置。</li><li>缓存形状与二维输入副本，计算并恢复前向输出。</li><li>累计参数梯度，返回原形状输入梯度。</li></ol>')+h('关键公式速查',aligned(r'Y_2&=X_2W+b',r'\mathrm dX_2&=G_2W^{\mathsf T},\quad\mathrm dW=X_2^{\mathsf T}G_2,\quad\mathrm db=\sum_n(G_2)_{n,:}'))
    extras=copy.deepcopy(lessons['rebuild_linear']['cases'][-2:])
    linear['cases']=[c for c in linear['cases'] if c.get('assessment') not in ['independent_linear_build','independent_linear_composition']]+extras
    linear['assessment_revision']=3;linear['assessment_migration']=dict(previous_revision=2,reason='构造检查合入唯一主线Linear，保留原8项，已有前后向实现继续使用。')
    linear['task_brief']['scope']='实现同一份Linear的构造、前向与反向'
    # 旧样例中有双重转义的LaTeX命令；保留真正的aligned换行。
    def clean_legacy_math(text):
        def fix(match):
            tex=html.unescape(match[3])
            if not re.search(r'\\\\[A-Za-z]',tex):return match[0]
            return math(tex.replace('\\\\','\\'),inline=match[2]=='inline')
        return re.sub(r'<(span|div) class="math-(inline|block)" data-tex="([^"]+)">.*?</\1>',fix,text,flags=re.S)
    for field in ['statement','sample_note']:
        linear['learning_page'][field]=clean_legacy_math(linear['learning_page'][field])
    scaffold(linear,linear['starter'],'import numpy as np\nfrom learner_core import Parameter, Module, initialize\n','_foundation_checks')

    meta=lessons.get('foundation_capstone',dict(id='foundation_capstone',title='基础层综合：组装并验收 MLP 分支',kind='layer',file='foundation_capstone.py',exports=['MLPBranch'],**{'class':'MLPBranch'}))
    meta.update(dependencies=DEPS,reference=BRANCH_REFERENCE,constructor_names=['C','hidden','eps'],assessment_revision=1,steps=[],derivation='',trace=[],goal='组合个人基础层，验证完整前后向和参数管理。',outcome='交付可复用的归一化与MLP分支。')
    meta['body']=h('1. 本节把已有成果接起来',p('你已经写过参数管理、初始化和各个基础层。本节新增一个 <code>MLPBranch</code>：保存一个个人 <code>RMSNorm</code> 和一个个人 <code>MLP</code>，把它们的返回值接起来。<code>MLP</code> 内部继续调用你的两个 <code>Linear</code> 和一个 <code>GELU</code>。')+p('这是后面 GPT Block 中的特征变换分支。最终 Block 会在分支外处理残差连接；现在先完成这条分支本身。'))
    meta['body']+=h('2. 构造：创建子层，之后一直复用',p('<code>MLPBranch(C, hidden, eps=1e-5)</code> 中，'+m(r'C')+' 是输入/输出特征宽度；<code>hidden</code> 为中间宽度，记为 '+m(r'M')+'；<code>eps</code> 是归一化分母中的正数。创建 <code>self.norm = RMSNorm(C, eps)</code> 与 <code>self.mlp = MLP(C, hidden)</code>。')+p('子层的参数与缓存由各自对象持有。参数列表依次是 <code>norm.w</code>、<code>mlp.fc1.W</code>、<code>mlp.fc1.b</code>、<code>mlp.fc2.W</code>、<code>mlp.fc2.b</code>；继承的 <code>parameters()</code> 负责递归收集这五个对象。'))
    meta['body']+=h('3. 前向：把返回值交给下一层',aligned(r'U&=\operatorname{RMSNorm}(X),\qquad Y=\operatorname{MLP}(U)',r'X&:(B,T,C)\to U:(B,T,C)\to Y:(B,T,C)',r'\operatorname{MLP}:\quad (B,T,C)&\to(B,T,M)\to(B,T,M)\to(B,T,C)')+p('<code>norm(x)</code> 得到 '+m(r'U')+'，把它传给 <code>mlp</code>，最终返回 '+m(r'Y')+'。完整保留所有位置；也支持二维和更多前置维度。子层已经保存了各自反向需要的输入，本组合只负责传值。'))
    meta['body']+=h('4. 反向：每个返回值都是前一层的上游',p('调用者给定 '+m(r'G=\partial L/\partial Y')+'。先执行 <code>mlp.backward(g)</code>，返回 '+m(r'\mathrm dU=\partial L/\partial U')+'；再把这个返回值交给 <code>norm.backward</code>，得到原输入梯度。')+aligned(r'\mathrm dU&=\operatorname{MLP.backward}(G)',r'\mathrm dX&=\operatorname{RMSNorm.backward}(\mathrm dU)')+p('各层自己累计参数梯度。本题复用它们已经实现的公式；整个分支返回 '+m(r'\mathrm dX')+'。新一轮梯度统计前，外部调用分支的 <code>zero_grad()</code>，五个参数同时清零，参数数值保持。'))
    meta['body']+=h('5. 本节怎样综合前面的能力','<div class="contract-table"><table><tr><th>已有成果</th><th>本节怎样实际检查</th></tr>'+''.join('<tr><td><code>'+a+'</code></td><td>'+z+'</td></tr>' for a,z in [('Parameter / Module','五个参数的真实身份、递归收集与原地清零'),('initialize / Linear','调用同一份个人实现，检查初始化、可选偏置和梯度'),('RMSNorm / MLP / GELU','实际分支前后向与所有参数梯度'),('Embedding','运行个人查表、重复编号回流及缓存检查'),('Scale','运行个人逐元素反向与共享标量梯度检查')])+'</table></div>'+p('测评分别复用各组件的真实接口。输入由测试给定，上游也由测试给定；模型训练中的损失接口在下一部分学习。'))
    statement=h('题目描述',p('实现 <code>MLPBranch</code>，将个人归一化层和特征变换层组合成可复用分支，并通过本部分基础层综合验收。'))+h('实现接口',p('<code>__init__(C, hidden, eps=1e-5)</code>、<code>forward(x)</code>、<code>backward(g)</code>。'))+h('输入格式',p('宽度为正整数，<code>eps</code> 为正实数。<code>x</code> 为至少二维、末轴长度为 '+m(r'C')+' 的实数数组；<code>g</code> 与本次输出同形。'))+h('输出格式',p('构造器保存 <code>norm</code>、<code>mlp</code> 两个子层；前向返回同形输出，反向返回同形输入梯度。'))+h('状态变化',p('前后向复用已有子层；参数梯度累计在子层中，统一清零由继承方法完成。输入、上游和参数数值保持原值。'))+h('动手任务清单','<ol><li>创建并保存两个个人子层。</li><li>前向先归一化，再传入特征变换层。</li><li>反向逆序传递返回值。</li><li>提交综合测评；若某个基础层失败，回到对应原文件修正。</li></ol>')+h('关键公式速查',aligned(r'X\xrightarrow{\rm norm}U&\xrightarrow{\rm mlp}Y',r'G\xrightarrow{\rm mlp.backward}\mathrm dU&\xrightarrow{\rm norm.backward}\mathrm dX'))
    meta['learning_page']=dict(role='miniGPT 真实分支的组合任务',purpose='只编写新的组合部分，直接复用已完成的个人基础层。',problem_title='组装归一化与 MLP 分支',statement=statement,sample_note=p('首例完整使用 '+m(r'(B,T,C,M)=(2,3,4,6)')+'。测评先设定公开参数，再检查六个位置的输出、输入梯度和五个参数梯度。构造、实例身份、缓存与组件覆盖另有公开检查。'))
    meta['extension']=h('接入完整 Block',p('后续将这条真实分支与注意力分支、残差连接组合。当前完整模型仍通过个人 <code>RMSNorm</code> 和 <code>MLP</code> 使用同一份基础层；分支类的直接接入登记在 Part06 清单。'))
    meta['task_brief']=dict(scope='只组装已有RMSNorm和MLP',variables=[dict(name='x',meaning='分支输入特征',shape='(B,T,C)，支持更多前置轴'),dict(name='g',meaning='分支输出收到的上游',shape='与x同形')],output='同形输出与输入梯度。',conditions=['复用个人子层；参数梯度留在子层。'],flow=['X → RMSNorm → MLP → Y'],dimensions=dict(B='序列数',T='位置数',C='特征宽度'),operation='串联已有子层，逆序传回梯度。',next_use=dict(lesson='t_block',mode='practice'))
    from _capstone_checks import oracle
    rng=np.random.default_rng(632);x=rng.normal(size=(2,3,4));g=rng.normal(size=x.shape)
    weights=[rng.normal(size=s)*.3 for s in [(4,),(4,6),(6,),(6,4),(4,)]]
    y,dx,grads=oracle(x,g,weights,1e-5);names=['norm.w','mlp.fc1.W','mlp.fc1.b','mlp.fc2.W','mlp.fc2.b']
    cases=[dict(**{'class':'MLPBranch'},init=[4,6,1e-5],X=x.tolist(),G=g.tolist(),parameters={n:v.tolist() for n,v in zip(names,weights)},expected=dict(Y=y.tolist(),dX=dx.tolist(),**{n+'.grad':v.tolist() for n,v in zip(names,grads)}),label='完整双batch分支：输出、输入梯度与五个参数梯度',category='综合组合')]
    for kind,shape in [('branch_composition',[2,3,4]),('branch_gradient',[2,2,3]),('branch_state',[2,2,3,4])]:
        cases.append(dict(assessment=kind,input=dict(C=shape[-1],hidden=shape[-1]+2,shape=shape,eps=1e-4,seed=633),expected={'综合分支检查':True},label={'branch_composition':'真实子层身份、调用顺序与参数收集','branch_gradient':'整体输入与每个参数张量的中心差分','branch_state':'四维输入、缓存隔离、梯度累计与统一清零'}[kind],category='综合验证'))
    for dep in DEPS:cases.append(dict(assessment='foundation_gate',input=dict(lesson=dep,operation='执行当前个人模块的全部公开用例，不注入参考层。'),expected={'复用个人模块':dep,'通过检查数':len(lessons[dep]['cases'])},label='复用已有成果：'+lessons[dep]['title'],category='基础层覆盖'))
    meta['cases']=cases
    imports='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n\n# 网页测评已依次加载个人前置模块；本地运行从原文件导入。\nif __name__ != "student":\n'
    for dep in DEPS[2:]:
        item=lessons[dep];imports+='    from '+item['file'][:-3]+' import '+', '.join(item.get('exports',[item.get('class','')]))+'\n'
    scaffold(meta,BRANCH_STARTER,imports,'_foundation_checks');lessons['foundation_capstone']=meta
    lessons['rebuild_linear']['superseded_by']='foundation_capstone'
    for stage in book['stages']:
        for mod in stage['modules']:mod['tasks']=[k for k in mod['tasks'] if k not in ['foundation_capstone','rebuild_linear']]
        stage['modules']=[mod for mod in stage['modules'] if mod['tasks']]
        if stage['title'].startswith('第四'):stage['modules'].append(dict(name='基础层综合验收',tasks=['foundation_capstone']))
    history=next((s for s in book['stages'] if s.get('historical')),None)
    if history is None:
        history=dict(title='可选 · 历史独立重写练习',optional=True,historical=True,modules=[]);book['stages'].append(history)
    history['modules']=[dict(name='保留旧代码与成绩',tasks=['rebuild_linear'])]+[v for v in history['modules'] if 'rebuild_linear' not in v['tasks']]
    apply_probability(book,math,aligned)
    return book


def apply_probability(book,math,aligned):
    m=lambda s:math(s,inline=True);p=lambda s:'<p>'+s+'</p>';h=lambda t,s:'<h3>'+t+'</h3>'+s
    for key in ['t_softmax','t_loss']:
        meta=book['lessons'][key];meta['derivation']='';meta['steps']=[];meta['trace']=[]
        meta['assessment_revision']=3;meta['assessment_migration']=dict(previous_revision=2,reason='保留原用例，新增缓存/返回值隔离及标量上游与极端输入检查。')
        cls='Softmax' if key=='t_softmax' else 'CrossEntropy'
        body=h('1. 本层接收什么，返回什么',p('前面的网络会为每个位置输出一行类别分数，记为 '+m(r'X\in\mathbb R^{B\times T\times V}')+'。'+m(r'B')+' 是序列数，'+m(r'T')+' 是每条序列的位置数，'+m(r'V')+' 是类别数。在语言模型里，类别就是词表中的 token 编号。每行独立处理，全部 '+m(r'N=BT')+' 个位置保留。'))
        body+=p('对更一般的输入 '+m(r'X:(s_0,\ldots,s_k,V)')+'，最后一轴仍为类别轴；前面的全部位置数为 '+m(r'N=\prod_{\ell=0}^k s_\ell')+'。三维时它就是 '+m(r'N=BT')+'；标签形状是相同的前置形状，反向恢复原分数形状。')
        if key=='t_softmax':
            meta['title']='Softmax：从分数到概率与反向'
            body+=p('<code>Softmax</code> 把分数变成同形概率 '+m(r'P')+'：每格非负，每一行之和为 '+m(r'1')+'。本层没有可训练参数；前向保存概率，反向接收损失对这些概率的梯度。')
            body+=h('2. 前向：平移、指数化、按行归一化',aligned(r'm_{b,t,0}&=\max_j X_{b,t,j}',r'Z&=X-m,\qquad E=\exp(Z)',r'S_{b,t,0}&=\sum_jE_{b,t,j},\qquad P=E/S')+p('先从本行每格减去本行最大值，最大的结果为 '+m(r'0')+'；指数后各项都是正数。再除以本行指数之和，得到概率。<code>max</code> 和 <code>sum</code> 都使用 <code>axis=-1, keepdims=True</code>：保留 '+m(r'(B,T,1)')+'，才能沿类别轴广播。指数使用 <code>np.exp</code>，不是逐行 Python 循环。')+p('减最大值使指数输入不大于零，避免大正数溢出；概率比例保持。完整等价性证明放在本节拓展。'))
            body+=h('3. 缓存和返回值',p('保存概率副本到 <code>self._p</code>，返回与输入同形的概率数组。返回值与缓存应独立：调用者修改返回的概率，不应破坏反向。本层没有需要手写的构造器，继承 <code>Module</code> 即可。'))
            body+=h('4. 为什么同一行的概率相互影响',p('固定一个位置，省略 '+m(r'b,t')+'。提高某个分数 '+m(r'x_j')+'，其指数增大；这一项也在公共分母里，所以 '+m(r'p_j')+' 增大，同一行其他概率会减小。其他位置没有共享分母，不受影响。')+aligned(r'p_i&=\frac{e^{x_i}}{\sum_k e^{x_k}}',r'\frac{\partial p_i}{\partial x_j}&=\frac{\delta_{ij}e^{x_i}\sum_ke^{x_k}-e^{x_i}e^{x_j}}{(\sum_ke^{x_k})^2}',r'&=p_i(\delta_{ij}-p_j)')+p(m(r'\delta_{ij}')+' 在 '+m(r'i=j')+' 时为 '+m(r'1')+'，其余为 '+m(r'0')+'。因此同类导数为 '+m(r'p_j(1-p_j)')+'，异类导数为 '+m(r'-p_i p_j')+'。'))
            body+=h('5. 反向：收集这行所有概率的上游',p('调用 <code>backward(g)</code> 时，'+m(r'G=\partial L/\partial P')+' 已由后续层给定，与概率同形。一个分数影响本行全部概率，所以要收齐这些路径：')+aligned(r'\mathrm dX_j&=\sum_iG_i\frac{\partial p_i}{\partial x_j}',r'&=G_jp_j-p_j\sum_iG_ip_i',r'&=p_j\left(G_j-\sum_iG_ip_i\right)')+p('先计算逐元素乘积，再沿最后轴求和并保留单例轴，得到每行共用的 '+m(r'A=\sum_iG_iP_i')+'。最后按 '+m(r'\mathrm dX=P\odot(G-A)')+' 计算同形输入梯度。'+m(r'\odot')+' 表示逐元素相乘。上游不额外平均，也不修改上游或缓存。'))
            summary=aligned(r'Z&=X-\max(X,\mathrm{axis}=-1,\mathrm{keepdims}=\mathrm{True})',r'P&=\exp(Z)/\sum_j\exp(Z_j)',r'A&=\sum_jG_jP_j\quad\text{（保留类别轴）}',r'\mathrm dX&=P\odot(G-A)')
            inputs='<code>forward(x)</code> 接收至少二维的实数分数数组；<code>backward(g)</code> 接收与概率同形的实数上游。类别数为正，先前向再反向。'
            outputs='前向返回同形概率数组；反向返回同形输入梯度。参数列表为空。'
            actions=['前向沿最后轴平移、指数化并归一化。','保存独立概率缓存，返回同形概率。','反向计算行内加权上游和，再返回输入梯度。']
            state='保存本次概率缓存；修改返回概率或原输入不改变缓存。反向保留上游原值，多次反向返回一致结果。'
            meta['extension']=h('减最大值为什么不改变概率',aligned(r'\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}&=\frac{e^{-m}e^{x_i}}{e^{-m}\sum_j e^{x_j}}=\frac{e^{x_i}}{\sum_j e^{x_j}}')+p('反向公式直接由未平移的等价函数推导，无需单独实现最大值算子的反向。术语“向量雅可比积”指把上游与这一行完整局部导数组合；实现时只需要上面的归约公式。'))
        else:
            body+=p('<code>CrossEntropy</code> 还接收整数标签 <code>targets</code>，形状为 '+m(r'(B,T)')+'；每个编号指出该位置正确的下一 token。前向把所有位置的预测压缩成一个平均损失，反向把这个标量的梯度传回每个类别分数。')
            body+=h('2. 从概率看单个位置的损失',p('正确类别编号记为 '+m(r'y')+'。它的概率越大，负对数损失越小：')+math(r'\ell=-\log p_y')+p('这里 '+m(r'\log')+' 是自然对数，NumPy 使用 <code>np.log</code>。在同一行，标签只选择一个类别作为正确答案，但所有类别分数都会影响这个类别的概率。'))
            body+=h('3. 稳定地直接计算对数概率',p('先将每行分数减去本行最大值，得到 '+m(r'Z')+'。设本行指数之和为 '+m(r'S')+'，则：')+aligned(r'Z&=X-\max(X,\mathrm{axis}=-1,\mathrm{keepdims}=\mathrm{True})',r'S&=\sum_j\exp(Z_j)',r'\log P&=Z-\log S')+p('求和使用 <code>axis=-1, keepdims=True</code>，以 '+m(r'(B,T,1)')+' 广播回类别轴。这与上一题概率的对数等价；直接计算对数概率能保留极小正确类概率对应的有限损失。用它得到反向所需的 '+m(r'P=\exp(\log P)')+' 并保存到 <code>self._p</code>。'))
            body+=h('4. 按标签取正确类，再对全部位置平均',p('先把 <code>targets</code> 变为 <code>targets[..., None]</code>，形状从 '+m(r'(B,T)')+' 变为 '+m(r'(B,T,1)')+'。<code>np.take_along_axis(logp, targets[..., None], axis=-1)</code> 在每个位置的最后轴取对应类别，返回 '+m(r'(B,T,1)')+'。')+aligned(r'Q_{b,t,0}&=(\log P)_{b,t,\,\mathrm{targets}_{b,t}}',r'L&=-\frac{1}{BT}\sum_{b,t}Q_{b,t,0}')+p('取负号并对这全部 '+m(r'N=BT')+' 个值求均值，返回零维 NumPy 数组，即 <code>np.array(...)</code>，形状 '+m(r'()')+'。标签保留整数类型；将其独立副本保存到 <code>self._targets</code>，反向还要知道每行的正确类别。'))
            body+=h('5. 反向来源：所有类别贡献概率，正确类再减一',p('将位置展平成 '+m(r'n')+'，先看一个位置的损失。对数归一化项依赖所有类别，正确类分数单独减去：')+aligned(r'\ell_n&=\log\sum_k e^{x_{n,k}}-x_{n,y_n}',r'\frac{\partial}{\partial x_{n,j}}\log\sum_ke^{x_{n,k}}&=\frac{e^{x_{n,j}}}{\sum_ke^{x_{n,k}}}=p_{n,j}',r'\frac{\partial\ell_n}{\partial x_{n,j}}&=p_{n,j}-\mathbf1[j=y_n]')+p(m(r'\mathbf1[j=y_n]')+' 表示“这个类别是否正确”：正确为 '+m(r'1')+'，其他为 '+m(r'0')+'。因此从概率的独立副本开始，每行只在标签指出的位置减一；其他类别仍保留其概率。'))
            body+=h('6. 平均与标量上游只在这里乘一次',p('<code>backward(g=1.0)</code> 的 '+m(r'g')+' 是损失标量收到的上游，默认 '+m(r'1')+'，也可以为零或负数。平均损失对每个位置分配 '+m(r'1/N')+'，所以：')+math(r'\mathrm dX_{n,j}=\frac{g}{N}\left(p_{n,j}-\mathbf1[j=y_n]\right)')+p('恢复全部原前置维度后返回输入梯度。后续基础层直接接收这个结果，保留其中已有的平均系数。整数标签没有梯度。每次反向都从缓存概率的副本开始，才能重复调用而不把“减一”累积到缓存中。'))
            summary=aligned(r'\log P&=Z-\log\sum_j e^{Z_j},\quad Z=X-\max_jX_j',r'X&:(s_0,\ldots,s_k,V),\quad N=\prod_{\ell=0}^k s_\ell',r'L&=-\frac1N\sum_n\log P_{n,y_n}',r'\mathrm dX_{n,j}&=\frac gN(P_{n,j}-\mathbf1[j=y_n])')
            inputs='<code>forward(logits, targets)</code> 接收至少二维实数分数数组和去掉最后轴后的整数标签数组；标签范围为 '+m(r'0\le y<V')+'。<code>backward(g=1.0)</code> 接收实数标量上游。先前向再对应反向。'
            outputs='前向返回零维 NumPy 平均损失；反向返回与原分数同形的梯度。参数列表为空。'
            actions=['直接计算稳定对数概率。','按标签取正确类，取负后对全部位置平均。','缓存独立概率和整数标签副本。','反向从概率副本在正确类减一，乘标量上游并除位置数，恢复形状。']
            state='缓存概率与标签副本；原输入或标签被修改不影响对应反向。每次反向从缓存副本开始，保持原缓存和输入上游。'
            meta['extension']=h('为什么稳定对数概率重要',p('极端错误预测可能使正确类概率在浮点数中下溢为零，先算概率再取对数会失去有限损失。'+m(r'\log P=Z-\log\sum_j e^{Z_j}')+' 可以直接保留这个对数值。完整训练和参数更新在后面的训练章节组合。'))
        meta['body']=body
        statement=h('题目描述',p('实现 <code>'+cls+'</code> 的前向与反向，作为 miniGPT 中的概率/损失组件。'))+h('实现接口',p('<code>'+('forward(x)</code>、<code>backward(g)' if key=='t_softmax' else 'forward(logits, targets)</code>、<code>backward(g=1.0)')+'</code>。'))+h('输入格式',p(inputs))+h('输出格式',p(outputs))+h('状态变化',p(state))+h('动手任务清单','<ol>'+''.join('<li>'+a+'</li>' for a in actions)+'</ol>')+h('关键公式速查',summary)
        meta['learning_page']=dict(role='miniGPT 的概率组件' if key=='t_softmax' else 'miniGPT 的训练损失组件',purpose='把网络分数变为逐位置概率，并把上游沿同一行的全部影响传回。' if key=='t_softmax' else '用整数标签选择正确类别，得到平均损失并返回每个分数的梯度。',problem_title='实现稳定概率与反向' if key=='t_softmax' else '实现稳定平均交叉熵',statement=statement,sample_note=p('本例完整保留 '+m(r'(B,T,V)=(2,3,4)')+' 的六个位置。'+('每行四个概率归一化；给定上游回传同形梯度。' if key=='t_softmax' else '六个标签分别选择本行类别，六项负对数概率共同平均；反向每行都保留全部四类。')))
        meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='probability_contract']
        for shape in [[2,3,5],[2,2,3,4],[3,2,1]]:
            meta['cases'].append(dict(assessment='probability_contract',input=dict(**{'class':cls},shape=shape,seed=725),expected={'概率损失接口与缓存':True},label='接口与缓存：'+str(tuple(shape))+('，概率返回值隔离/独立位置' if key=='t_softmax' else '，非单位/零/负上游及极端标签'),category='缓存与接口'))
        starter='class '+cls+'(Module):\n    def '+('forward(self, x)' if key=='t_softmax' else 'forward(self, logits, targets)')+':\n        raise NotImplementedError("请实现前向与缓存")\n\n    def '+('backward(self, g)' if key=='t_softmax' else 'backward(self, g=1.0)')+':\n        raise NotImplementedError("请实现反向")\n'
        scaffold(meta,starter,'import numpy as np\nfrom learner_core import Module\n','_transformer_checks')
