"""GPT组合课：先追溯变量来源，再沿同一例子贯通前向。"""


def implementation_checklist(math, aligned, forward):
    m=lambda s:math(s,inline=True)
    p=lambda s:'<p>'+s+'</p>'
    def group(title,items):
        return '<h4>'+title+'</h4><ol>'+''.join('<li>'+x+'</li>' for x in items)+'</ol>'
    actions=group('一、构造器 __init__：创建并初始化模型',[
        '保存 <code>self.context</code>；创建一个 <code>np.random.default_rng(seed)</code>，计算出口系数 '+m(r'1/\sqrt{2L}')+'。',
        '创建 <code>self.token</code> 与 <code>self.position</code> 两个个人 <code>Embedding</code>，都传入同一个 <code>rng</code>。',
        '创建 '+m('L')+' 个独立 <code>Block</code>，各自传入同一个 <code>rng</code>、<code>init_std=0.02</code> 和出口 <code>init_scale</code>；再创建最终 <code>self.norm = RMSNorm(C)</code>。',
        '初始化由上一题升级的个人组件在构造时完成。GPT 无需遍历内部参数。'])
    actions+=group('二、forward(ids)：从编号算出预测分数',[
        '从 <code>ids.shape</code> 读出 '+m('B,T')+'。调用 <code>self.token(ids)</code> 得到字符向量；用 <code>np.arange(T)</code> 产生位置编号，交给 <code>self.position</code> 查表。',
        '把位置向量扩成 '+m('(1,T,C)')+'，与字符向量广播相加，得到 '+m('X_0:(B,T,C)')+'。',
        '按列表顺序调用各个 <code>Block</code>，每次把上一个返回值交给下一个；再调用 <code>self.norm</code>，得到 '+m('Z')+' 并保存为 <code>self._h</code>。',
        '用 '+m('Z')+' 乘以字符表权重的转置，返回 '+m('S:(B,T,V)')+'。输出直接使用 <code>self.token.weight.data</code>，与输入查表共享参数。'])
    actions+=group('三、backward(g)：接好已有组件的反向调用',[
        '传入的 <code>g</code> 就是 '+m(r'G=\nabla S')+'。读取缓存的 '+m('Z')+'；把二者的批次与位置轴合并，得到 '+m('G_2:(BT,V)')+' 和 '+m('Z_2:(BT,C)')+'。',
        '计算输出头给字符表的贡献 '+m(r'G_2^{\mathsf T}Z_2')+'，累加到 <code>self.token.weight.grad</code>。再计算 '+m('GE')+'，作为传给最终归一化层的上游。',
        '调用 <code>self.norm.backward</code>，再从最后一个 <code>Block</code> 到第一个依次调用 <code>backward</code>。持续传递返回值，最后得到 '+m(r'\nabla X_0')+'，代码中可记作 <code>dx</code>。',
        '调用 <code>self.position.backward(dx.sum(axis=0))</code>：先沿批次轴求和，再交给位置查表层。',
        '调用 <code>self.token.backward(dx)</code>：字符查表层按编号累计输入侧贡献，与已存入的输出头贡献相加。最后返回 <code>None</code>。'])
    actions+=p('各子层沿用自己的缓存与反向实现。需要清空梯度时调用已有 <code>Module.zero_grad</code>；本次反向负责累计参数梯度。')
    formulas='<h4>初始化与完整前向</h4>'+aligned(r'W_{\mathrm{2D}}\sim\mathcal N(0,0.02^2),\qquad W_{\mathrm{exit}}\leftarrow W_{\mathrm{exit}}/\sqrt{2L}')+forward
    formulas+='<h4>完整反向</h4>'+p('令上游 '+m(r'G=\nabla S')+'，则：')+aligned(
        r'\nabla E_{\mathrm{head}}&=G_2^{\mathsf T}Z_2,\qquad\nabla Z=GE',
        r'\nabla X_L&=\operatorname{norm.backward}(\nabla Z)',
        r'\nabla X_\ell&=\operatorname{Block}_\ell.\operatorname{backward}(\nabla X_{\ell+1}),\quad\ell=L-1,\ldots,0',
        r'\nabla P_{t,:}&=\sum_b(\nabla X_0)_{b,t,:}',
        r'(\nabla E_{\mathrm{lookup}})_{v,:}&=\sum_{\mathrm{ids}_{b,t}=v}(\nabla X_0)_{b,t,:}',
        r'\nabla E&=\nabla E_{\mathrm{head}}+\nabla E_{\mathrm{lookup}}')
    formulas+=p('其中 '+m(r'G_2,Z_2')+' 表示将批次与位置两轴合并后的二维数组。')
    formulas+=p('读法：输出头 → 最终归一化 → 逆序经过各个 <code>Block</code> → 两张查表；字符表的两路贡献相加。')
    return actions,formulas


def explanation(math, aligned, initialization):
    m=lambda s:math(s,inline=True)
    p=lambda s:'<p>'+s+'</p>'
    h=lambda t,s:'<h3>'+t+'</h3>'+s
    fold=lambda t,s,opened=False:'<details class="block-detail"'+(' open' if opened else '')+'><summary>'+t+'</summary><div class="block-detail-body">'+s+'</div></details>'
    initialization=initialization.replace(' Block',' <code>Block</code>').replace(' MLP',' <code>MLP</code>')
    def table(head,rows):
        return '<div class="gpt-origin-table"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

    body=h('这节完成什么：输入编号，输出下一字符的分数',
        p('输入一批字符编号，输出每个位置对“下一个字符”的预测分数。例如输入“猫喜欢”，三个位置分别预测：看到“猫”后、看到“猫喜”后、看到“猫喜欢”后，下一个字符可能是什么。')+
        p('你已经写好的 <code>Block</code> 负责结合上下文。本节把它与查表、最终归一化及输出部分接成完整模型。先看输入从哪里来，再看构造器创建什么，随后沿同一个例子完成前向。'))

    body+=h('准备输入 · 外面传进来的东西',
        p('上一题的词表按“喜、欢、猫、鱼”排序，对应编号 '+m('0,1,2,3')+'。取“猫喜欢”和“鱼猫喜”作为一批输入，传给 <code>forward(ids)</code> 的数组是：')+
        math(r'\mathrm{ids}=\begin{bmatrix}2&0&1\\3&2&0\end{bmatrix}')+
        p('从当前输入形状直接读出 '+m('B=2')+'、'+m('T=3')+'：这一批有两段文本，每段三个字符。这两个数来自本次输入；下一次可以不同。')+
        p('另一些数在创建模型时选定。下面是贯穿本页的示例配置：')+
        table(['构造参数','示例值','含义与来源'],[
            ['<code>V</code>',m('4'),'词表大小，由上一题建立的词表决定。'],
            ['<code>C</code>',m('6'),'每个位置的特征宽度，由我们选择。'],
            ['<code>H</code>',m('2'),'每个注意力层的头数，由我们选择。'],
            ['<code>hidden</code>',m('12'),'<code>MLP</code> 内部扩展后的宽度。'],
            ['<code>layers</code>',m('2'),'堆叠几个 <code>Block</code>，公式中记作 '+m('L')+'。'],
            ['<code>context</code>',m('5'),'模型支持的最大输入长度。'],
            ['<code>seed</code>',m('0'),'初始化随机数的种子。']])+
        p(m('T=3')+' 是这次输入三个字符，<code>context</code> 为 '+m('5')+' 表示最多支持五个字符。'+m('C=6')+' 表示每个位置使用六维向量；六个特征没有人为指定的固定语义，会在训练中共同形成有用的表示。'))

    body+=h('创建模型 · 参数和子层从哪里来',
        p('构造器 <code>__init__</code> 创建并保存下面这些对象，之后每次前向复用它们：')+
        table(['对象','创建后持有什么'],[
            ['<code>self.token = Embedding(V, C, rng=rng)</code>','字符查表层：每个字符保存一个特征向量。'],
            ['<code>self.position = Embedding(context, C, rng=rng)</code>','位置查表层：每个位置保存一个特征向量。'],
            ['<code>self.blocks</code>','两个独立的个人 <code>Block</code> 实例，各有参数和缓存。'],
            ['<code>self.norm = RMSNorm(C)</code>','最后一个独立的归一化层。'],
            ['<code>self.context</code>','保存最大长度限制。']])+
        p('上表中的赋值是对象接口说明；构造和组合仍由你实现。现在给两张参数表起数学名字：')+
        aligned(r'E&=\texttt{self.token.weight.data}\in\mathbb R^{4\times6}',r'P&=\texttt{self.position.weight.data}\in\mathbb R^{5\times6}')+
        p(m('E')+' 的四行分别属于四个字符；'+m('P')+' 的五行分别属于五个位置。它们由构造器创建、初始化，训练时再修改数值。它们不是本次文本计算出的临时结果。')+
        fold('实现准备 · 复用上一题的初始化接口',initialization))

    body+=h('前向 ① · 字符向量加位置向量，得到初始表示',
        p('第一段输入是“猫喜欢”。字符编号来自 <code>ids</code>；位置编号由当前长度 '+m('T=3')+' 生成，<code>np.arange(T)</code> 得到 '+m('[0,1,2]')+'。因此三个位置分别查表并相加：')+
        aligned(r'\text{位置 }0&:E_{2,:}+P_{0,:}',r'\text{位置 }1&:E_{0,:}+P_{1,:}',r'\text{位置 }2&:E_{1,:}+P_{2,:}')+
        p(m('E_{2,:}')+' 是“猫”的向量，'+m('P_{0,:}')+' 是“处在第零个位置”的向量。两段输入一起计算：')+
        math(r'X_0=\begin{bmatrix}E_{2,:}+P_{0,:}&E_{0,:}+P_{1,:}&E_{1,:}+P_{2,:}\\E_{3,:}+P_{0,:}&E_{2,:}+P_{1,:}&E_{0,:}+P_{2,:}\end{bmatrix}')+
        p('上式每一项都是六维向量，因此 '+m('X_0:(2,3,6)')+'。'+m('X_0')+' 是相加后算出的初始表示，下标零表示还没有经过第一个 <code>Block</code>。')+
        p('位置查表结果为 '+m('(T,C)')+'；加 <code>[None, :, :]</code> 变为 '+m('(1,T,C)')+'，再广播到 '+m('(B,T,C)')+'。两段文本的相同位置共用一行位置参数，各段的字符内容仍各自不同。'))

    body+=h('前向 ② · 依次经过已有 Block，再做最终归一化',
        p('先把 '+m('X_0')+' 交给第一个 <code>Block</code>，再把返回值交给第二个：')+
        aligned(r'X_1&=\operatorname{Block}_0(X_0)',r'X_2&=\operatorname{Block}_1(X_1)')+
        p(m('X_0,X_1,X_2')+' 是同一批位置在不同处理阶段的表示，形状始终为 '+m('(2,3,6)')+'。两个 <code>Block</code> 结构相同，各自有独立参数；后一个继续加工前一个的输出。因果规则使每个位置只结合自己及前面位置的信息。')+
        p('接着调用最终归一化，把结果命名为 '+m('Z')+'：')+
        math(r'Z=\operatorname{RMSNorm}(X_2),\qquad Z:(2,3,6)')+
        p('代码用 <code>self._h</code> 缓存这个 '+m('Z')+'，供反向使用。这里统一用 '+m('Z')+'，避免把最终表示与注意力头数 '+m('H')+' 混淆。'))

    body+=h('前向 ③ · 与同一词表的每一行打分',
        p('每个位置现在有一个六维向量。把它与 '+m('E')+' 的四行分别做点积，就得到四个候选字符的分数：')+
        aligned(r'S&=ZE^{\mathsf T}',r'S_{b,t,v}&=\sum_{c=0}^{5}Z_{b,t,c}E_{v,c}')+
        p('下标 '+m('b')+' 表示哪段文本，'+m('t')+' 表示其中哪个位置，'+m('v')+' 表示哪个候选字符，'+m('c')+' 表示哪个特征。整体形状变化为：')+
        math(r'(2,3,6)\ @\ (6,4)\longrightarrow(2,3,4)')+
        p('例如 '+m('S_{0,2,:}')+' 是第一段文本最后一个位置的四个分数：看过“猫喜欢”后，给“喜、欢、猫、鱼”作为下一字符各打多少分。它们还不是概率，训练时交给 <code>CrossEntropy</code>。')+
        p('输入查表与输出打分使用同一份 '+m('E')+'，也就是同一个 <code>token.weight</code> 参数。这就是权重共享。'))

    reuse=p('已有 <code>Embedding</code>、<code>Block</code>、<code>RMSNorm</code> 的反向沿用个人实现。本题只把返回值按前向的相反顺序传递。<code>CrossEntropy.backward</code> 提供输出分数的上游 <code>g</code>，记作 '+m(r'G=\nabla S')+'，形状为 '+m('(B,T,V)')+'。')
    reuse+=p('输出头对应前向为 '+m(r'S=ZE^{\mathsf T}')+'。本层用 '+m('G')+' 与缓存的 '+m('Z')+' 计算输出头贡献，再调用已有组件：')
    reuse+=aligned(r'G_2&=\operatorname{reshape}(G,(BT,V)),\quad Z_2=\operatorname{reshape}(Z,(BT,C))',r'\nabla E_{\mathrm{head}}&=G_2^{\mathsf T}Z_2,\qquad\nabla Z=GE')
    reuse+=p('下标 '+m('2')+' 表示合并批次与位置后得到的二维视图。先把输出头贡献累加到 <code>token.weight.grad</code>；将 '+m(r'\nabla Z')+' 交给 <code>norm.backward</code>，再依次调用 <code>reversed(blocks)</code> 中各层的 <code>backward</code>，得到 '+m(r'\nabla X_0')+'。')
    reuse+=p('输入处对应前向为 '+m(r'(X_0)_{b,t,:}=E_{\mathrm{ids}_{b,t},:}+P_{t,:}')+'。调用 <code>position.backward(dx.sum(axis=0))</code> 归约共享位置的批次贡献，再调用 <code>token.backward(dx)</code> 按编号回填。字符查表层会处理重复编号累加，保留此前输出头贡献。')
    reuse+=aligned(r'(\nabla E_{\mathrm{lookup}})_{v,c}&=\sum_{(b,t):\,\mathrm{ids}_{b,t}=v}(\nabla X_0)_{b,t,c}',r'\nabla E&=\nabla E_{\mathrm{head}}+\nabla E_{\mathrm{lookup}}')
    reuse+=p('整数编号不求导，<code>GPT.backward</code> 返回 <code>None</code>；参数梯度保存在各自的 <code>grad</code> 中。')
    body+=h('实现准备 · 反向复用已有组件',fold('展开调用顺序与 GPT 外层的接线动作',reuse))
    body+=h('复用来源 · 本次只接成完整模型','<div data-gpt-prerequisites></div>'+p('查表、上下文处理与归一化使用当前个人模块；构造、前向和整网反向由你完成。'))

    def node(label,formula,note='',cls=''):
        return '<div class="gpt-map-node '+cls+'"><strong>'+label+'</strong><div class="gpt-map-formula">'+m(formula)+'</div>'+(p(note) if note else '')+'</div>'
    arrow='<div class="gpt-map-arrow" aria-hidden="true">↓</div>'
    flow='<figure class="gpt-forward-map" aria-label="完整 GPT 前向流程图">'
    flow+='<p class="gpt-map-input">输入：上一题提供 '+m(r'\mathrm{ids}:(B,T)')+'；由长度 '+m('T')+' 生成位置编号。</p>'
    flow+='<div class="gpt-map-sources">'+node('按字符编号查同一词表',r'E[\mathrm{ids}]:(B,T,C)','来自 <code>self.token</code>','gpt-map-shared')+node('由长度生成位置，再查位置表',r'P[0:T]:(T,C)','来自 <code>self.position</code>；向批次广播')+'</div>'+arrow
    flow+=node('相加 · 内容与位置合为初始表示',r'X_0=E[\mathrm{ids}]+P[0:T]')+arrow
    flow+=node('逐层更新 · 独立 Block 的输出接给下一层',r'X_0\xrightarrow{\operatorname{Block}_0}X_1\longrightarrow\cdots\xrightarrow{\operatorname{Block}_{L-1}}X_L','全程形状 '+m('(B,T,C)'))+arrow
    flow+=node('最终归一化 · 得到用于打分的表示',r'Z=\operatorname{RMSNorm}(X_L)','缓存为 <code>self._h</code>')+arrow
    flow+=node('共享输出头 · 再使用输入侧那一份词表',r'S=ZE^{\mathsf T},\qquad S:(B,T,V)','这里的 '+m('E')+' 与上方查表共用一个参数。','gpt-map-shared')+arrow
    flow+='<figcaption>训练：分数与目标交给 <code>CrossEntropy</code>。生成：取末位置分数，选择下一编号，再用原词表解码。</figcaption></figure>'
    body+=h('核心总结 · 把整个前向重新连成一条线',
        p('<strong>把“有哪些字符、在什么位置”变成“结合前文后，下一个字符可能是什么”。</strong> 先看完整数据流，再对照下面四行公式。')+flow+
        aligned(r'X_0&=E[\mathrm{ids}]+P[0:T]',r'X_{\ell+1}&=\operatorname{Block}_{\ell}(X_\ell),\quad \ell=0,\ldots,L-1',r'Z&=\operatorname{RMSNorm}(X_L)',r'S&=ZE^{\mathsf T}')+
        p('两张表提供内容与位置；各 <code>Block</code> 更新上下文表示；最终归一化后，与同一词表的全部行打分。以本例第一段最后一个位置为例，整条流程最终给出 '+m('S_{0,2,:}')+'：看过“猫喜欢”以后，对四个下一字符候选的分数。'))
    # 同级内容按阅读职责分组；具体步骤折叠，整体总结始终可见。
    sections=[v.split('</h3>',1) for v in body.split('<h3>')[1:]]
    return (h(*sections[0])+
        h('准备与来源 · 先认清输入，再认清模型对象',
          fold('01 输入与配置：哪些来自数据，哪些在构造时选定',sections[1][1],True)+
          fold('02 模型对象：两张表和已有子层从哪里来',sections[2][1],True))+
        h('前向主线 · 沿同一批输入走完三个步骤',
          ''.join(fold(title.replace('前向 ','',1),content,True) for title,content in sections[3:6]))+
        h('实现准备 · 复用与反向调用',sections[6][1]+sections[7][1])+
        h(*sections[8]))
