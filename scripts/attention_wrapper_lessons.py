"""完整注意力外层与Block：先建立直观认识，再落到接口和梯度来源。"""


def apply_wrapper_explanations(book, math, aligned):
    m = lambda tex: math(tex, inline=True)
    p = lambda text: '<p>' + text + '</p>'
    h = lambda title, text: '<h3>' + title + '</h3>' + text
    ul = lambda items: '<ul>' + ''.join('<li>' + item + '</li>' for item in items) + '</ul>'
    mha, block = (book['lessons'][key] for key in ('t_mha', 't_block'))

    forward = aligned(
        r'Z&=\operatorname{qkv}(X)\in\mathbb R^{B\times T\times3C}',
        r'(Q_0,K_0,V_0)&=\operatorname{split}(Z),\quad Q_0,K_0,V_0\in\mathbb R^{B\times T\times C}',
        r'(Q,K,V)&=\operatorname{splitHeads}(Q_0,K_0,V_0),\quad Q,K,V\in\mathbb R^{B\times H\times T\times d}',
        r'O&=\operatorname{core}(Q,K,V),\quad M=\operatorname{merge}(O),\quad Y=\operatorname{proj}(M)')
    reverse = aligned(
        r'\nabla M&=\operatorname{proj.backward}(G)',
        r'\nabla O&=\operatorname{splitHeads}(\nabla M)',
        r'(\nabla Q,\nabla K,\nabla V)&=\operatorname{core.backward}(\nabla O)',
        r'\nabla Z&=\operatorname{concat}(\operatorname{merge}(\nabla Q),\operatorname{merge}(\nabla K),\operatorname{merge}(\nabla V))',
        r'\nabla X&=\operatorname{qkv.backward}(\nabla Z)')
    body = h('1. 上一节的核心，现在要装成怎样的一层',
        p('上一节的 <code>AttentionCore</code> 已经会做一件完整的事：拿到提问、索引、内容三组向量，算出每个位置读回的信息。但调用它之前，仍要有人准备这三组向量。') +
        p('这一节的 <code>MultiHeadAttention</code> 直接接收 token 特征 '+m('X')+'：自己生成三组向量，让多个头各自读取上下文，再把结果整理成后续层可接收的特征。你要复用已有核心和 <code>Linear</code>，把前后这两端接起来。') +
        aligned(r'X\xrightarrow{\text{可训练投影}}(Q_0,K_0,V_0)\xrightarrow{\text{拆头与读取}}O\xrightarrow{\text{合头与输出投影}}Y') +
        p(m('X,Y:(B,T,C)')+'。'+m('B')+' 是批次数，'+m('T')+' 是每条序列的位置数，'+m('C')+' 是每个 token 的特征数。位置数和特征数保持，输出中每个位置的内容已经融合了允许读取的上下文。'))
    body += h('为什么同一个位置需要多种读取方式',
        p('一个头对当前 token 会算出一份读取比例，再按照这份比例组合各位置的内容。这个头输出向量的所有特征，都共用这份位置比例。理解一个 token 时，我们可能同时需要关注不同的信息。') +
        '<blockquote>小明拿着一本红色的书</blockquote>' +
        p('处理“书”时，模型可能需要关注“红色”提供的描述，也需要关注“小明”提供的人物信息。多头允许它同时产生几份不同的读取比例：') +
        ul(['一个头可以更关注描述信息。', '另一个头可以更关注人物或其他上下文关系。']) +
        p('这些只是帮助理解的例子。头的职责由训练形成，没有预先分工，也不保证各头一定学到不同用途。各头读回的信息先分别保留，再由输出投影学习怎样组合。') +
        aligned(r'\text{同一份输入}\longrightarrow\text{多种读取方式}\longrightarrow\text{组合成新的表示}') +
        p('<strong>拆头的目的，是让同一个 token 同时拥有多套关注上下文的方式。</strong>前面的可训练投影负责产生不同表示，拆头把它们组织成独立的读取过程，合头与输出投影负责整合结果。') +
        p('各头拥有不同的投影参数，但它们一起完成预测、通过同一个损失训练。每个头都能读取完整的允许上下文；训练时，预测损失的梯度沿合头与投影返回各头，一起调整这些参数。'))
    body += h('2. 多个头分别看什么：先投影，再分特征',
        p('沿用一个例子：'+m('B=2,T=3,C=4,H=2')+'。每条序列有三个 token，每个 token 四个特征；'+m('H')+' 是头数，每头的宽度为 '+m('d=C/H=2')+'。每个头仍处理三个 token，只是每个 token 在这个头里用两个特征表示。') +
        p('先用可训练矩阵变换原始四个特征，再把变换后的结果分给两个头。因此每个头的特征都可以由原输入的全部特征共同产生。不同头对应不同的投影列，训练可以让它们形成不同的匹配和读取方式。') +
        p('以一个 token 投影后的提问向量为例，下面的字母只是区分位置：') +
        aligned(r'Q_0[b,t,:]&=\begin{bmatrix}q_0&q_1&q_2&q_3\end{bmatrix}',
                r'Q[b,0,t,:]&=\begin{bmatrix}q_0&q_1\end{bmatrix},\qquad Q[b,1,t,:]=\begin{bmatrix}q_2&q_3\end{bmatrix}') +
        p('拆头没有新增参数，也没有丢掉特征；它给投影后的特征划分归属。每个头随后有自己的一张 query/key 匹配表。头的具体用途由训练形成，不预先指定某个头专门识别人名或语法。'))
    body += h('3. 三组向量为什么能用一个投影生成',
        p('<code>qkv = Linear(C, 3*C)</code> 将一个 token 的 '+m('C')+' 个特征变成 '+m('3C')+' 个。最后一轴按提问、索引、内容顺序分成三段，每段都是 '+m('C')+' 个特征：') +
        aligned(r'Z&=XW_{\mathrm{qkv}}+b_{\mathrm{qkv}},\quad W_{\mathrm{qkv}}:(C,3C),\quad b_{\mathrm{qkv}}:(3C,)',
                r'Z[b,t,:]&=\begin{bmatrix}Q_0[b,t,:]&K_0[b,t,:]&V_0[b,t,:]\end{bmatrix}') +
        p('这等价于三个独立的仿射投影合并计算：权重矩阵的三段输出列各自学习，偏置也分成三段。使用 <code>np.split(z, 3, axis=-1)</code> 取出三组，保持这个顺序。'))
    body += h('4. 拆头：把特征归属变成核心需要的轴',
        p('上一节核心的输入轴顺序是批次、头、token、头内特征。因此每组向量需要两步整理：') +
        aligned(r'(B,T,C)&\xrightarrow{\operatorname{reshape}}(B,T,H,d)\xrightarrow{\operatorname{transpose}(0,2,1,3)}(B,H,T,d)',
                r'(2,3,4)&\longrightarrow(2,3,2,2)\longrightarrow(2,2,3,2)') +
        p('第一步把最后四个特征分成两组，每组两个；第二步把头轴挪到 token 轴前。<code>reshape</code> 拆分特征轴，<code>transpose</code> 交换轴的位置，两步共同保证每个特征仍属于原来的 token。对三组向量做相同整理后，一次调用持久的 <code>core(q, k, v)</code>，得到 '+m('O:(B,H,T,d)')+'。') +
        p('核心在各批次、各头内分别打分、屏蔽未来、计算概率并读取内容。它已经支持这两个前置轴，所以不用另写循环逐个调用；一次批量调用也让它保存本次完整缓存。'))
    body += h('5. 合头和输出投影各自做什么',
        p('现在每个 token 有两个头各自读回的两个特征。合头要把同一个 token 的这两段并排放回一行：') +
        aligned(r'O[b,0,t,:]&=\begin{bmatrix}a_0&a_1\end{bmatrix},\qquad O[b,1,t,:]=\begin{bmatrix}b_0&b_1\end{bmatrix}',
                r'M[b,t,:]&=\begin{bmatrix}a_0&a_1&b_0&b_1\end{bmatrix}',
                r'(B,H,T,d)&\xrightarrow{\operatorname{transpose}(0,2,1,3)}(B,T,H,d)\xrightarrow{\operatorname{reshape}}(B,T,C)') +
        p('必须先把同一 token 的各头放相邻，再合并最后两轴。直接从头在前的布局变形，会把不同 token 的片段放错位置。') +
        p('拼接只负责摆好结果。输出投影 <code>proj = Linear(C, C)</code> 再学习怎样组合各头提供的信息：') +
        math(r'Y=MW_O+b_O,\qquad W_O:(C,C),\quad b_O:(C,)') +
        p('此时一个输出特征可以同时使用两个头的内容。整个多头层由此回到 '+m('(B,T,C)')+'，可以交给下一层。'))
    body += h('6. 构造一次，前后向使用同一批对象',
        ul(['<code>qkv</code>：生成三组投影，保存自己的输入缓存。',
            '<code>core</code>：读取上下文，保存三组输入、概率等缓存。',
            '<code>proj</code>：混合各头结果，保存合头后的输入缓存。']) +
        p('三个对象在 <code>__init__</code> 中创建，之后每次前向复用。每层反向需要找到刚才前向的缓存，投影参数也要在训练中持续保留。外层保存原输入形状 <code>self._shape</code>，供反向恢复轴。') +
        p('两层 <code>Linear</code> 共管理四个参数张量：两份权重、两份偏置；核心没有可训练参数。你在外层连接返回值，已有子层负责各自的参数梯度累计。'))
    body += h('7. 反向先过输出投影，再把合头布局倒回去',
        p('设最终标量损失为 '+m('L')+'，输出收到 '+m(r'G=\nabla Y=\partial L/\partial Y')+'；'+m(r'\nabla X')+' 表示损失对 '+m('X')+' 的梯度，与 '+m('X')+' 同形。先看对应前向：') +
        aligned(r'O\xrightarrow{\operatorname{merge}}M,\qquad Y&=MW_O+b_O',
                r'\nabla M&=GW_O^{\mathsf T}=\operatorname{proj.backward}(G)') +
        p('<code>proj.backward(g)</code> 返回合头结果的梯度，并累计输出投影的参数梯度。合头只重新摆过位置，所以反向把每个位置的梯度摆回同一个来源格；它不引入需要相乘的数值系数：') +
        aligned(r'\text{前向：}\quad M_{b,t,hd+c}&=O_{b,h,t,c}',
                r'\text{反向：}\quad(\nabla O)_{b,h,t,c}&=(\nabla M)_{b,t,hd+c}',
                r'(B,T,C)&\longrightarrow(B,T,H,d)\longrightarrow(B,H,T,d)') +
        p('将恢复好的 '+m(r'\nabla O')+' 交给 <code>core.backward</code>，按上一节已实现的读取、概率、打分链得到三组输入梯度。'))
    body += h('8. 三组梯度为什么要拼接，最后又怎样回到同一输入',
        p('对应前向是先将投影结果分成三段，再各自拆头并交给核心：') +
        aligned(r'Z&\xrightarrow{\operatorname{split}}(Q_0,K_0,V_0)\xrightarrow{\operatorname{splitHeads}}(Q,K,V)',
                r'O&=\operatorname{core}(Q,K,V)',
                r'(\nabla Q,\nabla K,\nabla V)&=\operatorname{core.backward}(\nabla O)') +
        p('先将每组梯度从头在前的布局合回 '+m('(B,T,C)')+'，得到 '+m(r'\nabla Q_0,\nabla K_0,\nabla V_0')+'。前向的三组占据 '+m('Z')+' 的不同列，因此反向也放回各自列，按原顺序拼接：') +
        math(r'\nabla Z=\begin{bmatrix}\nabla Q_0&\nabla K_0&\nabla V_0\end{bmatrix}\quad:(B,T,3C)') +
        p('三组投影都来自同一个输入，最后由投影层收齐三条用途。对应前向与输入梯度为：') +
        aligned(r'Z&=XW_{\mathrm{qkv}}+b_{\mathrm{qkv}},\qquad W_{\mathrm{qkv}}=\begin{bmatrix}W_Q&W_K&W_V\end{bmatrix}',
                r'\nabla X&=\nabla ZW_{\mathrm{qkv}}^{\mathsf T}',
                r'&=\nabla Q_0W_Q^{\mathsf T}+\nabla K_0W_K^{\mathsf T}+\nabla V_0W_V^{\mathsf T}') +
        p('调用一次 <code>qkv.backward</code> 就会完成这次合流，并累计输入投影的参数梯度。这里三组投影梯度先拼成不同输出列，回到共享输入时才相加；两种动作发生在不同对象上。'))
    mha['body'] = body
    mha['learning_page']['sample_note'] = p('贯穿例子使用 '+m('B=2,T=3,C=4,H=2,d=2')+'：每组投影 '+m('(2,3,4)')+' 拆成 '+m('(2,2,3,2)')+'，核心同形输出，再合回 '+m('(2,3,4)')+'。反向沿相同位置关系返回。图示符号用于区分特征归属；下方公开样例按各自输入完整评测。')
    _replace_tail(mha, h, ul, [
        '在 <code>__init__</code> 保存宽度与头数；创建持久 <code>qkv</code>、<code>core</code>、<code>proj</code>，要求 '+m('C,H>0')+' 且 '+m(r'C\bmod H=0')+'。',
        '前向保存原形状；投影后按提问、索引、内容分割；每组先分特征轴再交换头/token轴。',
        '调用一次完整核心，先交换轴再合头，最后调用输出投影。',
        '反向先过输出投影，恢复头布局，再调用核心；三组梯度各自合头并按原顺序拼接，交给输入投影。',
        '返回同输入形状的梯度；参数累计和清零复用已有子层，保持输入及上游原值。'], forward + reverse)

    bf = aligned(r'F_1(X)&=\operatorname{attn}(\operatorname{norm1}(X)),\qquad X_1=X+F_1(X)',
                 r'F_2(X_1)&=\operatorname{MLPBranch}(X_1)=\operatorname{mlp}(\operatorname{norm2}(X_1)),\qquad Y=X_1+F_2(X_1)')
    bb = aligned(r'G&=\nabla Y',
                 r'G_1=\nabla X_1&=G+\operatorname{MLPBranch.backward}(G)',
                 r'\nabla X&=G_1+\operatorname{norm1.backward}(\operatorname{attn.backward}(G_1))')
    body = h('1. 把读取上下文和加工特征组成一个完整单元',
        p('刚完成的 <code>MultiHeadAttention</code> 让每个 token 从允许的位置读回上下文。第四部分的 <code>MLPBranch</code> 则先归一化，再用 <code>MLP</code> 加工每个 token 自己的整组特征。现在把两种能力按顺序连接，形成后续 GPT 会反复堆叠的一个 <code>Block</code>。') +
        p('可以把这个单元读成：先读取上下文，补充当前表示；再加工补充后的特征，再次补充表示。输入和输出都是 '+m('(B,T,C)')+'，所以一个单元的输出可以继续送入下一个单元。'))
    body += h('2. 残差加法：在原表示上加入分支给出的补充',
        p('先只看一条分支。分支 '+m('F')+' 接收当前表示，算出同形结果；直连路径保留原表示，两者逐元素相加：') +
        math(r'R=X+F(X)') +
        p('这就是残差连接。分支可以学习怎样修改当前表示，直连让原有内容仍有通路。如果分支本次输出全零，结果就等于原输入。相加不会增加特征数，也不是把两个向量拼成长向量。') +
        p('例如本章沿用 '+m('B=2,T=3,C=4')+'：某个 token 的四维表示与分支给它的四维补充相加，仍是同一个位置的四维表示。'))
    body += h('3. 第一条分支：先归一化，再读取上下文',
        aligned(r'U&=\operatorname{norm1}(X),\qquad A=\operatorname{attn}(U)',r'X_1&=X+A') +
        p('<code>norm1</code> 是已有的 <code>RMSNorm</code>，按每个 token 的特征控制数值尺度；<code>attn</code> 是刚实现的多头注意力。它们把同形输入变成同形补充。') +
        p('标题中的 Pre-Norm 表示归一化放在分支计算之前。分支读的是归一化后的 '+m('U')+'，直连保留的是原始 '+m('X')+'。所以相加时回到 '+m('X+A')+'，不能把直连也替换成归一化结果。') +
        p('直观上，归一化让分支接收尺度较稳定的输入；直连继续携带当前表示。这一节先落实这两条路径，训练稳定性的更深分析放在拓展。'))
    body += h('4. 第二条分支接收的是更新后的表示',
        aligned(r'V&=\operatorname{norm2}(X_1),\qquad D=\operatorname{mlp}(V)',r'Y&=X_1+D') +
        p('第一步得到 '+m('X_1')+' 后，再交给已有的 <code>MLPBranch</code>。它内部已经包含 <code>norm2</code> 和 <code>mlp</code>，一次调用就完成这条分支。第二次直连也保留 '+m('X_1')+'。') +
        p('因此第二条分支加工的是已加入上下文的信息，不是再次从原始 '+m('X')+' 开始。<code>MLP</code> 的内部宽度由 <code>hidden</code> 指定：') +
        math(r'C\xrightarrow{\text{第一层Linear}}\mathrm{hidden}\xrightarrow{\mathrm{GELU}}\mathrm{hidden}\xrightarrow{\text{第二层Linear}}C') +
        p('虽然内部会改变特征宽度，分支出口回到 '+m('C')+'，才能与直连相加。它逐 token 加工特征；跨 token 的读取已经由前面的注意力负责。') +
        p('把两次补充连起来看，完整前向就是：') + bf)
    body += h('5. 你需要创建和复用哪些对象',
        p('<code>Block(C, H, hidden)</code> 接收特征宽度、注意力头数和 <code>MLP</code> 隐藏宽度。构造时创建一次：') +
        ul(['<code>norm1 = RMSNorm(C)</code>：第一条分支的归一化。',
            '<code>attn = MultiHeadAttention(C, H)</code>：第一条分支的上下文读取。',
            '<code>ffn = MLPBranch(C, hidden)</code>：调用第四部分前置任务中的个人归一化与特征加工分支。']) +
        p('第二个归一化层已由分支创建，它与 <code>norm1</code> 是两个不同实例，各自学习参数、保存缓存。分支反向需要最近一次前向的缓存，所以前向和反向必须使用构造时创建的同一批对象。') +
        p('属性名可以按你的封装选择。将分支保存在公开的 <code>ffn</code> 中，就可以直接调用 <code>ffn</code> 和 <code>ffn.backward</code>；已有 <code>Module.parameters()</code> 会递归进入分支收集参数。测评按实际子层身份核对两种分支及其参数，不要求额外的 <code>norm2</code>、<code>mlp</code> 别名。') +
        p('折叠参考沿用了旧接口：用私有 <code>_ffn</code> 调用分支，并以公开别名暴露内部子层。这只是一种兼容写法。由于个人 <code>Module</code> 跳过下划线属性，若选用私有分支，须通过公开属性让参数可被找到。后续组装 GPT 时，应按你自己的对象结构访问出口权重，例如公开分支中的 <code>ffn.mlp.fc2.W</code>。'))
    body += h('6. 反向为什么是两条路径相加',
        p('先回到一条残差的原始前向。设损失为 '+m('L')+'，结果收到上游 '+m(r'G_R=\nabla R')+'：') +
        math(r'R=X+F(X)') +
        p('输入影响结果有两条用途：一条直接经过加法，一条先经过分支。加法对两侧的局部导数都是一，所以两侧各收到完整上游。直连将它原样带回；分支经过自己的反向计算，返回对同一输入的贡献。最后将两份贡献相加：') +
        math(r'\nabla X=G_R+\operatorname{F.backward}(G_R)') +
        p('这里相加的是同一输入通过不同用途产生的梯度，并非把上游平均分给两边。若分支始终输出零，其输入梯度也为零，直连仍能将完整上游送回输入。'))
    body += h('7. 先回第二次加法，得到第一阶段的上游',
        p('反向从最终输出开始。对应前向是：') +
        aligned(r'D&=\operatorname{MLPBranch}(X_1),\qquad Y=X_1+D',r'G&=\nabla Y') +
        p('直连把 '+m('G')+' 送回 '+m('X_1')+'；另一份 '+m('G')+' 送入分支的 <code>backward</code>，例如 <code>ffn.backward</code>。该方法已负责先逆 <code>MLP</code>、再逆 <code>RMSNorm</code>，返回的就是第二条分支对 '+m('X_1')+' 的梯度。合流得到：') +
        math(r'G_1=\nabla X_1=G+\operatorname{MLPBranch.backward}(G)') +
        p('此时 '+m('G_1')+' 才是第一阶段输出真正收到的完整上游。下一步的两条路径都应使用它。'))
    body += h('8. 再回第一次加法，得到整个单元的输入梯度',
        p('对应前向再次列在这里：') +
        aligned(r'U&=\operatorname{norm1}(X),\qquad A=\operatorname{attn}(U),\qquad X_1=X+A') +
        p('直连将 '+m('G_1')+' 原样送回 '+m('X')+'。分支则先过注意力反向得到归一化输出的梯度，再过归一化反向，才能回到原输入：') +
        aligned(r'\nabla U&=\operatorname{attn.backward}(G_1)',r'\nabla X&=G_1+\operatorname{norm1.backward}(\nabla U)') +
        p('最后返回 '+m(r'\nabla X:(B,T,C)')+'。已有子层在各自反向中累计参数梯度；外层负责调用顺序、传递返回值和两次直连合流。第一条分支要按注意力、归一化的逆序调用，不能跳过归一化，也不能继续使用尚未合流的 '+m('G')+'。') +
        p('现在再看整个实现任务：创建持久子层，前向完成两次同形补充，反向从第二次补充开始逆序合流。计算细节沿用已完成的个人模块。'))
    block['body'] = body
    block['extension'] = h('为什么残差常与Pre-Norm一起使用', p('直连使表示与梯度有直接通路；归一化控制分支输入尺度。它们常用于帮助深层网络训练，但不保证训练必然稳定。这里的必要实现依据是前向的两条用途及其反向合流，具体训练表现还与初始化、层数和优化设置有关。'))
    block['learning_page']['sample_note'] = p('沿用 '+m('B=2,T=3,C=4,H=2')+'，隐藏宽度可以取 '+m(r'\mathrm{hidden}=8')+'。注意力补充、第一次相加、特征加工补充和第二次相加的出口始终为 '+m('(2,3,4)')+'。某条分支输出为零时，该次加法保留直连原值；实际公开检查同时覆盖零分支和非零分支，核对完整输出、输入梯度及全部参数梯度。')
    _replace_tail(block, h, ul, [
        '构造并保留两条分支的个人子层；属性名可自选，通过公开分支或公开子层让 <code>parameters()</code> 找到全部参数。',
        '先将归一化输入交给注意力，再与原输入相加得到 '+m('X_1')+'。',
        '将 '+m('X_1')+' 交给已有 <code>MLPBranch</code>，与 '+m('X_1')+' 相加得到输出。',
        '反向先将输出上游与第二条分支返回梯度相加，得到 '+m('G_1')+'；再逆注意力和第一层归一化，与直连 '+m('G_1')+' 相加。',
        '返回输入梯度；复用子层缓存和参数累计，继承参数收集与清零，不更新权重或修改输入。'], bf + bb)
    purpose = h('整体功能：从已有特征中，读回当前需要的上下文',
        p('从模型的输入输出看，'+m('X')+' 是各个 token 进入这一层时已有的特征，'+m('Y')+' 是这一层结合上下文算出的新特征。例如在“小明拿着一本红色的书”中，“书”所在的位置可以根据当前表示，读取前面描述或人物位置提供的信息。输出仍是一组向量，继续交给后续模型计算。') +
        p('<strong>注意力在做的工作，是让每个位置根据当前输入，选择并汇集允许位置提供的信息。</strong>读取比例由当前向量之间的匹配计算出来，换一份上下文，读取结果也可能变化。') +
        aligned(r'X\xrightarrow{\text{投影与拆头}}(Q,K,V)\xrightarrow{\operatorname{AttentionCore}}O\xrightarrow{\text{合头与输出投影}}Y') +
        p('这条完整的 '+m('X')+' 到 '+m('Y')+' 流程由下一节的 <code>MultiHeadAttention</code> 负责。<code>AttentionCore.forward(q, k, v)</code> 专门负责中间的读取：接收准备好的提问、索引和内容，按匹配程度组合内容，返回每个 query 读回的向量。图中的 '+m('O')+' 是核心读回的内容；本节单独讲核心时，将它记为 '+m('Y')+'。') +
        ul(['<strong>外层准备读取方式：</strong>用可训练投影从 '+m('X')+' 产生三组向量，并整理成各头布局。',
            '<strong>核心完成读取：</strong>'+m('Q,K')+' 决定从哪里读、读多少，'+m('V')+' 提供实际取回的内容。',
            '<strong>外层整理读取结果：</strong>合并各头，再经输出投影生成完整层的输出。']) +
        p('核心本身没有可训练参数，它使用的是外层投影得到的向量。它返回读回的信息；在后面的 <code>Block</code> 中，残差连接再把完整注意力层给出的补充加入原表示。'))
    core_body = book['lessons']['t_attention']['body']
    marker = '<h3>两种“权重”：保存的计算规则与本次读取比例</h3>'
    book['lessons']['t_attention']['body'] = core_body.replace(marker, purpose + marker)
    book['lessons']['t_attention']['body'] += h('核心总结：匹配、分配比例、读取内容',
        p('<strong>注意力根据匹配程度，按比例读取内容。</strong>把前面的步骤合起来，就是：') +
        math(r'\boxed{\operatorname{Attention}(Q,K,V)=\underbrace{\operatorname{softmax}_{\text{逐行}}\!\left(\frac{QK^{\mathsf T}}{\sqrt d}+M\right)}_{\text{各位置的读取比例}}V}') +
        p(m('d')+' 是每头的匹配特征数；因果掩码 '+m('M')+' 在未来位置为 '+m(r'-\infty')+'，其余位置为 '+m('0')+'。非因果模式不屏蔽位置，即 '+m('M=0')+'。') +
        ul(['<strong>匹配：</strong>'+m('Q')+' 和 '+m('K')+' 决定当前位置与哪些位置相关。',
            '<strong>分配比例：</strong>屏蔽未来，再用 <code>Softmax</code> 得到每行的读取权重。',
            '<strong>读取：</strong>用权重组合 '+m('V')+'，得到每个 query 读回的向量。']) +
        p('<strong>'+m('Q,K')+' 决定从哪里读、读多少，'+m('V')+' 决定读回什么。</strong>核心的反向沿这同一条读取链返回三组输入梯度。'))
    mha['body'] += h('核心总结：各头分别读取，再合成一份表示',
        p('<strong>多头注意力从输入生成多组读取方式，将各头结果组合成同一份输出。</strong>整体计算可以收成两行：') +
        aligned(r'O_h&=\operatorname{Attention}(Q_h,K_h,V_h),\quad h=1,\ldots,H',
                r'Y&=\boxed{\operatorname{Concat}(O_1,\ldots,O_H)W_O+b_O}') +
        p(m('Q_h,K_h,V_h')+' 由输入 '+m('X')+' 经过可训练投影、分割和拆头得到；'+m('O_h')+' 是第 '+m('h')+' 个头读回的内容。'+m('W_O,b_O')+' 是输出投影的权重与偏置；这里的 <code>Concat</code> 表示将同一 token 的各头结果沿特征拼接。') +
        ul(['<strong>生成：</strong>投影生成提问、索引、内容，再把特征分给各头。',
            '<strong>读取：</strong>每个头都看完整序列中的允许位置，分别算出读取结果。',
            '<strong>组合：</strong>同一 token 的各头结果拼回一行，再由输出投影混合。']) +
        p('<strong>输入和输出都是 '+m('(B,T,C)')+'，中间按 '+m('H')+' 个头分别读取上下文。</strong>反向先逆输出投影，再恢复各头布局，最后将三组投影用途汇回输入。'))
    block['body'] += h('核心总结：读取后补充一次，加工后再补充一次',
        p('<strong>一个 <code>Block</code> 依次读取上下文、加工当前特征，每一步都在原表示上加入补充。</strong>整个单元可以记成：') +
        math(r'\boxed{\begin{aligned}X_1&=X+\operatorname{MultiHeadAttention}(\operatorname{RMSNorm}_1(X))\\Y&=X_1+\operatorname{MLP}(\operatorname{RMSNorm}_2(X_1))\end{aligned}}') +
        ul(['<strong>先读取：</strong>第一条分支先归一化，再让 token 读取上下文；与 '+m('X')+' 相加得到 '+m('X_1')+'。',
            '<strong>再加工：</strong>第二条分支处理已经更新的 '+m('X_1')+'，再与 '+m('X_1')+' 相加得到输出。',
            '<strong>保留直连：</strong>每次归一化都在分支内部，直连保留该阶段的原表示；每次输出保持 '+m('(B,T,C)')+'。']) +
        p('公式第二条分支就是已完成的 <code>MLPBranch</code>。<strong>读上下文、加工特征，两次残差把结果加入当前表示。</strong>反向从第二次加法开始，先收齐直连与分支贡献，再回到第一次加法。'))
    from block_overview_layout import arrange_block
    return arrange_block(book, math)


def _replace_tail(meta, h, ul, actions, formula):
    """保留原OI接口及契约，更新题末可独立使用的任务与公式速查。"""
    statement = meta['learning_page']['statement']
    meta['learning_page']['statement'] = statement.split('<h3>动手任务清单</h3>')[0] + h('动手任务清单', ul(actions)) + h('关键公式速查', formula)
