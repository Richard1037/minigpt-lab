"""Scale → Linear 前向 → 完整 Linear：保留原代码、测评和个人文件。"""


def apply_scale_linear_lessons(book, math, aligned):
    def m(tex):
        return math(tex, inline=True)

    def p(text):
        return '<p>' + text + '</p>'

    def section(title, content):
        return '<h3>' + title + '</h3>' + content

    def page(role, purpose, title, interface, description, inputs, outputs, states, sample):
        return dict(role=role, purpose=purpose, problem_title=title,
                    statement=section('题目描述', p(description))
                    + section('实现接口', p(interface))
                    + section('输入格式', inputs)
                    + section('输出格式', outputs)
                    + section('状态变化', states), sample_note=p(sample))

    scale = book['lessons']['layer_scale']
    scale['body'] = section('1. 这个层保存什么，计算什么',
        p('先做一个只有一个可训练数值的层：<code>Scale</code> 保存参数 <code>w</code>，把输入数组里的每个元素都乘以它。所有位置共用这一个参数。')
        + p('<code>Scale(value)</code> 创建对象时，给定构造器把 <code>value</code> 包装成 <code>Parameter</code>：<code>self.w.data</code> 保存数值，<code>self.w.grad</code> 保存梯度，初始为零。<code>self</code> 就是这一个层对象。')
        + p('输入 '+m(r'X\in\mathbb R^{B\times T\times C}')+' 是一批序列的特征；'+m('B')+' 是序列数，'+m('T')+' 是每条序列的位置数，'+m('C')+' 是每个位置的特征数。参数 '+m('w')+' 是标量。'))
    scale['body'] += section('2. 前向：返回结果，留下本次输入',
        math(r'Y_{b,t,c}=wX_{b,t,c},\qquad \operatorname{shape}(Y)=\operatorname{shape}(X)')
        + p('<code>forward(x)</code> 中的 <code>x</code> 对应公式里的 '+m('X')+'。计算乘积作为返回值，同时把这次输入的一份独立副本存在 <code>self._x</code> 中。这就是缓存：反向时重新使用前向的数据。')
        + p('反向计算参数梯度需要前向时的原始输入。<code>self._x = x</code> 让两者指向同一个数组；如果传入数组的那段代码随后原地修改 <code>x</code>，缓存也会改变。用 <code>x.copy()</code> 保存独立副本，就能保留这次前向的输入。')
        + p('<code>_x</code> 的下划线是本项目的缓存命名约定。我们在 <code>Module.parameters()</code> 中写了跳过这类属性的判断；这是自己实现的规则，并非 Python 自动跳过。'))
    scale['body'] += section('3. 反向：一个返回值，一处参数梯度',
        p('后面的网络已经算好损失 '+m('L')+'，传回 '+m(r'G=\partial L/\partial Y')+'。<code>backward(g)</code> 收到的 <code>g</code> 就是这个数组，与输出同形；它告诉你每个输出对最终损失的影响。')
        + p('反向需要完成两件事：把输入梯度传给前一层；把参数梯度存到当前层。记号 '+m(r'\mathrm dX')+' 和 '+m(r'\mathrm dw')+' 分别代表 '+m(r'\partial L/\partial X')+' 和 '+m(r'\partial L/\partial w')+'。')
        + p('先固定一个位置，对当前乘法求导：输入改变时，输出按参数的倍数改变。再乘上后面传来的上游梯度，就是损失对这个输入的梯度：')
        + aligned(r'\frac{\partial Y_{b,t,c}}{\partial X_{b,t,c}}&=w',
                  r'\mathrm dX_{b,t,c}=\frac{\partial L}{\partial X_{b,t,c}}&=\frac{\partial L}{\partial Y_{b,t,c}}\frac{\partial Y_{b,t,c}}{\partial X_{b,t,c}}=G_{b,t,c}w')
        + p('一个输入元素只影响同位置的输出，所以输入梯度仍与输入同形。一个 '+m('w')+' 却影响所有输出，所以它从所有位置收到的贡献要相加：')
        + p('参数改变时，该位置输出变化的倍数是前向输入。因此先求出每个位置对参数的贡献，再把这些路径相加：')
        + aligned(r'\frac{\partial Y_{b,t,c}}{\partial w}&=X_{b,t,c}',
                  r'\mathrm dw=\frac{\partial L}{\partial w}&=\sum_{b,t,c}\frac{\partial L}{\partial Y_{b,t,c}}\frac{\partial Y_{b,t,c}}{\partial w}=\sum_{b,t,c}G_{b,t,c}X_{b,t,c}')
        + p('<code>backward</code> 返回 '+m(r'\mathrm dX')+'；把 '+m(r'\mathrm dw')+' 加到 <code>self.w.grad</code>。相乘用缓存的输入，求和覆盖全部轴，得到标量。'))
    scale['body'] += section('4. 梯度保存与参数更新是两个动作',
        p('<code>grad</code> 是累计区：每次反向将本次贡献加入已有值。新一轮梯度统计开始前，调用继承的 <code>zero_grad()</code> 清零。改变参数 <code>data</code> 的工作会在后面的优化器章节实现。')
        + aligned(r'\mathrm{grad}_{\mathrm{new}}&=\mathrm{grad}_{\mathrm{old}}+\mathrm dw',
                  r'\text{一次前向}&\longrightarrow\text{反向计算并累加}\longrightarrow\text{后续由优化器更新参数}'))
    scale['derivation'] = ''  # 必需推导已贴近对应计算，避免末尾重复。
    scale['learning_page'] = page('必要的反向传播练习',
        'Scale 用一个共享参数练习缓存、输入梯度和参数梯度；下一步将同样的职责扩展为 miniGPT 的 Linear 层。',
        '实现共享标量层的前向与反向',
        '<code>Scale.forward(x)</code>、<code>Scale.backward(g)</code>；构造器 <code>Scale(value=2.0)</code> 已给出。',
        '完成 Scale 层，使其逐元素缩放输入，并在反向时计算输入梯度和共享参数梯度。',
        p('<code>value</code> 为初始实数。<code>x</code> 为实数 NumPy 数组，支持二维、三维及更多维；<code>g</code> 是与前向输出同形的实数数组。先调用前向，再调用对应的反向。'),
        p('<code>forward</code> 返回同形 NumPy 数组 '+m('Y')+'；<code>backward</code> 返回同形 NumPy 数组 '+m(r'\mathrm dX')+'。'),
        p('前向保存输入副本。反向把所有位置的贡献累加到标量 <code>w.grad</code>；<code>zero_grad()</code> 清零。输入、上游数组与参数 <code>data</code> 保留原值。'),
        '首例使用两条序列、每条三个位置、每个位置四个特征。参数为 2，上游元素全为 1；输出是输入的两倍，输入梯度元素全为 2，参数梯度来自完整输入的元素和。')

    linear = book['lessons']['linear']
    linear['body'] = section('1. Linear 把每个位置的特征换成另一组特征',
        p('上一步 Scale 用一个数缩放所有元素。Linear 使用一张权重矩阵，把每个位置的 '+m('C')+' 个输入特征组合成 '+m('D')+' 个输出特征。同一个规则用于所有序列、所有位置。')
        + aligned(r'X&\in\mathbb R^{B\times T\times C},\qquad W\in\mathbb R^{C\times D},\qquad b\in\mathbb R^D',
                  r'Y&\in\mathbb R^{B\times T\times D}')
        + p(m('B')+' 是序列数，'+m('T')+' 是每条序列的位置数；'+m('W')+' 的每一列给出一个输出特征的组合权重，'+m('b')+' 为每个输出特征提供一个偏置。序列和位置保持各自的身份。'))
    linear['body'] += section('2. 先看一个输出来自哪里',
        math(r'Y_{b,t,j}=\sum_{i=0}^{C-1}X_{b,t,i}W_{i,j}+b_j')
        + p('固定序列 '+m('b')+'、位置 '+m('t')+' 和输出特征 '+m('j')+'：取这个位置的整行输入，与权重第 '+m('j')+' 列逐项相乘并求和，再加对应偏置。求和轴是输入特征 '+m('i')+'。公式中的下标 '+m('b')+' 表示序列编号，单独的向量 '+m('b')+' 表示偏置。'))
    linear['body'] += section('3. 三维张量怎样交给二维矩阵乘法',
        p('把每个位置的一行特征排进同一张二维表。这里只合并前两个轴，行内特征及位置顺序都保留。先记录原始形状，以便计算后放回原位置。')
        + aligned(r'N&=BT,\qquad n=bT+t',r'X_{b,t,i}&\longleftrightarrow (X_2)_{n,i}',
                  r'(B,T,C)&\longrightarrow (BT,C)\longrightarrow (BT,D)\longrightarrow (B,T,D)')
        + p('例如 '+m(r'B=2,\ T=3,\ C=4,\ D=5')+'：两个 batch 的六个 token 排成六行，依次来自 '+m(r'(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)')+'；矩阵乘法把每行四个特征换为五个，最后六行回到两个 batch。')
        + p('<code>X.reshape(-1, C)</code> 让 NumPy 推算行数。偏置加到每一行的对应输出特征；返回时用保存的前置形状与输出宽度恢复。二维输入本来就是一张位置表，采用相同计算。'))
    linear['derivation'] = section('计算关系', aligned(
        r'X_2&=\operatorname{reshape}(X,(N,C))',r'Y_2&=X_2W+b',
        r'Y&=\operatorname{reshape}(Y_2,(B,T,D))')
        + p('对于更多前置轴，'+m(r'N=\prod_{a=0}^{k-1}s_a')+'，原始输入形状为 '+m(r'(s_0,\ldots,s_{k-1},C)')+'；返回时保留这些前置轴，仅将最后一轴替换为 '+m('D')+'。'))
    linear['learning_page'] = page('构建 Linear 所需的 NumPy 应用练习',
        '先写清楚 Linear 内部的前向计算。下一题把同一计算放入可缓存、可反向传播的 Linear 类。',
        '实现保留前置维度的 Linear 前向', '<code>solve(X, W, b)</code>。',
        '对每个位置应用同一权重和偏置，返回变换后的特征数组。',
        p('<code>X</code> 为至少二维的实数 NumPy 数组，末轴长度为 '+m('C')+'；<code>W</code> 为形状 '+m('(C,D)')+' 的实数数组；<code>b</code> 为形状 '+m('(D,)')+' 的实数数组。各维长度为正整数。'),
        p('返回实数 NumPy 数组；输入形状 '+m(r'(s_0,\ldots,s_{k-1},C)')+' 对应输出形状 '+m(r'(s_0,\ldots,s_{k-1},D)')+'。'),
        p('计算使用传入的参数；输入数组保持原值。'),
        '公开样例按上方数组展示。二维用例便于逐项核对乘法与偏置；三维用例检验全部序列与位置都能正确展平、变换和恢复。')

    layer = book['lessons']['layer_linear']
    layer['body'] = section('1. 把上一题的计算变成一个层对象',
        p('上一题由调用方传入权重和偏置。这一题由 <code>Linear</code> 对象持有它们，并保存反向需要的数据。这就是 miniGPT 后续投影层与 MLP 使用的基础层。')
        + p('给定构造器 <code>Linear(C, D, use_bias=True)</code> 在创建时调用你上一节实现的 <code>initialize((C, D))</code>，再用 <code>Parameter</code> 包装成 <code>self.W</code>。启用偏置时创建长度为 '+m('D')+' 的全零参数 <code>self.b</code>；关闭时 <code>self.b</code> 为 <code>None</code>。')
        + p('构造只发生在创建对象时；之后的多次 <code>forward</code> 复用同一组参数。权重数值在 <code>W.data</code>，梯度累计在 <code>W.grad</code>；偏置同理。'))
    layer['body'] += section('2. 前向与缓存：沿用展平、变换、恢复',
        p('输入 '+m(r'X\in\mathbb R^{B\times T\times C}')+'；'+m('B,T')+' 分别是序列数和位置数，'+m('C,D')+' 分别是输入和输出特征宽度。把全部位置排成 '+m(r'N=BT')+' 行，完成上一题的计算：')
        + aligned(r'X_2&=\operatorname{reshape}(X,(N,C))',r'Y_2&=X_2W+b',r'Y&=\operatorname{reshape}(Y_2,(B,T,D))')
        + p('保存两样东西：原始输入形状，用于恢复输入梯度；二维输入的独立副本，用于计算权重梯度。<code>reshape</code> 的结果可能与原数组共享存储，因此缓存时再调用 <code>copy()</code>。关闭偏置时使用 '+m(r'Y_2=X_2W')+'。'))
    layer['body'] += section('3. 收到上游后，分别计算三类梯度',
        p('后面的网络传回 '+m(r'G=\partial L/\partial Y')+'，形状与输出相同。把它按前向相同的行顺序排成 '+m(r'G_2\in\mathbb R^{N\times D}')+'。'+m(r'\mathrm dX,\mathrm dW,\mathrm db')+' 分别表示损失对输入、权重和偏置的梯度。')
        + p('<strong>输入梯度：</strong>一个输入特征参与本位置的所有输出特征，因此沿输出特征轴汇总。连接权重来自原来的 '+m('W')+'：')
        + math(r'\underbrace{\mathrm dX_2}_{N\times C}=\underbrace{G_2}_{N\times D}\underbrace{W^{\mathsf T}}_{D\times C}')
        + p('转置让每个输出的梯度沿原来的连接回到输入。最后把 '+m(r'\mathrm dX_2')+' 恢复成原始输入形状，作为 <code>backward</code> 的返回值。')
        + p('<strong>权重梯度：</strong>同一个 '+m('W_{i,j}')+' 被所有位置复用。每处贡献是“该位置输入特征”乘“该位置输出梯度”，再沿位置轴求和：')
        + math(r'\underbrace{\mathrm dW}_{C\times D}=\underbrace{X_2^{\mathsf T}}_{C\times N}\underbrace{G_2}_{N\times D}')
        + p('这里的输入来自前向缓存；矩阵乘法自动汇总所有 batch 和 token 的贡献。')
        + p('<strong>偏置梯度：</strong>每个输出特征加了同一个偏置，所以把该特征在全部位置收到的梯度加起来：')
        + math(r'\mathrm db_j=\sum_{n=0}^{N-1}G_{2,n,j},\qquad \operatorname{shape}(\mathrm db)=(D,)')
        + p('<code>G2.sum(axis=0)</code> 消去位置轴，留下输出特征轴。关闭偏置时只计算输入和权重两条路径。'))
    layer['body'] += section('4. 返回梯度，累加参数，等待下一步',
        p('将本次 '+m(r'\mathrm dW,\mathrm db')+' 分别加入 <code>W.grad</code>、<code>b.grad</code>。<code>backward</code> 的返回值只有输入梯度，前一层接着使用它；参数梯度留在当前对象中供优化器读取。')
        + p('上游已经包含损失层选择的平均系数，本层按路径求和。一次前向配合对应反向；再次前向会替换缓存。新一轮统计前调用 <code>zero_grad()</code>，同一轮内多次反向使用累加。')
        + p('支持更多维输入时，'+m('N')+' 是所有前置轴长度的乘积。恢复形状时使用缓存的完整形状，因此 batch、token 等轴全部保留。'))
    layer['derivation'] = section('逐元素定义',
        p('下面用一个位置 '+m('n')+'、输入特征 '+m('i')+' 和输出特征 '+m('j')+' 展开三条来源。'+m('L')+' 是最终标量损失。')
        + aligned(r'Y_{2,nj}&=\sum_iX_{2,ni}W_{ij}+b_j',r'G_{2,nj}&=\frac{\partial L}{\partial Y_{2,nj}}'))
    layer['derivation'] += section('输入梯度', aligned(
        r'\frac{\partial Y_{2,nj}}{\partial X_{2,ni}}&=W_{ij}',
        r'\mathrm dX_{2,ni}&=\sum_j\frac{\partial L}{\partial Y_{2,nj}}\frac{\partial Y_{2,nj}}{\partial X_{2,ni}}=\sum_jG_{2,nj}W_{ij}',
        r'\mathrm dX_2&=G_2W^{\mathsf T}')
        + p('求和遍历同位置的所有输出特征，保留位置 '+m('n')+' 和输入特征 '+m('i')+'。'))
    layer['derivation'] += section('权重与偏置', aligned(
        r'\frac{\partial Y_{2,nj}}{\partial W_{ij}}&=X_{2,ni},\qquad \frac{\partial Y_{2,nj}}{\partial b_j}=1',
        r'\mathrm dW_{ij}&=\sum_nG_{2,nj}X_{2,ni},\qquad \mathrm dW=X_2^{\mathsf T}G_2',
        r'\mathrm db_j&=\sum_nG_{2,nj}')
        + p('共享参数从所有位置收集贡献。'+m(r'\mathrm dW')+' 保留输入与输出特征两轴，'+m(r'\mathrm db')+' 只保留输出特征轴。'))
    layer['learning_page'] = page('miniGPT 组成部分',
        '你实现的 Linear 用于后续 MLP 和注意力投影；这里完成缓存、前向、输入梯度及参数梯度。后续独立 Linear 任务再补写构造器。',
        '实现 Linear 的前向与反向',
        '<code>Linear.forward(x)</code>、<code>Linear.backward(g)</code>；构造器 <code>Linear(C, D, use_bias=True)</code> 已给出。',
        '使同一 Linear 对象完成特征变换，并依据上游梯度计算输入与参数的梯度。',
        p('<code>C</code>、<code>D</code> 为正整数，<code>use_bias</code> 为布尔值。<code>x</code> 为至少二维的实数 NumPy 数组，末轴为 '+m('C')+'；<code>g</code> 为与前向输出同形的实数 NumPy 数组。先调用前向，再调用对应反向。'),
        p('<code>forward</code> 返回 '+m(r'(s_0,\ldots,s_{k-1},D)')+' 的数组；<code>backward</code> 返回与原输入 '+m(r'(s_0,\ldots,s_{k-1},C)')+' 同形的输入梯度。'),
        p('前向保存原形状及二维输入副本；反向累加 <code>W.grad</code> 和启用时的 <code>b.grad</code>。<code>use_bias=False</code> 时 <code>b</code> 为 <code>None</code>。<code>zero_grad()</code> 清零已收集参数的梯度，参数数值及传入数组保留原值。'),
        '首例的六个位置全部参与计算：输入为 (2,3,4)，输出为 (2,3,5)，二维位置表有六行。测评将权重与偏置设为样例中的确定数值；前向使用这些当前参数，反向权重梯度汇总六行，偏置梯度按列汇总六行。')
    # 用实际公开样例衔接抽象公式；保留全部 batch，仅挑一个坐标追溯来源。
    layer['learning_page']['sample_note'] = section('样例解释：跟踪同一个位置和参数',
        p('上方完整样例使用 '+m(r'(B,T,C,D)=(2,3,4,5)')+'。先看第一个 batch、第一个 token 的第一个输出；所需输入来自该 token，权重来自第一列：')
        + math(r'Y_{0,0,0}=(-1)(-0.5)+(-0.9)(0)+(-0.8)(0.5)+(-0.7)(1)+0=-0.6')
        + p('同位置第一个输入特征的梯度，沿五个输出特征收集贡献：')
        + math(r'\mathrm dX_{0,0,0}=(-1)(-0.5)+(-0.9)(-0.4)+(-0.8)(-0.3)+(-0.7)(-0.2)+(-0.6)(-0.1)=1.3')
        + p('参数 '+m('W_{0,0}')+' 在六个位置共用。下面两行的六项，依次对应两个 batch 中的全部 token：')
        + aligned(r'\mathrm dW_{0,0}&=(-1)(-1)+(-0.6)(-0.5)+(-0.2)(0)+(0.2)(0.5)+(0.6)(1)+(1)(1.5)=3.5',
                  r'\mathrm db_0&=-1-0.5+0+0.5+1+1.5=1.5')
        + p('初始参数梯度为零时，累加后相应元素就是上述结果。完整 '+m(r'\mathrm dX')+' 保留 '+m('(2,3,4)')+'，'+m(r'\mathrm dW')+' 为 '+m('(4,5)')+'，'+m(r'\mathrm db')+' 为 '+m('(5,)')+'。下方演示可切换坐标，逐项追踪来源。'))
    from linear_forward_lesson import apply_linear_forward_lesson
    apply_linear_forward_lesson(book, math, aligned)
    from linear_layer_lesson import apply_linear_layer_lesson
    return apply_linear_layer_lesson(book, math, aligned)
