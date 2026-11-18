"""完整 Linear：局部导数、链式法则与矩阵实现逐项对应。"""


def apply_linear_layer_lesson(book, math, aligned):
    layer = book['lessons']['layer_linear']
    m = lambda tex: math(tex, inline=True)
    p = lambda text: '<p>' + text + '</p>'
    h = lambda title, text: '<h3>' + title + '</h3>' + text
    layer['body'] = h('1. 这次把计算放进一个有状态的对象',
        p('上一题 <code>solve(X, W, b)</code> 每次接收输入和参数，返回输出。这次 <code>Linear</code> 自己保存参数和前向缓存：<code>forward(x)</code> 计算输出，<code>backward(g)</code> 接收后面传来的梯度，返回输入梯度，并将参数梯度存进对象。')
        + p('给定构造器 <code>Linear(C, D, use_bias=True)</code> 在创建对象时执行一次：调用你实现的 <code>initialize((C, D))</code> 创建权重，再包装为 <code>self.W</code>；启用偏置时创建全零的 <code>self.b</code>，否则它为 <code>None</code>。两者都是 <code>Parameter</code>，数值保存在 <code>data</code>，梯度累计在 <code>grad</code>。')
        + aligned(r'W&\in\mathbb R^{C\times D},\qquad b\in\mathbb R^D',r'\operatorname{shape}(W.\mathrm{grad})&=(C,D),\qquad\operatorname{shape}(b.\mathrm{grad})=(D,)')
        + p(m('C')+' 是输入特征数，'+m('D')+' 是输出特征数。之后多次调用前向都使用这组当前参数。这里补写前向和反向；后续独立 Linear 题会让你补写构造器。'))
    layer['body'] += h('2. 前向：沿用上一题，并保存反向要用的数据',
        p('输入 '+m(r'X\in\mathbb R^{B\times T\times C}')+'；'+m('B')+' 是本批序列数，'+m('T')+' 是每条序列的位置数。把全部位置展平成二维表，每行对应一个 token：')
        + aligned(r'N&=BT,\qquad n=qT+t',r'X_2&=\operatorname{reshape}(X,(N,C))',r'Y_2&=X_2W+b',r'Y&=\operatorname{reshape}(Y_2,(B,T,D))')
        + p(m('q')+' 是序列下标，'+m('t')+' 是 token 下标，'+m('n')+' 是展平行号。本页仍用 '+m(r'B=2,\ T=3,\ C=4,\ D=5')+' 的完整样例：六个位置全部参与计算。')
        + p('前向保存两样东西：<code>self._shape</code> 保存原始输入形状，供输入梯度恢复；<code>self._x2</code> 保存二维输入的独立副本，供权重梯度计算。<code>reshape</code> 的结果可能与原数组共享数据，用 <code>copy()</code> 才能保留本次前向的输入。')
        + p('这些名称以 <code>_</code> 开头，按项目约定作为缓存，参数收集会跳过它们。关闭偏置时，前向计算 '+m(r'Y_2=X_2W')+'。'))
    layer['body'] += h('3. 反向先接收什么',
        p('损失 '+m('L')+' 是整个网络最终得到的标量。后面的网络已经算出每个输出对损失的影响，并把这个数组传给 <code>backward(g)</code>：')
        + aligned(r'G&=\frac{\partial L}{\partial Y},\qquad \operatorname{shape}(G)=(B,T,D)',r'G_2&=\operatorname{reshape}(G,(N,D))')
        + p('展平顺序与前向一致，因此 '+m('X_2')+' 和 '+m('G_2')+' 的同一行始终对应同一个 token。下面的 '+m(r'\mathrm dX,\mathrm dW,\mathrm db')+' 分别表示损失对输入、权重和偏置的梯度。')
        + p('先固定一行 '+m('n')+' 与输出特征 '+m('j')+'，把前向乘法展开；'+m('i')+' 遍历输入特征：')
        + math(r'Y_{2,nj}=\sum_{i=0}^{C-1}X_{2,ni}W_{ij}+b_j'))
    layer['body'] += h('4. 输入梯度：同一位置的所有输出，沿连接传回来',
        p('一个输入特征 '+m('X_{2,ni}')+' 参与这一行的每个输出。固定某个输出 '+m('j')+'，它对这个输入的局部导数就是连接权重 '+m('W_{ij}')+'；再接上该输出的上游梯度，对输出特征求和：')
        + aligned(r'\frac{\partial Y_{2,nj}}{\partial X_{2,ni}}&=W_{ij}',r'\mathrm dX_{2,ni}&=\sum_{j=0}^{D-1}\frac{\partial L}{\partial Y_{2,nj}}\frac{\partial Y_{2,nj}}{\partial X_{2,ni}}=\sum_{j=0}^{D-1}G_{2,nj}W_{ij}')
        + p('结果的行仍是位置 '+m('n')+'，列变成输入特征 '+m('i')+'。把刚才的求和写成矩阵乘法：')
        + math(r'\underbrace{\mathrm dX_2}_{N\times C}=\underbrace{G_2}_{N\times D}\underbrace{W^{\mathsf T}}_{D\times C}')
        + p('转置让原来权重第 '+m('i')+' 行成为第 '+m('i')+' 列，恰好与上游的一行沿输出特征乘加。得到二维输入梯度后，用缓存的 <code>self._shape</code> 恢复全部原始轴，作为 <code>backward</code> 的返回值。'))
    layer['body'] += h('5. 权重梯度：所有位置对同一个参数的贡献相加',
        p('一个 '+m('W_{ij}')+' 在所有序列、所有 token 中共用。固定一行 '+m('n')+'，对应输出对这个参数的导数是该行的输入特征；将每行接上上游后的贡献相加：')
        + aligned(r'\frac{\partial Y_{2,nj}}{\partial W_{ij}}&=X_{2,ni}',r'\mathrm dW_{ij}&=\sum_{n=0}^{N-1}\frac{\partial L}{\partial Y_{2,nj}}\frac{\partial Y_{2,nj}}{\partial W_{ij}}=\sum_{n=0}^{N-1}G_{2,nj}X_{2,ni}')
        + p('这次消去的是位置轴，留下输入和输出特征两轴。输入使用前向缓存：')
        + math(r'\underbrace{\mathrm dW}_{C\times D}=\underbrace{X_2^{\mathsf T}}_{C\times N}\underbrace{G_2}_{N\times D}')
        + p('转置后的输入第 '+m('i')+' 行，列出这个特征在所有位置的值；上游第 '+m('j')+' 列，列出对应输出在所有位置的梯度。两者乘加，正好收齐这个参数的全部贡献。将结果累加到 <code>self.W.grad</code>。'))
    layer['body'] += h('6. 偏置梯度：沿位置轴直接求和',
        p('同一个 '+m('b_j')+' 加到了所有位置的第 '+m('j')+' 个输出上。加法对偏置的局部导数为一，因此每处贡献就是那里的上游梯度：')
        + aligned(r'\frac{\partial Y_{2,nj}}{\partial b_j}&=1',r'\mathrm db_j&=\sum_{n=0}^{N-1}\frac{\partial L}{\partial Y_{2,nj}}\frac{\partial Y_{2,nj}}{\partial b_j}=\sum_{n=0}^{N-1}G_{2,nj}',r'\operatorname{shape}(\mathrm db)&=(D,)')
        + p('<code>G2.sum(axis=0)</code> 消去位置轴，保留输出特征轴。把它累加到 <code>self.b.grad</code>。当 <code>self.b is None</code> 时，参数只有权重，反向计算输入梯度和权重梯度两条路径。'))
    layer['body'] += h('7. 最后对齐返回值与对象状态',
        '<div class="contract-table"><table><thead><tr><th>动作</th><th>返回什么</th><th>对象里保存什么</th></tr></thead><tbody>'
        '<tr><td><code>forward(x)</code></td><td>输出 '+m('Y')+'</td><td>本次输入形状与二维输入副本。</td></tr>'
        '<tr><td><code>backward(g)</code></td><td>与输入同形的 '+m(r'\mathrm dX')+'</td><td>累加 <code>W.grad</code> 和启用时的 <code>b.grad</code>。</td></tr>'
        '<tr><td><code>zero_grad()</code></td><td><code>None</code></td><td>继承自 Module，将参数梯度原地清零。</td></tr></tbody></table></div>'
        + p('参数梯度留给后续优化器读取；<code>backward</code> 使用当前权重计算输入梯度，参数数值保持原值。本层按路径求和，上游已经包含损失层的平均系数。')
        + p('开始新一轮梯度统计前清零；同一轮内的多次反向把贡献相加。反向使用最近一次前向的缓存，再次前向会替换它。输入与上游数组保持原值。')
        + p('二维或更多维输入沿用同一规则：展平全部前置轴，保留末尾特征轴；返回时用缓存还原完整形状。'))
    layer['derivation'] = ''
    return book
