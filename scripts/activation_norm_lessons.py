"""激活与归一化：教学、实现契约和持久骨架；保留个人文件。"""
def apply_activation_norm_lessons(book, math, aligned):
    def m(t): return math(t, inline=True)
    def p(t): return '<p>'+t+'</p>'
    def h(t,s): return '<h3>'+t+'</h3>'+s
    g=book['lessons']['layer_gelu'];r=book['lessons']['layer_rmsnorm']
    g['body']=h('1. GELU 在做什么',p('Linear 把输入特征混合成新特征，GELU 接着对每个元素独立做一次平滑的非线性变换。较大的正值接近原样保留，较小的负值趋近零。它在 miniGPT 的 MLP 中连接两个 Linear。')+p('输入 '+m(r'X\in\mathbb R^{B\times T\times C}')+'，输出 '+m(r'Y')+' 与它同形。'+m(r'B')+' 是序列数，'+m(r'T')+' 是每条序列的位置数，'+m(r'C')+' 是特征数。本层没有可训练参数，沿用 Module 的空参数列表和清零方法。'))
    g['body']+=h('2. 前向：同一个函数应用到全部元素',p('先固定任意一个元素 '+m(r'x')+'。'+m(r'a,k')+' 是常数，'+m(r'u,t')+' 是为这次计算起名的中间量：')+aligned(r'a&=0.044715,\qquad k=\sqrt{2/\pi}',r'u&=k(x+ax^3)',r't&=\tanh(u)',r'y&=\tfrac12x(1+t)')+p('对数组执行平方、乘法、<code>np.tanh</code> 都是逐元素操作；中间数组 '+m(r'U,T_{\tanh}')+' 与输入同形。这里用 '+m(r't')+' 表示 tanh 的结果，序列长度仍记为大写 '+m(r'T')+'。实现时缓存输入副本与 tanh 结果，用于对应的反向。'))
    g['body']+=h('3. 反向：输出对输入有两条依赖',p('后续网络传回 '+m(r'G=\partial L/\partial Y')+'。看 '+m(r'y=\tfrac12x(1+t)')+'：输入既直接出现在乘法里，也通过 '+m(r'x\to u\to t')+' 改变乘法另一项。因此乘积法则有两项：')+aligned(r'\frac{\mathrm du}{\mathrm dx}&=k(1+3ax^2),\qquad\frac{\mathrm dt}{\mathrm du}=1-t^2',r'\frac{\mathrm dy}{\mathrm dx}&=\underbrace{\tfrac12(1+t)}_{\text{直接路径}}+\underbrace{\tfrac12x(1-t^2)k(1+3ax^2)}_{\text{经过 }u,t\text{ 的路径}}',r'\mathrm dX&=G\odot\operatorname{GELU}^{\prime}(X)')+p(m(r'\odot')+' 表示同位置元素相乘。所有输出只依赖同位置的输入，所以这里没有跨特征或跨位置求和。'+m(r'x=0')+' 时局部导数为 '+m(r'1/2')+'；上游仍要乘这个导数。返回 '+m(r'\mathrm dX')+' 给前一层。'))
    g['derivation']='';g['steps']=[]
    g['extension']=h('非线性与近似式',p('连续的线性层可以合并为一个线性映射，非线性激活使模型能够表达更复杂的关系。本课实现 tanh 近似 GELU；精确 GELU 使用高斯分布的累积分布函数，两者公式不同。'))
    g['learning_page']=dict(role='miniGPT 组成部分',purpose='实现 MLP 两个 Linear 之间的非线性层。',problem_title='实现 GELU 前向与反向',statement=h('题目描述',p('使用本页 tanh 近似式实现逐元素 GELU，并将上游梯度转换为输入梯度。'))+h('实现接口',p('<code>GELU.forward(x)</code>、<code>GELU.backward(g)</code>。对象用 <code>GELU()</code> 创建。'))+h('输入格式',p('<code>x</code> 为实数 NumPy 数组，常用形状 '+m(r'(B,T,C)')+'，也支持其他形状。<code>g</code> 是与本次前向输出同形的实数数组；先前向，再反向。'))+h('输出格式',p('<code>forward</code> 返回同形浮点数组 '+m(r'Y')+'；<code>backward</code> 返回同形数组 '+m(r'\mathrm dX')+'。'))+h('状态变化',p('缓存本次输入的独立副本与 tanh 结果；前后向保留传入数组原值。本层参数列表为空，反向只返回输入梯度。')),sample_note=p('本例完整保留两个 batch。每个格子沿相同的逐元素规则计算；零、正值和负值用于核对曲线及局部导数。'))
    r['body']=h('1. RMSNorm 在做什么',p('每个 token 有一行特征。RMSNorm 先用这一行自己的均方根调节整体大小，再乘一个可训练的逐特征缩放向量。每行独立统计；同一个缩放向量在所有位置共享。miniGPT 在注意力与 MLP 前使用它。')+p('构造 <code>RMSNorm(C, eps=1e-5)</code>：创建 '+m(r'w\in\mathbb R^C')+'，初值全为 '+m(r'1')+'，用 <code>Parameter</code> 包装，并保存正数 <code>eps</code>。参数的 <code>grad</code> 由 Parameter 初始化为同形零数组。'+m(r'\epsilon')+' 是固定稳定项，不是可训练参数。'))
    r['body']+=h('2. 前向：先理解一行，再同时处理全部行',p('设输入 '+m(r'X:(B,T,C)')+'。下面固定一个位置，写它的输入行 '+m(r'x')+'，索引 '+m(r'i,j')+' 都表示特征。先平方，再沿特征轴平均，得到均方 '+m(r'm')+'；逆均方根 '+m(r'r')+' 作为这行所有特征共用的缩放因子：')+aligned(r'm&=\frac1C\sum_{j=0}^{C-1}x_j^2',r'r&=\frac1{\sqrt{m+\epsilon}}',r'\widehat x_i&=x_i r,\qquad y_i=\widehat x_i w_i')+p('在 NumPy 中，<code>np.mean(x*x, axis=-1, keepdims=True)</code> 得到均方；<code>axis=-1</code> 选最后的特征轴，<code>keepdims=True</code> 保留长度为一的轴，让这一行的统计量能广播回全部特征。')+math(r'X:(B,T,C)\ \longrightarrow\ m,r:(B,T,1)\ \longrightarrow\ \widehat X,Y:(B,T,C)')+p('这里的 '+m(r'm')+' 是平方后的均值，不是特征均值。例如 '+m(r'x=[1,2,2]')+' 的均方为 '+m(r'3')+'。每个位置分别重复这一过程；两个 batch 的所有位置都保留。缓存 '+m(r'r,\widehat X')+' 即可反向，它们是本次计算新产生的数组。'))
    r['body']+=h('3. 先计算参数梯度：哪些位置用到了同一个缩放',p('后续网络传入同形上游 '+m(r'G=\partial L/\partial Y')+'。缩放 '+m(r'w_c')+' 参与所有位置的第 '+m(r'c')+' 个输出，每条路径的局部导数是归一化后的特征。因此同位置相乘，再沿全部前置位置相加：')+math(r'\mathrm dw_c=\sum_{b,t}G_{b,t,c}\widehat X_{b,t,c}')+p('得到的向量形状为 '+m(r'(C,)')+'，累加到 <code>w.grad</code>。可以将乘积展平为 '+m(r'(BT,C)')+' 再沿行求和；特征轴保留。'))
    r['body']+=h('4. 输入梯度：直接缩放与统计量两条路径汇合',p('先通过最后的逐特征缩放：'+m(r'u=G\odot w')+'，'+m(r'u')+' 是损失对 '+m(r'\widehat X')+' 的梯度，与输入同形。对一个位置，'+m(r'\widehat x_i=x_i r')+'：输入一方面直接参与乘法，另一方面通过均方改变 '+m(r'r')+'，进而影响本行所有特征。')+aligned(r'\left(\mathrm dx_i\right)_{\mathrm{direct}}&=ru_i',r'\frac{\partial L}{\partial r}&=\sum_j u_jx_j',r'\frac{\partial r}{\partial m}&=-\tfrac12r^3,\qquad\frac{\partial m}{\partial x_i}=\frac{2x_i}{C}',r'\left(\mathrm dx_i\right)_{\mathrm{stat}}&=\left(\sum_j u_jx_j\right)\left(-\tfrac12r^3\right)\frac{2x_i}{C}',r'&=-r^3x_i\operatorname{mean}_j(u_jx_j)')+p('间接路径中的求和覆盖本行所有输出，因为它们都用了同一个 '+m(r'r')+'。把两条路径相加，并代入 '+m(r'\widehat x=rx')+'，就得到数组实现形式：')+aligned(r'\mathrm dx_i&=ru_i-r^3x_i\operatorname{mean}_j(u_jx_j)',r'\mathrm dX&=r\left[u-\widehat X\odot\operatorname{mean}_{c}(u\odot\widehat X)\right]')+p('这里的 <code>mean</code> 仍然只沿最后特征轴，并保留轴。输入梯度的统计范围与参数梯度不同：'+m(r'\mathrm dX')+' 在各 token 内合并特征影响；'+m(r'\mathrm dw')+' 跨 token 汇总共享参数贡献。输入梯度保持原形状返回。'))
    r['body']+=h('5. 状态如何保留',p('前向保存本次的逆均方根和归一化特征，反向使用这些缓存并复用当前 <code>w.data</code>。每次反向将本次参数梯度加到已有 <code>w.grad</code>；新一轮统计开始前，外部调用 <code>zero_grad()</code>。参数更新交给后面的优化器。'))
    r['derivation']='';r['steps']=[]
    r['extension']=h('与 LayerNorm 的区别',p('LayerNorm 先减去本行均值，再按方差归一化，常带缩放和偏置。这里的 RMSNorm 使用均方根；两者统计量和反向路径不同。后面会单独给出 LayerNorm 对照任务。'))+h('为什么使用稳定项',p('全零输入时 '+m(r'r=1/\sqrt\epsilon')+'，输出仍为零，输入梯度也有定义。加入稳定项后，归一化特征的均方通常略小于一，不应把它当作精确等于一的测试标准。'))
    r['learning_page']=dict(role='miniGPT 组成部分',purpose='实现 Block 与最终输出归一化使用的 RMSNorm。',problem_title='实现 RMSNorm 的构造、前向与反向',statement=h('题目描述',p('按 token 计算均方根归一化，应用共享逐特征缩放，反向返回输入梯度并累加缩放参数梯度。'))+h('实现接口',p('<code>RMSNorm.__init__(C, eps=1e-5)</code>、<code>forward(x)</code>、<code>backward(g)</code>。'))+h('输入格式',p('<code>C</code> 为正整数，<code>eps</code> 为正实数。<code>x</code> 为最后轴长度为 '+m(r'C')+' 的浮点 NumPy 数组，支持二维、三维和更多前置维度；<code>g</code> 与本次前向输出同形。先前向，再反向。'))+h('输出格式',p('构造后对象持有 <code>w</code> 参数：'+m(r'(C,)')+'，初值全一，梯度全零；<code>forward</code> 返回同形 '+m(r'Y')+'，<code>backward</code> 返回同形 '+m(r'\mathrm dX')+'。'))+h('状态变化',p('每个实例独立持有参数。前向缓存本次 '+m(r'r,\widehat X')+'，反向累加 <code>w.grad</code>；<code>zero_grad()</code> 清零梯度。传入数组与参数数值保留原值。')),sample_note=p('公开样例使用 '+m(r'(B,T,C)=(2,3,4)')+'，六个 token 独立统计，最后一个输入行全零。样例为检查缩放作用，将当前 '+m(r'w')+' 改为非全一向量；构造时的初值仍须全一。'))
    g['learning_page']['statement']+=h('动手任务清单',
        '<ol><li><code>forward(x)</code>：保存输入副本，计算并缓存 tanh 中间结果，返回同形输出。</li><li><code>backward(g)</code>：将两条路径的局部导数相加，乘同位置上游，返回同形输入梯度。</li><li>提交测评：核对正负值、零点、高维形状与缓存隔离。</li></ol>')
    g['learning_page']['statement']+=h('关键公式速查',aligned(
        r'a&=0.044715,\qquad k=\sqrt{2/\pi}',
        r'u&=k(x+ax^3),\quad t=\tanh u,\quad y=\tfrac12x(1+t)',
        r'f^{\prime}(x)&=\tfrac12(1+t)+\tfrac12x(1-t^2)k(1+3ax^2)',
        r'\mathrm dX&=G\odot f^{\prime}(X)'))
    r['learning_page']['statement']+=h('动手任务清单',
        '<ol><li><code>__init__(C, eps)</code>：创建全一的逐特征 Parameter，保存 eps。</li><li><code>forward(x)</code>：沿最后特征轴计算均方与逆均方根，缓存归一化特征，返回缩放结果。</li><li><code>backward(g)</code>：累加参数梯度，合并直接和统计量两条输入路径，返回同形输入梯度。</li><li>提交测评：核对独立构造、全零输入、高维形状、缓存隔离与梯度累加。</li></ol>')
    r['learning_page']['statement']+=h('关键公式速查',aligned(
        r'w&\leftarrow\mathbf 1_C,\qquad m=\operatorname{mean}_c(X\odot X)',
        r'r&=(m+\epsilon)^{-1/2},\quad\widehat X=X\odot r,\quad Y=\widehat X\odot w',
        r'u&=G\odot w,\qquad\mathrm dw_c=\sum_{\text{全部位置}}G_{\ldots,c}\widehat X_{\ldots,c}',
        r'\mathrm dX&=r\left[u-\widehat X\odot\operatorname{mean}_c(u\odot\widehat X)\right]')
        +p('两处 '+m(r'\operatorname{mean}_c')+' 都使用 <code>axis=-1, keepdims=True</code>；'+m(r'\mathrm dw')+' 沿全部前置位置求和并加到 <code>w.grad</code>。'))
    for key,meta in [('layer_gelu',g),('layer_rmsnorm',r)]:
        old=meta['starter']
        if key=='layer_rmsnorm':
            meta['starter']='''class RMSNorm(Module):
    def __init__(self, C, eps=1e-5):
        # TODO：创建全一的逐特征 Parameter，保存 eps。
        raise NotImplementedError("请实现 RMSNorm 构造")

    def forward(self, x):
        # TODO：计算并缓存逆均方根和归一化输入，返回缩放后的输出。
        raise NotImplementedError("请实现前向")

    def backward(self, g):
        # TODO：累加参数梯度，合并两条输入路径，返回输入梯度。
        raise NotImplementedError("请实现反向")
'''
            meta['task_brief']['scope']='实现 RMSNorm.__init__、forward 和 backward'
            meta['assessment_revision']=3
            meta['assessment_migration']=dict(previous_revision=2,reason='增加缩放参数独立构造检查；完成构造、前向和反向后提交。')
            meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='rmsnorm_initialization']
            meta['cases'].append(dict(assessment='rmsnorm_initialization',category='参数构造',label='独立构造：全一缩放、零梯度、eps 与实例隔离',input=dict(configs=[[1,0.01],[3,0.0001],[5,0.1]],operation='构造不同特征宽度与eps的RMSNorm，检查初值和注册；改动当前参数后前向核对复用及eps。'),expected={'构造与复用':True}))
        full='import numpy as np\nfrom learner_core import Parameter, Module\n\n'+meta['starter']+'\n\n# 直接运行本文件时执行本节测评；被模型导入时跳过。\nif __name__ == "__main__":\n    from _foundation_checks import run_local\n    run_local("'+key+'", globals())\n'
        source=meta.get('local_code','')
        legacy='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n\n'+old+'\n\nif __name__ == "__main__":\n    from _foundation_checks import run_local\n    run_local("'+key+'", globals())\n'
        untouched=source.replace('\r\n','\n')==legacy or source==meta.get('scaffold_notes',{}).get('source')
        meta['scaffold_notes']=dict(source=source if untouched else '',code=full,starter=full)
        meta['sync_empty_scaffold']=True
    return book
