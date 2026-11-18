"""Embedding 的完整实现题；只生成课程，不改写已有个人练习文件。"""


STARTER = '''class Embedding(Module):
    def __init__(self, V, C):
        # TODO：创建局部随机数生成器，生成词表并包装成 Parameter。
        raise NotImplementedError("请实现词表参数构造")

    def forward(self, ids):
        # TODO：保存编号副本，返回各位置对应的完整特征行。
        raise NotImplementedError("请实现 Embedding 前向")

    def backward(self, g):
        # TODO：将每个位置的上游梯度累加回对应词表行。
        raise NotImplementedError("请实现 Embedding 反向")
'''

LEGACY_CONSTRUCTOR = '''class Embedding(Module):

    def __init__(self, V, C):
        self.weight = Parameter(np.random.default_rng(0).normal(0, 0.02, (V, C)))

    def forward(self, ids):
        """TODO：按网页公式实现此方法。"""
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        """TODO：按网页公式实现此方法。"""
        raise NotImplementedError('请实现 backward')
'''


def apply_embedding_lesson(book, math, aligned):
    meta = book['lessons']['layer_embedding']
    def m(tex):
        return math(tex, inline=True)
    def p(text):
        return '<p>' + text + '</p>'
    def section(title, content):
        return '<h3>' + title + '</h3>' + content
    def matrix(rows):
        return r'\begin{bmatrix}' + r'\\'.join(' & '.join(map(str,row)) for row in rows) + r'\end{bmatrix}'

    meta['body'] = section('1. Embedding 把编号变成一行可训练的特征',
        p('文本经过编码后，每个 token 都有一个整数编号。Embedding 收到这些编号，从一张参数表中为每个位置取出一行实数，交给后面的网络计算。这里的 token 可以是一个字符，也可以是分词得到的一段文本。')
        + p('这张表记作 '+m('E')+'，代码中保存在 <code>self.weight.data</code>。'+m('V')+' 是可用编号的数量，'+m('C')+' 是每行的特征数；编号 '+m('v')+' 对应行索引 '+m('v')+'。所有编号和下标都从 '+m('0')+' 开始。')
        + math(r'E\in\mathbb R^{V\times C},\qquad 0\le v<V')
        + p('同一个编号在不同位置出现时，都读取同一行参数。训练时，这一行里的实数会被优化器调整，逐渐成为有用的特征表示。'))

    meta['body'] += section('2. 构造时创建一张词表',
        p('调用 <code>Embedding(V, C)</code> 创建对象。在 <code>__init__</code> 中创建一个种子为 '+m('0')+' 的局部随机数生成器，生成形状为 '+m('(V,C)')+'、均值为 '+m('0')+'、标准差为 '+m('0.02')+' 的正态样本。随后用 <code>Parameter</code> 包装，保存为 <code>self.weight</code>。')
        + p('NumPy 的 <code>rng.normal(loc, scale, size)</code> 中，<code>loc</code> 是均值，<code>scale</code> 是标准差，<code>size</code> 是数组形状。这些初值在创建对象时生成一次。')
        + math(r'E_{v,c}\sim\mathcal N(0,0.02^2),\qquad \operatorname{shape}(E)=(V,C)')
        + p('<code>Parameter</code> 会保存一份浮点数据，并创建同形的零梯度：<code>weight.data</code> 用于查表，<code>weight.grad</code> 用于累计这张表收到的梯度。之后每次前向都使用对象中当前的参数。'))

    meta['body'] += section('3. 前向：编号指向哪一行，就把哪一行放到输出',
        p('输入 <code>ids</code> 是整数数组。常用形状为 '+m('(B,T)')+'：'+m('B')+' 条序列，每条有 '+m('T')+' 个 token。每个编号查出 '+m('C')+' 个特征，因此输出多出最后一条特征轴：')
        + math(r'\mathrm{ids}:(B,T)\quad\longrightarrow\quad Y:(B,T,C)')
        + p('固定序列编号 '+m('b')+' 和 token 位置 '+m('t')+'。先读取 '+m(r'v=\mathrm{ids}_{b,t}')+'，再把 '+m('E')+' 中行索引为 '+m('v')+' 的完整一行放到 '+m(r'Y_{b,t,:}')+'：')
        + math(r'Y_{b,t,c}=E_{\mathrm{ids}_{b,t},c}')
        + p('NumPy 支持用整数数组索引参数表：索引数组的每个位置各取一行，输出会保留索引数组的全部轴，再接上词表的特征轴。'))

    e = matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
    ids = matrix([[1,2,1],[0,1,3]])
    meta['body'] += section('4. 沿本页的两个 batch 样例走一遍',
        p('本页公开样例的当前词表与编号如下。测评把词表设为这些确定数值，便于逐行追踪来源；前向始终读取当前的 <code>weight.data</code>。')
        + math(r'E='+e+r',\qquad \mathrm{ids}='+ids)
        + p('第一条序列依次取行索引 '+m('1,2,1')+'，第二条依次取 '+m('0,1,3')+'；输出的两个 batch 都保留：')
        + aligned(r'Y_{0,:,:}&='+matrix([[3,4,5],[6,7,8],[3,4,5]]),
                  r'Y_{1,:,:}&='+matrix([[0,1,2],[3,4,5],[9,10,11]]))
        + p('输出形状为 '+m('(2,3,3)')+'。例如 '+m(r'Y_{0,0,:}=Y_{0,2,:}=Y_{1,1,:}=E_{1,:}')+'，这三个位置都来自同一行。'))

    meta['body'] += section('5. 缓存：保存这次前向实际使用的编号',
        p('反向需要知道：每个输出位置是从词表的哪一行取来的。因此前向保存 <code>ids</code> 的独立副本，例如放进 <code>self._ids</code>。若只保存引用，传入数组的那段代码后来原地修改编号，反向就可能把梯度送到另一行。')
        + p('<code>np.asarray(ids)</code> 取得 NumPy 数组表示；再调用 <code>copy()</code>，得到不随外部原地修改而变化的副本。以当前这份缓存完成对应反向；下一次前向会保存新的编号。'))

    meta['body'] += section('6. 反向：每个位置的梯度沿原路回到词表行',
        p('设最终损失为标量 '+m('L')+'。后面的网络把输出梯度 '+m(r'G=\partial L/\partial Y')+' 传给 <code>backward(g)</code>；它与输出同形 '+m('(B,T,C)')+'。记 '+m(r'\mathrm dE=\partial L/\partial E')+' 为本次要加到词表的梯度。')
        + p('一个输出元素就是词表中某个元素的值。固定词表行 '+m('v')+' 和特征 '+m('j')+'，只有“编号等于 '+m('v')+'，且特征也等于 '+m('j')+'”的输出依赖这个元素：')
        + math(r'\frac{\partial Y_{b,t,c}}{\partial E_{v,j}}=\begin{cases}1&\mathrm{ids}_{b,t}=v\ \text{且}\ c=j\\0&\text{其他情况}\end{cases}')
        + p('用链式法则把上游乘以这个局部导数，再汇总所有输出路径：')
        + aligned(r'\mathrm dE_{v,j}&=\sum_{b,t,c}G_{b,t,c}\frac{\partial Y_{b,t,c}}{\partial E_{v,j}}',
                  r'&=\sum_{\substack{b,t\\\mathrm{ids}_{b,t}=v}}G_{b,t,j}')
        + p('特征求和中的其他项都乘以 '+m('0')+'，只留下 '+m('c=j')+'；位置求和只保留编号等于 '+m('v')+' 的位置。因此每个位置的整行上游，直接加回该编号对应的词表梯度行。'))

    meta['body'] += section('7. 重复编号：三处来源汇总到同一行',
        p('样例中，编号 '+m('1')+' 出现在 '+m('(0,0),(0,2),(1,1)')+'。它们从同一行读出特征，也共同影响这一行的参数梯度：')
        + aligned(r'\mathrm dE_{1,:}&=G_{0,0,:}+G_{0,2,:}+G_{1,1,:}',
                  r'&=\begin{bmatrix}0&1&2\end{bmatrix}+\begin{bmatrix}6&7&8\end{bmatrix}+\begin{bmatrix}12&13&14\end{bmatrix}',
                  r'&=\begin{bmatrix}18&21&24\end{bmatrix}')
        + p('这同时包含两个 batch 的来源。其他行按各自编号收集贡献；本次没有被访问的行，收到的新增贡献为零。已有梯度仍保留，再加上本次贡献。'))

    meta['body'] += section('8. 用散射累加实现，再把状态留给优化器',
        p('这是前面“散射累加”练习的用途。把位置轴合并后，每个编号恰好对应一行上游：')
        + aligned(r'N&=BT',r'\mathrm{ids}_{\mathrm{flat}}&=\operatorname{reshape}(\mathrm{ids},(N,))',
                  r'G_2&=\operatorname{reshape}(G,(N,C))')
        + p('两者都按先 batch、后 token 的相同顺序排。样例的编号顺序是 '+m(r'[1,2,1,0,1,3]')+'，'+m('G_2')+' 的六行分别来自相同的六个位置。')
        + p('<code>np.add.at</code> 的三个参数依次是：要修改的梯度缓冲、每行贡献要去的行编号、要加回的贡献数组。它会把每一次出现都加进去，适合重复编号；缓冲的形状始终是 '+m('(V,C)')+'。')
        + p('反向把这些贡献累加到 <code>weight.grad</code>，返回 <code>None</code>。编号是离散地址，训练改变的是词表里的实数参数，本接口没有返回给整数编号的梯度。')
        + p('开始新一轮梯度统计前，用继承的 <code>zero_grad()</code> 清零。反向保留 <code>weight.data</code> 的数值，优化器稍后读取梯度并更新参数；输入编号与传入的上游数组也保留原值。'))
    meta['derivation'] = ''
    meta['extension'] = section('高级索引的加法与散射累加',
        p('NumPy 的高级索引会先取得一份中间结果，再回写。若索引重复，普通索引加法不能保证每份贡献都被累计。<code>np.add.at</code> 逐次处理重复位置；这里对应同一个参数的多条梯度路径。'))
    meta['extension'] += section('更多前置轴与位置查表',
        math(r'\mathrm{ids}:(s_0,\ldots,s_{k-1})\quad\longrightarrow\quad Y:(s_0,\ldots,s_{k-1},C)')
        + p('反向把所有编号位置排成一列，把对应上游排成同样多行后散射。后面的 GPT 会使用你实现的 Embedding：一张表按 token 编号查行，另一张表按位置编号查行，再将两种特征相加。'))

    meta['learning_page'] = dict(role='miniGPT 组成部分',
        purpose='实现可训练的查表层。后续 GPT 的 token 表和位置表会直接使用这个 Embedding 类。',
        problem_title='实现 Embedding 的构造、前向与反向',
        statement=section('题目描述',p('创建一张可训练词表，按整数编号查出特征，并把反向传播收到的每处梯度累加回原来的词表行。'))
        + section('实现接口',p('<code>Embedding.__init__(V, C)</code>、<code>Embedding.forward(ids)</code>、<code>Embedding.backward(g)</code>。'))
        + section('输入格式',p('<code>V</code>、<code>C</code> 为正整数。构造时使用种子为 '+m('0')+' 的局部 <code>np.random.default_rng</code>，按 '+m(r'\mathcal N(0,0.02^2)')+' 生成 '+m('(V,C)')+' 数组并包装成 <code>Parameter</code>，属性名为 <code>weight</code>。')
            + p('<code>ids</code> 为至少一维的整数 NumPy 数组，每个元素满足 '+m(r'0\le\mathrm{ids}<V')+'；常用形状是 '+m('(B,T)')+'。<code>g</code> 为与这次前向输出同形的实数 NumPy 数组。先完成一次前向，再使用相应上游执行反向。'))
        + section('输出格式',p('构造方法按 Python 约定返回 <code>None</code>，对象持有 '+m('(V,C)')+' 的参数及同形零梯度。<code>forward</code> 返回浮点 NumPy 数组，形状为原编号形状末尾追加 '+m('C')+'；二维编号对应 '+m('(B,T,C)')+'。<code>backward</code> 返回 <code>None</code>。'))
        + section('状态变化',p('每个对象独立持有自己的 <code>weight</code>；前向缓存编号副本并复用当前参数，反向累加 <code>weight.grad</code>。反复出现的编号、跨 batch 的同一编号和多次反向都累计贡献。<code>zero_grad()</code> 原地清零梯度；输入、上游和参数数值在前后向中保持原值。')),
        sample_note=p('上方是讲解中相同的完整样例：'+m('(B,T,V,C)=(2,3,4,3)')+'。前向输出按两条序列分别查行；反向的三处编号 '+m('1')+' 汇总到行索引 '+m('1')+'。展开本例反向产出和下方演示，可以沿这三条来源核对。'))
    meta['starter'] = STARTER
    full = ('import numpy as np\nfrom learner_core import Parameter, Module\n\n' + STARTER
            + '\n# 直接运行文件时执行本节测评；被后续模型导入时跳过。\n'
              'if __name__ == "__main__":\n    from _foundation_checks import run_local\n'
              '    run_local("layer_embedding", globals())\n')
    source = meta.get('local_code', '')
    legacy_file = ('import numpy as np\nfrom learner_core import Parameter, Module, initialize\n\n'
                   + LEGACY_CONSTRUCTOR + '\nif __name__ == "__main__":\n'
                   '    from _foundation_checks import run_local\n'
                   '    run_local("layer_embedding", globals())\n')
    untouched = source.replace('\r\n', '\n') == legacy_file
    # 仅无个人草稿、无成绩且仍为旧原始模板时，网页默认显示新待填框架。
    # 当前目录接口实时读取的个人文件若有任何变化，source 比较会保护它。
    meta['scaffold_notes'] = dict(source=source if untouched else '', code=full, starter=full)
    meta['sync_empty_scaffold'] = True
    meta['assessment_revision'] = 3
    meta['assessment_migration'] = dict(previous_revision=2, reason='本节新增词表构造测评，完成三个方法后提交。')
    meta['cases'] = [c for c in meta['cases'] if c.get('assessment') != 'embedding_initialization']
    meta['cases'].append(dict(assessment='embedding_initialization', category='参数构造',
        label='独立构造：不同词表尺寸、确定初值、零梯度与参数复用',
        input=dict(configs=[[2,3],[5,2],[1,4]], seed=0, std=0.02,
                   operation='分别构造 Embedding(V,C)，直接检查尚未被覆盖的参数；改变当前参数再做两次查表，核对参数对象与数值复用。'),
        expected={'参数形状':[[2,3],[5,2],[1,4]], '初值符合给定种子':True,
                  '初始梯度为零':True, 'Parameter已注册':True, '实例相互独立':True,
                  '前向复用当前参数':True, '全局随机状态保持':True}))
    meta['steps'] = []
    meta['hint'] = '先用缓存的编号与上游按相同顺序展平，再用 np.add.at 把每行贡献加到 weight.grad 对应行。'
    meta['task_brief'].update(scope='实现 Embedding.__init__、forward 和 backward',
        operation='构造词表；前向按编号查行并缓存；反向把每处上游散射累加到词表行。')
    return book
