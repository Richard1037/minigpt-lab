"""Linear 前向的单题精修源；不改实现、测评或个人文件。"""


def apply_linear_forward_lesson(book, math, aligned):
    lesson = book['lessons']['linear']
    m = lambda tex: math(tex, inline=True)
    p = lambda text: '<p>' + text + '</p>'
    h = lambda title, text: '<h3>' + title + '</h3>' + text
    case = lesson['cases'][lesson.get('sample_index', 0)]
    x, w, bias = case['args']
    # 讲解直接使用当前公开三维样例，避免正文与题面使用不同数据。
    B, T, C, D = len(x), len(x[0]), len(w), len(w[0])
    matrix = lambda rows: r'\begin{bmatrix}' + r' \\ '.join(' & '.join(f'{v:g}' for v in row) for row in rows) + r'\end{bmatrix}'
    lesson['body'] = h('1. 先认识输入、参数和输出',
        p('这一题实现 <code>solve(X, W, b)</code>：把每个 token 的一行输入特征，变换成一行输出特征。之后 miniGPT 的 Linear 层会把这段计算放进 <code>forward</code>。')
        + p('一个 batch 是一次处理的一组序列；每条序列有多个 token 位置，每个位置保存一个特征向量。'+m('B')+' 是本批序列数，'+m('T')+' 是每条序列的位置数，'+m('C')+' 是输入特征数，'+m('D')+' 是输出特征数。')
        + aligned(r'X&\in\mathbb R^{B\times T\times C}',r'W&\in\mathbb R^{C\times D},\qquad b\in\mathbb R^D',r'Y&\in\mathbb R^{B\times T\times D}')
        + p('<code>X</code> 是输入；<code>W</code> 是组合特征的权重，'+m('W_{i,j}')+' 决定第 '+m('i')+' 个输入特征对第 '+m('j')+' 个输出特征的贡献；<code>b</code> 是每个输出特征的偏置。参数由题目传入，所有位置共用同一组。'))
    lesson['body'] += h('2. 一个输出来自输入的一行和权重的一列',
        p('用 '+m('q')+' 表示序列编号，'+m('t')+' 表示位置编号，'+m('j')+' 表示输出特征编号。取该位置输入的一整行，与权重第 '+m('j')+' 列逐项相乘并求和，最后加对应偏置：')
        + math(r'Y_{q,t,j}=\underbrace{\sum_{i=0}^{C-1}X_{q,t,i}W_{i,j}}_{\text{沿输入特征乘加}}+b_j')
        + p('这里 '+m('q')+' 是序列下标；偏置始终记作 '+m('b')+'，避免把两者混淆。Linear 改变每个位置的特征，序列和 token 位置保持对应。'))
    lesson['body'] += h('3. 展平：把每个 token 的特征排成一行',
        p('为了使用二维矩阵乘法，先保存输入形状，再把前两个位置轴合并。特征轴保留，每一行仍对应一个完整 token。行号从零开始：')
        + aligned(r'N&=BT,\qquad n=qT+t',r'(X_2)_{n,i}&=X_{q,t,i}',r'(B,T,C)&\longrightarrow(N,C)')
        + p('本页样例有 '+m(rf'B={B},\ T={T},\ C={C},\ D={D}')+'，所以位置表有 '+m(f'N={B*T}')+' 行。按序列、再按 token 排列：')
        + math(r'\begin{array}{c|cccccc}n&0&1&2&3&4&5\\(q,t)&(0,0)&(0,1)&(0,2)&(1,0)&(1,1)&(1,2)\end{array}')
        + math(r'X_2='+matrix([row for batch in x for row in batch]))
        + p('<code>X.shape</code> 取得原始形状；<code>X.shape[-1]</code> 是末轴长度。<code>X.reshape(-1, C)</code> 保留每行的特征数，让 NumPy 推算行数。这里是重新排列形状，元素的先后次序不变。'))
    lesson['body'] += h('4. 计算：六行分别使用同一组权重和偏置',
        aligned(r'Y_2&=X_2W+b',r'(N,C)\,(C,D)&\longrightarrow(N,D)')
        + p('<code>@</code> 执行矩阵乘法。输入表的每一行，与权重的每一列乘加；偏置沿位置轴广播，加到每一行的对应输出特征。')
        + p('例如第一个 token 的第一个输出，使用输入表第零行、权重第零列、偏置第零项：')
        + math(r'(Y_2)_{0,0}=(-1)(-0.5)+(-0.9)(0)+(-0.8)(0.5)+(-0.7)(1)+0=-0.6')
        + p('其他五个位置也各用自己的输入行计算，权重与偏置仍是同一组。'))
    lesson['body'] += h('5. 恢复：让每一行回到原来的序列和位置',
        aligned(r'Y&=\operatorname{reshape}(Y_2,(B,T,D))',r'Y_{q,t,j}&=(Y_2)_{qT+t,j}',r'(6,5)&\longrightarrow(2,3,5)')
        + p('前面三行回到第一条序列，后面三行回到第二条。例如二维表第 '+m('4')+' 行来自 '+m('(q,t)=(1,1)')+'，恢复后也放回这个位置。')
        + p('<code>input_shape[:-1]</code> 取出输入的全部前置轴；<code>W.shape[1]</code> 取得输出宽度。调用 <code>reshape(*input_shape[:-1], D)</code> 时，<code>*</code> 把保存的形状元组拆成多个参数，例如把前置形状 '+m('(2,3)')+' 与宽度 '+m('5')+' 组合成 '+m('(2,3,5)')+'。')
        + p('这就完成了：保存形状 → 排成位置表 → 矩阵乘法与偏置 → 恢复并返回。二维输入本身就是位置表；更多维输入同样保留全部前置轴，只替换末轴。'))
    lesson['derivation'] = ''
    lesson['extension'] = h('更多前置轴时，位置数怎样计算',
        p('若输入有多个前置轴，先把它们的长度相乘得到位置数。末轴仍是一个位置的输入特征，输出时还原全部前置轴：')
        + aligned(r'X&:\ (s_0,\ldots,s_{k-1},C)',r'N&=\prod_{a=0}^{k-1}s_a',r'X_2&:\ (N,C),\qquad Y_2:\ (N,D)',r'Y&:\ (s_0,\ldots,s_{k-1},D)')
        + p('NumPy 的 <code>X @ W + b</code> 也能直接对齐这些轴；本题展开位置表，是为了给下一题的缓存及参数梯度汇总建立清晰的行对应。'))
    lesson['learning_page']['sample_note'] = p('上方样例与讲解使用同一组输入。输入 '+m('(2,3,4)')+' 展平为 '+m('(6,4)')+'，乘权重并加偏置得到 '+m('(6,5)')+'，恢复后返回 '+m('(2,3,5)')+'。输出的第一个位置第一个特征为 '+m('-0.6')+'。')
    lesson['checkpoint_storage_key'] = 'understand-linear-forward-v1'
    lesson['checkpoints'] = [
        dict(prompt='输入形状为 (2,3,4)，展平成二维位置表后是什么形状？', options=['(6,4)', '(2,12)', '(4,6)'], answer=0, explain='两条序列各有三个位置，共六行；每行保留四个输入特征。'),
        dict(prompt='权重形状为 (4,5)，一个输出来自哪些数据？', options=['输入某行与权重某列乘加，再加对应偏置', '只把输入和权重逐元素相乘', '把不同 token 的输入加在一起'], answer=0, explain='每个位置独立应用相同变换；乘加汇总的是输入特征轴。'),
        dict(prompt='二维输出中，行索引为 4 的那一行应恢复到哪个位置？行索引、序列编号和 token 编号均从 0 开始；每条序列有 3 个 token。', options=['序列0、token1', '序列1、token1', '序列1、token0'], answer=1, explain='行号 n=q×3+t；4=1×3+1，因此恢复到第二条序列的第二个位置。'),
    ]
    return book
