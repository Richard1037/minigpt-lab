"""独立 Linear：完整对象契约、必要推导和安全同步的待填框架。"""
STARTER='''class Linear(Module):
    def __init__(self, C, D, use_bias=True):
        # TODO：调用 initialize 创建权重并包装为 Parameter；创建可选偏置。
        raise NotImplementedError("请实现 Linear 构造")

    def forward(self, x):
        # TODO：保存原形状和二维输入副本，计算输出并恢复前置维度。
        raise NotImplementedError("请实现 Linear 前向")

    def backward(self, g):
        # TODO：累加参数梯度，返回恢复为原输入形状的输入梯度。
        raise NotImplementedError("请实现 Linear 反向")
'''

def apply_independent_linear_lesson(book,math,aligned):
    meta=book['lessons']['rebuild_linear'];old=meta['starter']
    def m(t):return math(t,inline=True)
    def p(t):return '<p>'+t+'</p>'
    def h(t,s):return '<h3>'+t+'</h3>'+s
    meta['body']=h('1. 这次从接口构造一个完整层',p('前面的 <code>Linear</code> 任务给出了构造器。这次你要在 <code>rebuild_linear.py</code> 中独立实现 <code>Linear.__init__</code>、<code>forward</code> 和 <code>backward</code>，让一个对象自己持有参数、保存本次前向缓存，并把梯度传给前一层。')+p('你已经实现的 <code>Parameter</code>、<code>Module</code> 与 <code>initialize</code> 由 <code>learner_core</code> 提供。继承 <code>Module</code> 后，可以使用参数收集、清零和 <code>layer(x)</code> 调用；本题自己创建参数并完成计算。'))
    meta['body']+=h('2. 创建对象：先确定每个属性是什么',p('<code>Linear(C, D, use_bias=True)</code> 中，'+m('C')+' 是输入特征宽度，'+m('D')+' 是输出特征宽度；<code>use_bias</code> 决定是否启用偏置。构造时创建以下对象：')
        +'<div class="contract-table"><table><thead><tr><th>属性</th><th>数值与初值</th><th>梯度</th></tr></thead><tbody><tr><td><code>self.W</code></td><td>'+m('(C,D)')+'；调用 <code>initialize((C, D))</code>，再包装成 <code>Parameter</code></td><td>'+m('(C,D)')+'，由 <code>Parameter</code> 初始化为零</td></tr><tr><td><code>self.b</code></td><td>启用时是 '+m('(D,)')+' 的全零 <code>Parameter</code>；关闭时为 <code>None</code></td><td>启用时为同形零数组</td></tr></tbody></table></div>'
        +p('构造器接入你前面实现的初始化函数，其规则是 '+m(r'W_{i,j}\sim\mathcal N(0,\sigma^2),\ \sigma=\sqrt{2/C}')+'。在本课接口中不传随机数生成器，沿用该函数的默认行为。偏置全零，不使用权重的随机初始化规则。')
        +p('参数数值读取 <code>W.data</code> 与可选的 <code>b.data</code>，梯度累计到各自的 <code>grad</code>。参数只在构造时创建，后续前后向复用这些对象。两个不同的层实例各自持有独立参数。'))
    meta['body']+=h('3. 前向：保留全部位置，变换最后一轴',p('设输入 '+m(r'X\in\mathbb R^{B\times T\times C}')+'；'+m('B')+' 是序列数，'+m('T')+' 是每条序列的位置数。所有位置共用同一个 '+m('W')+' 和可选偏置。把每个位置排成一行：')
        +aligned(r'N&=BT,\qquad X_2=\operatorname{reshape}(X,(N,C))',r'Y_2&=X_2W+b',r'Y&=\operatorname{reshape}(Y_2,(B,T,D))')
        +p('一行输入与 '+m('W')+' 的每一列组合，得到该位置的全部输出特征。偏置沿所有位置广播相加。输入也可以是 '+m(r'(s_0,\ldots,s_k,C)')+'；同样只把前置位置合并，输出为 '+m(r'(s_0,\ldots,s_k,D)')+'。')
        +p('代码中，<code>x.shape[:-1]</code> 取得全部前置维度，<code>x.reshape(-1, C)</code> 将它们合并；<code>-1</code> 让 NumPy 根据元素总数计算位置数。恢复输出时，在原前置维度后接上输出宽度 '+m('D')+'。'))
    meta['body']+=h('4. 缓存：反向还需要哪两样数据',p('在 <code>forward</code> 保存原始输入形状到 <code>self._shape</code>，并将展平后的输入独立副本保存到 <code>self._x2</code>。原形状用于恢复输入梯度，二维副本用于计算权重梯度。')
        +p('<code>reshape</code> 的结果可能与原输入共享存储；调用 <code>copy()</code> 后，外部代码修改原数组就不会改变这次前向的缓存。缓存名以下划线开头，按本项目 <code>Module.parameters()</code> 的规则跳过；只有参数对象会交给优化器。'))
    meta['body']+=h('5. 反向：先定义上游，再分别计算三类梯度',p('最终损失记为 '+m('L')+'。<code>backward(g)</code> 收到 '+m(r'G=\partial L/\partial Y')+'，它与本次输出同形。按相同位置顺序展平为 '+m('G_2:(N,D)')+'。记 '+m(r'\mathrm dX,\mathrm dW,\mathrm db')+' 为损失对输入、权重和偏置的梯度。'))
    meta['body']+=h('输入梯度：沿输出特征收回影响',p('在同一个位置，输入特征 '+m('i')+' 通过 '+m('W')+' 的这一行影响所有输出特征；局部导数是对应权重。将上游乘这些权重，再沿输出特征求和：')
        +aligned(r'\frac{\partial(Y_2)_{n,j}}{\partial(X_2)_{n,i}}&=W_{i,j}',r'(\mathrm dX_2)_{n,i}&=\sum_j(G_2)_{n,j}W_{i,j}',r'\mathrm dX_2&=G_2W^{\mathsf T}\quad:(N,D)(D,C)\to(N,C)')
        +p('转置让最后的输出特征轴收缩掉、输入特征轴保留下来。最后用缓存的 <code>self._shape</code> 恢复完整输入形状，作为返回值。'))
    meta['body']+=h('权重梯度：一处位置贡献一张表，再跨全部位置相加',p('每个位置的输入行都通过同一个 '+m('W')+' 影响输出。输出对权重的局部导数是对应输入，所以这一行输入与这一行上游两两相乘，得到整张权重的梯度贡献。所有位置共享参数，贡献要相加：')
        +aligned(r'\frac{\partial(Y_2)_{n,j}}{\partial W_{i,j}}&=(X_2)_{n,i}',r'\mathrm dW&=\sum_{n=0}^{N-1}(X_2)_{n,:}^{\mathsf T}(G_2)_{n,:}',r'&=X_2^{\mathsf T}G_2\quad:(C,N)(N,D)\to(C,D)')
        +p('这里使用前向缓存的输入，计算本次贡献后加到 <code>self.W.grad</code>。上游已经包含损失层的平均系数，本层直接汇总贡献。'))
    meta['body']+=h('偏置梯度：同一特征的上游跨位置相加',p('每个位置的第 '+m('j')+' 个输出都加同一个 '+m('b_j')+'，局部导数为 '+m('1')+'。因此保留输出特征轴，对位置轴求和：')
        +math(r'\mathrm db_j=\sum_n(G_2)_{n,j},\qquad\mathrm db=\operatorname{sum}(G_2,\mathrm{axis}=0)')
        +p('将本次贡献加到 <code>self.b.grad</code>。当 <code>self.b</code> 为 <code>None</code> 时，前向直接返回权重乘积，反向只累计权重梯度并返回输入梯度；参数列表只有权重。'))
    meta['body']+=h('6. 返回值、梯度累计与更新分清',p('<code>forward</code> 返回输出；<code>backward</code> 返回输入梯度，并把参数梯度留在对象中。多次反向会累计贡献；外部在新一轮统计前调用继承的 <code>zero_grad()</code>。前后向保留参数 <code>data</code>，参数更新由后面的优化器负责。'))
    meta['body']+=h('7. 完整样例与两实例组合',p('首例使用 '+m('(B,T,C,D)=(2,3,4,5)')+'，两条序列的六个位置都参与计算。展平前后：')
        +math(r'X:(2,3,4)\to X_2:(6,4)\to Y_2:(6,5)\to Y:(2,3,5)')
        +p('样例先创建对象，再将参数设为下方给定矩阵，便于核对前后向；这组演算参数与构造时的随机初值是两个检查环节。独立构造测评会在覆盖参数之前验证初值和实际初始化函数调用。')
        +p('组合测评还会创建两个本题的 <code>Linear</code>，按 '+m('(2,3,3)\to(2,3,5)\to(2,3,2)')+' 串联，分别检查输入梯度与两层参数梯度，确认每个对象独立保存自己的参数和缓存。'))
    meta['derivation']='';meta['steps']=[]
    meta['extension']=h('后续独立模型的接线',p('本题产物是 <code>rebuild_linear.py</code> 中的 <code>Linear</code>。当前组合测评已在同一进程实际使用该类建立两层网络。后续独立 <code>Block</code> 与 <code>GPT</code> 的导入链仍需在对应章节整合，接线任务保留在课程清单。'))+h('直接批量矩阵乘法',p('NumPy 的 <code>matmul</code> 支持保留前置维度的前向计算；展平实现的作用是显式对齐全部位置，并让参数梯度的汇总轴保持清楚。两种写法的输入输出契约一致。'))
    statement=h('题目描述',p('从三个方法接口实现一个可复用的 <code>Linear</code> 层，创建参数，完成任意前置维度的特征变换，并正确传递与累计梯度。'))+h('实现接口',p('<code>Linear.__init__(C, D, use_bias=True)</code>、<code>forward(x)</code>、<code>backward(g)</code>。'))+h('输入格式',p('<code>C</code>、<code>D</code> 为正整数，<code>use_bias</code> 为布尔值。<code>x</code> 是至少二维、最后一轴长度为 '+m('C')+' 的实数 NumPy 数组；<code>g</code> 是与本次前向输出同形的实数数组。先执行一次前向，再执行对应反向。'))+h('输出格式',p('构造器返回 <code>None</code>，对象持有 <code>W</code> 和可选的 <code>b</code>。<code>forward</code> 返回前置形状不变、最后轴为 '+m('D')+' 的数组；<code>backward</code> 返回与原输入同形的 '+m(r'\mathrm dX')+'。'))+h('状态变化',p('构造时调用当前 <code>initialize((C, D))</code>，将结果包装成 <code>Parameter</code>；偏置启用时创建全零参数，关闭时设为 <code>None</code>。前向缓存原形状与二维输入副本，前后向复用同一组参数。反向累计 <code>W.grad</code> 和可选的 <code>b.grad</code>；输入、上游与参数数值保持原值。'))
    statement+=h('动手任务清单','<ol><li><code>__init__</code>：创建权重与可选偏置，参数初始梯度由 <code>Parameter</code> 创建。</li><li><code>forward</code>：保存原形状和二维输入副本，矩阵乘法、可选加偏置、恢复输出形状。</li><li><code>backward</code>：上游展平，累计两类参数梯度，恢复并返回输入梯度。</li><li>提交测评：独立构造、可选偏置、完整高维、缓存隔离、累计清零和双实例组合。</li></ol>')+h('关键公式速查',aligned(r'W&=\operatorname{initialize}((C,D)),\quad b=\mathbf0_D\ \text{（启用偏置时）}',r'X&:(s_0,\ldots,s_k,C),\quad N=\prod_{\ell=0}^{k}s_\ell,\quad X_2:(N,C),\quad G_2:(N,D)',r'Y_2&=X_2W+b,\qquad Y:(s_0,\ldots,s_k,D)',r'\mathrm dX_2&=G_2W^{\mathsf T},\quad\mathrm dW=X_2^{\mathsf T}G_2,\quad\mathrm db=\sum_n(G_2)_{n,:}')+p('输出和输入梯度都恢复原前置维度。参数梯度是本次贡献，使用累计加法写入 <code>grad</code>。关闭偏置时省去加偏置与偏置梯度。'))
    meta['learning_page']=dict(role='miniGPT 基础模块的独立实现',purpose='从参数构造到前向缓存和反向传播，完成一个可以由其他模块实例化的基础层。',problem_title='独立实现完整 Linear 层',statement=statement,sample_note=p('这组完整双batch样例沿用前面的矩阵计算，核对全部输出与三类梯度。新增构造和双实例测评分别验证初始对象、可选偏置与实际组合使用。'))
    prefix='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n\n'
    ending='\n\nif __name__ == "__main__":\n    from _foundation_checks import run_local\n    run_local("rebuild_linear", globals())\n'
    source=meta.get('local_code','');legacy=old+ending
    full='# 使用本人的公共接口与初始化函数。\n'+prefix+STARTER+'\n# 直接运行执行本节测评；被其他模块导入时跳过。'+ending
    untouched=source.replace('\r\n','\n')==legacy or source==meta.get('scaffold_notes',{}).get('source')
    meta['starter']=STARTER;meta['scaffold_notes']=dict(source=source if untouched else '',code=full,starter=full);meta['sync_empty_scaffold']=True
    meta['assessment_revision']=2;meta['assessment_migration']=dict(previous_revision=1,reason='新增独立构造/初始化调用及两个个人Linear的组合检查。')
    meta['task_brief']['next_use']=dict(lesson='rebuild_block',mode='contrast')
    meta['task_brief']['conditions']=['位置顺序一致；前向缓存独立副本；参数梯度跨全部位置相加。']
    meta['hint']='构造时创建参数，前向保存形状与二维副本；反向复用这份缓存，参数梯度累计、输入梯度返回。'
    meta['cases']=[c for c in meta['cases'] if c.get('assessment') not in ['independent_linear_build','independent_linear_composition']]
    meta['cases'].append(dict(assessment='independent_linear_build',category='独立构造',label='构造与初始化接入：有/无偏置、当前Parameter、参数与函数复用',input=dict(configs=[[2,5,True],[4,3,False],[1,2,True]],operation='在覆盖参数前检查权重初值、偏置、零梯度及注册。随后以可追踪initialize返回不同权重，检查构造确实调用当前函数；两次前向保留参数对象/当前数值/已有梯度。'),expected={'独立构造与初始化接入':True}))
    meta['cases'].append(dict(assessment='independent_linear_composition',category='组合复用',label='两个个人Linear串联：不同宽度、完整双batch与独立缓存',input=dict(C=3,hidden=5,D=2,shape=[2,3,3],seed=219,operation='构造当前提交的两个Linear，使用固定种子生成输入/上游并覆盖不同参数，执行串联前向及逆序反向，核对最终输出、输入梯度与两层的全部参数梯度。'),expected={'两个个人Linear已串联':True,'对象和缓存独立':True}))
    return book
