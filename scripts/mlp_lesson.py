"""MLP 组合层教学与完整实现题；不写个人文件。"""
STARTER='''class MLP(Module):
    def __init__(self, C, hidden):
        # TODO：创建 fc1、act、fc2 三个独立子层，保存为公开属性。
        raise NotImplementedError("请实现子层构造")

    def forward(self, x):
        # TODO：按 fc1 → act → fc2 传递每一步返回的数组。
        raise NotImplementedError("请实现 MLP 前向")

    def backward(self, g):
        # TODO：按 fc2 → act → fc1 回传每一步返回的梯度。
        raise NotImplementedError("请实现 MLP 反向")
'''

def apply_mlp_lesson(book,math,aligned):
    meta=book['lessons']['layer_mlp'];old=meta['starter']
    def m(t):return math(t,inline=True)
    def p(t):return '<p>'+t+'</p>'
    def h(t,s):return '<h3>'+t+'</h3>'+s
    meta['body']=h('1. MLP 把已经实现的层组合起来',p('你已经实现 Linear 和 GELU。MLP 对每个 token 先用 Linear 扩展特征，经过 GELU 非线性变换，再用另一个 Linear 回到原来的宽度。后面的 Block 会把这个 MLP 作为一个子模块使用。')+p('构造参数 <code>C</code> 是输入和输出宽度，<code>hidden</code> 是中间宽度，公式记为 '+m(r'M')+'。'+m(r'B,T')+' 仍表示序列数和每条序列的位置数；本层只混合特征，不把不同 token 的数据混在一起。'))
    meta['body']+=h('2. 构造：创建三个子层，后续一直复用',
        '<div class="contract-table"><table><thead><tr><th>公开属性</th><th>子层</th><th>作用</th></tr></thead><tbody><tr><td><code>fc1</code></td><td><code>Linear(C, hidden)</code></td><td>把输入宽度扩展到中间宽度</td></tr><tr><td><code>act</code></td><td><code>GELU()</code></td><td>逐元素非线性变换</td></tr><tr><td><code>fc2</code></td><td><code>Linear(hidden, C)</code></td><td>回到原来的宽度</td></tr></tbody></table></div>'
        +p('<code>self.fc1</code> 等属性保存的是子层对象。两个 Linear 各自持有权重和偏置；GELU 没有参数。继承的 <code>parameters()</code> 会沿这些公开属性收集四个参数张量，<code>zero_grad()</code> 会清零它们。')
        +p('构造方法在创建 MLP 时执行一次。前后向调用这些已有对象，缓存与参数才能属于同一组层。Linear 内部继续使用你前面实现的初始化方法。'))
    meta['body']+=h('3. 前向：先给每个中间结果起名字',aligned(
        r'Z&=\mathrm{fc1}(X),& (B,T,C)&\to(B,T,M)',
        r'A&=\operatorname{GELU}(Z),&(B,T,M)&\to(B,T,M)',
        r'Y&=\mathrm{fc2}(A),&(B,T,M)&\to(B,T,C)')
        +p('第一次调用把 '+m(r'X')+' 交给 <code>fc1</code>，它返回 '+m(r'Z')+'；第二次把这个返回值交给 <code>act</code>，得到 '+m(r'A')+'；第三次把 '+m(r'A')+' 交给 <code>fc2</code>。MLP 最终返回 '+m(r'Y')+'。<code>layer(x)</code> 使用继承的 <code>__call__</code> 调用 <code>layer.forward(x)</code>。')
        +p('三个子层分别保存自己反向需要的缓存：<code>fc1</code> 留下原输入 '+m(r'X')+'；<code>act</code>（<code>GELU</code>）留下中间输入 '+m(r'Z')+' 及 '+m(r'\tanh')+' 结果；<code>fc2</code> 留下激活后的输入 '+m(r'A')+'。MLP 的职责是传递这些返回值。'))
    meta['body']+=h('4. 反向：每层收到的是对它自身输出的梯度',p('设后续网络传回 '+m(r'G=\partial L/\partial Y')+'。它的形状是 '+m(r'(B,T,C)')+'，先交给最后一层。用 '+m(r'\mathrm dA,\mathrm dZ,\mathrm dX')+' 命名每个返回值：')
        +aligned(r'\mathrm dA&=\mathrm{fc2.backward}(G),&(B,T,C)&\to(B,T,M)',r'\mathrm dZ&=\mathrm{act.backward}(\mathrm dA),&(B,T,M)&\to(B,T,M)',r'\mathrm dX&=\mathrm{fc1.backward}(\mathrm dZ),&(B,T,M)&\to(B,T,C)')
        +p('为什么顺序反过来？前向的最后一步首先接到上游；它计算损失对自己输入的梯度，这个输入正是前一个子层的输出。因此它返回的数组就是前一个子层需要的上游。')
        +p('将全部位置合并为 '+m(r'N=BT')+' 行，可用已经学过的三个局部公式核对整条链：')
        +aligned(r'Z_2&=X_2W_1+b_1,\quad A_2=f(Z_2),\quad Y_2=A_2W_2+b_2',r'\mathrm dA_2&=G_2W_2^{\mathsf T}',r'\mathrm dZ_2&=\mathrm dA_2\odot f^{\prime}(Z_2)',r'\mathrm dX_2&=\mathrm dZ_2W_1^{\mathsf T}')
        +p('这是链式法则在三个模块之间的传递。实现 MLP 时调用已有子层的 <code>backward</code>，由它们完成各自的导数、形状恢复与参数梯度累计。'))
    meta['body']+=h('5. 参数梯度由两个 Linear 各自保存',aligned(
        r'\mathrm dW_2&=A_2^{\mathsf T}G_2,&\mathrm db_2&=\sum_n(G_2)_{n,:}',
        r'\mathrm dW_1&=X_2^{\mathsf T}\mathrm dZ_2,&\mathrm db_1&=\sum_n(\mathrm dZ_2)_{n,:}')
        +p('<code>fc2.backward</code> 累加第二层的参数梯度，<code>fc1.backward</code> 累加第一层的参数梯度；GELU 只返回输入梯度。MLP 返回最终 '+m(r'\mathrm dX')+'，四个参数梯度留在对应子层中。外部在新一轮统计前调用 MLP 的 <code>zero_grad()</code>。'))
    meta['body']+=h('6. 完整样例的形状走读',p('首个公开样例取 '+m(r'(B,T,C,M)=(2,3,4,6)')+'，所以两条序列的六个 token 都经过同一组子层：')+math(r'X:(2,3,4)\to Z:(2,3,6)\to A:(2,3,6)\to Y:(2,3,4)')+math(r'G:(2,3,4)\to\mathrm dA:(2,3,6)\to\mathrm dZ:(2,3,6)\to\mathrm dX:(2,3,4)')+p('样例的数值输出和四个参数梯度在下方公开测评中核对。不同位置共享同一组参数，但各自的特征和上游保留原位置。'))
    meta['derivation']='';meta['steps']=[]
    meta['extension']=h('中间宽度与更高维输入',p('常见 GPT 使用比输入更宽的中间特征，但本接口由 <code>hidden</code> 显式指定，不固定为某个倍数。对于更多前置轴，所有子层继续只变换最后特征轴；最后的输出与输入同形。'))
    statement=h('题目描述',p('实现可被 miniGPT Block 调用的 MLP：组合两个 Linear 和一个 GELU，完成前向串联与反向梯度传递。'))+h('实现接口',p('<code>MLP.__init__(C, hidden)</code>、<code>forward(x)</code>、<code>backward(g)</code>。'))+h('输入格式',p('<code>C</code>、<code>hidden</code> 为正整数。<code>x</code> 为最后一轴长度为 '+m(r'C')+' 的实数 NumPy 数组，支持二维、三维及更多前置轴。<code>g</code> 与本次前向输出同形；先调用前向，再调用对应反向。'))+h('输出格式',p('构造器按 Python 约定返回 <code>None</code>，创建 <code>fc1</code>、<code>act</code>、<code>fc2</code>。<code>forward</code> 返回同形 '+m(r'Y')+'，<code>backward</code> 返回同形 '+m(r'\mathrm dX')+'。'))+h('状态变化',p('每个实例独立持有子层；前后向使用已有子层。缓存由子层保存，两个 Linear 的参数梯度由子层累加；MLP 的 <code>parameters()</code> 收集四个参数张量。前后向保留输入、上游与参数数值。'))
    statement+=h('动手任务清单','<ol><li>构造三个子层，保存为 <code>fc1</code>、<code>act</code>、<code>fc2</code>。</li><li><code>forward</code>：逐层传入上一步返回值，返回最后结果。</li><li><code>backward</code>：<code>fc2</code> → <code>act</code> → <code>fc1</code>，逐层传递返回的梯度。</li><li>提交测评：构造与真实子层复用、全部梯度、完整形状、缓存隔离及累加。</li></ol>')+h('关键公式速查',aligned(r'Z&=\mathrm{fc1}(X),\quad A=f(Z),\quad Y=\mathrm{fc2}(A)',r'\mathrm dA_2&=G_2W_2^{\mathsf T},\quad\mathrm dZ_2=\mathrm dA_2\odot f^{\prime}(Z_2),\quad\mathrm dX_2=\mathrm dZ_2W_1^{\mathsf T}',r'\mathrm dW_2&=A_2^{\mathsf T}G_2,\quad\mathrm dW_1=X_2^{\mathsf T}\mathrm dZ_2',r'\mathrm db_2&=\sum_n(G_2)_{n,:},\qquad\mathrm db_1=\sum_n(\mathrm dZ_2)_{n,:}')+p('下标 '+m(r'2')+' 表示展平全部前置位置，'+m(r'\odot')+' 表示同位置相乘。偏置梯度沿所有位置求和；参数梯度由子层累计，输入梯度恢复原形状。'))
    meta['learning_page']=dict(role='miniGPT 组成部分',purpose='组合已完成的基础层，供 Block 中的特征变换分支使用。',problem_title='构造并串联 MLP 子层',statement=statement,sample_note=p('同一个完整双batch样例同时检查最终输出、输入梯度、两个权重梯度和两个偏置梯度。展开反向产出可核对四个参数张量。'))
    meta['starter']=STARTER
    prefix='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n\nif __name__ != "student":\n    from layer_linear import Linear\n    from layer_gelu import GELU\n\n'
    ending='\n\nif __name__ == "__main__":\n    from _foundation_checks import run_local\n    run_local("layer_mlp", globals())\n'
    source=meta.get('local_code','');untouched=source.replace('\r\n','\n')==prefix+old+ending or source==meta.get('scaffold_notes',{}).get('source')
    full=prefix.replace('if __name__ != "student":','# 本地运行时导入个人子层；网页测评时由运行环境注入相同依赖。\nif __name__ != "student":')+STARTER+ending.replace('if __name__ == "__main__":','# 直接运行本文件时执行测评；作为子模块导入时跳过。\nif __name__ == "__main__":')
    meta['scaffold_notes']=dict(source=source if untouched else '',code=full,starter=full)
    meta['sync_empty_scaffold']=True
    meta['task_brief']['scope']='实现 MLP.__init__、forward 和 backward'
    meta['task_brief']['dimensions'].pop('d',None)
    meta['assessment_revision']=3;meta['assessment_migration']=dict(previous_revision=2,reason='增加独立构造及子层调用顺序/复用检查。')
    meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='mlp_composition']
    meta['cases'].append(dict(assessment='mlp_composition',category='构造与组合',label='独立构造与实际串联：子层形状、参数注册、调用顺序和对象复用',input=dict(configs=[[2,5],[5,3]],operation='检查不同宽度的三个子层、四个参数张量和实例隔离；用可追踪的子层替身核对前向及反向调用顺序、返回值传递和重复调用中的对象复用。'),expected={'构造与串联':True}))
    # 只转换自然语言节点中的接口标识，跳过已有代码和数学节点。
    from html.parser import HTMLParser
    import re
    class CodeLabels(HTMLParser):
        def __init__(self):super().__init__(convert_charrefs=False);self.out=[];self.skip=[]
        def handle_starttag(self,tag,attrs):
            self.out.append(self.get_starttag_text())
            if tag in ('code','pre') or any(k=='class' and 'math-' in v for k,v in attrs):self.skip.append(tag)
        def handle_endtag(self,tag):
            self.out.append('</'+tag+'>')
            if self.skip and self.skip[-1]==tag:self.skip.pop()
        def handle_data(self,text):
            if not self.skip:text=re.sub(r'(?<![A-Za-z0-9_])(fc1|fc2|act|Linear|GELU|forward|backward)(?![A-Za-z0-9_])',r'<code>\1</code>',text)
            self.out.append(text)
        def handle_entityref(self,name):self.out.append('&'+name+';')
        def handle_charref(self,name):self.out.append('&#'+name+';')
    for obj,key in [(meta,'body'),(meta,'extension'),(meta['learning_page'],'statement')]:
        parser=CodeLabels();parser.feed(obj[key]);obj[key]=''.join(parser.out)
    return book
