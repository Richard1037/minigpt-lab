"""先学明白，再按真实模型接口完成题目；仅生成学习内容。"""
from parameter_lesson import lesson as parameter_explanation


def apply_learner_layout(book, math, aligned):
    infra = book['lessons']['infrastructure']
    # 单独提供带注释框架，个人代码与测评版本保持原值。
    source=infra['local_code']
    notes={
        '    def __call__(self, *args):':'    # 调用 module(X) 时会自动执行 __call__，相当于 module.forward(X)。\n    # 定义中的 *args 把传入的位置参数收集成一个元组。',
        '        return self.forward(*args)':'        # 调用中的 *args 把元组拆成参数，传给 forward，并返回计算结果。',
    }
    def annotate(code):
        return '\n'.join((notes[line]+'\n' if line in notes else '')+line for line in code.split('\n'))
    infra['scaffold_notes']={'source':source,'code':annotate(source),'starter':annotate(infra['starter'])}
    infra['body'] = parameter_explanation()
    infra['body'] = infra['body'].replace('<h3>补充：如何用 DFS 查找参数</h3>', '''
<h3>沿一个实际用例走一遍</h3>
<p>用 <code>[[1, 2]]</code> 创建参数 <code>p</code>，它的 <code>data</code> 形状是 <code>(1, 2)</code>，<code>grad</code> 初始为 <code>[[0, 0]]</code>。让 <code>root.child.weight</code>、<code>root.shared</code> 和 <code>root.blocks[0].weight</code> 都引用它。</p>
<ol><li>将 <code>p.grad</code> 的全部元素改成 5，模拟已有梯度。</li><li>调用 <code>root.parameters()</code>：沿三个引用都能找到 <code>p</code>，但返回列表只保留一个 <code>p</code>。</li><li>调用 <code>root.zero_grad()</code>：它会取得这个列表，再对 <code>p</code> 清零。</li><li>最终 <code>p.grad</code> 为 <code>[[0, 0]]</code>，<code>p.data</code> 仍为 <code>[[1, 2]]</code>。</li></ol>
<p>下方题目的第一个公开样例就使用这一组数据。</p>
<h3>实现时用到的 Python 与 NumPy</h3>
<p><code>self</code> 是当前对象；<code>self.data</code> 和 <code>self.grad</code> 是存在这个对象里的两个属性。创建对象时执行 <code>__init__</code>，之后调用方法会继续使用这些属性。</p>
<p><code>np.array(data, dtype=np.float64, copy=True)</code> 把输入复制成浮点数组；<code>np.zeros_like(array)</code> 创建同形、同类型的全零数组。<code>grad[...] = 0</code> 把已有数组里的全部元素改成 0，其他持有该数组的引用也能看到清零结果。</p>
<h3>补充：如何用 DFS 查找参数</h3>''')
    infra['learning_page'] = {
        'role': 'miniGPT 组成部分',
        'purpose': '你实现的 Parameter 和 Module 会被后续 Linear、Embedding、Block 与 GPT 使用；优化器通过 parameters() 取得实际要更新的参数。',
        'problem_title': '实现 miniGPT 的参数管理接口',
        'statement': '''<h3>题目描述</h3>
<p>实现 <code>Parameter</code> 与 <code>Module</code>，让模型能够创建参数、收集全部可训练参数，并统一清零梯度。</p>
<h3>实现接口</h3>
<div class="contract-table"><table><thead><tr><th>方法</th><th>完成的动作</th></tr></thead><tbody>
<tr><th><code>Parameter.__init__(data)</code></th><td>复制输入得到 <code>float64</code> 的 <code>data</code>，创建同形的全零 <code>grad</code>。</td></tr>
<tr><th><code>Parameter.zero_grad()</code></th><td>原地清零当前参数的 <code>grad</code>。</td></tr>
<tr><th><code>Module.parameters()</code></th><td>深度优先收集参数；按首次遇到的顺序返回，同一个对象只收集一次。</td></tr>
<tr><th><code>Module.zero_grad()</code></th><td>清零本模块收集到的每个参数的 <code>grad</code>。</td></tr>
</tbody></table></div>
<h3>输入格式</h3>
<ul><li>创建 <code>Parameter</code> 时传入 <code>data</code>：可转换成 NumPy 实数数组的数值、列表或数组，形状任意。</li>
<li>调用 <code>Module</code> 方法时，输入来自当前对象 <code>self</code> 的属性。公开属性可包含 <code>Parameter</code>、子 <code>Module</code>、列表、元组、字典，以及普通数值或数组。</li>
<li>列表和元组按元素顺序访问；字典按插入顺序访问值；<code>Module</code> 按属性插入顺序访问公开属性。</li></ul>
<h3>输出格式</h3>
<p><code>parameters()</code> 返回 <code>Parameter</code> 对象组成的 Python 列表；没有参数时返回空列表。两个 <code>zero_grad()</code> 方法均返回 <code>None</code>。</p>
<h3>状态变化与约定</h3>
<ul><li>创建后 <code>data</code> 与传入数组独立，<code>grad</code> 与 <code>data</code> 同形。</li>
<li>清零后 <code>grad</code> 的元素全为 0，原 <code>grad</code> 数组仍被复用，参数 <code>data</code> 保留原值。</li>
<li><code>Module</code> 中以下划线开头的属性按缓存处理并跳过；普通数值和 NumPy 数组在参数收集中跳过。</li>
<li>共享关系按对象身份判断。模块与容器之间不含循环引用。</li></ul>''',
        'sample_input': '''<p>创建参数 <code>p</code>：<code>data</code> 为 <code>[[1, 2]]</code>，形状 <code>(1, 2)</code>。下面三个位置引用同一个 <code>p</code>：</p>
<div class="lesson-tree"><code>root.child.weight</code> → <code>p</code><br><code>root.shared</code> → <code>p</code><br><code>root.blocks[0].weight</code> → <code>p</code></div>
<p>将 <code>p.grad</code> 全部设为 5，再依次调用 <code>root.parameters()</code>、<code>root.zero_grad()</code>。</p>''',
        'sample_note': '<p>三个引用都指向 <code>p</code>，所以参数列表长度为 1。清零作用于 <code>p.grad</code>，得到 <code>[[0, 0]]</code>；<code>p.data</code> 仍为 <code>[[1, 2]]</code>。这是本页第一项公开测评的输入与结果。</p>',
    }
    from initialization_lesson import apply_initialization_lesson
    apply_initialization_lesson(book, math, aligned)
    from scale_linear_lessons import apply_scale_linear_lessons
    apply_scale_linear_lessons(book, math, aligned)
    from embedding_lesson import apply_embedding_lesson
    apply_embedding_lesson(book, math, aligned)
    from activation_norm_lessons import apply_activation_norm_lessons
    apply_activation_norm_lessons(book, math, aligned)
    from mlp_lesson import apply_mlp_lesson
    apply_mlp_lesson(book, math, aligned)
    from independent_linear_lesson import apply_independent_linear_lesson
    apply_independent_linear_lesson(book, math, aligned)
    from part45_lessons import apply_part45
    apply_part45(book, math, aligned)
    from part06_lessons import apply_part06
    apply_part06(book, math, aligned)
    from part07_lessons import apply_part07
    apply_part07(book, math, aligned)
    from part08_lessons import apply_part08
    apply_part08(book, math, aligned)
    from part09_lessons import apply_part09
    return apply_part09(book, math, aligned)
