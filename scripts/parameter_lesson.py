"""Parameter 与 Module 的学习者讲解。"""
import html

def lesson():
    return '''<div class="lesson-explanation">
<p>这一节的核心是：<strong>给模型里的可训练数值建立一套管理方式，方便反向传播记录梯度、训练时统一更新参数。</strong></p>
<p>我们从一个 Linear 层开始。</p>
<h3>1. Parameter：管理一个参数及其梯度</h3>
<p>Linear 层有权重 <code>W</code> 和偏置 <code>b</code>。训练时，每个参数都需要保存两份东西：</p>
<ul><li><strong><code>data</code></strong>：当前参数值，用于前向计算。</li><li><strong><code>grad</code></strong>：损失对这个参数的梯度，用于更新参数。</li></ul>
<p>例如，一个权重矩阵及其梯度：</p>
<div class="lesson-example"><div><strong><code>W.data</code></strong><div>1　2<br>3　4</div></div><div><strong><code>W.grad</code> · 刚创建时</strong><div>0　0<br>0　0</div></div></div>
<p>前向传播读取 <code>W.data</code>；反向传播计算梯度，并累加到 <code>W.grad</code>；优化器再根据 <code>W.grad</code> 更新 <code>W.data</code>。</p>
<p>因此 <code>Parameter</code> 要完成两件事：</p>
<ol><li><code>__init__(data)</code>：复制输入作为 <code>self.data</code>，并创建同形状、全零的 <code>self.grad</code>。</li><li><code>zero_grad()</code>：把 <code>self.grad</code> 清零，<strong>参数值 <code>self.data</code> 保持不变</strong>。</li></ol>
<p>这里的“复制”是为了让参数有自己的数据。例如你用数组创建 <code>W</code> 后，再修改原数组，不应意外修改 <code>W.data</code>。</p>
<h3>2. Module：管理一个层或模型</h3>
<p>一个 Linear 层不止有一个参数，它有 <code>W</code> 和 <code>b</code>；一个 miniGPT 又由很多层组成。</p>
<p><code>Module</code> 是这些层和模型的共同基础。可以把它理解成一个能包含参数和其他模块的对象：</p>
<div class="lesson-tree">GPT<br>├── Embedding<br>│　 └── <code>weight</code> → <code>Parameter</code><br>├── Block<br>│　 ├── Linear 1 → <code>W</code>、<code>b</code><br>│　 └── Linear 2 → <code>W</code>、<code>b</code><br>└── …</div>
<p>这里有两个层次：</p><ul><li><code>Parameter</code> 管理<strong>一个参数</strong>的数值和梯度。</li><li><code>Module</code> 管理<strong>一个层或模型</strong>中包含的所有参数和子模块。</li></ul>
<h3>3. 为什么 Module 要有 <code>parameters()</code></h3>
<p>训练模型时，优化器需要拿到所有要更新的参数。我们希望调用 <code>model.parameters()</code> 就能得到类似这样的列表：</p>
<div class="lesson-tree">[Embedding 的 <code>weight</code>, Linear 1 的 <code>W</code>, Linear 1 的 <code>b</code>, ...]</div>
<p>问题是，参数可能藏在子模块里，子模块也可能放在列表、元组或字典里。所以 <code>parameters()</code> 要从当前模块出发，逐层查找：</p>
<ul><li>遇到 <code>Parameter</code> → 收集</li><li>遇到 <code>Module</code> → 继续检查它的公开属性</li><li>遇到容器 → 继续检查容器中的元素</li></ul>
<p>当前项目约定：名字以下划线开头的模块属性用于缓存，不参与参数收集。</p>
<h3>4. 为什么要给共享参数去重</h3>
<p>同一个 <code>Parameter</code> 可以被多个地方引用。例如：</p><div class="lesson-tree">模型<br>├── 输入词嵌入 ──┐<br>└── 输出层　　──┴── 指向同一个 <code>Parameter</code></div>
<p>这是<strong>同一个对象被使用了两次</strong>，不是两个数值恰好相同的对象。</p>
<p><code>parameters()</code> 应只返回它一次：两处使用产生的梯度都累加到同一个 <code>grad</code> 中，然后优化器根据汇总的梯度更新一次。如果列表里出现两次，优化器就可能把同一个参数更新两次。</p>
<p>本项目按<strong>对象身份</strong>去重，而不是按参数数值去重：两个独立参数即使数值一样，也仍然是两个参数。</p>
<h3>5. Module 的 <code>zero_grad()</code> 做什么</h3>
<p>既然 <code>parameters()</code> 能找到整个模型的参数，<code>model.zero_grad()</code> 就可以遍历这个列表，依次调用每个参数的 <code>zero_grad()</code>。</p>
<p>一次常见的训练步骤可以这样理解：</p><div class="lesson-tree"><code>model.zero_grad()</code> 清除上一步留下的梯度<br>↓<br><code>forward</code> 用当前 <code>data</code> 计算预测<br>↓<br>计算损失<br>↓<br><code>backward</code> 将梯度累加到各参数的 <code>grad</code><br>↓<br>优化器更新 根据 <code>grad</code> 更新各参数的 <code>data</code></div>
<p>这里用“累加”是因为同一个参数可能在一次计算中被多处使用；有时也会有意累积多个 batch 的梯度。只有准备开始一轮新的梯度累积时，才需要清零。</p>
<p>所以这一节的四个方法连起来就是：</p><ul><li><code>Parameter.__init__</code> → 给每个参数准备 <code>data</code> 和 <code>grad</code></li><li><code>Parameter.zero_grad</code> → 清除一个参数的梯度</li><li><code>Module.parameters</code> → 找到模块里的所有参数，共享参数只返回一次</li><li><code>Module.zero_grad</code> → 清除模块里所有参数的梯度</li></ul>
<h3>补充：如何用 DFS 查找参数</h3>
<p>这是深度优先遍历：从当前 <code>Module</code> 出发，沿属性和容器逐层往下找；遇到 <code>Parameter</code> 就收集，然后返回继续搜索其他分支，直到所有分支都处理完。</p>
<div class="lesson-tree"><code>model</code>（Module）<br>├─ <code>weight</code> → Parameter A<br>├─ <code>blocks</code> → 列表<br>│　├─ 第 0 项（Module）→ <code>W</code>: Parameter B，<code>b</code>: Parameter C<br>│　└─ 第 1 项（Module）→ <code>W</code>: Parameter A（同一对象）<br>└─ <code>_cache</code> → 缓存</div>
<ol><li>从 <code>model</code> 开始，检查它的公开属性。</li><li>遇到 <code>weight</code>，它是 Parameter A，收集。</li><li>遇到列表 <code>blocks</code>，进入列表，依次检查元素。</li><li>进入第一个子模块，收集 B 和 C。</li><li>进入第二个子模块，又遇到 A。它已经收集过，跳过。</li><li><code>_cache</code> 以下划线开头，按本项目约定不检查。</li></ol>
<p>最终返回 [Parameter A, Parameter B, Parameter C]。<code>vars(value).items()</code> 提供对象属性；<code>id(value)</code> 用来识别同一个对象。模块与容器循环引用需要额外记录已访问对象，本任务不包含这种结构。</p>
''' + '<details class="lesson-code"><summary>展开 DFS 代码示例</summary><pre>' + html.escape('''def parameters(self):
    found = []
    seen = set()

    def visit(value):
        if isinstance(value, Parameter):
            if id(value) not in seen:
                seen.add(id(value))
                found.append(value)
        elif isinstance(value, Module):
            for name, child in vars(value).items():
                if not name.startswith("_"):
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    visit(self)
    return found''') + '</pre></details></div>'
