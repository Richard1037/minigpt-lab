"""基础 Linear 初始化：实际代码交付与个人依赖接线。"""
import copy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REFERENCE='''def initialize(shape, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    return rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])
'''
STARTER='''import numpy as np


def initialize(shape, rng=None):
    # rng=None 表示调用方没有传生成器，需在这里创建 default_rng(0)。
    # TODO：生成 shape 对应的标准正态数组，按输入宽度缩放并返回。
    raise NotImplementedError("请实现基础 Linear 权重初始化")


# 直接运行文件时检查当前实现；被 Linear 等模块导入时跳过测评。
if __name__ == "__main__":
    from _initialization_checks import run_local
    run_local(initialize)
'''


def matrix_tex(a):
    return r'\begin{bmatrix}'+r' \\ '.join(' & '.join(f'{v:.5f}' for v in row) for row in a)+r'\end{bmatrix}'


def apply_initialization_lesson(book, math, aligned):
    meta=book['lessons']['initialization']
    if meta['kind']=='concept':
        meta['legacy_assessment']={'kind':'concept','questions':copy.deepcopy(meta['questions'])}
    meta.pop('questions',None)
    meta.update(title='实现 Linear 的权重初始化',kind='initialization',file='initialize_weights.py',
        exports=['initialize'],starter=STARTER,reference=REFERENCE,dependencies=[],assessment_revision=3,
        goal='生成一份初始权重，交给后续 Linear 层使用。',steps=[],derivation='',constructor_names=[])
    target=ROOT/'exercises'/meta['file']
    if not target.exists():target.write_text(STARTER,encoding='utf-8')
    # 目录接口会实时读取个人文件；内容重建不把学习者的新答案写入课程快照。
    meta.setdefault('local_code', target.read_text(encoding='utf-8'))
    cases=[]
    configs=[
        ('C=4、D=3：按给定 RNG 生成完整权重',dict(seed=7,shape=[4,3])),
        ('改变输入与输出宽度：C=3、D=5',dict(seed=19,shape=[3,5])),
        ('非默认种子：C=5、D=2',dict(seed=2026,shape=[5,2])),
        ('同一个 RNG 连续初始化两份不同形状权重',dict(seed=31,shapes=[[4,3],[3,2]])),
        ('输入宽度为 1 的边界',dict(seed=11,shape=[1,4])),
        ('省略或传 None：默认调用均可复现',dict(seed=0,shapes=[[2,3],[2,3]],default_rng=True)),
    ]
    for label,config in configs:
        rng=np.random.default_rng(config['seed']);expected={}
        for i,shape in enumerate(config.get('shapes',[config.get('shape')])):
            generator=np.random.default_rng(0) if config.get('default_rng') else rng
            expected['W' if i==0 else f'W{i+1}']=generator.normal(0,np.sqrt(2/shape[0]),tuple(shape)).tolist()
        cases.append(dict(label=label,input=config,expected=expected,category='权重初始化与随机状态'))
    meta['cases']=cases
    z=np.random.default_rng(7).standard_normal((4,3));w=z*np.sqrt(2/4)
    meta['body']='''<div class="lesson-explanation">
<p>这一节做的事是：<strong>在创建 Linear 层时，为它生成一份初始权重数组。</strong>上一节的 <code>Parameter</code> 负责保存参数，这一节实现产生初始数值的函数。后续 Linear 会用它创建 <code>W.data</code>。</p>
<h3>1. 我们要准备什么</h3>
<p>假设每个位置有 <code>C=4</code> 个输入特征，要得到 <code>D=3</code> 个输出特征。权重 <code>W</code> 的形状就是 <code>(4, 3)</code>：每一列为一个输出特征提供 4 个权重。</p>
<p>函数接收形状 <code>shape=(C,D)</code>，返回这一份浮点权重。偏置由 Linear 的构造器另行创建；本函数的返回值是 <code>W</code>。</p>
<h3>2. 先采样，再按输入宽度缩放</h3>
<p>把 <code>C</code> 称为 <strong>fan-in</strong>，意思是一个输出连接了多少个输入。对于权重 <code>(C,D)</code>，它就是 <code>shape[0]</code>。</p>
<ol><li>用传入的随机数生成器 <code>rng</code>，生成形状为 <code>shape</code> 的标准正态样本 <code>Z</code>。标准正态的均值为 0、标准差为 1，样本中可有正数和负数。</li><li>根据输入宽度 <code>C</code> 计算缩放系数 σ。</li><li>把 <code>Z</code> 的每个元素乘以 σ，得到同形的 <code>W</code> 并返回。</li></ol>
'''+aligned(r'Z_{i,j}&\sim\mathcal N(0,1)',r'\sigma&=\sqrt{\frac{2}{C}}',r'W_{i,j}&=\sigma Z_{i,j}')+'''
<p>σ 是初始化后每个权重所服从的分布的标准差。乘同一个系数会调整数值大小，保留每个位置不同的随机值。</p>
<h3>3. 沿同一组样例走一遍</h3>
<p>本页样例使用 <code>shape=(4,3)</code>，调用方先用种子 7 创建 <code>rng</code>，再把它传给你的函数。</p>
'''+math(r'\sigma=\sqrt{\frac{2}{4}}\approx0.70710678')+'''
<p>生成器采到的完整样本矩阵如下；显示保留五位小数，测评使用原始精度：</p>'''+math(r'Z\simeq'+matrix_tex(z))+'''
<p>每个元素都乘以同一个缩放系数，矩阵形状保持不变：</p>'''+math(r'Z\xrightarrow{\ \times\sqrt{2/4}\ }W,\qquad Z,W\in\mathbb R^{4\times3}')+math(r'W\simeq'+matrix_tex(w))+'''
<p>以第一行第二列为例（代码坐标从 0 开始）：</p>'''+math(r'W_{0,1}=\sqrt{\frac{2}{4}}\,Z_{0,1}\approx0.21124')+'''
<p>这是下方题目第一项公开测评的同一组输入和输出。不要手填这组数：改变形状或 RNG 后，应按同一计算过程得到新的数组。</p>
<h3>4. 随机数生成器由谁创建</h3>
<p><code>rng</code> 是一个 Generator 对象，不是种子数字。调用方传入它时，函数直接从这个对象继续采样。连续调用两次，就使用同一段随机序列中的前一份、后一份，不能在函数内重新播种。</p>
<p>为了兼容后续 Linear 的默认调用，接口允许 <code>rng=None</code>：只在没有传入生成器时，创建一个局部 <code>np.random.default_rng(0)</code>。相同默认调用可复现。</p>
<h3>5. 写代码时用到的操作</h3>
<ul><li><code>shape[0]</code> 取输入宽度 <code>C</code>；<code>shape[1]</code> 是输出宽度 <code>D</code>。</li><li><code>rng.standard_normal(shape)</code> 生成指定形状的标准正态 NumPy 数组，默认类型为 <code>float64</code>。</li><li><code>np.sqrt</code> 求平方根；NumPy 数组乘一个数，会让全部元素分别乘以它。</li></ul>
<p>后续构造过程是：调用本函数得到 <code>W</code> → 用 <code>Parameter</code> 保存 <code>W</code> → Linear 前向读取这份权重计算输出。初始化只在创建参数时做；同一层之后的多次前向继续使用已经保存的权重。</p>
</div>'''
    meta['extension']='<h3>为什么输入越宽，初始权重尺度越小</h3><p>分析单个输出的方差时，假设输入分量独立、均值为 0；权重独立、均值为 0，并与输入独立；各输入方差相同、各权重方差相同，偏置为 0。</p>'+aligned(r'Y_j&=\sum_iX_iW_{i,j}',r'\operatorname{Var}(Y_j)&=\sum_i\operatorname{Var}(X_iW_{i,j})',r'&=C\operatorname{Var}(X_i)\operatorname{Var}(W_{i,j})')+'<p>如果权重尺度不变，C 增加会放大输出方差，因此将权重方差与 C 反向缩放。系数 2 来自 ReLU 保留约一半二阶矩的近似补偿；它不是 GELU 的精确方差恒等式。更复杂模型的配置到对应模型组装任务再讲。</p>'
    meta['task_brief']={'scope':'实现 initialize(shape, rng=None)','variables':[{'name':'shape','meaning':'权重的输入宽度和输出宽度','shape':'二元整数元组 (C,D)，C,D≥1'},{'name':'rng','meaning':'调用方已有的随机数生成器；None 时创建局部 default_rng(0)','shape':'np.random.Generator 或 None'}], 'operation':'按 shape 采标准正态数组，再乘以输入宽度对应的缩放系数。','output':'NumPy float64 数组 W，形状 (C,D)。','conditions':['使用传入 RNG 的当前状态，按次序采样。'],'flow':['shape/rng → Z → 缩放 → W'],'dimensions':{},'next_use':{'lesson':'layer_linear','mode':'reuse'}}
    meta['learning_page']={'role':'miniGPT 基础模块','purpose':'实现后续 Linear 实际调用的初始化函数；从生成参数数值开始，接到 Parameter 和 Linear。','problem_title':'生成 Linear 的初始权重','statement':'''<h3>题目描述</h3><p>实现 <code>initialize(shape, rng=None)</code>，生成形状为 <code>shape</code> 的标准正态数组，再按输入宽度缩放，返回权重。</p><h3>实现接口</h3><p><code>initialize(shape, rng=None)</code></p><h3>输入格式</h3><ul><li><code>shape=(C,D)</code>：两个正整数，分别为输入宽度、输出宽度。</li><li><code>rng</code>：已有的 <code>np.random.Generator</code>。省略或传 <code>None</code> 时，使用局部 <code>np.random.default_rng(0)</code>。</li></ul><h3>计算要求</h3>'''+math(r'W=Z\sqrt{\frac{2}{C}},\quad Z_{i,j}\sim\mathcal N(0,1)')+'''<h3>输出格式</h3><p>返回 <code>np.ndarray</code>，形状为 <code>(C,D)</code>，数据类型为 <code>float64</code>。本题所有输入均满足 <code>C,D≥1</code>。</p><h3>状态变化</h3><p>传入 RNG 时，应通过该生成器的一次同形采样得到 Z，RNG 状态相应推进；后续调用接着采样。返回新数组。</p>''','sample_note':'<p>测评器用 seed 创建 RNG，shape 作为第一个参数传入 initialize。连续调用用例将同一个 RNG 按 shapes 的顺序传给函数，W、W2 分别对应两次返回值；省略 rng 的用例由函数按默认约定创建生成器。数组显示可能舍入，判断使用完整精度，绝对容差为 10⁻¹²。</p>'}
    meta.pop('formula_fragments',None)
    # 数学对象使用公式；函数名、属性名和 shape 等程序接口保持代码样式。
    meta['body']=meta['body'].replace('σ',math(r'\sigma',inline=True))
    meta['learning_page']['statement']=meta['learning_page']['statement'].replace('<code>C,D≥1</code>',math(r'C,D\ge1',inline=True))
    meta['learning_page']['sample_note']=meta['learning_page']['sample_note'].replace('10⁻¹²',math(r'10^{-12}',inline=True))
    # 构造器保持原值；只接通确实需要初始化的依赖及其已有使用方。
    requiring={'layer_linear','rebuild_linear'}
    for stage in book['stages']:
        for module in stage['modules']:
            for name in module['tasks']:
                other=book['lessons'][name]
                if name in requiring or any(d in requiring for d in other.get('dependencies',[])):
                    requiring.add(name)
                    deps=[d for d in other.get('dependencies',[]) if d!='initialization']
                    deps.insert(1 if deps and deps[0]=='infrastructure' else 0,'initialization')
                    other['dependencies']=deps
                    if other.get('experiment_dependencies'):
                        deps=[d for d in other['experiment_dependencies'] if d!='initialization'];deps.insert(1 if deps and deps[0]=='infrastructure' else 0,'initialization');other['experiment_dependencies']=deps
    return book
