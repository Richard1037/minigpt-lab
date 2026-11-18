"""Part07的单一接口升级任务；参考只用于显式查看/隔离验证。"""
import ast
import copy
import re
from html.parser import HTMLParser


def format_prose(source, identifiers=()):
    """检查普通文本节点；已有代码和数学节点保持。"""
    class Labels(HTMLParser):
        def __init__(self):super().__init__(convert_charrefs=False);self.out=[];self.skip=[]
        def handle_starttag(self,tag,attrs):
            self.out.append(self.get_starttag_text())
            if tag in ('code','pre') or any(k=='class' and 'math-' in (v or '') for k,v in attrs):self.skip.append(tag)
        def handle_endtag(self,tag):
            self.out.append('</'+tag+'>')
            if self.skip and self.skip[-1]==tag:self.skip.pop()
        def handle_data(self,text):
            if not self.skip:
                names=('MultiHeadAttention','AttentionCore','MLPBranch','Embedding','RMSNorm','Parameter','Linear','Module','Block','MLP','GPT','rng','init_std','init_scale')+tuple(identifiers)
                pattern='|'.join(re.escape(v) for v in sorted(set(names),key=len,reverse=True))
                text=re.sub(r'(?<![A-Za-z0-9_.])('+pattern+r')(?![A-Za-z0-9_])',r'<code>\1</code>',text)
            self.out.append(text)
        def handle_entityref(self,name):self.out.append('&'+name+';')
        def handle_charref(self,name):self.out.append('&#'+name+';')
    parser=Labels();parser.feed(source);return ''.join(parser.out)

MODULES=[('layer_linear','Linear'),('layer_embedding','Embedding'),('layer_mlp','MLP'),('foundation_capstone','MLPBranch'),('t_mha','MultiHeadAttention'),('t_block','Block')]
CONSTRUCTORS={
'Linear':'''def __init__(self, C, D, use_bias=True, *, rng=None, init_std=None, init_scale=1.0):
    if init_std is None:
        values = initialize((C, D), rng)
    else:
        if rng is None: rng = np.random.default_rng(0)
        values = rng.normal(0, init_std, (C, D))
    self.W = Parameter(values * init_scale)
    self.b = Parameter(np.zeros(D)) if use_bias else None
''',
'Embedding':'''def __init__(self, V, C, *, rng=None, init_std=0.02):
    if rng is None: rng = np.random.default_rng(0)
    self.weight = Parameter(rng.normal(0, init_std, (V, C)))
''',
'MLP':'''def __init__(self, C, hidden, *, rng=None, init_std=None, init_scale=1.0):
    self.fc1 = Linear(C, hidden, rng=rng, init_std=init_std)
    self.act = GELU()
    self.fc2 = Linear(hidden, C, rng=rng, init_std=init_std, init_scale=init_scale)
''',
'MLPBranch':'''def __init__(self, C, hidden, eps=1e-5, *, rng=None, init_std=None, init_scale=1.0):
    self.norm = RMSNorm(C, eps)
    self.mlp = MLP(C, hidden, rng=rng, init_std=init_std, init_scale=init_scale)
''',
'MultiHeadAttention':'''def __init__(self, C, H, *, rng=None, init_std=None, init_scale=1.0):
    if C % H: raise ValueError('C 必须能被 H 整除')
    self.C, self.H = C, H
    self.qkv = Linear(C, 3*C, rng=rng, init_std=init_std)
    self.core = AttentionCore(True)
    self.proj = Linear(C, C, rng=rng, init_std=init_std, init_scale=init_scale)
''',
'Block':'''def __init__(self, C, H, hidden, *, rng=None, init_std=None, init_scale=1.0):
    self.norm1 = RMSNorm(C)
    self.attn = MultiHeadAttention(C, H, rng=rng, init_std=init_std, init_scale=init_scale)
    self.ffn = MLPBranch(C, hidden, rng=rng, init_std=init_std, init_scale=init_scale)
'''}


def apply_upgrade(book,math,aligned):
    lessons=book['lessons']
    code=[]
    for key,name in MODULES:
        tree=ast.parse(lessons[key]['reference'])
        for cls in tree.body:
            if isinstance(cls,ast.ClassDef) and cls.name==name:
                cls=copy.deepcopy(cls)
                cls.body=[ast.parse(CONSTRUCTORS[name]).body[0] if isinstance(n,ast.FunctionDef) and n.name=='__init__' else n for n in cls.body]
                source=ast.unparse(ast.fix_missing_locations(cls))
                if name=='Block':source=source.replace('self._ffn','self.ffn')
                code.append(source)
    # 检查入口只导入个人模块，不包含构造器替换或计算实现。
    driver='import numpy as np\nfrom learner_core import Parameter, Module, initialize\nif __name__ != "student":\n'+''.join('    from '+lessons[key]['file'][:-3]+' import '+name+'\n' for key,name in MODULES)+'    from layer_rmsnorm import RMSNorm\n    from layer_gelu import GELU\n    from t_attention import AttentionCore\n\nif __name__ == "__main__":\n    from _transformer_checks import run_local\n    run_local("t_init_upgrade", globals())\n'
    p=lambda x:'<p>'+x+'</p>'
    h=lambda t,x:'<h3>'+t+'</h3>'+x
    m=lambda x:math(x,inline=True)
    rows=''.join('<tr><td><code>'+lessons[k]['file']+'</code></td><td><code>'+n+'</code></td><td>'+('实际采样一次，创建自己的参数' if n in ('Linear','Embedding') else '创建子层时传递配置；只将出口缩放交给分支最后一层')+'</td></tr>' for k,n in MODULES)
    body=h('先看整体：为什么现在完善构造接口',p('基础层已能独立工作。组装完整 GPT 时，两张表和所有投影需要共用一个随机数流，并让两个残差出口更小。现在为已有模块增加可选配置，让参数在创建时就得到正确初值；GPT 只提供配置、组装组件。')+p('本任务修改原文件的构造器，复用已有前后向与缓存。不是再写一套层，也不是让 GPT 拆开子层重新赋值。'))
    body+=h('配置的来源：从 GPT 一直传到创建参数的层',p('<code>rng</code> 是 GPT 用 <code>np.random.default_rng(seed)</code> 创建的生成器对象，各层传递同一对象。<code>init_std</code> 是指定标准差；<code>init_scale</code> 是只在初始化时乘的系数。')+aligned(r'\sigma=0.02,\qquad s_{\mathrm{exit}}=\frac{1}{\sqrt{2L}}',r'W\leftarrow\operatorname{Normal}(0,\sigma,\mathrm{shape}(W))\,s')+p(m('L')+' 是完整模型的 Block 数；普通权重取 '+m('s=1')+'，残差出口取 '+m(r's=s_{\mathrm{exit}}')+'。各组件无需知道总层数，只接收计算好的系数。'))
    body+=h('接口约定 · 谁负责什么','<table><thead><tr><th>原文件</th><th>类</th><th>本次职责</th></tr></thead><tbody>'+rows+'</tbody></table>'+p('<code>Linear</code> 保留 <code>use_bias=True</code>，增加仅关键字参数 <code>rng=None, init_std=None, init_scale=1.0</code>。未指定标准差时调用已有 <code>initialize(shape, rng)</code>；指定时直接采样，避免先 fan-in 初始化再重采样。偏置仍为零。')+p('<code>Embedding</code> 增加 <code>rng=None, init_std=0.02</code>。无生成器时沿用固定种子零的旧行为；有生成器时直接使用传入对象。')+p('<code>MLP</code>、<code>MLPBranch</code>、<code>MultiHeadAttention</code>、<code>Block</code> 都增加仅关键字参数 <code>rng=None, init_std=None, init_scale=1.0</code>。<code>MLPBranch</code> 原有 <code>eps</code> 位置与默认值保持。')+p('<code>MLP</code> 只让第二层接收出口系数；注意力只让输出投影接收出口系数。<code>Block</code> 将系数交给两个分支。各模块的内部属性名沿用你的实现。'))
    body+=h('取数顺序 · 一个生成器连续使用',p('GPT 先创建字符表、位置表，再按层创建 Block。每个 Block 先创建注意力，再创建 MLP 分支；各分支内部按前向顺序创建两个投影。归一化与零偏置不消耗随机数。')+p('顺序是：字符表 → 位置表 → 每层的注意力输入投影、注意力输出投影、MLP 第一层、MLP 第二层。你当前 Block 若先创建 MLP 分支，需要把注意力创建放在它前面；前向运算顺序保持。')+p('例如 '+m('L=2')+' 时出口系数为 '+m(r'\frac{1}{2}')+'。普通矩阵采样后原样保存，出口矩阵采样后乘该系数。每张矩阵只采样一次；同形状的不同权重从随机流中取不同片段。'))
    signatures=''.join('<details><summary><code>'+name+'</code> · 构造接口提示</summary><pre>'+CONSTRUCTORS[name].split('\n')[0]+'\n    # TODO：在原构造器中补配置处理，保留已有前后向方法。\n</pre></details>' for _,name in MODULES)
    body+=h('动手前 · 怎样在原构造器上补参数',p('参数表中的 <code>*</code> 表示它后面的参数必须按名字传入，例如 <code>rng=rng</code>。旧位置参数仍然可用。下面只列构造签名；在原文件增量修改，已有方法继续使用。')+signatures)
    body+=h('核心总结：配置向下传，参数由所属模块创建',math(r'\mathrm{GPT}\longrightarrow\mathrm{Block}\longrightarrow\{\mathrm{Attention},\mathrm{MLPBranch}\}\longrightarrow\mathrm{Linear}')+p('两张表由 Embedding 自行初始化；投影由 Linear 自行初始化；组合模块负责传递配置与标记出口。已有默认调用仍可运行，完整 GPT 得到指定初值，外层不再寻找内部权重。'))
    actions='<ol>'+''.join('<li>'+v+'</li>' for v in ['在 <code>Linear</code> 和 <code>Embedding</code> 构造器补上上述可选参数；每个权重只走一条初始化分支。','在 <code>MLP</code>、<code>MLPBranch</code>、<code>MultiHeadAttention</code> 中向下传递同一生成器和标准差，并只缩放指定出口。','在 <code>Block</code> 构造器接收配置，先创建注意力、再创建 MLP 分支；保留个人内部命名。','在下方编辑区切换六个原文件并完成修改，再运行本节接口检查。前后向方法不清空、不复制；下一题在原 <code>t_gpt.py</code> 中完成 GPT。'])+'</ol>'
    statement=h('题目描述',p('增量完善上表六个个人文件中的构造器。第七个文件 <code>t_gpt.py</code> 留给下一题组装；本节检查程序只导入和调用你的模块。'))+h('实现接口',p('使用上方列出的可选关键字参数；类的原有位置参数与默认调用保持。'))+h('输入格式',p('旧构造参数，以及可选随机数生成器、标准差、初始化缩放系数。有效标准差非负，缩放系数为实数。'))+h('输出格式',p('构造得到的个人模块。初始参数、同形零梯度及前后向接口均有效；检查入口返回逐项结果。'))+h('状态变化',p('构造消耗传入生成器的连续状态，每张随机权重仅采样一次。前向不重新初始化；参数由个人 Module 收集与清零。'))+h('动手任务清单',actions)+h('关键公式速查',aligned(r'W_{\text{普通}}\sim\mathcal N(0,0.02^2)',r'W_{\text{出口}}=\frac{W_{\text{采样}}}{\sqrt{2L}}',r'\text{旧 Linear 默认：}\quad\sigma=\sqrt{2/C}'))
    meta=dict(id='t_init_upgrade',title='完善初始化接口：让已有模块支持完整 GPT',goal='升级已有构造接口，集中检查个人文件。',kind='project',file='t_init_upgrade.py',exports=[n for _,n in MODULES],dependencies=list(dict.fromkeys(['infrastructure','initialization','layer_embedding',*lessons['t_block']['dependencies'],'t_block'])),integration_only=True,integration_files=[key for key,_ in MODULES],body=body,definition='',derivation='',steps=[],trace=[],constructor_names=[],starter=driver,local_code=driver,reference='\n\n'.join(code),extension='',hint='',assessment_revision=1,task_brief=dict(scope='升级原模块构造器',variables=[dict(name='rng',meaning='同一个随机数生成器对象',shape='Generator')],operation='构造并检查原模块',output='各项行为结果',conditions=[],next_use=None),learning_page=dict(problem_title='完善初始化接口：让已有模块支持完整 GPT',role='miniGPT 接口升级',purpose='在原有层中增加可选初始化配置，随后复用它们组装 GPT。',statement=statement,sample_input='',sample_note=p('本例权重形状为 '+m('(3,4)')+'，种子为 '+m('11')+'，采样标准差为 '+m('0.07')+'，初始化系数为 '+m('0.5')+'。应只采样一次，再乘系数保存；下一个层继续使用生成器的后续状态。')))
    meta['cases']=[dict(label=label,assessment='initialization_upgrade',input=dict(variant=v),expected=dict(要求=label),category='初始化接口') for v,label in [('legacy','旧默认接口、参数与前向保持'),('linear','Linear 直接初始化、生成器与缩放'),('embedding','Embedding 共用随机流与独立参数'),('chain','Block 全链路传递、仅两出口缩放'),('state','原模块反向、梯度与重复前向状态')]]
    lessons['t_init_upgrade']=meta
    meta['sample_index']=1
    meta['cases'][1]['input'].update(seed=11,shape=[3,4],init_std=.07,init_scale=.5)
    for key,_ in MODULES:lessons[key]['upgrade_followup']='t_init_upgrade'
    meta['body']=format_prose(meta['body'])
    for field in ('statement','sample_note'):meta['learning_page'][field]=format_prose(meta['learning_page'][field])
    return meta
