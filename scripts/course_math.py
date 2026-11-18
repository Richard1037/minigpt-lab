"""课程公式的持久排版源；只整理讲义 HTML，不改代码、用例或个人文件。"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:sys.path.insert(0,str(ROOT/'scripts'))
from concise_course import apply_concise


def gradient_notation(tex):
    tex=re.sub(r'\\mathrm\{d([A-Zwbxqkv])\}',r'\\nabla \1',tex)
    tex=re.sub(r'\\mathrm\s*(?:\{d\}|d)\s*([A-Zwbxqkv])',r'\\nabla \1',tex)
    tex=re.sub(r'\\nabla_\{?([A-Zwbxqkv])\}?\s*L',r'\\nabla \1',tex)
    return re.sub(r'(?<![A-Za-z])d([A-Zwbxqkv])(?=[^A-Za-z]|$)',r'\\nabla \1',tex)


def math(tex, inline=False):
    tex=gradient_notation(tex)
    tag = 'span' if inline else 'div'
    return f'<{tag} class="math-{"inline" if inline else "block"}" data-tex="{html.escape(tex, quote=True)}">{html.escape(tex)}</{tag}>'


def p(text):
    return '<p>' + text + '</p>'


def aligned(*lines):
    return math(r'\begin{aligned}' + r' \\ '.join(lines) + r'\end{aligned}')


def apply_math(book):
    """可重复应用；教材重建后仍生成同一套正式公式。"""
    lessons = book['lessons']
    edits = {}
    extras = {}

    def section(id, field, title, content):
        edits.setdefault(id, {}).setdefault(field, {})[title] = content

    def body(id, title, content):
        section(id, 'body', title, content)

    def deriv(id, title, content):
        section(id, 'derivation', title, content)

    deriv('axes', '计算关系', math(r'y_n=\sum_{c=0}^{C-1}X_{n,c}') + p('axis=1 指定特征轴被归约。输入（N,C），输出（N，）；保留样本轴。'))
    deriv('broadcast', '计算关系', math(r'Y_{n,c}=X_{n,c}+b_c') + p('从末尾维度对齐：（N,C）与（C，）在特征轴匹配。每个特征有自己的偏置 b[c]，同一组 b 被各条样本复用。广播不要求先复制 N 份偏置。') + p('相同规则也适用于轴长为 1 的情况：将（N,1）与（N,C）对齐时，每行的单个值沿该行的 C 个特征复用。'))
    deriv('matmul', '计算关系', math(r'Y_{n,j}=\sum_{i=0}^{C-1}X_{n,i}W_{i,j}') + p('X 的第 n 行和 W 的第 j 列沿输入特征 i 乘加。形状（N,C）乘（C,D），得到（N,D）。'))
    body('reshape', '元素不变', math(r'X_{b,t,c}=(X_2)_{n,c},\qquad n=bT+t') + p('（B,T,C）→（B×T,C）。-1 让 NumPy 根据元素总数推算轴长。按 b 再按 t 排行，同一 token 的特征仍在同一行；不交换轴。'))
    body('transpose', '轴重排', math(r'Y_{b,c,t}=X_{b,t,c}') + p('本任务轴排列为 (0,2,1)。新轴依次取原来的 batch、特征、token。三维 X.T 会逆序全部轴，不能当作仅转置最后两轴。'))
    body('t_attention', '掩码', aligned(r'S&=\frac{QK^{\mathsf T}}{\sqrt d}', r'\widetilde S_{b,h,t,s}&=\begin{cases}S_{b,h,t,s}&s\le t\\-\infty&s>t\end{cases}') + p('最后两轴分别是 query 与 key token。因果模式在 Softmax 前屏蔽未来位置，每行保留自己；不能用 0 代替 −∞。非因果模式不应用此掩码。'))
    deriv('t_attention', '前向', aligned(r'A&=\operatorname{softmax}_{s}(\widetilde S)', r'Y&=AV') + p('每个 batch/head 独立计算。点积有 d 项；分量独立且方差约为 1 时，其方差随 d 增长，除以 √d 控制尺度。'))
    deriv('t_attention', '从输出逆序推导', aligned(r'\mathrm dV&=A^{\mathsf T}G', r'\mathrm dA&=GV^{\mathsf T}', r'\mathrm dS&=A\odot\big[\mathrm dA-\operatorname{sum}_{s}(A\odot\mathrm dA)\big]') + p('沿 key 轴 s 求和并保留该轴为 1。因果掩码位置的概率和 dS 都为 0；每个 batch/head 保留。'))
    deriv('t_attention', '回到 Q/K', aligned(r'\mathrm dQ&=\frac{\mathrm dS\,K}{\sqrt d}', r'\mathrm dK&=\frac{\mathrm dS^{\mathsf T}Q}{\sqrt d}') + p('转置只交换最后两轴。每条路径保留全部 batch/head，不用第一张切片代替完整张量。'))
    body('t_block', '模块', aligned(r'X_1&=X+\operatorname{Attention}(\operatorname{Norm}_1(X))', r'Y&=X_1+\operatorname{MLP}(\operatorname{Norm}_2(X_1))') + p('Block(C,H,hidden) 持有 norm1、attn、norm2、mlp。输入输出（B,T,C）。两次归一化用独立实例，避免覆盖前一次前向缓存。'))
    deriv('t_block', '第二条残差', aligned(r'F_2&=\operatorname{MLP}\circ\operatorname{Norm}_2', r'G_1&=G+J_{F_2}(X_1)^{\mathsf T}G') + p('G₁=G+norm2.backward(mlp.backward(G))。J 表示子网络的雅可比；实际实现通过已有 backward 连续计算，不构造它。'))
    deriv('t_block', '第一条残差', aligned(r'F_1&=\operatorname{Attention}\circ\operatorname{Norm}_1', r'\mathrm dX&=G_1+J_{F_1}(X)^{\mathsf T}G_1') + p('dX=G₁+norm1.backward(attn.backward(G₁))。直连与子网络分别接收完整上游梯度，回到同一输入时相加。参数梯度由各子层累加。'))
    deriv('t_dropout', '倒置 Dropout', aligned(r'q_i&\sim\operatorname{Bernoulli}(1-p),\qquad m_i=\frac{q_i}{1-p}', r'Y&=X\odot m,\qquad\mathbb E[Y\mid X]=X', r'\mathrm dX&=G\odot m') + p('训练反向复用当前前向的同一掩码；评估时 m=1，不再缩放，也不推进随机状态。'))
    deriv('t_layernorm', '前向', aligned(r'\mu&=\operatorname{mean}_j(x_j)', r'r&=\big(\operatorname{mean}_j[(x_j-\mu)^2]+\epsilon\big)^{-1/2}', r'h&=(x-\mu)r,\qquad y=h\odot w+b'))
    deriv('t_layernorm', '反向', aligned(r'u&=G\odot w', r'\mathrm dX&=r\big[u-\operatorname{mean}_j(u)-h\odot\operatorname{mean}_j(uh)\big]', r'\mathrm dw_c&=\sum_{b,t}G_{b,t,c}h_{b,t,c}', r'\mathrm db_c&=\sum_{b,t}G_{b,t,c}') + p('均值只沿特征轴；参数梯度沿 batch/token 归约并累加。'))
    deriv('t_layernorm', '位置公式对照', aligned(r'\operatorname{PE}(p,2i)&=\sin\left(\frac p{10000^{2i/C}}\right)', r'\operatorname{PE}(p,2i+1)&=\cos\left(\frac p{10000^{2i/C}}\right)') + p('p 是位置编号。正弦位置编码没有可训练参数；本项目主线的位置表通过位置编号查表，反向汇总所有 batch 的位置梯度。'))
    body('t_gpt','输入输出',math(r'X_{b,t,c}=E_{\mathrm{ids}_{b,t},c}+P_{t,c}')+p('ids 为（B,T）整数，T≤context。token 与位置查表相加得到（B,T,C），经模型输出 logits（B,T,V）。backward(G) 返回 None，同时累加全部参数梯度。'))
    body('t_gpt','缓存与初始化',math(r'W_{\mathrm{2D}}\sim\mathcal N(0,0.02^2),\qquad W_{\mathrm{exit}}\leftarrow\frac{W_{\mathrm{exit}}}{\sqrt{2L}}')+p('同一 seed 的单一随机生成器依次初始化二维权重，L=layers。仅各残差分支的出口投影缩放一次。各 Block 独立保存本次前向缓存，供紧接着的一次 backward。'))
    deriv('t_gpt', '共享头反向', aligned(r'\mathrm{logits}&=H_{\mathrm{out}}E^{\mathsf T}', r'\mathrm dE_{\mathrm{head}}&=G_2^{\mathsf T}H_{\mathrm{out},2}', r'\mathrm dH_{\mathrm{out}}&=GE', r'\mathrm dE&=\mathrm dE_{\mathrm{head}}+\mathrm dE_{\mathrm{lookup}}') + p('H_out 是末尾激活（B,T,C），E 是词表（V,C）；下标 2 表示展平全部位置。经 Norm 与反序 Blocks 回流后，再散射到同一词表的 grad。'))
    deriv('t_gpt', '位置反向', math(r'\mathrm dP_{t,c}=\sum_{b=0}^{B-1}\mathrm dX_{b,t,c}') + p('P 是位置表。相同位置向量被 B 条序列复用，先沿 batch 求和，再按位置编号累加。整数 ids 不求导；共享参数在 parameters() 中只出现一次。'))
    deriv('t_optim', '裁剪', aligned(r'n&=\sqrt{\sum_{\theta}\sum_i g_{\theta,i}^{\,2}}', r's&=\min\left(1,\frac{\mathrm{limit}}{n+10^{-12}}\right)', r'g_{\theta,i}&\leftarrow s\,g_{\theta,i}') + p('所有参数共用一个缩放比例，保持整体梯度方向；不能逐张量分别裁剪。'))
    deriv('t_optim', 'AdamW', aligned(r'm_t&=\beta_1m_{t-1}+(1-\beta_1)g_t', r'v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2', r'\widehat m_t&=\frac{m_t}{1-\beta_1^t},\qquad\widehat v_t=\frac{v_t}{1-\beta_2^t}', r'\theta_t&=(1-\eta\lambda)\theta_{t-1}', r'&\quad-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}') + p('步数从 t=1 开始。g²、平方根及除法均逐元素；不参与衰减的参数取 λ=0。衰减与损失梯度分离，step 不清零 grad。'))
    deriv('t_train', '梯度下降与困惑度', math(r'\theta_{t+1}=\operatorname{Optimizer}(\theta_t,\nabla_{\theta_t}L),\qquad\operatorname{PPL}=e^L') + p('固定小批次 loss 应能下降，但不能单靠下降排除因果泄漏。困惑度只在相同 tokenizer、数据和平均方式下可比较。'))
    deriv('t_train', '学习率扩展', aligned(r'\eta_{\mathrm{warmup}}&=\eta_{\max}\frac{\mathrm{step}}{\mathrm{warmup}}', r'\eta_{\mathrm{cosine}}&=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})', r'&\quad\cdot(1+\cos(\pi u))') + p('u 是衰减阶段的完成比例。先用固定学习率建立基线，再加入调度。'))
    deriv('t_sample', '概率', math(r'p_i=\frac{\exp(z_i/\tau)}{\sum_{j\in\mathcal K}\exp(z_j/\tau)},\qquad i\in\mathcal K') + p('𝒦 为保留的类别集合：top_k=0 时是全部类别，否则是分数最高的 k 类；集合外概率为 0。温度 τ>0，越小越尖锐。k=1 才退化为选定最大类别。'))
    for id, replacements in edits.items():
        if id in ('t_attention','t_block','t_dropout','t_layernorm','t_optim','t_train','t_sample') or (id == 't_gpt' and 'data-gpt-prerequisites' in lessons[id].get('body','')):
            continue  # Part06 专用教学源统一维护正文及就地推导。
        if id not in lessons:
            continue  # 允许仅构建基础章节时复用。
        for field, sections in replacements.items():
            value = lessons[id].get(field, '')
            for title, content in sections.items():
                pattern = r'(<h3>' + re.escape(title) + r'</h3>).*?(?=<h3>|$)'
                if (id,field,title) in {('layer_gelu','body','激活函数对照'),('t_layernorm','derivation','位置公式对照')}:
                    value=re.sub(pattern,'',value,flags=re.S)
                    extras[id]='<h3>'+title+'（拓展）</h3>'+content
                    continue
                value, count = re.subn(pattern, lambda m: m[1] + content, value, flags=re.S)
                if count != 1:
                    raise ValueError(f'公式章节未找到或重复：{id}/{field}/{title}')
            lessons[id][field] = value
    # 独立实现的推导与对应模块保持一致，不重复维护第二份数学公式。
    for independent, base in [('rebuild_gpt', 't_gpt')]:
        if independent in lessons and 'data-gpt-prerequisites' not in lessons[base].get('body',''):
            lessons[independent]['derivation'] = lessons[base]['derivation']
            if base == 't_block':
                original = lessons[independent]['body']
                lessons[independent]['body'] = original.split('<h3>模块</h3>')[0] + lessons[base]['body']
    for id,base in [('fix_shared','t_gpt'),('fix_clear','t_train')]:
        if id in lessons and lessons[base].get('derivation') and 'data-gpt-prerequisites' not in lessons[base].get('body',''):lessons[id]['derivation']=lessons[base]['derivation']
    for id in ('rebuild_gpt','fix_shared'):
        if id in lessons and 'data-gpt-prerequisites' not in lessons['t_gpt'].get('body',''):
            for title in ('输入输出','缓存与初始化'):
                pattern=r'(<h3>'+title+r'</h3>).*?(?=<h3>|$)'
                source=re.search(pattern,lessons['t_gpt']['body'],flags=re.S)[0]
                lessons[id]['body']=re.sub(pattern,lambda _:source,lessons[id]['body'],flags=re.S)
    for id, formula, explanation in [
        ('att_scores', r'S_{b,h,t,s}=\frac1{\sqrt d}\sum_{i=0}^{d-1}Q_{b,h,t,i}K_{b,h,s,i}', '只转置 K 的最后两轴；query 长度 Tq 与 key 长度 Tk 可以不同。'),
        ('att_weighted', r'Y_{b,h,t,j}=\sum_{s=0}^{T_k-1}A_{b,h,t,s}V_{b,h,s,j}', '沿 key 位置 s 汇总内容。value 特征宽度可以与打分特征宽度不同。')]:
        if id in lessons:
            lessons[id]['derivation'] = math(formula) + p(explanation)
    if 'att_scores' in lessons:
        lessons['att_scores']['body'] = '<h3>Q、K、V 的来源与作用</h3>' + p('Q 是每个 query 位置用来寻找信息的向量；K 是每个 key 位置用来匹配的向量；V 是被读取位置提供的内容。完整多头注意力将输入 X 分别投影得到 Q、K、V。') + p('本任务由调用方直接给出 Q 和 K。固定一个 batch 与 head，Q 的一行与 K 的一行点积，得到这两个位置的匹配分数；后续任务将分数变为概率，再读取 V。')
    extension = '<h3>数值误差分析（拓展）</h3>' + p('浮点数不能精确表示所有实数，乘加顺序也影响末位。float64 便于教学核对；训练可按需求使用 float32。不要用逐元素 == 判断一般浮点等价。')
    extension += '<h3>有限差分（拓展）</h3>' + p('Y 是输出，G 是与 Y 同形的上游梯度；固定 G，用它们构造标量 L。θ 是待检查的一个输入或参数元素，h 是施加在 θ 上的微小扰动。') + aligned(r'L&=\sum_iY_iG_i', r'\frac{\partial L}{\partial\theta}&\approx\frac{L(\theta+h)-L(\theta-h)}{2h}') + p('分别将 θ 加上和减去 h，重新计算 L，再用上式近似梯度，与反向传播的结果比较。h 太大有截断误差，太小会因相减丢失有效位。公开测评的配置、容差与坐标选择见具体用例；抽样检查不是对所有输入的数学证明。')
    book['common_extension'] = extension
    for lesson in lessons.values():
        lesson['extension'] = ''
    for id,content in extras.items():lessons[id]['extension']+=content
    lessons['reduce']['body']='<h3>逐行去均值：本节只做前向</h3>'+aligned(r'\mu_{n,0}&=\frac1C\sum_{j=0}^{C-1}X_{n,j}',r'Y_{n,j}&=X_{n,j}-\mu_{n,0}')+p('X 为（N,C），μ 为（N,1）。轴长 1 保留每行均值的位置，使它沿本行 C 个特征广播，得到 Y（N,C）。若去掉 keepdims，μ 变为（N,），它会从末轴与 C 对齐，可能报错，也可能在 N=C 时错误地按列相减。')
    lessons['reduce']['extension']+='<h3>逐行去均值的反向</h3>'+math(r'\mathrm dX_{n,j}=G_{n,j}-\frac1C\sum_kG_{n,k}')+p('G 是损失对去均值结果的上游梯度，dX 是传回输入 X 的梯度。后续去均值反向任务会推导并实现这条路径。')
    book['math_revision'] = 1
    apply_concise(book)
    from learner_layout import apply_learner_layout
    apply_learner_layout(book, math, aligned)
    def normalize_html(value):
        return re.sub(r'<(span|div) class="math-(inline|block)" data-tex="([^"]+)">.*?</\1>',lambda v:math(html.unescape(v[3]),inline=v[2]=='inline'),value,flags=re.S)
    for meta in lessons.values():
        for field in ['body','derivation','extension']:
            if isinstance(meta.get(field),str):meta[field]=normalize_html(meta[field])
        for field,value in meta.get('learning_page',{}).items():
            if isinstance(value,str):meta['learning_page'][field]=normalize_html(value)
        for field,value in meta.get('formula_fragments',{}).items():meta['formula_fragments'][field]=gradient_notation(value)
    return book


if __name__ == '__main__':
    target = ROOT / 'webapp/foundations.json'
    book = json.loads(target.read_text(encoding='utf-8'))
    apply_math(book)
    target.write_text(json.dumps(book, ensure_ascii=False, indent=2), encoding='utf-8')
    # 教学骨架更新后只迁移精确匹配的旧空模板，先备份；个人实现保留。
    from sync_scaffolds import sync_scaffolds
    synced=sync_scaffolds(book,ROOT/'exercises')
    from part06_lessons import sync_part06_imports
    import_sync=sync_part06_imports(book,ROOT/'exercises')
    if synced['updated'] or import_sync:
        target.write_text(json.dumps(book,ensure_ascii=False,indent=2),encoding='utf-8')
    coverage=ROOT/'docs/能力覆盖对照.md'
    if coverage.exists():
        meta=book['lessons']['initialization']
        row=f'| initialization · {meta["title"]} | 实现 initialize(shape, rng=None)，供个人 Linear 使用 | {len(meta["cases"])} 项：数值、类型、形状与随机状态 | layer_linear |'
        content=re.sub(r'^\| initialization ·.*$',lambda _:row,coverage.read_text(encoding='utf-8'),flags=re.M)
        embedding=book['lessons']['layer_embedding']
        embedding_row=f'| layer_embedding · {embedding["title"]} | 实现 Embedding.__init__、forward、backward | {len(embedding["cases"])} 项：查表、梯度、缓存、累加与独立构造 | t_gpt |'
        coverage.write_text(re.sub(r'^\| layer_embedding ·.*$',lambda _:embedding_row,content,flags=re.M),encoding='utf-8')
    print('Typeset definitions and derivations; code and grading data preserved.')
