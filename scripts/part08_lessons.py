"""Part08精修：保留优化器/固定批次训练两题及主接口。"""
import ast
from part45_lessons import scaffold
from initialization_upgrade import format_prose


def optimizer_lesson(math, aligned, p, h, ul, fold):
    m = lambda tex: math(tex, inline=True)
    clip_formula = aligned(
        r'n&=\sqrt{\sum_{p\in\mathcal P}\sum_i R_{p,i}^2}',
        r'a&=\min\!\left(1,\frac{\ell}{n+10^{-12}}\right)',
        r'G_p&=aR_p',
    )
    moments = aligned(
        r'm_t&=\beta_1 m_{t-1}+(1-\beta_1)G',
        r'v_t&=\beta_2 v_{t-1}+(1-\beta_2)G^{\odot 2}',
    )
    correction = aligned(
        r'\widehat m_t&=\frac{m_t}{1-\beta_1^t}',
        r'\widehat v_t&=\frac{v_t}{1-\beta_2^t}',
    )
    update = aligned(
        r'U_t&=\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}',
        r'\theta_t&=(1-\eta\lambda_p)\theta_{t-1}-U_t',
    )

    body = h('梯度怎样变成下一轮用的权重',
        p('上一节的模型用参数做前向，交叉熵给出损失 '+m('L')+'，再由反向把损失对每份参数的梯度写进 <code>p.grad</code>。这里 <code>p</code> 是一个 <code>Parameter</code> 对象。<code>p.data</code> 是模型下一次前向会读的参数值，<code>p.grad</code> 与它同形，记录改变各个参数元素会怎样影响当前损失。反向结束时，当前参数值仍在 <code>p.data</code> 里，更新才刚要开始。')+
        p('本题也是用梯度来更新 <code>data</code>，中间加了两件事。<code>clip_grad</code> 先控制全模型这次梯度的总大小，再由 <code>AdamW.step()</code> 结合当前梯度和过去几步的记录，算出每个参数实际要减去多少。'))

    clipping = (
        p('<code>clip_grad(parameters, limit)</code> 中，<code>parameters</code> 由调用方从 <code>model.parameters()</code> 取得，参数唯一性由 <code>Module.parameters()</code> 保证。裁剪直接遍历这个列表两次。用 '+m(r'\mathcal P')+' 表示按对象身份去重后的参数集合，'+m('p')+' 指其中一份参数，'+m('i')+' 遍历该参数的全部元素。若输入词表和输出头共用同一个对象，这份梯度已在反向合流，求范数和缩放都只处理一次。')+
        p('把参数的当前值记为 '+m(r'\theta_p')+'。记 '+m(r'R_p=\nabla_{\theta_p}L')+' 为 <code>p.grad</code> 中的原始梯度，'+m('n')+' 为全部原始梯度合在一起的二范数，'+m(r'\ell')+' 为调用方给定的上限 <code>limit</code>。先求 '+m('n')+'，再求共同的缩放比例 '+m('a')+'。结果 '+m('G_p')+' 是写回 <code>p.grad</code> 的裁剪后梯度。')+
        clip_formula+
        p('把全模型所有梯度接成一条向量，二范数就是它的长度。超出上限时，统一缩放压短这条向量，同时保持各分量的相对比例。比例上限为 '+m('1')+'，小梯度不会被放大。这里不是给每个层单独求一个系数，也无需真的拼接数组。分母中的 '+m('10^{-12}')+' 防止零范数时除以零。'))

    clipping_example = (
        p('用一个微型模型贯穿下面几步。设它只有二维权重 '+m('W')+' 和一维偏置 '+m('b')+'，初始值和本次反向结果如下。上限取 '+m(r'\ell=1')+'。')+
        aligned(r'W_0&=\begin{bmatrix}1&2\end{bmatrix},\quad b_0=(1)',
                r'R_W&=\begin{bmatrix}3&0\end{bmatrix},\quad R_b=(-4)',
                r'n&=\sqrt{3^2+0^2+(-4)^2}=5,\quad a\approx0.2',
                r'G_W&\approx\begin{bmatrix}0.6&0\end{bmatrix},\quad G_b\approx(-0.8)')+
        p('调用后，两个 <code>grad</code> 里分别存着 '+m('G_W')+' 与 '+m('G_b')+'，两个 <code>data</code> 还是原值。函数返回的是缩放前的范数 '+m('5')+'，便于训练时观察梯度有多大。下一步直接从原参数对象的 <code>grad</code> 读裁剪结果。'))

    updating = (
        p('假设某个权重的梯度为正，表示在当前位置把该权重稍微调大，会使损失的一阶近似变大。最简单的梯度下降用学习率 '+m(r'\eta')+' 乘梯度，再从权重中减去这一步。若当前参数值记为 '+m(r'\theta')+'，这个对照公式是')+
        math(r'\theta_{\mathrm{new}}=\theta-\eta\,\nabla_{\theta}L')+
        p('优化器在训练循环外创建一次。<code>Module.parameters()</code> 已按对象身份收集唯一参数。构造时用 <code>self.params = parameters</code> 保存这个列表，沿用其中的参数引用和顺序，为每份参数建立两个同形的零数组，分别放到 <code>self.m</code> 和 <code>self.v</code>。<code>self.t</code> 从零开始，记录整个优化器已执行多少次更新。每次 <code>step()</code> 先把它加一，然后用同一个步数处理所有参数。')+
        p('下面固定其中一份参数，省去下标 '+m('p')+'。'+m('G')+' 就是它本轮裁剪后的梯度 '+m('G_p')+'，从该参数的 <code>grad</code> 读取。如果代码里把这个值叫作 <code>g</code>，它对照的就是这里的 '+m('G')+'。'+m('t')+' 是本次更新的编号，'+m(r'\theta_{t-1}')+' 是更新前的 <code>data</code>。带下标 '+m('t-1')+' 的历史来自上一步，带下标 '+m('t')+' 的历史写回优化器，留给下一步。')+
        ul([
            '用 '+m('m_t')+' 保存有符号梯度的加权平均。正负梯度会互相抵消，连续几步同方向的梯度会留下更稳定的方向。配置 <code>beta1</code> 对应 '+m(r'\beta_1')+'，它决定保留多少旧记录。',
            '用 '+m('v_t')+' 保存梯度平方的加权平均。平方消除正负号，持续偏大的梯度会使这份记录偏大。配置 <code>beta2</code> 对应 '+m(r'\beta_2')+'，它决定平方记录更新得有多快。'+m(r'G^{\odot2}')+' 表示每个元素分别平方。',
        ])+moments+
        p('两份历史起初都是零，第一轮只吸收了本轮梯度的一部分。例如 '+m(r'm_1=(1-\beta_1)G')+'。到了第 '+m('t')+' 步，累进权重之和分别是 '+m(r'1-\beta_1^t')+' 和 '+m(r'1-\beta_2^t')+'。除去这两个因子，得到用于本次计算的校正值 '+m(r'\widehat m_t,\widehat v_t')+'。优化器留给下一步的是校正前的 '+m('m_t,v_t')+'。')+
        correction+
        p('记 '+m('U_t')+' 为最终要减去的梯度更新量。它用 '+m(r'\widehat m_t')+' 提供方向，用 '+m(r'\sqrt{\widehat v_t}')+' 提供各元素近期梯度的尺度。持续梯度较大的元素也会有较大的分母，因而更新会随各元素自身的梯度历史调节。'+m(r'\eta')+' 来自 <code>lr</code>，控制更新强度。'+m(r'\epsilon')+' 来自 <code>eps</code>，用于分母为零或很小时的数值保护。')+
        p('最后，为每份参数选定有效衰减系数 '+m(r'\lambda_p')+'。本项目的规则是二维及更高维参数取构造配置 <code>decay</code>，一维偏置和归一化缩放等取零。先把旧参数值乘衰减因子，再减去 '+m('U_t')+'，把 '+m(r'\theta_t')+' 原地写回 <code>data</code>。权重衰减直接作用在参数值上，这就是这里说的“解耦”。')+
        update+
        p('裁剪只在开头做一次全局归约。之后每个参数各自维护同形的 '+m('m_t,v_t')+'，用逐元素的乘、平方、开方和除法算出同形的 '+m('U_t')+'。裁剪上限约束的是进入历史统计的这一轮梯度，最终更新量还取决于学习率和历史。<code>step()</code> 完成后，下一轮前向会读到新 <code>data</code>。下一轮开始前由训练循环清零 <code>grad</code>，优化器的历史仍然保留。'))

    worked_example = clipping_example + (
        p('接着使用刚才裁剪后的 '+m('G_W')+'、'+m('G_b')+'。为便于手算，例中取 '+m(r'\beta_1=\beta_2=0.5')+'、'+m(r'\eta=0.1')+'，<code>decay</code> 取 '+m('0.1')+'，<code>eps</code> 用默认值。第一步从零历史开始，校正后 '+m(r'\widehat m_1=G')+'、'+m(r'\widehat v_1=G^{\odot2}')+'，开方得到各梯度元素的绝对值。因此示例中两个非零梯度分别产生约 '+m('0.1')+' 和 '+m('-0.1')+' 的扣减量。')+
        aligned(r'W_1&\approx0.99\begin{bmatrix}1&2\end{bmatrix}-\begin{bmatrix}0.1&0\end{bmatrix}',
                r'&=\begin{bmatrix}0.89&1.98\end{bmatrix}',
                r'b_1&\approx(1)-(-0.1)=(1.1)')+
        p('注意权重的第二个元素：本轮梯度为零，仍因衰减从 '+m('2')+' 变为 '+m('1.98')+'。偏置的梯度为负，减去负的更新量后，偏置变大。下一轮要用这些新值重新前向、反向，再裁剪。')+
        p('假设第二轮重新反向得到的全局范数小于上限，权重第一个元素这轮的梯度是 '+m('-0.1')+'。它第一步存下的方向历史是 '+m(r'm_1\approx0.5\times0.6=0.3')+'，第二步更新为 '+m(r'm_2\approx0.5\times0.3+0.5\times(-0.1)=0.1')+'。这一刻的梯度已经是负的，方向历史仍为正。第二步的扣减量使用这份新历史、对应的平方历史和步数 '+m('t=2')+'。这就是为什么 <code>step()</code> 要接着上一轮的状态继续算。'))

    limits = p('空参数集合或全零梯度的原始范数为 '+m('0')+'。有非零原始梯度时，把上限设为 '+m('0')+' 会把所有 <code>grad</code> 缩为零。此时若先前留有方向历史，或者权重仍参与衰减，一次 <code>step()</code> 仍可能改变参数。')

    body += h('一次更新的主线',
        p('反向已经把原始梯度写进 <code>grad</code>。更新按下面两步进行，展开可看每个量的定义和计算。')+
        fold('第一步 · 全局裁剪梯度', clipping)+
        p('<code>clip_grad()</code> 给全模型梯度用同一个比例。缩放后的梯度留在 <code>grad</code>，供优化器读取。')+
        fold('第二步 · 用 AdamW 更新参数', updating)+
        p('<code>AdamW.step()</code> 把当前梯度纳入方向和尺度的历史记录，算出扣减量并写回 <code>data</code>。新权重用于下一轮前向，历史用于下一次更新。'))
    body += h('按需核对',
        fold('同一组权重和偏置怎样连续更新两步', worked_example)+
        fold('空参数、零梯度和上限为零', limits))

    body += h('核心总结',
        aligned(r'R_p&=\nabla_{\theta_p}L',
                r'G_p&=aR_p',
                r'\theta_t&=(1-\eta\lambda_p)\theta_{t-1}-U_t')+
        p('反向写入原始梯度 '+m('R_p')+'，裁剪用全模型共用的比例 '+m('a')+' 将其改为 '+m('G_p')+'。优化器用 '+m('G_p')+' 更新历史并计算扣减量 '+m('U_t')+'，再更新参数。新权重供下一轮前向使用，新历史供下一次 <code>step()</code> 使用。'))

    recap = (
        p('裁剪时，'+m('R_p')+' 与 '+m('G_p')+' 分别表示同一份参数缩放前后的梯度，'+m(r'\ell')+' 是 <code>limit</code>。原地写回 '+m('G_p')+'，返回 '+m('n')+'。')+
        clip_formula+
        p('更新时逐参数计算，省去参数下标。'+m('G')+' 来自当前 <code>grad</code>，'+m('m_0=v_0=0')+'，先递增 <code>t</code>。历史数组、梯度和参数同形。')+
        moments+correction+update+
        p('其中 '+m(r'\eta,\beta_1,\beta_2,\epsilon')+' 对应 <code>lr</code>、<code>beta1</code>、<code>beta2</code>、<code>eps</code>。'+m(r'\lambda_p')+' 在参数维数至少为二时取 <code>decay</code>，其余取零。'+m('U_t')+' 是参数的扣减量。'))
    return body, recap


def apply_part08(book,math,aligned):
    lessons=book['lessons'];opt=lessons['t_optim'];train=lessons['t_train']
    # Keep the old experiment entry available for replaying saved snapshots.
    if train.get('lab') and 't_generate' in lessons:
        train['lab_replay_only'] = True
        lessons['t_generate'].update(
            lab=True,
            experiment_dependencies=[key for key in train['experiment_dependencies'] if key != 't_generate'],
        )
    opt['goal']='亲手建立优化器的历史状态，把全模型梯度转为连续参数更新。'
    train['goal']='复用已有个人组件，完成固定批次的一次到多次训练，并区分训练和验证状态。'
    m=lambda x:math(x,inline=True)
    p=lambda x:'<p>'+x+'</p>'
    h=lambda t,x:'<h3>'+t+'</h3>'+x
    ul=lambda xs:'<ol>'+''.join('<li>'+x+'</li>' for x in xs)+'</ol>'
    fold=lambda t,x:'<details class="block-detail"><summary>'+t+'</summary><div class="block-detail-body">'+x+'</div></details>'
    opt['body'], optimizer_recap = optimizer_lesson(math, aligned, p, h, ul, fold)
    # 裁剪和优化器都直接使用模型提供的唯一参数列表。
    ref = opt['reference'].replace('    parameters = list(parameters)\n', '', 1)
    opt['reference'] = ref.replace('self.params = list(parameters)', 'self.params = parameters')
    train['body']=h('先看整体：把个人模型变成一次完整训练',p('第七部分得到的 GPT 接收编号，输出下一 token 的分数。现在调用你已经实现的交叉熵、反向和优化器，让参数真的改变。本题固定同一批 x/y 反复训练，用来检查整条链能否降低损失；它是一个小批次过拟合实验。')+p('x、y 来自上一部分 make_batch：x 是输入编号，y 是逐位置的下一编号目标；model 是调用方传入的个人 GPT 实例。train 不重新创建模型。steps 决定更新次数，lr 交给一个持久的个人 AdamW。'))
    train['body']+=h('准备与来源 · 循环外创建什么',p('先取得 model.parameters()，构造一个 CrossEntropy 和一个 AdamW，创建空的 losses 列表。优化器在整个循环中共用，保证 m/v/t 连续；不能放进每步循环，也不能用反复调用 train(..., steps=1) 来冒充同一条连续训练。')+p('本题输入 '+m(r'x,y:(B,T)')+'，分数 '+m('S:(B,T,V)')+'，交叉熵返回标量损失。函数返回 '+m(r'\mathrm{losses}:(\mathrm{steps},)')+' 的 NumPy 数组；零步返回空数组。'))
    steps=[('清零','model.zero_grad()','丢弃上一轮累计梯度，保留权重与优化器历史。'),('前向','model(x)','用当前权重计算分数；模型各子层保存本轮缓存。'),('损失','loss_fn(logits, y)','保存本次交叉熵缓存，记录更新前损失。'),('反向','model.backward(loss_fn.backward())','先由损失得到分数上游，再沿个人模型返回并累计参数梯度。'),('裁剪','clip_grad(model.parameters(), 1.0)','统一缩放本轮梯度；返回范数可用于记录。'),('更新','opt.step()','使用同一优化器更新一次权重，历史状态推进一步。')]
    flow='<div class="p8-flow">'+''.join('<div><strong>'+str(i+1)+' · '+a+'</strong><code>'+b+'</code><p>'+c+'</p></div>' for i,(a,b,c) in enumerate(steps))+'</div>'
    train['body']+=h('训练主线 · 每一步沿这条链走完',flow+p('把本轮 loss 转为 Python float 加到 losses。这个值描述更新前模型的表现；下一轮重新前向后，才得到更新后模型对应的损失。不要先更新再拿旧分数冒充新损失。'))
    train['body']+=h('训练与验证 · 缓存什么时候可以被覆盖',p('本轮训练前向、损失前向、损失反向、模型反向必须连着完成。中途插入验证前向，会覆盖模型或损失缓存，使反向使用错误的一轮数据。独立验证放在完整更新之后或训练之前，只前向计算，不调用 backward/step，不改变权重和已有梯度。下一轮训练仍应重新前向。')+p('验证批次来自独立的数据区域，窗口和右移目标不能跨边界；对一组固定窗口计算平均损失，便于比较不同训练时刻。本题实现的是固定批次 train；网页实验室与 run_gpt 的随机抽样、验证记录和绘图循环仍由平台控制，它们调用个人模型/损失/优化器，并未直接调用本题 train。接口接通的后续工作单独保留，不把平台循环当作你已实现的成果。'))
    tf=aligned(r'S_k&=\operatorname{model}_{\theta_k}(x),\qquad\ell_k=\operatorname{CrossEntropy}(S_k,y)',r'\nabla\theta_k&\xleftarrow{\text{个人反向}}\nabla S_k',r'g_k&=\operatorname{clip}(\nabla\theta_k),\quad(\theta_{k+1},s_{k+1})=\operatorname{AdamW}(\theta_k,g_k,s_k)',r'\mathrm{losses}[k]&=\ell_k')
    train['body']+=h('核心总结：同一模型、同一优化器，完成连续更新',tf+p('其中 '+m(r's_k=(m_k,v_k,t_k)')+' 是优化器历史；'+m(r'\ell_k')+' 是第 '+m('k')+' 轮更新前损失，'+m(r'\nabla S_k=\partial\ell_k/\partial S_k')+'。')+p('清零 → 前向 → 损失 → 反向 → 裁剪 → 更新。返回每次更新前的损失；模型留下更新后的参数，优化器在本次调用内跨步保留状态。'))
    train['extension']=h('以后再扩展：随机批次、梯度累计与学习率调度',p('真实训练会逐步重新抽取批次，并在多次控制调用间保留优化器。梯度累计需要按有效位置数加权，调度需要明确全局步数；这些都超出当前固定批次函数的交付。不要每个 micro-batch 重建优化器。后续接口升级应复用个人训练步骤，再由平台记录图表。'))
    def page(meta,interface,inputs,outputs,state,actions,formula):
        meta.update(definition='',derivation='',steps=[],trace=[],hint='按题末清单实现；组件反向沿用已有个人模块。')
        meta['learning_page']=dict(problem_title=meta['title'],role='miniGPT 真实组成',purpose=meta['goal'],statement=h('题目描述',p('实现 '+interface+'，将本节主线落实为可测代码。'))+h('输入格式',p(inputs))+h('输出格式',p(outputs))+h('状态变化',p(state))+h('动手任务清单',ul(actions))+h('关键公式速查',formula),sample_input='',sample_note=p('公开样例按给定配置检查结果；每个用例都重新构造对象。同一次多步更新中，状态必须连续保留。'))
    page(opt,'<code>clip_grad(parameters, limit)</code>、<code>AdamW(parameters, lr=.001, beta1=.9, beta2=.999, eps=1e-8, decay=.01)</code> 和 <code>AdamW.step()</code>',
        '裁剪和优化器都接收 <code>Module.parameters()</code> 返回的唯一参数列表。裁剪阈值非负。优化器默认配置为 <code>lr=.001, beta1=.9, beta2=.999, eps=1e-8, decay=.01</code>。',
        '裁剪返回零维 NumPy 范数数组；构造得到优化器，step 无显式返回值。',
        '裁剪只改 grad；step 原地更新 data、m、v，整体步数加一；梯度保持裁剪后的值。公开状态为 params（按输入顺序保存的参数引用列表）、m/v（与参数一一对应的同形数组列表）、t（累计步数），供本题检查和后续保存读取；配置保存在同名属性 lr/beta1/beta2/eps/decay。',
        ['直接遍历传入的参数列表，先求所有梯度元素的共同范数，再统一原地缩放，返回原范数。','构造时保存传入的参数列表，保留引用和顺序，记录配置，为每个参数创建独立、同形的全零 m/v，t 设零。','step 将 t 加一，原地更新 m/v，按当前 t 做偏差校正。','仅对二维及更高维 data 衰减，再减自适应步长；保留参数和数组身份。'],optimizer_recap)
    page(train,'<code>train(model, x, y, steps, lr)</code>',
        '当前个人模型、二维整数输入与同形目标、非负训练步数和正学习率。批次有效且不超过模型上下文。',
        '长度等于 steps 的一维 NumPy 损失数组，每项是对应更新前损失。',
        '模型参数更新，最后一步梯度保留裁剪后的值；每步先清零。一个优化器在循环外创建，每步仅更新一次。',
        ['循环外创建个人 CrossEntropy、个人 AdamW 与空 losses 列表。','每步先 <code>model.zero_grad()</code>，再前向和损失；不要在反向前插入验证。','用 <code>loss_fn.backward()</code> 的结果调用 <code>model.backward(g)</code>，再裁剪全部参数梯度到阈值1。','调用 <code>opt.step()</code> 一次，保存本轮更新前损失；最后返回 NumPy 数组。'],tf)
    opt['learning_page']['sample_note']=p('本例两组参数分别是二维权重和一维偏置。第一步全局梯度范数为 '+m(r'\sqrt{35}\approx5.91608')+'，共同裁到阈值一；第二步范数约 '+m('0.76811')+'，保持原梯度。只衰减二维权重；两步后 '+m('t=2')+'，样例同时检查参数和历史矩。')
    train['learning_page']['sample_note']=p('本例有 '+m('B=2,T=3,V=5')+'，同一批输入和右移目标连续训练 '+m('12')+' 步。返回数组首项约 '+m('1.60994')+'，是第一次更新前损失；末项约 '+m('1.13323')+'，是第十二次更新前损失。测评随后再前向，检查十二次更新后的分数。')
    for key in ['t_optim','t_train']:
        meta=lessons[key]
        brief=meta['task_brief']
        if key=='t_optim':
            brief['scope']='实现 clip_grad、AdamW.__init__ 与 AdamW.step'
            brief['dimensions']={}
            brief['variables'][1]['shape']='非负实数'
        else:
            brief['variables'][3]['shape']='非负整数'
            brief['conditions']=['CrossEntropy；AdamW decay=.01；clip_grad limit=1。','每轮先反向再评估，避免覆盖训练缓存；优化器只在循环外创建一次。']
            brief.pop('next_use',None)
        tree=ast.parse(meta['reference'])
        for n in ast.walk(tree):
            if isinstance(n,ast.FunctionDef) and n.name in ('clip_grad','__init__','step','train'):n.body=ast.parse('raise NotImplementedError("请完成 '+n.name+'")').body
        meta['starter']=ast.unparse(ast.fix_missing_locations(tree))
        imports='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n'
        other_deps=[d for d in meta['dependencies'] if d!='infrastructure']
        if other_deps:
            imports+='if __name__ != "student":\n'+''.join('    from '+lessons[d]['file'][:-3]+' import '+', '.join(lessons[d].get('exports',[lessons[d].get('class','')]))+'\n' for d in other_deps)
        scaffold(meta,meta['starter'],imports,'_transformer_checks')
        identifiers=('Parameter.grad','Parameter.data','model.parameters()','model.zero_grad','model.backward','loss_fn.backward','train(..., steps=1)','CrossEntropy','AdamW','clip_grad','parameters','data','grad','step','steps','lr','beta1','beta2','eps','decay','losses','loss','float','model','make_batch','train','backward','m','v','t','x','y','run_gpt','NaN','micro-batch','params','opt.step','loss_fn','logits')
        for field in ('body','extension'):meta[field]=format_prose(meta.get(field,''),identifiers)
        meta['learning_page']['statement']=format_prose(meta['learning_page']['statement'],identifiers)
        meta['cases']=[c for c in meta['cases'] if c.get('assessment')!='part08_contract']
        for variant in (['state','empty'] if key=='t_optim' else ['order','zero']):meta['cases'].append(dict(label={'state':'三步历史状态、全局范数与共享参数复用','empty':'空参数、零梯度与数组身份','order':'个人组件调用顺序与循环外优化器','zero':'零步不更新与输入保持'}[variant],assessment='part08_contract',input=dict(lesson=key,variant=variant),expected=dict(要求='满足本题输入输出及状态约定'),category='状态与复用'))
        if key == 't_optim':
            meta['assessment_revision'] = 7
            meta['assessment_migration'] = dict(
                previous_revision=6, reason='AdamW 测评直接传入模型参数列表，保留旧成绩。')
        else:
            meta['assessment_revision']=3;meta['assessment_migration']=dict(previous_revision=2,reason='保留原数值检查，补构造/共享去重与训练调用状态，旧成绩保留。')
    return book
