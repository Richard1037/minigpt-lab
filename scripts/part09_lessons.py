"""Part09 main path and optional practice, built from the current personal interfaces."""
from copy import deepcopy
from pathlib import Path

from initialization_upgrade import format_prose


def generation_scores(case):
    import numpy as np
    from transformer_layers import GPT, sample

    inputs = case['input']
    model = GPT(*inputs['init'])
    rng = np.random.default_rng(inputs['seed'])
    result = np.asarray(case['expected']['ids'], dtype=np.int64)
    prompt_length = len(inputs['ids'][0])
    scores = []
    for end in range(prompt_length, prompt_length + inputs['steps']):
        window = result[:, max(0, end - model.context):end]
        logits = model(window)[:, -1, :]
        chosen = sample(logits, inputs['temperature'], inputs['top_k'], rng)
        if not np.array_equal(chosen, result[:, end]):
            raise ValueError(f'Generation demo diverged from its public sample at position {end}')
        scores.append(logits.tolist())
    return scores


def apply_part09(book, math, aligned):
    lessons = book['lessons']
    keys = ['t_sample', 't_generate', 't_checkpoint', 't_project', 'rebuild_gpt']
    if not all(key in lessons for key in keys):
        return book
    def inline(tex):
        return math(tex, inline=True)
    def p(text):
        return '<p>' + text + '</p>'
    def h(title, content):
        return '<h3>' + title + '</h3>' + content
    def fold(title, content):
        return '<details class="block-detail"><summary>' + title + '</summary><div class="block-detail-body">' + content + '</div></details>'
    def items(parts):
        return '<ol>' + ''.join('<li>' + part + '</li>' for part in parts) + '</ol>'
    def code(text):
        return '<code>' + text + '</code>'
    def page(key, purpose, body, interface, inputs, outputs, state, actions, recap, sample_note, extension=''):
        meta = lessons[key]
        identifiers = ('Softmax', 'sample', 'generate', 'checkpoint.npz', 'meta.json', 'report.json',
                       'run_gpt.py', 'AdamW', 'CrossEntropy', 'project_step', 'clip_grad')
        def fmt(text):
            return format_prose(text, identifiers)
        meta.update(body=fmt(body), definition='', derivation='', steps=[], trace=[], extension=fmt(extension))
        meta['learning_page'] = dict(
            problem_title=meta['title'], role='miniGPT 实现题', purpose=purpose,
            statement=fmt(h('题目描述', p(purpose)) + h('实现接口', p(interface))
                          + h('输入格式', p(inputs)) + h('输出格式', p(outputs))
                          + h('状态变化', p(state)) + h('动手任务清单', items(actions))
                          + h('关键公式速查', recap)),
            sample_input='', sample_note=fmt(p(sample_note)),
        )

    sample_math = aligned(
        r'Z_{b,i}&=S_{b,i}/\tau',
        r'\widetilde Z_{b,i}&=\begin{cases}Z_{b,i}&i\in K_b\\-\infty&i\notin K_b\end{cases}',
        r'P_{b,:}&=\operatorname{Softmax}(\widetilde Z_{b,:}),\quad I_b\sim P_{b,:}',
    )
    sample_recap = p('下标 ' + inline('b') + ' 是序列，' + inline('i') + ' 是词表编号。' +
        inline('S') + ' 是模型分数，' + inline(r'\tau') + ' 是温度，' + inline('K_b') +
        ' 是本行保留的编号，' + inline('P') + ' 是沿词表轴归一化后的概率，' +
        inline('I_b') + ' 是这一行抽到的编号。') + sample_math
    sample_body = h('把一行分数变成一个可追加的编号',
        p('GPT 已经为每个候选编号打了分。生成时，采样需要为批次里的每条序列选一个编号。温度决定概率有多集中，保留数量决定哪些候选还能参与抽样。每条序列独立计算概率，再按批次顺序从同一个随机生成器抽取。') +
        items(['用温度缩放分数。温度越低，分数差距在归一化前越大，概率更集中在高分项。',
               '按行选出得分最高的候选。保留原词表位置，把其余位置的分数设为负无穷，因此归一化后这些位置概率为零。',
               '调用已实现的个人 Softmax，沿最后一轴得到概率。每行调用一次 ' + code('rng.choice(V, p=row)') + '，输出的是词表编号。'])) + \
        fold('用两行分数走一遍温度、筛选和抽样',
            p('课堂样例用两条序列、四个候选，温度与保留数量都取 ' + inline('2') + '。两行保留的编号可以不同。') +
            aligned(r'S&=\begin{bmatrix}0&2&0&4\\4&0&2&0\end{bmatrix}',
                    r'Z&=\begin{bmatrix}0&1&0&2\\2&0&1&0\end{bmatrix}',
                    r'\widetilde Z&=\begin{bmatrix}-\infty&1&-\infty&2\\2&-\infty&1&-\infty\end{bmatrix}',
                    r'P&\approx\begin{bmatrix}0&0.269&0&0.731\\0.731&0&0.269&0\end{bmatrix}') +
            p('第一行只能抽到编号 ' + inline('1,3') + '，第二行只能抽到 ' + inline('0,2') + '。较高的概率表示更容易被选中。具体抽到哪个编号取决于传入的随机状态。')) + \
        fold('NumPy 操作和并列规则',
            p(code('np.argsort(z, axis=-1, kind="stable")') + ' 返回每行按分数从低到高排列的原始编号。' +
              code('[..., -top_k:]') + ' 取排序结果的尾部。分数并列时，稳定排序保留原编号顺序，截取尾部就优先留下编号较大的并列项。' +
              code('top_k == 0') + ' 时直接保留全部候选。') +
            p(code('np.take_along_axis') + ' 可读出这些编号上的分数，' + code('np.put_along_axis') +
              ' 可按原编号写入预填负无穷的数组。随后把每行概率交给 ' + code('rng.choice') +
              '。抽样推进调用方传来的生成器，下一行和下一轮继续使用推进后的状态。')) + \
        h('核心总结', sample_recap)
    page('t_sample', '把每条序列的下一编号分数变成一个可追加的 token 编号。', sample_body,
         code('sample(logits, temperature, top_k, rng)'),
         '二维分数 ' + inline(r'S\in\mathbb R^{B\times V}') + '，正温度、从零到词表大小的整数保留数，以及调用方创建的 NumPy 随机生成器。公开样例的种子用于创建这个对象。',
         '形状为 ' + inline('(B,)') + ' 的 NumPy ' + code('int64') + ' 数组，每条序列一个编号。',
         '分数输入保持原值，随机生成器随抽样推进。词表轴在筛选前后保持长度与编号位置。',
         ['温度除法得到新分数数组。', '按行稳定排序筛选候选，布置掩码。',
          '复用个人 Softmax，按批次顺序用同一个生成器抽样并返回整数数组。'],
         sample_recap, '样例每行有四个候选，只保留两项。输出由该用例的种子与逐行抽样顺序共同决定。')

    gen_math = aligned(
        r'W_j&=R_j[:, -K:],\quad K=\text{上下文上限}',
        r'S_j&=\operatorname{GPT}(W_j)[:, -1, :]',
        r'I_j&=\operatorname{sample}(S_j,\tau,k,\mathrm{rng})',
        r'R_{j+1}&=\operatorname{concat}_{\mathrm{token}}(R_j,I_j[:,\mathrm{None}])',
    )
    gen_note = p(inline('R_j') + ' 是第 ' + inline('j') + ' 轮前保存的完整编号序列，' + inline('W_j') +
        ' 是本轮送进模型的尾部窗口，' + inline('S_j') + ' 是每条序列末位置的词表分数，' + inline('I_j') +
        ' 是各行抽到的新编号。每轮从同一个生成器取样。')
    gen_body = h('用已有前缀续写完整序列',
        p('训练的右移标签来自已有文本。生成只有提示，下一编号由模型预测并采样得到，再接回前缀。完整序列用于保存输出，尾部窗口用于控制每次模型输入的长度。') +
        items(['用 ' + code('result[:, -model.context:]') + ' 为所有批次取各自末尾窗口。',
               '前向得到每个位置的分数，用 ' + code('[:, -1, :]') + ' 取最后一个位置。最后一个位置的分数预测当前前缀之后的编号。',
               '调用个人 sample。用 ' + code('token[:, None]') + ' 把每批一个编号变成一列，再用 ' +
               code('np.concatenate(..., axis=1)') + ' 追加到完整结果。下一轮重新截取窗口。'])) + \
        fold('两条提示连续生成两轮',
            p('上下文上限为 ' + inline('3') + '，两条提示原本各有四个编号。假设两轮抽出的列依次是 ' +
              inline(r'[2,4]^{\mathsf T}') + ' 和 ' + inline(r'[1,0]^{\mathsf T}') + '。新编号来自当轮抽样。') +
            aligned(r'R_0&=\begin{bmatrix}4&0&1&3\\1&2&4&0\end{bmatrix}',
                    r'W_0&=\begin{bmatrix}0&1&3\\2&4&0\end{bmatrix}',
                    r'R_1&=\begin{bmatrix}4&0&1&3&2\\1&2&4&0&4\end{bmatrix}',
                    r'W_1&=\begin{bmatrix}1&3&2\\4&0&4\end{bmatrix}',
                    r'R_2&=\begin{bmatrix}4&0&1&3&2&1\\1&2&4&0&4&0\end{bmatrix}') +
            p('两轮窗口都只有三列，最终结果有六列。追加的位置是完整序列末尾，最初的提示四列完整保留。')) + \
        h('核心总结', gen_note + gen_math) + \
        p('完成本题后，可从目录打开训练实验室。网站用你的个人模块训练一个新模型，再调用 ' + code('generate') + ' 续写你给出的文本开头。')
    page('t_generate', '复用个人 GPT 和采样，在上下文限制内批量续写。', gen_body,
         code('generate(model, ids, steps, temperature=1., top_k=0, seed=0)'),
         '模型拥有正整数 ' + code('context') + '。提示是形状 ' + inline('(B,T_0)') + ' 的二维整数数组，' +
         inline(r'T_0\ge1') + '，可长于上下文。追加步数为非负整数，采样参数沿用上一题约定。',
         '形状 ' + inline(r'(B,T_0+\mathrm{steps})') + ' 的整数数组，包含原提示和新增编号。零步返回提示的独立副本。',
         '先复制提示。本次调用用种子创建一个生成器，各轮推进同一随机状态。每轮前向可以刷新模型缓存，权重和已有梯度保持原值。',
         ['复制提示并创建生成器。', '逐轮截取末尾窗口，前向并取末位置分数。',
          '调用个人 sample；把新编号变成一列，按 token 轴追加到完整结果。'],
         gen_note+gen_math, '核对输出前几列是否完整保留提示，以及追加列数是否等于给定步数。输出涵盖样例的全部批次。',
         h('上下文窗口里的位置编号', p('本课程每轮重新计算完整窗口，位置编号从零起。键值缓存可以减少重复计算，还要处理窗口变化和位置对齐，留作后续拓展。')))

    generation = lessons['t_generate']
    generation['generation_scores'] = generation_scores(generation['cases'][generation.get('sample_index', 0)])
    reference = Path(__file__).resolve().parents[1] / 'solutions' / 'train_my_gpt_reference.py'
    generation['lab_reference'] = reference.read_text(encoding='utf-8')

    checkpoint = lessons['t_checkpoint']
    checkpoint.update(
        provided=True,
        derivation='',
        extension='',
        body=h('保存与恢复接口',
            p(code('save_checkpoint(path, model, optimizer, rng)') +
              ' 将模型权重、AdamW 的步数和历史、抽批次的随机状态保存到一个 NPZ 文件。') +
            p(code('load_checkpoint(path, model, optimizer, rng)') +
              ' 将上述状态原地载入。先创建同配置的模型、绑定其参数的优化器，以及同类型的随机生成器。') +
            p('两者都返回 ' + code('None') + '。续训时使用原训练语料和词表。')) +
        fold('调用例子',
            '<pre><code>from t_checkpoint import save_checkpoint, load_checkpoint\n\n'
            'save_checkpoint("checkpoint.npz", model, optimizer, rng)\n'
            '# Restart with the same configuration.\n'
            'load_checkpoint("checkpoint.npz", model, optimizer, rng)</code></pre>' +
            p('恢复后沿用传入的对象，抽取下一批并继续更新。')) +
        p('个人训练程序会在保存和续训时调用这两个接口。学完可从页尾进入训练实验室查看成果交付。'))
    for key in ('t_generate', 't_checkpoint'):
        lessons[key]['task_brief'].pop('next_use', None)
    
    step_math = aligned(
        r'S&=\operatorname{GPT}_{\theta}(\mathrm{ids})',
        r'L&=\operatorname{CrossEntropy}(S,\mathrm{targets})',
        r'\nabla S&=\operatorname{CrossEntropy.backward}()',
        r'\mathrm{GPT.backward}(\nabla S)&\longrightarrow \nabla\theta',
    )
    step_note = p(inline('S') + ' 是分数，' + inline('L') + ' 是本次更新前的平均损失。梯度写入参数后，进行全局裁剪与一次优化器更新，返回先前得到的 ' + inline('L') + '。')
    project_body = h('把已经完成的模块接成一次训练更新',
        p('第八部分的 ' + code('train') + ' 自己创建优化器并反复使用固定批次。本题把一次更新封装成 ' +
          code('project_step') + '。调用方持有模型和优化器，可以在下一次传入不同批次，并让 AdamW 的历史随调用连续推进。') +
        items(['清空旧梯度，用当前模型计算分数，再求更新前的平均交叉熵。',
               '复用个人损失与模型的反向方法，累积当前批次的梯度。',
               '用个人全局裁剪和传入的优化器完成一次更新，返回更新前的损失。'])) + \
        h('用分项检查定位问题',
          p('第一项检查训练一步。其余用例可用于核对梯度、因果性、共享权重、固定批次降损失、评估、生成和精确续训。成果交付在训练实验室进行。')) + \
        h('核心总结', step_math+step_note)
    lessons['t_project']['title'] = '训练单步与分项检查'
    page('t_project', '用已有个人模块完成一次可重复调用的训练更新。', project_body,
         code('project_step(model, optimizer, ids, targets)'),
         '已创建的个人 GPT、绑定其参数的 AdamW、同形的二维整数输入和下一编号目标。输入的位置长度在模型上下文内。',
         '返回更新前的零维 NumPy 平均损失。',
         '就地更新传入模型和同一个优化器。步数推进一次，之后可用同一组对象训练下一个批次。',
         ['从传入模型清空旧梯度，完成本次前向和交叉熵计算。',
          '调用个人反向并把全局裁剪阈值设为 ' + inline('1') + '。',
          '调用传入优化器更新一次，返回刚才得到的损失。'],
         step_math+step_note, '样例预先在梯度缓冲区写入旧值。需要先清零，再比较一次更新后的权重和更新前损失。')
    lessons['t_project']['task_brief']['next_use'] = dict(lesson='rebuild_gpt', mode='contrast')

    refine_optional(book, page, h, p, fold, aligned, inline, code)
    return book


def refine_optional(book, page, h, p, fold, aligned, inline, code):
    lessons = book['lessons']
    meta = lessons['rebuild_gpt']
    for stage in book['stages']:
        for module in stage['modules']:
            module['tasks'] = [key for key in module['tasks'] if key not in ('rebuild_gpt', 't_project')]
        stage['modules'] = [module for module in stage['modules'] if module['tasks']]
    history = next((stage for stage in book['stages'] if stage.get('id') == 'optional-project-step'), None)
    if history is None:
        history = dict(id='optional-project-step', title='历史 · 训练单步与分项检查', optional=True)
        book['stages'].append(history)
    history['modules'] = [dict(name='训练单步与分项检查', tasks=['t_project'])]
    optional = next((stage for stage in book['stages'] if stage.get('id') == 'optional-gpt-rebuild'), None)
    if optional is None:
        optional = dict(id='optional-gpt-rebuild', title='选修 · miniGPT 独立组装', optional=True,
                        outcome='按需练习从接口重新组装。')
        book['stages'].append(optional)
    optional['outcome'] = '按需练习从接口重新组装。'
    optional['modules'] = [dict(name='独立组装练习', tasks=['rebuild_gpt'])]
    # Optional exercise uses the same upgraded personal components as the main GPT.
    deps = [key for key in lessons['t_project']['dependencies'] if key != 't_gpt'] + ['t_project']
    meta.update(dependencies=deps, experiment_dependencies=deps, reference=lessons['t_gpt']['reference'],
                assessment_revision=2,
                assessment_migration=dict(previous_revision=1, reason='选修复用已升级的个人子层，新增现有初始化与反向来源检查。'))
    meta['cases'] = [case for case in meta['cases'] if case.get('assessment') not in ('part07_contract', 'initialization_upgrade')]
    meta['cases'] += deepcopy([case for case in lessons['t_gpt']['cases']
                              if case.get('assessment') in ('part07_contract', 'initialization_upgrade')])
    recap = aligned(r'X_0&=E[\mathrm{ids}]+P[\mathrm{positions}]',
                    r'X_{j+1}&=\operatorname{Block}_j(X_j)',
                    r'H&=\operatorname{RMSNorm}(X_L),\qquad S=HE^{\mathsf T}',
                    r'\nabla E&=(\nabla S)_{\mathrm{flat}}^{\mathsf T}H_{\mathrm{flat}}+\nabla E_{\mathrm{lookup}}')
    note = p(inline('E,P')+' 是词表和位置表，轴为位置或编号、特征。'+inline('L')+' 是层数，'+inline('H')+
             ' 是最终归一化后的激活，'+inline('S')+' 是词表分数。下标 '+inline(r'\mathrm{flat}')+
             ' 表示合并批次和位置轴。查表梯度由已有 Embedding 按编号累加。')
    body = h('用另一份 GPT 类检验组装能力',
        p('主线的 GPT 已在第七部分组装完成。这道选修适合想从空方法再组装一次的学习者。选修调用相同的个人子层和训练工具。')) + \
        fold('构造：把初始化配置传入已有组件',
             p('以同一个种子创建一个随机生成器，先给词表和位置表，再按模型层顺序交给各个已有 Block。每个 Block 的两个出口用 '+
               inline(r'1/\sqrt{2L}')+' 缩放。子层负责采样初始化，GPT 负责传入配置。')) + \
        fold('前向与反向：追踪两条共享词表路径',
             p('两个查表结果相加后依次通过各 Block 和末尾 RMSNorm。用同一词表转置完成输出投影。根据下方前向公式，反向先把输出头梯度加到词表，再逆序调用已有模块；位置查表梯度沿批次求和，词表查表按编号再累加。')) + \
        h('核心总结', note+recap)
    page('rebuild_gpt', '选修：使用同一份个人基础层，从已有接口独立组装一个 GPT。', body,
         code('GPT(V, C, H, hidden, layers, context, seed=0)') + '、' + code('forward(ids)') + '、' + code('backward(g)'),
         '词表大小、模型宽度、注意力头数、中间宽度、Block 层数、上下文与随机种子；前向接收二维整数编号，反向接收与分数同形的上游梯度。',
         '前向返回分数，反向累加参数梯度并返回 ' + code('None') + '。',
         '模型持有共享词表和独立子层。一次前向的缓存供对应反向使用。',
         ['构造查表和各子层，传入第七部分已实现的初始化配置。',
          '串联各子层并复用词表作为输出权重。',
          '逆序调用已有反向，在同一词表上汇总查表和输出头的梯度。'],
         note+recap, '检查使用新的模型宽度、头数和输入长度时，层间接口仍能配合。')
