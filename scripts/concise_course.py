"""学习文案终稿：实现入口、变量、计算、输出；保留公式和测评数据。"""
import re


OPERATIONS = {
    'map':'沿训练和生成两条路径，辨认编号、向量、分数、损失和梯度。',
    'architecture':'按数据流识别 Embedding、Block、归一化和输出头。',
    'shape':'将 values 转为 float64 数组，读取各轴长度。',
    'axes':'沿特征轴求和，得到每条样本的总和。',
    'index':'按 ids 的顺序取行，再保留前 k 列。',
    'reduce':'求每行均值，再从该行的每个特征中减去均值。',
    'broadcast':'将 b 的每个值加到 X 对应的特征列。',
    'matmul':'将 X 的每一行与 W 的每一列沿 C 轴乘加。',
    'linear':'合并位置轴，计算 X2@W+b，再恢复输出形状。',
    'reshape':'按 batch、token 顺序，将前两个轴合并成位置轴。',
    'restore':'按 out_shape，将二维表的每一行放回原位置。',
    'transpose':'保持 batch 轴，交换 token 轴和特征轴。',
    'heads':'把特征轴拆成 H×d，再将 head 轴移到 token 轴之前。',
    'copy':'复制 X，将副本的 [0,0] 改为 −99。',
    'scatter':'复制 table，将每条 G 加到 ids 指定的行；重复编号继续累加。',
    'random':'用 seed 创建局部 Generator，按 dims 采样标准正态数组。',
    'random_sequence':'创建一个 Generator，连续采样两份数组，按先后顺序堆叠。',
    'infrastructure':'创建 data/grad，递归收集并去重参数，按接口清零梯度。',
    'initialization':'根据输入宽度确定初始尺度，再识别残差出口的缩放规则。',
    'layer_scale':'前向缓存 X 并缩放；反向返回 dX，将各位置贡献累加到 w.grad。',
    'layer_linear':'前向缓存 X2 和原形状；反向计算 dX，并累加 W.grad、可选 b.grad。',
    'layer_embedding':'按 ids 查词表；反向将各位置的 G 累加到对应词表行。',
    'layer_gelu':'前向计算 tanh 近似 GELU；反向将 G 乘局部导数。',
    'layer_rmsnorm':'逐 token 计算均方根并缩放；反向合并两条输入路径，累加 w.grad。',
    'layer_mlp':'前向依次调用 fc1、GELU、fc2；反向按相反顺序传递梯度。',
    't_softmax':'沿类别轴稳定归一化；反向用缓存概率计算行内耦合梯度。',
    't_loss':'计算稳定 log 概率，按 targets 取值并平均；反向返回 logits 梯度。',
    'att_scores':'交换 K 的最后两轴，与 Q 相乘，再除以 √d。',
    'att_probs':'将未来位置设为 −∞，沿 key 轴计算稳定 Softmax。',
    'att_weighted':'沿 key 轴，用 A 对 V 的内容向量加权。',
    'att_merge':'先交换 head/token 轴，再按 head 顺序拼接特征。',
    'mean_backward':'将 G 除以 C，再沿特征轴广播到 C 个位置。',
    'center_backward':'求 G 沿特征轴的均值，再用 G 减去该均值。',
    'att_value_backward':'交换 A 的最后两轴，与 G 相乘，汇总所有 query 的贡献。',
    'att_score_backward':'求每行 A*dA 的和，计算 A*(dA−该和)。',
    't_attention':'前向完成打分、掩码、归一化和加权；反向依次计算 dV、dA、dS、dQ、dK。',
    't_mha':'串联 QKV 投影、拆头、注意力、合头和输出投影；反向逐级回流。',
    't_block':'前向完成两次残差相加；反向在每个分叉汇合直连与子层梯度。',
    't_dropout':'训练时采样并缓存缩放掩码，评估时传递原值；反向复用本次掩码。',
    't_layernorm':'逐 token 中心化、归一化并缩放；反向计算 dX，累加 w.grad、b.grad。',
    't_gpt':'查 token/位置表，经 Blocks 和 Norm 后用共享词表输出；反向汇合共享参数梯度。',
    't_data':'排序去重建立字符词表，再按 starts 截取输入和右移一位的标签。',
    't_optim':'先按全局范数裁剪，再更新 AdamW 的 m、v、t 和参数 data。',
    't_train':'每步清零梯度、前向、记录损失、反向、裁剪并更新；优化器在循环外创建。',
    't_sample':'温度缩放、保留 top-k、归一化，再用传入 rng 逐行抽样。',
    't_generate':'截取末尾上下文，取最后位置 logits，采样并追加，重复 steps 次。',
    't_checkpoint':'保存参数、优化器和 RNG 状态；读取时核对形状，恢复状态并清零旧梯度。',
    't_project':'调用个人模块完成一次前向、反向、全局裁剪和参数更新。',
    'rebuild_linear':'创建 W 和可选 b，实现前向缓存、dX 及参数梯度累加。',
    'rebuild_block':'创建两个 Norm、Attention 和 MLP，实现两条残差及反向合流。',
    'rebuild_gpt':'创建词表、位置表、Blocks 和 Norm，完成共享输出头的前向与完整反向。',
    'fix_bias':'沿 batch/token 轴汇总 G，保留输出特征轴。',
    'fix_heads':'先交换 head/token 轴，再合并特征轴，恢复正确坐标。',
    'fix_residual':'将 direct 和 branch 两路贡献相加。',
    'fix_shared':'将输出头和输入查表的贡献累加到同一个词表 grad。',
    'fix_clear':'在每步前向、反向开始前清零旧梯度，保留跨步优化器状态。'
}

SCOPES = {
    'map':'辨认训练与生成的数据流', 'architecture':'辨认 GPT 与 Block 的结构',
    'initialization':'判断初始化规则',
    'infrastructure':'实现 Parameter.__init__、zero_grad 和 Module.parameters、zero_grad',
    't_data':'实现 tokenize(text) 和 make_batch(ids, starts, T)',
    't_optim':'实现 clip_grad(parameters, limit) 和 AdamW.step()',
    't_train':'实现 train(model, x, y, steps, lr)',
    't_sample':'实现 sample(logits, temperature, top_k, rng)',
    't_generate':'实现 generate(model, ids, steps, temperature, top_k, seed)',
    't_checkpoint':'实现 save_checkpoint 和 load_checkpoint',
    't_project':'实现 project_step(model, optimizer, ids, targets)',
    'rebuild_linear':'实现 Linear 的 __init__、forward 和 backward',
    'rebuild_block':'实现 Block 的 __init__、forward 和 backward',
    'rebuild_gpt':'实现 GPT 的 __init__、forward 和 backward',
    'fix_shared':'修正 GPT.backward(G)', 'fix_clear':'修正 train(model, x, y, steps, lr)'
}

CONDITIONS = {
    'map':[], 'architecture':[], 'shape':[],
    'axes':['axis=1，keepdims=False。'],
    'index':['行列编号从 0 开始；列范围为 [0,k)。'],
    'reduce':['axis=-1，keepdims=True。'],
    'broadcast':['b 的长度等于 X 最后一轴的长度。'],
    'matmul':['X 的列数等于 W 的行数。'],
    'heads':['C 能被 H 整除；d=C/H。'],
    'random':['使用 default_rng(seed)。'],
    'random_sequence':['第一轴表示两次采样，后续轴保留 dims。'],
    'infrastructure':['按对象身份去重；zero_grad 保持 data 原值。'],
    'initialization':['基础层使用 fan-in 初始化；GPT 的二维权重使用均值 0、标准差 0.02 的正态分布，再缩放残差出口。'],
    'layer_scale':['缓存前向 X。'],
    'layer_linear':['N 是所有前置轴的乘积；上游 G 已包含损失的平均系数。'],
    'layer_mlp':['每个 token 复用同一组子层参数。'],
    't_attention':['causal=True 时 Tq=Tk；causal=False 时 Tq、Tk 可以不同。','缓存 Q/K/V、A；因果模式还需保存掩码。'],
    't_mha':['各头与位置共享投影参数。'],
    't_dropout':['训练与反向使用同一 mask；评估使用全 1 掩码。'],
    't_optim':['全局统一裁剪；二维及更高维权重参与衰减。'],
    't_sample':['并列时按稳定 argsort 保留末尾 k 项；沿用传入 rng 的状态。'],
    't_generate':['每次送入模型的序列长度 ≤context；返回完整提示和新增编号。'],
    't_project':['使用调用方提供的 model 和 optimizer。'],
    'fix_heads':['Y[b,t,h*d+j]=X[b,h,t,j]。'],
    'fix_residual':['direct、branch 是同一输入的两路梯度。'],
    'fix_shared':['两条路径复用同一个词表 Parameter。'],
    'center_backward':['沿最后特征轴求均值，keepdims=True。'],
    'att_weighted':['沿 Tk 轴乘加；V 的特征宽度为 dv。'],
    'att_score_backward':['沿最后 key 轴求和，keepdims=True。']
}

REPLACEMENTS = {
    '输入特征；是函数参数，不是特殊关键字':'输入特征',
    '整数 token 编号，不是连续特征':'整数 token 编号',
    '(B,T)，不求导':'(B,T)，int64',
    '保留前几列，不是固定写死的 2':'保留的列数',
    '原始样本表':'N 行、C 列的输入数组',
    '要依次取出的原行编号，可重复':'依次取出的行编号，共 M 个，可重复',
    '两份样本堆叠，形状 (2,*dims)，float64。':'两份样本组成的 float64 数组，形状 (2,*dims)。',
    'forward 只返回同形 Y；backward 只返回同形 dX。':'forward 返回同形 Y；backward 返回同形 dX。',
    'forward 返回同形 Y；backward 只返回同形 dX。':'forward 返回同形 Y；backward 返回同形 dX。',
    '，不返回标签梯度':'',
    '，不对整数 targets 求导':'',
    '，不对整数 ids 求导':'',
    '只累加 weight.grad':'累加 weight.grad',
    '输入不变':'输入保持原值',
    '原输入保持不变':'原输入保持原值',
    '不修改 X':'保持 X 原值',
    '不修改输入':'保持输入原值',
    '不改变参数 data':'参数 data 保持原值',
    '不能直接用 A*dA':'使用完整的行内耦合梯度',
    '不能只将 G 除以 C':'合并直接与均值两路贡献',
    '，不能重采样':'',
    '；不在 backward 更新 w.data':'；参数更新由优化器执行',
    '，不是原始 G':'',
    '，不构造完整雅可比':'',
    '不能覆盖':'按 += 累加',
    '不更新 weight.data':'weight.data 保持原值',
    '无返回数组':'返回 None',
    '本节只做前向':'计算过程',
    '本节不提交':'拓展',
    '不要把激活或梯度当成新参数。':'',
    '同一个偏置被各条样本复用，计算规则不要求先复制 N 份偏置。':'各样本复用同一组偏置。',
    'X.shape=(2,3) 说明有两个轴，不是两个元素。':'X.shape=(2,3) 表示 2 行、3 列。',
    '三维 X.T 会逆序全部轴，不能当作“仅转置最后两轴”。':'',
    '不要每次 forward 重置种子。':'',
    '不能只缓存输出，因为 w=0 时无法从输出恢复 X。':'缓存 X 用于参数梯度计算。',
    '参数梯度是求和，不是求均值。':'参数梯度沿所有位置求和。',
    '不能用 weight.grad[ids] += G 代替重复索引累加。':'',
    '每层收到的是损失对它自身输出的梯度，不是都传入原始 G。':'将各层返回的梯度传给前一层。',
    '不能直接 reshape 到（B,H,T,d），那会改变 token 与 head 的对应。':'',
    '这里不是把整个三维张量归一化成一个分布。':'每个 token 独立归一化。',
    'targets[b,t] 指定该位置正确类别，不是 one-hot。':'targets[b,t] 是正确类别的整数编号。',
    '不要先 softmax 再 log，极小概率会下溢。':'使用 LogSumExp 稳定计算 logp。',
    '为什么不能相加':'QKV 梯度拼接',
    '目标移位本身不能替代掩码。':'标签右移一位，掩码限制读取范围。',
    '评估只 forward 和计算 loss，不能 backward/step。':'评估执行 forward 和损失计算。',
    '固定小批次 loss 应能下降，但不能单靠下降排除因果泄漏。':'过拟合检查损失下降；因果性使用独立测评检查。',
    '不能先 Softmax 再直接清零，否则每行概率和改变。':'',
}


def rewrite(text):
    for old,new in REPLACEMENTS.items():text=text.replace(old,new)
    return text


def apply_concise(book):
    lessons=book['lessons']
    if set(lessons)-{'foundation_capstone','t_init_upgrade'}!=set(OPERATIONS):raise ValueError('精简文案未覆盖全部任务')
    methods={
        'layer_scale':('Scale','X'),'layer_linear':('Linear','X'),
        'layer_embedding':('Embedding','ids'),'layer_gelu':('GELU','X'),
        'layer_rmsnorm':('RMSNorm','X'),'layer_mlp':('MLP','X'),
        't_softmax':('Softmax','X'),'t_attention':('AttentionCore','Q, K, V'),
        't_mha':('MultiHeadAttention','X'),'t_block':('Block','X'),
        't_dropout':('Dropout','X, training'),'t_layernorm':('LayerNorm','X'),
        't_gpt':('GPT','ids')}
    for id,m in lessons.items():
        if id in ('foundation_capstone','t_init_upgrade'):continue  # 由part45_lessons统一生成。
        t=m['task_brief']
        if m['kind']=='array':t['scope']=('修正' if m.get('repair') else '实现')+' solve('+', '.join(m['args'])+')'
        elif id in methods:
            cls,args=methods[id];t['scope']=f'实现 {cls}.forward({args}) 和 backward(G)'
        elif id=='t_loss':t['scope']='实现 CrossEntropy.forward(logits, targets) 和 backward(g)'
        else:t['scope']=SCOPES[id]
        t['operation']=OPERATIONS[id]
        t['output']=rewrite(t['output'])
        t['conditions']=[rewrite(c) for c in CONDITIONS.get(id,t['conditions'])]
        for v in t['variables']:
            v['meaning']=rewrite(v['meaning']);v['shape']=rewrite(v['shape'])
        if id=='index':
            t['variables'][2]['shape']='整数标量，1≤k≤C'
            t['output']='NumPy 数组，形状 (M,k)。'
            t['dimensions']={}
            m['steps']=[]
            m['body']=''
        for field in ('body','derivation'):
            text=m.get(field,'')
            def prune(match):
                title,content=match[1],match[2]
                redundant={'输入','操作与输出','接口','输入输出','输入与输出','变量与形状','构造参数','最终交付','接口练习','组装练习','最终独立实现','统一形状','前置依赖'}
                extra=(id=='reshape' and title=='恢复') or (id=='transpose' and title in {'批量 @','反向'}) or (id=='reduce' and title=='为什么保留最后一轴') or (id=='heads' and title in {'四个轴','还原'})
                if field=='body' and 'data-tex=' not in content and (title in redundant or extra):return ''
                return match[0]
            text=re.sub(r'<h3>(.*?)</h3>(.*?)(?=<h3>|$)',prune,text,flags=re.S)
            for old,new in {'这次只写反向':'梯度计算','仅实现对 V 的反向':'V 的梯度','仅实现 Softmax 回到分数的反向':'分数梯度','只恢复形状':'位置恢复','推进状态，不重置种子':'随机状态推进','不能省掉的累加':'重复编号累加','逐行去均值：本节只做前向':'逐行去均值'}.items():text=text.replace('<h3>'+old+'</h3>','<h3>'+new+'</h3>')
            if field=='body' and id=='fix_clear':
                text=re.sub(r'本页下方实验室|(?:训练闭环|自回归生成)页的实验室','目录中的训练实验室',text)
            m[field]=rewrite(text)
            if field=='body' and id=='t_generate':m[field]=re.sub(r'KV cache[^。]*。','',m[field])
        if id in {'mean_backward','center_backward','restore','att_value_backward','att_score_backward'}:m['body']=''
        m['steps']=[rewrite(s) for s in m.get('steps',[])]
        m['goal']=t['operation']
    book['language_revision']=1
    return book
