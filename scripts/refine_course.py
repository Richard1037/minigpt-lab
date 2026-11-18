"""逐页任务契约与可执行步骤；仅改教材，不改个人实现或成绩。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def apply_refinements(book):
    lessons = book['lessons']
    lessons['reduce']['title']='逐行去均值：mean 与 keepdims'
    specs = {}

    def add(id, scope, variables, output, conditions, steps, flow):
        specs[id] = dict(scope=scope, variables=[dict(zip(('name','meaning','shape'), row)) for row in variables], output=output, conditions=conditions, flow=flow)
        if id in lessons:
            lessons[id]['steps'] = steps
            lessons[id]['task_brief'] = specs[id]

    X = ('X', '输入特征；是函数参数，不是特殊关键字', '(B,T,C)，或本节指定形状')
    G = ('G（代码 g）', '损失对本层输出的上游梯度，由调用者给定', '与本层输出同形')
    ids = ('ids', '整数 token 编号，不是连续特征', '(B,T)，不求导')
    lr = ('lr', '优化器学习率', '正实数标量')
    dims = {'B':'批次中序列数','T':'每条序列的 token 数','C':'输入特征宽度','D':'输出特征宽度','H':'注意力头数','d':'每头特征宽度','V':'词表大小','N':'样本数或展平的位置数，按本节定义','M':'索引输出行数','L':'文本长度或层数，按本节定义','Tq':'query token 数','Tk':'key/value token 数','dv':'value 特征宽度'}
    add('map','概念任务：区分训练和生成的数据流，不写代码。',[('ids','文本编码后的编号','(B,T)'),('logits','每个位置对词表中各类的分数','(B,T,V)'),('targets','每个位置的下一 token 编号','(B,T)')],'完成本页选择题。',['B 是序列数，T 是序列长度；C 是内部特征宽度，V 是词表大小。'],[],['训练：文本 → 编号窗口与右移标签 → logits → loss → 参数梯度 → 更新','生成：提示编号 → 末位置 logits → 采样编号 → 追加到提示'])
    add('architecture','概念任务：识别本项目的 decoder-only GPT 和 Block 数据流，不写代码。', [('X','Block 输入','(B,T,C)'),('H','头数，d=C/H','整数，C 可被 H 整除')],'完成结构对照题。',['主线：Pre-Norm、RMSNorm、GELU、可学习位置表；原始 encoder-decoder 是对照。'],[],['编号 → token/position 查表 → 多层 Block → 末尾 Norm → 共享输出头','Block：Norm→Attention→残差相加→Norm→MLP→残差相加'])
    add('shape','仅实现 solve(values)：创建数组并返回形状；不写 backward。',[('values','嵌套 Python 列表；网页测评可能已将其转为数组','规则多维结构'),('X','由 values 转成的实数数组','与 values 结构一致，float64')],'轴长度组成的一维 int64 数组，长度等于 X.ndim。',['返回 shape，不返回 X 本身。'],['将 values 转成 float64 NumPy 数组。','读取 X.shape。','将形状元组转成 int64 数组并返回。'],['嵌套列表 → X 的轴结构 → 每轴长度 → 形状数组'])
    add('axes','仅实现 solve(X)：对每行求和；不写 backward。',[('X','N 条样本，每条 C 个特征','(N,C)')],'每条样本的总和，形状 (N,)。',['固定 axis=1；keepdims=False。输入不变。'],['读取二维 X 的样本轴与特征轴。','沿 axis=1 求和。','返回一维 NumPy 数组。'],['X(N,C) → 每行的一组 C 个特征 → 沿特征轴归约 → y(N,)'])
    add('index','仅实现 solve(X,ids,k)：取行后截列；不写 backward。',[('X','原始样本表','(N,C)'),('ids','要依次取出的原行编号，可重复','(M,)，整数'),('k','保留前几列，不是固定写死的 2','整数标量，1≤k≤C')],'按指定顺序取行并保留前 k 列，形状 (M,k)。',['行顺序由 ids 决定；列区间 [0,k)。','网页/本地 k 为零维整数数组，可用 int(k)。旧草稿需补 k 参数。'],['按 ids 依次取出行，保留重复编号。','把 k 转成整数，保留列 0 到 k−1。','返回新数组，不修改输入。'],['X 和 ids → (M,C) 取行结果 → 按 k 截列 → (M,k)'])
    add('reduce','仅实现 solve(X)：每一行减去该行均值；本节不写 backward。',[('X','N 条样本，每条 C 个特征','(N,C)'),('μ','逐行均值，中间结果','(N,1)'),('Y','各行去均值后的结果','(N,C)')],'Y=X−μ，返回与 X 同形的数组。',['固定 axis=-1（二维中即 1），keepdims=True。','只求均值再相减，不缩放、不加偏置。'],['沿最后一轴求 mean，同时保留这一轴。','核对 μ 的形状是 (N,1)。','计算 X−μ，由广播将每个均值对齐到本行特征。','返回 Y，不修改 X。'],['X(N,C) → 沿特征求均值 μ(N,1) → 各行均值沿 C 复用 → 相减 Y(N,C)'])
    add('broadcast','仅实现 solve(X,b)：将同一个偏置加到所有位置；不写 backward。',[('X','输入；所有前置轴表示不同位置','(…,C)，本页基础测评为 (N,C)'),('b','每个特征对应一个偏置，各位置复用同一组偏置','(C,)')],'新数组 Y，与 X 同形。',['末轴长度相同；从右向左对齐。','使用加法返回新数组，不用 X+=b。'],['核对 b 的长度与 X 最后一轴相同。','直接进行数组加法，让 NumPy 沿前置位置广播。','返回结果，保持 X、b 不变。'],['X(…,C) + b(C,) → 末轴对齐 → 每个位置复用同一 b → Y(…,C)'])
    add('matmul','仅实现 solve(X,W)：矩阵乘法；不写 backward。',[('X','N 条样本的 C 个输入特征','(N,C)'),('W','将 C 个输入特征组合为 D 个输出特征','(C,D)')],'Y=X@W，形状 (N,D)。',['沿共同特征 C 乘加；不是逐元素 X*W。'],['核对 X 的列数等于 W 的行数。','用矩阵乘法得到每行与每列的乘加。','返回 (N,D) 数组。'],['X 的第 n 行 + W 的第 j 列 → 沿 C 乘加 → Y[n,j]'])
    add('linear','仅实现 solve(X,W,b) 的前向；完整 backward 在“Linear：前向、缓存与反向”任务实现。',[X,('W','共享权重','(C,D)'),('b','共享偏置','(D,)')],'Y 形状 (B,T,D)；二维边界输入则返回 (N,D)。',['保留原始前置形状；所有位置共享同一 W、b。'],['记录 X 原始形状，合并前置轴为 N。','用二维 X2@W+b 得到 Y2。','把输出恢复为原前置轴加 D，返回 Y。'],['X(B,T,C) → X2(B*T,C) → Y2(B*T,D) → Y(B,T,D)'])
    add('reshape','仅实现 solve(X)：合并 batch/token 轴并返回二维表；本节不写 backward 或恢复函数。',[('X','全部 batch/token 的特征','(B,T,C)')],'X2，形状 (B*T,C)。',['按 NumPy 默认 C 顺序，行 n=b*T+t。'],['读取 B、T、C。','仅合并前两个轴，保持特征顺序。','返回二维表；后续 Linear 再练习恢复形状。'],['(b,t,c) → 行号 n=b*T+t、列号 c → (N,C)'])
    add('transpose','仅实现 solve(X)：交换 token 与特征轴；本节不额外计算矩阵乘法。',[('X','三维特征张量','(B,T,C)')],'Y，形状 (B,C,T)。',['固定轴顺序 (0,2,1)，保留 batch。'],['确认输入轴依次为 B、T、C。','指定新轴顺序为 0、2、1。','返回换轴结果，不用三维 X.T。'],['X[b,t,c] → 坐标重新解释 → Y[b,c,t]'])
    add('heads','仅实现 solve(X,H)：先拆分特征，再换轴；本节不写注意力或 backward。',[('X','每个 token 的完整特征','(B,T,C)'),('H','头数；由调用方给定','整数标量'),('d','每头宽度，由 C/H 得到','整数')],'多头张量，形状 (B,H,T,d)。',['C 必须可被 H 整除。测试 H=2、3、4。','浏览器旧草稿需把 solve(X) 改为 solve(X,H)。'],['读取 B、T、C，将 H 转成整数，计算 d=C//H。','reshape 成 (B,T,H,d)，只拆最后一轴。','transpose(0,2,1,3)，把 head 轴移到 token 前。','返回全部 batch/head，不取单个切片。'],['(B,T,C) → 特征拆成 H×d → (B,T,H,d) → (B,H,T,d)'])
    add('copy','仅实现 solve(X)：修改独立副本；不写 backward。',[('X','原数组，不允许修改','二维非空数组'),('Y','独立副本','与 X 同形')],'副本 Y，其中 Y[0,0] 被替换成 −99。',['固定坐标 (0,0)，固定替换值 −99。'],['创建与 X 不共享修改的副本。','仅修改副本左上角。','返回副本并保留 X。'],['X → 独立副本 Y → 修改 Y[0,0] → 返回；X 保留原值'])
    add('scatter','仅实现 solve(ids,G,table)：执行给定梯度的散射累加，不新增 backward 方法。',[('ids','每条 G 回到哪一行','(M,)，整数'),('G','待加到词表中的各条贡献','(M,C)'),('table','累加起点，可能不是全零','(V,C)')],'table 的副本加上所有贡献，形状 (V,C)。',['重复编号要多次相加；table 的初始值保留。'],['复制 table，保留原始输入。','第 n 条 G 加到 ids[n] 指定的行。','用 np.add.at 等正确处理重复索引。','返回累加后的副本。'],['ids[n] 定位行 → G[n] 送到该行 → 在 table 初始值上累加 → 返回'])
    add('random','仅实现 solve(seed,dims)：创建局部随机生成器并采样；不写权重初始化层。',[('seed','重建随机序列使用的种子','整数标量'),('dims','输出每个轴的长度','一维整数数组')],'标准正态样本数组，形状 tuple(dims)。',['用 default_rng(seed)，不改变全局随机状态。'],['将 seed 转成整数，dims 转成整数形状元组。','用给定种子创建局部 Generator。','按 dims 采标准正态样本并返回。'],['seed → 局部 Generator → standard_normal(dims) → 指定形状样本'])
    add('infrastructure','实现 Parameter.__init__/zero_grad 和 Module.parameters/zero_grad；本节不实现网络前向或反向。',[('data','参数数值的独立 float64 数组','任意实数形状'),('grad','梯度累加缓冲区，初始全零','与 data 同形'),('parameters()','递归收集的可训练 Parameter','按顺序列表，共享对象去重')],'按接口创建/清零参数；parameters 返回参数列表。',['只遍历参数和子 Module；跳过普通数组缓存。','zero_grad 不改变 data；共享参数只出现一次。'],['Parameter 复制 data，创建同形零 grad。','zero_grad 原地将 grad 清零。','递归遍历 Module 属性及容器，按对象身份收集去重的 Parameter。','Module.zero_grad 调用所有参数的 zero_grad。'],['模块属性/容器 → 找到 Parameter → 按身份去重 → 参数列表 → 清零 grad'])
    add('initialization','概念任务：判断 fan-in 与残差出口的初始化规则，不写代码。',[('C','Linear 输入宽度','正整数'),('L','残差 Block 层数','正整数')],'完成初始化理解题。',['本页基础层 fan-in 标准不等于 GPT 词表及所有权重的固定初始化。','GPT 的二维权重按均值 0、标准差 0.02 初始化，残差出口再缩放。'],[],['输入宽度 C → 基础层初始标准差 → 残差出口的层数缩放'])
    add('layer_scale','实现 Scale.forward(X) 与 backward(G)：返回输入梯度，并累加标量参数梯度。',[X,G,('w','可训练缩放参数','标量'),('cached X','本次前向输入，用于 dw','与 X 同形')],'forward 返回同形 Y；backward 返回同形 dX，同时累加 w.grad。',['缓存 X，不依赖 w=0 时无法逆推的 Y。'],['forward 保存 X，返回 w*X。','backward 计算 G*w。','将所有位置的 G*X 求和，累加到 w.grad。','返回 dX；不在 backward 更新 w.data。'],['X,w → Y → 给定 G → dX=G*w；各位置贡献汇总到 dw'])
    add('layer_linear','实现 Linear.forward(X) 与 backward(G)：补齐缓存、输入梯度、权重和偏置梯度。',[X,G,('W','可训练共享权重','(C,D)'),('b','可选可训练偏置','(D,)，use_bias=False 时不存在'),('X2/G2','合并前置位置后的输入/上游','(N,C)/(N,D)')],'forward 返回 (…,D)；backward 返回与 X 同形 dX，累加 W.grad 和可选 b.grad。',['构造器已给出，保留属性名；支持任意前置轴。','N 为所有前置轴之积；损失若已平均，本层不再平均。'],['前向缓存原始形状和 X2，计算 Y2 并恢复输出形状。','把 G 变成 G2，与 X2 的位置行一一对应。','计算 dX2=G2@W.T 并恢复原形。','将 X2.T@G2 累加到 W.grad；可选偏置沿 N 求和并累加。','返回 dX，不修改输入和参数 data。'],['X → 展平位置 → Y2 → 恢复；G → dX，X/G → dW，G 沿位置归约 → db'])
    add('layer_embedding','实现 Embedding.forward(ids) 与 backward(G)：整数编号不求导，backward 返回 None。',[ids,G,('weight/E','可训练词表','(V,C)'),('V/C','词表行数/每行特征宽度','正整数')],'forward 返回 (B,T,C)；backward 返回 None，只累加 weight.grad。',['重复编号指向同一行，跨 batch 也要累加。'],['前向缓存 ids，按编号取 weight 对应行。','反向把 ids 和 G 的位置轴展平。','按 ids 将每条 G 散射加到 weight.grad。','返回 None，不对整数 ids 求导，也不更新 weight.data。'],['ids[b,t] → E 对应行 → Y[b,t,:]；G[b,t,:] → 相同 E 行的梯度'])
    add('layer_gelu','实现 GELU.forward(X) 与 backward(G)；本层没有参数，不累加参数梯度。',[X,G,('k/a','近似公式的固定系数','k=√(2/π)，a=0.044715'),('u/t','中间量 u=k(x+ax³)、t=tanh(u)','与 X 同形')],'forward 返回同形 Y；backward 只返回同形 dX。',['使用 tanh 近似式；缓存 X 与 t。'],['前向计算 u、t，缓存 X 与 t。','返回 0.5*X*(1+t)。','按乘积法则和链式法则求局部导数。','逐元素乘上游 G，返回 dX。'],['X → u → tanh 得 t → Y；G 沿乘积的两条路径回到 X'])
    add('layer_rmsnorm','实现 RMSNorm 的前向、输入梯度及 w.grad 累加。',[X,G,('w','可训练逐特征缩放','(C,)'),('m/r/xhat','均方/逆均方根/归一化输入','m、r 为 (…,1)，xhat 与 X 同形'),('eps','避免除零的固定构造值','正实数，默认 1e−5')],'返回同形 Y、dX；累加 w.grad。',['mean 只沿最后特征轴，keepdims=True；dw 沿所有前置位置求和。'],['计算每个 token 的均方、r 与 xhat，缓存 r/xhat。','输出 xhat*w。','计算 u=G*w，合并直接和经过均方值的输入梯度路径。','沿所有位置将 G*xhat 累加到 w.grad，返回 dX。'],['每个 token 的 X → m(…,1) → r → xhat → Y；反向合并直接/均方两路'])
    add('layer_mlp','用自己的 Linear/GELU 实现 MLP.forward 和 backward；本节组合已有层。',[X,G,('hidden','中间特征宽度','正整数'),('fc1/act/fc2','两层 Linear 和一层 GELU','C→hidden→C')],'Y/dX 与 X 同形；两个 Linear 的参数梯度由子层累加。',['前置模块用个人草稿；每个 token 共享同一组参数，不在此混合位置。'],['前向依次调用 fc1、act、fc2，返回最终 Y。','反向把 G 送入 fc2.backward。','将得到的梯度送入 act.backward。','将新梯度送入 fc1.backward，返回 dX。'],['X → fc1 → GELU → fc2 → Y；G 按相反顺序回流'])
    add('t_softmax','实现 Softmax.forward(X) 和 backward(G)；本层无参数。',[('X','每个 token 的类别分数 logits','(B,T,V)'),G,('P','缓存的逐位置概率','与 X 同形')],'forward 返回 P；backward 返回同形 dX。',['只沿最后类别轴；减最大值、求和均 keepdims=True。'],['逐位置减去类别最大值。','指数化，除以同位置全部类别的指数和，缓存 P。','反向先求 sum(G*P,axis=-1,keepdims=True)。','返回 P*(G−该和)，不构造完整雅可比。'],['logits → 行最大值平移 → 指数 → 行归一化 P；G → 行内耦合归约 → dX'])
    add('t_loss','实现 CrossEntropy.forward(logits,targets) 与 backward(g)。',[('logits（代码 X）','每个位置各类别分数','(B,T,V)'),('targets','正确类别整数编号','(B,T)'),('g','损失标量上的上游梯度，默认 1','标量'),('N','参与平均的位置总数','N=B*T')],'forward 返回零维 NumPy 平均损失；backward 返回 (B,T,V)，不返回标签梯度。',['稳定计算 log 概率；N 只除一次。'],['合并位置轴，逐位置减最大值并求 LogSumExp。','按 targets 取正确类别的 log 概率。','取负值，对全部 N 个位置平均并返回零维数组。','反向在概率的正确类别处减 1，再乘 g/N 并恢复 logits 形状。'],['logits → 稳定 logp → targets 选中一类 → N 个位置平均 loss → g → dlogits'])
    add('att_scores','仅实现 solve(Q,K)：注意力打分前向；不写 backward。',[('Q','query 特征','(B,H,Tq,d)'),('K','key 特征','(B,H,Tk,d)'),('d','匹配特征宽度','Q/K 最后一轴的长度')],'S，形状 (B,H,Tq,Tk)。',['仅转置 K 的最后两轴；Tq 与 Tk 可不同。'],['读取 d=Q.shape[-1]。','把 K 最后两轴交换为 (…,d,Tk)。','计算 Q@Kᵀ。','除以 √d，返回完整 S。'],['Q 的 query 行 + K 的 key 行 → 沿 d 点积 → 缩放 → 每对位置的分数'])
    add('att_probs','仅实现 solve(S)：因果掩码与逐行 Softmax 前向；不写 backward。',[('S','未掩码分数','(B,H,T,T)'),('mask','key 位置 s>query 位置 t 的未来区域','(T,T)，广播到 B/H')],'A，与 S 同形；未来位置为 0，每行和为 1。',['仅接受方形因果自注意力。先掩码再 Softmax。'],['依据最后两轴的行/列编号，生成 s>t 掩码。','将未来位置分数置 −∞。','沿 key 轴减最大值、指数化、除以行和。','返回 A，不修改 S。'],['S → 标出未来区域 → −∞ 掩码分数 → 沿 key 稳定 Softmax → A'])
    add('att_weighted','仅实现 solve(A,V)：用概率读取 value；不写 backward。',[('A','每个 query 对 key 的读取概率','(B,H,Tq,Tk)'),('V','每个 key 位置携带的内容','(B,H,Tk,dv)')],'Y，形状 (B,H,Tq,dv)。',['沿 key 轴 Tk 乘加，dv 不必等于打分宽度 d。'],['核对 A 最后一轴和 V 倒数第二轴一致。','用 A@V，逐 query 对 value 向量加权。','保留 B、H、Tq 和 dv 四个输出轴。'],['A 的 query 行 → 指向全部 key/value → 按读取概率组合内容 → Y 的 query 向量'])
    add('att_merge','仅实现 solve(X)：恢复 token 的多头输出；不写 backward。',[('X','各 head 的输出','(B,H,T,d)')],'Y，形状 (B,T,H*d)。',['先交换 H/T，再合并 H*d；保持 head 顺序。'],['读取 B、H、T、d。','transpose(0,2,1,3) 得到 (B,T,H,d)。','reshape 为 (B,T,H*d)，返回结果。'],['所有 head 的同一 token → 先放在同一 token 下面 → 按 head 顺序拼接特征'])
    add('t_attention','实现 AttentionCore.forward(Q,K,V) 与 backward(G)，返回三路梯度。',[('Q/K','query/key 特征','(B,H,Tq,d)/(B,H,Tk,d)'),('V','内容特征','(B,H,Tk,dv)'),G,('causal','是否屏蔽未来位置','布尔构造条件')],'forward 返回 (B,H,Tq,dv)；backward 按 dQ,dK,dV 顺序返回元组。',['无参数；缓存 Q/K/V、A、mask。非因果模式可用不同 Tq/Tk。'],['计算缩放 QKᵀ，按 causal 决定是否在 Softmax 前掩码。','逐 query 行计算 A，返回 A@V 并缓存。','反向先计算 dV=AᵀG、dA=GVᵀ。','使用 Softmax 的耦合导数得到 dS，并清零屏蔽位置。','计算带缩放的 dQ、dK，按规定顺序返回。'],['Q/K → S → mask → A；A/V → Y；G → dV/dA → dS → dQ/dK'])
    add('t_mha','实现 MultiHeadAttention.forward/backward：串联个人 Linear 与 AttentionCore。',[X,G,('C/H/d','模型宽度/头数/每头宽度','d=C/H，C%H=0'),('qkv','一次得到 Q/K/V 的投影层','C→3C'),('proj/core','输出投影/注意力核心','C→C / 每头独立')],'Y、dX 均为 (B,T,C)；各投影参数梯度由子层累加。',['所有头和位置共享各投影层；构造器已提供。'],['qkv 投影后沿特征切成 Q/K/V 三块。','各块先拆 (H,d)，再换到 (B,H,T,d)。','调用 core，把其输出先换轴再合头，送入 proj。','反向先 proj，再拆头并调用 core.backward。','将 dQ/dK/dV 分别逆向合头，按原顺序拼接后交给 qkv.backward。'],['X → qkv → Q/K/V 分头 → core → 合头 → proj → Y；反向逐级逆序'])
    add('t_block','实现 Block.forward/backward：两条残差各自保留直连和子层路径。',[X,G,('X1','第一条残差相加后的中间结果','(B,T,C)'),('norm1/attn/norm2/mlp','四个独立子模块','输入输出保持 (B,T,C)')],'Y、dX 同形；所有子层参数梯度累加。',['两个 Norm 用不同实例；每次分叉传完整上游梯度。'],['计算 X1=X+attn(norm1(X))。','计算 Y=X1+mlp(norm2(X1))。','从 Y 回流：将 G 经过 mlp、norm2 后，与直连 G 相加。','把 G1 经过 attn、norm1 后，与直连 G1 相加并返回。'],['X → 第一残差得到 X1 → 第二残差得到 Y；每次反向分叉→子层→同一变量处相加'])
    add('t_dropout','实现 Dropout.forward(X,training) 和 backward(G)；没有参数。',[X,G,('p/seed','丢弃概率/局部 RNG 种子','0≤p<1 / 整数'),('mask','保留为 1/(1−p)、丢弃为 0 的缓存','与 X 同形'),('training','本次是否训练','布尔值')],'Y、dX 与 X 同形。',['前向与反向复用同一 mask；评估不采样。'],['训练时采保留掩码，按保留概率倒数缩放并缓存。','输出 X*mask；评估输出同值数组并使用全 1 掩码。','反向返回 G*当前缓存 mask，不能重采样。'],['training 开关 → 本次 mask → X 相乘 → Y；同一 mask → G 相乘 → dX'])
    add('t_layernorm','实现 LayerNorm.forward/backward，包括 w、b 参数梯度。',[X,G,('μ/r/h','特征均值/逆标准差/标准化输入','μ、r 为 (…,1)，h 与 X 同形'),('w/b','缩放/偏置','各 (C,)'),('eps','除零保护','正实数')],'Y、dX 同形；累加 w.grad、b.grad。',['特征轴上中心化；参数梯度归约全部前置位置。'],['沿特征求均值，计算中心化输入和方差。','加 eps，计算 r/h 并缓存；输出 h*w+b。','反向计算 u=G*w，合并中心化与方差路径。','将 G*h、G 沿前置位置求和并累加，返回 dX。'],['X → μ → 中心化 → 方差/r → h → Y；G → 两路归约合流 → dX/dw/db'])
    gpt_vars=[ids,G,('V/C/H','词表大小/模型宽度/头数','正整数，C%H=0'),('hidden/layers','MLP 中间宽度/Block 层数','正整数'),('context/seed','最大序列长度/初始化 RNG 种子','正整数 / 整数'),('token/position','共享 token 表/位置表','(V,C)/(context,C)')]
    gpt_steps=['按 ids 查 token 表，按位置查 position 表并沿 batch 广播相加。','依次通过全部 Block 和末尾 Norm。','用同一 token.weight 的转置计算 logits。','反向先累计输出头对共享表的梯度，再得到末尾激活梯度。','经 Norm 与反序 Blocks 回流，将输入查表和位置表梯度分别累加。','返回 None；共享参数只收集一次。']
    add('t_gpt','实现 GPT.forward(ids)/backward(G)；普通任务的构造器已给出。',gpt_vars,'forward 返回 (B,T,V) logits；backward 返回 None，累加全部参数梯度。',['T≤context；token 表与输出头共享同一个 Parameter。'],gpt_steps,['ids/位置 → 两个表相加 → Blocks → Norm → 共享头 → logits；两条共享表梯度相加'])
    add('t_data','实现 tokenize(text) 和 make_batch(ids,starts,T) 两个函数；不写 backward。',[('text','用于建字符词表的文本','字符串'),('ids','编码后的完整文本序列','(L,)，int64'),('starts','各条样本窗口起点','(B,)，整数'),('T','窗口长度','正整数')],'tokenize 返回 (vocab,ids)：排序字符列表与编码数组；make_batch 返回 X/Y 两个 (B,T) 整数数组。',['每个起点 s 满足 0≤s 且 s+T<L；Y 比 X 向后移一个字符。'],['将 text 字符排序去重，建立字符到编号的映射。','按词表编码 text 为 int64 ids。','对每个 starts[b]，截取长度 T 的输入窗口。','从 starts[b]+1 截取标签窗口，堆叠所有 batch 并返回 X,Y。'],['text → 词表/ids → starts 指定窗口 X → 向右一位的标签 Y'])
    add('t_optim','实现 clip_grad 和 AdamW.step；不实现网络 backward。',[('parameters','去重的可训练参数列表','每项有 data/grad'),('limit','全局梯度范数上限','正实数'),lr,('m/v/t','一阶矩/二阶矩/累计更新步数','各参数同形数组/整数'),('beta1/beta2/eps/decay','矩平滑系数/除零保护/衰减系数','构造配置标量')],'clip_grad 原地缩放 grad 并返回裁剪前零维范数；step 原地更新 data，无返回数组。',['全局统一裁剪；step 不清零 grad。高维权重衰减，一维 bias/norm 不衰减。'],['汇总全部 grad 的平方，开方得到全局范数。','计算一个公共缩放系数，原地裁剪所有 grad。','AdamW 每次 step 增加 t，更新 m/v 并做偏差校正。','按参数维数决定衰减，再用校正矩更新 data。'],['全部 grad → 一个范数/比例 → 裁剪 → m/v/t → 偏差校正 → data 更新'])
    train_vars=[('model','已构造的个人 GPT','模型对象'),('x/ids','固定批次输入编号','(B,T)'),('y/targets','下一 token 标签','(B,T)'),('steps','要更新的步数','正整数'),lr]
    train_steps=['循环外创建一次 CrossEntropy 和 AdamW，不在每步重建优化器。','每步先 model.zero_grad，再计算 logits 和平均 loss。','记录更新前的 loss 标量。','loss_fn.backward 产生 dlogits，交给 model.backward。','全局裁剪到 1，执行 optimizer.step；最后返回损失数组。']
    add('t_train','实现 train(model,x,y,steps,lr)：调用已完成的个人模块，完成多步训练。',train_vars,'长度 steps 的 float64 损失数组；模型参数在函数中更新。',['CrossEntropy；AdamW decay=.01；clip_grad limit=1。','先反向再评估，避免覆盖训练缓存；优化器跨步保留 m/v/t。'],train_steps,['清零 → GPT → loss并记录 → loss反向 → GPT反向 → 全局裁剪 → 更新；循环 steps 次'])
    add('t_sample','仅实现 sample(logits,temperature,top_k,rng)：每行抽一个 token；不写 backward。',[('logits（代码 X）','各 batch 下一 token 类别分数','(B,V)'),('temperature','温度 τ','正实数'),('top_k','只保留最大几类，0 表示全保留','整数，0…V'),('rng/seed','调用方的随机生成器/公开用例创建它的种子','Generator / 整数')],'每行一个抽样编号，形状 (B,)，int64。',['并列按稳定 argsort 保留末尾 k 项；不能在函数内重置 RNG。'],['复制分数并除以温度。','top_k>0 时将其他类别置 −∞。','逐行计算稳定 Softmax。','使用传入 rng 按概率采样，每行返回一个编号。'],['logits → 温度缩放 → top-k 保留集合 → 行概率 → rng 抽样编号'])
    add('t_generate','仅实现 generate(model,ids,steps,temperature,top_k,seed)：调用个人模型与采样；不写 backward。',[('model','已构造、已有参数的个人 GPT','模型对象，含 context'),('ids','完整提示编号','(B,T0)，int64，T0≥1'),('steps','追加多少个 token','非负整数'),('temperature/top_k/seed','采样温度/保留类别数/本次 RNG 种子','正实数/整数/整数'),('context','每次送入模型的最大窗口长度','model.context')],'完整提示加新 token，形状 (B,T0+steps)。',['只前向；每轮最多取末 context 个编号；输出不能截掉原提示。'],['复制提示，按 seed 创建一次局部 RNG。','每轮截取当前序列最后 context 个 token。','模型前向后只取最后位置的 logits。','调用 sample 得到每个 batch 的新编号并追加。','重复 steps 次，返回完整序列。'],['完整序列 → 截取末尾窗口 → 末位置 logits → 采样 → 追加到完整序列 → 下一轮'])
    add('t_checkpoint','实现 save_checkpoint 和 load_checkpoint；本节保存/恢复状态，不写网络 backward。',[('path','检查点文件路径','路径'),('model/optimizer/rng','个人 GPT、AdamW、由 default_rng 创建的生成器','已有对象'),('状态','参数、m/v/t、学习率配置及 RNG 状态','数组和具名元数据')],'save 写检查点；load 原地恢复已有对象的状态，清零遗留 grad。',['参数按固定顺序保存为 p0、p1…，读取时核对形状；用 allow_pickle=False。'],['按模型参数顺序收集 data 和优化器 m/v。','把 t、学习率配置、RNG 状态写入具名元数据；参数形状随数组保存。','核对对应参数的形状，恢复 data/m/v/t/配置。','恢复 RNG，清零旧 grad；验证同一步续训结果一致。'],['模型/优化器/RNG → 数组+具名元数据 → 检查点 → 检查匹配 → 恢复 → 精确续训'])
    add('t_project','只实现 project_step(model,optimizer,ids,targets)：组合个人模块完成一次更新，不重写 GPT。',[('model/optimizer','调用方已经创建的个人模型/AdamW','对象'),ids,('targets','下一 token 标签','(B,T)')],'更新前平均损失的零维 NumPy 数组；同时完成一次参数更新。',['其他综合检查由测评器调用；不是要求在 project_step 内重复执行八类检查。'],['清空 model 旧梯度。','计算 logits 和 CrossEntropy 损失。','经 loss 与 GPT 完整反向。','全局裁剪到 1，再 optimizer.step。','返回这次更新前的 loss。'],['个人模型和输入 → 一次前向/损失 → 全反向 → 裁剪/更新 → 返回更新前 loss'])
    # 独立实现复用定义，但构造器也属于交付，不能沿用普通任务的范围描述。
    for id, base, scope, initial in [
        ('rebuild_linear','layer_linear','独立实现 Linear.__init__/forward/backward：构造器和两个方法都要写。','创建 W 与可选 b，使用约定初始化；保存输入/输出特征宽度。'),
        ('rebuild_block','t_block','独立实现 Block.__init__/forward/backward：自己创建四个子模块。','创建两个独立 Norm、Attention、MLP，并保存为公开属性。'),
        ('rebuild_gpt','t_gpt','独立实现 GPT.__init__/forward/backward：构造器也需自行实现。','创建 token/position、独立 Blocks 和 Norm；用 seed 对二维权重依次初始化，共享输出词表并缩放残差出口一次。')]:
        base_spec=specs[base]
        conditions=[c for c in base_spec['conditions'] if '构造器已' not in c]
        add(id,scope,[(v['name'],v['meaning'],v['shape']) for v in base_spec['variables']],base_spec['output'],conditions,[initial]+lessons[base]['steps'],base_spec['flow'])
    add('fix_bias','拓展修错：只修复 solve(G) 的偏置归约轴，不重写 Linear。',[('G','损失对 Linear 输出的梯度','(B,T,D)')],'db，形状 (D,)。',['仅归约 batch/token，保留输出特征 D。'],['检查错误实现保留了哪些轴。','沿所有前置位置求和，保留末特征轴。','返回 db。'],['所有位置同一输出特征的 G → 沿 B/T 合流 → db[D]'])
    add('fix_heads','拓展修错：修复 solve(X) 合头的 token/head 对应关系。',[('X','多头输出','(B,H,T,d)')],'恢复的 token 特征，形状 (B,T,H*d)。',['数量相同不代表位置正确；目标 Y[b,t,h*d+j]=X[b,h,t,j]。'],['先交换 head 与 token 两轴。','再合并 H*d 特征。','核对同一 token 的头特征按 head 顺序拼接。'],['(b,h,t,j) → (b,t,h,j) → (b,t,h*d+j)'])
    add('fix_residual','拓展修错：合并同一输入收到的两条梯度，不额外执行子层反向。',[('direct','直连分支回到输入的梯度','与输入同形'),('branch','子网络已算出的输入梯度','与 direct 同形')],'两条梯度之和，与输入同形。',['两路均已给定，不能只保留一路或除以 2。'],['确认 direct、branch 是同一输入的两路贡献。','相加并返回，保持两者不变。'],['直连贡献 + 子网络贡献 → 同一输入处相加'])
    add('fix_shared','拓展修错：修复给定 GPT.backward 的共享词表梯度，保留个人前置模块。',gpt_vars,'backward 返回 None；E.grad 包含输出头和输入查表两路贡献。',['不新建另一份词表，不通过修改前置模块绕过错误。'],['保留输出头对词表的梯度。','将梯度经 Norm 与反序 Blocks 回流。','把输入查表贡献累加到同一 E.grad，不能覆盖输出头贡献。'],['共享输出头 dE + 输入查表 dE → 同一词表参数 grad'])
    add('fix_clear','拓展修错：修复给定 train 的跨步梯度残留；训练接口保持不变。',train_vars,'更新前损失数组 (steps,)；模型按正确梯度更新。',['每步反向前清零旧梯度，优化器状态跨步保留。'],['定位新一步训练开始处。','在本步前向/反向前调用 model.zero_grad。','保留当前步的反向、裁剪、更新和损失记录顺序。'],['上一步更新完 → 清零旧 grad → 当前步前向/反向 → 当前步更新'])

    add('mean_backward','只实现 solve(G,C)：均值操作的反向，不重算前向。',[('G','损失对每个 token 均值结果的梯度','(B,T,1)，可有更多前置轴'),('C','前向被归约的特征数','正整数标量')],'各输入特征收到的 dX，形状 (B,T,C)，保留全部前置轴。',['前向固定 axis=-1、keepdims=True；均值导数为 1/C。'],['将 C 转成整数。','把 G 除以 C。','将末轴长度 1 广播到 C，保留其他轴。','返回独立数组 dX。'],['均值结果上的 G(…,1) → 每项分得 G/C → 沿特征轴广播 → dX(…,C)'])
    add('center_backward','只实现 solve(G)：对应 Y=X−mean(X) 的反向。',[('G','损失对去均值输出 Y 的梯度','(B,T,C)，支持更多前置轴')],'dX，与 G 同形。',['沿最后特征轴求均值，keepdims=True；本任务不求参数梯度。'],['直接路径保留完整 G。','计算 G 沿最后特征轴的均值，保留轴长 1。','用 G 减去该均值，合并直接和均值两条路径。','返回 dX，不能只将 G 除以 C。'],['直接贡献 G + 均值路径的负贡献 → 合流 → dX；每行梯度和为零'])
    add('restore','仅实现 solve(Y2,out_shape)：将二维位置表恢复；不重算 Linear。',[('Y2','所有位置排成的二维输出表','(N,D)'),('out_shape','目标全部轴长度，包含输出特征 D','整数数组，例如 [B,T,D]')],'按 out_shape 恢复后的数组。',['元素总数一致；三维时 N=B*T。按默认 C 顺序。'],['将 out_shape 转为整数形状元组。','让 Y2.reshape 到目标形状。','返回恢复结果，保留各 token 的特征顺序。'],['二维行 n → 原位置 b,t → 同一特征 j → Y[b,t,j]'])
    add('random_sequence','仅实现 solve(seed,dims)：同一 RNG 连续采样两份数组。',[('seed','本次序列的种子','整数标量'),('dims','每份样本的轴长度','一维整数数组'),('rng','只创建一次的局部生成器','Generator')],'两份样本堆叠，形状 (2,*dims)，float64。',['第一轴 2 是采样轮次；不要将 batch 或其他原轴设为 1 简化。'],['根据 seed 创建一次局部 Generator。','按 dims 采第一份标准正态数组。','继续使用同一个 rng 采第二份。','按先后顺序 np.stack，返回两份。'],['seed → 初始 RNG → 第一份并推进状态 → 第二份 → 按轮次堆叠'])
    add('att_value_backward','仅实现 solve(A,G)：加权读取的 dV，不重算注意力前向。',[('A','各 query 读取各 key 的概率','(B,H,Tq,Tk)'),('G','损失对加权输出 Y 的梯度','(B,H,Tq,dv)')],'dV，形状 (B,H,Tk,dv)。',['沿 query 位置汇总；只交换 A 的最后两轴。'],['将 A 最后两轴转置为 (…,Tk,Tq)。','与 G 矩阵乘法，沿 Tq 汇总。','返回全部 batch/head 的 dV。'],['同一 key 被多个 query 读取 → 各 query 的 G 按 A 回流 → 汇总到 dV[key]'])
    add('att_score_backward','仅实现 solve(A,dA)：Softmax 的反向，返回分数梯度 dS。',[('A','前向缓存的行概率','(B,H,Tq,Tk)'),('dA','损失对概率 A 的上游梯度','与 A 同形')],'dS，与 A 同形。',['沿最后 key 轴求和，keepdims=True；不构造雅可比。'],['逐元素计算 A*dA。','沿最后 key 轴求和，保留轴长 1。','从 dA 减去该和，再逐元素乘 A。','返回 dS；不能直接用 A*dA。'],['dA 与 A → 行内公共耦合项 → 从每项 dA 扣除 → 乘 A → dS'])
    if 'foundation_capstone' in lessons:
        add('foundation_capstone','组装个人归一化与MLP分支',[X,G],'同形输出与输入梯度。',['复用个人子层；参数梯度留在子层。'],[],['X → RMSNorm → MLP → Y'])
    if 't_init_upgrade' in lessons:
        specs['t_init_upgrade']=lessons['t_init_upgrade']['task_brief']
    if set(specs) != set(lessons):
        raise ValueError('任务契约未覆盖课程：'+str(set(lessons)^set(specs)))
    for id, meta in lessons.items():
        names=' '.join(v['shape'] for v in specs[id]['variables'])
        specs[id]['dimensions']={name:meaning for name,meaning in dims.items() if name in names}
        if id in ('layer_embedding','layer_gelu','rebuild_gpt'):
            meta['goal']=specs[id]['scope']
    lessons['rebuild_gpt']['body']=lessons['rebuild_gpt']['body'].replace('构造器给定，完成 forward/backward。','构造器也需要自己实现，再完成 forward/backward。')
    lessons['rebuild_linear']['task_brief']['conditions']=[c for c in lessons['rebuild_linear']['task_brief']['conditions'] if '构造器' not in c]
    lessons['infrastructure']['exports']=['Parameter','Module']
    for id, meta in lessons.items():
        if meta['kind'] in ('layer','project'):
            meta['dependencies']=list(dict.fromkeys(['infrastructure']+meta.get('dependencies',[])))
            meta['starter']=meta['starter'].replace('from foundation_core import','from learner_core import')
            if meta.get('experiment_dependencies'):
                meta['experiment_dependencies']=list(dict.fromkeys(['infrastructure']+meta['experiment_dependencies']))
        cls=meta.get('class') or (meta.get('cases') or [{}])[meta.get('sample_index',0)].get('input',{}).get('class') if meta.get('kind')=='project' else meta.get('class')
        if id in ('t_train','t_generate','t_checkpoint','t_project','fix_clear'):cls='GPT'
        meta['constructor_names']={
            'Scale':['value'],'Linear':['C','D','use_bias'],'Embedding':['V','C'],'GELU':[],
            'RMSNorm':['C','eps'],'MLP':['C','hidden'],'Softmax':[], 'CrossEntropy':[],
            'AttentionCore':['causal'],'MultiHeadAttention':['C','H'],'Block':['C','H','hidden'],
            'Dropout':['p','seed'],'LayerNorm':['C','eps'], 'GPT':['V','C','H','hidden','layers','context','seed']
        }.get(cls,[])
    lessons['t_generate']['sample_fields']=['init','ids','steps','temperature','top_k','seed']
    integrations={
        'shape':'layer_linear','axes':'layer_linear','index':'layer_embedding','reduce':'center_backward',
        'mean_backward':'layer_rmsnorm','center_backward':'t_layernorm','broadcast':'layer_linear',
        'matmul':'layer_linear','linear':'layer_linear','reshape':'restore','restore':'layer_linear',
        'transpose':'t_mha','heads':'att_merge','copy':'layer_linear','scatter':'layer_embedding',
        'random':'random_sequence','random_sequence':'t_checkpoint','infrastructure':'layer_linear',
        'initialization':'rebuild_gpt','layer_scale':'layer_linear','layer_linear':'layer_mlp',
        'layer_embedding':'t_gpt','layer_gelu':'layer_mlp','layer_rmsnorm':'t_block','layer_mlp':'t_block',
        'att_scores':'t_attention','att_probs':'t_attention','att_weighted':'att_value_backward',
        'att_merge':'t_mha','att_value_backward':'t_attention','att_score_backward':'t_attention',
        't_softmax':'att_score_backward','t_loss':'t_train','t_attention':'t_mha','t_mha':'t_block',
        't_block':'t_gpt','t_dropout':'t_gpt','t_layernorm':'architecture','t_gpt':'t_train',
        't_data':'t_train','t_optim':'t_train','t_train':'t_project','t_sample':'t_generate',
        't_generate':'t_project','t_checkpoint':'t_project','rebuild_linear':'rebuild_block',
        'rebuild_block':'rebuild_gpt','t_project':'rebuild_gpt','map':'t_gpt','architecture':'t_block'}
    for id,to in integrations.items():
        lessons[id]['task_brief']['next_use']=dict(lesson=to,mode='contrast' if id in ('t_dropout','t_layernorm') else 'practice' if lessons[id]['kind']=='array' or id=='layer_scale' else 'reuse')
    for id, dtype in [('shape','int64'),('random','float64'),('random_sequence','float64')]:
        for case in lessons[id]['cases']:
            case['expected_dtype']=dtype
    # 用例名称只描述实际覆盖的行为，避免共用模板误导学习者。
    for id, meta in lessons.items():
        for case in meta.get('cases',[]):
            label=case.get('label','')
            if label=='零值/不同轴长/重复编号的结构检查':
                case['label']={'index':'不同列数与重复行编号','random':'新种子与不同维数的输出','heads':'H=3 与不同 token 数的坐标映射','shape':'四维输入的轴长度','axes':'全零输入与不同特征数','broadcast':'不同特征宽度的末轴对齐','matmul':'零权重矩阵与非方形输出','linear':'零权重时偏置沿所有位置复用','reduce':'常数行去均值应全零','reshape':'不同轴长的展平顺序','transpose':'不同轴长的换轴结果','copy':'全零输入复制后仅改一格','scatter':'非零初始表与重复编号累加'}.get(id,'不同输入结构检查')
            if meta.get('class')=='GELU' and '所有输入与参数梯度' in label:
                case['label']=label.replace('所有输入与参数梯度','所有输入梯度')
            if meta.get('class')=='Embedding' and '所有输入与参数梯度' in label:
                case['label']='零上游：词表梯度为零，整数编号梯度返回 None'
    book['clarity_revision']=1
    return book


if __name__=='__main__':
    target=ROOT/'webapp/foundations.json'
    book=json.loads(target.read_text(encoding='utf-8'))
    apply_refinements(book)
    target.write_text(json.dumps(book,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays=json.loads((ROOT/'exercises/cases.json').read_text(encoding='utf-8'))
    for id,meta in book['lessons'].items():
        if meta['kind']=='array':arrays[id]=meta['cases']
    (ROOT/'exercises/cases.json').write_text(json.dumps(arrays,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 能力、练习与最终复用对照','', '这里记录实际课程映射。数组短题练操作，后续层需自己写入这些操作，不自动调用短题的 solve。代码测评通过不等于学习者已验收动画。','', '| 任务 | 本次动手范围 | 公开检查 | 后续应用/对照 |','|---|---|---|---|']
    for stage in book['stages']:
        for module in stage['modules']:
            for id in module['tasks']:
                meta=book['lessons'][id];t=meta['task_brief'];use=t.get('next_use',{}).get('lesson')
                categories='、'.join(dict.fromkeys(c.get('category','数值与形状') for c in meta.get('cases',[])))
                count=len(meta.get('questions',[])) if meta['kind']=='concept' else len(meta['cases'])
                lines.append(f'| {id} · {meta["title"]} | {t["scope"]} | {count} 项：{categories or "概念判断"} | {use or ("拓展修错，不计主线" if stage.get("optional") else "最终独立交付")} |')
    lines+=['','尚待补齐的连续过程动画见 webapp/visual-roadmap.json；本表不把文字流程当作动画已完成。','']
    (ROOT/'docs/能力覆盖对照.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Refined task scope, variables, conditions and code steps for all',len(book['lessons']),'lessons.')
