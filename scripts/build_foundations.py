"""构建第一、二、四部分的课程目录和本地骨架；绝不覆盖已有个人练习。"""
import ast
import inspect
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'exercises'))
sys.path.insert(0,str(ROOT/'solutions'))
import foundation_layers as ref
import foundation_core as core

legacy=json.loads((ROOT/'webapp/legacy-lessons.json').read_text(encoding='utf-8'))
cases=json.loads((ROOT/'exercises/cases.json').read_text(encoding='utf-8'))
lessons={}


def add(id,title,goal,body,**kw):
    lessons[id]=dict(id=id,title=title,goal=goal,body=body,**kw)


def section(title,text):
    return f'<h3>{title}</h3><p>{text}</p>'


add('map','从文本到训练和生成','区分 token、编号、向量，以及训练与生成的输入输出。',
section('一条具体数据流','例如“我喜欢猫”：字符级 tokenizer 先得到“我 / 喜 / 欢 / 猫”，再查词表得到编号 [0,1,2,3]。编号只是索引，不表示大小或距离。Embedding 将每个编号查成 C 个可训练数字。')+
section('模型中的位置','编号（B,T）→ token 向量加位置向量（B,T,C）→ 多个 Transformer Block → 最终归一化 → 输出 logits（B,T,V）。logits 是分数，Softmax 后才是概率。')+
section('训练','输入“我喜欢”，目标“喜欢猫”：位置 0 预测“喜”，位置 1 预测“欢”，位置 2 预测“猫”。计算损失 → 反向得到梯度 → 优化器更新参数。训练时用因果掩码禁止当前位置读取未来输入。')+
section('生成','给出前缀，取最后位置的概率选择下一个 token，再追加到输入，重复执行。生成不计算参数梯度。')+
section('五种数据','输入由调用方提供；参数是模型可训练数值；激活是前向中间结果；梯度是损失对某个数值的导数；优化器状态是更新参数时积累的动量等辅助量。不要把激活或梯度当成新参数。'),
kind='concept',questions=[dict(prompt='输入编号的形状是（2,3），Embedding 维度 C=4，输出是什么？',options=['（2,3,4）的向量数组','（2,4）的概率','（2,3）的损失'],answer=0,explain='两个样本，每个三个 token，每个 token 查出四个特征。'),dict(prompt='一次训练中，哪一步真正修改模型参数？',options=['Embedding 查表','反向计算梯度','优化器执行更新'],answer=2,explain='forward 计算激活，backward 累加梯度，优化器根据梯度更新参数。')])
add('architecture','本项目的结构与原始 Transformer','认识当前实现范围，区分自注意力、交叉注意力与前馈网络。',
section('本项目 GPT','采用 decoder-only：可学习位置嵌入、因果自注意力、Pre-Norm、RMSNorm、GELU 前馈网络、共享 token embedding 的输出头。没有 encoder，也没有交叉注意力。后续实现围绕这一明确结构。')+
section('原始 encoder-decoder','encoder 的自注意力可读取源序列所有位置；decoder 的自注意力使用因果掩码，另有交叉注意力读取 encoder 输出。交叉注意力中 Q 来自 decoder，K/V 来自 encoder。原始结构常用 LayerNorm、正弦位置编码；本项目的变体会逐项对照。')+
section('Block 内部','令 x 为输入。先 x₁=x+Attention(RMSNorm(x))，再 y=x₁+MLP(RMSNorm(x₁))。残差产生两条梯度路径，反向需要相加。MLP 对每个 token 独立作用；注意力让 token 之间交换信息。')+
section('维度表','B=样本条数，T=序列长度，C=模型特征数，D=某个线性层的输出特征数，H=注意力头数，V=词表大小。多头中每头特征数为 C/H，要求 C 能被 H 整除。'),
kind='concept',questions=[dict(prompt='哪一部分让不同 token 交换信息？',options=['Linear 单独作用于最后一轴','注意力','逐位置 GELU'],answer=1,explain='Linear 和 GELU 逐位置计算；注意力通过对其他 token 的 value 加权实现信息交换。'),dict(prompt='本项目是否需要实现 encoder 才能训练 GPT？',options=['需要','不需要'],answer=1,explain='decoder-only GPT 不需要 encoder。原始 Transformer 的 encoder-decoder 结构另作对照。')])

for id in ['axes','broadcast','matmul','linear']:
    old=legacy['lessons'][id]
    note=legacy['notes'][id]
    add(id,old['title'],old['goal'],section('变量与形状',note['intro']+' '+note['meaning']+' '+note['axes']),
        kind='array',starter=old['starter'],reference=legacy['references'][id],steps=old['steps'],hint=old['hint'],cases=cases[id],args=['X'] if id=='axes' else ['X','b'] if id=='broadcast' else ['X','W'] if id=='matmul' else ['X','W','b'],file={'axes':'01_axes.py','broadcast':'02_broadcast.py','matmul':'04_matmul.py','linear':'03_linear.py'}[id],derivation=section('计算关系',old['why']))


def array(id,title,goal,args,case_list,reference,body,steps,hint):
    cases[id]=case_list
    add(id,title,goal,body,kind='array',args=args,cases=case_list,reference=reference,starter=f'def solve({", ".join(args)}):\n    # TODO：实现本节计算，返回 NumPy 数组。\n    raise NotImplementedError("请完成本节 TODO")',steps=steps,hint=hint,file=f'numpy_{id}.py')


array('shape','数组创建、类型与坐标','把列表变成 float64 数组，返回它的形状。',['values'],
 [dict(label='两条三特征样本',args=[[[1,2,3],[4,5,6]]],expected=[2,3]),dict(label='三维输入',args=[[[[1,2],[3,4]]]],expected=[1,2,2])],
 'def solve(values):\n    X = np.asarray(values, dtype=np.float64)\n    return np.array(X.shape, dtype=np.int64)',
 section('数组与类型','Python 列表组织对象；NumPy 数组还记录统一 dtype 与每个轴的长度。np.array 创建数组，np.asarray 将已有对象解释成数组。float64 用于本课程实数计算；token 编号保持整数。')+section('坐标','X.shape=(2,3) 说明有两个轴，不是两个元素。X[0,1] 是第一行第二列。三维 X[b,t,c] 要提供三个坐标才能取到一个数。'),
 ['将 values 转成 float64 数组。','读取 shape，将形状元组转换为一维整数数组返回。'], 'shape 是属性，不要写成 shape()。')
array('index','切片与按编号取行','按 ids 取行，再保留前 k 列。',['X','ids','k'],
 [dict(label='ids=[2,0,2]，k=2：重排、重复并截列',args=[[[10,11,12],[20,21,22],[30,31,32]],[2,0,2],2],dtypes=['float64','int64','int64'],expected=[[30,31],[10,11],[30,31]]),dict(label='k=1：仅保留第 0 列',args=[[[1,2,3],[4,5,6]],[1,0],1],dtypes=['float64','int64','int64'],expected=[[4],[1]]),dict(label='k=列数：保留全部列',args=[[[1,2,3],[4,5,6]],[1,1],3],dtypes=['float64','int64','int64'],expected=[[4,5,6],[4,5,6]])],
 'def solve(X, ids, k):\n    return X[ids, :int(k)]',
 section('输入','X 是二维样本表（N,C）；ids 是一维整数行编号，长度 M，可重排或重复；k 是整数标量，表示保留前几列，1≤k≤C。样例 k=2，保留列编号 0、1，不包含编号 2。')+section('操作与输出','先按 ids 的顺序取行，得到（M,C）；再截取列区间 [0,k)，得到（M,k）。重复编号会在输出中出现多次。')+section('接口','实现 solve(X,ids,k)，返回 NumPy 数组，输入均不可修改。网页和本地测评传入的 k 为零维整数数组，可用 int(k) 作为切片终点。旧接口 solve(X,ids) 需要增加 k。')+section('切片和高级索引','整数数组索引会按编号取出副本；:k 表示从第 0 列到第 k−1 列。'),
 ['读取 ids 指定的行，保留给定顺序和重复项。','将 k 转成整数，截取前 k 列。','返回形状（len(ids),k）的数组。'],'X[ids, :int(k)]；切片终点不包含在结果中。')
array('reduce','mean、keepdims 与广播反向','每行减去它自身的特征均值，保留原形状。',['X'],
 [dict(label='行均值为 2 与 5',args=[[[1,2,3],[4,5,6]]],expected=[[-1,0,1],[-1,0,1]]),dict(label='一行四个特征',args=[[[1,3,5,7]]],expected=[[-3,-1,1,3]])],
 'def solve(X):\n    mean = np.mean(X, axis=-1, keepdims=True)\n    return X - mean',
 section('形状','X 为（N,C）。沿最后一轴求均值并 keepdims=True，得到（N,1）；减法时这个均值沿每行广播。若去掉 keepdims，得到（N,），不一定能与原数组正确对齐。')+section('归约的反向','若 y=Σⱼxⱼ，则每个 ∂y/∂xⱼ=1，故上游梯度传给每项；若 y=(Σⱼxⱼ)/C，则传给每项的是上游梯度除以 C。')+section('广播的反向','若 Yₙⱼ=Xₙⱼ+bⱼ，则 ∂L/∂bⱼ=Σₙ∂L/∂Yₙⱼ。前向重复使用同一变量，反向要把所有路径的贡献加起来。'),
 ['沿最后一轴求 mean，保留被归约的轴。','用 X 减去均值返回新数组。'],'axis=-1，keepdims=True。')
array('reshape','合并与恢复维度','把三维 X 的 B、T 合并，保留 C。',['X'],
 [dict(label='两条序列，各两个 token',args=[[[[1,2],[3,4]],[[5,6],[7,8]]]],expected=[[1,2],[3,4],[5,6],[7,8]]),dict(label='每个 token 三个特征',args=[[[[1,2,3],[4,5,6]]]],expected=[[1,2,3],[4,5,6]])],
 'def solve(X):\n    return X.reshape(-1, X.shape[-1])',
 section('元素不变','（B,T,C）→（B×T,C）。-1 让 NumPy 根据元素总数推算这一轴。顺序按 b 再按 t 排行，同一 token 的特征仍在同一行。')+section('恢复','若输出表为 Y2（B×T,D），用记录的原始形状恢复为（B,T,D）。reshape 只改变形状解释，不完成转置；不要用它交换 token 轴与通道轴。'),['读取最后一轴的长度 C。','reshape 成（-1,C）返回。'],'不要硬编码 B、T、C。')
array('transpose','转置与批量矩阵乘法','把每条序列的 token 轴与特征轴交换。',['X'],
 [dict(label='（1,2,3）变（1,3,2）',args=[[[[1,2,3],[4,5,6]]]],expected=[[[1,4],[2,5],[3,6]]]),dict(label='两个 batch 独立交换',args=[[[[1,2],[3,4]],[[5,6],[7,8]]]],expected=[[[1,3],[2,4]],[[5,7],[6,8]]])],
 'def solve(X):\n    return X.transpose(0, 2, 1)',
 section('轴重排','X.transpose(0,2,1) 让新轴依次对应旧轴 0、2、1。三维 X.T 会把所有轴逆序，不能当作“仅转置最后两轴”。')+section('批量 @','（B,T,C）@（C,D）得到（B,T,D）；（B,H,T,d）@（B,H,d,T）得到（B,H,T,T）。@ 对最后两轴做矩阵乘法，其余轴按广播规则匹配。')+section('反向','reshape 的反向恢复原形状；transpose 的反向应用轴置换的逆置换。例如 0,2,1 的逆置换仍是 0,2,1。'),['保持 batch 轴不动。','交换原轴 1 与 2。'],'使用 transpose(0,2,1) 或 swapaxes(1,2)。')
array('copy','视图、复制与原地修改','复制 X，把副本左上角改为 -99；原输入保持不变。',['X'],
 [dict(label='修改副本',args=[[[1,2],[3,4]]],expected=[[-99,2],[3,4]]),dict(label='单元素',args=[[[8]]],expected=[[-99]])],
 'def solve(X):\n    Y = X.copy()\n    Y[0, 0] = -99\n    return Y',section('变量与数据','Y=X 只是两个名字引用同一数组。Y=X[:] 常是共享内存的视图。Y=X.copy() 才把当前数值复制到独立数组。np.shares_memory 可检查是否共享。')+section('缓存','若缓存保存输入引用，调用方之后原地修改输入会改变反向所需数值。教学层采用复制关键缓存；真实工程可通过严格的生命周期约定减少复制。'),['复制 X。','只修改副本左上角，返回副本。'],'不能先修改 X 再复制。')
array('scatter','重复索引与梯度散射','将每个编号对应的增量累加到词表行中。',['ids','G','table'],
 [dict(label='编号 1 出现两次',args=[[1,1,0],[[1,2],[3,4],[5,6]],[[0,0],[0,0],[0,0]]],dtypes=['int64','float64','float64'],expected=[[5,6],[4,6],[0,0]]),dict(label='只有一个编号',args=[[0],[[2,3]],[[0,0],[0,0]]],dtypes=['int64','float64','float64'],expected=[[2,3],[0,0]])],
 'def solve(ids, G, table):\n    result = table.copy()\n    np.add.at(result, ids, G)\n    return result',section('输入','ids 的长度为 N，G 形状（N,C），table 形状（V,C）。第 n 行 G 应加入 table[ids[n]]。重复编号必须多次累加。')+section('不能省掉的累加','result[ids] += G 使用高级索引的临时数组，重复编号可能只回写一次。np.add.at 或明确循环可以按每次出现累加。'),['复制 table。','按 ids 将 G 散射相加，返回副本。'],'使用 np.add.at(result, ids, G)。')
array('random','随机数与可复现初始化','用固定 seed 生成指定形状的标准正态数组。',['seed','dims'],
 [dict(label='种子 7，形状（2,3）',args=[7,[2,3]],dtypes=['int64','int64'],expected=np.random.default_rng(7).standard_normal((2,3)).tolist()),dict(label='种子 0，形状（1,2）',args=[0,[1,2]],dtypes=['int64','int64'],expected=np.random.default_rng(0).standard_normal((1,2)).tolist())],
 'def solve(seed, dims):\n    rng = np.random.default_rng(int(seed))\n    return rng.standard_normal(tuple(int(d) for d in dims))',section('随机生成器','每个 Generator 持有自己的状态。相同 seed 创建的新生成器，执行相同采样顺序可复现；在同一个生成器上连续调用会推进状态。不要每次 forward 重置种子。')+section('数值类型','本课程用 float64 做实数运算，编号保持 int64。权重初始化一般均值为 0，方差按输入特征数缩放；下一部分会用 fan-in 初始化。浮点误差分析放在拓展。'),['从 seed 创建 Generator。','按 dims 指定形状采样，不要调用旧的全局 np.random.seed。'],'seed 是零维整数数组，先转成 int；dims 转成整数元组。')


class Hollow(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name in self.methods:
            node.body=[ast.Expr(value=ast.Constant(value='TODO：按网页公式实现此方法。')),ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError',ctx=ast.Load()),args=[ast.Constant(value='请实现 '+node.name)],keywords=[]),cause=None)]
        return node


def skeleton(source,methods):
    tree=ast.parse(source);h=Hollow();h.methods=methods;tree=h.visit(tree);ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def numeric_case(cls,init,X,G,params,label,repeat=False):
    obj=getattr(ref,cls)(*init)
    for name,value in params.items():
        target=obj
        for part in name.split('.'):
            target=getattr(target,part)
        target.data[...] = value
    x=np.array(X,dtype=np.int64 if cls=='Embedding' else np.float64);g=np.array(G,dtype=float)
    y=obj.forward(x).copy();expected={'Y':y.tolist()}
    # Independent central differences of L=sum(Y*G); no use of reference backward.
    def diff(arr):
        result=np.zeros_like(arr,dtype=float)
        for idx in np.ndindex(arr.shape):
            value=arr[idx];arr[idx]=value+1e-6;plus=np.sum(obj.forward(x)*g)
            arr[idx]=value-1e-6;minus=np.sum(obj.forward(x)*g);arr[idx]=value
            result[idx]=(plus-minus)/2e-6
        return result
    expected['dX']=None if cls=='Embedding' else diff(x).tolist()
    def parameters(module,prefix=''):
        for name,value in vars(module).items():
            if isinstance(value,core.Parameter):yield prefix+name,value
            elif isinstance(value,core.Module):yield from parameters(value,prefix+name+'.')
    for name,p in parameters(obj):expected[name+'.grad']=(diff(p.data)*(2 if repeat else 1)).tolist()
    return dict(label=label,**{'class':cls},init=init,X=X,G=G,parameters=params,expected=expected,repeat=repeat,check_zero=True,expected_source='前向按公式计算；梯度来自标量目标 L=Σ(Y⊙G) 的中心差分，生成后固定保存。')


infra=inspect.getsource(core.Parameter)+'\n\n'+inspect.getsource(core.Module)
add('infrastructure','Parameter 与 Module','建立参数容器、参数去重和梯度清零接口。',
 section('Parameter','data 是可训练数组；grad 是同形状梯度缓冲区，初始全零。zero_grad 只清梯度，不改变 data。初始化时复制数据，统一为 float64。')+
 section('Module','forward 产生激活并缓存；backward 返回输入梯度，并累加参数梯度；parameters 递归访问子模块的公开属性，按对象身份去重。下划线属性是缓存，不参与参数收集。')+
 section('共享参数','同一个 Parameter 被两个属性引用，仍只应收集一次，否则优化器会更新两次。列表、元组和字典中的子模块也要递归访问。')+
 section('缓存生命周期','一次 forward 对应紧接着的一次 backward。连续 forward 会替换缓存；反向结束后才更新权重。多个 batch 累加梯度时，每批都重新 forward，参数梯度在累计开始前清零。'),
 kind='infrastructure',starter=skeleton(infra,{'__init__','zero_grad','parameters'}),reference=infra,file='layer_infrastructure.py',dependencies=[],steps=['完成 Parameter 的 data、grad 初始化和清零。','实现 Module 参数递归收集，按 id 去重，跳过缓存。','实现 Module.zero_grad 调用各 Parameter 的清零。'],cases=[dict(label='共享参数递归去重与清零',input='Parameter([[1,2]])；child.weight、root.shared、root.blocks[0].weight 指向同一个参数；先将梯度置为 5，再 root.zero_grad()',expected={'参数形状':[1,2],'共享参数计数':1,'清零后梯度':[[0,0]]})])

add('initialization','fan-in 初始化与残差出口缩放','按输入特征数初始化权重，正确限定残差缩放的时机。',
 section('fan-in','本课程 Linear 的 W 形状为（C,D）。若输入各分量独立、均值 0，且每项方差为 v，则单个输出的方差约为 C·Var(W)·v。权重若不随 C 缩放，输出尺度会随输入维数增长。')+
 section('本课程约定','使用标准差 √(2/C) 的正态权重，bias 为 0。这个系数来自 ReLU 的近似方差补偿，不是 GELU 的精确方差恒等式；GELU 网络中仍需观察激活和梯度。')+
 section('残差出口','若 L 层、每层两个残差分支，常用起始尺度 1/√(2L) 缩小出口投影。只作用于各分支的最后一个投影权重，初始化后调用一次；不能缩放全部权重，也不能在 forward 中重复缩放。'),
 kind='concept',questions=[dict(prompt='C=4 时本课程权重初始化的标准差是多少？',options=['2','√(2/4)','1/4'],answer=1,explain='W 的每个元素从标准正态采样后乘 √(2/C)。'),dict(prompt='残差出口缩放应该何时执行？',options=['每次 forward','每次 backward','初始化时对出口权重执行一次'],answer=2,explain='重复缩放会不断压小权重，改变训练问题。')],reference=inspect.getsource(core.initialize)+'\n\n'+inspect.getsource(core.scale_residual_outputs))


def layer(id,cls,init,body,derivation,case_list,deps=[]):
    source=inspect.getsource(getattr(ref,cls))
    add(id,cls+'：前向、缓存与反向','实现可复用的 '+cls+' 层，正确返回输入梯度并累加参数梯度。',body,
        kind='layer',**{'class':cls},starter=skeleton(source,{'forward','backward'}),reference=source,file='layer_'+id.removeprefix('layer_')+'.py',dependencies=deps,cases=case_list,derivation=derivation,
        steps=['保留已给定的构造函数和属性名。','按前向公式实现 forward，保存反向需要的中间量。','实现 backward：返回输入梯度，参数梯度使用 +=。','运行全部公开测评，确认重复调用累加与 zero_grad 均正确。'])

layer('layer_scale','Scale',[],section('接口','Scale(value) 持有一个标量参数 w。forward(x) 接受任意形状实数数组，输出同形数组；backward(G) 返回 dX，并更新 w.grad。')+section('缓存','保存 X；不能只缓存输出，因为 w=0 时无法从输出恢复 X。'),
 section('前向与推导','Yᵢ=wXᵢ。给定 Gᵢ=∂L/∂Yᵢ，链式法则给出 ∂L/∂Xᵢ=Gᵢw；同一 w 被所有位置使用，∂L/∂w=ΣᵢGᵢXᵢ。参数梯度为标量求和，不是求均值。'),
 [numeric_case('Scale',[2.],[[1.,-2.]],[[3.,4.]],{},'w=2：输入梯度与标量参数梯度'),numeric_case('Scale',[0.],[[1.,2.]],[[1.,1.]],{},'零权重仍有参数梯度；两轮累加',True)])

layer('layer_linear','Linear',[],section('统一形状','W 固定为（C,D），b 为（D，），X 为（…，C）。新课程沿用前向练习的存储方向。旧项目 src 中 W 为（D,C），公式需转置；不要把旧类的方法直接混入本类。')+section('接口','Linear(C,D,use_bias=True)。持有 W、可选 b；forward 返回（…，D），backward 接受同形 G，返回与 X 同形的 dX。')+section('缓存','保存原始形状与二维 X2（N,C），N 是所有前置维度乘积。参数归优化器更新，backward 不改 data。支持二维、三维和没有 bias 的情况。'),
 section('逐元素定义','Yₙⱼ=ΣᵢX2ₙᵢWᵢⱼ+bⱼ。令 G2=∂L/∂Y2，形状（N,D）。')+section('输入梯度','∂L/∂X2ₙᵢ=ΣⱼG2ₙⱼWᵢⱼ，因此 dX2=G2Wᵀ，得到（N,C），最后恢复 X 的形状。')+section('权重与偏置','∂L/∂Wᵢⱼ=ΣₙX2ₙᵢG2ₙⱼ，因此 dW=X2ᵀG2；∂L/∂bⱼ=ΣₙG2ₙⱼ，因此 db=sum(G2,axis=0)。这里不再次除以 N，上游损失决定是否平均。'),
 [numeric_case('Linear',[2,3],[[[1.,2.],[0.,1.]]],[[[1.,2.,3.],[4.,5.,6.]]],{'W':[[1.,0.,2.],[0.,2.,1.]],'b':[1.,2.,3.]},'三维、非方阵：同时检查 Y、dX、dW、db'),numeric_case('Linear',[2,2,False],[[1.,-1.]],[[2.,3.]],{'W':[[1.,2.],[3.,4.]]},'无偏置；重复前向反向累加',True)])

layer('layer_embedding','Embedding',[],section('输入输出','Embedding(V,C) 的 weight 为（V,C）。输入 ids 是整数数组（B,T），输出（B,T,C）。编号不是连续数值特征；对整数编号不求导，backward 返回 None。')+section('查表','Y[b,t,c]=weight[ids[b,t],c]。缓存 ids；相同编号出现多次时对应同一行参数。'),
 section('反向推导','某个参数 weight[v,c] 只影响 ids 等于 v 的输出。因此 dweight[v,c]=Σ_{b,t:ids[b,t]=v}G[b,t,c]。这是散射累加，不是覆盖。')+section('重复编号','先展平 ids 为（B×T，），G 为（B×T,C），再 np.add.at(weight.grad, ids_flat, G_flat)。不能用 weight.grad[ids] += G 代替重复索引累加。'),
 [numeric_case('Embedding',[3,2],[[1,1,0]],[[[1.,2.],[3.,4.],[5.,6.]]],{'weight':[[1.,2.],[3.,4.],[5.,6.]]},'重复编号：编号 1 的两份梯度相加'),numeric_case('Embedding',[3,2],[[2]],[[[2.,3.]]],{'weight':[[1.,2.],[3.,4.],[5.,6.]]},'未访问行梯度为零；两轮累加',True)])

layer('layer_gelu','GELU',[],section('输入输出','无可训练参数。输入、输出及上游梯度均为同形实数数组。使用 tanh 近似 GELU；缓存 X 与 t。')+section('激活函数对照','sigmoid=1/(1+e⁻ˣ)，导数 s(1−s)；tanh 导数 1−t²；ReLU=max(0,x)，本课程取 x=0 时导数为 0。饱和区导数小，线性层简单堆叠仍是线性映射，因此需要非线性。')+section('标准公式','k=√(2/π)，a=0.044715；u=k(x+ax³)，t=tanh(u)，y=½x(1+t)。这是近似式，并非精确 erf 定义。'),
 section('完整链式法则','du/dx=k(1+3ax²)，dt/du=1−t²。乘积求导得到 dy/dx=½(1+t)+½x(1−t²)k(1+3ax²)。最后 dX=G⊙dy/dx。x=0 时 dy/dx=½；遗漏乘积法则的任何一项都会出错。'),
 [numeric_case('GELU',[],[[-1.,0.,1.]],[[2.,3.,4.]],{},'负值、零、正值；检查零点导数为 1/2'),numeric_case('GELU',[],[[[.2,-.5]]],[[[1.,-2.]]],{},'三维形状与非单位上游梯度')])

layer('layer_rmsnorm','RMSNorm',[],section('归一化轴','RMSNorm(C,eps=1e-5)。每个 token 只沿最后一轴的 C 个特征计算均方，不跨 batch 或 token。w 为（C，），逐特征缩放，默认全 1。')+section('前向','m=mean(x²)，r=(m+ε)⁻¹ᐟ²，x̂=xr，y=x̂⊙w。m、r 保留最后一轴为 1，以正确广播。ε 用于避免全零输入除以零，保留在计算中。')+section('LayerNorm 对照','LayerNorm 先减均值，再除以标准差，通常还有缩放和偏置；RMSNorm 不减均值。本项目实现 RMSNorm，不把两种层说成同一个公式。'),
 section('导数路径','令 u=G⊙w，dL/dr=Σⱼuⱼxⱼ；dr/dm=−½r³；dm/dxᵢ=2xᵢ/C。因此 dXᵢ=ruᵢ−r³xᵢ·mean(u⊙x)。')+section('便于实现的形式','用 x̂=rx，得到 dX=r[u−x̂·mean(u⊙x̂)]。dW=Σ所有前置位置(G⊙x̂)，保留特征轴。缓存 r、x̂；参数梯度累加。'),
 [numeric_case('RMSNorm',[2],[[[3.,4.],[0.,0.]]],[[[1.,2.],[3.,4.]]],{'w':[2.,.5]},'非单位缩放、零输入与 epsilon'),numeric_case('RMSNorm',[3],[[1.,-2.,3.]],[[2.,1.,-1.]],{'w':[1.,2.,3.]},'三特征；两轮累加',True)])

layer('layer_mlp','MLP',[],section('组成','MLP(C,hidden)：Linear(C→hidden) → GELU → Linear(hidden→C)。输入与输出均为（B,T,C），中间为（B,T,hidden）。它逐 token 计算，不混合不同位置。')+section('前置依赖','使用你在 layer_linear.py 与 layer_gelu.py 中实现的类。网页提交会一并读取这两个任务的个人草稿；未完成时会明确提示，不能悄悄使用参考答案。')+section('缓存','MLP 持有 fc1、act、fc2，子层各自保存缓存。Module.parameters 会递归收集两个 Linear 的参数。'),
 section('反向串联','令 h₁=fc1(x)，h₂=GELU(h₁)，y=fc2(h₂)。依次计算 G₂=fc2.backward(G)，G₁=act.backward(G₂)，dX=fc1.backward(G₁)。每个子层收到的是损失对其输出的梯度，不是原始 G。')+section('同一套模块','前向与反向都复用已有层。验收同时检查最终输出、输入梯度、两个权重梯度和两个偏置梯度，防止某个子层没有接入计算。'),
 [numeric_case('MLP',[2,3],[[[1.,2.],[0.,-1.]]],[[[1.,2.],[3.,4.]]],{'fc1.W':[[.1,.2,.3],[.2,-.1,.4]],'fc1.b':[.1,0.,-.1],'fc2.W':[[.2,.3],[-.1,.4],[.5,-.2]],'fc2.b':[.1,.2]},'组合：全部子层梯度与输入梯度'),numeric_case('MLP',[2,3],[[.1,.2]],[[1.,-1.]],{'fc1.W':[[.1,.2,.3],[.2,-.1,.4]],'fc1.b':[.1,0.,-.1],'fc2.W':[[.2,.3],[-.1,.4],[.5,-.2]],'fc2.b':[.1,.2]},'重复调用的参数梯度累加',True)],deps=['layer_linear','layer_gelu'])

# 默认样例保留非退化的 batch、token、feature 轴；简单边界例子另行保留。
X3=np.arange(24,dtype=float).reshape(2,3,4)/10-1
W45=np.arange(20,dtype=float).reshape(4,5)/10-.5
b5=np.arange(5,dtype=float)/10
G35=np.arange(30,dtype=float).reshape(2,3,5)/10-1
large_linear=dict(label='B=2、T=3、C=4、D=5：批次与 token 均完整保留',args=[X3.tolist(),W45.tolist(),b5.tolist()],expected=(X3@W45+b5).tolist())
# 不反复追加，也不删除原二维及单样本边界测评。
if not any(c['label']==large_linear['label'] for c in cases['linear']):cases['linear'].append(large_linear)
lessons['linear']['cases']=cases['linear']
lessons['linear']['sample_index']=next(i for i,c in enumerate(cases['linear']) if c['label']==large_linear['label'])
lessons['linear']['body']=section('变量与形状','X 为（B,T,C）=（2,3,4），包含两个 batch，每个三个 token，每个 token 四个特征。W 为（4,5），b 为（5，），输出 Y 为（2,3,5）。B 和 T 都大于 1；整理成二维仅是计算步骤，输入输出仍按原 batch 展示。')
lessons['reshape']['cases'][0]=dict(label='（2,3,4）→（6,4）：行 n=b×3+t',args=[X3.tolist()],expected=X3.reshape(6,4).tolist())
lessons['transpose']['output_axes']=['batch b','特征 c','token t']
lessons['transpose']['cases'][0]=dict(label='（2,3,4）→（2,4,3），两个 batch 分别转置',args=[X3.tolist()],expected=X3.transpose(0,2,1).tolist())
lessons['shape']['cases'][1]=dict(label='真实三维：每个轴均大于 1',args=[X3.tolist()],expected=[2,3,4])
lessons['layer_scale']['cases'][0]=numeric_case('Scale',[2.],X3.tolist(),np.ones_like(X3).tolist(),{},'B=2、T=3、C=4：所有位置参与标量参数梯度')
lessons['layer_linear']['cases'][0]=numeric_case('Linear',[4,5],X3.tolist(),G35.tolist(),{'W':W45.tolist(),'b':b5.tolist()},'B=2、T=3、C=4、D=5：检查全部输出及三类梯度')
lessons['layer_embedding']['cases'][0]=numeric_case('Embedding',[4,3],[[1,2,1],[0,1,3]],np.arange(18,dtype=float).reshape(2,3,3).tolist(),{'weight':np.arange(12,dtype=float).reshape(4,3).tolist()},'两个 batch 中重复编号 1：跨 batch 累加到同一参数行')
lessons['layer_gelu']['cases'].insert(0,numeric_case('GELU',[],X3.tolist(),np.ones_like(X3).tolist(),{},'B=2、T=3、C=4：所有 batch 与 token 的逐元素非线性'))
normX=X3.copy();normX[1,2,:]=0
lessons['layer_rmsnorm']['cases'][0]=numeric_case('RMSNorm',[4],normX.tolist(),(np.ones_like(X3)*2).tolist(),{'w':[1.,2.,.5,1.5]},'（2,3,4）：六个 token 独立归一化，最后一个为零向量')
lessons['layer_mlp']['cases'].insert(0,numeric_case('MLP',[4,6],X3.tolist(),np.ones_like(X3).tolist(),{},'B=2、T=3、C=4、hidden=6：完整三维 MLP 与全部参数梯度'))

heads=np.arange(48,dtype=float).reshape(2,3,8)
array('heads','四维张量：拆头、换轴与还原','按输入头数 H，把（B,T,C）拆成（B,H,T,d），保留全部 batch、head 和 token。',['X','H'],
 [dict(label='H=2：（2,3,8）→（2,2,3,4）',args=[heads.tolist(),2],dtypes=['float64','int64'],expected=heads.reshape(2,3,2,4).transpose(0,2,1,3).tolist()),dict(label='H=3：（2,2,6）→（2,3,2,2）',args=[np.arange(24).reshape(2,2,6).tolist(),3],dtypes=['float64','int64'],expected=np.arange(24).reshape(2,2,3,2).transpose(0,2,1,3).tolist())],
 'def solve(X,H):\n    B, T, C = X.shape\n    H = int(H)\n    d = C // H\n    return X.reshape(B, T, H, d).transpose(0, 2, 1, 3)',
 section('四个轴','B=2 个 batch，H=2 个 head，T=3 个 token，d=4 个每头特征。样例输出明确展示全部四张 T×d 切片，不省略第二个 batch 或第二个 head。')+section('分两步操作','先 reshape：（2,3,8）→（2,3,2,4），只把最后一轴拆为 H,d。再 transpose(0,2,1,3)：（B,T,H,d）→（B,H,T,d）。不能直接 reshape 到（B,H,T,d），那会改变 token 与 head 的对应。')+section('还原','先 transpose(0,2,1,3) 回到（B,T,H,d），再 reshape(B,T,C)。每个位置的值应与原输入一致。这里仅练习轴操作，注意力的 Q/K/V 计算留到后续章节。'),
 ['读取 B、T、C，使用传入的 H，计算 d=C//H。','先拆分最后一轴，再把 head 轴移到 token 轴之前。','返回完整四维数组，不取任何一个 batch 或 head 的切片。'],'将 H 转成整数；reshape(B,T,H,C//H) 后 transpose(0,2,1,3)。')

extensions=section('数值误差分析（拓展）','浮点数不能精确表示所有实数；矩阵乘加顺序也会影响末位。float64 适合教学核对，实际训练可按需求使用 float32。不要用数组逐元素 == 判断一般浮点计算等价。')+section('有限差分（拓展）','对 L=Σ(Y⊙G)，数值导数约为 [L(θ+h)−L(θ−h)]/(2h)。h 太大会带来截断误差，太小会因浮点相减丢失有效位。误差的解释、步长和容差调节在这里学习，不作为每页正文负担。')+section('测评中使用','固定样例检查具体值、形状、输入不变和状态；基础层梯度预期由独立中心差分生成并保存。网站只报告通过或失败；需要排查时展开具体结果。数值检查是发现错误的工具，不是对所有输入的数学证明。')
for id,meta in lessons.items():
    meta.setdefault('extension',extensions)
    meta.setdefault('steps',[])
    if meta['kind']=='concept':continue
    if meta['kind']=='array':
        main=f'from _checks import run_local\n\nif __name__ == "__main__":\n    run_local("{id}", solve)\n'
        code='import numpy as np\n\n'+meta['starter']+'\n\n'+main
    else:
        deps=''
        if meta.get('dependencies'):
            deps='if __name__ != "student":\n    from layer_linear import Linear\n    from layer_gelu import GELU\n\n'
        support='foundation_core' if meta['kind']=='infrastructure' else 'learner_core'
        code=f'import numpy as np\nfrom {support} import Parameter, Module, initialize\n\n'+deps+meta['starter']+f'\n\nif __name__ == "__main__":\n    from _foundation_checks import run_local\n    run_local("{id}", globals())\n'
    # Existing personal files are never rewritten by this builder.
    target=ROOT/'exercises'/meta['file']
    if not target.exists():target.write_text(code,encoding='utf-8')
    meta['local_code']=target.read_text(encoding='utf-8')

stages=[
 dict(title='第一部分 · 整体架构',modules=[dict(name='模型与数据流',tasks=['map','architecture'])]),
 dict(title='第二部分 · NumPy 与张量',modules=[dict(name='数组、轴与索引',tasks=['shape','axes','index','reduce']),dict(name='广播与矩阵运算',tasks=['broadcast','matmul','reshape','transpose','heads']),dict(name='内存、散射与随机数',tasks=['copy','scatter','random'])]),
 dict(title='第三部分 · 标量网络',paused=True,description='按你的学习进度暂缓：标量自动微分、直线与 sin 曲线拟合保留在原项目，以后再整理。',modules=[]),
 dict(title='第四部分 · 完整基础层',modules=[dict(name='公共接口与初始化',tasks=['infrastructure','initialization']),dict(name='Scale 与 Linear',tasks=['layer_scale','linear','layer_linear']),dict(name='Embedding 查表',tasks=['layer_embedding']),dict(name='激活与归一化',tasks=['layer_gelu','layer_rmsnorm']),dict(name='组合为 MLP',tasks=['layer_mlp'])]),
 dict(title='第五部分起 · 后续课程',paused=True,description='损失函数、注意力、GPT、训练、检查点与生成尚未在新站开放；原文件保留。',modules=[])
]
catalog=dict(version=2,stages=stages,lessons=lessons)
(ROOT/'webapp/foundations.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'exercises/cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Built {len(lessons)} lessons; preserved existing exercise files.')

if __name__=='__main__':
    from _foundation_checks import evaluate_layer,evaluate_infrastructure
    scope=dict(vars(ref));scope.update(Parameter=core.Parameter,Module=core.Module)
    for id,meta in lessons.items():
        if meta['kind']=='layer':
            rows=evaluate_layer(scope,meta['cases'])
            assert all(r['passed'] for r in rows),(id,rows)
        elif meta['kind']=='infrastructure':assert all(r['passed'] for r in evaluate_infrastructure(scope,meta['cases']))
    print('Reference layer implementations match all independent gradient expectations.')

# 基础重建完成后接回后续课程，保留所有已有个人代码。
if __name__ == "__main__":
    import runpy
    runpy.run_path(str(ROOT/"scripts/build_transformer.py"),run_name="__main__")
