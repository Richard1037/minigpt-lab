"""补齐从数组操作到完整层的六个主线台阶；只新建缺失练习。"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from course_math import math, aligned, p
target=ROOT/'webapp/foundations.json'
book=json.loads(target.read_text(encoding='utf-8'))
new={}
rng=np.random.default_rng(20261109)


def add(id,title,args,reference,body,derivation,inputs,dtypes):
    scope={'np':np};exec(reference,scope)
    cases=[]
    labels=['多 batch/token 的完整来源','改变轴长检验迁移','零上游或恒定输入的边界','不同结构，禁止记住样例']
    for label,values in zip(labels,inputs):
        values=[np.asarray(v,dtype=t) for v,t in zip(values,dtypes)]
        cases.append(dict(label=label,args=[v.tolist() for v in values],dtypes=dtypes,expected=scope['solve'](*values).tolist(),category='基础构建与迁移'))
    starter='def solve('+', '.join(args)+'):\n    # TODO：按本节来源与形状关系实现。\n    raise NotImplementedError("请实现 '+title+'")'
    filename='numpy_'+id+'.py'
    local='import numpy as np\n\n'+starter+'\n\nif __name__ == "__main__":\n    from _checks import run_local\n    run_local("'+id+'", solve)\n'
    path=ROOT/'exercises'/filename
    if not path.exists():path.write_text(local,encoding='utf-8')
    new[id]=dict(id=id,title=title,goal=title+'：先按来源完成短实现，再接入完整模块。',kind='array',args=args,reference=reference,body=body,derivation=derivation,cases=cases,starter=starter,local_code=path.read_text(encoding='utf-8'),file=filename,steps=[],hint='核对每个轴的含义，保持输入不变。',dependencies=[],assessment_revision=1)


add('mean_backward','均值反向：把上游分回每个特征',['G','C'],
    'def solve(G,C):\n    return np.broadcast_to(G/int(C), (*G.shape[:-1],int(C))).copy()',
    '<h3>这次只写反向</h3>'+p('前向已固定为沿最后一轴求均值、keepdims=True。G 是损失对均值结果的梯度，不是原始输入；C 是该轴原来的长度。'),
    p('固定一个 batch 和 token。μ 是该位置 C 个特征的均值：')
    +math(r'\mu_{b,t,0}=\frac1C\sum_{j=0}^{C-1}X_{b,t,j}')
    +p('改变其中一个特征 X[b,t,c]，均值对它的局部导数为 1/C：')
    +math(r'\frac{\partial\mu_{b,t,0}}{\partial X_{b,t,c}}=\frac1C')
    +p('G 是损失 L 对这个均值 μ 的上游梯度：')
    +math(r'G_{b,t,0}=\frac{\partial L}{\partial\mu_{b,t,0}}')
    +p('沿 L → μ → X 使用链式法则：')
    +aligned(r'\mathrm dX_{b,t,c}&=\frac{\partial L}{\partial X_{b,t,c}}',r'&=\frac{\partial L}{\partial\mu_{b,t,0}}\frac{\partial\mu_{b,t,0}}{\partial X_{b,t,c}}',r'&=G_{b,t,0}\cdot\frac1C')
    +p('μ 连接上游和输入；两个偏导数分别写成 G 和 1/C 后，结果中就不再单独出现 μ。'),
    [[rng.normal(size=(2,3,1)),4],[rng.normal(size=(3,2,1)),5],[np.zeros((2,4,1)),3],[rng.normal(size=(2,2,2,1)),3]],['float64','int64'])
add('center_backward','去均值反向：合并直接与均值路径',['G'],
    'def solve(G):\n    return G-G.mean(axis=-1,keepdims=True)',
    '<h3>这次只写反向</h3>'+p('对应 reduce 的前向 Y=X−mean(X)。G 是损失对 Y 的梯度。每个 x 既直接影响本位置输出，也通过均值影响该行全部输出。'),
    p('固定一个 batch 和 token，i、j 都是同一位置的特征编号；Gᵢ 是损失 L 对输出 Yᵢ 的梯度。')
    +aligned(r'\mu&=\frac1C\sum_{k=0}^{C-1}X_k',r'Y_i&=X_i-\mu,\qquad G_i=\frac{\partial L}{\partial Y_i}')
    +p('直接路径：Xⱼ 直接影响 Yⱼ，贡献为 Gⱼ。均值路径：所有输出都减去同一个 μ，每条路径贡献 −Gᵢ，先汇合到 μ：')
    +math(r'\frac{\partial L}{\partial\mu}=-\sum_{i=0}^{C-1}G_i')
    +p('再经过均值的局部导数 ∂μ/∂Xⱼ=1/C，得到均值路径对 Xⱼ 的贡献。将两条路径相加：')
    +aligned(r'\mathrm dX_j&=G_j+\frac{\partial L}{\partial\mu}\frac{\partial\mu}{\partial X_j}',r'&=G_j-\frac1C\sum_{i=0}^{C-1}G_i'),
    [[rng.normal(size=(2,3,4))],[rng.normal(size=(3,2,5))],[np.ones((2,3,4))],[rng.normal(size=(2,2,3,4))]],['float64'])
add('restore','形状恢复：让每一行回到原 batch/token',['Y2','out_shape'],
    'def solve(Y2,out_shape):\n    return Y2.reshape(tuple(int(s) for s in out_shape))',
    '<h3>只恢复形状</h3>'+p('Y2 是计算后的二维位置表。out_shape 给定要恢复的全部轴长度，三维时为（B,T,D）。行号仍对应 n=b*T+t，不改变特征顺序。'),
    math(r'Y_{b,t,j}=Y_{2,bT+t,j}'),
    [[rng.normal(size=(6,5)),[2,3,5]],[rng.normal(size=(6,4)),[3,2,4]],[np.zeros((8,3)),[2,4,3]],[rng.normal(size=(12,5)),[2,2,3,5]]],['float64','int64'])
add('random_sequence','随机状态：同一生成器连续采两次',['seed','dims'],
    'def solve(seed,dims):\n    rng=np.random.default_rng(int(seed))\n    shape=tuple(int(d) for d in dims)\n    return np.stack([rng.standard_normal(shape),rng.standard_normal(shape)])',
    '<h3>推进状态，不重置种子</h3>'+p('用 seed 只创建一次局部 Generator，连续采样两份同形标准正态数组。将它们按采样先后堆叠。第二次必须延续第一次的状态。'),
    p('输出第 0 份来自初始状态，第 1 份来自采过第一份后的状态。同一个 seed 重跑整个函数可复现；函数内部两份样本不应由两个重新播种的生成器产生。'),
    [[7,[2,3,4]],[19,[3,2]],[0,[2,4]],[61,[2,2,3,4]]],['int64','int64'])
add('att_value_backward','注意力反向一：所有 query 汇入 dV',['A','G'],
    'def solve(A,G):\n    return A.swapaxes(-1,-2) @ G',
    '<h3>仅实现对 V 的反向</h3>'+p('前向 Y=A@V。A 是读取概率，G 是损失对 Y 的上游梯度。本任务只返回 dV，不计算 dA、dQ 或 dK。'),
    aligned(r'Y_{b,h,t,j}&=\sum_s A_{b,h,t,s}V_{b,h,s,j}',r'\mathrm dV_{b,h,s,j}&=\sum_t A_{b,h,t,s}G_{b,h,t,j}',r'\mathrm dV&=A^{\mathsf T}G'),
    [[rng.random((2,2,3,4)),rng.normal(size=(2,2,3,5))],[rng.random((3,2,2,5)),rng.normal(size=(3,2,2,4))],[rng.random((2,3,4,2)),np.zeros((2,3,4,3))],[rng.random((2,3,5,4)),rng.normal(size=(2,3,5,2))]],['float64','float64'])
As=[rng.random(s) for s in [(2,2,3,4),(3,2,2,5),(2,3,4,3),(2,3,5,4)]]
As=[a/a.sum(axis=-1,keepdims=True) for a in As]
add('att_score_backward','注意力反向二：同一行概率的耦合',['A','dA'],
    'def solve(A,dA):\n    return A*(dA-(A*dA).sum(axis=-1,keepdims=True))',
    '<h3>仅实现 Softmax 回到分数的反向</h3>'+p('前向 A=Softmax(S)。A 已给出，dA 是损失对 A 的梯度。返回损失对未归一化分数 S 的梯度 dS；本任务不计算 Q/K 的梯度。'),
    aligned(r'\mathrm dS_j&=A_j\left(\mathrm dA_j-\sum_kA_k\mathrm dA_k\right)'),
    [[As[0],rng.normal(size=As[0].shape)],[As[1],rng.normal(size=As[1].shape)],[As[2],np.ones_like(As[2])],[As[3],np.zeros_like(As[3])]],['float64','float64'])
# 这一步练反向位置来源，A 也应为合法逐行概率。
for case in new['att_value_backward']['cases']:
    a=np.asarray(case['args'][0]);a/=a.sum(-1,keepdims=True)
    case['args'][0]=a.tolist();case['expected']=(a.swapaxes(-1,-2)@np.asarray(case['args'][1])).tolist()
new['random_sequence']['cases'][2]['label']='种子 0：第二份延续第一份随机状态'
new['random_sequence']['cases'][3]['label']='四维样本完整保留全部前置轴'
new['att_value_backward']['input_axes']={'A':['batch b','head h','query t','key s'],'G':['batch b','head h','query t','value 特征 j']}
new['att_value_backward']['output_axes']=['batch b','head h','key s','value 特征 j']
new['att_score_backward']['input_axes']={key:['batch b','head h','query t','key s'] for key in ['A','dA']}
new['att_score_backward']['output_axes']=['batch b','head h','query t','key s']
book['lessons'].update(new)
after={'reduce':['mean_backward','center_backward'],'reshape':['restore'],'random':['random_sequence'],'att_merge':['att_value_backward','att_score_backward']}
for stage in book['stages']:
    for module in stage['modules']:
        tasks=[]
        for id in module['tasks']:
            if id in new:continue
            tasks.append(id);tasks.extend(after.get(id,[]))
        module['tasks']=tasks
# 基础操作先于依赖它们的组合和反向；完整注意力放在反向分解练习之后。
for stage in book['stages']:
    if stage['title'].startswith('第二部分'):
        first, second = stage['modules'][:2]
        moved = ['reduce', 'mean_backward', 'center_backward']
        first['tasks'] = [task for task in first['tasks'] if task not in moved]
        second['tasks'] = [task for task in second['tasks'] if task not in moved]
        at = second['tasks'].index('broadcast') + 1
        second['tasks'][at:at] = moved
        second['name'] = '广播、归约与矩阵运算'
    elif stage['title'].startswith('第六部分'):
        tasks = stage['modules'][0]['tasks']
        tasks.remove('t_attention')
        tasks.insert(tasks.index('att_score_backward') + 1, 't_attention')
target.write_text(json.dumps(book,ensure_ascii=False,indent=2),encoding='utf-8')
arrays=json.loads((ROOT/'exercises/cases.json').read_text(encoding='utf-8'))
arrays.update({id:m['cases'] for id,m in new.items()})
(ROOT/'exercises/cases.json').write_text(json.dumps(arrays,ensure_ascii=False,indent=2),encoding='utf-8')
print('Added six main-path bridge exercises; preserved all existing learner implementations.')
