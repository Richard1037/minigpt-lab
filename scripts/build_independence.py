"""独立实现、修错与迁移练习。重复构建不覆盖个人文件。"""
import ast,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'solutions')]
import transformer_layers as ref
from _transformer_checks import probe
from _foundation_checks import pack
b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
new={}
def save(meta,source):
    imports='import numpy as np\nfrom learner_core import Parameter, Module, initialize\n'
    if meta.get('dependencies'):
        imports+='if __name__ != "student":\n'
        for d in meta['dependencies']:
            item=new.get(d,b['lessons'].get(d));names=item.get('exports',[item.get('class','')])
            imports+=f"    from {Path(item['file']).stem} import {', '.join(names)}\n"
    meta['starter']=imports+'\n'+source
    checker='_checks' if meta['kind']=='array' else '_foundation_checks' if meta['kind']=='layer' else '_transformer_checks'
    arg='solve' if meta['kind']=='array' else 'globals()'
    code=meta['starter']+f'\n\nif __name__ == "__main__":\n    from {checker} import run_local\n    run_local("{meta["id"]}", {arg})\n'
    path=ROOT/'exercises'/meta['file']
    if not path.exists():path.write_text(code,encoding='utf-8')
    meta['local_code']=path.read_text(encoding='utf-8');meta['assessment_revision']=1
    meta.setdefault('steps',['根据输入输出约定完成实现。','运行全部公开用例，在新尺寸下验证。'])
    meta.setdefault('extension','<p>独立完成后，再展开参考实现对照。差分理论放在基础层拓展中。</p>')
    new[meta['id']]=meta

def array_bug(id,title,args,wrong,correct,body,inputs):
    ns={'np':np};exec(correct,ns);cases=[]
    for i,items in enumerate(inputs):
        arrays=[np.asarray(v,dtype=float) for v in items]
        cases.append(dict(label=['真实多批次定位错误','改变尺寸检验修复','零值/不同上游检验'][i],args=[v.tolist() for v in arrays],expected=ns['solve'](*arrays).tolist(),category='修复与迁移'))
    save(dict(id=id,title=title,goal=body,body='<h3>待修复行为</h3><p>'+body+'</p><p>编辑器给出可运行但不正确的实现。定位原因后修复，并在不同尺寸上通过测试。</p>',kind='array',args=args,cases=cases,reference=correct,file=id+'.py',repair=True,dependencies=[]),wrong)

array_bug('fix_bias','修错：偏置梯度求和轴',['G'],
'def solve(G):\n    # 此实现能运行，但梯度轴不正确。\n    return G.sum(axis=-1)',
'def solve(G):\n    return G.sum(axis=(0,1))',
'G（B,T,D）是 Y=XW+b 的上游梯度；返回 db（D，）。同一偏置在所有 batch/token 被重复使用。',
[[np.arange(24).reshape(2,3,4)/10],[np.arange(30).reshape(3,2,5)/7],[np.zeros((2,4,3))]])
array_bug('fix_heads','修错：合并 head 时打乱 token',['X'],
'def solve(X):\n    B,H,T,d=X.shape\n    return X.reshape(B,T,H*d)',
'def solve(X):\n    B,H,T,d=X.shape\n    return X.transpose(0,2,1,3).reshape(B,T,H*d)',
'X（B,H,T,d）返回（B,T,H×d）。值的数量正确仍可能对应到错误的 token。',
[[np.arange(48).reshape(2,2,3,4)],[np.arange(72).reshape(3,3,2,4)],[np.arange(48).reshape(2,3,4,2)]])
array_bug('fix_residual','修错：残差漏掉直连梯度',['direct','branch'],
'def solve(direct,branch):\n    return branch.copy()',
'def solve(direct,branch):\n    return direct+branch',
'direct 与 branch 都为（B,T,C），分别是直连和子网络回流；返回两者相加的新数组，保持两个输入原值。',
[[np.ones((2,3,4)),np.full((2,3,4),2)],[np.arange(30).reshape(3,2,5),np.full((3,2,5),-1)],[np.ones((2,2,3)),np.zeros((2,2,3))]])
for id,base,title,wrong,body in [
 ('fix_shared','t_gpt','修错：共享词表漏掉输出头梯度',lambda s:s.replace('self.token.weight.grad += g.reshape(-1,g.shape[-1]).T @ self._h.reshape(-1,C)','pass  # 此处漏掉一条共享参数梯度路径'), '代码能前向运行，但共享词表只收到输入查表路径的梯度。修复 backward，让未查表行也能收到输出头梯度。'),
 ('fix_clear','t_train','修错：训练时忘记清空梯度',lambda s:s.replace('model.zero_grad()','pass  # 此处没有清除上一步梯度'), '第一步可能正常，之后梯度不断累加。修复每步的状态管理，保持训练返回值和更新顺序不变。')]:
    meta=copy.deepcopy(b['lessons'][base]);meta.update(id=id,title=title,goal=body,body='<h3>待修复行为</h3><p>'+body+'</p>'+meta['body'],file=id+'.py',repair=True)
    meta.pop('lab',None);meta.pop('experiment_dependencies',None)
    meta['steps']=['先预测哪个测评会失败，再提交观察定位。','修复当前模块，不修改前置模块以绕过错误。']
    save(meta,wrong(meta['reference']))

def empty_class(source):
    tree=ast.parse(source)
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef):
            methods=[]
            for fn in cls.body:
                if isinstance(fn,ast.FunctionDef):
                    # 保留接口，移除构造和方法的现成实现。
                    fn.body=ast.parse('raise NotImplementedError("TODO：独立实现 '+fn.name+'")').body;methods.append(fn)
            cls.body=methods
    return ast.unparse(ast.fix_missing_locations(tree))
for id,base,title,level,body in [
 ('rebuild_linear','layer_linear','独立实现：从接口写出 Linear','接口练习','只给 Linear(C,D,use_bias=True) 接口。自己创建 W（C,D）与可选 b（D，），实现缓存、前向、输入梯度和参数梯度累加。'),
 ('rebuild_block','t_block','独立组装：从接口写出 Block','组装练习','自己构造 norm1、attn、norm2、mlp 四个子模块；采用 Pre-Norm 和两条残差。forward/backward 接受（B,T,C），构造参数为 C,H,hidden。'),
 ('rebuild_gpt','t_gpt','独立实现：从空方法组装 miniGPT','最终独立实现','自己完成 GPT 的构造、前向和完整反向。参数 V,C,H,hidden,layers,context,seed；公开属性 token、position、blocks、norm、context。二维权重按 seed 的单一 RNG 依次 N(0,0.02) 初始化，两个残差出口除 √(2×layers)。词表共享、位置归约和参数去重均需自己处理。')]:
    meta=copy.deepcopy(b['lessons'][base]);meta.update(id=id,title=title,goal=body,body='<h3>'+level+'</h3><p>'+body+'</p>'+meta['body'],file=id+'.py',independent=True)
    meta.pop('residual_trace',None);meta['steps']=['先画出子模块、参数和缓存清单。','自行实现构造函数与全部方法；参考实现默认收起。','通过原始接口验收及新增迁移输入。']
    if id=='rebuild_gpt':
        meta['dependencies']=[('rebuild_block' if d=='t_block' else d) for d in meta['dependencies']]
        # 完整综合测试需要训练、采样、保存接口；它们仍是个人实现。
        deps=b['lessons']['t_project']['dependencies']+['t_project']
        meta['dependencies']=list(dict.fromkeys([d for d in deps if d not in ['t_gpt','t_block']]+['rebuild_block']))
        for c in b['lessons']['t_project']['cases']:meta['cases'].append(copy.deepcopy(c))
        r=np.random.default_rng(120)
        for conf,shape in [([7,6,3,11,2,7,13],(3,4)),([9,8,4,13,1,6,21],(2,5))]:
            ids=r.integers(0,conf[0],size=shape);g=r.normal(size=(*shape,conf[0]))
            c=dict(mode='module',label=f'迁移：V={conf[0]},C={conf[1]},H={conf[2]},B/T={shape}',input=dict(**{'class':'GPT'},init=conf,ids=ids.tolist(),G=g.tolist()),category='新尺寸迁移')
            c['expected']=pack(probe(vars(ref),c));meta['cases'].append(c)
    save(meta,empty_class(meta['reference']))
b['lessons'].update(new)
# 专项插在已学模块之后；独立组装按逐步减少支持的顺序安排。
for stage in b['stages']:
    for mod in stage['modules']:
        mod['tasks']=[id for id in mod['tasks'] if id not in new]
    stage['modules']=[mod for mod in stage['modules'] if mod['tasks']]
    if stage['title'].startswith('第四'):stage['modules'].append(dict(name='独立实现基础层',tasks=['rebuild_linear']))
    if stage['title'].startswith('第六'):stage['modules'].append(dict(name='独立组装 Block',tasks=['rebuild_block']))
    if stage['title'].startswith('第九'):stage['modules'].append(dict(name='最终独立实现',tasks=['rebuild_gpt']))
b['stages']=[s for s in b['stages'] if not s.get('optional')]
b['stages'].append(dict(title='拓展 · 典型错误修复',optional=True,outcome='按需练习排错，不计入主线进度，也不作为主线前置。',modules=[dict(name='拓展排错练习',tasks=['fix_bias','fix_heads','fix_residual','fix_shared','fix_clear'])]))
new['rebuild_gpt']['lab']=True
new['rebuild_gpt']['experiment_dependencies']=new['rebuild_gpt']['dependencies']
quizzes={
 'layer_linear':[
  dict(prompt='dW[i,j] 应汇总哪些位置的贡献？',options=['所有 batch 和 token','只汇总输出特征 j','只取第一个 batch'],answer=0,explain='W[i,j] 在每个位置共享，dW[i,j]=Σ_b,t X[b,t,i]G[b,t,j]。'),
  dict(prompt='求 dX[b,t,i] 时，应该对哪个轴求和？',options=['batch','输出特征 j','token'],answer=1,explain='该输入分量影响同一 token 的全部输出特征，所以对 j 累加。'),
  dict(prompt='上游 G 已来自平均损失，Linear.backward 是否再次除 B×T？',options=['再除一次','不再平均'],answer=1,explain='平均系数已包含在 G 中，再除会把梯度缩小一次。')],
 't_attention':[
  dict(prompt='同一个 V[s,d] 被多个 query 读取，dV 应怎样计算？',options=['只保留最后一个 query 的贡献','汇总所有 query：AᵀG','把贡献取平均'],answer=1,explain='这是共享值的梯度汇总。对 query 轴求和，而不是覆盖或平均。'),
  dict(prompt='因果注意力中，一行表示什么？',options=['一个 query 对各 key 的读取概率','一个 batch','一个特征的参数'],answer=0,explain='最后两轴是 query 和 key；每行在允许读取的 key 上归一化。'),
  dict(prompt='Softmax 反向能否只做 G⊙P？',options=['可以','不可以，需要同一行的耦合项'],answer=1,explain='一个 score 会改变该行全部概率，必须减去 sum(G⊙P)。')],
 't_block':[
  dict(prompt='y=x+F(x)，回到 x 的梯度是什么？',options=['只用 F.backward(G)','G+F.backward(G)','两条路径各分一半 G'],answer=1,explain='两条分支分别接收完整 G，回到同一变量时相加。'),
  dict(prompt='两次 Norm 前向能否复用一个会覆盖缓存的实例？',options=['始终可以','不能，否则第一条路径的缓存可能丢失'],answer=1,explain='每个实例需保留与自己这次前向对应的缓存。')],
 't_gpt':[
  dict(prompt='共享词表 E 的梯度来自哪里？',options=['只有输入查表','只有输出头','输入查表和输出头两条路径'],answer=2,explain='logits=HEᵀ 也依赖 E，未查表行也可能有输出头梯度。'),
  dict(prompt='位置向量广播到所有 batch，反向怎样回到位置表？',options=['取 batch=0','沿 batch 求和','沿特征求和'],answer=1,explain='同一位置向量被 B 条序列使用，贡献需相加。')]
}
quizzes['linear']=copy.deepcopy(quizzes['layer_linear'])
for id,qs in quizzes.items():b['lessons'][id]['checkpoints']=qs
b['version']=5
(ROOT/'webapp/foundations.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf-8')
arrays=json.loads((ROOT/'exercises/cases.json').read_text(encoding='utf-8'))
for id,m in new.items():
    if m['kind']=='array':arrays[id]=m['cases']
(ROOT/'exercises/cases.json').write_text(json.dumps(arrays,ensure_ascii=False,indent=2),encoding='utf-8')
print('Independent curriculum:',len(b['lessons']),'lessons;',sum(len(m.get('cases',m.get('questions',[]))) for m in b['lessons'].values()),'checks')

# 所有章节生成后统一排版；不会触碰个人代码或测评。
import runpy
runpy.run_path(str(ROOT/'scripts/build_bridges.py'),run_name='__main__')
runpy.run_path(str(ROOT/'scripts/refine_course.py'),run_name='__main__')
runpy.run_path(str(ROOT/'scripts/course_math.py'),run_name='__main__')
