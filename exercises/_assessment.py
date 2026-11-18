"""行为/梯度验收：只执行个人模块，不导入 solutions。"""
import traceback
import numpy as np
from foundation_core import Parameter, Module

CATEGORIES={'gradient':'反向传播','accumulation':'状态管理','cache':'缓存隔离','softmax':'数学性质','causal':'因果性','shared':'参数共享','overfit':'训练行为','evaluation':'评估状态','generation':'生成行为','checkpoint':'恢复一致性','dropout':'训练与评估','infrastructure':'参数管理','project_step':'完整训练入口'}

def named_parameters(obj,scope=None):
    P=(scope or {}).get('Parameter',Parameter)
    M=(scope or {}).get('Module',Module)
    found=[];seen=set()
    def visit(v,path=''):
        if isinstance(v,P):
            if id(v) not in seen:seen.add(id(v));found.append((path,v))
        elif isinstance(v,M):
            for k,child in vars(v).items():
                if not k.startswith('_'):visit(child,(path+'.' if path else '')+k)
        elif isinstance(v,(list,tuple,dict)):
            for k,child in (v.items() if isinstance(v,dict) else enumerate(v)):visit(child,path+'.'+str(k))
    visit(obj);return found

def diagnose(actual,expected,path='输出',atol=2e-6,rtol=1e-6):
    errors=[]
    if isinstance(expected,dict):
        if not isinstance(actual,dict):return [path+'：应返回字典']
        for k,v in expected.items():
            if k not in actual:errors.append(path+'.'+k+'：缺少此项')
            else:errors.extend(diagnose(actual[k],v,path+'.'+k,atol,rtol))
        for k in actual.keys()-expected.keys():errors.append(path+'.'+k+'：多余返回项')
        return errors
    if expected is None:return [] if actual is None else [path+'：应返回 None']
    if isinstance(expected,(str,bool)):return [] if actual==expected else [path+'：预期 '+str(expected)+'，实际 '+str(actual)]
    if not isinstance(actual,np.ndarray):return [path+'：应返回 NumPy 数组，实际 '+type(actual).__name__]
    e=np.asarray(expected)
    if actual.shape!=e.shape:return [f'{path}：形状应为 {e.shape}，实际 {actual.shape}']
    ok=np.isclose(actual,e,atol=atol,rtol=rtol)
    if not np.all(ok):
        index=tuple(int(i) for i in np.argwhere(~ok)[0]) if actual.ndim else ()
        return [f'{path}{index}：预期 {e[index]:.8g}，实际 {actual[index]:.8g}；最大绝对误差 {np.max(np.abs(actual-e)):.4g}']
    return []

def behavior(scope,c):
    if c['assessment']=='part08_contract':
        from _part08_contract_checks import behavior as check
        return check(scope,c)
    if c['assessment']=='initialization_upgrade':
        from _initialization_upgrade_checks import behavior as check
        return check(scope,c)
    if c['assessment']=='part07_contract':
        from _gpt_contract_checks import behavior as check
        return check(scope,c)
    if c['assessment']=='attention_component':
        from _attention_component_checks import behavior as check
        return check(scope,c)
    if c['assessment']=='part06_contract':
        from _attention_contract_checks import behavior as check
        return check(scope,c)
    if c['assessment'] in ('foundation_gate','branch_composition','branch_gradient','branch_state'):
        from _capstone_checks import behavior as check
        return check(scope,c)
    if c['assessment']=='probability_contract':
        from _probability_checks import behavior as check
        return check(scope,c)
    if c['assessment'] in ('independent_linear_build','independent_linear_composition'):
        from _independent_linear_checks import check_build,check_composition
        return (check_build if c['assessment']=='independent_linear_build' else check_composition)(scope,c)
    if c['assessment']=='mlp_composition':
        from _mlp_checks import check_composition
        return check_composition(scope,c)
    if c['assessment']=='rmsnorm_initialization':
        from _rmsnorm_checks import check_initialization
        return check_initialization(scope,c)
    if c['assessment']=='embedding_initialization':
        from _embedding_checks import check_initialization
        return check_initialization(scope,c)
    p=c['input'];kind=c['assessment'];rng=np.random.default_rng(p.get('seed',41));report={}
    def check(condition,message):
        if not condition:raise AssertionError(message)
    def close(a,b,label,atol=3e-6):
        a,b=np.asarray(a),np.asarray(b)
        check(a.shape==b.shape,f'{label}：形状 {a.shape} 应为 {b.shape}')
        if not np.allclose(a,b,rtol=3e-4,atol=atol):
            index=tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(a-b)),a.shape)) if a.ndim else ()
            raise AssertionError(f'{label}{index}：解析值 {a[index]:.8g}，对照值 {b[index]:.8g}')
    def model():return scope[p.get('class','GPT')](*p.get('init',[]))
    if kind=='gradient':
        obj=model();cls=p['class'];shape=tuple(p['shape'])
        if cls=='AttentionCore':
            args=[rng.normal(size=shape) for _ in range(3)]
        elif cls=='GPT':args=[rng.integers(0,p['init'][0],size=shape)]
        elif cls=='Embedding':args=[rng.integers(0,p['init'][0],size=shape)]
        else:args=[rng.normal(size=shape)]
        targets=rng.integers(0,shape[-1],size=shape[:-1]) if cls=='CrossEntropy' else None
        forward=lambda:obj.forward(args[0],targets) if targets is not None else obj.forward(*args)
        y=forward();g=rng.normal(size=np.asarray(y).shape);dx=obj.backward(g)
        derivatives=list(dx) if cls=='AttentionCore' else [dx]
        tensors=[]
        if cls not in ['GPT','Embedding']:tensors += [(f'dInput[{i}]',x,d) for i,(x,d) in enumerate(zip(args,derivatives))]
        tensors += [(name+'.grad',param.data,param.grad.copy()) for name,param in named_parameters(obj,scope)]
        h=1e-5;tested=0;max_error=0.
        for name,array,grad in tensors:
            check(isinstance(grad,np.ndarray),name+' 必须是 NumPy 数组')
            check(grad.shape==array.shape,name+' 形状错误')
            count=min(array.size,p.get('coordinates',4))
            indices=rng.choice(array.size,size=count,replace=False)
            for flat in indices:
                index=tuple(int(i) for i in np.unravel_index(int(flat),array.shape));old=array[index]
                try:
                    array[index]=old+h;plus=float(np.sum(forward()*g))
                    array[index]=old-h;minus=float(np.sum(forward()*g))
                finally:array[index]=old
                numeric=(plus-minus)/(2*h);error=abs(float(grad[index])-numeric)
                max_error=max(max_error,error);tested+=1
                close(np.array(grad[index]),np.array(numeric),name+str(index))
        report=dict(检查坐标数=tested,参数张量数=len(named_parameters(obj,scope)),最大绝对误差=max_error)
    elif kind=='cache':
        obj=model();other=model();cls=p['class'];shape=tuple(p['shape'])
        left=named_parameters(obj,scope);right=named_parameters(other,scope)
        check([n for n,_ in left]==[n for n,_ in right],'重复构造的参数路径不一致')
        for (_,source),(_,dest) in zip(left,right):dest.data[...]=source.data
        x=rng.integers(0,p['init'][0],size=shape) if cls=='Embedding' else rng.normal(size=shape)
        original=x.copy();out=obj.forward(x);g=rng.normal(size=out.shape)
        if cls=='Embedding':x[...] = (x+1)%p['init'][0]
        else:x[...] = x+37
        actual=obj.backward(g)
        other.forward(original);expected=other.backward(g)
        if cls=='Embedding':check(actual is None,'整数 ids 的反向返回值必须是 None')
        else:close(actual,expected,'使用原始前向缓存的 dX')
        for (name,a),(_,b) in zip(left,right):close(a.grad,b.grad,name+' 使用原始缓存的参数梯度')
        report=dict(调用方输入已修改=True,输入梯度符合原始前向=True,参数张量数=len(left))
    elif kind=='accumulation':
        obj=model();shape=tuple(p['shape']);integer=p['class']=='Embedding'
        x=rng.integers(0,p['init'][0],size=shape) if integer else rng.normal(size=shape)
        y=obj(x);g=rng.normal(size=y.shape);weights=[v.data.copy() for v in obj.parameters()]
        obj.backward(g);first=[v.grad.copy() for v in obj.parameters()]
        obj.zero_grad();y=obj(x);obj.backward(2*g);second=[v.grad.copy() for v in obj.parameters()]
        obj(x);obj.backward(g)
        for (name,param),a,b,w in zip(named_parameters(obj,scope),first,second,weights):
            close(param.grad,a+b,name+'.grad 累加');close(param.data,w,name+'.data 不得更新')
        obj.zero_grad()
        for name,param in named_parameters(obj,scope):check(np.all(param.grad==0),name+' 清零失败')
        report=dict(参数张量数=len(weights),不同上游梯度累加正确=True,清零不改权重=True)
    elif kind=='softmax':
        obj=scope['Softmax']();x=rng.normal(size=(2,3,5))*3;a=obj(x).copy();b=obj(x+10000)
        close(a,b,'Softmax 平移不变');close(b.sum(-1),np.ones((2,3)),'概率行和')
        check(np.all(b>=0),'概率不能为负');obj(x);close(obj.backward(np.ones_like(x)),np.zeros_like(x),'概率和的输入梯度')
        report=dict(六个位置概率和均为一=True,大常数平移不变=True,全一上游对应零梯度=True)
    elif kind=='causal':
        obj=model()
        if p['class']=='GPT':
            ids=np.array([[0,1,2,3],[1,2,3,4]]);other=ids.copy();other[:,-1]=0
            a=obj(ids).copy();b=obj(other);close(a[:,:-1],b[:,:-1],'未来 token 泄漏',1e-10)
            g=np.zeros_like(a);g[:,0]=rng.normal(size=g[:,0].shape);obj.zero_grad();obj(ids);obj.backward(g)
            close(obj.position.weight.grad[1:],np.zeros_like(obj.position.weight.grad[1:]),'首位置不能给未来位置梯度',1e-10)
        else:
            q,k,v=[rng.normal(size=(2,2,4,3)) for _ in range(3)]
            a=obj(q,k,v).copy();k2=k.copy();v2=v.copy();k2[:,:,-1]+=50;v2[:,:,-1]-=20
            b=obj(q,k2,v2);close(a[:,:,:-1],b[:,:,:-1],'未来 K/V 泄漏',1e-10)
            obj(q,k,v);g=np.zeros_like(a);g[:,:,0]=1;dq,dk,dv=obj.backward(g)
            close(dk[:,:,1:],np.zeros_like(dk[:,:,1:]),'未来 dK 必须为零',1e-10);close(dv[:,:,1:],np.zeros_like(dv[:,:,1:]),'未来 dV 必须为零',1e-10)
        report=dict(未来输入变化不影响过去=True,未来位置梯度为零=True)
    elif kind=='shared':
        obj=model();ids=np.array([[0,1],[1,0]]);y=obj(ids);g=rng.normal(size=y.shape);obj.backward(g)
        check(sum(v is obj.token.weight for v in obj.parameters())==1,'共享词表必须只收集一次')
        # 未查表行仅走输出头，使用中心差分直接验证该路径。
        param=obj.token.weight;index=(p['init'][0]-1,0);analytic=param.grad[index];old=param.data[index];h=1e-5
        param.data[index]=old+h;a=float(np.sum(obj(ids)*g));param.data[index]=old-h;b=float(np.sum(obj(ids)*g));param.data[index]=old
        close(np.array(analytic),np.array((a-b)/(2*h)),'未查表行仍应收到输出头梯度')
        check(abs(float(analytic))>1e-9,'输出头梯度意外缺失')
        report=dict(共享参数只出现一次=True,未查表行也收到输出头梯度=True)
    elif kind in ['overfit','evaluation']:
        obj=model();ids=np.array([[0,1,2],[1,2,3]]);targets=(ids+1)%p['init'][0]
        if kind=='overfit':
            initial=[v.data.copy() for v in obj.parameters()]
            losses=scope['train'](obj,ids,targets,p.get('steps',45),.02)
            check(isinstance(losses,np.ndarray) and losses.shape==(p.get('steps',45),),'train 必须返回每步损失数组')
            check(np.all(np.isfinite(losses)),'损失非有限')
            check(losses[-1]<losses[0]*.5,f'固定批次未明显降损失：{losses[0]:.5f} → {losses[-1]:.5f}')
            check(any(not np.array_equal(v.data,w) for v,w in zip(obj.parameters(),initial)),'参数没有更新')
            report=dict(起始损失=float(losses[0]),最终损失=float(losses[-1]),要求='最终损失 < 起始损失 × 0.5')
        else:
            for param in obj.parameters():param.grad[...] = .7
            weights=[v.data.copy() for v in obj.parameters()];grads=[v.grad.copy() for v in obj.parameters()]
            loss=np.array(np.mean(obj(ids)**2))
            for param,w,g in zip(obj.parameters(),weights,grads):
                check(np.array_equal(param.data,w),'评估 forward 修改了权重');check(np.array_equal(param.grad,g),'评估 forward 修改了参数梯度')
            report=dict(检查用标量_mean_logits平方=float(loss),权重与已有梯度保持不变=True)
    elif kind=='dropout':
        obj=scope['Dropout'](.3,5);x=np.ones((2,3,8));a=obj.forward(x).copy();b=obj.forward(x).copy()
        check(not np.array_equal(a,b),'训练 RNG 每次 forward 必须推进，不能重置种子')
        close(obj.backward(np.ones_like(x)),b,'反向必须复用最近一次掩码')
        close(obj.forward(x,training=False),x,'评估时禁用 Dropout');close(obj.backward(x),x,'评估反向为恒等')
        report=dict(训练掩码随调用推进=True,反向复用同一掩码=True,评估关闭掩码=True)
    elif kind=='generation':
        obj=model();ids=np.array([[0,1,2],[1,2,3]]);before=ids.copy()
        weights=[v.data.copy() for v in obj.parameters()];grads=[v.grad.copy() for v in obj.parameters()]
        out=scope['generate'](obj,ids,6,.8,1,7)
        check(out.shape==(2,9) and np.issubdtype(out.dtype,np.integer),'生成必须返回 (B,T+steps) 整数数组')
        check(np.array_equal(out[:,:3],ids) and np.array_equal(ids,before),'提示必须保留且不修改输入')
        check(np.all((out>=0)&(out<p['init'][0])),'生成编号越界')
        repeat=scope['generate'](obj,ids,6,.8,1,7);check(np.array_equal(out,repeat),'相同种子/参数应复现')
        for param,w,g in zip(obj.parameters(),weights,grads):check(np.array_equal(param.data,w) and np.array_equal(param.grad,g),'生成不应更新权重或梯度')
        report=dict(双批次滑窗输出形状=list(out.shape),提示保留=True,权重梯度不变=True,可复现=True)
    elif kind=='checkpoint':
        import tempfile
        from pathlib import Path
        obj=model();opt=scope['AdamW'](obj.parameters(),lr=.01)
        for param in obj.parameters():param.grad[...] = rng.normal(size=param.grad.shape)
        opt.step();weights=[v.data.copy() for v in obj.parameters()];ms=[v.copy() for v in opt.m];vs=[v.copy() for v in opt.v]
        expected_rng=np.random.default_rng();expected_rng.bit_generator.state=rng.bit_generator.state
        temp_root=Path(__file__).resolve().parents[1]/'.check-tmp';temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path=str(Path(tmp)/'state.npz');scope['save_checkpoint'](path,obj,opt,rng)
            for param in obj.parameters():param.data[:]+=3;param.grad[:]=9
            for value in opt.m:value[:]+=5
            for value in opt.v:value[:]+=7
            opt.t=19;opt.lr=.4;rng.random(7)
            scope['load_checkpoint'](path,obj,opt,rng)
        focus=p['focus']
        if focus=='weights':
            for (name,param),w in zip(named_parameters(obj,scope),weights):close(param.data,w,name+' 权重恢复',1e-12)
            check(all(np.all(v.grad==0) for v in obj.parameters()),'加载必须清零旧梯度')
        if focus=='optimizer':
            check(opt.t==1 and opt.lr==.01,'时间步或学习率未恢复')
            for i,(m,v,em,ev) in enumerate(zip(opt.m,opt.v,ms,vs)):close(m,em,f'm[{i}]',1e-12);close(v,ev,f'v[{i}]',1e-12)
        if focus=='rng':close(rng.random(12),expected_rng.random(12),'恢复后的下一段随机数',1e-12)
        report=dict(检查对象=focus,原地扰动后成功恢复=True)
    elif kind=='project_step':
        obj=model();ids=np.array([[0,1,2],[1,2,3]]);targets=(ids+1)%p['init'][0]
        opt=scope['AdamW'](obj.parameters(),lr=.01)
        ce=scope['CrossEntropy']();before=ce(obj(ids),targets)
        weights=[v.data.copy() for v in obj.parameters()]
        # 预置旧梯度，检查入口是否在更新前清零。
        for param in obj.parameters():param.grad[:]=13
        loss=scope['project_step'](obj,opt,ids,targets)
        close(loss,before,'返回值应为更新前平均损失')
        check(opt.t==1,'每步只能执行一次优化器更新')
        check(any(not np.array_equal(v.data,w) for v,w in zip(obj.parameters(),weights)),'训练入口没有更新权重')
        # 从相同配置独立执行指定顺序，对照所有参数。
        other=model();other_opt=scope['AdamW'](other.parameters(),lr=.01)
        other.zero_grad();ce(other(ids),targets);other.backward(ce.backward());scope['clip_grad'](other.parameters(),1.);other_opt.step()
        for (name,a),b in zip(named_parameters(obj,scope),other.parameters()):close(a.data,b.data,name+' 一次完整训练更新',1e-10)
        report=dict(返回更新前损失=True,清除旧梯度=True,完整反向与裁剪=True,仅更新一次=True)
    elif kind=='infrastructure':
        P,M=scope['Parameter'],scope['Module'];x=np.array([[1.,2.],[3.,4.]])
        p1=P(x);x[:]=99;check(not np.any(p1.data==99),'Parameter 必须复制输入')
        if p.get('focus')=='copy':
            check(p1.data.dtype==np.float64 and p1.grad.shape==p1.data.shape and np.all(p1.grad==0),'dtype/grad 初始化错误')
            return dict(输入复制=True,统一_float64=True,同形零梯度=True)
        p2=P([2.,3.]);child=M();child.w=p1;root=M();root.list=[child,{'other':p2}];root.shared=p1;root._cache=P([8.])
        if p.get('focus')=='empty':
            empty=M();empty._cache=P([3.]);check(empty.parameters()==[],'空模块/缓存不应返回参数');empty.zero_grad();return dict(空模块可清零=True,缓存参数被排除=True)
        items=root.parameters();check(len(items)==2 and set(map(id,items))=={id(p1),id(p2)},'列表/字典递归、共享去重或跳过缓存错误')
        for v in items:v.grad[:]=3
        root.zero_grad();check(all(np.all(v.grad==0) for v in items),'嵌套清零失败')
        report=dict(输入复制=True,列表字典递归=True,共享去重=True,缓存不当参数=True)
    else:raise ValueError(kind)
    return report

def evaluate_special(scope,c):
    try:
        out=behavior(scope,c)
        return dict(label=c['label'],passed=True,actual=out,category=c.get('category'),diagnostics=[])
    except Exception as e:
        return dict(label=c['label'],passed=False,errorText=traceback.format_exc(limit=4),category=c.get('category'),diagnostics=[str(e)])
