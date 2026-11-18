/* 过程图：保留完整张量，用位置移动表达来源；不改变 NumPy 的内存语义。 */
(function(root){
 'use strict';
 const supported=new Set(['shape','axes','reduce','mean_backward','center_backward','broadcast','reshape','restore','transpose','heads','copy','att_merge','layer_embedding','scatter','t_loss','t_data','t_sample','t_generate','t_layernorm','t_dropout','att_scores','att_probs','att_weighted','att_value_backward','att_score_backward']);
 const shape=x=>Array.isArray(x)?[x.length,...(x.length?shape(x[0]):[])]:[];
 const coords=s=>s.length?slice(s[0]).flatMap(i=>coords(s.slice(1)).map(rest=>[i,...rest])):[[]];
 const slice=n=>Array.from({length:n},(_,i)=>i);
 const at=(a,c)=>c.reduce((part,i)=>part[i],a);
 const build=(s,f,p=[])=>s.length?slice(s[0]).map(i=>build(s.slice(1),f,[...p,i])):f(p);
 const copy=x=>JSON.parse(JSON.stringify(x));
 const key=c=>c.join(',');
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const fmt=x=>x===null?'?':typeof x==='string'?x==='\n'?'↵':x===' '?'␠':x:x===-Infinity?'−∞':Number(Number(x).toFixed(5)).toString();
 function makePlan(id,c,generationScores){
  const panels=[],events=[],args=c.args||[],X=args[0];
  const add=(id,title,values,axes,initial='empty')=>{panels.push({id,title,values,axes,initial,shape:shape(values)});return id;};
  const link=(from,source,to,target,mode='replace',value)=>({from,source,to,target,mode,value});
  const event=(title,note,visible,moves,extra=[],tex='',clear=[])=>events.push({title,note,visible,moves,extra,tex,clear});
  let note='',expected={};
  if(id==='t_dropout'){
   const input=c.input,X=input.X,G=input.G,[B,T,C]=shape(X),p=input.init[0],seed=input.init[1],training=input.kwargs?.training??true,axes=['batch b','token t','特征 c'];
   const mask=build([B,T,C],([b,t,j])=>G[b][t][j]!==0?c.expected.dX[b][t][j]/G[b][t][j]:X[b][t][j]!==0?c.expected.Y[b][t][j]/X[b][t][j]:null);
   add('x','X · 前向输入',X,axes,'full');add('mask','本公开样例的缓存掩码（由输出还原）',mask,axes,'full');add('y','Y · 前向输出',c.expected.Y,axes);add('g','G · 上游梯度',G,axes,'full');add('dx','dX · 使用同一掩码',c.expected.dX,axes);
   for(let b=0;b<B;b++)for(let t=0;t<T;t++){
    const extra=slice(C).map(j=>({panel:'mask',coord:[b,t,j]}));
    event(`前向：b=${b}, t=${t}`,training?`训练模式：这个位置的 X 与缓存掩码逐项相乘；保留项按 1/(1−p) 缩放。`:'评估模式：掩码为 1，输入原样传递。',['x','mask','y'],slice(C).map(j=>link('x',[b,t,j],'y',[b,t,j],'replace',X[b][t][j]*(mask[b][t][j]??0))),extra,String.raw`Y=X\odot M`);
    event(`反向：b=${b}, t=${t}`,'将同一次前向缓存的掩码乘到 G 对应位置；本图不重新采样。',['g','mask','dx'],slice(C).map(j=>link('g',[b,t,j],'dx',[b,t,j],'replace',G[b][t][j]*(mask[b][t][j]??0))),extra,String.raw`\mathrm dX=G\odot M`);
   }
   note=`公开样例：p=${p}，种子=${seed}，${training?'训练':'评估'}模式，${B} 个 batch × ${T} 个 token。掩码由公开输出还原，非浏览器重新采样；“?” 表示该位置 X 和 G 都为 0，无法从输出推断掩码。`;expected.y=c.expected.Y;expected.dx=c.expected.dX;
  }else if(id==='t_layernorm'){
   const input=c.input,X=input.X,G=input.G,[B,T,C]=shape(X),eps=input.init[1]??1e-5,small=[B,T,1],axes=['batch b','token t','特征 c'];
   const mean=build(small,([b,t])=>X[b][t].reduce((s,v)=>s+v,0)/C);
   const center=build([B,T,C],([b,t,j])=>X[b][t][j]-mean[b][t][0]);
   const r=build(small,([b,t])=>1/Math.sqrt(center[b][t].reduce((s,v)=>s+v*v,0)/C+eps));
   const h=build([B,T,C],([b,t,j])=>center[b][t][j]*r[b][t][0]);
   const gmean=build(small,([b,t])=>G[b][t].reduce((s,v)=>s+v,0)/C);
   const ghmean=build(small,([b,t])=>G[b][t].reduce((s,v,j)=>s+v*h[b][t][j],0)/C);
   const dx=build([B,T,C],([b,t,j])=>r[b][t][0]*(G[b][t][j]-gmean[b][t][0]-h[b][t][j]*ghmean[b][t][0]));
   add('x','X · 所有位置的输入',X,axes,'full');add('mean','μ · 每个位置的均值',mean,axes);add('center','z=X−μ',center,axes);add('r','r · 逆标准差',r,axes);add('h','归一化结果 h；初始 w=1、b=0 时也是 Y',h,axes);
   add('g','G · 输出上游梯度',G,axes,'full');add('gm','mean(G)',gmean,axes);add('ghm','mean(G*h)',ghmean,axes);add('dx','dX · 三条贡献合并',dx,axes);
   add('dw','w.grad · 跨位置累计',slice(C).map(()=>0),['特征 c'],'full');add('db','b.grad · 跨位置累计',slice(C).map(()=>0),['特征 c'],'full');
   for(let b=0;b<B;b++)for(let t=0;t<T;t++){
    const pos=`b=${b}, t=${t}`,row=key=>slice(C).map(j=>({panel:key,coord:[b,t,j]}));
    event(`求均值：${pos}`,`仅汇总这个 token 的 ${C} 个特征，其他 batch/token 各自独立。`,['x','mean'],slice(C).map(j=>link('x',[b,t,j],'mean',[b,t,0],'replace',mean[b][t][0])),[],String.raw`\mu=\operatorname{mean}_c(X)`);
    event(`中心化：${pos}`,'本位置每个特征减去同一个均值。',['x','mean','center'],slice(C).map(j=>link('x',[b,t,j],'center',[b,t,j],'replace',center[b][t][j])),[{panel:'mean',coord:[b,t,0]}],String.raw`z=X-\mu`);
    event(`求缩放：${pos}`,`先求 z 的均方，加 ε=${eps}，再取平方根的倒数。`,['center','r'],slice(C).map(j=>link('center',[b,t,j],'r',[b,t,0],'replace',r[b][t][0])),[],String.raw`r=(\operatorname{mean}_c(z^2)+\epsilon)^{-1/2}`);
    event(`归一化：${pos}`,'z 的每个特征乘同一 r。公开样例中缩放 w 全为 1、偏置 b 全为 0，因此 h 同时是前向输出。',['center','r','h'],slice(C).map(j=>link('center',[b,t,j],'h',[b,t,j],'replace',h[b][t][j])),[{panel:'r',coord:[b,t,0]}],String.raw`h=z\,r,\qquad Y=h\odot w+b`);
    event(`反向汇总：${pos}`,'分别汇总当前 token 的 G 和逐项乘积 G*h；两个均值都只沿特征轴计算。',['g','h','gm','ghm'],slice(C).flatMap(j=>[link('g',[b,t,j],'gm',[b,t,0],'replace',gmean[b][t][0]),link('g',[b,t,j],'ghm',[b,t,0],'replace',ghmean[b][t][0])]),row('h'),String.raw`\operatorname{mean}(G),\quad\operatorname{mean}(G\odot h)`);
    event(`回到输入：${pos}`,'G 直接贡献与中心化、尺度两条路径汇合；本例 w=1，故 u=G。',['g','h','r','gm','ghm','dx'],slice(C).map(j=>link('g',[b,t,j],'dx',[b,t,j],'replace',dx[b][t][j])),[...row('h'),{panel:'r',coord:[b,t,0]},{panel:'gm',coord:[b,t,0]},{panel:'ghm',coord:[b,t,0]}],String.raw`\mathrm dX=r\left[G-\operatorname{mean}(G)-h\operatorname{mean}(G\odot h)\right]`);
    event(`参数梯度汇入：${pos}`,'这个 token 向 w.grad 累加 G*h，向 b.grad 累加 G；同一组参数由所有位置共享。',['g','h','dw','db'],slice(C).flatMap(j=>[link('g',[b,t,j],'dw',[j],'add',G[b][t][j]*h[b][t][j]),link('g',[b,t,j],'db',[j],'add',G[b][t][j])]),row('h'),String.raw`\mathrm dw_c\mathrel{+}=G_c h_c,\qquad\mathrm db_c\mathrel{+}=G_c`);
   }
   note=`公开样例使用 w=1、b=0、ε=${eps}；${B} 个 batch × ${T} 个 token 的 ${C} 个特征全部保留。`;expected.h=c.expected.Y;expected.dx=c.expected.dX;expected.dw=c.expected['w.grad'];expected.db=c.expected['b.grad'];
  }else if(id==='t_sample'){
   const input=c.input,X=input.X,[B,V]=shape(X),tau=input.temperature,k=input.top_k,scaled=X.map(row=>row.map(v=>v/tau));
   const masked=scaled.map(row=>{const order=slice(V).sort((a,b)=>row[a]-row[b]||a-b),kept=new Set(k?order.slice(-k):order);return row.map((v,j)=>kept.has(j)?v:-Infinity);});
   const prob=masked.map(row=>{const max=Math.max(...row),exp=row.map(v=>Math.exp(v-max)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(v=>v/sum);});
   add('x','logits · 每条序列的类别分数',X,['batch b','类别 j'],'full');add('scaled','温度缩放后',scaled,['batch b','类别 j']);add('masked','top-k 后的分数',masked,['batch b','类别 j']);add('p','采样概率',prob,['batch b','类别 j']);add('y','样例采样结果 · 类别编号',c.expected.ids,['batch b']);
   for(let b=0;b<B;b++){
    event(`温度缩放：batch ${b}`,`本行所有分数除以温度 ${tau}，只改变这一条序列的分布。`,['x','scaled'],slice(V).map(j=>link('x',[b,j],'scaled',[b,j],'replace',scaled[b][j])),[],String.raw`z_j=x_j/${tau}`);
    event(`保留候选：batch ${b}`,k?`保留分数最高的 ${k} 类，其他位置设为 −∞；同分时遵循稳定排序的末尾规则。`:'top-k=0，保留所有类别。',['scaled','masked'],slice(V).map(j=>link('scaled',[b,j],'masked',[b,j],'replace',masked[b][j])),[],k?String.raw`\mathcal K=\operatorname{topk}(z)` : String.raw`\mathcal K=\{0,\ldots,V-1\}`);
    event(`归一化：batch ${b}`,'对保留分数执行稳定 Softmax，屏蔽位置的概率为 0。',['masked','p'],slice(V).map(j=>link('masked',[b,j],'p',[b,j],'replace',prob[b][j])),[],String.raw`p_j=\operatorname{softmax}(z)_j`);
    const chosen=c.expected.ids[b];event(`抽样：batch ${b}`,`本用例使用种子 ${input.seed} 的 Python 随机生成器，公开样例抽到类别 ${chosen}；此图展示记录的抽样结果，未在浏览器重新实现 NumPy 随机数。`,['p','y'],[link('p',[b,chosen],'y',[b],'replace',chosen)],[],String.raw`\mathrm{id}_{${b}}\sim p_{${b},:}`);
   }
   note=`B=${B}，V=${V}，温度=${tau}，top-k=${k}，种子=${input.seed}；每条序列独立计算概率，并按顺序使用同一个随机生成器。`;expected.y=c.expected.ids;
  }else if(id==='shape'){
   const dims=shape(X),axes=dims.map((_,i)=>'轴 '+i);
   add('x','X · 由 values 转成的数组',X,axes,'full');add('y','shape · 各轴长度',dims,['轴序号']);
   for(let i=0;i<dims.length;i++)event(`读取轴 ${i} 的长度`,`保持其他坐标为 0，沿轴 ${i} 从编号 0 到 ${dims[i]-1}，共有 ${dims[i]} 个位置；将轴长写入形状数组第 ${i} 项。`,['x','y'],slice(dims[i]).map(j=>link('x',dims.map((_,axis)=>axis===i?j:0),'y',[i],'replace',dims[i])),[],String.raw`\operatorname{shape}(X)_{${i}}=${dims[i]}`);
   note='按轴的顺序读取长度；形状数组的元素是轴长，不是 X 中的数据。';expected.y=c.expected;
  }else if(id==='restore'){
   const Y=args[0],target=args[1].map(Number),prefix=target.slice(0,-1),D=target[target.length-1],N=Y.length;
   const axes=[...prefix.map((_,i)=>i===0?'batch b':i===1?'token t':'位置 '+i),'输出特征 j'];
   const unravel=n=>{const co=Array(prefix.length);for(let i=prefix.length-1;i>=0;i--){co[i]=n%prefix[i];n=Math.floor(n/prefix[i]);}return co;};
   add('x','Y₂ · 已展平的位置表',Y,['位置 n','输出特征 j'],'full');add('shape','out_shape · 目标各轴长度',target,['目标轴'],'full');add('y','Y · 恢复后的全部位置',c.expected,axes);
   for(let n=0;n<N;n++){const pos=unravel(n),label=pos.map((v,i)=>`${axes[i]}=${v}`).join('，');event(`恢复第 ${n} 行 → ${label}`,`按目标形状与 C 顺序，将整行 ${D} 个输出特征放回对应位置；同一位置的特征保持顺序。`,['shape','x','y'],slice(D).map(j=>link('x',[n,j],'y',[...pos,j])),[],String.raw`Y_{\text{位置},j}=(Y_2)_{${n},j}`);}
   note=`目标形状 (${target.join(',')})；${N} 行各自恢复到全部前置位置，最后特征轴宽度 ${D}。`;expected.y=c.expected;
  }else if(id==='copy'){
   const [N,C]=shape(X);add('x','X · 原数组',X,['行','列'],'full');add('y','Y · 独立副本',c.expected,['行','列']);
   event('复制原数组',`将全部 ${N}×${C} 个元素复制到独立数组 Y；此时每个坐标的数值与 X 对应。`,['x','y'],coords([N,C]).map(co=>link('x',co,'y',co)));
   event('修改副本的左上角','只将 Y[0,0] 替换成 −99；其余位置和原数组 X 保持原值。',['x','y'],[link('y',[0,0],'y',[0,0],'replace',-99)],[],String.raw`Y_{0,0}=-99`);
   note='数组副本与原数组独立；移动表示复制数值，后续修改不会回写到 X。';expected.y=c.expected;
  }else if(id==='att_value_backward'){
   const A=args[0],G=args[1],[B,H,Tq,Tk]=shape(A),dv=shape(G)[3],zero=build([B,H,Tk,dv],()=>0);
   add('a','A · query 读取 key 的概率',A,['batch b','head h','query t','key s'],'full');add('g','G · 输出上游梯度',G,['batch b','head h','query t','特征 j'],'full');add('y','dV · 各 query 的贡献累计',zero,['batch b','head h','key s','特征 j'],'full');
   for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let s=0;s<Tk;s++)for(let t=0;t<Tq;t++)event(`b=${b}, h=${h}：query ${t} → key ${s}`,`这个 query 以概率 A[${b},${h},${t},${s}] 读取 key ${s}；将该概率乘 G 的各特征，累加到同一个 dV 行。`,['a','g','y'],slice(dv).map(j=>link('g',[b,h,t,j],'y',[b,h,s,j],'add',A[b][h][t][s]*G[b][h][t][j])),[{panel:'a',coord:[b,h,t,s]}],String.raw`\mathrm dV_{${b},${h},${s},j}\mathrel{+}=A_{${b},${h},${t},${s}}G_{${b},${h},${t},j}`);
   note=`每个 key 的内容可能被 ${Tq} 个 query 使用；沿 query 汇总，全部 ${B} 个 batch、${H} 个 head 保留。`;expected.y=c.expected;
  }else if(id==='att_score_backward'){
   const A=args[0],G=args[1],[B,H,Tq,Tk]=shape(A);
   const dot=build([B,H,Tq,1],([b,h,t])=>slice(Tk).reduce((sum,s)=>sum+A[b][h][t][s]*G[b][h][t][s],0));
   const diff=build([B,H,Tq,Tk],([b,h,t,s])=>G[b][h][t][s]-dot[b][h][t][0]);
   const result=build([B,H,Tq,Tk],([b,h,t,s])=>A[b][h][t][s]*diff[b][h][t][s]);
   add('a','A · 前向概率',A,['batch b','head h','query t','key s'],'full');add('g','dA · 概率的上游梯度',G,['batch b','head h','query t','key s'],'full');
   add('dot','行内耦合项 sum(A*dA)',dot,['batch b','head h','query t','保留轴']);add('diff','dA 减去行内耦合项',diff,['batch b','head h','query t','key s']);add('y','dS · 逐位置分数梯度',result,['batch b','head h','query t','key s']);
   for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let t=0;t<Tq;t++){
    event(`汇总耦合项：b=${b}, h=${h}, query ${t}`,`这一行各 key 的 A 与 dA 逐项相乘并求和，得到一个共享的耦合项。`,['a','g','dot'],slice(Tk).map(s=>link('g',[b,h,t,s],'dot',[b,h,t,0],'replace',dot[b][h][t][0])),slice(Tk).map(s=>({panel:'a',coord:[b,h,t,s]})),String.raw`u_{${b},${h},${t},0}=\sum_s A_{${b},${h},${t},s}\,\mathrm dA_{${b},${h},${t},s}`);
    event(`扣除耦合项：b=${b}, h=${h}, query ${t}`,`从这一行每个 dA 减去相同的 u；保留 key 轴。`,['g','dot','diff'],slice(Tk).map(s=>link('g',[b,h,t,s],'diff',[b,h,t,s],'replace',diff[b][h][t][s])),[{panel:'dot',coord:[b,h,t,0]}],String.raw`v_{${b},${h},${t},s}=\mathrm dA_{${b},${h},${t},s}-u_{${b},${h},${t},0}`);
    event(`回到分数：b=${b}, h=${h}, query ${t}`,`每个位置再乘自己的前向概率 A，得到对应的 dS。`,['a','diff','y'],slice(Tk).map(s=>link('diff',[b,h,t,s],'y',[b,h,t,s],'replace',result[b][h][t][s])),slice(Tk).map(s=>({panel:'a',coord:[b,h,t,s]})),String.raw`\mathrm dS_{${b},${h},${t},s}=A_{${b},${h},${t},s}\,v_{${b},${h},${t},s}`);
   }
   note=`沿最后的 key 轴（长度 ${Tk}）耦合；保留全部 ${B} 个 batch、${H} 个 head 和 ${Tq} 个 query。`;expected.y=c.expected;
  }else if(id==='center_backward'){
   const G=args[0],dims=shape(G),prefix=dims.slice(0,-1),C=dims[dims.length-1],small=[...prefix,1];
   const axes=[...prefix.map((_,i)=>i===0?'batch b':i===1?'token t':'位置 '+i),'特征 c'];
   const means=build(small,co=>slice(C).reduce((sum,j)=>sum+at(G,[...co.slice(0,-1),j]),0)/C);
   const branch=build(dims,co=>-at(means,[...co.slice(0,-1),0]));
   add('g','G · 直接路径与均值路径的上游',G,axes,'full');add('mean','每个位置的 mean(G)',means,axes);add('branch','均值路径 · −mean(G)',branch,axes);add('y','dX · 两条路径合流',c.expected,axes);
   for(const pos of coords(prefix)){
    const label=pos.map((v,i)=>`${axes[i]}=${v}`).join('，'),one=[...pos,0];
    event(`汇总均值路径：${label}`,`这一位置的 ${C} 个输出都减去同一个均值；它们的上游贡献先汇总，再按均值导数除以 ${C}。`,['g','mean'],slice(C).map(j=>link('g',[...pos,j],'mean',one,'replace',at(means,one))),[],String.raw`\operatorname{mean}(G)=\frac1{${C}}\sum_i G_i`);
    event(`分回输入：${label}`,`均值路径带负号，将 −mean(G) 送到该位置的每个输入特征。`,['mean','branch'],slice(C).map(j=>link('mean',one,'branch',[...pos,j],'replace',at(branch,[...pos,j]))),[],String.raw`\mathrm dX_{\text{均值},j}=-\operatorname{mean}(G)`);
    event(`合流：${label}`,`每个输入特征还接收同位置的直接梯度 G；两条路径相加得到 dX。`,['g','branch','y'],slice(C).map(j=>link('g',[...pos,j],'y',[...pos,j],'replace',at(G,[...pos,j])+at(branch,[...pos,j]))),slice(C).map(j=>({panel:'branch',coord:[...pos,j]})),String.raw`\mathrm dX_j=G_j-\operatorname{mean}(G)`);
   }
   note=`沿最后特征轴 C=${C} 计算；保留所有前置 batch/token 位置。每条路径的来源和合流分别展示。`;expected.y=c.expected;
  }else if(id==='att_scores'){
   const Q=args[0],K=args[1],[B,H,Tq,d]=shape(Q),Tk=shape(K)[2],scores=build([B,H,Tq,Tk],([b,h,t,s])=>slice(d).reduce((sum,i)=>sum+Q[b][h][t][i]*K[b][h][s][i],0)/Math.sqrt(d));
   add('q','Q · 查询向量',Q,['batch b','head h','query t','特征 i'],'full');add('k','K · 匹配向量',K,['batch b','head h','key s','特征 i'],'full');
   add('y','S · 缩放后的匹配分数',c.expected,['batch b','head h','query t','key s']);
   for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let t=0;t<Tq;t++)for(let s=0;s<Tk;s++)event(`b=${b}, h=${h}：query ${t} 对 key ${s}`,`当前 Q 行与 K 行的 ${d} 对特征逐项相乘并求和，再除以 √${d}；得到这两个位置的一格分数。`,['q','k','y'],slice(d).map(i=>link('q',[b,h,t,i],'y',[b,h,t,s],'replace',scores[b][h][t][s])),slice(d).map(i=>({panel:'k',coord:[b,h,s,i]})),String.raw`S_{${b},${h},${t},${s}}=\frac{1}{\sqrt{${d}}}\sum_i Q_{${b},${h},${t},i}K_{${b},${h},${s},i}`);
   note=`Q/K 宽度 d=${d}；Tq=${Tq}，Tk=${Tk}。所有 ${B} 个 batch 和 ${H} 个 head 独立计算。`;expected.y=c.expected;
  }else if(id==='att_probs'){
   const S=args[0],[B,H,T,Tk]=shape(S),masked=build([B,H,T,Tk],([b,h,t,s])=>s>t?-Infinity:S[b][h][t][s]);
   const probs=masked.map(batch=>batch.map(head=>head.map(row=>{const max=Math.max(...row),exp=row.map(v=>Math.exp(v-max)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(v=>v/sum);}))); 
   add('x','S · 原始分数',S,['batch b','head h','query t','key s'],'full');add('masked','掩码后分数',masked,['batch b','head h','query t','key s']);add('y','A · 逐行概率',c.expected,['batch b','head h','query t','key s']);
   for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let t=0;t<T;t++){
    event(`掩码：b=${b}, h=${h}, query ${t}`,`本行 key 编号大于 ${t} 的未来位置变成 −∞；其余分数保持原值。`,['x','masked'],slice(Tk).map(s=>link('x',[b,h,t,s],'masked',[b,h,t,s],'replace',masked[b][h][t][s])),[],String.raw`\widetilde S_{t,s}=\begin{cases}S_{t,s}&s\le t\\-\infty&s>t\end{cases}`);
    event(`归一化：b=${b}, h=${h}, query ${t}`,`沿这一行 key 轴进行稳定 Softmax；未来位置概率为 0，允许位置的概率加起来为 1。`,['masked','y'],slice(Tk).map(s=>link('masked',[b,h,t,s],'y',[b,h,t,s],'replace',probs[b][h][t][s])),[],String.raw`A_{t,:}=\operatorname{softmax}(\widetilde S_{t,:})`);
   }
   note=`因果自注意力：T=${T}；保留全部 ${B} 个 batch、${H} 个 head。先掩码，再归一化。`;expected.y=c.expected;
  }else if(id==='att_weighted'){
   const A=args[0],V=args[1],[B,H,Tq,Tk]=shape(A),dv=shape(V)[3],out=build([B,H,Tq,dv],([b,h,t,j])=>slice(Tk).reduce((sum,s)=>sum+A[b][h][t][s]*V[b][h][s][j],0));
   add('a','A · 每个 query 读取各 key 的概率',A,['batch b','head h','query t','key s'],'full');add('v','V · 各 key 的内容',V,['batch b','head h','key s','特征 j'],'full');add('y','Y · 每个 query 读到的内容',c.expected,['batch b','head h','query t','特征 j']);
   for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let t=0;t<Tq;t++)for(let j=0;j<dv;j++)event(`b=${b}, h=${h}：query ${t} 的特征 ${j}`,`用本 query 行的 ${Tk} 个概率，分别乘对应 key 的 V 特征 ${j}，再沿 key 位置求和。`,['a','v','y'],slice(Tk).map(s=>link('a',[b,h,t,s],'y',[b,h,t,j],'replace',out[b][h][t][j])),slice(Tk).map(s=>({panel:'v',coord:[b,h,s,j]})),String.raw`Y_{${b},${h},${t},${j}}=\sum_s A_{${b},${h},${t},s}V_{${b},${h},s,${j}}`);
   note=`所有 ${B} 个 batch、${H} 个 head 独立计算；key 长度 ${Tk}，value 特征宽度 ${dv}。`;expected.y=c.expected;
  }else if(id==='t_generate'){
   const input=c.input,prompt=input.ids,output=c.expected.ids,[B,T0]=shape(prompt),steps=input.steps,context=input.init[5],width=Math.min(context,T0+Math.max(0,steps-1));
   const window=build([B,width],()=>0),samples=output.map(row=>row.slice(T0));
   add('prompt','完整提示',prompt,['批次','提示位置'],'full');
   add('window','末尾窗口',window,['批次','窗口位置']);
   if(generationScores?.length){
    add('scores','最后位置的分数',generationScores[0],['批次','候选编号']);
    add('probabilities','筛选后的抽样概率',generationScores[0],['批次','候选编号']);
   }
   if(steps)add('sample','逐轮采样记录',samples,['批次','生成轮次']);
   add('result','完整序列',output,['批次','完整位置']);
   const visible=panels.map(panel=>panel.id);
   const currentRound=visible.filter(panel=>['window','scores','probabilities'].includes(panel));
   event('保留完整提示',`将每条提示的 ${T0} 个编号写入完整序列。之后只截取送入模型的窗口，完整序列继续保留。`,visible,coords([B,T0]).map(([b,t])=>link('prompt',[b,t],'result',[b,t])));
   for(let step=0;step<steps;step++){
    const len=T0+step,start=Math.max(0,len-context),size=len-start;
    event(`第 ${step} 轮 · 截取窗口`,`当前完整长度 ${len}，从位置 ${start} 起取末尾 ${size} 个编号送入模型。`,visible,slice(B).flatMap(b=>slice(size).map(t=>link('result',[b,start+t],'window',[b,t]))),[],String.raw`\mathrm{window}=\mathrm{result}[:,\,-\mathrm{context}:]`,currentRound);
    const logits=generationScores?.[step];
    const candidates=[],selected=[],draws=[],scoring=[];
    if(logits){
     logits.forEach((row,b)=>{
      const ranked=slice(row.length).sort((a,c)=>row[a]-row[c]||a-c);
      const kept=ranked.slice(-(input.top_k||row.length)),chosen=samples[b][step];
      const max=row[kept.at(-1)],weight=j=>Math.exp((row[j]-max)/input.temperature);
      const total=kept.reduce((sum,j)=>sum+weight(j),0);
      const percent=j=>kept.includes(j)?`${(weight(j)/total*100).toFixed(1)}%`:'0%';
      candidates.push(...kept.map(j=>({panel:'scores',coord:[b,j]})));
      selected.push({panel:'scores',coord:[b,chosen]},{panel:'probabilities',coord:[b,chosen]});
      row.forEach((value,j)=>{
       scoring.push(link(null,[],'scores',[b,j],'display',value.toFixed(3)));
       scoring.push(link(null,[],'probabilities',[b,j],'display',percent(j)));
      });
      draws.push(`第 ${b} 行抽中 ${chosen}（概率 ${percent(chosen)}）`);
     });
    }
    const rule=input.top_k?`每行保留分数最高的 ${input.top_k} 个候选`:'每行保留全部候选';
    event(`第 ${step} 轮 · 计算抽样概率`,`取模型最后位置的分数。${rule}。${logits?'蓝色标出保留项，其余候选的概率为零。':''}温度 ${input.temperature} 下，保留项通过 Softmax 得到抽样概率。`,visible,scoring,candidates,String.raw`\mathrm{logits}=\mathrm{model}(\mathrm{window})[:,\,-1,:]`);
    const outputs=slice(B).flatMap(b=>[
     link(null,[],'sample',[b,step],'replace',samples[b][step]),
     link(null,[],'result',[b,T0+step],'replace',samples[b][step])
    ]);
    const drawNote=draws.length?`${draws.join('，')}。`:'';
    event(`第 ${step} 轮 · 抽样并追加`,`${drawNote}把抽中的编号追加到各自的完整序列。下一轮使用追加后的序列。`,visible,outputs,selected);
   }
   note=`沿公开样例逐轮走一遍。每轮截取窗口、筛选高分候选，再抽样并追加。高分候选有更高的抽中概率。上下文上限 ${context}，共生成 ${steps} 轮。`;
   expected.result=output;if(steps)expected.sample=samples;
  }else if(id==='t_data'){
   const text=Array.from(c.input.text),starts=c.input.starts,T=c.input.T,ids=c.expected.ids,vocab=Array.from(c.expected.vocab),B=starts.length;
   add('text','text · 原始字符',text,['字符位置'],'full');add('vocab','vocab · 编号对应字符',vocab,['类别编号'],'full');
   add('ids','ids · 编码后的完整文本',ids,['字符位置']);add('starts','starts · 各样本起点',starts,['batch b'],'full');
   add('x','X · 每个输入窗口',c.expected.X,['batch b','token t']);add('y','targets · 每个位置的下一个字符',c.expected.targets,['batch b','token t']);
   for(let pos=0;pos<text.length;pos++)event(`编码第 ${pos} 个字符`,`字符“${fmt(text[pos])}”在词表中的编号是 ${ids[pos]}。词表去重排序后固定，重复字符使用同一编号。`,['text','vocab','ids'],[link('text',[pos],'ids',[pos],'replace',ids[pos])],[{panel:'vocab',coord:[ids[pos]]}]);
   for(let b=0;b<B;b++){
    const s=starts[b];
    event(`截取输入：batch ${b}`,`起点 starts[${b}]=${s}，从 ids[${s}] 起取 T=${T} 个编号；其他 batch 使用各自起点。`,['starts','ids','x'],slice(T).map(t=>link('ids',[s+t],'x',[b,t])),[{panel:'starts',coord:[b]}],String.raw`X_{${b},t}=\mathrm{ids}_{${s}+t}`);
    event(`截取目标：batch ${b}`,`从同一起点向后移动一格，targets[${b},t] 是 X[${b},t] 的下一个字符。`,['starts','ids','x','y'],slice(T).map(t=>link('ids',[s+t+1],'y',[b,t])),slice(T).map(t=>({panel:'x',coord:[b,t]})),String.raw`\mathrm{targets}_{${b},t}=\mathrm{ids}_{${s}+t+1}`);
   }
   note=`公开样例：${B} 个起点，窗口长度 T=${T}；字符先编码，再分别截取输入及右移一格的目标。␠ 表示空格，↵ 表示换行。`;
   expected.ids=ids;expected.x=c.expected.X;expected.y=c.expected.targets;
  }else if(id==='t_loss'){
   const logits=c.input.X,targets=c.input.targets,[B,T,V]=shape(logits),N=B*T;
   const probs=logits.map(batch=>batch.map(row=>{const max=Math.max(...row),exp=row.map(v=>Math.exp(v-max)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(v=>v/sum);}));
   const losses=logits.map((batch,b)=>batch.map((row,t)=>{const max=Math.max(...row),logsum=Math.log(row.reduce((sum,v)=>sum+Math.exp(v-max),0));return logsum-(row[targets[b][t]]-max);}));
   const dx=probs.map((batch,b)=>batch.map((row,t)=>row.map((p,j)=>(p-(j===targets[b][t]?1:0))/N)));
   add('x','logits · 每个位置的类别分数',logits,['batch b','token t','类别 j'],'full');
   add('targets','targets · 正确类别编号',targets,['batch b','token t'],'full');
   add('p','p · 每个位置的类别概率',probs,['batch b','token t','类别 j']);
   add('losses','每个位置的损失 −log p[target]',losses,['batch b','token t']);
   add('loss','平均损失 L · 累计每项/N',[[0]],['汇总','结果'],'full');
   add('dx','dLogits · 对全部类别回传',dx,['batch b','token t','类别 j']);
   for(let b=0;b<B;b++)for(let t=0;t<T;t++){
    const target=targets[b][t],pos=`b=${b}, t=${t}`;
    event(`归一化 ${pos}`,`同一位置的 ${V} 个类别分数经稳定 Softmax，得到这一行概率。`,['x','p'],slice(V).map(j=>link('x',[b,t,j],'p',[b,t,j],'replace',probs[b][t][j])),[],String.raw`p_{${b},${t},j}=\operatorname{softmax}(x_{${b},${t},:})_j`);
    event(`选择目标类别：${pos}`,`targets[${b},${t}]=${target}，只选择该位置第 ${target} 类的概率，取负对数得到此位置损失。`,['targets','p','losses'],[link('p',[b,t,target],'losses',[b,t],'replace',losses[b][t])],[{panel:'targets',coord:[b,t]}],String.raw`\ell_{${b},${t}}=-\log p_{${b},${t},${target}}`);
   }
   for(let b=0;b<B;b++)for(let t=0;t<T;t++)event(`汇总损失：b=${b}, t=${t}`,`将此位置损失除以 ${N}，加入平均值；全部 ${B} 个 batch × ${T} 个 token 都参与。`,['losses','loss'],[link('losses',[b,t],'loss',[0,0],'add',losses[b][t]/N)],[],String.raw`L=\frac{1}{${N}}\sum_{b,t}\ell_{b,t}`);
   for(let b=0;b<B;b++)for(let t=0;t<T;t++){
    const target=targets[b][t];
    event(`反向：b=${b}, t=${t}`,`从该位置每个类别的概率出发；目标类别 ${target} 减 1，再将整行除以 ${N}。本例标量上游 g=1。`,['targets','p','dx'],slice(V).map(j=>link('p',[b,t,j],'dx',[b,t,j],'replace',dx[b][t][j])),[{panel:'targets',coord:[b,t]}],String.raw`\frac{\partial L}{\partial x_{${b},${t},j}}=\frac{p_{${b},${t},j}-\mathbf 1[j=${target}]}{${N}}`);
   }
   note=`同一公开样例：B=${B}，T=${T}，V=${V}，平均位置数 N=B×T=${N}；展示全部位置。反向使用默认上游 g=1。`;
   expected.loss=[[c.expected.loss]];expected.dx=c.expected.dLogits;
  }else if(id==='mean_backward'){
   const G=args[0],C=Number(args[1]),dims=shape(G),prefix=dims.slice(0,-1);
   const axes=[...prefix.map((_,i)=>i===0?'batch b':i===1?'token t':'位置 '+i),'特征 c'];
   const divided=build(dims,co=>at(G,co)/C);
   add('g','G · 均值结果的上游梯度',G,axes,'full');
   add('part','每个特征分得的梯度 · G/C',divided,axes);
   add('y','dX · 每个输入特征的梯度',c.expected,axes);
   for(const position of coords(prefix)){
    const source=[...position,0],label=position.map((v,i)=>`${axes[i]}=${v}`).join('，');
    event(`均分：${label}`,`G 是损失对均值 μ 的梯度；μ 对每个输入特征的导数是 1/${C}。两者相乘，得到 G/${C}。`,['g','part'],[link('g',source,'part',source,'replace',at(divided,source))],[],String.raw`\frac{\partial L}{\partial X_c}=\underbrace{\frac{\partial L}{\partial\mu}}_{G}\underbrace{\frac{\partial\mu}{\partial X_c}}_{1/C}=\frac{G}{C}`);
    event(`分回特征：${label}`,`将同一个 G/C 对齐到此位置的 ${C} 个特征；其他 batch 和 token 保留各自的梯度。`,['part','y'],slice(C).map(j=>link('part',source,'y',[...position,j])),[],String.raw`\mathrm dX_{b,t,c}=\frac{G_{b,t,0}}{C}`);
   }
   note=`C=${C}；G 的末轴长度为 1，输出末轴长度为 C。所有前置位置独立计算。`;expected.y=c.expected;
  }else if(id==='axes'||id==='reduce'){
   const N=X.length,C=X[0].length,mean=id==='reduce';
   add('x','X · 每行 C 个来源',X,['样本 n','特征 c'],'full');
   const reduced=X.map(row=>row.reduce((a,b)=>a+b,0)/(mean?C:1));
   add('r',mean?'μ · 保留特征轴长 1':'y · 特征轴被归约',mean?reduced.map(v=>[v]):reduced,mean?['样本 n','特征轴（长 1）']:['样本 n']);
   if(mean)add('y','Y · 每行减去自己的均值',c.expected,['样本 n','特征 c']);
   for(let n=0;n<N;n++)event(`归约第 ${n} 行`,mean?`本行 ${C} 个特征共同产生一个均值 μ[${n},0]。其他行不参与。`:`第 ${n} 行全部特征汇入 y[${n}]；特征轴消失。`,['x','r'],slice(C).map(j=>link('x',[n,j],'r',mean?[n,0]:[n],'replace',reduced[n])),[],mean?String.raw`\mu_{${n},0}=\frac1{${C}}\sum_{c=0}^{${C-1}}X_{${n},c}`:String.raw`y_{${n}}=\sum_{c=0}^{${C-1}}X_{${n},c}`);
   if(mean)for(let n=0;n<N;n++)event(`均值对齐第 ${n} 行，再相减`,`同一个 μ[${n},0] 沿本行的特征轴复用；每个目标仍保留原 X[${n},c]，计算 X−μ。`,['x','r','y'],slice(C).map(j=>link('r',[n,0],'y',[n,j],'replace',c.expected[n][j])),slice(C).map(j=>({panel:'x',coord:[n,j]})),String.raw`Y_{${n},c}=X_{${n},c}-\mu_{${n},0}`);
   note=mean?'固定条件：axis=-1，keepdims=True；(N,C) → μ(N,1) → Y(N,C)。本节只前向。':'固定条件：axis=1，keepdims=False；(N,C) → y(N,)。';
   expected[mean?'y':'r']=c.expected;
  }else if(id==='broadcast'){
   const b=args[1],N=X.length,C=X[0].length;
   add('x','X · 不同样本位置',X,['样本 n','特征 c'],'full');add('b','b · 只有这一组偏置',b,['特征 c'],'full');add('y','Y · 末轴对齐后相加',c.expected,['样本 n','特征 c']);
   event('先对齐末尾特征轴','b[c] 对齐 X 的第 c 列。样本轴上没有新的偏置参数。',['b','x','y'],[],[],String.raw`(N,C)+(C,)\longrightarrow(N,C)`);
   for(let j=0;j<C;j++)event(`复用偏置 b[${j}]`,`每条样本的第 ${j} 个特征都加同一个 b[${j}]。移动的标记代表逻辑复用，不是创建 N 个独立参数。`,['b','x','y'],slice(N).map(n=>link('b',[j],'y',[n,j],'replace',c.expected[n][j])),slice(N).map(n=>({panel:'x',coord:[n,j]})),String.raw`Y_{n,${j}}=X_{n,${j}}+b_{${j}}`);
   note='固定条件：从右向左对齐，b 的长度等于 X 的特征宽度。偏置反向在完整 Linear 的 db 演示中汇总全部使用位置。';expected.y=c.expected;
  }else if(id==='reshape'||id==='transpose'){
   const [B,T,C]=shape(X),trans=id==='transpose';
   add('x','X · 原坐标 (b,t,c)',X,['batch b','token t','特征 c'],'full');add('y',trans?'Y · 新坐标 (b,c,t)':'X₂ · 行 n=b×T+t',c.expected,trans?['batch b','特征 c','token t']:['位置 n','特征 c']);
   for(let b=0;b<B;b++)for(let t=0;t<T;t++)event(`跟踪 b=${b}, t=${t} 的全部特征`,trans?`原 token 行中的每个 c，移到同一 batch 的特征行 c、token 列 ${t}。`:`本 token 的全部特征一起对应到行 n=${b}×${T}+${t}=${b*T+t}；不同 token 不混合。`,['x','y'],slice(C).map(j=>link('x',[b,t,j],'y',trans?[b,j,t]:[b*T+t,j])),[],trans?String.raw`X_{${b},${t},c}=Y_{${b},c,${t}}`:String.raw`X_{${b},${t},c}=(X_2)_{${b*T+t},c}`);
   note=trans?'固定排列：(0,2,1)。图中移动表示坐标映射，不表示 NumPy 必须复制内存。':'目标形状：(B*T,C)，按 C 顺序。图中移动表示坐标映射；reshape 可能共享内存。';expected.y=c.expected;
  }else if(id==='heads'||id==='att_merge'){
   const merge=id==='att_merge',dims=shape(X),B=dims[0],H=merge?dims[1]:Number(args[1]),T=merge?dims[2]:dims[1],d=merge?dims[3]:dims[2]/H,C=H*d;
   const middle=build([B,T,H,d],([b,t,h,j])=>merge?X[b][h][t][j]:X[b][t][h*d+j]);
   add('x',merge?'各 head 的输出 X':'完整 token 特征 X',X,merge?['batch b','head h','token t','特征 j']:['batch b','token t','特征 c'],'full');
   add('m','中间布局 · (B,T,H,d)',middle,['batch b','token t','head h','特征 j']);add('y',merge?'合头 Y · (B,T,C)':'分头 Y · (B,H,T,d)',c.expected,merge?['batch b','token t','特征 c']:['batch b','head h','token t','特征 j']);
   for(let b=0;b<B;b++)for(let t=0;t<T;t++)event(`${merge?'逆向换轴':'拆特征轴'}：b=${b}, t=${t}`,merge?'先让同一 token 的全部 head 放到一起；此时仍保留 H、d 两轴。':`把完整 C=${C} 个特征分成 H=${H} 组，每组 d=${d} 个。先得到 B,T,H,d，不提前交换轴。`,['x','m'],slice(H).flatMap(h=>slice(d).map(j=>link('x',merge?[b,h,t,j]:[b,t,h*d+j],'m',[b,t,h,j]))),[],merge?String.raw`X_{${b},h,${t},j}=M_{${b},${t},h,j}`:String.raw`X_{${b},${t},hd+j}=M_{${b},${t},h,j}`);
   if(merge){for(let b=0;b<B;b++)for(let t=0;t<T;t++)event(`拼接特征：b=${b}, t=${t}`,'已放到同一 token 下面的 head 按顺序拼接，列号 c=h*d+j。',['m','y'],slice(H).flatMap(h=>slice(d).map(j=>link('m',[b,t,h,j],'y',[b,t,h*d+j]))),[],String.raw`Y_{${b},${t},hd+j}=M_{${b},${t},h,j}`);}
   else{for(let b=0;b<B;b++)for(let h=0;h<H;h++)event(`换轴：b=${b}, h=${h}`,`把这个 head 在全部 ${T} 个 token 的特征放入同一张输出切片；token 顺序不变。`,['m','y'],slice(T).flatMap(t=>slice(d).map(j=>link('m',[b,t,h,j],'y',[b,h,t,j]))),[],String.raw`M_{${b},t,${h},j}=Y_{${b},${h},t,j}`);}
   note=`H=${H}，d=C/H=${d}，全部 B=${B}、T=${T} 保留。${merge?'先逆向换轴，再合特征。':'先拆特征，再换轴。'}图示表示坐标映射，不断言复制内存。`;expected.y=c.expected;
  }else if(id==='layer_embedding'){
   const ids=c.X,E=c.parameters.weight,G=c.G,[B,T]=shape(ids),[V,C]=shape(E),Y=build([B,T,C],([b,t,j])=>E[ids[b][t]][j]),dE=E.map(row=>row.map(()=>0));
   add('ids','ids · 编号决定来源行',ids,['batch b','token t'],'full');add('e','E / weight · 参数数据',E,['词表行索引 v','特征 c'],'full');add('y','Y · 查表结果',Y,['batch b','token t','特征 c']);
   add('g','G · 各输出位置的上游',G,['batch b','token t','特征 c'],'full');add('de','dE / weight.grad · 梯度缓冲',dE,['词表行索引 v','特征 c'],'full');
   for(let b=0;b<B;b++)for(let t=0;t<T;t++){const v=ids[b][t];event(`查表：ids[${b},${t}]=${v}`,`编号指向 E 中行索引为 ${v} 的一行，再把这行特征放到输出位置 (${b},${t})。相同编号始终指向同一行参数。`,['ids','e','y'],slice(C).map(j=>link('e',[v,j],'y',[b,t,j])),[{panel:'ids',coord:[b,t]}],String.raw`Y_{${b},${t},c}=E_{${v},c}`);}
   for(let round=0;round<(c.repeat?2:1);round++)for(let b=0;b<B;b++)for(let t=0;t<T;t++){const v=ids[b][t];for(let j=0;j<C;j++)dE[v][j]+=G[b][t][j];event(`反向${round?'第 2 轮':''}：位置 (${b},${t}) → 行索引 ${v}`,`本位置的 G 回到词表行索引 ${v} 的梯度。重复编号的贡献继续相加，不覆盖，也不改变 E.data。`,['ids','g','de'],slice(C).map(j=>link('g',[b,t,j],'de',[v,j],'add')),[{panel:'ids',coord:[b,t]}],String.raw`(\mathrm dE)_{${v},c}\mathrel{+}=G_{${b},${t},c}`);}
   // dE 的期望单独保存；面板初始值始终为零。
   panels.find(p=>p.id==='de').values=E.map(row=>row.map(()=>0));expected.y=Y;expected.de=copy(dE);
   note=`V=${V}，C=${C}；完整 ids(${B},${T})。前向取参数行，反向把 G 送回对应行的 grad；整数 ids 不求导。`;
  }else if(id==='scatter'){
   const ids=args[0],G=args[1],table=args[2],V=table.length,C=table[0].length;
   add('ids','ids · 每条贡献的目标行',ids,['贡献编号 n'],'full');add('g','G · 待累加贡献',G,['贡献编号 n','特征 c'],'full');add('table','table · 初始值',table,['词表行 v','特征 c'],'full');add('y','result · 在初始值上累加',table,['词表行 v','特征 c']);
   event('先复制初始 table','result 从 table 的现有数值开始，不能假定初始为全零。',['table','y'],coords([V,C]).map(c=>link('table',c,'y',c)));
   for(let n=0;n<ids.length;n++)event(`第 ${n} 条贡献 → 行 ${ids[n]}`,`将 G[${n},:] 加到 result[${ids[n]},:]。之前的初始值和同编号贡献都保留。`,['ids','g','y'],slice(C).map(j=>link('g',[n,j],'y',[ids[n],j],'add')),[{panel:'ids',coord:[n]}],String.raw`\mathrm{result}_{${ids[n]},c}\mathrel{+}=G_{${n},c}`);
   note='返回独立副本；重复索引逐次累加，原 table、G、ids 保持不变。';expected.y=c.expected;
  }
  return {panels,events,note,expected};
 }
 function mount(id,c,generationScores){
  const host=document.getElementById('animation'),plan=makePlan(id,c,generationScores);if(!host||!plan.events.length)return;
  const box=document.createElement('div');box.className='process-animation';if(id==='t_generate')box.classList.add('generation-process');host.append(box);
  let step=0,playing=false,busy=false,disposed=false;const moving=new Set(),animations=new Set(),values=new Map();
  const cellKey=(p,co)=>p+':'+key(co);
  box.innerHTML='<h2>看来源怎样到达输出</h2><p class="small">'+esc(plan.note)+'</p><div class="controls process-controls"><button data-next>下一步 →</button><button data-play>自动演示</button><button data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><div data-formula></div></div><div class="process-panels"></div>';
  const find=(p,co)=>box.querySelector(`[data-cell="${cellKey(p,co)}"]`);
  function panelHTML(p){
   const S=p.shape,n=S.length;
   const grid=(prefix,rows,columns)=>`<div class="process-slice"><strong>${prefix.map((v,i)=>p.axes[i]+'='+v).join(' · ')}</strong><div class="process-matrix" style="grid-template-columns:auto repeat(${columns},minmax(var(--process-cell-width,34px),1fr))"><span class="process-coordinate">${esc(p.axes[n-1]||'坐标')} →</span>${slice(columns).map(j=>`<span class="process-coordinate">${j}</span>`).join('')}${slice(rows).map(i=>`<span class="process-coordinate">${n===1?'':' '+esc(p.axes[n-2])+'='+i}</span>`+slice(columns).map(j=>{const co=n===1?[j]:[...prefix,i,j],k=cellKey(p.id,co),value=at(p.values,co);if(p.initial==='full')values.set(k,value);return `<span class="process-cell ${p.initial==='full'?'':'process-empty'}" data-cell="${k}" title="${esc(p.title)}[${co.join(',')}]">${p.initial==='full'?esc(fmt(value)):'·'}</span>`;}).join('')).join('')}</div></div>`;
   const slices=n<=2?grid([],n===1?1:S[0],S[n-1]):coords(S.slice(0,-2)).map(prefix=>grid(prefix,S[n-2],S[n-1])).join('');
   return `<section class="process-panel" data-panel="${p.id}"><h3>${esc(p.title)}</h3><p class="small">形状 (${S.join(', ')}) · 轴：${esc(p.axes.join(' / '))}</p><div class="process-slices">${slices}</div></section>`;
  }
  const next=box.querySelector('[data-next]'),play=box.querySelector('[data-play]'),resetButton=box.querySelector('[data-reset]');
  function buttons(){next.disabled=busy||step===plan.events.length;resetButton.disabled=busy;play.disabled=busy&&!playing;next.textContent=step===plan.events.length?'✓ 演示完成':'下一步 →';play.textContent=playing?'暂停':'自动演示';box.querySelector('[data-count]').textContent=`${step}/${plan.events.length}`;}
  function reset(){playing=false;values.clear();step=0;box.querySelector('.process-panels').innerHTML=plan.panels.map(panelHTML).join('');const visible=plan.events[0].visible;box.querySelectorAll('[data-panel]').forEach(el=>el.hidden=!visible.includes(el.dataset.panel));box.querySelector('[data-title]').textContent='从完整输入开始';box.querySelector('[data-note]').textContent='点击下一步，观察高亮来源移动到目标位置。源数组保留不变。';box.querySelector('[data-formula]').innerHTML='';buttons();}
  function set(target,value){const el=find(target.to,target.target),k=cellKey(target.to,target.target);values.set(k,value);el.textContent=fmt(value);el.classList.remove('process-empty');if(target.mode!=='display')el.classList.add('process-destination');}
  async function move(m){const source=find(m.from,m.source),target=find(m.to,m.target);source.classList.add('process-source');target.classList.add('process-destination');const s=source.getBoundingClientRect(),t=target.getBoundingClientRect();
   const ghost=source.cloneNode(true);ghost.removeAttribute('data-cell');ghost.className='process-moving';Object.assign(ghost.style,{left:s.left+'px',top:s.top+'px',width:s.width+'px',height:s.height+'px'});document.body.append(ghost);moving.add(ghost);
   const duration=root.matchMedia('(prefers-reduced-motion: reduce)').matches?0:650;
   try{const anim=ghost.animate([{transform:'translate(0,0)',opacity:1},{transform:`translate(${t.left-s.left}px,${t.top-s.top}px)`,opacity:.75}],{duration,easing:'cubic-bezier(.4,0,.2,1)',fill:'forwards'});animations.add(anim);await anim.finished;animations.delete(anim);}catch{}
   moving.delete(ghost);ghost.remove();
  }
  async function advance(){if(disposed||busy||step===plan.events.length)return;busy=true;const event=plan.events[step];buttons();
   box.querySelectorAll('.process-source,.process-destination').forEach(el=>el.classList.remove('process-source','process-destination'));
   for(const panel of event.clear)box.querySelectorAll(`[data-panel="${panel}"] [data-cell]`).forEach(el=>{
    values.delete(el.dataset.cell);el.textContent='·';el.classList.add('process-empty');
   });
   box.querySelectorAll('[data-panel]').forEach(el=>el.hidden=!event.visible.includes(el.dataset.panel));box.querySelector('[data-title]').textContent=event.title;box.querySelector('[data-note]').textContent=event.note;
   const formula=box.querySelector('[data-formula]');if(event.tex&&root.CourseMath)root.CourseMath.set(formula,event.tex);else formula.textContent='';
   event.extra.forEach(v=>find(v.panel,v.coord).classList.add('process-source'));
   await Promise.all(event.moves.filter(m=>m.from!==null).map(move));
   if(!disposed){for(const m of event.moves){const old=values.get(cellKey(m.to,m.target))||0,value=m.value??values.get(cellKey(m.from,m.source));set(m,m.mode==='add'?old+value:value);}step++;busy=false;buttons();}
  }
  next.onclick=advance;resetButton.onclick=reset;
  play.onclick=async()=>{if(playing){playing=false;buttons();return;}if(step===plan.events.length)reset();playing=true;buttons();while(playing&&!disposed&&step<plan.events.length){await advance();if(!playing||disposed)break;await new Promise(resolve=>setTimeout(resolve,450));}playing=false;if(!disposed)buttons();};
  root.understandingCleanup=()=>{disposed=true;playing=false;animations.forEach(a=>a.cancel());moving.forEach(el=>el.remove());};reset();
 }
 root.ProcessAnimations={supports:id=>supported.has(id),mount,makePlan};
 if(typeof module!=='undefined')module.exports={makePlan,shape,coords};
})(typeof window==='undefined'?globalThis:window);
