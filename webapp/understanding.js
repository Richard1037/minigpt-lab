/* 可解释的计算过程。数值由当前公开样例计算，不取第一个 batch 代替完整张量。 */
(function(root){
'use strict';
const zeros=(r,c)=>Array.from({length:r},()=>Array(c).fill(0));
const mm=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0)));
const tr=a=>a[0].map((_,j)=>a.map(row=>row[j]));
function linearBackward(X,W,G){const B=X.length,T=X[0].length,C=W.length,D=W[0].length;
 const dX=X.map(batch=>batch.map(()=>Array(C).fill(0))),dW=zeros(C,D),db=Array(D).fill(0);
 for(let b=0;b<B;b++)for(let t=0;t<T;t++)for(let j=0;j<D;j++){db[j]+=G[b][t][j];for(let i=0;i<C;i++){dX[b][t][i]+=G[b][t][j]*W[i][j];dW[i][j]+=X[b][t][i]*G[b][t][j];}}
 return {dX,dW,db};}
function attentionBackward(Q,K,V,G,causal=true){const S=[],A=[],Y=[],dQ=[],dK=[],dV=[],dS=[];
 for(let b=0;b<Q.length;b++){for(const list of [S,A,Y,dQ,dK,dV,dS])list.push([]);
 for(let h=0;h<Q[b].length;h++){const q=Q[b][h],k=K[b][h],v=V[b][h],g=G[b][h],scale=Math.sqrt(q[0].length);
 const scores=mm(q,tr(k)).map(row=>row.map(x=>x/scale));
 const a=scores.map((row,i)=>{const z=row.map((x,j)=>causal&&j>i?-Infinity:x),m=Math.max(...z),e=z.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);});
 const da=mm(g,tr(v));const ds=da.map((row,i)=>{const dot=row.reduce((s,x,j)=>s+x*a[i][j],0);return row.map((x,j)=>a[i][j]*(x-dot));});
 S[b].push(scores);A[b].push(a);Y[b].push(mm(a,v));dV[b].push(mm(tr(a),g));dS[b].push(ds);
 dQ[b].push(mm(ds,k).map(row=>row.map(x=>x/scale)));dK[b].push(mm(tr(ds),q).map(row=>row.map(x=>x/scale)));
 }}return {S,A,Y,dQ,dK,dV,dS};}
const fmt=x=>Math.abs(x)<1e-10?'0':Number(x.toFixed(5)).toString();
const selector=(id,label,n)=>`<label>${label}<select id="${id}">${Array.from({length:n},(_,i)=>`<option value="${i}">${i}</option>`).join('')}</select></label>`;
const mathSet=(el,tex)=>root.CourseMath?root.CourseMath.set(el,tex):(el.textContent=tex);
const mathBlock=(tex,inline=false)=>root.CourseMath?root.CourseMath.block(tex,inline):esc(tex);
function install(meta,id,c){
 if(root.understandingCleanup)root.understandingCleanup();root.understandingCleanup=null;
 const host=document.getElementById('animation');if(!host)return;
 if(['linear','layer_linear','rebuild_linear'].includes(id))linear(host,c,id);
 if(id==='t_attention')attention(host,c);
 if(id==='t_mha')multiheadFlow(host,c);
 if(['t_gpt','rebuild_gpt'].includes(id))sharedEmbedding(host,c);
 if(id==='t_optim')optimizerFlow(host,c);
 if(['random','random_sequence'].includes(id))randomFlow(host,c,id);
 if(['t_train','t_checkpoint','t_project'].includes(id))workflowFlow(host,c,id);
 if(meta.residual_trace||c.residual_trace)residual(host,c.residual_trace||meta.residual_trace);
}
function workflowFlow(host,c,id){
 const input=c.input||{},isInfra=id==='infrastructure',ids=isInfra?null:input.ids,targets=isInfra?null:(input.targets||(id==='t_project'?ids.map(row=>row.map(v=>(v+1)%input.init[0])):null));
 const specs={
  t_train:{title:'固定批次的训练循环',summary:`公开样例：${input.steps} 步，学习率 ${input.lr}。每步记录更新前损失；优化器状态跨步保留。`,nodes:{clear:'清空旧梯度',forward:'GPT 前向 → logits',loss:'交叉熵 → loss',back:'loss / GPT 反向',clip:'全局裁剪',update:'AdamW 更新',state:'m / v / t 跨步保留'},frames:[
   ['清空本步旧梯度','每步开始清零模型梯度；优化器的 m、v 和步数不清零。','zero_grad → 当前步',['clear','state']],
   ['前向与损失','将本页两个 batch 的 ids 输入模型，按 targets 求平均交叉熵并记录更新前损失。','ids → GPT → logits → loss',['forward','loss']],
   ['反向传播','损失层把梯度传给 logits；GPT 按相反顺序传播并累加参数梯度。','loss.backward → GPT.backward',['loss','back']],
   ['裁剪并更新','先用全局范数裁剪梯度，再由同一个 AdamW 更新参数和 m/v/t。','全部 grad → clip → optimizer.step',['back','clip','update','state']],
   ['进入下一步','继续使用同一个优化器，下一步重新清零梯度后重复；最终返回逐步损失。','保留 m/v/t → 下一步 zero_grad',['update','state','clear']]]},
  t_checkpoint:{title:'检查点：保存后恢复同一步状态',summary:'公开用例固定 AdamW 学习率 0.01、default_rng(42)，保存前更新一次；恢复后复现下一次更新。',nodes:{update:'第 1 次更新',save:'保存 checkpoint',params:'参数 data',optim:'优化器 m / v / t / 配置',rng:'随机生成器状态',next:'第 2 次更新参照',load:'加载并清零旧 grad',compare:'重做第 2 次并比较'},frames:[
   ['先完成一次更新','先更新模型，让参数与优化器状态对应同一个训练步。','模型 + 优化器 → 更新 1',['update','params','optim']],
   ['保存三类状态','同一检查点记录参数、优化器 m/v/t 与配置，以及随机生成器状态。','参数 + 优化器 + RNG → 文件',['params','optim','rng','save']],
   ['继续一次作为参照','在保存后的状态再更新一次，记录 logits 与 RNG 的下一段结果。','保存后状态 → 更新 2 → 参照',['save','next','rng']],
   ['加载并恢复','将参数、优化器和随机状态恢复到保存时，清除遗留 grad。','文件 → 参数 + 优化器 + RNG → 清零 grad',['save','load','params','optim','rng']],
   ['重复下一次并对照','用恢复后的状态重复相同步骤；结果与参照一致才说明这一续训样例通过。','恢复 → 更新 2 → 对照 logits 与 RNG',['load','compare','next']]]},
  t_project:{title:'综合任务：一次真实训练更新',summary:'本用例使用两条序列、个人模型和调用方的 AdamW；返回更新前的平均损失。',nodes:{inputs:'ids + targets',model:'个人 GPT',loss:'交叉熵 loss',grad:'参数梯度',clip:'全局裁剪 limit=1',optim:'传入的 AdamW',output:'更新前 loss'},frames:[
   ['清除旧梯度','本用例预置旧梯度；首先清零，防止它混入当前步骤。','model.zero_grad → 新的一步',['model','grad']],
   ['前向和损失','用 ids 计算 logits，按 targets 求平均损失；该标量稍后作为返回值。','ids → GPT → logits + targets → loss',['inputs','model','loss','output']],
   ['完整反向','从 loss 回传 logits 梯度，再经过 GPT 的全部模块累加参数梯度。','loss.backward → GPT.backward → grad',['loss','model','grad']],
   ['裁剪并更新','统一裁剪全局梯度至上限 1，调用传入的优化器执行一次更新。','grad → clip_grad → optimizer.step',['grad','clip','optim','model']],
   ['返回更新前损失','返回此前计算的零维 loss；模型参数已经完成一次更新。','loss → 返回值；参数保持更新后的状态',['loss','output','model']]]}
 };
 const spec=specs[id],panel=document.createElement('div');panel.className='understanding shared-flow';
 panel.innerHTML=`<h2>${esc(spec.title)}</h2><p>${esc(spec.summary)}</p><div class="controls"><button type="button" data-back>← 上一步</button><button type="button" data-next>下一步 →</button><button type="button" data-play>自动演示</button><button type="button" data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><p data-route></p></div><div class="shared-nodes">${Object.entries(spec.nodes).map(([key,label])=>`<div data-node="${key}">${esc(label)}</div>`).join('')}</div>${ids?'<details><summary>展开公开样例的输入与目标</summary><div data-example></div></details>':''}`;
 host.append(panel);const $=key=>panel.querySelector('[data-'+key+']');if(ids)$('example').innerHTML=arrayView('ids',ids)+(targets?arrayView('targets',targets):'');
 let index=0,timer=null;const stop=()=>{if(timer)clearInterval(timer);timer=null;$('play').textContent='自动演示';};
 function draw(){const [title,note,route,active]=spec.frames[index];$('title').textContent=title;$('note').textContent=note;$('route').textContent=route;panel.querySelectorAll('[data-node]').forEach(el=>el.classList.toggle('active',active.includes(el.dataset.node)));$('count').textContent=`${index+1}/${spec.frames.length}`;$('back').disabled=index===0;$('next').disabled=index===spec.frames.length-1;}
 $('back').onclick=()=>{stop();index=Math.max(0,index-1);draw();};$('next').onclick=()=>{stop();index=Math.min(spec.frames.length-1,index+1);draw();};$('reset').onclick=()=>{stop();index=0;draw();};$('play').onclick=()=>{if(timer){stop();return;}if(index===spec.frames.length-1)index=0;draw();timer=setInterval(()=>{index++;draw();if(index===spec.frames.length-1)stop();},2000);$('play').textContent='暂停';};root.understandingCleanup=stop;draw();
}
function randomFlow(host,c,id){
 const seed=Number(c.args[0]),dims=c.args[1],multiple=id==='random_sequence',samples=multiple?c.expected:[c.expected],panel=document.createElement('div');panel.className='understanding shared-flow';
 panel.innerHTML=`<h2>随机生成器：种子与状态如何推进</h2><p>公开样例：seed=${seed}，每份样本形状 (${dims.join(', ')})。图中展示测评记录的样本，不在浏览器重新实现 NumPy 随机数。</p><div class="controls"><button type="button" data-back>← 上一步</button><button type="button" data-next>下一步 →</button><button type="button" data-play>自动演示</button><button type="button" data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><p data-route></p></div><div class="shared-nodes"><div data-node="seed">seed<br>${seed}</div><div data-node="s0">生成器初始状态 S₀</div><div data-node="first">第 1 次采样<br>形状 (${dims.join(', ')})</div><div data-node="s1">采样后状态 S₁</div>${multiple?`<div data-node="second">第 2 次采样<br>形状 (${dims.join(', ')})</div><div data-node="s2">继续推进的状态 S₂</div>`:''}</div><div data-output></div>`;
 host.append(panel);const $=key=>panel.querySelector('[data-'+key+']');
 const frames=[{title:'用 seed 创建一次生成器',note:'种子确定生成器的初始状态。状态由生成器持有，不是输入数组的一个轴。',route:'seed → 默认生成器 → 初始状态 S₀',nodes:['seed','s0'],count:0},{title:'第一次采样并推进状态',note:'从 S₀ 生成第一份指定形状的标准正态样本；生成器随后进入 S₁。',route:'S₀ → 第一份样本 + S₁',nodes:['s0','first','s1'],count:1}];
 if(multiple)frames.push({title:'沿用状态继续第二次采样',note:'第二次从 S₁ 继续生成，而不是重新用 seed 创建生成器；最终将两份样本按先后顺序堆叠。',route:'S₁ → 第二份样本 + S₂ → 堆叠',nodes:['s1','second','s2'],count:2});
 let index=0,timer=null;const stop=()=>{if(timer)clearInterval(timer);timer=null;$('play').textContent='自动演示';};
 function draw(){const f=frames[index];$('title').textContent=f.title;$('note').textContent=f.note;$('route').textContent=f.route;panel.querySelectorAll('[data-node]').forEach(el=>el.classList.toggle('active',f.nodes.includes(el.dataset.node)));$('output').innerHTML=samples.slice(0,f.count).map((a,i)=>arrayView(`第 ${i+1} 份样本 · 全部位置`,a)).join('');$('count').textContent=`${index+1}/${frames.length}`;$('back').disabled=index===0;$('next').disabled=index===frames.length-1;}
 $('back').onclick=()=>{stop();index=Math.max(0,index-1);draw();};$('next').onclick=()=>{stop();index=Math.min(frames.length-1,index+1);draw();};$('reset').onclick=()=>{stop();index=0;draw();};$('play').onclick=()=>{if(timer){stop();return;}if(index===frames.length-1)index=0;draw();timer=setInterval(()=>{index++;draw();if(index===frames.length-1)stop();},2000);$('play').textContent='暂停';};root.understandingCleanup=stop;draw();
}
function optimizerTrace(input){
 const cfg=input.config||{},lr=cfg.lr??.001,b1=cfg.beta1??.9,b2=cfg.beta2??.999,eps=cfg.eps??1e-8,decay=cfg.decay??.01,limit=input.limit;
 const map=(a,f)=>Array.isArray(a)?a.map(x=>map(x,f)):f(a),zip=(a,b,f)=>Array.isArray(a)?a.map((x,i)=>zip(x,b[i],f)):f(a,b),flat=a=>Array.isArray(a)?a.flatMap(flat):[a];
 let weights=JSON.parse(JSON.stringify(input.weights)),m=weights.map(w=>map(w,()=>0)),v=weights.map(w=>map(w,()=>0));const frames=[];
 input.gradients.forEach((grad,idx)=>{
  const before=JSON.parse(JSON.stringify(weights)),norm=Math.sqrt(grad.flatMap(flat).reduce((sum,x)=>sum+x*x,0)),scale=Math.min(1,limit/(norm+1e-12)),clipped=grad.map(g=>map(g,x=>x*scale)),step=idx+1;
  m=m.map((old,i)=>zip(old,clipped[i],(a,g)=>b1*a+(1-b1)*g));v=v.map((old,i)=>zip(old,clipped[i],(a,g)=>b2*a+(1-b2)*g*g));
  weights=weights.map((w,i)=>{const mh=map(m[i],x=>x/(1-b1**step)),vh=map(v[i],x=>x/(1-b2**step)),correction=zip(mh,vh,(x,y)=>lr*x/(Math.sqrt(y)+eps)),factor=Array.isArray(w[0])?1-lr*decay:1;return zip(w,correction,(x,delta)=>x*factor-delta);});
  frames.push({step,before,grad,norm,scale,clipped,m:JSON.parse(JSON.stringify(m)),v:JSON.parse(JSON.stringify(v)),weights:JSON.parse(JSON.stringify(weights))});
 });return {frames,weights,m,v};
}
function optimizerFlow(host,c){
 const input=c.input,trace=optimizerTrace(input),panel=document.createElement('div');panel.className='understanding optimizer-flow';
 panel.innerHTML='<h2>从梯度到参数：连续更新</h2><p>使用本页公开样例，保留全部参数张量；先合并计算全局范数，再更新每个参数的状态。</p><div class="controls"><button type="button" data-back>← 上一步</button><button type="button" data-next>下一步 →</button><button type="button" data-play>自动演示</button><button type="button" data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><div data-formula></div></div><div data-arrays></div>';
 host.append(panel);const $=key=>panel.querySelector('[data-'+key+']');let index=0,timer=null,total=trace.frames.length*4;const stop=()=>{if(timer)clearInterval(timer);timer=null;$('play').textContent='自动演示';};
 const group=(title,arrays)=>arrays.map((a,i)=>arrayView(`${title} · 参数 ${i}`,a)).join('');
 function draw(){const f=trace.frames[Math.floor(index/4)],stage=index%4;
  const titles=['所有参数共同计算范数','用一个比例裁剪梯度','更新本参数的 m 与 v','使用校正后的状态更新参数'];
  const notes=[`第 ${f.step} 步：所有梯度平方相加再开方，裁剪前范数 ${fmt(f.norm)}；上限 ${input.limit}。`,`统一缩放比例 ${fmt(f.scale)}；每个参数使用同一比例。`,`m、v 在每步沿用上一步状态，当前步 t=${f.step}；图中显示更新后的状态。`,`第 ${f.step} 步更新前后的所有参数；二维及更高维权重按配置衰减，一维参数不衰减。`];
  const formulas=[String.raw`n=\sqrt{\sum_{\theta,i}g_{\theta,i}^{2}}`,String.raw`g\leftarrow g\min(1,\mathrm{limit}/(n+10^{-12}))`,String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`,String.raw`\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}`];
  $('title').textContent=`第 ${f.step} 次更新 · ${titles[stage]}`;$('note').textContent=notes[stage];mathSet($('formula'),formulas[stage]);
  $('arrays').innerHTML=stage===0?group('原始梯度',f.grad):stage===1?group('裁剪后的梯度',f.clipped):stage===2?group('一阶状态 m',f.m)+group('二阶状态 v',f.v):group('更新前参数',f.before)+group('更新后参数',f.weights);
  $('count').textContent=`${index+1}/${total}`;$('back').disabled=index===0;$('next').disabled=index===total-1;
 }
 $('back').onclick=()=>{stop();index=Math.max(0,index-1);draw();};$('next').onclick=()=>{stop();index=Math.min(total-1,index+1);draw();};$('reset').onclick=()=>{stop();index=0;draw();};$('play').onclick=()=>{if(timer){stop();return;}if(index===total-1)index=0;draw();timer=setInterval(()=>{index++;draw();if(index===total-1)stop();},1800);$('play').textContent='暂停';};root.understandingCleanup=stop;draw();
}
function multiheadFlow(host,c){
 const X=c.input.X,[B,T,C]=[X.length,X[0].length,X[0][0].length],H=c.input.init[1],d=C/H,panel=document.createElement('div');panel.className='understanding shared-flow';
 panel.innerHTML=`<h2>多头注意力：前向与反向的形状路径</h2><p>本例 B=${B}、T=${T}、C=${C}、H=${H}，每头 d=${d}。所有 batch、token 和 head 均保留；这里演示位置与梯度的路径，数值由个人实现计算。</p><div class="controls"><button type="button" data-back>← 上一步</button><button type="button" data-next>下一步 →</button><button type="button" data-play>自动演示</button><button type="button" data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><div data-formula></div><p data-route></p></div><div class="shared-nodes"><div data-node="x">输入 X<br>(${B},${T},${C})</div><div data-node="qkv">qkv 投影<br>(${B},${T},${3*C})</div><div data-node="split">拆分 Q/K/V<br>各 (${B},${T},${C})</div><div data-node="heads">拆头、换轴<br>各 (${B},${H},${T},${d})</div><div data-node="core">注意力核心<br>(${B},${H},${T},${d})</div><div data-node="merge">先换轴再合头<br>(${B},${T},${C})</div><div data-node="proj">输出投影 proj<br>(${B},${T},${C})</div><div data-node="g">上游梯度 G<br>(${B},${T},${C})</div><div data-node="dx">输入梯度 dX<br>(${B},${T},${C})</div></div><details><summary>展开本例全部输入 X</summary><div data-x></div></details>`;
 host.append(panel);const $=key=>panel.querySelector('[data-'+key+']');$('x').innerHTML=arrayView('X · 全部 batch/token',X);
 const frames=[
  {title:'一次投影得到三组特征',note:'同一个输入经过 qkv 线性层，最后一轴分成 Q、K、V 三个连续区间。',route:'X → qkv → Q / K / V',nodes:['x','qkv','split'],tex:String.raw`\operatorname{shape}(X)=\operatorname{shape}(Q)=\operatorname{shape}(K)=\operatorname{shape}(V)=(B,T,C)`},
  {title:'每一组分别拆头并换轴',note:'每个 token 的 C 个特征先拆为 H 组、每组 d 个；再交换 token 与 head 轴。',route:'Q / K / V → 拆成 H×d → 交换 head/token',nodes:['split','heads'],tex:String.raw`(B,T,C)\rightarrow(B,T,H,d)\rightarrow(B,H,T,d)`},
  {title:'每个 head 独立读取内容',note:'Q/K/V 按对应 batch、head 送入注意力核心，经过打分、掩码和加权读取；各头各自得到输出。',route:'各头 Q / K / V → AttentionCore',nodes:['heads','core'],tex:String.raw`Y_{\mathrm{head}}=\operatorname{Attention}(Q,K,V)`},
  {title:'先换轴再合头，最后输出投影',note:'将 head 轴移回 token 后，再把 H 与 d 拼成 C；proj 作用于每个位置。',route:'各头输出 → 换轴 → 合头 → proj → Y',nodes:['core','merge','proj'],tex:String.raw`(B,H,T,d)\rightarrow(B,T,H,d)\rightarrow(B,T,C)`},
  {title:'反向从输出投影回到各头',note:'G 先进入 proj.backward；将得到的梯度按合头的逆序拆头、换轴，交给 AttentionCore.backward。',route:'G → proj.backward → 拆头 → core.backward',nodes:['g','proj','merge','core'],tex:String.raw`G\rightarrow\mathrm dY_{\mathrm{head}}\rightarrow(\mathrm dQ,\mathrm dK,\mathrm dV)`},
  {title:'三路梯度各自还原并拼接',note:'dQ、dK、dV 各自换轴合头，按照前向切分的顺序拼回 qkv 输出梯度；再交给 qkv.backward。',route:'dQ / dK / dV → 分别合头 → 拼接 → qkv.backward → dX',nodes:['core','heads','split','qkv','dx'],tex:String.raw`\mathrm dQKV=\operatorname{concat}(\mathrm dQ,\mathrm dK,\mathrm dV)`}
 ];let index=0,timer=null;const stop=()=>{if(timer)clearInterval(timer);timer=null;$('play').textContent='自动演示';};
 function draw(){const f=frames[index];$('title').textContent=f.title;$('note').textContent=f.note;$('route').textContent=f.route;mathSet($('formula'),f.tex);panel.querySelectorAll('[data-node]').forEach(el=>el.classList.toggle('active',f.nodes.includes(el.dataset.node)));$('count').textContent=`${index+1}/${frames.length}`;$('back').disabled=index===0;$('next').disabled=index===frames.length-1;}
 $('back').onclick=()=>{stop();index=Math.max(0,index-1);draw();};$('next').onclick=()=>{stop();index=Math.min(frames.length-1,index+1);draw();};$('reset').onclick=()=>{stop();index=0;draw();};$('play').onclick=()=>{if(timer){stop();return;}if(index===frames.length-1)index=0;draw();timer=setInterval(()=>{index++;draw();if(index===frames.length-1)stop();},2000);$('play').textContent='暂停';};root.understandingCleanup=stop;draw();
}
function sharedEmbedding(host,c){
 const ids=c.input.ids,[V,C]=c.input.init,[B,T]=[ids.length,ids[0].length],panel=document.createElement('div');panel.className='understanding shared-flow';
 panel.innerHTML=`<h2>同一词表的两条梯度路径</h2><p>同一个参数 E（${V},${C}）既用于输入查表，也用于输出头。两条贡献最终累加到同一个 E.grad。</p><div class="controls"><button type="button" data-back>← 上一步</button><button type="button" data-next>下一步 →</button><button type="button" data-play>自动演示</button><button type="button" data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><div data-formula></div><p data-route></p></div><div class="shared-nodes"><div data-node="ids">输入 ids<br>(${B},${T})</div><div data-node="table">共享词表 E<br>(${V},${C})</div><div data-node="lookup">输入查表 + 位置表<br>(${B},${T},${C})</div><div data-node="blocks">Blocks + 最终 Norm<br>(${B},${T},${C})</div><div data-node="head">输出头：H Eᵀ<br>(${B},${T},${V})</div><div data-node="upstream">上游 G<br>(${B},${T},${V})</div><div data-node="headgrad">输出头贡献<br>dE_head · (${V},${C})</div><div data-node="lookupgrad">输入查表贡献<br>dE_lookup · (${V},${C})</div><div data-node="sum">同一个 E.grad<br>(${V},${C})</div></div><details><summary>展开本例全部输入编号</summary><div data-ids></div></details>`;
 host.append(panel);const $=key=>panel.querySelector('[data-'+key+']');$('ids').innerHTML=arrayView('ids · 各 batch/token',ids);
 const frames=[
  {title:'输入路径：编号查同一张表',note:'每个 ids[b,t] 找到 E 的对应行，加上位置向量；重复编号指向同一组参数。',nodes:['ids','table','lookup'],tex:String.raw`X_{b,t,c}=E_{\mathrm{ids}_{b,t},c}+P_{t,c}`},
  {title:'前向：经过 Block 与输出头',note:'查表结果经过全部 Block 与末尾 Norm；输出头仍使用同一个 E 的转置，产生各类别分数。',nodes:['lookup','blocks','table','head'],tex:String.raw`\mathrm{logits}=H_{\mathrm{out}}E^{\mathsf T}`},
  {title:'输出头路径：G 回到词表',note:'每个位置的 G 与末尾激活共同贡献 dE_head，沿全部 batch 和 token 求和。即使某一行未在输入中查到，也可能收到输出头贡献。',nodes:['upstream','head','blocks','headgrad','table'],tex:String.raw`\mathrm dE_{\mathrm{head}}=G_2^{\mathsf T}H_{\mathrm{out},2}`},
  {title:'输出头路径：G 回到末尾激活',note:'同一份 G 也经过共享 E 回到 H，再按相反顺序通过 Norm 和所有 Block，得到输入位置的 dX。',nodes:['upstream','head','table','blocks','lookup'],tex:String.raw`\mathrm dH_{\mathrm{out}}=GE`},
  {title:'输入查表路径：dX 散射回 E',note:'每个 batch/token 的 dX 按原 ids 送回词表对应行；重复编号的贡献累加。位置表梯度沿 batch 汇总。',nodes:['lookup','ids','lookupgrad','table'],tex:String.raw`(\mathrm dE_{\mathrm{lookup}})_{v,c}=\sum_{b,t:\,\mathrm{ids}_{b,t}=v}\mathrm dX_{b,t,c}`},
  {title:'两条贡献汇入同一个梯度',note:'把输出头和输入查表的贡献都累加到同一个 E.grad；参数收集和优化器更新时，这个 E 只登记一次。',nodes:['headgrad','lookupgrad','table','sum'],tex:String.raw`\mathrm dE=\mathrm dE_{\mathrm{head}}+\mathrm dE_{\mathrm{lookup}}`}
 ];
 let index=0,timer=null;const stop=()=>{if(timer)clearInterval(timer);timer=null;$('play').textContent='自动演示';};
 const routes=['ids + E → 查表 + 位置','查表 → Blocks / Norm → 输出头 ← E','G + H → 输出头贡献 → E.grad','G + E → H → Blocks / Norm → dX','dX + ids → 查表贡献 → E.grad','输出头贡献 + 查表贡献 → 同一个 E.grad'];
 function draw(){const f=frames[index];$('title').textContent=f.title;$('note').textContent=f.note;$('route').textContent=routes[index];mathSet($('formula'),f.tex);panel.querySelectorAll('[data-node]').forEach(el=>el.classList.toggle('active',f.nodes.includes(el.dataset.node)));$('count').textContent=`${index+1}/${frames.length}`;$('back').disabled=index===0;$('next').disabled=index===frames.length-1;}
 $('back').onclick=()=>{stop();index=Math.max(0,index-1);draw();};$('next').onclick=()=>{stop();index=Math.min(frames.length-1,index+1);draw();};$('reset').onclick=()=>{stop();index=0;draw();};$('play').onclick=()=>{if(timer){stop();return;}if(index===frames.length-1)index=0;draw();timer=setInterval(()=>{index++;draw();if(index===frames.length-1)stop();},2000);$('play').textContent='暂停';};root.understandingCleanup=stop;draw();
}
function linear(host,c,id){
 const X=id==='linear'?c.args[0]:c.X,W=id==='linear'?c.args[1]:c.parameters.W;
 if(!W||!Array.isArray(X[0][0]))return;
 const B=X.length,T=X[0].length,C=W.length,D=W[0].length;
 const G=id==='linear'?X.map((batch,b)=>batch.map((_,t)=>Array.from({length:D},(_,j)=>(1+b*T*D+t*D+j)/10))):c.G;
 const bias=id==='linear'?c.args[2]:c.parameters.b||Array(D).fill(0);
 const grads=linearBackward(X,W,G);
 const panel=document.createElement('div');panel.className='understanding';
 panel.innerHTML=`<h2>矩阵反向传播：一个格子的梯度从哪里来？</h2><p>同一个前向样例 X（${B},${T},${C}）、W（${C},${D}）。${id==='linear'?'为演算反向，额外给定下方上游梯度 G；它表示损失对 Y 的敏感程度。':'G 是损失对本层输出 Y 的梯度，由调用方给定。'}</p><div class="gradient-route"><span>X</span><span>→ XW+b →</span><span>Y</span><span>← ${mathBlock(String.raw`G=\frac{\partial L}{\partial Y}`,true)}</span></div><div class="visual-controls"><label>观察哪项计算<select id="linMode"><option value="forward">前向：一个 Y[b,t,j]</option><option value="dX">输入梯度 dX[b,t,i]</option><option value="dW">共享权重梯度 dW[i,j]</option><option value="db">共享偏置梯度 db[j]</option></select></label>${selector('linB','batch b',B)}${selector('linT','token t',T)}${selector('linI','输入特征 i',C)}${selector('linJ','输出特征 j',D)}</div><div id="linFormula" class="gradient-formula" aria-live="polite"></div><div id="linSources"></div><div class="visual-controls"><button id="linPrev">← 撤回一项</button><button id="linNext">累加下一项 →</button><button id="linPlay">自动累加</button><button id="linReset">重置累计</button><label>累计项数 <input id="linStep" type="range" min="0" value="0"></label><output id="linCount"></output></div><div id="linTerms"></div><div id="linAnswer" class="gradient-answer" aria-live="polite"></div><details><summary>查看完整 G 与最终梯度（所有 batch）</summary><div id="linAll"></div></details><p class="small">公式中的转置是把同一个输出位置的贡献对齐到输入/权重坐标。dX 对输出特征求和；dW、db 对所有 batch 和 token 求和。这里不额外除以 B×T。</p>`;
 if(id==='linear'){
  panel.querySelector('h2').textContent='Linear 前向：一个输出来自哪一行、哪一列？';
  panel.querySelector('p').textContent='选择 batch、token 和输出特征，观察该位置的输入与权重一列逐项相乘，累加后加上偏置。';
  panel.querySelector('.gradient-route').remove();
  panel.querySelectorAll('#linMode option:not([value=forward])').forEach(option=>option.remove());
  panel.querySelector('details').remove();
  panel.querySelector('p.small').textContent='每个位置使用同一组权重与偏置；batch 和 token 的编号保留。';
 }
 host.append(panel);let count=0,clock=null;
 const $=s=>panel.querySelector('#'+s);const stop=()=>{clearInterval(clock);clock=null;$('linPlay').textContent='自动累加';};root.understandingCleanup=stop;
 if(id!=='linear')$('linAll').innerHTML=arrayView('上游梯度 G',G)+arrayView('完整 dX',grads.dX)+arrayView('完整 dW',grads.dW)+arrayView('完整 db',grads.db);
 function draw(){const mode=$('linMode').value,b=+$('linB').value,t=+$('linT').value,i=+$('linI').value,j=+$('linJ').value;let terms=[],formula='',answer=0;
 for(const [key,active] of [['linB',!['dW','db'].includes(mode)],['linT',!['dW','db'].includes(mode)],['linI',['dX','dW'].includes(mode)],['linJ',mode!=='dX']])$(key).disabled=!active;
 if(mode==='forward'){formula=String.raw`Y_{${b},${t},${j}}&=\sum_{i=0}^{${C-1}}\textcolor{#2567a4}{X_{${b},${t},i}}\textcolor{#79449c}{W_{i,${j}}}+b_{${j}}`;terms=W.map((_,k)=>({label:`i=${k}`,left:X[b][t][k],right:W[k][j]}));answer=terms.reduce((s,v)=>s+v.left*v.right,bias[j]);}
 if(mode==='dX'){formula=String.raw`\mathrm dX_{${b},${t},${i}}&=\sum_{j=0}^{${D-1}}\textcolor{#b16b20}{G_{${b},${t},j}}\textcolor{#79449c}{W_{${i},j}}\\\mathrm dX_2&=G_2W^{\mathsf T}`;terms=G[b][t].map((v,k)=>({label:`输出特征 j=${k}`,left:v,right:W[i][k]}));answer=grads.dX[b][t][i];}
 if(mode==='dW'){formula=String.raw`\mathrm dW_{${i},${j}}&=\sum_{b=0}^{${B-1}}\sum_{t=0}^{${T-1}}\textcolor{#2567a4}{X_{b,t,${i}}}\textcolor{#b16b20}{G_{b,t,${j}}}\\\mathrm dW&=X_2^{\mathsf T}G_2`;terms=X.flatMap((batch,bb)=>batch.map((row,tt)=>({label:`b=${bb}, t=${tt} ↔ 展平行 n=${bb*T+tt}`,left:row[i],right:G[bb][tt][j]})));answer=grads.dW[i][j];}
 if(mode==='db'){formula=String.raw`\mathrm db_{${j}}&=\sum_{b=0}^{${B-1}}\sum_{t=0}^{${T-1}}\textcolor{#b16b20}{G_{b,t,${j}}}`;terms=G.flatMap((batch,bb)=>batch.map((row,tt)=>({label:`b=${bb}, t=${tt}`,left:row[j],right:1})));answer=grads.db[j];}
 count=Math.min(count,terms.length);$('linStep').max=terms.length;$('linStep').value=count;$('linCount').textContent=`${count}/${terms.length}`;
 const active=Math.max(0,count-1),bb=Math.floor(active/T),tt=active%T;
 const term=mode==='forward'?String.raw`\textcolor{#2567a4}{X_{${b},${t},${active}}}\textcolor{#79449c}{W_{${active},${j}}}`:mode==='dX'?String.raw`\textcolor{#b16b20}{G_{${b},${t},${active}}}\textcolor{#79449c}{W_{${i},${active}}}`:mode==='dW'?String.raw`\textcolor{#2567a4}{X_{${bb},${tt},${i}}}\textcolor{#b16b20}{G_{${bb},${tt},${j}}}`:String.raw`\textcolor{#b16b20}{G_{${bb},${tt},${j}}}`;
 mathSet($('linFormula'),String.raw`\begin{aligned}${formula}${count?String.raw`\\\mathrm{source}&=${term}`:''}\end{aligned}`);
 const label=(name,color)=>`<p class="math-source-label math-source-${color}">${name}</p>`,mark=(r,c)=>count?[r+','+c]:[];
 $('linSources').innerHTML=mode==='dW'?label('X₂：行 n=b×T+t，蓝色来源','x')+matrix(X.flat(),mark(active,i))+label('G₂：同一行 n，橙色来源','g')+matrix(G.flat(),mark(active,j)):mode==='dX'?label('当前位置的上游梯度 G，橙色来源','g')+matrix([G[b][t]],mark(0,active))+label('输入特征 i 对应的 W 行，紫色来源','w')+matrix([W[i]],mark(0,active)):mode==='forward'?label('当前 token 的输入 X，蓝色来源','x')+matrix([X[b][t]],mark(0,active))+label('输出特征 j 对应的 W 列，紫色来源','w')+matrix([W.map(row=>row[j])],mark(0,active)):label('所有 batch/token 的 G，橙色来源','g')+matrix([terms.map(v=>v.left)],mark(0,active));
 $('linTerms').innerHTML='<div class="contribution-list">'+terms.map((term,k)=>`<div class="contribution ${k<count?'included':''}"><small>${term.label}</small><span>${fmt(term.left)} × ${fmt(term.right)} = <strong>${fmt(term.left*term.right)}</strong></span><span>${k<count?'✓ 已计入':'待计入'}</span></div>`).join('')+'</div>';
 const partial=terms.slice(0,count).reduce((s,v)=>s+v.left*v.right,0);$('linAnswer').textContent=`当前乘加累计：${fmt(partial)}${mode==='forward'?`；最后加偏置 ${fmt(bias[j])}`:''}。${count===terms.length?'✓ 已收齐全部贡献，结果 '+fmt(answer):'完整结果应为 '+fmt(answer)+'，继续逐项累加。'}`;
 $('linPrev').disabled=count===0;$('linNext').disabled=count===terms.length;
 }
 ['linMode','linB','linT','linI','linJ'].forEach(key=>$(key).onchange=()=>{stop();count=0;draw();});
 $('linStep').oninput=()=>{stop();count=+$('linStep').value;draw();};$('linNext').onclick=()=>{count++;draw();};$('linPrev').onclick=()=>{count--;draw();};$('linReset').onclick=()=>{stop();count=0;draw();};
 $('linPlay').onclick=()=>{if(clock)return stop();if(count===+$('linStep').max)count=0;clock=setInterval(()=>{count++;draw();if(count>=+$('linStep').max)stop();},850);$('linPlay').textContent='暂停';};draw();
}
function attentionFlow(host,c,out){
 const {Q,K,V}=c.input,causal=!!c.input.init[0],B=Q.length,H=Q[0].length,T=Q[0][0].length,Tk=K[0][0].length,d=Q[0][0][0].length;
 const panel=document.createElement('div');panel.className='understanding attention-flow';
 panel.innerHTML=`<h2>同一组 Q、K、V 如何产生输出</h2><p>公开样例保留全部 ${B} 个 batch × ${H} 个 head；每张表的行是 query，列是 key。打分宽度 d=${d}。按步骤观察分数、掩码、概率和内容读取。</p><div class="controls"><button type="button" data-back>← 上一步</button><button type="button" data-next>下一步 →</button><button type="button" data-play>自动演示</button><button type="button" data-reset>重新演示</button><span data-count></span></div><div class="process-explain" aria-live="polite"><strong data-title></strong><p data-note></p><div data-formula></div></div><div data-flow></div><details><summary>展开同一样例的全部 Q、K 与 V</summary><div data-inputs></div></details>`;
 host.append(panel);const $=key=>panel.querySelector('[data-'+key+']');
 $('inputs').innerHTML=arrayView('Q · query 查询',Q)+arrayView('K · key 匹配',K)+arrayView('V · value 内容',V);
 let phase=0,timer=null;const stop=()=>{if(timer)clearInterval(timer);timer=null;$('play').textContent='自动演示';};
 const names=['Q 与 K 打分','屏蔽未来位置','每行归一化为概率','按概率读取 V'];
 function maps(values,kind){return '<div class="attention-maps">'+values.flatMap((heads,b)=>heads.map((matrix,h)=>`<div><strong>batch ${b} · head ${h} · 行=query / 列=key</strong><div class="attention-grid" style="grid-template-columns:repeat(${Tk},minmax(38px,1fr))">${matrix.flatMap((row,t)=>row.map((v,s)=>{const masked=kind==='masked'&&causal&&s>t;return `<span class="flow-cell ${masked?'flow-masked':''}">${masked?'−∞':fmt(v)}</span>`;})).join('')}</div></div>`)).join('')+'</div>';}
 function draw(){
  const titles=[`每个 query 与每个 key 点积，再除以 √${d}`,'按位置遮住未来的 key','沿每个 query 行计算 Softmax','用这一行概率加权所有 V'];
  const notes=[`Q 的每一行和 K 的每一行沿 d=${d} 乘加。下表保留所有 batch/head 和 ${T}×${Tk} 个位置对。`,causal?'只屏蔽 key 位置 s 大于 query 位置 t 的格子；其余原始分数不变。−∞ 经 Softmax 后得到 0。':'此样例关闭因果掩码，所有位置保留原分数。',causal?'对每一行的允许位置归一化；被屏蔽的未来位置概率为 0，每行概率之和为 1。':'对每一行的全部 key 归一化，每行概率之和为 1。','每个 query 的概率行分别乘同一 batch/head 的 V；沿 key 位置求和，得到该 query 的输出向量。'];
  const formulas=[String.raw`S=QK^{\mathsf T}/\sqrt{d}`,causal?String.raw`\widetilde S_{t,s}=\begin{cases}S_{t,s}&s\le t\\-\infty&s>t\end{cases}`:String.raw`\widetilde S=S`,String.raw`A_{t,s}=\operatorname{softmax}_s(\widetilde S_{t,s})`,String.raw`Y_{t,j}=\sum_s A_{t,s}V_{s,j}`];
  $('title').textContent=titles[phase];$('note').textContent=notes[phase];mathSet($('formula'),formulas[phase]);
  $('flow').innerHTML=`<p class="small">${names.map((n,i)=>`${i===phase?'▶ ':''}${esc(n)}`).join(' → ')}</p>`+(phase<3?maps(phase===2?out.A:out.S,phase===1?'masked':'values'):maps(out.A,'values')+arrayView('V · 各 key 的内容',V)+arrayView('Y · 每个 query 的输出',out.Y));
  $('flow').classList.remove('flow-transition');void $('flow').offsetWidth;$('flow').classList.add('flow-transition');
  $('count').textContent=`${phase+1}/4`;$('back').disabled=phase===0;$('next').disabled=phase===3;
 }
 $('back').onclick=()=>{stop();phase=Math.max(0,phase-1);draw();};$('next').onclick=()=>{stop();phase=Math.min(3,phase+1);draw();};$('reset').onclick=()=>{stop();phase=0;draw();};
 $('play').onclick=()=>{if(timer){stop();return;}if(phase===3)phase=0;draw();timer=setInterval(()=>{phase++;draw();if(phase===3)stop();},1800);$('play').textContent='暂停';};
 root.understandingCleanup=stop;draw();
}
function attention(host,c){
 const {Q,K,V,G}=c.input,causal=c.input.init[0];const out=attentionBackward(Q,K,V,G,causal),B=Q.length,H=Q[0].length,T=Q[0][0].length,d=V[0][0][0].length;
 attentionFlow(host,c,out);
 const panel=document.createElement('div');panel.className='understanding';panel.innerHTML=`<h2>注意力显微镜：读取内容与梯度回流</h2><p>下方保留全部 ${B} 个 batch × ${H} 个 head。前向选择 query 行观察读取；查看 dV 时选择 value/key 列，观察所有 query 的贡献如何汇入同一个 V。</p><div class="visual-controls">${selector('attB','batch b',B)}${selector('attH','head h',H)}${selector('attT','query token t',T)}${selector('attD','value 特征 d',d)}<label>阶段<select id="attPhase"><option value="A">概率与加权输出</option><option value="dV">反向：多个 query 汇入 dV</option><option value="dS">反向：Softmax 的耦合梯度</option><option value="dQ">反向：打分回到 Q/K</option></select></label></div><div id="attMap"></div><div id="attCalc" class="gradient-answer" aria-live="polite"></div><div id="attTensor"></div></div>`;host.append(panel);const $=id=>panel.querySelector('#'+id);
 function draw(){const phase=$('attPhase').value,maxT=phase==='dV'?K[0][0].length:T;let t=+$('attT').value; if($('attT').options.length!==maxT){$('attT').innerHTML=Array.from({length:maxT},(_,i)=>'<option value="'+i+'">'+i+'</option>').join('');t=Math.min(t,maxT-1);$('attT').value=t;}const b=+$('attB').value,h=+$('attH').value,j=+$('attD').value;
 $('attT').parentElement.firstChild.textContent=phase==='dV'?'value/key token s':'query token t';
 $('attMap').innerHTML='<div class="attention-maps">'+out.A.flatMap((heads,bb)=>heads.map((a,hh)=>`<div><strong>b=${bb}, h=${hh} · 行=query，列=key</strong><div class="attention-grid" style="grid-template-columns:repeat(${a[0].length},minmax(38px,1fr))">${a.flatMap((row,tt)=>row.map((v,kk)=>`<button type="button" data-att-b="${bb}" data-att-h="${hh}" data-att-t="${tt}" data-att-k="${kk}" title="query ${tt} 读取 key ${kk}：概率 ${fmt(v)}" style="--prob:${v}" class="${bb===b&&hh===h&&(phase==='dV'?kk===t:tt===t)?'selected':''}">${causal&&kk>tt?'屏蔽':fmt(v)}</button>`)).join('')}</div></div>`)).join('')+'</div>';
 panel.querySelectorAll('[data-att-t]').forEach(el=>el.onclick=()=>{$('attB').value=el.dataset.attB;$('attH').value=el.dataset.attH;$('attT').value=phase==='dV'?el.dataset.attK:el.dataset.attT;draw();});
 let tex='',note='';
 if(phase==='A'){tex=String.raw`Y_{${b},${h},${t},${j}}=\sum_{s=0}^{${K[0][0].length-1}}\textcolor{#2567a4}{A_{${b},${h},${t},s}}V_{${b},${h},s,${j}}`;note='高亮 query 行：该行的概率在本 batch/head 内加权读取 V。';}
 if(phase==='dV'){tex=String.raw`\begin{aligned}\mathrm dV_{${b},${h},${t},${j}}&=\sum_{q=0}^{${T-1}}\textcolor{#2567a4}{A_{${b},${h},q,${t}}}G_{${b},${h},q,${j}}\\\mathrm dV&=A^{\mathsf T}G\end{aligned}`;note='高亮 key 列：所有 query 对这个 value 的读取贡献回到同一位置，沿 query 汇总。';}
 if(phase==='dS'){tex=String.raw`\begin{aligned}\mathrm dA&=GV^{\mathsf T}\\\mathrm dS&=A\odot\left[\mathrm dA-\operatorname{sum}_s(A\odot\mathrm dA)\right]\end{aligned}`;note='高亮 query 行：一个 score 会影响该行全部概率。沿 key 轴求和并保留轴；被屏蔽位置的概率和 dS 都为零。';}
 if(phase==='dQ'){tex=String.raw`\begin{aligned}\mathrm dQ&=\frac{\mathrm dS\,K}{\sqrt{${Q[0][0][0].length}}}\\\mathrm dK&=\frac{\mathrm dS^{\mathsf T}Q}{\sqrt{${Q[0][0][0].length}}}\end{aligned}`;note='Q 汇总各 key 的贡献，K 汇总各 query 的贡献。分母来自打分特征宽度；保留全部 batch/head。';}
 $('attCalc').innerHTML=mathBlock(tex)+`<p>${esc(note)}</p>`;
 $('attTensor').innerHTML=phase==='dQ'?arrayView('完整 dQ',out.dQ)+arrayView('完整 dK',out.dK):arrayView(phase==='A'?'完整输出 Y':'完整 '+phase,phase==='A'?out.Y:out[phase],phase==='dS'?['batch b','head h','query t','key s']:null);
 }panel.querySelectorAll('select').forEach(el=>el.onchange=draw);draw();
}
function residual(host,frames){
 const panel=document.createElement('div');panel.className='understanding';
 panel.innerHTML='<h2>残差：一份上游梯度怎样沿两条路径回流？</h2><div class="visual-controls"><label>反向阶段<select id="resPhase"><option value="0">第二条残差：回到 x₁</option><option value="1">第一条残差：回到 x</option></select></label></div><div id="resBody"></div>';host.append(panel);
 const draw=()=>{const phase=+panel.querySelector('select').value,frame=frames[phase];
 const formula=phase===0?String.raw`G_1=\textcolor{#2567a4}{G}+\textcolor{#79449c}{J_{F_2}(X_1)^{\mathsf T}G}`:String.raw`\mathrm dX=\textcolor{#2567a4}{G_1}+\textcolor{#79449c}{J_{F_1}(X)^{\mathsf T}G_1}`;
 panel.querySelector('#resBody').innerHTML=mathBlock(formula)+`<div class="gradient-route"><span class="math-source-x">蓝：直连贡献</span><span>＋</span><span class="math-source-w">紫：经过子网络的贡献</span></div><p>${esc(frame.text)}</p>${arrayView('直连贡献',frame.direct)}${arrayView('子网络贡献',frame.branch)}${arrayView('两路相加',frame.sum)}<p class="small">J 表示本阶段子网络的雅可比；实现时串联已有 backward。每条路径收到完整上游梯度，在共同来源处相加。</p>`;
 };
 panel.querySelector('select').onchange=draw;draw();
}
root.installUnderstanding=install;root.GradientMath={linearBackward,attentionBackward,optimizerTrace};
if(typeof module!=='undefined')module.exports=root.GradientMath;
})(typeof window==='undefined'?globalThis:window);
