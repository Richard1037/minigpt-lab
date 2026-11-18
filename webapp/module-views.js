/* 每个展示只回答对应模块的理解难点。 */
(function(root){'use strict';
 const range=n=>Array.from({length:n},(_,i)=>i),shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
 const coords=s=>s.length?range(s[0]).flatMap(i=>coords(s.slice(1)).map(p=>[i,...p])):[[]];
 const tex=(s,inline=false)=>root.CourseMath.block(s,inline),dims=s=>'('+s.join(',')+')';
 function rmsRelation(mode,picked,N,C){
  const marks={};
  function add(id,cells,indirect=false){const v=marks[id]||(marks[id]={direct:[],indirect:[]});v[indirect?'indirect':'direct'].push(...cells);}
  const row=r=>range(C).map(j=>[r,j]),col=j=>range(N).map(r=>[r,j]);
  if(!picked)return marks;
  const {id,r,j}=picked;
  if(mode==='parameter'){
   if(id==='dw'){for(const k of ['G','xh','product'])add(k,col(j));add('dw',[[0,j]]);}
   else{for(const k of ['G','xh','product'])add(k,[[r,j]]);add('dw',[[0,j]]);}
  }else if(mode==='input'){
   if(id==='w'){
    add('w',[[0,j]]);for(const k of ['G','u','direct'])add(k,col(j));
    add('r',range(N).map(n=>[n,0]));add('xh',range(N).flatMap(row),true);
    for(const k of ['stat','dX'])add(k,range(N).flatMap(row),true);add('dX',col(j));return marks;
   }
   add('r',[[r,0]]);add('xh',row(r),true);
   if(id==='r'||id==='xh'){
    for(const k of ['G','u'])add(k,row(r));add('w',row(0));
    for(const k of ['stat','dX'])add(k,row(r),true);if(id==='r')add('direct',row(r));return marks;
   }
   if(['G','u','direct'].includes(id)){
    for(const k of ['G','u','direct'])add(k,[[r,j]]);
    add('w',[[0,j]]);add('stat',row(r),true);add('dX',row(r),true);add('dX',[[r,j]]);
   }else{
    for(const k of ['G','u']){add(k,row(r),true);if(id!=='stat')add(k,[[r,j]]);}
    add('w',row(0));add('stat',[[r,j]],true);add('dX',[[r,j]]);if(id!=='stat')add('direct',[[r,j]]);
   }
  }else{
   if(id==='X'){
    add('X',[[r,j]]);add('stats',[[r,0],[r,1]],true);add('w',row(0));
    for(const k of ['xh','Y']){add(k,row(r),true);add(k,[[r,j]]);}
   }else if(id==='w'){
    add('w',[[0,j]]);add('xh',col(j));add('Y',col(j));
   }else if(id==='stats'){
    add('X',row(r));add('stats',[[r,0],[r,1]]);add('w',row(0));for(const k of ['xh','Y'])add(k,row(r));
   }else{
    add('X',row(r),true);add('X',[[r,j]]);add('stats',[[r,0],[r,1]],true);add('w',[[0,j]]);for(const k of ['xh','Y'])add(k,[[r,j]]);
   }
  }return marks;
 }
 function mountRms(c){
  const host=document.getElementById('animation'),xd=shape(c.X),C=xd.at(-1),positions=coords(xd.slice(0,-1)),N=positions.length;
  host.innerHTML='<section class="source-map rms-view"><h2>RMSNorm：行内统计与跨位置共享</h2><div class="mv-tabs"><button data-rms-mode="forward">前向归一化</button><button data-rms-mode="input">输入梯度</button><button data-rms-mode="parameter">参数梯度</button><button data-rms-reset>看整体</button></div><div class="mv-intro"></div><div class="mv-parameter"></div><div class="mv-grid"></div><p class="mv-legend"><span>实线：直接关系</span><span>虚线：经过共享统计量的间接关系</span><span>深色外框：选中位置</span></p><div class="mv-note" aria-live="polite"></div></section>';
  const box=host.firstElementChild;let mode='forward',picked=null;
  function card(id,title,R,D,formula,original){
   const marked=rmsRelation(mode,picked,N,C)[id]||{direct:[],indirect:[]};
   const has=(key,r,j)=>marked[key].some(p=>p[0]===r&&p[1]===j);
   const cls=id==='w'||id==='dw'?'sm-w':['G','u'].includes(id)?'sm-g':'sm-x';
   return `<section class="lm-matrix ${cls}"><h3>${title}</h3><div class="lm-shape">${tex(original,true)}</div><div class="mv-operation">${tex(formula)}</div><div class="lm-bracket"><div class="lm-cells" style="grid-template-columns:auto repeat(${D},22px)"><span></span>${range(D).map(j=>`<span class="lm-index">${id==='stats'?['m','r'][j]:j}</span>`).join('')}${range(R).map(r=>`<span class="lm-index">${id==='w'||id==='dw'?'':`(${positions[r].join(',')})`}</span>`+range(D).map(j=>`<button type="button" class="lm-cell ${has('direct',r,j)?'lm-related':has('indirect',r,j)?'mv-indirect':''} ${picked?.id===id&&picked.r===r&&picked.j===j?'lm-picked':''}" data-rms-cell="${id},${r},${j}" aria-label="RMS ${id} 位置 ${r} 特征 ${j}">·</button>`).join('')).join('')}</div></div></section>`;
  }
  const names={forward:'每个 token 用自己的特征算均方；这一行的逆均方根广播回该行，再乘所有位置共享的缩放向量。',input:'先把上游通过逐特征缩放传回，再把直接路径与统计量路径相加。点击一个 G 格子，区分同格的直接贡献与本行的间接贡献。',parameter:'前向缓存的归一化特征与上游同位置相乘，再沿全部位置求和。点击一个 G 格子只贡献到 dw 的同一特征；点击 dw 则查看该列的全部来源。'};
  function draw(){
   box.querySelector('.mv-intro').innerHTML=`<p>${names[mode]}</p><p class="small">输入原形状 ${tex(dims(xd),true)}；下表每行保留原位置坐标，完整显示 ${N} 个 token。</p>`;
   box.querySelector('.mv-parameter').innerHTML=mode==='parameter'?'':card('w','共享缩放 w',1,C,'Y=\\widehat X\\odot w',dims([C]));
   const regular=dims(xd),statShape=dims([...xd.slice(0,-1),1]);let html='';
   if(mode==='forward')html=card('X','输入 X',N,C,'X\\odot X',regular)+card('stats','行内统计量 · 两列',N,2,String.raw`m=\operatorname{mean}_c(X^2),\ r=(m+\epsilon)^{-1/2}`,'m,r:\ '+statShape)+card('xh','归一化特征',N,C,String.raw`\widehat X=X\odot r`,regular)+card('Y','输出',N,C,String.raw`Y=\widehat X\odot w`,regular);
   if(mode==='input')html=card('G','收到的上游 G',N,C,String.raw`G=\partial L/\partial Y`,regular)+card('xh','前向缓存：归一化特征',N,C,String.raw`\widehat X`,regular)+card('r','前向缓存：逆均方根',N,1,'r',statShape)+card('u','传回归一化特征的梯度',N,C,String.raw`u=G\odot w`,regular)+card('direct','直接路径',N,C,String.raw`\mathrm dX_{\rm direct}=r\odot u`,regular)+card('stat','统计量路径',N,C,String.raw`\mathrm dX_{\rm stat}=-r\widehat X\operatorname{mean}_c(u\odot\widehat X)`,regular)+card('dX','两条路径相加',N,C,String.raw`\mathrm dX=\mathrm dX_{\rm direct}+\mathrm dX_{\rm stat}`,regular);
   if(mode==='parameter')html=card('xh','前向缓存：归一化特征',N,C,String.raw`\widehat X`,regular)+card('G','收到的上游 G',N,C,'G',regular)+card('product','每个位置的参数梯度贡献',N,C,String.raw`P=G\odot\widehat X`,regular);
   const grid=box.querySelector('.mv-grid');grid.className='mv-grid mv-'+mode;grid.innerHTML=html;
   let note=mode==='parameter'?tex(String.raw`\mathrm dw_c=\sum_{\text{全部位置}}P_{\ldots,c}`)+card('dw','求和后的本次参数梯度',1,C,String.raw`\mathrm dw=\operatorname{sum}_{\text{位置}}P`,dims([C]))+'<p>只有位置轴被归约，特征轴保留；把本次贡献累加到 w.grad。</p>':mode==='input'?'<p>u 与直接路径逐元素对应；统计量路径中的均值让同一行的所有特征相互关联。不同 token 的输入梯度互不混合。</p>':'<p>统计量只在当前行内计算，同一行的所有特征复用一个 r。w 的每个特征分量则被所有行共享。</p>';
   if(picked&&mode==='input'&&['G','u','direct'].includes(picked.id))note+='<p><strong>当前单格的直接贡献只到同格 u 和直接路径；经统计量路径影响本行所有 dX。本视图不计算 dw。</strong></p>';
   box.querySelector('.mv-note').innerHTML=note;
   box.querySelectorAll('[data-rms-cell]').forEach(b=>b.onclick=()=>{const [id,r,j]=b.dataset.rmsCell.split(',');picked={id,r:Number(r),j:Number(j)};draw();box.querySelector(`[data-rms-cell="${b.dataset.rmsCell}"]`)?.focus({preventScroll:true});});
   box.querySelectorAll('[data-rms-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rmsMode===mode)));root.CourseMath.renderAll(box);
  }
  box.querySelectorAll('[data-rms-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.rmsMode;picked=null;draw();});box.querySelector('[data-rms-reset]').onclick=()=>{picked=null;draw();};draw();
 }
 function mlpFlow(c){const input=shape(c.X),mid=[...input.slice(0,-1),c.init[1]];return {input,mid,output:input.slice()};}
 function mountMlp(c){
  const host=document.getElementById('animation'),f=mlpFlow(c);
  host.innerHTML=`<section class="source-map mlp-view"><h2>MLP：返回值怎样在三个子层之间传递</h2><p>每个框是构造时创建并保留的一个子层。前向传数组，反向传回损失对上一层输出的梯度。</p><h3>前向：从左到右</h3><div class="mv-chain">${[['fc1','X','Z',f.input,f.mid],['act','Z','A',f.mid,f.mid],['fc2','A','Y',f.mid,f.output]].map(([id,a,b,sa,sb])=>`<button data-mlp="${id}"><strong>${id==='act'?'GELU · act':'Linear · '+id}</strong>${tex(`${a}\ ${dims(sa)}\\longrightarrow ${b}\ ${dims(sb)}`)}</button>`).join('')}</div><h3>反向：从右到左</h3>${tex(String.raw`\mathrm dX\ \xleftarrow{\rm fc1.backward}\ \mathrm dZ\ \xleftarrow{\rm act.backward}\ \mathrm dA\ \xleftarrow{\rm fc2.backward}\ G`)}<p class="small">反向宽度：${tex(`${dims(f.output)}\\to ${dims(f.mid)}\\to ${dims(f.mid)}\\to ${dims(f.input)}`,true)}；全部 batch/token 位置保留。</p><div class="mv-note" aria-live="polite"></div></section>`;
  const box=host.firstElementChild;
  const detail={fc1:['第一层 <code>Linear</code>：先前向、最后反向',`前向收到原输入 ${tex('X',true)}，返回扩展特征 ${tex('Z',true)}。反向收到 <code>GELU</code> 返回的 ${tex(String.raw`\mathrm dZ`,true)}，累计自己的 ${tex('W_1,b_1',true)} 梯度，再返回 ${tex(String.raw`\mathrm dX`,true)}。`,String.raw`Z_2=X_2W_1+b_1,\quad\mathrm dX_2=\mathrm dZ_2W_1^{\mathsf T},\quad\mathrm dW_1=X_2^{\mathsf T}\mathrm dZ_2`],act:['<code>GELU</code>：连接两个 <code>Linear</code> 的返回值',`前向收到 ${tex('Z',true)}，返回 ${tex('A',true)}。反向必须使用第二层 <code>Linear</code> 返回的 ${tex(String.raw`\mathrm dA`,true)}，与缓存 ${tex('Z',true)} 的局部导数相乘得到 ${tex(String.raw`\mathrm dZ`,true)}。<code>GELU</code> 没有参数梯度。`,String.raw`A=f(Z),\qquad\mathrm dZ=\mathrm dA\odot f'(Z)`],fc2:['第二层 <code>Linear</code>：最后前向、最先反向',`前向收到激活后的 ${tex('A',true)}，返回 ${tex('Y',true)}。反向最先收到外部上游 ${tex('G',true)}，累计自己的 ${tex('W_2,b_2',true)} 梯度，并把 ${tex(String.raw`\mathrm dA`,true)} 返回给 <code>GELU</code>。`,String.raw`Y_2=A_2W_2+b_2,\quad\mathrm dA_2=G_2W_2^{\mathsf T},\quad\mathrm dW_2=A_2^{\mathsf T}G_2`]};
  function select(id){const [title,text,formula]=detail[id];box.querySelector('.mv-note').innerHTML=`<h3>${title}</h3><p>${text}</p>${tex(formula)}<p class="small">子层各自保存前向缓存；MLP 负责把本次返回值交给下一次调用。新一轮梯度统计前，由外部统一调用 <code>zero_grad()</code>。</p>`;box.querySelectorAll('[data-mlp]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mlp===id)));root.CourseMath.renderAll(box);}
  box.querySelectorAll('[data-mlp]').forEach(b=>b.onclick=()=>select(b.dataset.mlp));select('fc2');
 }
 root.ModuleViews={mount:(id,c)=>id==='layer_rmsnorm'?mountRms(c):mountMlp(c)};
 if(typeof module!=='undefined')Object.assign(module.exports,{rmsRelation,mlpFlow});
})(typeof window==='undefined'?globalThis:window);

/* 注意力：完整来源对象与收缩轴；不把逐坐标公式表当作主图。 */
(function(root){'use strict';
 const ids=new Set(['att_scores','att_probs','att_weighted','att_merge','att_value_backward','att_score_backward','t_attention','t_mha','t_block','t_dropout','t_layernorm','rebuild_block']);
 const range=n=>Array.from({length:n},(_,i)=>i),mat=(r,c,f)=>range(r).map(i=>range(c).map(j=>f(i,j)));
 const tr=a=>a[0].map((_,j)=>a.map(row=>row[j]));
 const mm=(a,b)=>mat(a.length,b[0].length,(i,j)=>a[i].reduce((v,x,k)=>v+x*b[k][j],0));
 const tex=(s,inline=false)=>root.CourseMath?root.CourseMath.block(s,inline):s;
 const shape=a=>Array.isArray(a)?[a.length,...shape(a[0])]:[];
 function attentionSlice(b=0,h=0){
  const Q=mat(3,2,(i,c)=>(i+1)*(c?-.35:.4)+b*.2+h*.1),K=mat(3,2,(j,c)=>(j+1)*(c?.3:-.2)+b*.1-h*.2),V=mat(3,2,(j,c)=>(j+1)*(c?-.4:.5)+b*.3+h*.1),G=mat(3,2,(i,c)=>(i+1)*.2+(c?-.3:.1)+b*.1-h*.2);
  const S=mm(Q,tr(K)).map(row=>row.map(x=>x/Math.sqrt(2))),masked=S.map((r,i)=>r.map((x,j)=>j<=i?x:-Infinity));
  const A=masked.map(row=>{const max=Math.max(...row),e=row.map(x=>Math.exp(x-max)),sum=e.reduce((a,v)=>a+v,0);return e.map(x=>x/sum);});
  const Y=mm(A,V),dV=mm(tr(A),G),dA=mm(G,tr(V)),R=A.map((r,i)=>[r.reduce((v,x,j)=>v+x*dA[i][j],0)]),dS=A.map((r,i)=>r.map((x,j)=>x*(dA[i][j]-R[i][0]))),dQ=mm(dS,K).map(r=>r.map(x=>x/Math.sqrt(2))),dK=mm(tr(dS),Q).map(r=>r.map(x=>x/Math.sqrt(2)));
  return {Q,K,V,G,S,masked,A,Y,dV,dA,R,dS,dQ,dK};
 }
 const layouts={scores:['Q','K','S'],probabilities:['S','masked','A'],read:['A','V','Y'],value:['A','G','dV'],scoreback:['A','dA','R','dS'],query:['dS','K','dQ'],key:['dS','Q','dK']};
 const formulas={scores:String.raw`S=QK^{\mathsf T}/\sqrt d`,probabilities:String.raw`S\ \xrightarrow{j>i\ \text{置为}-\infty}\ \widetilde S\ \xrightarrow{\operatorname{softmax}_{j}}\ A`,read:String.raw`Y_{i,:}=\sum_j A_{i,j}V_{j,:}`,value:String.raw`\mathrm dV_{j,:}=\sum_i A_{i,j}G_{i,:}\quad\Longrightarrow\quad\mathrm dV=A^{\mathsf T}G`,scoreback:String.raw`R_i=\sum_j A_{i,j}\mathrm dA_{i,j},\quad\mathrm dS=A\odot(\mathrm dA-R)`,query:String.raw`\mathrm dQ_{i,:}=\sum_j\mathrm dS_{i,j}K_{j,:}/\sqrt d`,key:String.raw`\mathrm dK_{j,:}=\sum_i\mathrm dS_{i,j}Q_{i,:}/\sqrt d`};
 const titles={scores:'匹配：每个 query 与全部 key',probabilities:'排除未来，再按 key 归一化',read:'读取：每行权重组合全部内容行',value:'内容回流：收齐全部 query 的用途',scoreback:'分数回流：同一行概率互相影响',query:'提问回流：收齐本 query 的所有匹配',key:'索引回流：收齐全部 query 的匹配'};
 const names={Q:'提问',K:'索引',V:'内容',S:'匹配表',masked:'屏蔽后分数',A:'读取权重',Y:'读取结果',G:'输出上游',dV:'内容梯度',dA:'权重上游',R:'行内公共项',dS:'分数梯度',dQ:'提问梯度',dK:'索引梯度'};
 const symbols={masked:String.raw`\widetilde S`,dV:String.raw`\mathrm dV`,dA:String.raw`\mathrm dA`,dS:String.raw`\mathrm dS`,dQ:String.raw`\mathrm dQ`,dK:String.raw`\mathrm dK`};
 function selection(mode,name,r,c){
  const yes=(a,b)=>a===b,all=()=>true,row=i=>(a)=>a===i,col=j=>(a,b)=>b===j,cell=(i,j)=>(a,b)=>a===i&&b===j;
  const out={};
  // 点输出看全部来源；点输入看它影响的完整区域。颜色不代表数值大小。
  if(mode==='scores'){
   if(name==='S'){out.S=cell(r,c);out.Q=row(r);out.K=row(c);}
   else if(name==='Q'){out.Q=cell(r,c);out.K=col(c);out.S=row(r);}
   else{out.K=cell(r,c);out.Q=col(c);out.S=col(r);}
  }else if(mode==='probabilities'||mode==='scoreback'){
   for(const key of layouts[mode])out[key]=key==='R'?row(r):(a,b)=>a===r&&b<=r;
   if(mode==='probabilities'&&c>r){out.S=cell(r,c);out.masked=cell(r,c);out.A=cell(r,c);}
   if(mode==='scoreback'&&name!=='R'&&c>r){for(const key of layouts[mode])out[key]=()=>false;out[name]=cell(r,c);}
  }else if(mode==='read'){
   if(name==='Y'){out.Y=cell(r,c);out.A=(a,b)=>a===r&&b<=r;out.V=(a,b)=>a<=r&&b===c;}
   else if(name==='A'){out.A=cell(r,c);out.V=c<=r?row(c):()=>false;out.Y=c<=r?row(r):()=>false;}
   else{out.V=cell(r,c);out.A=(a,b)=>a>=r&&b===r;out.Y=(a,b)=>a>=r&&b===c;}
  }else if(mode==='value'){
   if(name==='dV'){out.dV=cell(r,c);out.A=(a,b)=>a>=r&&b===r;out.G=(a,b)=>a>=r&&b===c;}
   else if(name==='A'){out.A=cell(r,c);out.G=c<=r?row(r):()=>false;out.dV=c<=r?row(c):()=>false;}
   else{out.G=cell(r,c);out.A=(a,b)=>a===r&&b<=r;out.dV=(a,b)=>a<=r&&b===c;}
  }else if(mode==='query'){
   if(name==='dQ'){out.dQ=cell(r,c);out.dS=(a,b)=>a===r&&b<=r;out.K=(a,b)=>a<=r&&b===c;}
   else if(name==='dS'){out.dS=cell(r,c);out.K=c<=r?row(c):()=>false;out.dQ=c<=r?row(r):()=>false;}
   else{out.K=cell(r,c);out.dS=(a,b)=>a>=r&&b===r;out.dQ=(a,b)=>a>=r&&b===c;}
  }else if(mode==='key'){
   if(name==='dK'){out.dK=cell(r,c);out.dS=(a,b)=>a>=r&&b===r;out.Q=(a,b)=>a>=r&&b===c;}
   else if(name==='dS'){out.dS=cell(r,c);out.Q=c<=r?row(r):()=>false;out.dK=c<=r?row(c):()=>false;}
   else{out.Q=cell(r,c);out.dS=(a,b)=>a===r&&b<=r;out.dK=(a,b)=>a<=r&&b===c;}
  }
  return out;
 }
 function matrix(name,a,b,h,numbers=false){
  const symbol=symbols[name]||name,keys=['S','masked','A','dA','dS'];
  return `<section class="av-matrix"><h4>${tex(symbol,true)} · ${names[name]}</h4><p class="small">${tex(`(${a.length},${a[0].length})`,true)} · ${keys.includes(name)?'行 query / 列 key':name==='R'?'行 query / 公共项':['K','V','dK','dV'].includes(name)?'行 key / 列特征':'行 query / 列特征'}</p><div class="av-cells" style="--av-cols:${a[0].length}">${a.flatMap((row,i)=>row.map((v,j)=>`<button data-av-cell="${b},${h},${name},${i},${j}" aria-label="批次 ${b} 头 ${h} ${name} ${i},${j}" class="${(name==='masked'||name==='A'||name==='dS')&&j>i?'av-blocked':''}">${numbers?(Number.isFinite(v)?v.toFixed(2):'−∞'):((name==='masked'||name==='A'||name==='dS')&&j>i?name==='masked'?'−∞':'0':tex(`${symbol}_{${i},${j}}`,true))}</button>`)).join('')}</div></section>`;
 }
 function mountAttention(id){
  const host=document.getElementById('animation');let mode=id==='t_attention'?'overview':({att_scores:'scores',att_probs:'probabilities',att_weighted:'read',att_value_backward:'value',att_score_backward:'scoreback'}[id]||'scores'),numbers=false,sliceB=0,sliceH=0;
  const modes=id==='t_attention'?['overview',...Object.keys(layouts)]:[mode];
  host.innerHTML=`<section class="source-map attention-view"><h2>先看完整前向，再沿原路理解反向</h2><p>贯通示例：${tex('B=2,H=2,T_q=T_k=3,d=d_v=2',true)}。完整张量形状保留；各批次和头采用相同规则，分项默认展示一个可切换的切片。图中符号表示位置，测评数值见上方样例。</p><div class="av-path">${tex(String.raw`(Q,K)\to S\to\widetilde S\to A\quad;\quad(A,V)\to Y`)}</div><div class="av-tabs">${modes.map(v=>`<button data-av-mode="${v}">${v==='overview'?'完整流程':titles[v]}</button>`).join('')}<label><input type="checkbox" data-av-numbers>显示数值核对</label></div><div class="av-rule"></div><p class="av-hint">点任意来源或结果格子核对对应行/列；主公式始终描述全部位置共同遵循的计算。</p><div class="av-slices"></div><p class="av-selection" aria-live="polite"></p></section>`;
  const box=host.firstElementChild;
  const picker=document.createElement('div');picker.className='av-slice-picker';picker.innerHTML='<span>查看一个切片</span>'+range(2).flatMap(b=>range(2).map(h=>`<button type="button" data-av-pick="${b},${h}">${tex(`b=${b},h=${h}`,true)}</button>`)).join('');box.querySelector('.av-slices').before(picker);
  function draw(){
   box.querySelector('.av-selection').textContent='';
   box.querySelectorAll('[data-av-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.avMode===mode)));
   box.querySelector('[data-av-numbers]').parentElement.hidden=mode==='overview';
   picker.hidden=mode==='overview';
   picker.querySelectorAll('[data-av-pick]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.avPick===`${sliceB},${sliceH}`)));
   if(mode==='overview'){
    const node=(name,action,shape)=>`<section><h4>${tex(name,true)}</h4><p>${action}</p><small>${tex(shape,true)}</small></section>`;
    box.querySelector('.av-rule').innerHTML='<h3>同一个前向方法：匹配 → 排除 → 分配权重 → 读取</h3>';
    box.querySelector('.av-slices').innerHTML=`<div class="av-whole"><div class="av-whole-flow">${node('Q,K','输入提问与索引','(2,2,3,2)')}${node('S','每行提问与所有索引点积，再缩放','(2,2,3,3)')}${node(String.raw`\widetilde S`,'未来位置改为负无穷','(2,2,3,3)')}${node('A','沿 key 轴归一化，得到读取权重','(2,2,3,3)')}${node('Y=AV','另一输入 V 提供内容；按权重组合整行','(2,2,3,2)')}</div><h3>同一个反向方法：先分两路，再把匹配梯度送回两侧</h3><div class="av-whole-back"><section><h4>读取结果收到上游</h4>${tex(String.raw`G=\partial L/\partial Y`)}</section><section><h4>内容直接收集读取用途</h4><p>对应前向</p>${tex(String.raw`Y=AV`)}${tex(String.raw`\mathrm dV=A^{\mathsf T}G`)}</section><section><h4>权重经过概率回到分数</h4><p>对应前向</p>${tex(String.raw`Y=AV,\quad A=\operatorname{softmax}(\widetilde S)`)}${tex(String.raw`\mathrm dA=GV^{\mathsf T}\to\mathrm dS`)}</section><section><h4>分数再回到提问和索引</h4><p>对应前向</p>${tex(String.raw`S=QK^{\mathsf T}/\sqrt d`)}${tex(String.raw`\mathrm dQ=\mathrm dSK/\sqrt d`)}${tex(String.raw`\mathrm dK=\mathrm dS^{\mathsf T}Q/\sqrt d`)}</section></div></div>`;
    box.querySelector('.av-hint').textContent='各中间量都属于这一个模块；上方分项按钮只用于核对来源，不切换任务或代码。';root.CourseMath.renderAll(box);return;
   }
   box.querySelector('.av-hint').textContent='点任意来源或结果格子核对对应行/列；主公式描述全部位置共同遵循的计算。';
   const forward={value:String.raw`Y=AV,\quad Y_{i,c}=\sum_jA_{i,j}V_{j,c}`,scoreback:String.raw`A=\operatorname{softmax}_{j}(\widetilde S),\quad A_{i,j}=\frac{e^{\widetilde S_{i,j}}}{\sum_ke^{\widetilde S_{i,k}}}`,query:String.raw`S=QK^{\mathsf T}/\sqrt d,\quad S_{i,j}=\sum_cQ_{i,c}K_{j,c}/\sqrt d`,key:String.raw`S=QK^{\mathsf T}/\sqrt d,\quad S_{i,j}=\sum_cQ_{i,c}K_{j,c}/\sqrt d`}[mode];
   box.querySelector('.av-rule').innerHTML=`<h3>${titles[mode]}</h3>${forward?'<p class="av-forward-label">对应的前向函数</p>'+tex(forward)+'<p class="av-forward-label">沿该函数回传梯度</p>':''}${tex(formulas[mode])}<p>${mode==='value'?'同一内容行被许多 query 读取。每条回流都乘前向读取权重，再沿 query 相加；这是转置权重表的来源。':mode==='query'||mode==='key'?'匹配分数同时依赖提问和索引；每个对象收齐所有用到自己的匹配，局部系数是另一侧向量与缩放。':mode==='scoreback'?'先把每行全部概率和上游相乘求和，得到一个公共项，再广播回该行；不同 query 的归一化互不影响。':mode==='probabilities'?'灰色是未来区域。它在分数阶段被排除，随后指数为零；允许的每行权重和为一。':mode==='scores'?'整张表每个格子取一行提问与一行索引。特征轴求和，query/key 两轴保留。':'同一行权重作用于各个内容行，沿 key 求和；输出保留 query 与内容特征。'}</p>`;
   {const b=sliceB,h=sliceH,d=attentionSlice(b,h);box.querySelector('.av-slices').innerHTML=`<article class="av-slice" data-av-slice="${b},${h}"><h3>${tex(`b=${b},h=${h}`,true)}</h3><div class="av-objects">${layouts[mode].map(n=>matrix(n,d[n],b,h,numbers)).join('')}</div></article>`;}
   box.querySelectorAll('[data-av-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.avMode===mode)));
   box.querySelectorAll('[data-av-cell]').forEach(button=>button.onclick=()=>{
    const [b,h,n,r,c]=button.dataset.avCell.split(','),rel=selection(mode,n,+r,+c);
    box.querySelectorAll('[data-av-cell]').forEach(el=>{const [eb,eh,en,er,ec]=el.dataset.avCell.split(',');el.classList.toggle('av-source',eb===b&&eh===h&&!!rel[en]?.(+er,+ec));el.setAttribute('aria-pressed',String(el===button));});
    box.querySelector('.av-selection').innerHTML=`当前核对 ${tex(`b=${b},h=${h}`,true)} 的 ${tex(`${symbols[n]||n}_{${r},${c}}`,true)}；蓝色标出对应来源/去向，其他批次和头独立。${mode==='probabilities'&&+c>+r?'该未来分数已屏蔽，原分数的变化不会传到任何概率。':''}`;root.CourseMath.renderAll(box.querySelector('.av-selection'));
   });root.CourseMath.renderAll(box);
  }
  picker.querySelectorAll('[data-av-pick]').forEach(b=>b.onclick=()=>{[sliceB,sliceH]=b.dataset.avPick.split(',').map(Number);draw();});
  box.querySelectorAll('[data-av-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.avMode;draw();});box.querySelector('[data-av-numbers]').onchange=e=>{numbers=e.target.checked;draw();};draw();
 }
 function headMapping(b,t,h,c,H=2,d=2){return {source:[b,h,t,c],target:[b,t,h*d+c]};}
 function mountMhaFlow(){
  const host=document.getElementById('animation');
  const stages=[
   ['输入投影','X','Z',String.raw`Z=\operatorname{qkv}(X)`,'(2,3,4)','(2,3,12)','每个 token 的四个特征，生成十二个投影特征。',String.raw`\nabla X=\operatorname{qkv.backward}(\nabla Z)`],
   ['分成三组','Z','(Q_0,K_0,V_0)',String.raw`(Q_0,K_0,V_0)=\operatorname{split}(Z)`,'(2,3,12)',String.raw`\text{各 }(2,3,4)`,'沿最后轴依次取提问、索引、内容三段，每段四个特征。',String.raw`\nabla Z=\operatorname{concat}(\nabla Q_0,\nabla K_0,\nabla V_0)`],
   ['每组拆出头轴','(Q_0,K_0,V_0)','(Q_r,K_r,V_r)',String.raw`Q_r=\operatorname{reshape}(Q_0)`,String.raw`\text{各 }(2,3,4)`,String.raw`\text{各 }(2,3,2,2)`,'每组的四个特征分成两头，每头两个。公式以提问为例，另两组相同。',String.raw`\nabla Q_0=\operatorname{reshape}(\nabla Q_r)`],
   ['把头轴移到前面','(Q_r,K_r,V_r)','(Q,K,V)',String.raw`Q=\operatorname{transpose}_{(0,2,1,3)}(Q_r)`,String.raw`\text{各 }(2,3,2,2)`,String.raw`\text{各 }(2,2,3,2)`,'轴从批次、token、头、特征，变成批次、头、token、特征；另两组相同。',String.raw`\nabla Q_r=\operatorname{transpose}_{(0,2,1,3)}(\nabla Q)`],
   ['各头读取上下文','(Q,K,V)','O',String.raw`O=\operatorname{core}(Q,K,V)`,String.raw`\text{各 }(2,2,3,2)`,'(2,2,3,2)','复用 AttentionCore：打分、屏蔽未来、Softmax、加权读取。每个头都保留三个 token。',String.raw`(\nabla Q,\nabla K,\nabla V)=\operatorname{core.backward}(\nabla O)`],
   ['把同一token的头放相邻','O','P',String.raw`P=\operatorname{transpose}_{(0,2,1,3)}(O)`,'(2,2,3,2)','(2,3,2,2)','将 token 轴移回头轴前面；同一 token 的两段结果放在一起。',String.raw`\nabla O=\operatorname{transpose}_{(0,2,1,3)}(\nabla P)`],
   ['合成一行特征','P','M',String.raw`M=\operatorname{reshape}(P)`,'(2,3,2,2)','(2,3,4)','合并最后的头轴和头内特征轴，每个 token 回到四个特征。',String.raw`\nabla P=\operatorname{reshape}(\nabla M)`],
   ['输出投影','M','Y',String.raw`Y=\operatorname{proj}(M)`,'(2,3,4)','(2,3,4)','学习混合两个头提供的信息，得到最终输出。',String.raw`\nabla M=\operatorname{proj.backward}(G),\quad G=\nabla Y`]
  ];
  host.innerHTML=`<section class="source-map attention-view mha-flow-view"><h2>多头注意力：从输入到输出，逐步接起来</h2><p>${tex('B=2,T=3,C=4,H=2,d=2',true)}。先沿完整流程看每一步接收和交出什么，再看下方特征片段如何换分组。</p><div class="av-tabs"><button data-mha-direction="forward">完整前向</button><button data-mha-direction="backward">反向逐步对应</button></div><div class="mha-stages"></div><h3>布局放大：每个 token 的两段特征去了哪里</h3><p>每块代表两个特征，块上直接写明 token 和头。拆合头只改变分组与轴位置，所有片段都保留。</p><div class="av-tabs"><button data-mha-layout="split">拆头：投影后怎样分给各头</button><button data-mha-layout="merge">合头：各头结果怎样回到token</button><label>查看批次 <select data-mha-batch aria-label="布局示例批次"><option value="0">0</option><option value="1">1</option></select></label></div><div class="mha-layout"></div><p class="mha-layout-note" aria-live="polite"></p></section>`;
  const box=host.firstElementChild;let direction='forward',layout='split',batch=0,picked=null;
  function drawStages(){
   const backward=direction==='backward',order=backward?stages.map((s,i)=>[s,i]).reverse():stages.map((s,i)=>[s,i]);
   box.querySelector('.mha-stages').innerHTML=`<p>${backward?'从输出上游开始，沿同一条前向链逆序返回；每行同时列出对应前向。第三、四步以提问为例，索引和内容执行相同的布局恢复。':`从 ${tex('X',true)} 开始，每行的输出就是下一行的输入。带 ${tex('r',true)} 下标的名字表示刚拆出头轴的布局，${tex('P',true)} 表示合头前换好轴的结果。`}</p><ol>${order.map(([s,i])=>`<li data-mha-stage="${i+1}"><span class="mha-stage-number">${i+1}</span><div><h4>${s[0]}</h4>${backward?`<div class="mha-forward-recall"><span>对应前向</span>${tex(s[3])}</div>${tex(s[7])}`:`<div class="mha-stage-io">${tex(String.raw`${s[1]}:${s[4]}\ \longrightarrow\ ${s[2]}:${s[5]}`)}</div>${tex(s[3])}<p>${s[6].replace('AttentionCore','<code>AttentionCore</code>').replace('Softmax','<code>Softmax</code>')}</p>`}</div></li>`).join('')}</ol>${backward?'<p>布局步骤把梯度放回原来源格。三组梯度先拼回投影的不同输出列，再由 <code>qkv.backward</code> 汇回共享输入；两层 <code>Linear</code> 各自累计所有批次、token 的参数梯度。</p>':''}`;
   box.querySelectorAll('[data-mha-direction]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mhaDirection===direction)));root.CourseMath.renderAll(box.querySelector('.mha-stages'));
  }
  function drawLayout(){
   const chunk=(t,h)=>`<button class="mha-chunk av-head-${h}" data-mha-chunk="${t},${h}" aria-pressed="${picked===`${t},${h}`}" aria-label="批次${batch} token${t} 头${h} 的两个特征"><strong>token ${t}</strong><span>头 ${h} · 两个特征</span></button>`;
   const tokenView=`<section><h4>${layout==='split'?'投影后，每个token有四个特征':'合回后，每个token有四个特征'}</h4>${tex(layout==='split'?'Q_0:(2,3,4)':'M:(2,3,4)',true)}${range(3).map(t=>`<div class="mha-token-row"><span>token ${t}</span>${range(2).map(h=>chunk(t,h)).join('')}</div>`).join('')}</section>`;
   const headView=`<section><h4>${layout==='split'?'拆头后，每个头有完整序列':'核心输出，每个头有完整序列'}</h4>${tex(layout==='split'?'Q:(2,2,3,2)':'O:(2,2,3,2)',true)}${range(2).map(h=>`<div class="mha-head-group"><strong>头 ${h}</strong><div>${range(3).map(t=>chunk(t,h)).join('')}</div></div>`).join('')}</section>`;
   box.querySelector('.mha-layout').innerHTML=`${layout==='split'?tokenView:headView}<div class="mha-layout-arrow">${layout==='split'?'分组，再交换轴 →':'交换轴，再合并 →'}</div>${layout==='split'?headView:tokenView}`;
   box.querySelector('.mha-layout-note').innerHTML=picked?`批次 ${batch}、token ${picked.split(',')[0]}、头 ${picked.split(',')[1]} 的两个特征，前后始终是同一块；没有与其他 token 混合。`:`当前显示批次 ${batch}；另一批次遵循同一规则，完整形状中的两个批次均保留。${layout==='split'?'以提问为例，索引和内容执行相同拆头；每个头都保留三个 token。':'先把同一 token 的头 0、头 1 放相邻，再合并成四个特征；后续输出投影才会学习混合各头信息。'}`;
   box.querySelectorAll('[data-mha-chunk]').forEach(b=>b.onclick=()=>{picked=b.dataset.mhaChunk;drawLayout();});box.querySelectorAll('[data-mha-layout]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mhaLayout===layout)));root.CourseMath.renderAll(box.querySelector('.mha-layout'));
  }
  box.querySelectorAll('[data-mha-direction]').forEach(b=>b.onclick=()=>{direction=b.dataset.mhaDirection;drawStages();});box.querySelectorAll('[data-mha-layout]').forEach(b=>b.onclick=()=>{layout=b.dataset.mhaLayout;picked=null;drawLayout();});box.querySelector('[data-mha-batch]').onchange=e=>{batch=Number(e.target.value);picked=null;drawLayout();};drawStages();drawLayout();
 }
 function mountHeads(id){
  if(id==='t_mha')return mountMhaFlow();
  const host=document.getElementById('animation');
  host.innerHTML=`<section class="source-map attention-view"><h2>同一 token 的片段怎样合回</h2><p>${tex('B=2,T=3,C=4,H=2,d=2',true)}；每个头占两个特征。点击任意片段，在所有布局中标出同一来源。</p>${tex(String.raw`(B,H,T,d)\xrightarrow{\rm transpose}(B,T,H,d)\xrightarrow{\rm reshape}(B,T,Hd)`)}<div class="av-heads"></div><div class="av-head-note"></div></section>`;
  const box=host.firstElementChild;
  box.querySelector('.av-heads').innerHTML=range(2).map(b=>`<article><h3>${tex(`b=${b}`,true)}</h3><div class="av-head-columns">${['头为外层：'+tex('(H,T,d)',true),'token 为外层：'+tex('(T,H,d)',true),'同一 token 拼接：'+tex('(T,Hd)',true)].map((title,layout)=>`<section><h4>${title}</h4>${range(layout===0?2:3).map(outer=>`<div class="av-head-row">${tex(layout===0?`h=${outer}`:`t=${outer}`,true)}${range(layout===0?3:2).map(inner=>range(2).map(c=>{const h=layout===0?outer:inner,t=layout===0?inner:outer;return `<button class="av-head-${h}" data-av-head="${b},${t},${h},${c}">${tex(`x_{${t},${h},${c}}`,true)}</button>`;}).join('')).join('')}</div>`).join('')}</section>`).join('')}</div></article>`).join('');
  box.querySelectorAll('[data-av-head]').forEach(el=>el.onclick=()=>{const [b,t,h,c]=el.dataset.avHead.split(',').map(Number);box.querySelectorAll('[data-av-head]').forEach(v=>v.setAttribute('aria-pressed',String(v.dataset.avHead===el.dataset.avHead)));box.querySelector('.av-head-note').innerHTML=tex(String.raw`(${b},${h},${t},${c})\longleftrightarrow(${b},${t},${h},${c})\longleftrightarrow(${b},${t},${2*h+c})`);root.CourseMath.renderAll(box.querySelector('.av-head-note'));});root.CourseMath.renderAll(box);
 }
 function residualFlow(g,branch2,branch1){const g1=g+branch2;return {g1,dx:g1+branch1};}
 function mountBlock(){
  if(document.querySelector('.block-macro')){document.getElementById('animation').innerHTML='';return;}
  const host=document.getElementById('animation');host.innerHTML=`<section class="source-map attention-view"><h2>两次残差：前向分叉，反向合流</h2><p>每个框代表完整 ${tex('(B,T,C)',true)} 张量，两个批次和所有 token 一起经过相同结构。</p><div class="av-residuals">${[1,2].map(i=>`<button data-av-residual="${i}"><span>直连路径</span><strong>${tex(i===1?'X':'X_1',true)} ─────────────→ ＋ → ${tex(i===1?'X_1':'Y',true)}</strong><span>↘ ${i===1?'<code>norm1 → attn</code>':'<code>MLPBranch</code>（<code>norm2 → mlp</code>）'} ↗</span></button>`).join('')}</div><div class="av-residual-note"></div></section>`;
  const box=host.firstElementChild;
  const select=i=>{box.querySelectorAll('[data-av-residual]').forEach(v=>v.setAttribute('aria-pressed',String(+v.dataset.avResidual===i)));box.querySelector('.av-residual-note').innerHTML=i===2?`<h3>第二次加法先返回</h3><p>对应前向</p>${tex(String.raw`Y=X_1+\operatorname{MLPBranch}(X_1)`)}<p>输出上游同时送给直连和完整特征分支。直连直接返回上游；分支按自己的逆序返回另一份输入梯度。两份相加才是第一条残差收到的上游。</p>${tex(String.raw`G_1=\underbrace{G}_{\text{直连}}+\underbrace{\operatorname{MLPBranch.backward}(G)}_{\text{分支}}`)}`:`<h3>第一次加法接收已经合流的上游</h3><p>对应前向</p>${tex(String.raw`X_1=X+\operatorname{attn}(\operatorname{norm1}(X))`)}<p>这一次两条路径接收的都是第二次加法返回的完整梯度，分别直返与经过注意力/归一化后相加。子层负责自己的参数梯度。</p>${tex(String.raw`\mathrm dX=\underbrace{G_1}_{\text{直连}}+\underbrace{\operatorname{norm1.backward}(\operatorname{attn.backward}(G_1))}_{\text{分支}}`)}`;root.CourseMath.renderAll(box);};box.querySelectorAll('[data-av-residual]').forEach(v=>v.onclick=()=>select(+v.dataset.avResidual));select(2);
 }
 function mountComparison(id){
  const host=document.getElementById('animation'),drop=id==='t_dropout';
  host.innerHTML=`<section class="source-map attention-view"><h2>${drop?'同一掩码控制前向与反向':'先移动到均值为零，再调整整行尺度'}</h2>${drop?'<p>确定的关系示例；不是 NumPy 随机序列回放。绿色保留，灰色阻断。全部批次/token 同时显示。</p>':`<p>全部 ${tex('(B,T,C)=(2,3,4)',true)} 保留。蓝色是正值，橙色是负值；先减去本行均值，原来全正的输入会分布到零两侧。再把整行乘相同倒标准差。示例使用 ${tex('w=1,b=0',true)}。</p>${tex(String.raw`X\xrightarrow{\text{减本行 }\mu}Z\xrightarrow{\times R}H\xrightarrow{\odot w+b}Y`)}`}<div class="av-comparison">${range(2).flatMap(b=>range(3).map(t=>drop?`<div>${tex(`(b,t)=(${b},${t})`,true)}${range(4).map(c=>`<span class="${(b+t+c)%3===0?'av-blocked':'av-keep'}">${tex(`x_{${c}}\ ${((b+t+c)%3===0)?String.raw`\xrightarrow{M=0}0`:String.raw`\xrightarrow{M=1/(1-p)}x_c/(1-p)`}`,true)}<br>${tex((b+t+c)%3===0?String.raw`G_c\to0`:String.raw`G_c\to G_c/(1-p)`,true)}</span>`).join('')}</div>`:normRow(b,t))).join('')}</div>${drop?tex(String.raw`\text{评估： }M=1,\quad X\to X,\quad G\to G`):`<p>这里每行只使用自己的统计量；最后的缩放和偏置则由全部位置共享，所以参数梯度沿同一特征列汇总。</p>${tex(String.raw`\mathrm dw_c=\sum_{b,t}G_{b,t,c}H_{b,t,c},\qquad\mathrm db_c=\sum_{b,t}G_{b,t,c}`)}`}</section>`;root.CourseMath.renderAll(host);
 }
 function normData(b,t){const X=[1+b+t,2+b+2*t,4+2*b+t,7+b+t],mu=X.reduce((a,v)=>a+v)/4,Z=X.map(v=>v-mu),R=1/Math.sqrt(Z.reduce((a,v)=>a+v*v)/4+1e-5);return {X,mu,Z,R,H:Z.map(v=>v*R)};}
 function normRow(b,t){const d=normData(b,t),strip=(a,label)=>`<span class="av-norm-strip"><strong>${tex(label,true)}</strong><span>${a.map(v=>`<i class="${v<0?'av-neg':'av-pos'}">${v.toFixed(2)}</i>`).join('')}</span></span>`;return `<div>${tex(`(${b},${t})`,true)}${strip(d.X,'X')}<span>减本行 ${tex(String.raw`\mu=${d.mu.toFixed(2)}`,true)} →</span>${strip(d.Z,'Z')}<span>乘 ${tex(`R=${d.R.toFixed(2)}`,true)} →</span>${strip(d.H,'H=Y')}</div>`;}
 root.AttentionViews={supports:id=>ids.has(id),mount:id=>['t_block','rebuild_block'].includes(id)?mountBlock():['att_merge','t_mha'].includes(id)?mountHeads(id):['t_dropout','t_layernorm'].includes(id)?mountComparison(id):mountAttention(id)};
 if(typeof module!=='undefined')Object.assign(module.exports,{attentionSlice,selection,headMapping,residualFlow,normData});
})(typeof window==='undefined'?globalThis:window);

/* 概率用行内敏感度，损失用标签选择路径，组合任务用持久子层。 */
(function(root){'use strict';
 const range=n=>Array.from({length:n},(_,i)=>i),shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
 const coords=s=>s.length?range(s[0]).flatMap(i=>coords(s.slice(1)).map(p=>[i,...p])):[[]];
 const tex=(s,inline=false)=>root.CourseMath.block(s,inline);
 function probabilityData(c){
  const x=c.input.X,s=shape(x),V=s.at(-1),positions=coords(s.slice(0,-1));
  const rows=x.flat(s.length-2),g=c.input.G?.flat(s.length-2)||rows.map(a=>a.map(()=>1));
  const p=rows.map(a=>{const e=a.map(v=>Math.exp(v-Math.max(...a))),sum=e.reduce((a,b)=>a+b,0);return e.map(v=>v/sum);});
  const target=c.input.targets?.flat(Infinity);
  const logp=rows.map(a=>{const z=a.map(v=>v-Math.max(...a)),logsum=Math.log(z.reduce((s,v)=>s+Math.exp(v),0));return z.map(v=>v-logsum);});
  return {s,V,positions,rows,p,g,target,logp,N:rows.length};
 }
 function probabilityRelation(data,row,j,loss=false,g=1){
  const p=data.p[row],up=data.g[row],y=data.target?.[row];
  if(loss)return {local:p.map((v,k)=>v-(k===y?1:0)),dx:p.map((v,k)=>g/data.N*(v-(k===y?1:0))),loss:-data.logp.reduce((s,a,n)=>s+a[data.target[n]],0)/data.N};
  const coefficients=p.map((v,i)=>v*((i===j?1:0)-p[j]));
  return {coefficients,contributions:coefficients.map((v,i)=>v*up[i]),dx:coefficients.reduce((s,v,i)=>s+v*up[i],0)};
 }
 function vector(values,symbol,selected=-1,colors=null){return `<div class="pv-vector" aria-label="${symbol}">${values.map((v,i)=>`<span class="pv-value ${i===selected?'pv-selected':''} ${colors?.[i]||''}" title="${Number(v).toPrecision(6)}">${tex(`${symbol}_{${i}}`,true)}<small>${Number(v).toFixed(3)}</small></span>`).join('')}</div>`;}
 function probabilityOverview(d,loss,backward){
  if(loss&&!backward)return lossGatherDiagram(d);
  const full='('+d.s.join(',')+')',reduced='('+[...d.s.slice(0,-1),1].join(',')+')';
  let steps,formula,note,headers,row;
  if(!loss&&!backward){
   steps=[['X','完整分数',full],['Z=X-\\max_V X','逐位置减本行最大值',full],['E=\\exp Z','全部类别逐元素指数化',full],['P=E/\\sum_V E','本行求和，广播相除',full]];
   formula=String.raw`S=\sum_V E\;:${'('+[...d.s.slice(0,-1),1].join(',')+')'},\qquad P=E/S`;
   note='每一行是一处完整的 token 特征。只沿类别轴归一化；batch 和 token 轴保留，各行互不混合。';
   headers=['完整位置','输入的一整行','归约类别轴','整行输出'];
   row=(co)=>[String.raw`X_{${co},:}`,String.raw`S_{${co},0}=\sum_j E_{${co},j}`,String.raw`P_{${co},:}=E_{${co},:}/S_{${co},0}`];
  }else if(!loss){
   steps=[['P\\odot G','所有位置逐元素配对',full],['A=\\sum_V(P\\odot G)','每个位置得到一个汇总量',reduced],['G-A','向本行所有类别广播减法',full],['\\mathrm dX=P\\odot(G-A)','返回完整输入梯度',full]];
   formula=String.raw`\mathrm dX=P\odot\left(G-\sum_V(P\odot G)\right)`;
   note='一个分数影响同一行所有概率，所以先收集整行上游的加权和。这个汇总量广播回原行，再与缓存概率相乘；全部位置同时使用同一规则。';
   headers=['完整位置','整行概率与上游','行内汇总成一个量','广播回整行梯度'];
   row=co=>[String.raw`P_{${co},:}\odot G_{${co},:}`,String.raw`A_{${co},0}=\sum_jP_{${co},j}G_{${co},j}`,String.raw`\mathrm dX_{${co},:}=P_{${co},:}\odot(G_{${co},:}-A_{${co},0})`];
  }else if(!backward){
   steps=[['X','所有位置的类别分数',full],['\\log P=Z-\\log\\sum_Ve^Z','稳定计算完整对数概率',full],['r=-\\operatorname{gather}(\\log P,\\mathrm{targets})','每个标签从自己那一行取一类','('+d.s.slice(0,-1).join(',')+')'],['L=\\operatorname{mean}(r)','全部位置汇总为一个平均损失','()']];
   formula=String.raw`L=-\frac1N\sum_{n=0}^{N-1}\log P_{n,y_n},\qquad N=${d.N}`;
   note='先为所有位置计算完整对数概率，再按每个位置自己的标签同时取值。得到每处一个损失后，对全部位置平均；类别选择和位置平均是两个不同操作。';
   headers=['完整位置','本位置的标签','从本行取出的正确类','该位置进入全局平均'];
   row=(co,n)=>[String.raw`y_{${co}}=${d.target[n]}`,String.raw`(\log P)_{${co},${d.target[n]}}`,String.raw`r_{${co}}/N\longrightarrow L`];
  }else{
   steps=[['P=\\exp(\\log P)','缓存的完整概率',full],['P-Q','所有位置在各自正确类减一',full],['g/N','损失上游与平均系数','()'],['\\mathrm dX=(g/N)(P-Q)','同一系数乘全部位置和类别',full]];
   formula=String.raw`Q_{n,j}=\mathbf1[j=y_n],\qquad\mathrm dX=\frac gN(P-Q)`;
   note='全局标量损失向每个位置传回同一个平均系数；每一行的标签决定该行哪里减一。所有类别都有概率项，整张梯度保留全部 batch/token。标签指示张量用于理解，实现时直接按标签索引减一。';
   headers=['完整位置','这一整行的概率','本行正确类减一','乘统一系数，回到原位置'];
   row=(co,n)=>[String.raw`P_{${co},:}`,String.raw`Q_{${co},j}=\mathbf1[j=${d.target[n]}]`,String.raw`\mathrm dX_{${co},:}=\frac gN(P_{${co},:}-Q_{${co},:})`];
  }
  return `<div class="pv-overview-flow">${steps.map(([f,t,s],i)=>`<section><small>步骤 ${i+1}</small><div>${tex(f)}</div><p>${t}</p><small>${tex(s,true)}</small></section>`).join('')}</div><p>${note}</p>${tex(formula)}<details class="pv-symbol-reference"><summary>位置符号对照（可选）</summary><div class="pv-overview-table"><table><thead><tr>${headers.map(t=>`<th>${t}</th>`).join('')}</tr></thead><tbody>${d.positions.map((p,n)=>{const co=p.join(',');return `<tr data-pv-position="${co}"><th>${tex(`(${co})`,true)}</th>${row(co,n).map(f=>`<td>${tex(f,true)}</td>`).join('')}</tr>`;}).join('')}</tbody></table></div></details>`;
 }
 function lossGatherDiagram(d){
  const T=d.s.at(-2),groups=Math.ceil(d.N/T),cell=(n,value,kind)=>`<button type="button" data-gather-row="${n}" class="gather-picked-value" aria-label="位置 ${d.positions[n].join(',')} ${kind}">${value}</button>`;
  const compact=(negative)=>`<div class="gather-small-grid" style="grid-template-columns:auto repeat(${T},minmax(46px,1fr))"><span></span>${range(T).map(t=>`<small>位置 ${t}</small>`).join('')}${range(groups).map(b=>`<small>序列 ${b}</small>`+range(T).map(t=>{const n=b*T+t;return cell(n,negative?(-d.logp[n][d.target[n]]).toFixed(2):d.target[n],negative?'损失项':'标签');}).join('')).join('')}</div>`;
  const selected=d.logp.map((a,n)=>-a[d.target[n]]),mean=selected.reduce((a,b)=>a+b,0)/d.N;
  return `<div class="gather-diagram"><p>每个位置有 ${d.V} 个候选类别。标签给出要取的列号：绿色格子被取出，其他类别留在原矩阵中。</p><section class="gather-labels"><h3>① 输入标签：每个位置一个列号</h3>${compact(false)}</section><div class="gather-operation">↓ 用标签定位本行的列</div><section><h3>② 完整对数概率 ${tex(String.raw`\log P`,true)}：每行取绿色的一格</h3><div class="gather-batches">${range(groups).map(b=>`<div class="gather-batch"><strong>序列 ${b}</strong><div class="gather-source-grid" style="grid-template-columns:auto repeat(${d.V},minmax(48px,1fr))"><span></span>${range(d.V).map(j=>`<small>类别 ${j}</small>`).join('')}${range(T).map(t=>{const n=b*T+t;return `<small data-pv-position="${d.positions[n].join(',')}">位置 ${t}</small>`+d.logp[n].map((v,j)=>`<span class="gather-source ${j===d.target[n]?'gather-selected':''}" ${j===d.target[n]?`data-gather-row="${n}"`:''}>${v.toFixed(2)}</span>`).join('');}).join('')}</div></div>`).join('')}</div></section><div class="gather-operation">↓ 每行选中的格子取负号，回到原来的序列与位置</div><section><h3>③ 得到全部 ${d.N} 个损失项</h3>${compact(true)}</section><div class="gather-operation">↓ 全部相加，再除以 ${d.N}</div><section class="gather-mean"><h3>④ 汇总为一个平均损失</h3>${tex(`L=\\frac{${selected.map(v=>v.toFixed(2)).join('+')}}{${d.N}}\\approx ${mean.toFixed(2)}`)}<p class="small">数值仅用于辨认来源，显示有舍入。点击任一标签或损失项，可同时框出它对应的原矩阵格子。</p></section></div>`;
 }
 function mountProbability(id,c){
  const d=probabilityData(c),loss=id==='t_loss';let n=0,j=loss?d.target[0]:0,upstream=1;
  const host=document.getElementById('animation');host.innerHTML=`<section class="source-map probability-view"><h2>${loss?'交叉熵整体：标签选择、损失汇总与反向':'整体：完整张量的逐行归一化与反向'}</h2><p class="small">${tex(`X:${'('+d.s.join(',')+')'},\\quad N=${d.N}`,true)}；下方保留所有位置，每行表示完整类别向量。</p><div class="pv-overview-modes"><button data-pv-whole="forward">看前向整体</button><button data-pv-whole="backward">看反向整体</button></div><div class="pv-overview"></div><details class="pv-detail"><summary>单个位置与类别的来源核对（可选）</summary><div class="pv-positions"></div><div class="pv-content"></div></details></section>`;
  const box=host.firstElementChild;
  function whole(backward){box.querySelector('.pv-overview').innerHTML=probabilityOverview(d,loss,backward);box.querySelectorAll('[data-pv-whole]').forEach(b=>b.setAttribute('aria-pressed',String((b.dataset.pvWhole==='backward')===backward)));box.querySelectorAll('[data-gather-row]').forEach(b=>b.onclick=()=>{box.querySelectorAll('[data-gather-row]').forEach(e=>e.classList.toggle('gather-focus',e.dataset.gatherRow===b.dataset.gatherRow));});root.CourseMath.renderAll(box);}
  box.querySelectorAll('[data-pv-whole]').forEach(b=>b.onclick=()=>whole(b.dataset.pvWhole==='backward'));whole(false);
  function draw(){
   box.querySelector('.pv-positions').innerHTML=d.positions.map((co,i)=>`<button data-prow="${i}" aria-pressed="${i===n}">${tex(`(${co.join(',')})`,true)}${loss?tex(`\\to y=${d.target[i]}\\to\\ell_{${i}}`,true):''}</button>`).join('');
   const result=probabilityRelation(d,n,j,loss,upstream),p=d.p[n];
   let html=`<h3>位置 ${tex(`(${d.positions[n].join(',')})`,true)}</h3>`;
   if(!loss){
    const color=result.coefficients.map(v=>v>0?'pv-positive':v<0?'pv-negative':'pv-zero');
    html+=`<p>点击类别，固定检查它的分数。蓝色表示正影响，橙色表示负影响；零影响单独标出。</p><div class="pv-categories">${range(d.V).map(k=>`<button data-pcat="${k}" aria-pressed="${j===k}">${tex(`x_{${k}}`,true)}</button>`).join('')}</div><h3>前向影响：提高所选分数</h3>${vector(d.rows[n],'x',j)}${vector(p,'p',-1,color)}<div class="pv-signs">${result.coefficients.map((v,i)=>`<span class="${color[i]}">${tex(`\\frac{\\partial p_${i}}{\\partial x_${j}}`,true)}：${v>0?'增加':v<0?'减少':'不变'}</span>`).join('')}</div>${tex(String.raw`\frac{\partial p_i}{\partial x_j}=p_i(\delta_{ij}-p_j)`)}<h3>反向来源：这行每个输出都有一条回流</h3><p>所选分数影响整行概率，因此这一行所有输出的上游，都要乘各自局部导数后相加。公共分母使其他类别也参与；其他位置没有这条路径。</p>${vector(d.g[n],'G')}${tex(`\\mathrm dx_{${j}}=\\sum_iG_i\\underbrace{p_i(\\delta_{i,${j}}-p_{${j}})}_{\\text{前向局部导数}}`)}${tex(String.raw`\mathrm dX=P\odot\left(G-\sum_iG_iP_i\right)`)}<details><summary>展开这行的系数与贡献核对</summary>${vector(result.coefficients,'J')}${vector(result.contributions,'GJ')}${tex(`\\mathrm dx_${j}=${result.dx.toFixed(6)}`)}</details>`;
   }else{
    const y=d.target[n],one=range(d.V).map(k=>k===y?1:0);
    html+=`<p>本行标签 ${tex(`y=${y}`,true)} 只选择正确类的对数概率；所有位置的负对数概率共同平均。</p>${vector(d.logp[n],String.raw`\log p`,y)}${tex(`\\ell_{${n}}=-\\log p_{${y}},\\qquad L=\\frac1{${d.N}}\\sum_{n=0}^{${d.N-1}}\\ell_n`)}<p class="small">上方每个位置都对同一损失贡献 ${tex(`1/${d.N}`,true)}；当前选择只用于查看来源。</p><h3>反向：概率减标签，再乘统一系数</h3><div class="pv-upstream">标量上游 ${[-2,0,1].map(v=>`<button data-pg="${v}" aria-pressed="${upstream===v}">${tex(`g=${v}`,true)}</button>`).join('')}</div><p>对数归一化项传回整行概率，正确类分数项在标签位置减一。这个局部结果再乘损失平均系数与标量上游。</p><div class="pv-loss-chain"><section><h4>${tex('P',true)}</h4>${vector(p,'p',y)}</section><section><h4>${tex(String.raw`q_j=\mathbf 1[j=y]`,true)}</h4>${vector(one,'q',y)}</section><section><h4>${tex(String.raw`\mathrm dX`,true)}</h4>${vector(result.dx,String.raw`\mathrm dx`,y,result.dx.map(v=>v>0?'pv-positive':v<0?'pv-negative':'pv-zero'))}</section></div>${tex(`\\mathrm dx_j=\\underbrace{${upstream}/${d.N}}_{g/N}\\left(p_j-\\mathbf1[j=${y}]\\right)`)}<p class="small">这里的标签向量只用于解释正确类减一；实现时按整数编号索引。${tex('g=0',true)} 时全部回传为零，负上游会反转方向。</p><details><summary>展开完整位置损失核对</summary><div class="pv-loss-list">${d.positions.map((co,i)=>`<span>${tex(`\\ell_{(${co.join(',')})}=${(-d.logp[i][d.target[i]]).toFixed(4)}`,true)}</span>`).join('')}</div>${tex(`L=${result.loss.toFixed(6)}`)}</details>`;
   }
   box.querySelector('.pv-content').innerHTML=html;
   box.querySelectorAll('[data-prow]').forEach(b=>b.onclick=()=>{n=+b.dataset.prow;if(loss)j=d.target[n];draw();});
   box.querySelectorAll('[data-pcat]').forEach(b=>b.onclick=()=>{j=+b.dataset.pcat;draw();});
   box.querySelectorAll('[data-pg]').forEach(b=>b.onclick=()=>{upstream=+b.dataset.pg;draw();});
   root.CourseMath.renderAll(box);
  }draw();
 }
 function mountBranch(c){
  const s=shape(c.X),mid=[...s.slice(0,-1),c.init[1]],host=document.getElementById('animation');
  host.innerHTML=`<section class="source-map branch-view"><h2>已有子层怎样组成一个分支</h2><div class="mv-chain"><button data-branch="norm"><strong><code>norm</code> · <code>RMSNorm</code></strong>${tex(`X\\ ${'('+s.join(',')+')'}\\to U\\ ${'('+s.join(',')+')'}`)}</button><button data-branch="mlp"><strong><code>mlp</code> · <code>MLP</code></strong>${tex(`U\\ ${'('+s.join(',')+')'}\\to Y\\ ${'('+s.join(',')+')'}`)}</button></div>${tex(String.raw`G\xrightarrow{\rm mlp.backward}\mathrm dU\xrightarrow{\rm norm.backward}\mathrm dX`)}<div class="mv-note"></div><p class="small">${tex(`\\operatorname{MLP}:(${s.join(',')})\\to(${mid.join(',')})\\to(${mid.join(',')})\\to(${s.join(',')})`,true)}；全部位置保留。五个参数属于已有子层，分支负责传递返回值。</p></section>`;
  const box=host.firstElementChild;
  function select(id){box.querySelector('.mv-note').innerHTML=id==='norm'?`<h3>先前向、后反向</h3><p><code>norm</code> 接收原输入并返回归一化特征。反向接收 <code>mlp.backward</code> 返回的 ${tex(String.raw`\mathrm dU`,true)}，累计自己的缩放参数梯度，再返回 ${tex(String.raw`\mathrm dX`,true)}。</p>`:`<h3>后前向、先反向</h3><p><code>mlp</code> 接收归一化特征，内部复用两个 <code>Linear</code> 和 <code>GELU</code>。反向先接收外部上游，累计四个参数梯度并返回 ${tex(String.raw`\mathrm dU`,true)}。</p>`;box.querySelectorAll('[data-branch]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.branch===id)));root.CourseMath.renderAll(box);}
  box.querySelectorAll('[data-branch]').forEach(b=>b.onclick=()=>select(b.dataset.branch));select('mlp');
 }
 root.CompactLearningViews={mount:(id,c)=>id==='foundation_capstone'?mountBranch(c):mountProbability(id,c)};
 if(typeof module!=='undefined')Object.assign(module.exports,{probabilityData,probabilityRelation,probabilityOverview});
})(typeof window==='undefined'?globalThis:window);
/* 第七部分：追踪窗口来源与共享词表的两种用途，不重复独立批次。 */
(function(root){'use strict';
 const range=n=>Array.from({length:n},(_,i)=>i),tex=(s,inline=false)=>root.CourseMath.block(s,inline);
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const char=c=>c===' '?'␠ 空格':c==='\n'?'↵ 换行':c;
 function mountData(c){
  const args=c.input,chars=Array.from(args.text),vocab=Array.from(new Set(chars)).sort((a,b)=>a.codePointAt(0)-b.codePointAt(0)),ids=chars.map(v=>vocab.indexOf(v)),starts=args.starts,T=args.T;
  let batch=0;const host=document.getElementById('animation');
  host.innerHTML=`<section class="source-map"><h2>窗口从哪里来：输入与目标相差一个位置</h2><p>词表与原文保留完整顺序。独立窗口采用同一规则，默认看一行，可切换起点；真实批次形状为 ${tex(`(${starts.length},${T})`,true)}。</p><div class="p7-vocab">${vocab.map((v,i)=>`<span>${esc(char(v))} → ${i}</span>`).join('')}</div><label>查看批次行 <select data-p7-batch aria-label="数据窗口批次">${starts.map((s,b)=>`<option value="${b}">${b} · 起点 ${s}</option>`).join('')}</select></label><div class="p7-windows"></div></section>`;
  function strip(positions,label,cls){return `<div class="p7-strip"><strong>${label}</strong><div>${positions.map(i=>`<span class="p7-token ${cls}"><small>原文位置 ${i}</small><b>${esc(char(chars[i]))}</b><small>编号 ${ids[i]}</small></span>`).join('')}</div></div>`;}
  function draw(){const s=starts[batch],input=range(T).map(i=>s+i),target=range(T).map(i=>s+i+1);
   host.querySelector('.p7-windows').innerHTML=`${strip(range(chars.length),'完整原文','')}${tex(String.raw`s=${s},\quad X=\mathrm{ids}[${s}:${s+T}],\quad\mathrm{targets}=\mathrm{ids}[${s+1}:${s+T+1}]`)}${strip(input,'输入 · 从起点开始','p7-input')}${strip(target,'目标 · 从后一个位置开始','p7-target')}<p>同一列的目标来自输入后一个原文位置；最后一个目标仍在原文中，不把第一项循环搬到末尾。</p>`;root.CourseMath.renderAll(host);
  }host.querySelector('select').onchange=e=>{batch=Number(e.target.value);draw();};draw();
 }
 root.Part07Views={mount:(id,c)=>{if(id==='t_data')mountData(c);}};
})(typeof window==='undefined'?globalThis:window);
