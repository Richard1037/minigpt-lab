/* 先展示完整关系，再点选目标追踪全部来源；特征数值不是默认视觉主体。 */
(function(root){'use strict';
 const attentionIds=['att_scores','att_probs','att_weighted','att_merge','att_value_backward','att_score_backward','t_attention','t_mha','t_block','t_dropout','t_layernorm','rebuild_block'];
 const supported=new Set(['layer_scale','linear','layer_linear','layer_embedding','layer_gelu','layer_rmsnorm','layer_mlp','rebuild_linear','foundation_capstone','t_softmax','t_loss','t_data',...attentionIds]);
 const shape=x=>Array.isArray(x)?[x.length,...(x.length?shape(x[0]):[])]:[];
 const product=a=>a.reduce((n,v)=>n*v,1);
 const range=n=>Array.from({length:n},(_,i)=>i);
 const coordinates=s=>s.length?range(s[0]).flatMap(i=>coordinates(s.slice(1)).map(rest=>[i,...rest])):[[]];
 const at=(a,c)=>c.reduce((v,i)=>v[i],a);
 const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const tex=(s,inline=false)=>root.CourseMath?root.CourseMath.block(s,inline):`<span data-tex="${escape(s)}">${escape(s)}</span>`;
 const dims=s=>'('+s.join(', ')+')';
 const eq=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
 function model(id,c){
  if(!supported.has(id))return null;
  const kind=id==='layer_embedding'?'embedding':id==='layer_scale'?'scale':'linear';
  const x=id==='linear'?c.args[0]:c.X,xd=shape(x);
  const prefix=kind==='embedding'?xd:xd.slice(0,-1),positions=coordinates(prefix),N=product(prefix);
  const C=kind==='embedding'?(c.parameters?.weight?.[0]?.length||c.init[1]):xd.at(-1);
  const D=kind==='linear'?(id==='linear'?shape(c.args[1])[1]:c.init[1]):C;
  const V=kind==='embedding'?(c.parameters?.weight?.length||c.init[0]):0;
  const bias=kind==='linear'&&(id==='linear'||c.init[2]!==false);
  return {id,kind,prefix,positions,N,C,D,V,bias,x,case:c,ids:kind==='embedding'?positions.map(p=>at(x,p)):[],
   modes:id==='linear'?['forward']:kind==='embedding'?['forward','dE']:kind==='scale'?['forward','dX','dw']:['forward','dX','dW',...(bias?['db']:[])]};
 }
 function relation(m,mode,target=[0,0]){
  const {N,C,D,V}=m;const [r,j]=target;
  const rows=(n,c,f)=>range(n).flatMap(i=>range(c).map(k=>f(i,k)));
  const panel=(id,title,dimensions,tone,selected=[],positionAxis=null,transposed=false)=>({id,title,dimensions,tone,selected,positionAxis,transposed});
  const row=(i,n)=>range(n).map(k=>[i,k]),col=(j,n)=>range(n).map(k=>[k,j]);
  if(m.kind==='linear'){
   const pos=m.positions[r],position=pos?`位置 (${pos.join(', ')})`:'全部位置';
   if(mode==='forward')return {formula:String.raw`Y_2=X_2W${m.bias?'+b':''}`,focus:String.raw`(Y_2)_{${r},${j}}=\sum_{i=0}^{${C-1}}(X_2)_{${r},i}W_{i,${j}}${m.bias?`+b_{${j}}`:''}`,
    note:`${position} 的一行输入，与 W 的一列沿输入特征 i 组合${m.bias?'，再加该输出特征的偏置':''}。同一组参数被全部位置共享。`,reduction:'沿输入特征 i 汇总',target:panel('Y2','Y₂ · 目标输出',[N,D],'target',[target],'row'),
    sources:[panel('X2','X₂ · 位置输入',[N,C],'x',row(r,C),'row'),panel('W','W · 共享权重',[C,D],'w',col(j,C)),...(m.bias?[panel('b','b · 共享偏置',[1,D],'bias',[[0,j]])]:[])],operators:m.bias?['×','+']:['×']};
   if(mode==='dX')return {formula:String.raw`\mathrm dX_2=G_2W^{\mathsf T}`,focus:String.raw`(\mathrm dX_2)_{${r},${j}}=\sum_{k=0}^{${D-1}}(G_2)_{${r},k}(W^{\mathsf T})_{k,${j}}`,
    note:`${position} 的一行上游，沿输出特征 k 汇总回输入特征 ${j}。Wᵀ 的这一列来自原 W 的行索引 ${j}；转置只交换特征轴。`,reduction:'沿输出特征 k 汇总',target:panel('dX2','dX₂ · 目标输入梯度',[N,C],'target',[target],'row'),
    sources:[panel('G2','G₂ · 输出的上游梯度',[N,D],'g',row(r,D),'row'),panel('WT','Wᵀ · 转置后的权重',[D,C],'w',col(j,D),null,true)],operators:['×']};
   if(mode==='dW')return {formula:String.raw`\mathrm dW=X_2^{\mathsf T}G_2`,focus:String.raw`(\mathrm dW)_{${r},${j}}=\sum_{n=0}^{${N-1}}(X_2^{\mathsf T})_{${r},n}(G_2)_{n,${j}}`,
    note:`X₂ᵀ 的这一行来自原 X₂ 的特征列 ${r}，与 G₂ 的输出特征列 ${j} 汇总全部 ${N} 个位置；所有 batch 的来源都会保留。`,reduction:'沿所有位置 n 汇总',target:panel('dW','dW · 目标权重梯度',[C,D],'target',[target]),
    sources:[panel('XT','X₂ᵀ · 转置后的位置输入',[C,N],'x',row(r,N),'column',true),panel('G2','G₂ · 输出的上游梯度',[N,D],'g',col(j,N),'row')],operators:['×']};
   return {formula:String.raw`\mathrm db=\operatorname{sum}(G_2,\mathrm{axis}=0)`,focus:String.raw`(\mathrm db)_{${j}}=\sum_{n=0}^{${N-1}}(G_2)_{n,${j}}`,
    note:`同一个偏置被所有位置相加使用。收集 G₂ 中输出特征列 ${j} 的全部位置，得到这个偏置的梯度。`,reduction:'归约位置轴，保留输出特征',target:panel('db','db · 目标偏置梯度',[1,D],'target',[[0,j]]),
    sources:[panel('G2','G₂ · 输出的上游梯度',[N,D],'g',col(j,N),'row')],operators:[],sum:true};
  }
  if(m.kind==='embedding'){
   const v=mode==='forward'?m.ids[r]:r,matching=m.ids.flatMap((id,n)=>id===v?[n]:[]);
   if(mode==='forward')return {formula:String.raw`\mathrm{ids}\longrightarrow E\text{ 的对应行}\longrightarrow Y`,focus:String.raw`\mathrm{ids}_{${m.positions[r].join(',')}}=${v}\quad\Longrightarrow\quad Y_{${m.positions[r].join(',')},:}=E_{${v},:}`,
    note:`位置 (${m.positions[r].join(', ')}) 的编号 ${v} 指向词表行索引 ${v}。这整行特征进入该输出位置；其他相同编号也共享这行参数。`,reduction:'按编号取整行',target:panel('Y2','Y · 目标位置的特征',[N,C],'target',row(r,C),'row'),
    sources:[panel('ids','ids · 每个位置的编号',[N,1],'address',[[r,0]],'row'),panel('E','E · 共享词表',[V,C],'w',row(v,C))],operators:['→'],routes:[{position:m.positions[r],row:v}],value:v};
   return {formula:String.raw`\mathrm dE_{v,:}=\sum_{b,t:\,\mathrm{ids}_{b,t}=v}G_{b,t,:}`,focus:String.raw`\mathrm dE_{${v},:}=${matching.length?matching.map(n=>`G_{${m.positions[n].join(',')},:}`).join('+'):String.raw`\mathbf 0`}`,
    note:matching.length?`编号 ${v} 有 ${matching.length} 处来源。高亮的每行上游都回到同一个词表梯度行，跨 batch 的重复编号也合流。`:`本次没有位置使用编号 ${v}，该行收到的新增梯度为零。`,reduction:'相同编号的整行贡献相加',target:panel('dE','dE · 目标词表梯度',[V,C],'target',row(v,C)),
    sources:[panel('ids','ids · 找到所有相同编号',[N,1],'address',matching.map(n=>[n,0]),'row'),panel('G2','G · 各位置的整行上游',[N,C],'g',matching.flatMap(n=>row(n,C)),'row')],operators:['→'],routes:matching.map(n=>({position:m.positions[n],row:v})),value:v};
  }
  const isWeight=mode==='dw',input=mode==='forward'?'X2':'G2';
  return {formula:mode==='forward'?String.raw`Y=wX`:isWeight?String.raw`\mathrm dw=\sum_{b,t,c}G_{b,t,c}X_{b,t,c}`:String.raw`\mathrm dX=Gw`,
   focus:isWeight?String.raw`\mathrm dw=\operatorname{sum}(G\odot X)`:String.raw`${mode==='forward'?'Y':'\\mathrm dX'}_{${m.positions[r].join(',')},${j}}=${mode==='forward'?'X':'G'}_{${m.positions[r].join(',')},${j}}w`,
   note:isWeight?'一个标量参数在所有位置、所有特征被复用；每一格的上游与原输入共同贡献到同一个参数梯度。':'选中的位置和特征，使用本位置的输入或上游，与所有位置共享的同一个 w 组合。',
   reduction:isWeight?'归约全部位置轴与特征轴':'逐元素对应，共享一个标量',
   target:isWeight?panel('dw','dw · 目标标量梯度',[1,1],'target',[[0,0]]):panel(mode==='forward'?'Y2':'dX2',mode==='forward'?'Y · 目标输出':'dX · 目标输入梯度',[N,C],'target',[target],'row'),
   sources:isWeight?[panel('G2','G · 上游梯度',[N,C],'g',rows(N,C,(a,b)=>[a,b]),'row'),panel('X2','X · 前向输入',[N,C],'x',rows(N,C,(a,b)=>[a,b]),'row')]:[panel(input,mode==='forward'?'X · 前向输入':'G · 上游梯度',[N,C],mode==='forward'?'x':'g',[target],'row'),panel('w','w · 所有位置共享',[1,1],'w',[[0,0]])],operators:[isWeight?'⊙':'×'],sum:isWeight};
 }
 function linearSelection(m,selection){
  const all=(R,C)=>range(R).flatMap(r=>range(C).map(j=>[r,j]));
  const row=(r,C)=>range(C).map(j=>[r,j]),col=(j,R)=>range(R).map(r=>[r,j]);
  const marks={};let kind=selection?.kind||'dW';
  if(!selection?.matrix){
   if(kind==='dW')Object.assign(marks,{X2:all(m.N,m.C),G2:all(m.N,m.D),dW:all(m.C,m.D),W:all(m.C,m.D),Y2:all(m.N,m.D)});
   if(kind==='dX')Object.assign(marks,{G2:all(m.N,m.D),W:all(m.C,m.D),dX2:all(m.N,m.C),X2:all(m.N,m.C),Y2:all(m.N,m.D)});
   if(kind==='db')Object.assign(marks,{G2:all(m.N,m.D),b:all(1,m.D),db:all(1,m.D),Y2:all(m.N,m.D)});
   if(kind==='forward')Object.assign(marks,{X2:all(m.N,m.C),W:all(m.C,m.D),Y2:all(m.N,m.D),...(m.bias?{b:all(1,m.D)}:{})});
  }else{
   const {matrix,r,j}=selection;
   if(['W','dW'].includes(matrix)){kind='dW';Object.assign(marks,{W:[[r,j]],dW:[[r,j]],X2:col(r,m.N),Y2:col(j,m.N),G2:col(j,m.N)});}
   if(['X2','dX2'].includes(matrix)){kind='dX';Object.assign(marks,{X2:[[r,j]],dX2:[[r,j]],W:row(j,m.D),Y2:row(r,m.D),G2:row(r,m.D)});}
   if(['b','db'].includes(matrix)){kind='db';Object.assign(marks,{b:[[0,j]],db:[[0,j]],Y2:col(j,m.N),G2:col(j,m.N)});}
   if(matrix==='Y2'){kind='forward';Object.assign(marks,{X2:row(r,m.C),W:col(j,m.C),Y2:[[r,j]],...(m.bias?{b:[[0,j]]}:{})});}
   if(matrix==='G2'){kind='dW';Object.assign(marks,{G2:[[r,j]],Y2:[[r,j]],X2:row(r,m.C),W:col(j,m.C),dW:col(j,m.C),dX2:row(r,m.C),...(m.bias?{db:[[0,j]],b:[[0,j]]}:{})});}
  }
  return {kind,marks};
 }
 function mountLinearMatrices(box,m){
  let selection={kind:'dW'};
  const descriptions={
   forward:{title:'整个输出：每个位置使用同一组权重',text:'每一行输入都产生一行输出。'+tex('W',true)+' 的每一列定义一个输出特征；所有位置共用同一组参数。把全部位置放在一起，就是一次矩阵乘法。',formula:String.raw`Y_2=X_2W${m.bias?'+b':''}`,reason:String.raw`(N,C)(C,D)\longrightarrow(N,D)`},
   dW:{title:'整个权重梯度：每个位置贡献一张同形矩阵，再相加',text:'在一个位置上，每个输入特征都参与了每个输出特征。前向里，输出随权重变化的倍数就是对应输入，所以反向的上游要乘这个输入。把这一行输入与这一行上游两两相乘，就得到一张 '+tex(String.raw`C\times D`,true)+' 的权重梯度贡献表。所有位置使用同一个 '+tex('W',true)+'，所以把这些表相加；'+tex(String.raw`X_2^{\mathsf T}G_2`,true)+' 一次完成全部配对与求和。',formula:String.raw`\mathrm dW=\sum_{n=0}^{N-1}\underbrace{(X_2)_{n,:}^{\mathsf T}(G_2)_{n,:}}_{\text{一个位置对整张 }W\text{ 的贡献}}=X_2^{\mathsf T}G_2`,reason:String.raw`\underbrace{(C,1)(1,D)}_{\text{一个位置得到 }(C,D)}\quad\xrightarrow{\text{累加全部 }N\text{ 个位置}}\quad(C,D)`},
   dX:{title:'整个输入梯度：把输出特征的影响传回各输入特征',text:'前向中，每个输入特征通过 '+tex('W',true)+' 的一行影响该位置的全部输出特征。反向把该位置的上游按这些权重分配回来，再沿输出特征求和。每个位置独立处理，位置轴保留；转置后，把输出特征收回为输入特征。',formula:String.raw`\mathrm dX_2=G_2W^{\mathsf T}`,reason:String.raw`(N,D)(D,C)\longrightarrow(N,C)`},
   db:{title:'整个偏置梯度：沿位置轴收集同一偏置的全部贡献',text:'同一个偏置向量被加到每一行输出。前向加法对偏置的局部导数是 1，所以反向直接把所有位置的上游向量相加，保留输出特征轴。',formula:String.raw`\mathrm db=\sum_{n=0}^{N-1}(G_2)_{n,:}`,reason:String.raw`(N,D)\xrightarrow{\text{沿位置轴求和}}(D,)`}
  };
  const specs=[['X2','X₂ · 输入',m.N,m.C,'x'],['W','W · 共享权重',m.C,m.D,'w'],...(m.bias?[['b','b · 共享偏置',1,m.D,'bias']]:[]),['Y2','Y₂ · 输出',m.N,m.D,'target'],['dX2','dX₂ · 输入梯度',m.N,m.C,'x'],['dW','dW · 权重梯度',m.C,m.D,'w'],...(m.bias?[['db','db · 偏置梯度',1,m.D,'bias']]:[]),['G2','G₂ · 输出收到的上游',m.N,m.D,'g']];
  box.className='source-map linear-matrices';
  box.innerHTML=`<h2>Linear：全部矩阵与计算关系</h2><p class="small">${tex(`N=${m.prefix.join('\\cdot ')}=${m.N},\\quad C=${m.C},\\quad D=${m.D}`,true)}。${tex(String.raw`X,Y,G,\mathrm dX`,true)} 的前置位置统一展平为行；每行保留原坐标，计算后恢复原形状。</p><div class="lm-equations">${tex(String.raw`Y_2=X_2W${m.bias?'+b':''}\qquad G_2=\frac{\partial L}{\partial Y_2}`)}${tex(String.raw`\mathrm dX_2=G_2W^{\mathsf T}\qquad\mathrm dW=X_2^{\mathsf T}G_2${m.bias?String.raw`\qquad\mathrm db=\sum_n(G_2)_{n,:}`:''}`)}</div><p class="small">点矩阵格子查看关联区域；下方解释整个矩阵的计算。蓝色：输入，紫色：权重，橙色：上游；深边框标出点击位置。</p><div class="lm-grid"></div><div class="lm-whole"><span>看整个矩阵：</span>${['forward','dW','dX',...(m.bias?['db']:[])].map(k=>`<button type="button" data-lm-whole="${k}">${tex({forward:'Y',dW:String.raw`\mathrm dW`,dX:String.raw`\mathrm dX`,db:String.raw`\mathrm db`}[k],true)}</button>`).join('')}</div><section class="lm-explain" aria-live="polite"></section>`;
  if(m.id==='rebuild_linear')box.querySelector('h2').textContent='独立 Linear：构造后的参数与梯度关系';
  const symbols={X2:'X_2',W:'W',b:'b',Y2:'Y_2',dX2:String.raw`\mathrm dX_2`,dW:String.raw`\mathrm dW`,db:String.raw`\mathrm db`,G2:'G_2'};
  const label=(id,title)=>tex(symbols[id],true)+' · '+title.split(' · ')[1];
  const grid=box.querySelector('.lm-grid');
  function draw(){
   const state=linearSelection(m,selection),d=descriptions[state.kind];
   grid.innerHTML=specs.map(([id,title,R,C,tone])=>`<section class="lm-matrix sm-${tone}"><h3>${label(id,title)}</h3><div class="lm-shape">${tex(id==='b'||id==='db'?`(${C},)`:dims([R,C]),true)}</div><div class="lm-bracket"><div class="lm-cells" style="grid-template-columns:auto repeat(${C},22px)"><span></span>${range(C).map(j=>`<span class="lm-index">${j}</span>`).join('')}${range(R).map(r=>`<span class="lm-index" title="${['X2','Y2','G2','dX2'].includes(id)?'原坐标 ('+m.positions[r].join(',')+')':'行索引 '+r}">${['X2','Y2','G2','dX2'].includes(id)?`${r}<small>(${m.positions[r].join(',')})</small>`:r}</span>`+range(C).map(j=>`<button type="button" class="lm-cell ${(state.marks[id]||[]).some(c=>eq(c,[r,j]))?'lm-related':''} ${selection.matrix===id&&selection.r===r&&selection.j===j?'lm-picked':''}" data-lm-cell="${id},${r},${j}" aria-label="${id} 行 ${r} 列 ${j}" aria-pressed="${selection.matrix===id&&selection.r===r&&selection.j===j}">·</button>`).join('')).join('')}</div></div></section>`).join('');
   box.querySelector('.lm-explain').innerHTML=`<h3>${d.title}</h3><p>${d.text}</p>${tex(d.formula)}<div class="small">${tex(d.reason)}</div>${selection.matrix?`<p class="small">已选 ${tex(`(${symbols[selection.matrix]})_{${selection.r},${selection.j}}`,true)}：${selection.matrix==='G2'?`这一格上游会贡献到高亮的整列权重梯度、整行输入梯度${m.bias?'与对应偏置':''}。`:'高亮区域显示前向影响与反向计算的对应关系。'} 点击“看整个矩阵”恢复整体视图。</p>`:''}<p class="small">${tex(String.raw`\mathrm dW${m.bias?String.raw`,\mathrm db`:""}`,true)} 是本次贡献，累加进参数的 <code>grad</code>；${tex(String.raw`\mathrm dX_2`,true)} 恢复输入形状后返回。</p>`;
   box.querySelectorAll('[data-lm-whole]').forEach(b=>b.setAttribute('aria-pressed',String(!selection.matrix&&selection.kind===b.dataset.lmWhole)));
   grid.querySelectorAll('[data-lm-cell]').forEach(b=>b.onclick=()=>{const [matrix,r,j]=b.dataset.lmCell.split(',');selection={matrix,r:Number(r),j:Number(j)};draw();grid.querySelector(`[data-lm-cell="${b.dataset.lmCell}"]`)?.focus({preventScroll:true});});
   root.CourseMath?.renderAll(box);
  }
  box.querySelectorAll('[data-lm-whole]').forEach(b=>b.onclick=()=>{selection={kind:b.dataset.lmWhole};draw();});
  box.sourceMapFocus=(kind,coord)=>{selection={matrix:{dW:'dW',dX:'dX2',db:'db',forward:'Y2'}[kind],r:kind==='db'?0:kind==='dW'?(coord[0]||0):Math.max(0,m.positions.findIndex(p=>eq(p,coord.slice(0,-1)))),j:coord.at(-1)||0};draw();};
  draw();
 }
 function mount(id,c){
  if(id==='t_data'){root.Part07Views.mount(id,c);return;}
  if(['foundation_capstone','t_softmax','t_loss'].includes(id)){root.CompactLearningViews.mount(id,c);return;}
  if(attentionIds.includes(id)){root.AttentionViews.mount(id,c);return;}
  if(id==='layer_gelu'){mountGelu();return;}
  if(id==='layer_rmsnorm'||id==='layer_mlp'){root.ModuleViews.mount(id,c);return;}
  const m=model(id,c),host=document.getElementById('animation');if(!m||!host)return;
  const old=Array.from(host.childNodes);const numbers=document.createElement('details');numbers.className='source-numbers';
  numbers.innerHTML='<summary>展开本例数值与逐步核对（可选）</summary>';
  if(m.kind==='linear'&&!m.bias)old.forEach(node=>node.remove());
  else old.forEach(node=>numbers.append(node));
  if((!old.length||(m.kind==='linear'&&!m.bias))&&root.arrayView)numbers.innerHTML+=root.arrayView('X · 完整输入',m.x)+(c.G?root.arrayView('G · 完整上游',c.G):'')+(m.kind==='scale'?root.arrayView('w · 共享标量',c.init[0]):'')+root.arrayView('本例结果',c.expected);
  const box=document.createElement('section');box.className='source-map';host.prepend(box);host.append(numbers);
  if(id==='layer_linear'||id==='rebuild_linear'){numbers.remove();mountLinearMatrices(box,m);return;}
  const names={forward:'前向',dX:'输入梯度 dX',dW:'权重梯度 dW',db:'偏置梯度 db',dE:'词表梯度 dE',dw:'标量参数梯度 dw'};
  box.innerHTML=`<h2>先看整体，再点目标找来源</h2><div class="sm-modes" role="group" aria-label="数据流方向">${m.modes.map(mode=>`<button type="button" data-sm-mode="${mode}">${names[mode]}</button>`).join('')}</div><div class="sm-shape-flow"></div><div class="sm-formula"></div><p class="sm-instruction">点击绿色目标矩阵中的任意格子，高亮会一次显示它的全部来源。</p><div class="sm-boards"></div><div class="sm-explanation" aria-live="polite"><strong class="sm-reduction"></strong><p class="sm-note"></p><div class="sm-focus"></div><div class="sm-routes"></div></div>`;
  let mode='forward',target=[0,0];
  const $=s=>box.querySelector(s);
  function positions(axis,n){return axis?`n=${n} · (${m.positions[n].join(',')})`:String(n);}
  function board(p,isTarget){
   const [R,C]=p.dimensions;const columns=range(C).map(j=>`<span class="sm-index" title="${p.positionAxis==='column'?positions(true,j):'列索引 '+j}">${p.positionAxis==='column'?'n='+j:j}</span>`).join('');
   const originalShape=m.kind!=='linear'&&p.positionAxis==='row'?(p.id==='ids'?m.prefix:[...m.prefix,m.C]):null;
   const feature=m.kind==='linear'?'输出特征 j':'特征 c';
   const axes={X2:'行：位置 n · 列：输入特征 i',G2:'行：位置 n · 列：'+feature,W:'行：输入特征 i · 列：输出特征 j',WT:'行：输出特征 j · 列：输入特征 i',XT:'行：输入特征 i · 列：位置 n',Y2:'行：位置 n · 列：'+feature,dX2:'行：位置 n · 列：输入特征 i',dW:'行：输入特征 i · 列：输出特征 j',db:'列：输出特征 j',ids:'行：位置 n · 格内是 token 编号',E:'行：词表编号 v · 列：特征 c',dE:'行：词表编号 v · 列：特征 c'}[p.id];
   const cells=range(R).map(r=>`<span class="sm-index sm-row-index" title="${p.positionAxis==='row'?positions(true,r):'行索引 '+r}">${p.positionAxis==='row'?`n=${r}<small>(${m.positions[r].join(',')})</small>`:r}</span>`+range(C).map(j=>{
    const selected=p.selected.some(c=>eq(c,[r,j])),value=p.id==='ids'?m.ids[r]:isTarget?(selected?'●':'○'):'·';
    const attr=`class="sm-cell ${selected?'sm-selected':''}" title="${escape(p.title)} [${r},${j}]"`;
    return isTarget?`<button type="button" ${attr} data-sm-target="${r},${j}" aria-pressed="${selected}" aria-label="目标 ${p.id} 行索引 ${r} 列索引 ${j}">${value}</button>`:`<span ${attr}>${value}</span>`;
   }).join('')).join('');
   return `<div class="sm-board sm-${p.tone}"><h3>${escape(p.title)}</h3><div class="sm-dimensions">${tex(p.id==='w'||p.id==='dw'?String.raw`\text{标量}`:originalShape?String.raw`\text{原形状 }${dims(originalShape)}`:String.raw`\text{形状 }(${R},${C})`,true)}</div>${originalShape?`<p class="sm-axis-note">位置视图 ${dims([R,C])}，保留全部位置</p>`:''}${p.transposed?`<p class="sm-transpose">${p.id==='WT'?`W ${dims([m.C,m.D])} → Wᵀ ${dims([m.D,m.C])}`:`X₂ ${dims([m.N,m.C])} → X₂ᵀ ${dims([m.C,m.N])}`}：行列交换</p>`:''}${axes?`<p class="sm-axis-note">${axes}</p>`:''}<div class="sm-matrix-scroll"><div class="sm-matrix" style="grid-template-columns:auto repeat(${C},minmax(23px,1fr))"><span></span>${columns}${cells}</div></div>${p.positionAxis==='column'?'<p class="sm-axis-note">列 n 遍历下方全部位置映射</p>':p.positionAxis==='row'?'<p class="sm-axis-note">每行保留原位置坐标</p>':''}</div>`;
  }
  function syncNumbers(){
   const select=numbers.querySelector('#linMode');if(!select||!Array.from(select.options).some(o=>o.value===mode))return;
   select.value=mode;select.dispatchEvent(new Event('change'));
   const [r,j]=target,pos=m.positions[r]||[];
   const fields=mode==='dW'?{linI:r,linJ:j}:mode==='db'?{linJ:j}:{linB:pos[0],linT:pos[1],...(mode==='dX'?{linI:j}:{linJ:j})};
   for(const [id,value] of Object.entries(fields)){const el=numbers.querySelector('#'+id);if(el&&value!==undefined&&value<el.options.length){el.value=value;el.dispatchEvent(new Event('change'));}}
  }
  function draw(){
   box.querySelector('h2').textContent='先看整体，再点目标找来源';
   for(const s of ['.sm-shape-flow','.sm-formula','.sm-instruction','.sm-boards','.sm-explanation'])$(s).hidden=false;
   const relationData=relation(m,mode,target),outShape=[...m.prefix,m.kind==='linear'?m.D:m.C];
   let flow;
   if(m.kind==='embedding')flow=mode==='forward'?String.raw`\mathrm{ids}\ ${dims(m.prefix)}\ \longrightarrow\ E\ ${dims([m.V,m.C])}\ \longrightarrow\ Y\ ${dims(outShape)}`:String.raw`\left(G\ ${dims(outShape)},\ \mathrm{ids}\ ${dims(m.prefix)}\right)\ \xrightarrow{\text{按编号汇总}}\ \mathrm dE\ ${dims([m.V,m.C])}`;
   else if(mode==='forward')flow=String.raw`X\ ${dims([...m.prefix,m.C])}\ \xrightarrow{\text{保留全部位置并展平}}\ X_2\ ${dims([m.N,m.C])}\ \longrightarrow\ Y\ ${dims(outShape)}`;
   else if(mode==='dX')flow=String.raw`G\ ${dims(outShape)}\ \longrightarrow\ G_2\ ${dims([m.N,m.D])}\ \longrightarrow\ \mathrm dX\ ${dims([...m.prefix,m.C])}`;
   else flow=String.raw`G\ ${dims(outShape)}\ \longrightarrow\ G_2\ ${dims([m.N,m.D])}\ \xrightarrow{\text{汇总全部位置${m.kind==='scale'?'和特征':''}}}\ ${mode==='dw'?String.raw`\mathrm dw\ \text{标量}`:mode==='db'?String.raw`\mathrm db\ (${m.D},)`:String.raw`\mathrm dW\ ${dims([m.C,m.D])}`}`;
   $('.sm-shape-flow').innerHTML=tex(flow);
   $('.sm-formula').innerHTML=tex(relationData.formula);
   $('.sm-boards').innerHTML=relationData.sources.map((p,i)=>board(p,false)+(relationData.operators[i]?`<span class="sm-operator" aria-hidden="true">${relationData.operators[i]}</span>`:'')).join('')+`<span class="sm-operator sm-result-arrow">${relationData.sum?'Σ →':'→'}</span>`+board(relationData.target,true);
   $('.sm-reduction').textContent=relationData.reduction;
   $('.sm-note').textContent=relationData.note;
   $('.sm-focus').innerHTML=tex(relationData.focus);
   $('.sm-routes').innerHTML=(relationData.routes||[]).map(route=>`<span>位置 (${route.position.join(', ')}) <b>→</b> 行索引 ${route.row}</span>`).join('');
   if(m.kind==='linear'&&mode==='dW')$('.sm-routes').innerHTML=m.positions.map((p,n)=>`<span>n=${n} ↔ (${p.join(', ')})</span>`).join('');
   box.querySelectorAll('[data-sm-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.smMode===mode)));
   box.querySelectorAll('[data-sm-target]').forEach(b=>b.onclick=()=>{const coordinate=b.dataset.smTarget;target=coordinate.split(',').map(Number);draw();box.querySelector(`[data-sm-target="${coordinate}"]`)?.focus({preventScroll:true});});
   root.CourseMath?.renderAll(box);
   if(numbers.open)syncNumbers();
  }
  box.querySelectorAll('[data-sm-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.smMode;target=mode==='dE'?[m.ids[0],0]:[0,0];draw();});
  numbers.ontoggle=()=>{if(numbers.open)syncNumbers();};
  box.sourceMapFocus=(requested,coord=[])=>{
   if(!m.modes.includes(requested))return;mode=requested;
   if(mode==='dW')target=[coord[0]||0,coord[1]||0];
   else if(mode==='db')target=[0,coord[0]||0];
   else {const position=coord.slice(0,-1),n=m.positions.findIndex(p=>eq(p,position));target=[Math.max(0,n),coord.at(-1)||0];}
   const spec=relation(m,mode,[0,0]).target.dimensions;
   if(target[0]>=spec[0]||target[1]>=spec[1])target=[0,0];draw();
  };
  draw();
 }
 function geluDetails(x){
  const a=.044715,k=Math.sqrt(2/Math.PI),t=Math.tanh(k*(x+a*x*x*x));
  const direct=.5*(1+t),indirect=.5*x*(1-t*t)*k*(1+3*a*x*x);
  return {y:x*direct,direct,indirect,slope:direct+indirect};
 }
 function mountGelu(){
  const host=document.getElementById('animation');if(!host)return;
  host.innerHTML='<section class="source-map gelu-explorer"><h2>GELU 怎样改变输入和梯度</h2><p>同一个函数独立作用于所有元素，形状保持不变。拖动输入位置，看输出曲线和该处的局部斜率。</p><div class="ge-controls"><label>输入位置 x <input type="range" min="-4" max="4" step="0.05" value="0" aria-label="选择 GELU 输入位置"></label><button data-ge-x="-1.5">负值区域</button><button data-ge-x="0">零点附近</button><button data-ge-x="1.5">正值区域</button></div><div class="ge-plots"></div><div class="ge-explain" aria-live="polite"></div></section>';
  const box=host.firstElementChild;let x=0;
  function plot(derivative){
   const W=480,H=235,left=42,right=455,top=15,bottom=200,lo=derivative?-.4:-1,hi=derivative?1.5:4;
   const px=v=>left+(v+4)/8*(right-left),py=v=>bottom-(v-lo)/(hi-lo)*(bottom-top),d=geluDetails(x),v=derivative?d.slope:d.y;
   const points=range(161).map(n=>{const z=-4+n*.05,a=geluDetails(z);return `${px(z)},${py(derivative?a.slope:a.y)}`;}).join(' ');
   return `<div><h3>${derivative?'局部斜率：反向怎样缩放梯度':'函数曲线：输入怎样变成输出'}</h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${derivative?'GELU导数曲线':'GELU函数曲线和切线'}"><line x1="${left}" y1="${py(0)}" x2="${right}" y2="${py(0)}" class="ge-axis"/><line x1="${px(0)}" y1="${top}" x2="${px(0)}" y2="${bottom}" class="ge-axis"/><text x="${right}" y="${H-10}">x</text><text x="5" y="18">${derivative?"f′(x)":'f(x)'}</text>${derivative?`<line x1="${left}" x2="${right}" y1="${py(1)}" y2="${py(1)}" class="ge-guide"/><text x="5" y="${py(1)+4}">1</text>`:''}<polyline points="${points}" class="ge-curve"/><line x1="${px(x)}" y1="${py(0)}" x2="${px(x)}" y2="${py(v)}" class="ge-guide"/>${!derivative?`<line x1="${px(Math.max(-4,x-.65))}" y1="${py(d.y+d.slope*(Math.max(-4,x-.65)-x))}" x2="${px(Math.min(4,x+.65))}" y2="${py(d.y+d.slope*(Math.min(4,x+.65)-x))}" class="ge-tangent"/>`:''}<circle cx="${px(x)}" cy="${py(v)}" r="5" class="ge-point"/></svg></div>`;
  }
  function draw(){
   const d=geluDetails(x);box.querySelector('.ge-plots').innerHTML=plot(false)+plot(true);
   const effect=d.slope<0?'这里斜率为负：输入梯度相对上游反向；绝对值小于一时，幅度也缩小。':d.slope>1?'这里斜率大于一：输入梯度与上游同向，幅度略放大。':d.slope<.01?'这里斜率接近零：输出对输入变化不敏感，传回的梯度也接近零。':'这里斜率介于零和一：输入梯度与上游同向，幅度缩小。';
   box.querySelector('.ge-explain').innerHTML=`<p>绿色是函数曲线，橙色短线是当前点的切线。切线的斜率决定反向如何改变上游梯度；右图直接画出这个斜率。</p><p><strong>${effect}</strong></p>${tex(String.raw`f'(x)=\underbrace{\color{#28659a}{\tfrac12(1+t)}}_{\text{直接乘法路径}}+\underbrace{\color{#976020}{\tfrac12x(1-t^2)k(1+3ax^2)}}_{\text{经过 }u,t\text{ 的路径}}`)}<p>第一项来自乘法中的 x，第二项来自另一因子 t 随 x 的变化。两项相加是同一个输出对输入的总变化率。</p>${tex(String.raw`\mathrm dX=G\odot f'(X)`)}<details><summary>数值核对（可选）</summary><p>x=${x.toFixed(2)}，输出=${d.y.toFixed(4)}；两条路径的导数分别为 ${d.direct.toFixed(4)} 和 ${d.indirect.toFixed(4)}，相加得到 ${d.slope.toFixed(4)}。</p></details>`;
   root.CourseMath?.renderAll(box);
  }
  const slider=box.querySelector('input');slider.oninput=()=>{x=Number(slider.value);draw();};
  box.querySelectorAll('[data-ge-x]').forEach(b=>b.onclick=()=>{x=Number(b.dataset.geX);slider.value=x;draw();});draw();
 }
 root.SourceMaps={supports:id=>supported.has(id),mount,focus:(mode,coord)=>document.querySelector('#animation > .source-map')?.sourceMapFocus?.(mode,coord)};
 if(typeof module!=='undefined')module.exports={model,relation,linearSelection,geluDetails};
})(typeof window==='undefined'?globalThis:window);
