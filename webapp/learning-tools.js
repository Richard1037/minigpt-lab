/* 学习检查、完整形状地图和失败定位；不改写个人代码。 */
(function(root){'use strict';
function modelShapes(c){const {B,T,C,H,V,L,F,context}=c;if(C%H)throw Error('C 必须能被 H 整除');if(T>context)throw Error('T 不能超过 context');return {ids:[B,T],x:[B,T,C],qkv:[B,T,3*C],heads:[B,H,T,C/H],scores:[B,H,T,T],hidden:[B,T,F],logits:[B,T,V],N:B*T,parameters:V*C+context*C+L*(4*C*C+2*C*F+7*C+F)+C};}
const defaults={B:2,T:3,C:8,H:2,V:12,L:2,F:32,context:12};
function mount(ctx){
 map(ctx);quiz(ctx);
 const entry=document.getElementById('commonGuideEntry');
 entry.hidden=!ctx.book.common_extension?.trim();
 entry.onclick=()=>openCommonGuide(ctx.book);
}
function openCommonGuide(book){
 const box=document.createElement('dialog');box.id='commonGuideWindow';box.className='model-map-window';box.setAttribute('aria-labelledby','commonGuideTitle');
 box.innerHTML='<div class="model-map-heading"><h2 id="commonGuideTitle">数值误差与有限差分</h2><button type="button" class="map-close" aria-label="关闭数值误差与有限差分">关闭 ×</button></div>'+book.common_extension;
 document.body.append(box);box.querySelector('.map-close').onclick=()=>box.close();
 box.addEventListener('close',()=>box.remove(),{once:true});
 box.addEventListener('click',e=>{if(e.target===box){const r=box.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)box.close();}});
 if(root.CourseMath)root.CourseMath.renderAll(box);
 box.showModal();
}
function quiz({meta,current,read,put}){if(!meta.checkpoints?.length)return;
 const box=document.createElement('section');box.className='understanding-check';const qs=meta.checkpoints,key=meta.checkpoint_storage_key||'understand-'+current;
 const signature=JSON.stringify(qs),saved=read(key),answers=saved?.signature===signature?saved.answers:[];
 box.innerHTML='<h2>理解检查（可选）</h2><p class="small">这些选择题用于确认计算关系，单独保存理解进度；代码仍需通过正式测评。</p>'+qs.map((q,i)=>`<fieldset><legend>${i+1}. ${esc(q.prompt)}</legend>${q.options.map((o,j)=>`<label><input type="radio" name="understand-${i}" value="${j}" ${answers[i]===j?'checked':''}> ${esc(o)}</label>`).join('')}<button type="button" data-question="${i}">检查这一题</button><div class="quiz-feedback" aria-live="polite"></div></fieldset>`).join('')+'<p class="quiz-progress" aria-live="polite"></p>';
 const target=document.getElementById('understandingQuestions');if(!target)return;target.replaceChildren(box);
 const checked=new Set(saved?.signature===signature?saved.checked:[]);
 function progress(){const n=qs.filter((q,i)=>checked.has(i)&&answers[i]===q.answer).length;box.querySelector('.quiz-progress').textContent=`理解检查 ${n}/${qs.length} 通过`;put(key,JSON.stringify({signature,answers,checked:[...checked]}));}
 box.querySelectorAll('input').forEach(el=>el.onchange=()=>{const i=Number(el.name.split('-')[1]);answers[i]=+el.value;checked.delete(i);el.closest('fieldset').querySelector('.quiz-feedback').textContent='答案已修改，重新检查后计入理解进度。';progress();});
 box.querySelectorAll('[data-question]').forEach(button=>button.onclick=()=>{const i=+button.dataset.question,feedback=button.nextElementSibling;
 if(answers[i]===undefined){feedback.textContent='请先选择一个答案。';return;}
 checked.add(i);const ok=answers[i]===qs[i].answer;feedback.className='quiz-feedback '+(ok?'pass':'fail');feedback.textContent=(ok?'✓ 判断正确。':'需要再想一想。')+qs[i].explain;progress();});progress();
}
function map({book,current,go,status,read,put}){document.getElementById('modelMapWindow')?.remove();const box=document.createElement('dialog');box.id='modelMapWindow';box.className='model-map model-map-window';box.setAttribute('aria-labelledby','modelMapTitle');box.innerHTML='<div class="model-map-heading"><h2 id="modelMapTitle">完整 miniGPT：形状与梯度地图</h2><button type="button" class="map-close" aria-label="关闭模型地图">关闭 ×</button></div><p>点击节点查看输入、输出、参数和缓存；切换反向视图观察梯度返回路径。这里的尺寸可独立调整，不会修改训练配置。</p><div class="map-controls"></div><div class="map-note" aria-live="polite"></div><div class="model-nodes"></div><div class="node-inspector"></div>';
 document.body.append(box);let entry=document.getElementById('modelMapEntry');if(!entry){entry=document.createElement('button');entry.id='modelMapEntry';entry.type='button';entry.textContent='完整模型地图 ↗';document.getElementById('courseTools').append(entry);}entry.setAttribute('aria-haspopup','dialog');entry.onclick=()=>box.showModal();box.querySelector('.map-close').onclick=()=>box.close();box.addEventListener('click',e=>{if(e.target===box){const r=box.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)box.close();}});let config={...defaults,...read('model-map-config')},phase='forward',selected='ids';
 box.querySelector('.map-controls').innerHTML=Object.entries(config).map(([key,value])=>`<label>${esc({B:'batch B',T:'token T',C:'宽度 C',H:'头数 H',V:'词表 V',L:'层数 L',F:'MLP 宽度',context:'位置表长度'}[key])}<input type="number" data-shape="${key}" min="${['L','context'].includes(key)?1:2}" max="${key==='V'?100:64}" value="${value}"></label>`).join('')+'<label>计算方向<select id="mapPhase"><option value="forward">前向 →</option><option value="backward">反向：从损失回到输入</option></select></label>';
 function draw(){let s;try{s=modelShapes(config);}catch(e){box.querySelector('.map-note').textContent=e.message;return;}
 put('model-map-config',JSON.stringify(config));box.querySelector('.map-note').textContent=`N=B×T=${s.N}；每头 d=C/H=${config.C/config.H}；参数共 ${s.parameters} 个（共享词表只计算一次）。`;
 const f=a=>'('+a.join(', ')+')',nodes=[
 {id:'ids',title:'token 编号',shape:f(s.ids),lesson:'t_data',input:'文本经字符词表编码，整数 ids；目标为右移一位的编号。',output:f(s.ids)+' int64',params:'无',cache:'本节点没有可微分的连续输入。',gradient:'整数 ids 不求导。词表参数的梯度通过散射累加。'},
 {id:'embedding',title:'token + position',shape:f(s.x),lesson:'layer_embedding',input:'ids '+f(s.ids)+'；位置编号 (T,)；位置向量广播到 B 条序列。',output:f(s.x),params:`token (${config.V},${config.C})；position (${config.context},${config.C})`,cache:'token ids、位置 ids，供两次查表反向使用。',gradient:'dX '+f(s.x)+' 散射到 token 表；对 batch 求和后散射到 position 表。词表还需加上输出头的梯度。'},
 {id:'block',title:`Block × ${config.L}`,shape:f(s.x),lesson:'t_block',input:f(s.x)+' → Norm → Attention → 残差 → Norm → MLP → 残差。',output:f(s.x),params:'每层独立的两个 Norm、QKV/proj、MLP 两个 Linear。',cache:`Q/K/V ${f(s.heads)}；A ${f(s.scores)}；MLP 激活 ${f(s.hidden)}；各子层保存自己的缓存。`,gradient:'按相反层序执行 backward；每个残差处直连梯度与子网络梯度相加，不能平均或覆盖。'},
 {id:'norm',title:'最终 RMSNorm',shape:f(s.x),lesson:'layer_rmsnorm',input:f(s.x),output:f(s.x),params:`w (${config.C},)`,cache:'每个 token 的缩放因子 r、归一化激活。',gradient:'直接路径与经过均方尺度的路径合并；w.grad 对所有位置求和。'},
 {id:'head',title:'共享词表输出头',shape:f(s.logits),lesson:'t_gpt',input:'末尾激活 '+f(s.x)+'；词表 E 与输入查表使用同一个 Parameter。',output:'logits '+f(s.logits),params:'复用 token 表，不创建第二份参数。',cache:'末尾激活 H，反向计算 dE_head。',gradient:'上游分别回到末尾激活和共享词表；词表再叠加输入查表的梯度。'},
 {id:'loss',title:'平均交叉熵',shape:'标量',lesson:'t_loss',input:'logits '+f(s.logits)+'；targets '+f(s.ids),output:'零维平均损失；N='+s.N,params:'无',cache:'概率 P、目标编号。',gradient:'类别概率减去正确类别指示，再除位置总数；平均系数只应用一次。'}];
 const ordered=phase==='forward'?nodes:[...nodes].reverse();box.querySelector('.model-nodes').innerHTML=ordered.map(n=>`<button type="button" data-node="${n.id}" class="${selected===n.id?'selected':''}"><strong>${esc(n.title)}</strong><span>${phase==='backward'&&n.id==='ids'?'整数编号：不求导':(phase==='backward'&&n.id!=='loss'?'梯度 / ':'')+esc(n.shape)}</span></button>`).join('<span class="map-arrow">→</span>');
 const n=nodes.find(n=>n.id===selected);box.querySelector('.node-inspector').innerHTML=`<h3>${esc(n.title)}</h3><dl><dt>输入</dt><dd>${esc(n.input)}</dd><dt>输出</dt><dd>${esc(n.output)}</dd><dt>参数</dt><dd>${esc(n.params)}</dd><dt>缓存</dt><dd>${esc(n.cache)}</dd><dt>反向</dt><dd>${esc(n.gradient)}</dd></dl><p class="small">QKV ${f(s.qkv)} → 分头 ${f(s.heads)} → 分数 ${f(s.scores)}；所有 batch/head 都保留。</p><button type="button" class="map-open">打开对应章节：${esc(book.lessons[n.lesson].title)}</button><span class="small"> ${esc(status(n.lesson).text)}</span>`;
 const formulas={
 embedding:[String.raw`X_{b,t,c}=E_{\mathrm{ids}_{b,t},c}+P_{t,c}`,String.raw`\mathrm dP_{t,c}=\sum_b\mathrm dX_{b,t,c}`],
 block:[String.raw`Y=X+F(X)`,String.raw`\mathrm dX=G+J_F(X)^{\mathsf T}G`],
 norm:[String.raw`Y=\widehat X\odot w`,String.raw`\mathrm dX=r\big[u-\widehat X\odot\operatorname{mean}_c(u\odot\widehat X)\big]`],
 head:[String.raw`\mathrm{logits}=H_{\mathrm{out}}E^{\mathsf T}`,String.raw`\begin{aligned}\mathrm dH_{\mathrm{out}}&=GE\\\mathrm dE_{\mathrm{head}}&=G_2^{\mathsf T}H_{\mathrm{out},2}\end{aligned}`],
 loss:[String.raw`L=-\frac1{BT}\sum_{b,t}\log p_{b,t,\mathrm{targets}_{b,t}}`,String.raw`\mathrm d\mathrm{logits}=\frac{P-\operatorname{one\_hot}(\mathrm{targets})}{BT}`]
 };
 if(root.CourseMath&&formulas[n.id]){box.querySelector('.node-inspector h3').insertAdjacentHTML('afterend',root.CourseMath.block(formulas[n.id][phase==='forward'?0:1]));root.CourseMath.renderAll(box);}
 box.querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>{selected=el.dataset.node;draw();});box.querySelector('.map-open').onclick=()=>{box.close();go(n.lesson);};
 }
 box.querySelectorAll('[data-shape]').forEach(el=>el.onchange=()=>{const value=Number(el.value);if(!Number.isInteger(value)||value<Number(el.min)||value>Number(el.max)){box.querySelector('.map-note').textContent='请输入界限内的整数。';return;}config[el.dataset.shape]=value;draw();});box.querySelector('#mapPhase').onchange=e=>{phase=e.target.value;draw();};draw();
}
function wireDiagnostics(ctx){document.querySelectorAll('#tests details.failed').forEach((row,i)=>{const idx=+row.dataset.caseIndex,c=ctx.meta.cases[idx],record=ctx.read('pilot-result-'+ctx.current),result=record?.data?.rows?.[idx];if(!result)return;
 const message=(result.diagnostics||[]).join(' ');let target=null,mode=null,coord=[];
 const match=message.match(/(W\.grad|b\.grad|dX)\(([^)]+)\)/);if(match){target=match[1];coord=match[2].split(',').map(Number);mode=target==='W.grad'?'dW':target==='b.grad'?'db':'dX';}
 if(!mode)mode=/W\.grad/.test(message)?'dW':/b\.grad/.test(message)?'db':/dX/.test(message)?'dX':null;
 const isLinear=['linear','layer_linear','rebuild_linear'].includes(ctx.current),isAttention=ctx.current==='t_attention',isBlock=ctx.current==='t_block';
 if(!isLinear&&!isAttention&&!isBlock)return;
 const button=document.createElement('button');button.type='button';button.className='diagnostic-jump';button.textContent=isLinear&&ctx.current!=='rebuild_linear'&&!c.assessment&&shape(c.X||c.args?.[0]).length!==3?'查看该输入的展平与形状恢复':'用本测评输入查看梯度来源';row.querySelector('.case-body').append(button);
 button.onclick=()=>{if(isLinear&&ctx.current!=='rebuild_linear'&&!c.assessment&&shape(c.X||c.args?.[0]).length!==3){ctx.go('reshape');return;}const animation=document.querySelector('#animation');const fold=animation.closest('[data-lesson-fold]');if(fold){fold.hidden=false;fold.open=true;}animation.innerHTML='';
 // 使用失败用例，不将它冒充页面开头的默认样例。
 const title=document.createElement('p');title.className='diagnostic-context';title.textContent=(c.assessment?'需要定位的行为检查：':'当前展示失败用例：')+c.label;animation.append(title);
 if(c.assessment){animation.innerHTML+='<p>本项为行为/差分检查。先用本页固定数值样例定位对应的梯度路径；完整随机输入见失败项配置。</p>';ctx.install(ctx.meta,ctx.current,ctx.meta.cases[ctx.meta.sample_index||0]);}
 else {const visual=c.visual_parameters?{...c,parameters:{...c.parameters,...c.visual_parameters}}:c;ctx.install({...ctx.meta,residual_trace:c.residual_trace||ctx.meta.residual_trace},ctx.current,visual);}
 if(isLinear&&mode){const sel=document.querySelector('#linMode');if(sel){sel.value=mode;sel.dispatchEvent(new Event('change'));
 const fields=mode==='dW'?['linI','linJ']:mode==='db'?['linJ']:['linB','linT','linI'];fields.forEach((key,j)=>{const el=document.getElementById(key);if(el&&coord[j]!==undefined&&coord[j]<el.options.length){el.value=coord[j];el.dispatchEvent(new Event('change'));}});}}
 if(isAttention){const el=document.getElementById('attPhase');if(el){el.value=/dV/.test(message)?'dV':/dQ|dK/.test(message)?'dQ':'dS';el.dispatchEvent(new Event('change'));}}
 if(isLinear&&mode&&root.SourceMaps)root.SourceMaps.focus(mode,coord);
 animation.scrollIntoView({behavior:'smooth',block:'start'});
 };
 });}
root.LearningTools={mount,wireDiagnostics,modelShapes,openCommonGuide};if(typeof module!=='undefined')module.exports={modelShapes};
})(typeof window==='undefined'?globalThis:window);
