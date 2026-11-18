'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function get(key){try{return localStorage.getItem(key)}catch{return null}}
function put(key,value){try{localStorage.setItem(key,value)}catch{}}
function read(key){try{return JSON.parse(get(key)||'null')}catch{return null}}
let book,current,step=0,timer=null;
let editorResizeCleanup=null;
let readingHistory=null;
let labSidebar;
const running=new Set();
const m=()=>book.lessons[current];
const pageTitle=id=>id===CourseNavigation.trainingLab.id?CourseNavigation.trainingLab.title:book.lessons[id]?.title;
const order=()=>book.stages.filter(s=>!s.optional).flatMap(s=>s.modules.flatMap(x=>x.tasks));
function locationOf(id){for(const stage of book.stages)for(const module of stage.modules)if(module.tasks.includes(id))return {stage,module};}
function explanationState(id){return (book.visual_roadmap?.items||[]).find(item=>item.id.startsWith('learner-explanation-')&&['review','accepted'].includes(item.status)&&item.lessons.includes(id))?.status||'todo';}
function explanationReviewed(id){return explanationState(id)!=='todo';}
function isLegacyAnswers(id,raw){if(!book.lessons[id].legacy_assessment||raw===null)return false;try{const v=JSON.parse(raw);return Array.isArray(v)&&v.every(x=>Number.isInteger(x));}catch{return false;}}
function legacyRecord(id){const meta=book.lessons[id],record=read('pilot-result-'+id);return meta.legacy_assessment&&record?.version===JSON.stringify(meta.legacy_assessment.questions)?record:null;}
function preserveLegacy(id){
  if(!book.lessons[id].legacy_assessment)return;
  const key='pilot-history-'+id+'-concept-v1',raw=get('pilot-'+id),record=legacyRecord(id);
  if(!get(key)&&(record||isLegacyAnswers(id,raw)))put(key,JSON.stringify({draft:isLegacyAnswers(id,raw)?raw:null,result:record}));
}
function priorAssessmentRecord(id){
  const migration=book.lessons[id].assessment_migration;if(!migration)return null;
  const live=read('pilot-result-'+id);let version;
  try{version=JSON.parse(live?.version||'null');}catch{}
  if(Array.isArray(version)&&Number.isInteger(version[0])&&version[0]>=1&&version[0]<=migration.previous_revision)return live;
  for(let revision=migration.previous_revision;revision>=1;revision--){const archived=read('pilot-history-'+id+'-revision-'+revision);if(archived)return archived;}
  return null;
}
function preservePriorAssessment(id){
  const migration=book.lessons[id].assessment_migration;if(!migration)return;
  const record=priorAssessmentRecord(id);if(!record)return;
  let revision;try{revision=JSON.parse(record.version)[0];}catch{return;}
  if(!Number.isInteger(revision)||revision<1||revision>migration.previous_revision)return;
  const key='pilot-history-'+id+'-revision-'+revision;
  if(!get(key))put(key,JSON.stringify(record));
}
function draft(id){const meta=book.lessons[id];if(meta.integration_only||meta.provided)return meta.local_code||meta.starter;const saved=get('pilot-'+id);if(saved!==null&&!isLegacyAnswers(id,saved))return saved;if(meta.kind==='concept')return '[]';return meta.local_code||meta.scaffold_notes?.starter||meta.starter;}
function explainCall(code){
  if(current!=='infrastructure')return code;
  return code.replace(/^([ \t]*)def __call__\(self, \*args\):\r?\n\1    return self\.forward\(\*args\)/m,(_,indent)=>`${indent}# module(X) 自动调用这个方法，相当于 module.forward(X)。\n${indent}# 定义中的 *args 将位置参数收集成元组。\n${indent}def __call__(self, *args):\n${indent}    # 调用中的 *args 拆开元组，传给 forward，再返回结果。\n${indent}    return self.forward(*args)`);
}
function version(id){const meta=book.lessons[id];return JSON.stringify(meta.kind==='concept'?meta.questions:[meta.assessment_revision||1,meta.cases]);}
function dependencyCode(id,d){return draft(d);}
function dependencyVersion(id){return JSON.stringify((book.lessons[id].dependencies||[]).map(d=>[d,dependencyCode(id,d),version(d)]));}
function status(id){
  const record=read('pilot-result-'+id),meta=book.lessons[id];
  if(meta.provided)return {text:'已提供 · 可直接调用',kind:'success'};
  if(!record)return {text:'○ 未测评',kind:''};
  if(legacyRecord(id))return {text:'旧选择题记录保留 · 请提交初始化代码',kind:'warn'};
  if(record.version!==version(id))return {text:'↻ 验收标准已更新，需重测',kind:'warn'};
  if(record.code!==draft(id)||(meta.dependencies?.length&&record.dependencies!==dependencyVersion(id)))return {text:'↻ 已修改，待重测',kind:'warn'};
  const total=meta.kind==='concept'?meta.questions.length:meta.cases.length,passed=record.data.rows?.filter(r=>r.passed).length||0;
  if(passed===total&&record.data.auto_save?.ok===false)return {text:'! 测评通过 · 本地保存未完成',kind:'warn'};
  if(passed===total&&record.data.auto_save?.ok)return {text:`✓ 已通过 · ${passed}/${total} · 本地已验证并保存`,kind:'success'};
  return {text:passed===total?`✓ 已通过 · ${passed}/${total}`:`! 待修改 · ${passed}/${total}`,kind:passed===total?'success':'warn'};
}
function captureReading(){return {scrollTop:$('main').scrollTop,folds:Array.from($('main').querySelectorAll('details'),el=>el.open)};}
function restoreReading(reading){
  if(!reading){$('main').scrollTop=0;return;}
  $('main').querySelectorAll('details').forEach((el,i)=>{if(typeof reading.folds?.[i]==='boolean')el.open=reading.folds[i];});
  const lesson=current,top=reading.scrollTop||0;$('main').scrollTop=top;
  requestAnimationFrame(()=>{if(current===lesson)$('main').scrollTop=top;});
}
function visitLesson(id,reading){clearInterval(timer);timer=null;current=id;step=0;put('course-current',id);render();restoreReading(reading);}
function go(id,options){readingHistory.go(id,options);}
function setupReadingReturn(){
  const entry=readingHistory?.entry(),fromTitle=pageTitle(entry?.from);if(!entry?.review||!entry.depth||!fromTitle)return;
  const bar=document.createElement('nav');bar.className='reading-return';bar.setAttribute('aria-label','返回阅读位置');
  const button=document.createElement('button');button.type='button';button.textContent='← 返回 '+fromTitle;button.onclick=()=>readingHistory.back();
  bar.append(button);$('main').prepend(bar);
}
function navigation(){
  CourseNavigation.mount({book,current,status,go,explanationState});
  if($('taskTabs')){
    const {module}=locationOf(current);
    $('taskTabs').innerHTML=module.tasks.map((id,i)=>{
      const result=status(id),explanation=explanationState(id);
      return `<button type="button" data-task="${id}" class="module-task ${id===current?'active':''}" ${id===current?'aria-current="step"':''}><span class="task-number">${String(i+1).padStart(2,'0')}</span><strong class="task-title">${esc(book.lessons[id].title)}</strong><span class="task-status" data-state="${result.kind}">${esc(result.text)}</span><span class="task-explanation explanation-tag ${explanation==='accepted'?'explanation-accepted':explanation==='review'?'explanation-review':''}">${explanation==='accepted'?'讲解已验收':explanation==='review'?'讲解待验收':'讲解待调整'}</span></button>`;
    }).join('');
    document.querySelectorAll('[data-task]').forEach(b=>b.onclick=()=>go(b.dataset.task));
    const n=module.tasks.filter(id=>status(id).kind==='success').length;
    $('moduleProgress').textContent=`${n}/${module.tasks.length} 已通过`;
    $('moduleBar').style.width=100*n/module.tasks.length+'%';
    $('moduleProgressTrack').setAttribute('aria-valuenow',n);
    $('moduleProgressTrack').setAttribute('aria-valuetext',`${module.tasks.length} 项任务中 ${n} 项已通过`);
  }
}
function shape(value){return Array.isArray(value)?[value.length,...(value.length?shape(value[0]):[])]:[];}
function matrix(value,highlight=[],rows=null,columns=null){
  const count=(value[0]?.length||1)+(rows?1:0);
  const header=columns?(rows?'<div class="tensor-coordinate">坐标</div>':'')+columns.map(s=>`<div class="tensor-coordinate">${esc(s)}</div>`).join(''):'';
  return `<div class="array-block"><div class="matrix" style="grid-template-columns:repeat(${count},minmax(var(--matrix-cell-min,34px),auto))">${header}${value.map((r,i)=>(rows?`<div class="tensor-coordinate">${esc(rows[i])}</div>`:'')+r.map((v,j)=>`<div class="cell ${highlight.includes(i+','+j)?'hot':''}">${typeof v==='number'?Number(v.toFixed(5)):esc(v)}</div>`).join('')).join('')}</div></div>`;
}
function arrayView(label,value,axes=null){
  if(value===null)return `<div class="array-name">${esc(label)}</div><p>None（此项没有返回值）</p>`;
  if(typeof value==='string'||typeof value==='boolean')return `<p><strong>${esc(label)}</strong>：${esc(value)}</p>`;
  if(!Array.isArray(value)&&typeof value==='object')return Object.entries(value).map(([k,v])=>arrayView(k,v)).join('');
  const rectangular=v=>!Array.isArray(v)||v.every(child=>JSON.stringify(shape(child))===JSON.stringify(shape(v[0]))&&rectangular(child));
  if(Array.isArray(value)&&!rectangular(value))return `<div class="array-name">${esc(label)} · 分组列表（各组形状不同）</div>`+value.map((v,i)=>arrayView(label+'['+i+']',v)).join('');
  const dims=shape(value),heading=`<div class="array-name">${esc(label)} <span>· ${dims.length?'形状（'+dims.join('，')+'）':'标量'}</span></div>`;
  axes=axes||(dims.length===4?['batch b','head h','token t','特征 d']:['batch b','token t','特征 j']);
  if(!dims.length)return heading+`<p>${esc(value)}</p>`;
  if(dims.length===1)return heading+matrix([value]);
  if(dims.length===2)return heading+matrix(value);
  const slices=(part,prefix=[])=>{
    if(shape(part).length===2){
      const title=prefix.map((v,i)=>`${axes[i]||'轴 '+i}=${v}`).join(' · ');
      return `<div class="tensor-slice" data-coordinates="${prefix.join(',')}"><strong class="slice-title">${title}</strong>${matrix(part,[],part.map((_,i)=>axes[dims.length-2]+'='+i),part[0].map((_,i)=>axes[dims.length-1]+'='+i))}</div>`;
    }
    return part.map((child,i)=>slices(child,[...prefix,i])).join('');
  };
  return heading+`<div class="tensor-slices">${slices(value)}</div>`;
}
function example(){return m().cases[m().sample_index||0];}
function caseInput(c){
  if(m().kind==='initialization')return namedInput(c.input);
  if(c.assessment)return namedInput(c.input);
  if(c.mode==='checkpoint')return namedInput(c.input,true,['init','ids','targets'])+'<p class="small">本用例固定使用 AdamW 学习率 0.01、default_rng(42)：先更新一次并保存，再更新一次；加载后重复第二次更新并比较结果。</p>';
  if(m().kind==='project')return namedInput(c.input,true)+(m().call?`<p class="small">${esc(m().call)}</p>`:'');
  if(m().kind==='array')return c.args.map((a,i)=>arrayView(m().args[i],a,m().input_axes?.[m().args[i]])).join('');
  if(m().kind==='infrastructure')return m().learning_page?.sample_input||`<p>${esc(c.input)}</p>`;
  return constructorView(c.init)+arrayView(m().class==='Embedding'?'ids':'X',c.X)+arrayView('上游梯度 G',c.G)+Object.entries(c.parameters||{}).map(([k,v])=>arrayView(k,v)).join('')+`<p class="small">${c.repeat?'连续执行两轮 <code>forward</code> → <code>backward</code>，参数梯度应累加为两份。':'执行一轮 <code>forward</code> → <code>backward</code>。'}之后检查 <code>zero_grad</code>；输入和参数 <code>data</code> 保持原值。</p>`;
}
function constructorView(values){
  if(!values?.length)return '';
  const labels={V:'词表大小',C:'输入/模型特征宽度',D:'输出特征宽度',H:'头数',hidden:'MLP 中间宽度',layers:'Block 层数',context:'最大上下文长度',seed:'初始化种子',p:'丢弃概率',eps:'除零保护',causal:'因果掩码开关',value:'初始缩放',use_bias:'是否使用偏置'};
  return '<dl class="named-config">'+values.map((value,i)=>{const name=m().constructor_names?.[i]||'构造项 '+(i+1);return `<div><dt>${esc(name)} · ${esc(labels[name]||'本页构造条件')}</dt><dd>${esc(value)}</dd></div>`;}).join('')+'</dl>';
}
function namedInput(input,filterUnused=false,fields=null){
  if(!input||typeof input!=='object'||Array.isArray(input))return arrayView('输入',input);
  const names={X:['t_loss','t_softmax','t_sample'].includes(current)?'logits（代码 X）':'X',ids:'ids · token 编号',targets:'targets · 下一 token 标签',G:'G · 上游梯度',starts:'starts · 窗口起点',T:'T · 窗口长度',steps:'steps · 执行步数',lr:'lr · 学习率',seed:'seed · 用例/采样 RNG 种子',temperature:'temperature · 采样温度',top_k:'top_k · 保留类别数',shape:'shape · 检查张量的轴长度',coordinates:'coordinates · 每参数张量检查坐标数',weights:'weights · 初始参数',gradients:'gradients · 给定梯度',operation:'operation · 本用例执行的行为',text:'text · 建词表语料',limit:'limit · 全局裁剪上限'};
  if(m().kind==='initialization')Object.assign(names,{shape:'shape · 权重形状 (C,D)',shapes:'shapes · 连续调用时依次使用的权重形状',seed:'seed · 调用方创建 RNG 的种子',default_rng:'default_rng · 本用例省略 rng 参数'});
  const shown=fields||m().sample_fields;
  return Object.entries(input).filter(([key])=>key!=='class'&&(!filterUnused||!shown||shown.includes(key))).map(([key,value])=>key==='init'?constructorView(value):key==='shape'?`<p class="small"><strong>${esc(names[key])}</strong>：(${value.join(', ')})</p>`:arrayView(names[key]||key,value)).join('');
}
function taskBrief(meta){
  const t=meta.task_brief;if(!t)return '';
  return `<div class="task-overview"><h2>${esc(t.scope)}</h2><div class="contract-table"><table><thead><tr><th>参数 / 变量</th><th>含义</th><th>形状 / 类型</th></tr></thead><tbody>${t.variables.map(v=>`<tr><th>${esc(v.name)}</th><td>${esc(v.meaning)}</td><td>${esc(v.shape)}</td></tr>`).join('')}</tbody></table></div><p><strong>计算：</strong>${esc(t.operation)}</p><p><strong>输出：</strong>${esc(t.output)}</p>${t.conditions.length?`<p class="contract-conditions"><strong>约定：</strong>${t.conditions.map(esc).join(' ')}</p>`:''}${Object.keys(t.dimensions||{}).length?`<details class="dimension-note"><summary>维度符号</summary><p class="small">${Object.entries(t.dimensions).map(([k,v])=>esc(k)+'：'+esc(v)).join('；')}。</p></details>`:''}</div>`;
}
function lessonText(text){
  const fragments=m().formula_fragments;
  if(!fragments||!window.CourseMath)return esc(text);
  const keys=Object.keys(fragments).sort((a,b)=>b.length-a.length).map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'));
  return String(text).split(new RegExp('('+keys.join('|')+')','g')).map(part=>Object.hasOwn(fragments,part)?CourseMath.block(fragments[part],true):esc(part)).join('');
}
const lessonFoldStates=new Map();
function exampleSections(){
  return '<details class="lesson-fold" data-lesson-fold="sample"><summary>样例输入与输出</summary><div class="lesson-fold-content"><div id="sample"></div></div></details><details class="lesson-fold" id="understandingFold" data-lesson-fold="understanding" open><summary>理解检查 · 选择题</summary><div class="lesson-fold-content"><div id="understandingQuestions"></div></div></details><details class="lesson-fold" id="animationFold" data-lesson-fold="animation" open><summary>可视化与过程演示</summary><div class="lesson-fold-content"><div id="animation"></div></div></details>';
}
function setupLessonFolds(){
  // 正文中的独立样例走读，与题面样例共用折叠样式和页面内状态。
  document.querySelectorAll('#main h2,#main h3').forEach((heading,index)=>{
    if(heading.closest('details')||!/(样例|示例|例子|实际用例|走一遍)/.test(heading.textContent))return;
    const rank=Number(heading.tagName.slice(1));
    const box=document.createElement('details'),summary=document.createElement('summary'),content=document.createElement('div');
    box.className='lesson-fold';box.dataset.lessonFold='inline-'+index+'-'+heading.textContent;box.open=true;
    summary.textContent=heading.textContent;content.className='lesson-fold-content';
    heading.before(box);box.append(summary,content);
    let next=heading.nextSibling;
    while(next){
      if(next.nodeType===1&&/^H[1-6]$/.test(next.tagName)&&Number(next.tagName.slice(1))<=rank)break;
      const sibling=next.nextSibling;content.append(next);next=sibling;
    }
    heading.remove();
  });
  const lesson=current;
  document.querySelectorAll('[data-lesson-fold]').forEach(box=>{
    const key=lesson+':'+box.dataset.lessonFold;
    if(lessonFoldStates.has(key))box.open=lessonFoldStates.get(key);
    box.ontoggle=()=>lessonFoldStates.set(key,box.open);
  });
  for(const id of ['animationFold','understandingFold']){
    const box=$(id);
    if(box)box.hidden=!Array.from(box.querySelector('.lesson-fold-content').children).some(el=>el.children.length||el.textContent.trim());
  }
}
function setupBlockOverview(){
  if(!['t_block','t_gpt'].includes(current))return;
  const box=document.querySelector(current==='t_block'?'[data-block-prerequisites]':'[data-gpt-prerequisites]');if(!box)return;
  const ids=current==='t_block'?['layer_rmsnorm','t_mha','foundation_capstone']:['t_init_upgrade','layer_embedding','t_block','layer_rmsnorm'];
  box.innerHTML=ids.map(id=>`<div class="block-prereq-row"><button type="button" data-block-task="${id}">${esc(book.lessons[id].title)}</button><span><code>${esc(book.lessons[id].file)}</code> · ${get('pilot-'+id)!==null?'浏览器草稿':'本地练习文件'}</span><span>${esc(status(id).text)}</span></div>`).join('')+'<p class="small">测评状态表示当前版本的检查结果；是否已亲手实现并理解，请结合原题与自己的代码核对。</p>';
  document.querySelectorAll('[data-block-task]').forEach(button=>button.onclick=()=>go(button.dataset.blockTask,{review:true}));
}
function learningContent(meta,concept){
  if(meta.provided)return `<section class="card learning-explanation">${meta.body}</section>`;
  if(meta.learning_page){
    const page=meta.learning_page;
    return `<section class="card learning-explanation"><div class="section-label">01 / 先学明白</div><div class="lesson-purpose"><span>${esc(page.role)}</span><p>${esc(page.purpose)}</p></div>${meta.body}${meta.derivation?`<div class="derivation"><h2>公式推导</h2>${meta.derivation}</div>`:''}</section><section class="card lesson-problem"><div class="section-label">02 / ${concept?'规则应用题':'实现题'}</div><h2>${esc(page.problem_title)}</h2>${page.statement}${!concept?exampleSections():''}</section>`;
  }
  return `<section class="card goal task-contract"><div class="section-label">01 / 任务与计算</div>${taskBrief(meta)}${meta.body?`<div class="task-definition">${meta.body}</div>`:''}${meta.derivation?`<div class="derivation"><h2>公式推导</h2>${meta.derivation}</div>`:''}</section>${!concept?`<section class="card"><div class="section-label">02 / 样例与演算</div>${exampleSections()}</section>`:''}`;
}
function showLegacyRecord(){
 const meta=m();if(!meta.legacy_assessment)return;
 const history=read('pilot-history-'+current+'-concept-v1'),record=legacyRecord(current)||history?.result;
 const box=document.createElement('details');box.className='legacy-record';const summary=document.createElement('summary');summary.textContent='本节已改为代码任务 · 以前的选择题记录';box.append(summary);
 const note=document.createElement('p');note.textContent=record?`原选择题 ${record.data.rows?.filter(r=>r.passed).length||0}/${meta.legacy_assessment.questions.length} 项通过；${record.when||''}。旧记录保留，当前实现需要重新测评。`:'原选择题已由代码实现题替代。旧答题草稿会保留，当前编辑框用于实现 initialize。';box.append(note);
 if(record?.data.rows)for(const row of record.data.rows){const p=document.createElement('p');p.textContent=`${row.label}：${row.actual}（${row.passed?'通过':'未通过'}）`;box.append(p);}
 $('editorStatus').after(box);
}
function showPriorAssessmentRecord(){
 const record=priorAssessmentRecord(current);if(!record)return;
 const box=document.createElement('details');box.className='prior-assessment reference';
 const summary=document.createElement('summary');summary.textContent='旧测评记录 · 构造测评更新前';box.append(summary);
 const note=document.createElement('p'),rows=record.data?.rows||[];
 note.textContent=`原测评 ${rows.filter(row=>row.passed).length}/${rows.length} 项通过；${record.when||''}。${m().assessment_migration.reason}`;box.append(note);
 const source=document.createElement('pre');source.textContent=record.code||'';box.append(source);
 $('editorStatus').after(box);
}
function render(){
  if(window.understandingCleanup)window.understandingCleanup();
  labSidebar.onChange=null;
  if(current===CourseNavigation.trainingLab.id){renderTrainingLab();return;}
  const meta=m(),{stage,module}=locationOf(current),concept=meta.kind==='concept';
  $('main').innerHTML=`<div class="breadcrumb">${esc(stage.title)} → ${esc(module.name)}</div><section class="module-overview" aria-labelledby="moduleHeading"><div class="milestone"><div class="module-heading"><div class="module-heading-title"><span class="section-label">当前模块</span><h2 id="moduleHeading">${esc(module.name)}</h2></div><strong id="moduleProgress" class="module-progress"></strong></div>${stage.outcome?`<p class="module-summary">阶段目标：${esc(stage.outcome)}</p>`:""}<div class="bar" id="moduleProgressTrack" role="progressbar" aria-label="模块任务完成进度" aria-valuemin="0" aria-valuemax="${module.tasks.length}" aria-valuenow="0"><div id="moduleBar"></div></div></div><div class="task-tabs" id="taskTabs" role="group" aria-label="模块任务"></div></section><div class="hero"><div><div class="eyebrow">当前学习任务</div><h1>${esc(meta.title)}</h1>${meta.task_brief?'':`<p>${esc(meta.goal)}</p>`}</div></div>${learningContent(meta,concept)}<section class="card"><div class="section-label">${meta.learning_page?(concept?'03 / 提交答案':'03 / 编写与测评'):(concept?'02 / 检查理解':'03 / 动手实现')}</div>${!meta.learning_page&&meta.steps.length?'<ol>'+meta.steps.map(s=>`<li>${esc(s)}</li>`).join('')+'</ol>':''}<div id="editorStatus" class="editor-status" role="status"></div>${concept?`<div id="questions">${meta.questions.map((q,i)=>`<fieldset><legend>${i+1}. ${lessonText(q.prompt)}</legend>${q.options.map((o,j)=>`<label><input type="radio" name="q${i}" value="${j}"> ${lessonText(o)}</label>`).join('')}</fieldset>`).join('')}</div>`:`<p class="small">NumPy 别名 np；Tab 缩进。</p>${meta.dependencies?.length?'<div class="dependency-note">前置模块：'+meta.dependencies.map(d=>esc(book.lessons[d].title)).join('、')+'。</div>':''}<textarea id="code" spellcheck="false" aria-label="Python 代码编辑器"></textarea>${meta.hint?`<details><summary>提示</summary><p>${esc(meta.hint)}</p></details>`:''}<div class="controls"><button id="save">保存到本地练习</button><button id="load">读取本地文件</button><button id="restore">恢复骨架</button></div><p id="saveNote" class="small">本地：exercises/${esc(meta.file)}；保存会先备份旧文件。网站草稿自动保存在当前浏览器。</p><details><summary>本地运行与完整文件</summary><pre>python exercises/${esc(meta.file)}</pre><p>使用已安装 NumPy 的 Python，在项目目录运行。</p><pre id="localSource">${esc(meta.local_code)}</pre></details>`}<button class="primary" id="run">${concept?'检查答案':'提交代码并测评'}</button><p class="small">${concept?'所有题目答对即通过本任务。':meta.kind==='initialization'?'NumPy float64 数组 · 形状一致 · 数值与 RNG 状态一致 · 绝对容差 10⁻¹²。':meta.kind==='array'?'NumPy 数组 · 形状一致 · 输入保持原值 · 绝对容差 10⁻¹⁰。':meta.kind==='project'?'检查公开用例的输出、梯度和状态；参数按接口更新。':'检查前向、梯度、形状、梯度累加与清零；输入和权重保持原值。'}</p><details id="assessmentDetails" class="assessment-group"><summary id="results" class="assessment-heading" aria-live="polite"></summary><div id="tests"></div></details><div id="completion" class="completion success" hidden></div></section>${meta.reference?`<details class="card reference"><summary>参考实现</summary><pre>${esc(meta.reference)}</pre></details>`:''}${meta.extension?.trim()?`<details class="card"><summary>本节拓展</summary>${meta.extension}</details>`:''}<nav id="lessonPager" class="lesson-pager" aria-label="相邻任务"></nav><footer>第一至九部分已开放（第三部分标量网络暂缓）。核心计算与反向传播均使用 Python + NumPy。</footer>`;
  if(meta.provided){
    $('run').closest('.card').remove();
    setupEditorResize();navigation();
    if(window.PythonEditor)PythonEditor.highlightReadonly($('main'));
    if(window.LearningTools)LearningTools.mount(learningContext());
    if(window.VisualRoadmap)VisualRoadmap.mount(learningContext());
    setupReadingReturn();
    return;
  }
  if(meta.superseded_by||meta.practice_for){
    const target=meta.practice_for||meta.superseded_by;
    const note=document.createElement('aside');note.className='card optional-practice-note';note.innerHTML=meta.practice_for?'<p>这是按需使用的分步练习。完整讲解与实现题已合入主线模块，原代码和成绩继续保留。</p>':'<p>本题保留原代码与成绩，作为可选历史练习。主线使用已学个人模块完成综合组装。</p>';
    const button=document.createElement('button');button.type='button';button.textContent='前往'+book.lessons[target].title;button.onclick=()=>go(target);note.append(button);document.querySelector('#main .hero').after(note);
  }
  if(concept){
    const answers=read('pilot-'+current)||[];
    answers.forEach((v,i)=>{const input=document.querySelector(`[name=q${i}][value="${v}"]`);if(input)input.checked=true;});
    document.querySelectorAll('input[type=radio]').forEach(el=>el.onchange=()=>{put('pilot-'+current,JSON.stringify(answersNow()));refresh();});
  }else{
    $('code').value=explainCall(draft(current));
    $('code').oninput=()=>{preserveLegacy(current);put('pilot-'+current,$('code').value);if(window.editorHighlight)window.editorHighlight();refresh();};
    $('code').onkeydown=e=>{if(e.key==='Tab'&&!e.shiftKey){e.preventDefault();const t=e.target;t.setRangeText('    ',t.selectionStart,t.selectionEnd,'end');t.oninput();}};
    $('save').onclick=save;
    $('load').onclick=async()=>{const response=await fetch('/api/catalog');const data=await response.json();meta.local_code=data.lessons[current].local_code;$('code').value=meta.local_code;$('code').oninput();};
    $('restore').onclick=()=>{if(confirm('恢复本任务骨架？当前网页草稿将被替换，本地文件不变。')){$('code').value=meta.scaffold_notes?.starter||meta.starter;$('code').oninput();}};
    const c=example(),hasForward=c.expected&&Object.hasOwn(c.expected,'Y'),expected=hasForward?c.expected.Y:c.expected;
    $('sample').innerHTML=`<div class="sample-grid"><div class="sample-side"><h3>样例输入</h3>${caseInput(c)}</div><div class="sample-side expected"><h3>${hasForward?'forward 的预期输出 Y':'预期输出'}</h3>${arrayView('返回值',expected,meta.output_axes)}${hasForward?`<details><summary>本例 backward 的预期产出</summary>${Object.entries(c.expected).filter(([k])=>k!=='Y').map(([k,v])=>arrayView(k,v)).join('')}</details>`:''}</div></div>`;
    if(meta.learning_page?.sample_note)$('sample').insertAdjacentHTML('beforeend',meta.learning_page.sample_note);
    setupAnimation();if(window.installUnderstanding&&current!=='index'&&!['rebuild_linear','foundation_capstone','t_softmax','t_loss','t_gpt','t_data','t_optim','t_train','t_checkpoint','t_project','rebuild_gpt'].includes(current)&&!window.AttentionViews?.supports(current)&&!window.ProcessAnimations?.supports(current))installUnderstanding(meta,current,c);
    if(window.SourceMaps?.supports(current))SourceMaps.mount(current,c);
  }
  if(!concept){const note=document.createElement('p');note.className='small';note.textContent='全部通过后，自动备份并保存本次代码及所用前置模块，再按本地文件复测。';$('run').after(note);}
  showLegacyRecord();showPriorAssessmentRecord();$('run').onclick=run;$('run').disabled=running.has(current);paintCases();refresh();setupEditorResize();if(!concept&&window.PythonEditor&&window.PythonHighlighter)PythonEditor.mount($('code'));if(window.PythonEditor)PythonEditor.highlightReadonly($('main'));setupIntegrationTask();if(window.LearningTools)LearningTools.mount(learningContext());if(window.VisualRoadmap)VisualRoadmap.mount(learningContext());setupLessonFolds();setupBlockOverview();setupReadingReturn();
  const use=meta.task_brief?.next_use;
  if(use&&(stage.optional||!locationOf(use.lesson)?.stage.optional)){const link=document.createElement('button');link.type='button';link.className='next-application';link.textContent=(use.mode==='contrast'?'对照：':'应用：')+book.lessons[use.lesson].title;link.onclick=()=>go(use.lesson);document.querySelector('#main footer').append(link);}
}
function learningContext(id=current){const page=current;return {meta:book.lessons[id]||{},current:id,book,go:target=>go(target,{review:true}),status,read,put,draft,install:(meta,id,c)=>{if(!['rebuild_linear','foundation_capstone','t_softmax','t_loss','t_gpt','t_data','t_optim','t_train','t_checkpoint','t_project','rebuild_gpt'].includes(id)&&!window.AttentionViews?.supports(id))window.installUnderstanding?.(meta,id,c);if(window.SourceMaps?.supports(id))SourceMaps.mount(id,c);},isCurrent:()=>current===page};}
function answersNow(){return m().questions.map((_,i)=>{const checked=document.querySelector(`[name=q${i}]:checked`);return checked?Number(checked.value):-1;});}
function refresh(){
  navigation();if(current===CourseNavigation.trainingLab.id)return;
  const st=status(current);$('editorStatus').className='editor-status '+st.kind;$('editorStatus').textContent=st.text;
  const record=read('pilot-result-'+current),passed=st.kind==='success';
  $('completion').hidden=!passed;
  if(passed){
    const optional=locationOf(current).stage.optional,sequence=optional?locationOf(current).stage.modules.flatMap(x=>x.tasks):order(),next=sequence[sequence.indexOf(current)+1];
    const followup=book.visual_roadmap?.items?.find(item=>item.remind_after===current&&['todo','in_progress'].includes(item.status));
    const remaining=!next&&!optional?order().filter(id=>status(id).kind!=='success').length:0;
    const summary=next?'':optional?'本组拓展练习完成；不计入主线进度。':remaining?`综合任务通过；尚有 ${remaining} 个任务需要按当前标准验收。`:followup?'已完成当前课程全部任务。':'已完成当前课程全部任务。可回到训练实验室调整参数。';
    $('completion').innerHTML=`<strong>${esc(st.text)}</strong><p class="small">${esc(record.when)}</p>${summary?`<p>${summary}</p>`:''}${followup?`<p class="small">下一项实践是「${esc(followup.title)}」。计划见顶部「学习工具 → 课程完善清单 → 后续」。</p>`:''}`;
  }
  const meta=m(),count=meta.kind==='concept'?meta.questions.length:meta.cases.length;
  $('results').textContent=running.has(current)?'测评中…':legacyRecord(current)?`代码测评 · ${count} 项（旧选择题记录保留）`:!record?`测评用例 · ${count} 项`:st.kind==='warn'&&(st.text.includes('已修改')||st.text.includes('需重测'))?'内容已更新，请重测':record.data.error?'查看运行错误':`测评结果 · ${record.data.rows?.filter(r=>r.passed).length||0}/${count} 通过${record.data.rows?.some(r=>!r.passed)?' · 有未通过项':''}`;
}
function paintCases(){
  const meta=m(),record=read('pilot-result-'+current),data=record?.version===version(current)?record.data:null;
  if(meta.kind==='concept'){
    $('tests').innerHTML=(data?.rows||[]).map((r,i)=>`<details data-case-index="${i}" class="test-case ${r.passed?'':'failed'}" ${r.passed?'':'open'}><summary><span class="case-index">${i+1}</span><span class="case-title" title="${esc(r.label)}">${esc(r.label)}</span><span class="case-status ${r.passed?'pass':'fail'}">${r.passed?'✓ 通过':'! 未通过'}</span></summary><div class="case-body"><p class="case-full-label">${esc(r.label)}</p><p>你的选择：${lessonText(r.actual)}</p><p>正确答案：${lessonText(r.expected)}</p><p>${lessonText(r.explain)}</p></div></details>`).join('');return;
  }
  $('tests').innerHTML=(data?.error?`<div class="error-panel"><strong>运行未完成</strong><pre>${esc(data.error)}</pre><p>若错误来自前置练习，先完成对应模块。</p></div>`:'')+meta.cases.map((c,i)=>{
    const row=data?.rows?.[i],failed=row&&!row.passed;
    const label=compactCaseLabel(c.label);
    return `<details data-case-index="${i}" class="test-case ${failed?'failed':''}" ${failed?'open':''}><summary><span class="case-index">${i+1}</span><span class="case-title" title="${esc(label)}">${esc(label)}</span><span class="case-status ${row?(row.passed?'pass':'fail'):''}">${row?(row.passed?'✓ 通过':'! 未通过'):'未运行'}</span></summary><div class="case-body"><p class="case-full-label">${esc(label)} <small class="case-category">${esc(c.category||"数值与形状")}</small></p><div class="case-grid"><div><h3>输入</h3>${caseInput(c)}${meta.kind==='array'?`<p class="small">solve(${meta.args.join(', ')})</p>`:''}</div><div><h3>预期输出</h3>${arrayView('返回值',c.expected,meta.output_axes)}</div>${row?`<div><h3>实际输出</h3>${row.errorText?`<pre>${esc(row.errorText)}</pre>`:arrayView('返回值',row.actual,meta.output_axes)}</div>`:''}</div>${failed?'<div class="failure-location"><strong>检查位置</strong><ul>'+(row.diagnostics?.length?row.diagnostics:['检查形状、求和轴、缓存及参数梯度。']).map(d=>'<li>'+esc(d)+'</li>').join('')+'</ul></div>':''}${row&&'unchanged'in row?`<p class="small">输入${meta.kind==='layer'?'及权重':''}保持原值：${row.unchanged?'是':'否'}</p>`:''}</div></details>`;
  }).join('');
  if(window.LearningTools)LearningTools.wireDiagnostics(learningContext());
}
function compactCaseLabel(label){return label.replace('改变非方形尺寸，禁止硬编码原样例','非方形尺寸').replace('不同结构，禁止记住样例','不同结构').replace('更高维：保留所有前置轴，不展平后丢失形状','高维输入：保留所有前置轴');}
function setupIntegrationTask(){
  if(m().upgrade_followup){
    const link=document.createElement('button');link.type='button';link.className='next-application';
    link.textContent='后续升级：为完整 GPT 增加初始化配置';link.onclick=()=>go(m().upgrade_followup,{review:true});
    document.querySelector('#main footer').before(link);
  }
  if(!m().integration_only)return;
  const owner=current,ids=m().integration_files||[],editor=$('code');
  if(!ids.length)return;
  let selected=ids[0];
  const card=$('run').closest('.card');card.querySelector('.section-label').textContent='03 / 完善原模块并检查';
  card.querySelector('.dependency-note')?.setAttribute('hidden','');
  const picker=document.createElement('section');picker.className='integration-workspace';
  picker.innerHTML='<p>在这里修改六个原文件。切换文件会保留各自的网页草稿；保存只写回当前文件，并自动备份。</p><div class="integration-tabs" role="group" aria-label="选择要修改的原模块">'+ids.map(id=>'<button type="button" data-module-file="'+esc(id)+'">'+esc(book.lessons[id].file)+'</button>').join('')+'</div><p class="integration-current" aria-live="polite"></p>';
  $('editorFrame').before(picker);
  $('restore').hidden=true;
  $('save').textContent='保存当前文件';$('load').textContent='读取当前本地文件';$('run').textContent='检查六个模块的接口升级';
  const identity=()=>book.lessons[selected];
  function showFile(id){
    selected=id;editor.value=draft(id);
    picker.querySelectorAll('[data-module-file]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.moduleFile===id)));
    picker.querySelector('.integration-current').textContent='正在编辑：'+identity().file+' · '+(get('pilot-'+id)!==null?'网页草稿':'本地文件');
    $('localSource').textContent=identity().local_code;
    $('saveNote').textContent='保存位置：exercises/'+identity().file+'；保存前自动备份原文件。';
    if(window.editorHighlight)window.editorHighlight();
    refresh();
  }
  editor.oninput=()=>{preserveLegacy(selected);preservePriorAssessment(selected);put('pilot-'+selected,editor.value);if(window.editorHighlight)window.editorHighlight();refresh();};
  picker.querySelectorAll('[data-module-file]').forEach(b=>b.onclick=()=>showFile(b.dataset.moduleFile));
  $('load').onclick=async()=>{
    const id=selected;
    try{
      const response=await fetch('/api/catalog');if(!response.ok)throw Error('读取失败');
      const fresh=await response.json(),code=fresh.lessons[id].local_code;
      book.lessons[id].local_code=code;preserveLegacy(id);preservePriorAssessment(id);put('pilot-'+id,code);
      if(current===owner&&selected===id)showFile(id);
    }catch(e){if(current===owner)$('saveNote').textContent=e.message;}
  };
  $('save').onclick=async()=>{
    const id=selected,code=editor.value;const button=$('save');button.disabled=true;
    try{
      const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lesson:id,code})}),data=await response.json();
      if(!response.ok||data.error)throw Error(data.error||'保存失败');
      book.lessons[id].local_code=data.code;
      if(current===owner&&selected===id){$('saveNote').textContent='已保存 '+book.lessons[id].file+'；原文件已自动备份。';$('localSource').textContent=data.code;}
    }catch(e){if(current===owner)$('saveNote').textContent=e.message;}
    finally{if(current===owner)button.disabled=false;}
  };
  showFile(selected);
}

async function run(){
  const id=current,meta=m();if(running.has(id))return;running.add(id);$('run').disabled=true;
  preserveLegacy(id);preservePriorAssessment(id);
  let code=draft(id),depVersion=dependencyVersion(id);
  try{
    if(meta.integration_only){
      const fresh=await fetch('/api/catalog').then(r=>{if(!r.ok)throw Error('读取本地模块失败');return r.json();});
      for(const d of [id,...meta.dependencies])book.lessons[d].local_code=fresh.lessons[d].local_code;
      code=draft(id);depVersion=dependencyVersion(id);
    }
    const payload={lesson:id};
    if(meta.kind==='concept')payload.answers=answersNow();
    else{payload.code=code;payload.dependencies=Object.fromEntries((meta.dependencies||[]).map(d=>[d,dependencyCode(id,d)]));payload.auto_save=true;}
    refresh();
    const response=await fetch('/api/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();
    if(meta.kind!=='concept'&&!data.error&&data.rows?.length===meta.cases.length&&data.rows.every(row=>row.passed===true)){
      if(!data.auto_save)data.auto_save={ok:false,error:'本地服务尚未启用自动保存，请重启本地网站服务后重新测评。'};
      if(data.auto_save.ok)for(const item of data.auto_save.updated||[])book.lessons[item.lesson].local_code=item.code;
      if(current===id)$('saveNote').textContent=data.auto_save.ok?'本次通过版本已自动保存到本地，并通过本地运行验证；旧文件已备份。后续编辑需再次测评。':'测评通过，但本地保存未完成：'+data.auto_save.error;
    }
    put('pilot-result-'+id,JSON.stringify({code,version:version(id),dependencies:depVersion,data,when:new Date().toLocaleString()}));
  }catch(e){put('pilot-result-'+id,JSON.stringify({code,version:version(id),data:{error:'检查未完成：'+e.message},when:new Date().toLocaleString()}));}
  finally{running.delete(id);if(current===id){$('run').disabled=false;paintCases();}refresh();}
}

async function save(){
  const id=current,code=$('code').value;$('save').disabled=true;
  try{
    const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lesson:id,code})}),data=await response.json();
    if(!response.ok||data.error)throw Error(data.error||'保存失败');book.lessons[id].local_code=data.code;
    if(current===id){$('saveNote').textContent='已保存到 '+data.path+'；旧文件已自动备份。';$('localSource').textContent=data.code;if(window.PythonEditor)PythonEditor.highlightReadonly($('main'));}
  }catch(e){if(current===id)$('saveNote').textContent=e.message;}
  finally{if(current===id)$('save').disabled=false;}
}

function setupAnimation(){
  if(['t_gpt','t_data','t_optim','t_train','t_checkpoint','t_project','rebuild_gpt'].includes(current))return;
  if(window.AttentionViews?.supports(current))return;
  if(['foundation_capstone','t_softmax','t_loss'].includes(current))return;
  if(current==='index'){window.setupIndexAnimation(example());return;}
  if(window.ProcessAnimations?.supports(current)){ProcessAnimations.mount(current,example(),m().generation_scores);return;}
  if(['linear','layer_linear','layer_scale','t_attention','t_block'].includes(current))return;
  const supported=['axes','broadcast','matmul','reshape','transpose','heads','linear','layer_linear','layer_embedding','layer_gelu','layer_rmsnorm','layer_scale','layer_mlp'];
  if(!supported.includes(current)&&!m().trace?.length)return;
  $('animation').innerHTML='<h3 style="margin-top:24px">逐步演算</h3><div id="viz"></div><div id="explain" class="explanation" aria-live="polite"></div><div class="controls" style="margin-top:12px"><button id="prev">← 上一步</button><button id="next" class="primary">下一步 →</button><button id="play">自动演示</button><button id="reset">重置</button><span id="counter" class="small"></span></div>';
  $('prev').onclick=()=>{step=Math.max(0,step-1);draw();};$('next').onclick=()=>{step=Math.min(maxStep(),step+1);draw();};
  const stop=()=>{clearInterval(timer);timer=null;$('play').textContent='自动演示';};
  $('reset').onclick=()=>{stop();step=0;draw();};$('play').onclick=()=>{if(timer)return stop();if(step===maxStep())step=0;timer=setInterval(()=>{step++;draw();if(step>=maxStep())stop();},1300);$('play').textContent='暂停';};draw();
}
function maxStep(){if(m().trace?.length)return m().trace.length-1;if(current==='axes')return 2;if(current==='matmul')return 6;if(current==='broadcast')return 4;if(current==='layer_embedding')return example().X.flat().length;if(current==='layer_gelu')return 2;if(['linear','reshape','transpose','heads'].includes(current))return 3;return 4;}
function draw(){
  const meta=m(),c=example();let html='',text='';
  if(meta.trace?.length){const frame=meta.trace[step];html=arrayView(frame.label,frame.value);text=frame.text;
  }else if(current==='axes'){
    html=matrix(c.args[0],step?[0,1,2].map(j=>(step-1)+','+j):[])+arrayView('当前计算结果',c.expected.map((v,i)=>i<step?v:'?'));
    text=step?`第 ${step} 行的三个特征相加：${c.args[0][step-1].join(' + ')} = ${c.expected[step-1]}。`:'沿特征轴归约，每条样本得到一个总和。';
  }else if(current==='matmul'||current==='broadcast'){
    const columns=c.expected[0].length,row=Math.floor((step-1)/columns),col=(step-1)%columns;
    html='<div class="matrices">'+matrix(c.args[0],step?(current==='matmul'?[0,1].map(j=>row+','+j):[row+','+col]):[])+`<span>${current==='matmul'?'×':'+'}</span>`+matrix(current==='matmul'?c.args[1]:[c.args[1]],step?(current==='matmul'?[0,1].map(i=>i+','+col):['0,'+col]):[])+'</div>'+arrayView('当前计算结果',c.expected.map((r,i)=>r.map((v,j)=>i*columns+j<step?v:'?')));
    text=!step?'点击下一步，跟踪高亮元素。':current==='matmul'?`输出第 ${row+1} 行、第 ${col+1} 列：${c.args[0][row].map((v,k)=>`${v}×${c.args[1][k][col]}`).join(' + ')} = ${c.expected[row][col]}`:`第 ${row+1} 行、第 ${col+1} 列：${c.args[0][row][col]} + ${c.args[1][col]} = ${c.expected[row][col]}`;
  }else if(current==='heads'){
    const X=c.args[0],B=X.length,T=X[0].length,C=X[0][0].length,H=Number(c.args[1]),d=C/H;
    const split=X.map(batch=>batch.map(row=>Array.from({length:H},(_,h)=>row.slice(h*d,(h+1)*d))));
    const restored=c.expected.map(batch=>Array.from({length:T},(_,t)=>batch.flatMap(head=>head[t])));
    html=arrayView(step===0?'X · (B,T,C)':step===1?'拆分后的 (B,T,H,d)':step===2?'换轴后的 (B,H,T,d)':'还原后的 (B,T,C)',step===0?X:step===3?restored:step===1?split:c.expected,step===1?['batch b','token t','head h','特征 d']:null);
    text=[`输入有 B=${B}、T=${T}、C=${C}，给定 H=${H}。全部 batch 都保留。`,`只拆最后一轴：每连续 ${d} 个特征分给一个 head，共 ${H} 组；当前顺序为 B,T,H,d。`,`交换轴 1 和 2 后为 B,H,T,d。每张切片固定一个 batch 和 head，不是一个二维张量。`,`先 transpose(0,2,1,3) 回到 B,T,H,d，再 reshape(B,T,C)。还原后每个坐标的值与输入相同。`][step];
  }else if(current==='linear'||current==='reshape'||current==='transpose'){
    const X=c.args[0];html=arrayView('输入 X',X);
    if(step)html+=arrayView(current==='transpose'?'转置后的结果':current==='reshape'?'二维表 X2':step<3?'二维表':'恢复后的 Y',current==='linear'?(step===1?X.flat():step===2?c.expected.flat():c.expected):c.expected,current==='transpose'?m().output_axes:null);
    const B=X.length,T=X[0].length;
    if(current!=='transpose'&&step>0&&(current==='reshape'||step<3))html+='<div class="position-map">'+X.flat().map((_,n)=>`<span>行 n=${n} ← b=${Math.floor(n/T)}, t=${n%T}</span>`).join('')+'</div>';
    text=current==='linear'?[`记录 B=${B}、T=${T}、C=${X[0][0].length}，所有 batch 都参与计算。`,`把 B、T 合为 ${B*T} 行，n=b×${T}+t；每个 token 的特征仍在同一行。`,'Y2=X2@W+b，在二维上完成全部位置的计算。','恢复（B,T,D），各行回到自己的 batch、token 位置。'][step]:current==='reshape'?['先确认三轴含义，输入完整显示两个 batch。','每行都显示原 batch、token 坐标；第一批的 token 在前。','第二批接在后面；合并后没有跨 token 混合特征。','保存原形状，反向时每行按 n=b×T+t 回到原位置。'][step]:['轴顺序为（B,T,C），两个 batch 都完整显示。','新轴 0 仍取旧轴 0，batch 不变。','新轴 1 取旧轴 2，新轴 2 取旧轴 1；每个 batch 内单独转置。','元素换位置而不换值；反向应用逆置换。'][step];
  }else if(current==='layer_embedding'){
    const ids=c.X.flat(),G=c.G.flat(),table=c.parameters.weight,partial=table.map(row=>row.map(()=>0));
    for(let n=0;n<step;n++)for(let j=0;j<G[n].length;j++)partial[ids[n]][j]+=G[n][j];
    html=arrayView('token 编号 · 保留 (B,T)',c.X)+arrayView('当前 weight.grad · (V,C)',partial);
    const T=c.X[0].length;
    text=step?`位置 b=${Math.floor((step-1)/T)}, t=${(step-1)%T} 的 G=${JSON.stringify(G[step-1])} 加到编号 ${ids[step-1]} 的参数梯度行。不同 batch 中重复编号也继续相加。`:'前向查表，反向按 batch、token 逐次散射。编号本身的梯度为 None。';
  }else if(current==='layer_gelu'){
    const f=x=>.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x)));
    const points=Array.from({length:121},(_,i)=>{const x=-3+i*.05;return `${30+(x+3)*55},${180-f(x)*45}`;}).join(' ');
    html=`<svg viewBox="0 0 400 230" role="img" aria-label="GELU 函数曲线"><path d="M30 180H370 M195 20V215" stroke="#acb7a4"/><polyline points="${points}" fill="none" stroke="#256b51" stroke-width="3"/><text x="325" y="203">x</text><text x="205" y="30">GELU(x)</text><circle cx="195" cy="180" r="4" fill="#256b51"/></svg>`;
    if(step)html+=arrayView(step===1?'前向 Y':'dX = G ⊙ GELU′(X)',step===1?c.expected.Y:c.expected.dX);
    text=['GELU 对各元素独立作用，负输入不是全部截断成零。','计算 tanh 近似式，输入输出形状相同。','乘积法则和链式法则两项都要保留；零点局部导数为 1/2。'][step];
  }else if(current==='layer_rmsnorm'){
    const means=c.X.map(batch=>batch.map(row=>[row.reduce((s,x)=>s+x*x,0)/row.length])),r=means.map(batch=>batch.map(([v])=>[1/Math.sqrt(v+1e-5)]));
    html=step===0?arrayView('输入 X',c.X):step===1?arrayView('每个 token 的 mean(x²)',means):step===2?arrayView('r=1/√(mean(x²)+ε)',r):step===3?arrayView('输出 Y',c.expected.Y):arrayView('反向 dX',c.expected.dX);
    text=['只在每个 token 内沿特征轴计算。','求均方，不减均值。','加 epsilon 后取平方根的倒数；零输入也保持有限。','按特征乘可训练 w。','梯度有直接路径与经过 r 的间接路径，二者必须相加。'][step];
  }else{
    const output=step<2?c.expected.Y:step===2?c.expected.dX:Object.fromEntries(Object.entries(c.expected).filter(([k])=>k.endsWith('.grad')));
    html=arrayView(step<2?'前向 Y':step===2?'反向 dX':'参数梯度',output);
    text=current==='layer_linear'?['X reshape 成（N,C），N 为所有前置维度的乘积。','Y2=X2W+b；保存 X2 和输入形状。','G2Wᵀ 得到（N,C），再恢复 X 形状。','dW=X2ᵀG2，db=Σ行 G2；参数梯度使用 +=。','新一次梯度累加开始前 zero_grad；backward 本身不更新权重。'][step]:current==='layer_mlp'?['Linear(C→hidden) → GELU → Linear(hidden→C)。','三个子层前向，依次缓存。','从 fc2.backward 到 act.backward，再到 fc1.backward。','两个 Linear 的参数都必须收到梯度。','每个 token 独立通过相同网络；不会在这里交换 token 信息。'][step]:['Y=wX。','缓存 X 以求参数梯度。','dX=G·w。','dw=Σ(G⊙X)，贡献来自所有位置。','累加到 w.grad；清零由外部训练循环执行。'][step];
  }
  $('viz').innerHTML=html;$('explain').textContent=text;$('counter').textContent=`${step}/${maxStep()}`;$('prev').disabled=step===0;$('next').disabled=step===maxStep();
}

function renderTrainingLab(){
  $('main').innerHTML=`<div class="hero"><div><h1>${esc(CourseNavigation.trainingLab.title)}</h1><p>用一段语料训练 miniGPT，再输入文字开头，查看续写结果。</p></div></div><div id="labPage"></div>`;
  setupEditorResize();navigation();
  Laboratory.mount($('labPage'),{...learningContext('t_generate'),sidebar:labSidebar});
  if(window.LearningTools)LearningTools.mount(learningContext());
  if(window.VisualRoadmap)VisualRoadmap.mount(learningContext());
  setupReadingReturn();
}

// 讲义、样例、编辑与测评保持单一内容流，只调整代码框。
function setupEditorResize(){
  if(editorResizeCleanup)editorResizeCleanup();
  editorResizeCleanup=null;
  const code=$('code');if(!code)return;
  const frame=document.createElement('div');frame.className='editor-frame';frame.id='editorFrame';
  code.before(frame);frame.appendChild(code);code.setAttribute('wrap','off');
  const handle=document.createElement('button');handle.type='button';handle.className='editor-resize-handle';handle.textContent='↘';
  handle.setAttribute('aria-label','调整代码编辑框大小');handle.title='拖动调整宽高；方向键微调；Home 恢复默认';frame.appendChild(handle);
  const note=document.createElement('div');note.className='editor-size-note';
  note.innerHTML='<span>拖动右下角调整编辑框，宽度不超过内容区。</span><button type="button">恢复编辑框尺寸</button>';
  frame.after(note);
  const key='single-editor-size-'+current;
  const old=read(key);let ratio=Number.isFinite(old?.ratio)?old.ratio:1;
  let height=Number.isFinite(old?.height)?old.height:380;
  const clamp=(value,lo,hi)=>Math.max(lo,Math.min(hi,value));
  function available(){const style=getComputedStyle(frame.parentElement);return Math.max(120,frame.parentElement.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight));}
  function apply(){const width=available();ratio=clamp(ratio,Math.min(280,width)/width,1);height=clamp(height,160,1600);frame.style.width=width*ratio+'px';frame.style.height=height+'px';}
  const persist=()=>put(key,JSON.stringify({ratio,height}));
  function reset(){ratio=1;height=380;apply();persist();}
  note.querySelector('button').onclick=reset;
  handle.onpointerdown=e=>{
    e.preventDefault();handle.focus();handle.setPointerCapture(e.pointerId);
    const rect=frame.getBoundingClientRect(),x=e.clientX,y=e.clientY;
    handle.onpointermove=move=>{ratio=(rect.width+move.clientX-x)/available();height=rect.height+move.clientY-y;apply();};
    const finish=()=>{handle.onpointermove=null;persist();};handle.onpointerup=finish;handle.onpointercancel=finish;
  };
  handle.onkeydown=e=>{
    if(e.key==='Home'){e.preventDefault();reset();return;}
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
    e.preventDefault();const delta=e.shiftKey?8:32;
    ratio+=(e.key==='ArrowRight'?delta:e.key==='ArrowLeft'?-delta:0)/available();
    height+=e.key==='ArrowDown'?delta:e.key==='ArrowUp'?-delta:0;apply();persist();
  };
  window.addEventListener('resize',apply);editorResizeCleanup=()=>window.removeEventListener('resize',apply);apply();
}

if(window.CourseMath)CourseMath.observe($('main'));
window.addEventListener('popstate',event=>readingHistory?.restore(event.state));
window.addEventListener('pagehide',()=>readingHistory?.save());
fetch('/api/catalog').then(r=>{if(!r.ok)throw Error('catalog');return r.json();}).then(data=>{
 book=data;const valid=id=>Boolean(pageTitle(id));
 labSidebar=Laboratory.mountSidebar($('labSidebar'),{read,put});
 readingHistory=CourseNavigation.createReadingHistory({history:window.history,capture:captureReading,navigate:visitLesson,valid});
 const saved=get('course-current'),entry=readingHistory.start(valid(saved)?saved:'map');
 current=entry.lesson;put('course-current',current);render();restoreReading(entry.reading);
}).catch(e=>{$('main').textContent='无法加载课程。请重新启动本地网站，再刷新页面。';});
