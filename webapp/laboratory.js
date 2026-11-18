/* 可复现实验记录保存在当前浏览器；回放快照不覆盖个人草稿。 */
(function(root){'use strict';
function chart(series,title,{unit='',guide=null}={}){
 const points=series.flatMap(s=>s.points);if(!points.length)return '';
 const maxX=Math.max(1,...points.map(([x])=>x));
 const values=points.map(([,y])=>y);
 if(guide)values.push(guide.value);
 const minY=Math.min(...values),maxY=Math.max(...values);
 const xPosition=x=>60+x/maxX*570;
 const yPosition=y=>maxY===minY?125:210-(y-minY)/(maxY-minY)*170;
 const colors=['#256b51','#a65c37','#536bb0','#a87619'];
 const caption=y=>`${Number(y.toPrecision(3))}${unit}`;
 const lines=series.map((s,i)=>{
  const color=colors[i%colors.length];
  if(s.points.length===1){const [[x,y]]=s.points;return `<circle cx="${xPosition(x)}" cy="${yPosition(y)}" r="3" fill="${color}"/>`;}
  return `<polyline points="${s.points.map(([x,y])=>`${xPosition(x)},${yPosition(y)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.4"/>`;
 }).join('');
 const reference=guide?`<line x1="60" x2="630" y1="${yPosition(guide.value)}" y2="${yPosition(guide.value)}" stroke="#9a6443" stroke-width="1.6" stroke-dasharray="6 4"/>`:'';
 const tick=y=>`<text x="5" y="${yPosition(y)+4}">${esc(caption(y))}</text>`;
 return `<figure class="experiment-chart"><figcaption>${esc(title)}</figcaption><svg viewBox="0 0 660 250" role="img" aria-label="${esc(title)}，横轴训练步数"><path d="M60 25V210H635" fill="none" stroke="#8c9e85"/>${reference}${lines}${tick(maxY)}${minY===maxY?'':tick(minY)}<text x="60" y="235">0</text><text x="590" y="235">${maxX} 步</text></svg><div class="chart-legend">${series.map((s,i)=>`<span style="color:${colors[i%colors.length]}">● ${esc(s.name)}</span>`).join('')}${guide?`<span class="chart-guide-legend">${esc(guide.label)}</span>`:''}</div></figure>`;
}
function trainingDiagnostics(result){
 const steps=values=>values.map((value,index)=>[index+1,value]);
 const activations=Object.keys(result.activation_rms[0]||{}).map(name=>({name,points:steps(result.activation_rms.map(row=>row[name])).filter(([,value])=>Number.isFinite(value))}));
 const gradient=chart([{name:'裁剪前',points:steps(result.norms)}],'全局梯度范数',{guide:{value:1,label:'裁剪阈值 1.0'}});
 const updates=chart([{name:'每步更新',points:steps(result.relative_updates.map(value=>value*100))}],'相对参数更新幅度',{unit:'%'});
 const clippedSteps=result.norms.filter(value=>value>1).length;
 const rmsFormula=window.CourseMath?.block(String.raw`\sqrt{\operatorname{mean}(A^2)}`,true)??'平方后取平均再开方';
 const panel=(title,description,reading,plot)=>`<section class="lab-diagnostic"><div><h3>${title}</h3><p>${description}</p><p class="lab-diagnostic-reading">${reading}</p></div>${plot}</section>`;
 return `<details class="lab-more lab-diagnostics"><summary>训练诊断</summary>
  <p class="small">上方损失在固定窗口上计算，便于比较预测误差的变化。下面沿训练步骤看前向输出、反向梯度和实际更新。横轴都是训练步数。</p>
  ${panel('前向 · 每一层的输出有多大',`每条线对应一层，把这一步的输出张量取 RMS（${rmsFormula}，其中 ${window.CourseMath?.block('A',true)??'A'} 是该层输出）。数值越大，该层输出的整体幅度越大。`,'先在同一步比较各层，再沿一条线看同一层随训练怎样变化。若后面的 Block 持续放大输出，可回看残差、归一化与初始化。',chart(activations,'各层输出的激活 RMS'))}
  ${panel('反向 · 梯度裁剪前有多大','把所有参数的梯度合成一个量。虚线是阈值 1.0。梯度范数超过虚线时，裁剪会把所有梯度等比例缩小，再交给优化器。',`这次有 <strong>${clippedSteps} / ${result.norms.length}</strong> 步超过阈值。对照上方损失，可以看到裁剪频繁时训练是否仍在进展。`,gradient)}
  ${panel('更新 · 参数实际改了多少','把这一步的参数改变量与更新前参数的整体大小相比。比如 0.5% 表示更新量的范数约是原参数范数的 0.5%。','这条线记录优化器实际走了多大一步。如果幅度长时间接近零，且训练损失也几乎不动，可以检查学习率和优化器更新。',updates)}
 </details>`;
}
function progressMarkup(book,status){
 const stages=book.stages.filter(stage=>!stage.paused&&!stage.optional).map(stage=>{
  const ids=stage.modules.flatMap(module=>module.tasks);
  const ready=ids.filter(id=>status(id).kind==='success').length;
  const provided=ids.filter(id=>book.lessons[id].provided).length;
  return {title:stage.title,ready,provided,total:ids.length};
 });
 const readyCount=stages.reduce((sum,stage)=>sum+stage.ready,0);
 const taskCount=stages.reduce((sum,stage)=>sum+stage.total,0);
 const providedCount=stages.reduce((sum,stage)=>sum+stage.provided,0);
 const progress=stages.map(({title,ready,provided,total})=>{
  const label=provided?`${ready}/${total} 就绪，含 ${provided} 项课程提供的工具`:`${ready}/${total} 任务通过`;
  const [part,name]=title.split(' · ');
  return `<div class="lab-progress-item">
    <div class="lab-progress-ring" style="--ready:${100*ready/Math.max(total,1)}%" role="progressbar" aria-label="${esc(title)}" aria-valuemin="0" aria-valuemax="${total}" aria-valuenow="${ready}" aria-valuetext="${esc(label)}">
      <span aria-hidden="true"><b>${ready}</b><small>/${total}</small></span>
    </div>
    ${name?`<small class="lab-progress-part">${esc(part)}</small>`:''}<strong>${esc(name??part)}</strong>
  </div>`;
 }).join('');
 return `<section class="card lab-progress" aria-labelledby="labProgressTitle">
   <div class="lab-progress-heading"><h2 id="labProgressTitle">课程进度</h2><p><strong>${readyCount}</strong> / ${taskCount} 项就绪</p></div>
   <div class="lab-progress-grid">${progress}</div>
   ${providedCount?`<p class="lab-progress-note">含 ${providedCount} 项课程提供的工具。</p>`:''}
 </section>`;
}
function deliveryMarkup(book,reference){
 const plan=book.visual_roadmap?.items?.find(item=>item.id==='logic-part-08-training-reuse');
 const state={todo:'待实现',in_progress:'制作中',review:'待你验收',accepted:'已验收'}[plan?.status]??'待实现';
 const math=tex=>window.CourseMath?.block(tex,true)??'';
 return `<section class="card lab-delivery" aria-labelledby="labDeliveryTitle">
   <div class="lab-delivery-heading"><h2 id="labDeliveryTitle">成果交付</h2><span class="small">个人训练程序 · ${state}</span></div>
   <p>在 <code>train_my_gpt.py</code> 中把你写好的组件接成一个程序。先用本地命令运行，再把同一个入口接到网页。</p>
   <h3>输入和输出</h3>
   <ul><li>训练接收英文语料、训练步数和运行目录，输出每步损失，并保存词表与检查点。</li><li>续训读取原语料和运行目录，从保存的步数继续更新。</li><li>生成接收运行目录、一个或多个英文提示词和新增字符数，输出保留提示词的完整续写。</li></ul>
   <h3>动手清单</h3>
   <ol>
     <li><strong>训练并保存。</strong>从已有的 <code>train</code> 抽出一次更新供两处共用。按批读取英文语料，用同一组模型和优化器更新，保存检查点。</li>
     <li><strong>恢复并续训。</strong>恢复参数、优化器和抽批次的随机状态。连训五步与先训三步、重开后再训两步应得到一致结果。</li>
     <li><strong>加载并生成。</strong>用同一个检查点续写多个英文提示词。网页展示累计步数、损失和生成结果。</li>
   </ol>
   <h3>关键公式速查</h3>
   <p>${math('X')} 是输入批次，${math('Y')} 是右移一格的目标。对第 ${math('b')} 行，${math('i_b')} 是语料中的窗口起点，${math('T')} 是窗口长度。${math(String.raw`X_{b,:}=\mathrm{ids}[i_b:i_b+T],\quad Y_{b,:}=\mathrm{ids}[i_b+1:i_b+T+1]`)}。每步返回更新前的交叉熵 ${math(String.raw`L=\operatorname{CrossEntropy}(\operatorname{GPT}(X),Y)`)}。</p>
   ${reference?`<details class="lab-more lab-reference"><summary>参考代码 · 从英文语料到续写</summary>
     <h4>先看这三步</h4>
     <p>下面是从项目根目录调用参考程序的 Python 示例。先读入训练文本，再训练并保存，最后用检查点接着写 <code>the</code>。</p>
     <pre data-language="python" aria-label="Python 调用示例"><code>from pathlib import Path
from solutions.train_my_gpt_reference import train_text, continue_text

text = Path("data/corpus.txt").read_text(encoding="utf-8")
run_dir = Path("runs/from_python")
train_text(text, run_dir, steps=3)
output = continue_text(run_dir, ["the "], steps=24)
print(output[0])</code></pre>
     <details class="lab-more">
       <summary>在终端运行 · 训练、续训、生成</summary>
       <p>从项目根目录运行。先训练 3 步并保存，重开程序续训 2 步，再加载同一检查点生成 24 个字符。第一次训练用新目录，续训和生成沿用它。</p>
       <pre class="lab-terminal" aria-label="终端命令"><code>python solutions/train_my_gpt_reference.py train runs/example --text data/corpus.txt --steps 3
python solutions/train_my_gpt_reference.py resume runs/example --text data/corpus.txt --steps 2
python solutions/train_my_gpt_reference.py generate runs/example --prompt "the " --steps 24</code></pre>
     </details>
     <details class="lab-more">
       <summary>完整 Python 源码 · <code>train_my_gpt_reference.py</code></summary>
       <p>从 <code>train_text</code> 和 <code>continue_text</code> 读起。开头的 <code>sys.path</code> 帮脚本找到 <code>exercises</code> 中的课程模块，末尾的 <code>main</code> 把终端参数传给这两个函数。</p>
       <pre data-language="python" class="lab-full-source" aria-label="完整 Python 源码" tabindex="0"><code>${esc(reference)}</code></pre>
     </details>
   </details>`:''}
 </section>`;
}
const storyCore='the cat sits by the river.\nthe dog runs by the river.\nthe cat runs by the tree.\nthe dog rests by the tree.\n';
const storyValidation='the dog sits by the tree.\nthe cat rests by the river.\n';
const storyExtra='the rabbit hops by the pond.\nthe fox walks by the bridge.\nthe bird sings by the lake.\nthe duck swims by the pond.\nthe horse walks by the field.\nthe goat rests by the fence.\n';
const story={text:(storyCore.repeat(4)+storyExtra).repeat(2),
 validation_text:(storyValidation+'the rabbit walks by the pond.\n').repeat(3),prompts:['the cat sits']};
const oldStory='清晨，小猫来到河边。小猫看见了鱼，鱼游向桥下。午后，小狗沿着河边跑，小猫坐在桥边等它。夜晚，它们一起回家。'.repeat(5);
const oldValidation='午后，小猫沿着河边跑。小狗坐在桥下等它。小猫看见了鱼，夜晚它们一起回家。'.repeat(3);
const presetValidationCorpora=new Map([[story.validation_text,story.text],[storyValidation.repeat(3),storyCore.repeat(8)],[oldValidation,oldStory]]);
const storySettings={C:32,H:2,T:16,B:4,hidden:32,layers:1,steps:300,lr:0.01,temperature:0.5,top_k:3,tokens:13};
const repeated={text:'the cat naps.\nthe dog runs.\n'.repeat(20),validation_text:'',prompts:['the cat']};
const maxPrompts=3;
const parameterGroups=[
 {title:'训练设置',hint:'控制更新次数、更新幅度，以及每一步取多少个样本。',fields:[
  ['steps','训练步数',60,1,300],['lr','学习率',.01,.00001,.1,'any'],['B','批次大小',2,2,8]
 ]},
 {title:'模型结构',hint:'设置模型的规模和一次能读取的字符数。',fields:[
  ['C','模型宽度',8,4,32],['H','注意力头数',2,1,8],['T','上下文长度',8,2,24],
  ['hidden','MLP 隐藏宽度',32,4,128],['layers','Block 层数',1,1,3]
 ]},
 {title:'续写采样',hint:'温度越低越偏向高分字符。候选数设为 0 时保留整个词表。',fields:[
  ['temperature','采样温度',.8,.05,3,'any'],['top_k','保留候选数',0,0,null],['tokens','新增字符数',24,1,60]
 ]}
];
function replaceLegacyPreset(form){
 if(Array.isArray(form.prompts))return form;
 const stockPrompts=form.prompt===story.prompts[0]&&form.comparison_prompt==='the dog sits';
 if(stockPrompts&&form.text===storyCore.repeat(8)&&form.validation_text===storyValidation.repeat(3)){
  const previous={...storySettings,C:24,T:12,B:2,seed:7,temperature:0.8,top_k:1,tokens:12};
  if(Object.entries(previous).every(([key,value])=>form[key]==null||Number(form[key])===value))return {...form,...story,...storySettings};
 }
 if(stockPrompts&&form.text===story.text&&form.validation_text===story.validation_text)return {...form,prompts:story.prompts};
 if(form.comparison_prompt)return form;
 const {text,validation_text='',prompt=''}=form;
 const oldRepeated='春天来了。小猫喜欢鱼。小狗喜欢跑。我们一起学习。'.repeat(20);
 if(text===oldStory&&['',oldValidation].includes(validation_text)&&['','清晨，小猫'].includes(prompt))return {...form,...story};
 if(text===oldRepeated&&validation_text===''&&prompt==='春天来了。')return {...form,...repeated};
 return form;
}
function characterLabel(char){return ({' ':'空格','\n':'换行','\t':'制表符'})[char]??char;}
function probabilityMarkup(pair,cursor){
 const round=Math.max(cursor-1,0),step=pair.steps[round];
 if(!Array.isArray(step?.candidates))return '<p class="small">这次记录没有采样概率。</p>';
 const bar=(label,value,chosen=false)=>{
  const percent=Math.max(0,Math.min(1,Number(value)||0))*100;
  const display=percent>0&&percent<0.05?'&lt;0.1%':`${percent.toFixed(1)}%`;
  return `<div class="lab-prob-row${chosen?' is-chosen':''}"><span>${esc(label)}</span><span class="lab-prob-track" aria-hidden="true"><i style="width:${percent}%"></i></span><span>${display}</span></div>`;
 };
 const bars=step.candidates.filter(item=>typeof item?.char==='string'&&Number.isFinite(item.probability))
  .map(({char,probability})=>bar(characterLabel(char),probability,cursor>0&&char===step.chosen)).join('');
 const other=Number(step.other)>1e-8?bar('其余合计',step.other):'';
 return `<h4>第 ${round+1} 个字符</h4><p class="small">${cursor?`抽中 ${esc(characterLabel(step.chosen))}`:'等待抽样'}</p>${bars}${other}`;
}
function generationMarkup(pair,description){
 const count=pair.suffix.length;
 return `<section class="lab-generation-card">
  <div class="lab-generation-heading"><h3>${esc(pair.label)}</h3><span class="small">${description}</span></div>
  <p class="generated-text" aria-live="off"><span>${esc(pair.prefix)}</span><b data-lab-suffix></b><mark data-lab-character hidden></mark></p>
  ${count?`<div class="controls lab-replay-controls"><button type="button" data-step-back>上一步</button><button type="button" data-step-next>下一字符</button><button type="button" data-step-play>播放</button><button type="button" data-step-end>看完整续写</button><output aria-label="已显示的新增字符数">0 / ${count}</output></div>`:''}
  <details class="lab-more"><summary>下一个字符的概率分布</summary><p class="small">显示实际抽样概率。点「下一字符」查看这一轮选中了谁。</p><div class="lab-probability-distribution" aria-live="off"></div></details>
 </section>`;
}
function mountGeneration(card,pair,pauseAll,autoPlay){
 const find=selector=>card.querySelector(selector),count=pair.suffix.length;
 const previous=find('[data-step-back]'),next=find('[data-step-next]');
 const play=find('[data-step-play]'),end=find('[data-step-end]'),progress=find('output');
 const suffix=find('[data-lab-suffix]'),current=find('[data-lab-character]');
 const probabilities=find('details'),distribution=find('.lab-probability-distribution');
 let cursor=0,timer=null;
 function pause(){
  if(timer!==null)clearInterval(timer);
  timer=null;
  if(play)play.textContent='播放';
 }
 function renderStep(){
  suffix.textContent=pair.suffix.slice(0,Math.max(0,cursor-1)).join('');
  current.hidden=cursor===0;
  current.textContent=pair.suffix[cursor-1]??'';
  if(count){
   progress.textContent=`${cursor} / ${count}`;
   previous.disabled=cursor===0;
   next.disabled=end.disabled=cursor===count;
  }
  if(probabilities.open)distribution.innerHTML=probabilityMarkup(pair,cursor);
 }
 function start(){
  if(cursor===count){cursor=0;renderStep();}
  play.textContent='暂停';
  timer=setInterval(()=>{
   if(!card.isConnected){pause();return;}
   cursor++;
   renderStep();
   if(cursor===count)pause();
  },320);
 }
 if(count){
  previous.onclick=()=>{pauseAll();cursor--;renderStep();};
  next.onclick=()=>{pauseAll();cursor++;renderStep();};
  end.onclick=()=>{pauseAll();cursor=count;renderStep();};
  play.onclick=()=>{if(timer!==null)pause();else{pauseAll();start();}};
  if(autoPlay)start();
 }
 probabilities.ontoggle=()=>{if(probabilities.open){pauseAll();renderStep();}};
 renderStep();
 return pause;
}
function mountSidebar(sidebar,{read,put}){
 sidebar.innerHTML=`
  <h2 class="lab-settings-heading" id="labSettingsTitle"><button type="button" id="labSettingsToggle" aria-controls="labSettingsForm" aria-expanded="true"><span class="lab-settings-title">实验设置</span><span class="lab-settings-arrow" aria-hidden="true">›</span></button></h2>
  <form id="labSettingsForm" class="lab-settings-form">
   <div class="lab-settings-switch" role="group" aria-label="实验设置类别">
    <button type="button" id="labCorpusSelect" aria-pressed="true" aria-controls="labCorpusPanel">训练语料</button>
    <button type="button" id="labParametersSelect" aria-pressed="false" aria-controls="labParametersPanel">训练参数</button>
   </div>
   <div class="lab-settings-body" id="labCorpusPanel">
    <section class="lab-settings-group">
     <div class="controls lab-presets"><button type="button" id="loadNarrative">载入相似情景样例</button><button type="button" id="loadRepeated">载入英文短句</button></div>
     <label for="labText" class="lab-field-label">英文语料</label>
     <textarea id="labText" name="text" rows="9" aria-describedby="labTextError" required></textarea>
     <span id="labTextError" class="lab-setting-error" hidden></span>
     <p id="labValidationMode" class="small" role="status"></p>
     <details class="lab-more">
      <summary>独立验证文本（可选）</summary>
      <p id="labValidationHint" class="small">填完整英文句子，用已知的下一字符算验证损失，权重保持原样。字符来自训练词表。留空时用语料前 80% 训练，后 20% 算损失。</p>
      <textarea id="labValidation" name="validation_text" rows="4" aria-label="独立验证文本" aria-describedby="labValidationHint labValidationError"></textarea>
      <span id="labValidationError" class="lab-setting-error" hidden></span>
     </details>
    </section>
   </div>
   <div class="lab-settings-body" id="labParametersPanel" hidden>
    ${parameterGroups.map(({title,hint,fields})=>`<section class="lab-settings-group">
     <h3>${title}</h3><p class="small">${hint}</p>
     <div class="lab-fields">${fields.map(([name,label,value,min,max,step=1])=>`<div class="lab-field">
      <label for="labParam-${name}">${label}</label>
      <input type="number" id="labParam-${name}" name="${name}" value="${value}" min="${min}" ${max==null?'':`max="${max}"`} step="${step}" aria-describedby="labRange-${name} labError-${name}" required>
      <span class="small" id="labRange-${name}"></span><span id="labError-${name}" class="lab-setting-error" hidden></span>
     </div>`).join('')}</div>
    </section>`).join('')}
   </div>
   <p class="small lab-settings-footer">修改自动保存，下次运行时生效。</p>
  </form>`;

 const $=id=>sidebar.querySelector('#'+id),form=$('labSettingsForm'),toggle=$('labSettingsToggle');
 const corpusPanel=$('labCorpusPanel'),parametersPanel=$('labParametersPanel');
 const draftKey='gpt-lab-form-t_generate',viewKey='gpt-lab-sidebar';
 const saved=read(draftKey)??read('gpt-lab-form-t_train');
 const values=saved?replaceLegacyPreset(saved):{...story,...storySettings};
 let sampleValidation=values.sample_validation??presetValidationCorpora.has(values.validation_text);
 const view=read(viewKey)??{};
 const controller={
  prompts:values.prompts??[values.prompt,values.comparison_prompt].filter(Boolean),
  settings:()=>Object.fromEntries(new FormData(form)),
  validate(){updateValidity();return form.reportValidity();},
  onChange:null,
  savePrompts(prompts){controller.prompts=prompts;saveDraft();}
 };
 function fillSettings(values){for(const input of form.elements)if(input.name&&Object.hasOwn(values,input.name))input.value=values[input.name];}
 function saveDraft(){put(draftKey,JSON.stringify({...controller.settings(),prompts:controller.prompts,sample_validation:sampleValidation}));}
 function clearDetachedSampleValidation(){
  const {text,validation_text:validation}=form.elements;
  if(sampleValidation&&presetValidationCorpora.get(validation.value)!==text.value){
   validation.value='';
   sampleValidation=false;
  }
 }
 const parameterInputs=parameterGroups.flatMap(group=>group.fields.map(([name])=>form.elements[name]));
 const minWindow=Number(form.elements.T.min),maxWindow=Number(form.elements.T.max);
 function setError(input,errorNode,message){
  input.setCustomValidity(message);
  if(message)input.setAttribute('aria-invalid','true');
  else input.removeAttribute('aria-invalid');
  errorNode.textContent=message;
  errorNode.hidden=!message;
 }
 function updateValidity(){
  const {text,validation_text:validation,C,H,T,top_k:topK}=form.elements;
  const corpusLength=Array.from(text.value).length,heldOut=Array.from(validation.value).length;
  const vocabulary=new Set(text.value);
  const trainLength=heldOut?corpusLength:Math.floor(corpusLength*.8);
  const validationLength=heldOut||corpusLength-trainLength;
  T.max=Math.max(minWindow,Math.min(maxWindow,trainLength-1,validationLength-1));
  topK.max=vocabulary.size;
  $('labValidationMode').textContent=sampleValidation?'使用样例自带的验证文本。换语料会自动切换为划分验证。':heldOut?'使用下方填写的独立验证文本。':'自动用语料末尾 20% 作验证。';

  for(const [input,length,min,max,errorNode] of [[text,corpusLength,40,10000,$('labTextError')],[validation,heldOut,20,4000,$('labValidationError')]]){
   let message=(input.required||length>0)&&(length<min||length>max)?`请输入 ${min}–${max} 个字符。`:'';
   if(!message&&input===validation&&heldOut){
    const unknown=[...new Set(validation.value)].filter(char=>!vocabulary.has(char));
    if(unknown.length)message=`独立验证文本里的 ${unknown.slice(0,12).map(characterLabel).join('、')} 未出现在训练语料中。请修改这段文本，或留空以自动划分。`;
   }
   setError(input,errorNode,message);
  }
  for(const input of parameterInputs){
   const {name,min,max,step,validity}=input;
   const range=min===max?min:`${min}–${max}`;
   $('labRange-'+name).textContent=`范围 ${range}`;
   let message='';
   if(validity.badInput||validity.valueMissing)message='请输入数值。';
   else if(input===T&&validity.rangeOverflow&&Number(max)<maxWindow)message=`按当前训练和验证文本长度，最多 ${max} 个字符。`;
   else if(input===topK&&validity.rangeOverflow)message=`当前词表只有 ${max} 个字符。`;
   else if(validity.rangeUnderflow||validity.rangeOverflow)message=`请输入范围内的${step==='1'?'整数':'数值'}。`;
   else if(validity.stepMismatch)message='请输入整数。';
   else if(input===H&&C.validity.valid&&C.valueAsNumber%H.valueAsNumber!==0){
    const heads=Array.from({length:Number(H.max)},(_,i)=>i+1).filter(n=>C.valueAsNumber%n===0);
    message=`宽度 ${C.valueAsNumber} 可搭配 ${heads.join('、')} 个头。`;
   }
   setError(input,$('labError-'+name),message);
  }
 }
 function applyView(){
  sidebar.toggleAttribute('data-collapsed',!!view.collapsed);
  toggle.setAttribute('aria-expanded',String(!view.collapsed));
  form.hidden=!!view.collapsed;
  corpusPanel.hidden=!!view.parameters;
  parametersPanel.hidden=!view.parameters;
  $('labCorpusSelect').setAttribute('aria-pressed',String(!view.parameters));
  $('labParametersSelect').setAttribute('aria-pressed',String(!!view.parameters));
 }
 function setView(changes){Object.assign(view,changes);applyView();put(viewKey,JSON.stringify(view));}
 function loadPreset(preset){
  fillSettings(preset);
  sampleValidation=presetValidationCorpora.has(preset.validation_text);
  updateValidity();
  controller.prompts=preset.prompts;
  saveDraft();
  controller.onChange?.(true);
 }
 toggle.onclick=()=>setView({collapsed:!view.collapsed});
 $('labCorpusSelect').onclick=()=>setView({parameters:false});
 $('labParametersSelect').onclick=()=>setView({parameters:true});
 $('loadNarrative').onclick=()=>loadPreset({...story,...storySettings});
 $('loadRepeated').onclick=()=>loadPreset(repeated);
 form.oninput=event=>{
  if(event.target===form.elements.validation_text)sampleValidation=false;
  clearDetachedSampleValidation();
  updateValidity();saveDraft();controller.onChange?.();
 };
 form.onsubmit=event=>event.preventDefault();
 form.addEventListener('invalid',event=>{
  if(event.target===Array.from(form.elements).find(control=>control.willValidate&&!control.validity.valid)){
   setView({collapsed:false,parameters:parametersPanel.contains(event.target)});
   event.target.closest('details')?.setAttribute('open','');
  }
 },true);
 fillSettings(values);
 clearDetachedSampleValidation();
 updateValidity();
 applyView();
 return controller;
}
function mount(container,{meta,current,read,put,draft,book,go,status,sidebar}){
 container.className='training-lab';
 container.innerHTML=`
 ${progressMarkup(book,status)}
 <section class="card lab-workspace" aria-labelledby="labWorkspaceTitle">
   <h2 id="labWorkspaceTitle">英文续写实验</h2>
   <p class="small">选一份语料，填入开头，再运行训练和续写。</p>
   <form id="labForm">
     <p id="labDataSummary" class="small"></p>
     <section class="lab-input-panel" aria-labelledby="labPromptsTitle">
       <h3 id="labPromptsTitle">模型接什么</h3>
       <p id="labPromptHint" class="small">同一个模型会分别续写每条开头。全部留空时，从验证文本取一段作为开头。</p>
       <div id="labPromptList" class="lab-prompt-list"></div>
       <div class="lab-prompt-actions">
         <button type="button" id="labAddPrompt">＋ 添加开头</button>
         <p class="small">最多 ${maxPrompts} 条，每条使用语料里已有的字符。</p>
       </div>
       <details class="lab-more" id="labVocabulary"><summary>词汇库</summary><p class="small">点击英文词可插入最近使用的开头。模型按字符生成，下面也列出了可用字符。</p><div id="labVocabularyList"></div></details>
     </section>
     <div class="lab-run-row">
       <button type="submit" class="primary" id="labRun">训练并续写</button>
       <span class="small">每次自动抽取随机种子，从头训练一个模型。</span>
     </div>
   </form>
   <details class="lab-more"><summary>从输入文本到输出文本怎么调用</summary>
     <ol><li><code>tokenize</code> 把英文字符转成编号，<code>make_batch</code> 取输入窗口和右移一格的目标。</li><li>平台循环调用 <code>GPT</code>、<code>CrossEntropy</code>、<code>clip_grad</code> 和 <code>AdamW</code>，用下一字符的预测误差更新权重。</li><li>把每条开头编码后分别调用同一个模型的 <code>generate</code>，再用词表把输出编号还原为英文。</li></ol>
     <p class="small">每次运行使用你当前的课程代码，网页草稿优先。运行结束后可以展开查看实际执行的 Python 源码。</p>
     <div class="controls">${[['t_data','语料编码与窗口'],['t_gpt','GPT 模型'],['t_optim','梯度裁剪与优化器'],['t_generate','续写生成']].map(([id,label])=>`<button type="button" data-lab-lesson="${id}">${label} ↗</button>`).join('')}</div>
   </details>
 </section>
 <section class="card" id="labResult" aria-live="polite" hidden></section>
 <details class="card"><summary>历史实验与复现</summary>
   <p class="small">回放使用当次源码和配置，保存在当前浏览器的个人草稿仍可继续编辑。</p>
   <div id="labHistory"></div>
   <label class="lab-text-label" for="importExperiment">导入实验记录 JSON</label>
   <p class="small" id="importHint">导入后先预览，点击回放时才会运行快照中的源码。</p>
   <input type="file" id="importExperiment" aria-describedby="importHint" accept="application/json,.json">
   <p id="importNote" class="small" aria-live="polite"></p>
 </details>
 ${deliveryMarkup(book,meta.lab_reference)}`;
 const $=id=>container.querySelector('#'+id),form=$('labForm');
 const promptList=$('labPromptList');
 let targetPrompt,promptId=0;
 function updatePromptLabels(){
  const rows=Array.from(promptList.children);
  $('labAddPrompt').disabled=rows.length>=maxPrompts;
  rows.forEach((row,index)=>{
   row.querySelector('label').textContent=rows.length===1?'续写开头':`续写开头 ${index+1}`;
   const remove=row.querySelector('button');
   remove.hidden=rows.length===1;
   remove.setAttribute('aria-label',`移除开头 ${index+1}`);
  });
 }
 function appendPrompt(value=''){
  const row=document.createElement('div');
  const id=`labPrompt-${++promptId}`;
  row.className='lab-prompt-row';
  row.innerHTML=`<label for="${id}" class="lab-field-label"></label><button type="button" class="lab-remove-prompt">移除</button><textarea id="${id}" name="prompts" rows="2" placeholder="the cat sits" aria-describedby="labPromptHint"></textarea>`;
  const input=row.querySelector('textarea');
  input.value=value;
  input.onfocus=()=>{targetPrompt=input;};
  row.querySelector('button').onclick=()=>{
   const neighbor=row.nextElementSibling||row.previousElementSibling;
   row.remove();
   neighbor.querySelector('textarea').focus();
   updatePromptLabels();
   form.oninput();
  };
  promptList.append(row);
  return input;
 }
 function fillPrompts(entries){
  promptList.replaceChildren();
  if(entries.length)for(const value of entries)appendPrompt(value);
  else appendPrompt();
  targetPrompt=promptList.querySelector('textarea');
  updatePromptLabels();
 }
 const readPrompts=()=>new FormData(form).getAll('prompts');
 fillPrompts(sidebar.prompts);
 function updateDataSummary(settings){
  const text=Array.from(settings.text).length;
  const validation=Array.from(settings.validation_text).length;
  const training=validation?text:Math.floor(text*0.8);
  $('labDataSummary').textContent=`训练 ${training} 字符 · ${validation?'独立验证':'从语料末尾验证'} ${validation||text-training} 字符`;
 }
 const vocabulary=$('labVocabulary');
 function updateVocabulary(text=sidebar.settings().text){
  if(!vocabulary.open)return;
  const words=[...new Set(text.match(/[A-Za-z]+(?:['-][A-Za-z]+)*/g)||[])].sort((a,b)=>a.localeCompare(b,'en'));
  const chars=[...new Set(text)].sort();
  $('labVocabularyList').innerHTML=`<p class="small">语料中的 ${words.length} 个英文词</p><div class="lab-vocab-words">${words.map(word=>`<button type="button" data-lab-word="${esc(word)}">${esc(word)}</button>`).join('')}</div><p class="small lab-vocab-chars">${chars.length} 个可用字符 · ${chars.map(char=>esc(characterLabel(char))).join('　')}</p>`;
 }
 vocabulary.ontoggle=()=>updateVocabulary();
 $('labVocabularyList').onclick=event=>{
  const word=event.target.closest('[data-lab-word]')?.dataset.labWord;
  if(!word)return;
  const {value,selectionStart:start,selectionEnd:end}=targetPrompt;
  const before=/[A-Za-z]$/.test(value.slice(0,start))?' ':'';
  const after=/^[A-Za-z]/.test(value.slice(end))?' ':'';
  targetPrompt.setRangeText(before+word+after,start,end,'end');
  targetPrompt.focus();
  form.oninput();
 };
 sidebar.onChange=replacePrompts=>{
  if(replacePrompts)fillPrompts(sidebar.prompts);
  const settings=sidebar.settings();
  updateDataSummary(settings);
  updateVocabulary(settings.text);
 };
 sidebar.onChange();
 form.oninput=()=>sidebar.savePrompts(readPrompts());
 $('labAddPrompt').onclick=()=>{const input=appendPrompt();updatePromptLabels();input.focus();form.oninput();};
 window.PythonEditor?.highlightReadonly(container);
 container.querySelectorAll('[data-lab-lesson]').forEach(button=>button.onclick=()=>go(button.dataset.labLesson));
 function download(record){const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='minigpt-experiment-'+record.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 let stopReplay=()=>{};
 function show(record,comparison=null){
 stopReplay();
 const r=record.result,p=r.provenance;
 const text=String(record.config.text??'');
 const trainingText=record.config.validation_text?text:Array.from(text).slice(0,r.split?.boundary??undefined).join('');
 const seen=text=>text?(trainingText.includes(text)?'训练文本里出现过':'训练文本中的新组合'):'从验证文本选取开头';
 const runSeed=record.config.seed==null?'':` · 随机种子 ${esc(record.config.seed)}`;
 let html=`<h2>训练结果</h2><dl class="lab-summary"><div><dt>训练损失</dt><dd>${r.evaluation[0].train.toFixed(4)} → ${r.evaluation.at(-1).train.toFixed(4)}</dd></div><div><dt>验证损失</dt><dd>${r.validation_before.toFixed(4)} → ${r.validation_after.toFixed(4)}</dd></div></dl><p class="small">${record.replay?'历史源码回放':'平台实验'} · ${esc(record.when)} · 训练耗时 ${r.seconds.toFixed(2)} 秒${runSeed}</p>`;
 html+=chart([{name:'训练固定窗口',points:r.evaluation.map(p=>[p.step,p.train])},{name:'验证固定窗口',points:r.evaluation.map(p=>[p.step,p.validation])}],'训练与验证损失');
 const outputs=r.generated.map((output,i)=>({output,entered:record.config.prompts?.[i]??record.config.prompt}));
 if(typeof r.comparison==='string')outputs.push({output:r.comparison,entered:record.config.comparison_prompt});
 const replayPairs=outputs.map(({output,entered},i)=>{
  const label=outputs.length===1?'续写开头':`续写 ${i+1}`;
  const trace=r.generation_trace?.[i];
  const prefix=trace?.prefix??entered??Array.from(output).slice(0,Math.min(3,record.config.T??3)).join('');
  const matches=typeof prefix==='string'&&output.startsWith(prefix);
  const suffix=Array.from(matches?output.slice(prefix.length):output);
  const steps=matches&&Array.isArray(trace?.steps)&&trace.steps.length===suffix.length?trace.steps:[];
  return {prefix:matches?prefix:'',suffix,steps,label,entered};
 });
 html+=`<h2>续写结果</h2><div class="lab-generation-list">${replayPairs.map(pair=>generationMarkup(pair,seen(pair.entered))).join('')}</div>`;
 if(Array.isArray(r.batch_preview))html+=`<details class="lab-more"><summary>查看首批输入和右移目标</summary><table class="lab-batch"><thead><tr><th>输入</th><th>目标</th></tr></thead><tbody>${r.batch_preview.map(row=>`<tr><td><code>${esc(row.input)}</code></td><td><code>${esc(row.target)}</code></td></tr>`).join('')}</tbody></table></details>`;
 if(typeof r.pipeline_source==='string')html+=`<details class="lab-more"><summary>实际运行的 Python 源码</summary><p><code>_transformer_lab.py</code> 负责本次从语料到续写的完整调度。</p><pre data-language="python" class="lab-full-source" aria-label="实际运行的 Python 源码" tabindex="0"><code>${esc(r.pipeline_source)}</code></pre></details>`;
 if(comparison)html+=`<p class="${comparison.ok?'pass':'fail'}">${comparison.ok?'✓ 损失、验证点、梯度、更新幅度和生成文本与原记录一致。':'回放结果有差异。可展开完整实验记录，核对运行环境与源码。'}</p>`;
 html+=trainingDiagnostics(r);
 html+=`<details class="lab-more"><summary>完整实验记录</summary><p class="small">参数 ${r.parameters} · 词表 ${r.vocab} · 训练 ${r.train_chars} 字 · 验证 ${r.validation_chars} 字。 ${esc(r.split.policy)} ${esc(r.split.vocab_policy||'')}</p><p class="small">源码 ${esc(p.source_sha256.slice(0,12))} · 数据 ${esc(p.data_sha256.slice(0,12))} · Python ${esc(p.python)} / NumPy ${esc(p.numpy)}</p><button type="button" id="exportLast">导出完整实验记录 JSON</button><pre>${esc(JSON.stringify(record,null,2))}</pre></details>`;
 $('labResult').hidden=false;$('labResult').innerHTML=html;$('exportLast').onclick=()=>download(record);
 const pauseCallbacks=[];
 stopReplay=()=>pauseCallbacks.forEach(pause=>pause());
 const autoPlay=replayPairs.length===1&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches;
 $('labResult').querySelectorAll('.lab-generation-card').forEach((card,i)=>{
  pauseCallbacks.push(mountGeneration(card,replayPairs[i],stopReplay,autoPlay));
 });
 window.PythonEditor?.highlightReadonly($('labResult'));
 }
 function history(){const rows=read('gpt-experiment-history')||[];$('labHistory').innerHTML=rows.length?'<h3>最近五次实验</h3><div class="array-block"><table><thead><tr><th>C/H/T/层</th><th>步数 / 学习率</th><th>验证损失 前→后</th><th>记录</th></tr></thead><tbody>'+rows.map((r,i)=>`<tr><td>${esc([r.config.C,r.config.H,r.config.T,r.config.layers].join('/'))}</td><td>${esc(r.config.steps)} / ${esc(r.config.lr)}</td><td>${r.validation_before.toFixed(3)} → ${r.validation_after.toFixed(3)}</td><td>${r.snapshot?`<button type="button" data-show="${i}">查看</button> <button type="button" data-replay="${i}">回放源码快照</button>`:'旧记录仅保存指标'}</td></tr>`).join('')+'</tbody></table></div>':'';
 container.querySelectorAll('[data-show]').forEach(el=>el.onclick=()=>{show(rows[+el.dataset.show]);$('labResult').scrollIntoView({block:'start'});});container.querySelectorAll('[data-replay]').forEach(el=>el.onclick=()=>run(rows[+el.dataset.replay]));}
 function equal(a,b){const metrics=r=>[r.losses,r.norms,r.relative_updates,r.evaluation,r.activation_rms,r.generated,r.comparison??null];return JSON.stringify(metrics(a))===JSON.stringify(metrics(b));}
 async function run(previous=null){
 const config=previous?.config??{...sidebar.settings(),prompts:readPrompts(),seed:Math.floor(Math.random()*100000)};
 const snapshot=previous?.snapshot||{lesson:current,code:draft(current),dependencies:Object.fromEntries(meta.experiment_dependencies.map(d=>[d,draft(d)])),assessment_versions:Object.fromEntries(meta.experiment_dependencies.map(d=>[d,book.lessons[d].assessment_revision||1]))};
 stopReplay();
 $('labRun').disabled=true;container.querySelectorAll('[data-replay]').forEach(b=>b.disabled=true);$('labResult').hidden=false;$('labResult').textContent=previous?'正在使用历史源码快照回放…':'正在训练，完成后会在这里显示续写结果…';
 $('labResult').scrollIntoView({block:'nearest'});
 try{const response=await fetch('/api/experiment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lesson:snapshot.lesson,code:snapshot.code,dependencies:snapshot.dependencies,config})}),result=await response.json();
 if(result.error)throw Error(result.error);
 if(!response.ok)throw Error(`请求失败（${response.status}）`);
 const record={id:Date.now(),when:new Date().toLocaleString(),replay:!!previous,config:result.config,snapshot,result,validation_before:result.validation_before,validation_after:result.validation_after};
 const rows=read('gpt-experiment-history')||[];rows.unshift(record);put('gpt-experiment-history',JSON.stringify(rows.slice(0,5)));
 if(container.isConnected){show(record,previous?{ok:equal(result,previous.result)}:null);history();}
 }catch(e){$('labResult').innerHTML='<h2>实验未完成</h2><pre>'+esc(e.message)+'</pre>';}finally{$('labRun').disabled=false;container.querySelectorAll('[data-replay]').forEach(b=>b.disabled=false);}}
 function validateRecord(r){const out=r?.result;if(!r?.config||!book.lessons[r.snapshot?.lesson]?.lab||typeof r.snapshot.code!=='string'||!r.snapshot.dependencies||typeof r.snapshot.dependencies!=='object'||!out?.provenance)throw Error('不是受支持的完整实验记录。');
 for(const key of ['losses','norms','relative_updates'])if(!Array.isArray(out[key])||!out[key].length||!out[key].every(Number.isFinite))throw Error('实验曲线格式不正确：'+key);
 if(!Array.isArray(out.activation_rms)||out.activation_rms.length!==out.losses.length||!out.activation_rms.every(v=>v&&Object.values(v).every(Number.isFinite)))throw Error('激活记录格式不正确。');
 if(!Array.isArray(out.evaluation)||!out.evaluation.length||!out.evaluation.every(p=>[p.step,p.train,p.validation].every(Number.isFinite)))throw Error('验证曲线格式不正确。');
 if(!Array.isArray(out.generated)||!out.generated.every(v=>typeof v==='string')||!out.split||typeof out.split.policy!=='string')throw Error('生成结果或数据划分缺失。');
 for(const key of ['validation_before','validation_after','parameters','vocab','train_chars','validation_chars','seconds'])if(!Number.isFinite(out[key]))throw Error('实验指标缺失：'+key);
 for(const key of ['source_sha256','data_sha256','lab_sha256','python','numpy'])if(typeof out.provenance[key]!=='string')throw Error('运行指纹缺失：'+key);
 for(const value of Object.values(r.snapshot.dependencies))if(typeof value!=='string')throw Error('源码快照格式不正确。');}
 $('importExperiment').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>2000000)throw Error('记录超过 2 MB，请使用较小记录。');const record=JSON.parse(await file.text());validateRecord(record);const rows=read('gpt-experiment-history')||[];rows.unshift({...record,id:Date.now()});put('gpt-experiment-history',JSON.stringify(rows.slice(0,5)));show(record);history();$('importNote').textContent='已载入记录供查看，尚未执行源码。可在历史表点击“回放源码快照”。';}catch(error){$('importNote').textContent=error.message;}};
 form.onsubmit=e=>{
  e.preventDefault();
  form.oninput();
  if(sidebar.validate())run();
 };
 history();
}
root.Laboratory={mount,mountSidebar,chart};
})(typeof window==='undefined'?globalThis:window);
