// 课程层级与任务进度。原任务 ID 保持不变，以沿用草稿和测评记录。
lessons.matmul = {
  title: '把一行与一列相乘，再求和', short: '矩阵乘法',
  goal: '计算二维输入 X 与权重 W 的矩阵乘积，不加偏置。',
  input: 'X：形状（N，C），每行一条样本；W：形状（C，D），每列定义一个输出特征。',
  output: '形状（N，D）的 NumPy 数组。第 i 行、第 j 列是 X 第 i 行与 W 第 j 列逐项相乘后的和。',
  steps: ['定义 solve(X, W)，保持输入不变。', '核对 X 的列数与 W 的行数相同；它们是参与求和的特征数。', '使用矩阵乘法返回新数组。此任务不加偏置，下一模块再组合为 Linear。'],
  starter: 'def solve(X, W):\n    # TODO：返回矩阵乘积，注意不是逐元素相乘\n    raise NotImplementedError("请实现矩阵乘法")',
  hint: '使用 X @ W。@ 表示矩阵乘法，* 表示逐元素相乘。',
  why: '矩阵乘法把多个加权和同时计算出来。X 的一行提供一条样本的所有输入特征，W 的一列提供某个输出特征的全部权重。中间的 C 轴被乘加，留下 N 与 D。'
};
references.matmul = 'def solve(X, W):\n    return X @ W';
// Linear 的教学主线显式整理维度，为二维形式的反向传播做准备。
lessons.linear.title='从三维输入实现 Linear 前向';
lessons.linear.goal='保存原始形状，将所有 token 排成二维表，完成线性计算后恢复输入的前置维度。';
lessons.linear.steps=[
  '记录 X.shape，取最后一轴 C；W 的形状为（C，D），b 的形状为（D，）。',
  '将 X 变为 X2，形状（所有位置数，C）：三维输入时，所有位置数为 B×T。只整理行，不合并特征。',
  '用 X2 @ W + b 计算 Y2，形状（所有位置数，D）。',
  '把 Y2 恢复为 X 的前置维度加 D，然后返回。二维输入也要支持。'
];
lessons.linear.hint='先用 X.reshape(-1, X.shape[-1]) 排成二维表。输出用 reshape(*X.shape[:-1], W.shape[1]) 恢复。-1 表示让 NumPy 推算这一轴的长度。';
lessons.linear.starter='def solve(X, W, b):\n    # 1. 保存形状并排成二维表\n    # 2. 矩阵乘法与偏置\n    # 3. 恢复前置维度后返回\n    raise NotImplementedError("请实现 Linear 前向的三个步骤")';
references.linear='def solve(X, W, b):\n    input_shape = X.shape\n    C = input_shape[-1]\n    D = W.shape[1]\n    X2 = X.reshape(-1, C)\n    Y2 = X2 @ W + b\n    return Y2.reshape(*input_shape[:-1], D)';
lessons.linear.why='直接写 X @ W + b 也是正确的 NumPy 前向实现。本课程先显式整理为二维，便于看清每个 token 的去向，以及反向时所有 token 如何共同贡献参数梯度。两种实现计算相同，不要求通过源码形式判断正确性。';
notes.matmul = {
  intro: 'X 是样本数组；W 是权重数组。它们不是特殊关键字，而是函数接收的两个变量名。',
  meaning: '本例 X 有两条样本，每条两个特征；W 有两行三列，表示把两个输入特征组合为三个输出特征。',
  axes: '输出第 1 行、第 3 列：1×2 + 2×1 = 4。矩阵乘法的内侧维度必须相等：（2，2）乘（2，3）得到（2，3）。',
  code: 'X = np.array([[1, 2], [3, 4]], dtype=float)\nW = np.array([[1, 0, 2], [0, 2, 1]], dtype=float)\nresult = solve(X, W)\nprint(result)  # [[1., 4., 4.], [3., 8., 10.]]'
};
notes.linear.meaning='三维输入（B，T，C）表示 B 条序列，每条 T 个 token，每个 token 有 C 个特征。本例（1，2，2）：一条序列、两个 token、每个两个特征。W 为（2，2），b 为（2，）。';
notes.linear.axes='先把（B，T，C）变成（B×T，C），乘 W 后得到（B×T，D），再恢复成（B，T，D）。允许合并前两维，不能丢失恢复所需的原始形状。';
notes.linear.code='X = np.array([[[1, 2], [0, 1]]], dtype=float)\nW = np.array([[1, 0], [0, 2]], dtype=float)\nb = np.array([3, 4], dtype=float)\nresult = solve(X, W, b)\nprint(result)  # [[[4., 8.], [3., 6.]]]';

const modules = [
  {id:'numpy', stage:'阶段一', title:'掌握数组计算', name:'NumPy 起步',
   goal:'看懂数组的轴，并用归约、广播与矩阵乘法处理数据。',
   tasks:['axes','broadcast','matmul'], future:[],
   acceptance:'三个任务全部通过各自测评，即完成本模块的基础验收。组合应用在下一模块的 Linear 前向中检验。'},
  {id:'layers', stage:'阶段二', title:'构建网络层', name:'Linear 层',
   goal:'组合矩阵乘法与偏置，实现一个可以前向、反向并验证梯度的层。',
   tasks:['linear'], future:['反向计算','梯度检查'],
   acceptance:'完整模块需要前向、反向与梯度检查全部通过。本次只开放前向，前向通过不会标记整个模块完成。'}
];
const taskOrder = modules.flatMap(m=>m.tasks);
function activeModule(){return modules.find(m=>m.tasks.includes(current));}
function doneCount(m){return m.tasks.filter(id=>status(id).kind==='success').length;}
function moduleState(m){let n=doneCount(m),total=m.tasks.length+m.future.length;return {n,total,done:n===total};}
function goTask(id){current=id;render();window.scrollTo({top:0,behavior:'smooth'});}

// 任务、模块、阶段使用同一份代码关联状态；编辑后向上同步失效。
refreshStatus = function(){
  $('nav').innerHTML = modules.map(m=>{
    const st=moduleState(m),active=m.tasks.includes(current);
    return `<section class="stage-nav"><div class="eyebrow">${m.stage}</div><h3>${m.title}</h3><button data-module="${m.id}" class="module-button ${active?'active':''}" ${active?'aria-current="page"':''}><strong>${m.name}</strong><small>${st.done?'✓ 模块完成':`${st.n}/${st.total} 个任务完成`}</small></button></section>`;
  }).join('') + '<div class="future-stage">阶段三 · 构建注意力<br><small>尚未开放</small></div><div class="future-stage">阶段四 · 拼装 GPT<br><small>尚未开放</small></div><div class="future-stage">阶段五 · 训练并生成<br><small>尚未开放</small></div>';
  document.querySelectorAll('[data-module]').forEach(b=>b.onclick=()=>{
    let m=modules.find(m=>m.id===b.dataset.module);
    if(!m.tasks.includes(current)) goTask(m.tasks.find(id=>status(id).kind!=='success')||m.tasks[0]);
  });
  const m=activeModule(),state=moduleState(m);
  if($('moduleProgress')){
    $('moduleProgress').textContent=state.done?'✓ 模块完成 · 本阶段基础验收通过':`${state.n}/${state.total} 个任务完成${m.future.length?' · 后续任务尚未开放':''}`;
    $('moduleBar').style.width=(state.n/state.total*100)+'%';
  }
  if($('taskTabs')){
    $('taskTabs').innerHTML=m.tasks.map((id,i)=>{
      const st=status(id);
      return `<button data-task="${id}" class="${id===current?'active':''}" ${id===current?'aria-current="step"':''}><strong>${i+1}. ${lessons[id].short}</strong><small>${st.text.replace('本节','任务')}</small></button>`;
    }).join('')+m.future.map((name,i)=>`<div class="unavailable">${m.tasks.length+i+1}. ${name}<br><small>尚未开放</small></div>`).join('');
    document.querySelectorAll('[data-task]').forEach(b=>b.onclick=()=>goTask(b.dataset.task));
  }
  if(!$('completion'))return;
  let st=status(current),r=readResult(current);
  if($('editorStatus')){
    $('editorStatus').className='editor-status '+st.kind;
    $('editorStatus').textContent=st.text.replace('本节','任务');
  }
  // Only successful, current code gets the end-of-task action.
  $('completion').hidden=st.kind!=='success';
  $('completion').className='completion '+st.kind;
  $('completion').innerHTML=`<strong>${esc(st.text.replace('本节','任务'))}</strong>${r?`<div class="small">最近提交：${esc(r.when)} · 当前浏览器保存</div>`:''}`;
  if(st.kind==='success'){
    let next=taskOrder[taskOrder.indexOf(current)+1];
    if(next){let b=document.createElement('button');b.textContent=(next==='linear'?'进入下一模块：':'下一个任务：')+lessons[next].short;b.onclick=()=>goTask(next);$('completion').appendChild(b);}
    else{const note=document.createElement('p');note.textContent='当前样章已完成。下一任务：Linear 反向计算（尚未开放）；完整模块仍待反向与梯度检查通过。';$('completion').appendChild(note);}
  }
  updateAssessmentHeading();
};

const renderTask=render;
render=function(){
  renderTask();
  const m=activeModule(),index=m.tasks.indexOf(current);
  const hero=document.querySelector('.hero');
  hero.insertAdjacentHTML('beforebegin',`<div class="breadcrumb">${m.stage} / ${m.title} → ${m.name} → 任务 ${index+1}</div><section class="milestone"><div class="section-label">当前模块 / ${m.stage}</div><h2>${m.name}</h2><p>${m.goal}</p><div class="bar"><div id="moduleBar"></div></div><strong id="moduleProgress"></strong><p class="module-summary">验收：${m.acceptance}</p></section><div class="task-tabs" id="taskTabs" aria-label="模块内练习任务"></div>`);
  hero.querySelector('.step-number').textContent=`${index+1} / ${m.tasks.length+m.future.length}`;
  hero.querySelector('.eyebrow').textContent='当前练习任务';
  const route=document.querySelector('.route');
  if(m.id==='numpy')route.innerHTML='<span class="current">数组计算基础</span> → <span>Linear：矩阵乘法 + 偏置</span> → <span>注意力与 GPT</span>';
  if(current==='matmul')$('viz').closest('section').querySelector('h2').textContent='一行 × 一列 → 一个输出元素';
  renderSample();
  if(current==='linear')renderLinearBridge();
  $('run').disabled=runningLessons.has(current);
  refreshStatus();
};

const drawTask=draw;
total=function(){return current==='linear'?3:current==='matmul'?6:current==='axes'?2:4;};
draw=function(){
  if(current==='linear')return drawLinear();
  if(current!=='matmul'){
    drawTask();
    // The fixed expected result is shown above; this is the animation's work in progress.
    document.querySelectorAll('#viz .matrix-label').forEach(el=>{
      if(el.textContent.startsWith('Y ·')||el.textContent.startsWith('输出 ·'))el.textContent='当前计算结果 · '+el.textContent.split('·')[1].trim();
    });
    return;
  }
  const X=[[1,2],[3,4]],W=[[1,0,2],[0,2,1]],Y=[[1,4,4],[3,8,10]];
  const r=Math.floor((step-1)/3),c=(step-1)%3,valid=step>0;
  const partial=Y.map((row,i)=>row.map((v,j)=>i*3+j<step?v:'?'));
  $('viz').innerHTML='<div class="formula">Yᵢⱼ = Σₖ Xᵢₖ Wₖⱼ</div><div class="matrices">'+
    matrix('X ·（2 条样本，2 个特征）',X,valid?[r+',0',r+',1']:[])+
    '<div class="operator">×</div>'+matrix('W ·（2 个输入，3 个输出）',W,valid?['0,'+c,'1,'+c]:[])+
    '<div class="operator">=</div>'+matrix('当前计算结果 ·（2 条样本，3 个输出）',partial,[],valid?[r+','+c]:[])+'</div>';
  $('explain').textContent=valid?`输出第 ${r+1} 行、第 ${c+1} 列：${X[r][0]} × ${W[0][c]} + ${X[r][1]} × ${W[1][c]} = ${Y[r][c]}。绿色标记对应的一行与一列。`:'点击下一步：一行与一列的对应元素相乘，再相加。输出形状为（2，3）。';
  $('counter').textContent=`${step} / 6`;$('prev').disabled=step===0;$('next').disabled=step===6;
};

// Data is rendered from the same cases that the Python checker executes.
function arrayView(label,value){
  const dims=shape(value),name=`<div class="array-name">${esc(label)} <span>· 形状（${dims.join('，')}）</span></div>`;
  if(dims.length===1)return name+`<div class="array-block">${matrix('一维数组',[value])}</div>`;
  if(dims.length===2)return name+`<div class="array-block">${matrix('',value)}</div>`;
  if(dims.length===3)return name+value.map((v,i)=>`<div class="array-block">${matrix('样本 '+(i+1),v)}</div>`).join('');
  return name+`<pre>${esc(JSON.stringify(value))}</pre>`;
}
function argNames(){return current==='axes'?['X']:current==='broadcast'?['X','b']:current==='matmul'?['X','W']:['X','W','b'];}
function sampleIndex(){return current==='linear'?1:0;}
function renderSample(){
  const example=cases[current][sampleIndex()],names=argNames();
  $('sample').innerHTML=`<div class="sample-grid"><div class="sample-side"><h3>样例输入</h3>${example.args.map((v,i)=>arrayView(names[i],v)).join('')}</div><div class="sample-side expected"><h3>预期输出</h3>${arrayView('返回值',example.expected)}<p class="small">这是最终应得到的结果。下面的动画使用同一组输入，逐步算出它。</p></div></div>`;
}
function updateAssessmentHeading(){
  if(!$('results'))return;
  const r=readResult(current),same=r&&r.code===$('code').value&&r.version===JSON.stringify(cases[current]);
  $('results').textContent=runningLessons.has(current)?'正在执行你的代码…':!r?'测试用例 · 展开可查看输入与预期输出':!same?'上次提交的测试结果 · 代码已修改，需重新测评':r.data.error?'本次未完成测评 · 请查看错误后重试':'测试结果 · 通过项默认收起，失败项展开';
}
function paintAssessment(){
  if(!$('tests'))return;
  const r=readResult(current),valid=r&&r.version===JSON.stringify(cases[current]);
  const rows=valid?(r.data.rows||[]):[],names=argNames();
  $('tests').innerHTML=(valid&&r.data.error?`<div class="error-panel"><strong>运行未完成</strong><pre>${esc(r.data.error)}</pre></div>`:'')+cases[current].map((c,i)=>{
    const row=rows[i],failed=row&&!row.passed;
    const input=c.args.map((v,j)=>arrayView(names[j],v)).join('');
    let actual='';
    if(row){
      actual=row.errorText?`<div><h3>执行异常</h3><pre>${esc(row.errorText)}</pre></div>`:`<div><h3>你的实际输出</h3>${arrayView('返回值',row.actual)}<p class="small">返回形状（${row.shape.join('，')}）</p></div>`;
    }
    const diagnosis=row&&!row.errorText?`<p class="small">最大绝对误差：${row.error===null?'形状不符，未比较':row.error} ｜ 返回 NumPy 数组：${row.array?'是':'否'} ｜ 输入未修改：${row.unchanged?'是':'否'}</p>`:'';
    return `<details class="test-case ${failed?'failed':''}" ${failed?'open':''}><summary><span>${i===sampleIndex()?'样例测评':'其他用例 '+(i+1)} · ${esc(c.label)}</span><span class="case-status ${row?(row.passed?'pass':'fail'):''}">${row?(row.passed?'✓ 通过':'! 未通过'):'未运行'}</span></summary><div class="case-body"><div class="case-grid"><div><h3>传入函数的输入</h3>${input}</div><div><h3>预期输出</h3>${arrayView('返回值',c.expected)}</div>${actual}</div>${diagnosis}${failed?'<p>检查顺序：输入与输出形状 → 计算轴或乘法规则 → 是否原地修改了输入。</p>':''}</div></details>`;
  }).join('');
  updateAssessmentHeading();
}

function drawLinear(){
  const c=cases.linear[1],X=c.args[0],W=c.args[1],b=c.args[2],Y=c.expected;
  const X2=X.flat(),Y2=Y.flat();
  const stages=['输入 X ·（B，T，C）=（1，2，2）','整理为 X2 ·（B×T，C）=（2，2）','计算 Y2 ·（B×T，D）=（2，2）','恢复 Y ·（B，T，D）=（1，2，2）'];
  $('viz').innerHTML='<div class="route">'+stages.map((s,i)=>`<span class="${i===step?'current':''}">${s}</span>`).join(' → ')+'</div>'+
    (step===0?arrayView('X',X):step===1?arrayView('X2：同一 token 的特征仍在同一行',X2):step===2?'<div class="matrices">'+matrix('X2',X2)+' × '+matrix('W',W)+' + '+matrix('b',[b])+' = '+matrix('Y2',Y2)+'</div>':arrayView('当前计算结果 Y',Y));
  $('explain').textContent=[
    '开始：有 1 条序列，每条 2 个 token，每个 token 有 2 个输入特征。先记录原始形状（1，2，2）。',
    '整理：第 1 行仍是第 1 个 token [1, 2]，第 2 行仍是第 2 个 token [0, 1]。reshape 不求和、不混合特征；只是改变数组的形状视图（必要时可能复制）。',
    '计算：第一行 [1×1+2×0+3, 1×0+2×2+4] = [4,8]；第二行 [0×1+1×0+3, 0×0+1×2+4] = [3,6]。',
    '恢复：把两行结果放回那一条序列，得到 [[[4,8],[3,6]]]。输出形状（1，2，2），输入的 B、T 保留，最后一轴为 D。'
  ][step];
  $('counter').textContent=`${step} / 3`;$('prev').disabled=step===0;$('next').disabled=step===3;
}

function renderLinearBridge(){
  $('viz').closest('section').querySelector('h2').textContent='三维 → 二维计算 → 恢复三维';
  $('code').closest('section').insertAdjacentHTML('beforebegin',`<section class="card"><div class="section-label">实现路线 / 从前向接到反向</div><h2>同一套维度约定贯穿整个 Linear 层</h2><p>当前先完成纯函数前向，下一任务再把它装入层对象并实现反向。纯函数返回值不是完整可训练层；本次通过只代表前向正确。</p><div class="grid"><div><h3>前向时保存</h3><p>原始输入形状、二维输入 X2。W 和 b 是层持有的参数；反向必须使用这次前向对应的权重，计算完反向后才能更新。</p></div><div><h3>反向时传入</h3><p>G = ∂L/∂Y，上游传来的损失对输出的梯度，形状必须与 Y 相同。G 不是标签，也不是 Y。</p></div></div><ol><li>将 G 整理成 G2，形状（B×T，D）。</li><li>输入梯度：dX2 = G2 @ Wᵀ，形状（B×T，C），然后恢复为原始 X 的形状。</li><li>权重梯度：dW = X2ᵀ @ G2，形状（C，D），汇总所有样本、所有 token 的贡献。</li><li>偏置梯度：db = G2 沿 axis=0 求和，形状（D，）。不要在此额外求平均，上游损失已经决定是否平均。</li><li>层内把 dW、db 累加到参数梯度；训练循环在一次新的梯度累加开始前清零，优化器负责更新参数。</li></ol><p class="small">完整模块还必须通过输入、W、b 的有限差分校验，以及重复调用时的梯度累加与清零检查。反向任务与这些测评尚未开放，不会用“前向通过”代替它们。</p><details><summary>对照：为什么直接三维 @ 也正确</summary><p>NumPy 会把最后两轴当作矩阵维度，前面的轴用于批量计算；共享的二维 W 作用于每个 token。因此 X @ W + b 与本节显式 reshape 的结果相同。先学显式维度整理，是为了下一步能手写并检查梯度。</p></details></section>`);
}
