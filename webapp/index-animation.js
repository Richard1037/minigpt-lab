(function(){'use strict';
window.setupIndexAnimation=function(c){
 const host=document.getElementById('animation'),X=c.args[0],ids=c.args[1],k=Number(c.args[2]),C=X[0].length;
 let step=0,playing=false,busy=false,disposed=false;
 if(window.understandingCleanup)window.understandingCleanup();window.understandingCleanup=()=>{playing=false;disposed=true;};
 const cells=(values,columns=C)=>Array.from({length:columns},(_,j)=>`<span class="slice-cell" data-col="${j}">${values?esc(values[j]):'·'}</span>`).join('');
 host.innerHTML=`<div class="index-animation"><h2>实际取行，再截取前 k 列</h2><p>输入变量：k = <strong>${k}</strong>，保留列编号 0…${k-1}；列编号 ${k} 不包含在切片中。</p><div class="slice-ids">${ids.map((id,i)=>`<span data-id-index="${i}">ids[${i}] = ${id}</span>`).join('')}</div><div class="slice-flow"><section><h3>原始 X · (${X.length}, ${C})</h3><div class="slice-columns" style="--columns:${C}">${Array.from({length:C},(_,j)=>`<span>列 ${j}</span>`).join('')}</div>${X.map((row,i)=>`<div class="slice-label">原行 ${i}</div><div class="slice-row source-row" data-source="${i}" style="--columns:${C}">${cells(row)}</div>`).join('')}</section><section><h3>按 ids 取行 · (${ids.length}, ${C})</h3><div class="slice-columns" style="--columns:${C}">${Array.from({length:C},(_,j)=>`<span>列 ${j}</span>`).join('')}</div>${ids.map((id,i)=>`<div class="slice-label">新行 ${i} ← 原行 ${id}</div><div class="slice-row selected-row" data-selected="${i}" style="--columns:${C}">${cells(null)}</div>`).join('')}</section><section><h3>截取前 k 列 · (${ids.length}, ${k})</h3><div class="slice-columns" style="--columns:${k}">${Array.from({length:k},(_,j)=>`<span>列 ${j}</span>`).join('')}</div>${ids.map((id,i)=>`<div class="slice-label">输出行 ${i} ← 原行 ${id}</div><div class="slice-row output-row" data-output="${i}" style="--columns:${k}">${cells(null,k)}</div>`).join('')}</section></div><p class="slice-explanation" aria-live="polite"></p><div class="controls"><button type="button" class="primary" id="slice-next">取下一行 →</button><button type="button" id="slice-play">自动演示</button><button type="button" id="slice-reset">重新演示</button></div></div>`;
 const panel=host.querySelector('.index-animation'),note=panel.querySelector('.slice-explanation'),next=panel.querySelector('#slice-next'),play=panel.querySelector('#slice-play');
 function tell(){next.disabled=busy||step>=ids.length+2;next.textContent=step<ids.length?'取下一行 →':step===ids.length?'标出截取范围 →':step===ids.length+1?'截取并移入输出 →':'✓ 演示完成';play.textContent=playing?'暂停（本步完成后）':'自动演示';}
 async function travel(source,target,values,crop=false){
  const base=panel.getBoundingClientRect(),s=source.getBoundingClientRect(),t=target.getBoundingClientRect();
  const clone=document.createElement('div');clone.className='slice-row slice-moving';clone.style.cssText=`--columns:${C};width:${s.width}px;height:${s.height}px;left:${s.left-base.left}px;top:${s.top-base.top}px`;clone.innerHTML=cells(values);panel.append(clone);
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches,cut=crop?s.width*(C-k)/C:0;
  try{
   if(crop)await clone.animate([{clipPath:'inset(0 0px 0 0)'},{clipPath:`inset(0 ${cut}px 0 0)`}],{duration:reduce?0:550,fill:'forwards'}).finished;
   await clone.animate([{transform:'translate(0,0)'},{transform:`translate(${t.left-s.left}px,${t.top-s.top}px)`}],{duration:reduce?0:1100,fill:'forwards',easing:'ease-in-out'}).finished;
  }finally{clone.remove();}
 }
 async function advance(){if(busy||disposed||step>=ids.length+2)return;busy=true;tell();
  try{
   if(step<ids.length){const i=step,id=ids[i],source=panel.querySelector(`[data-source="${id}"]`),target=panel.querySelector(`[data-selected="${i}"]`);panel.querySelectorAll('.source-row,.slice-ids span').forEach(el=>el.classList.remove('slice-active'));source.classList.add('slice-active');panel.querySelector(`[data-id-index="${i}"]`).classList.add('slice-active');note.textContent=`读取 ids[${i}]=${id}：复制原行 ${id}，放到取行结果的新行 ${i}。${i>0&&ids.slice(0,i).includes(id)?'该编号重复出现，因此再次复制，原始行仍保留。':''}`;await travel(source,target,X[id]);target.innerHTML=cells(X[id]);
   }else if(step===ids.length){panel.querySelectorAll('.selected-row').forEach(row=>row.querySelectorAll('[data-col]').forEach(cell=>{const j=+cell.dataset.col;cell.classList.add(j<k?'slice-kept':'slice-dropped');if(j===k-1)cell.classList.add('slice-boundary');}));note.textContent=`取行已完成。现在标出 [:k]：k=${k}，保留列 0 到 ${k-1}；${k<C?'右侧其余列不进入输出。':'本例保留全部列。'}`;
   }else{note.textContent=`沿标记边界截取：每行保留前 ${k} 列，再移入输出。原始 X 不改变。`;await Promise.all(ids.map(async(id,i)=>{const source=panel.querySelector(`[data-selected="${i}"]`),target=panel.querySelector(`[data-output="${i}"]`);await travel(source,target,X[id],true);target.innerHTML=cells(X[id].slice(0,k),k);}));note.textContent=`✓ 返回形状 (${ids.length},${k})。先按 ids 取行，再取列区间 [0,${k})；原始 X 保持不变。`;}
   step++;
  }finally{busy=false;tell();}
 }
 next.onclick=advance;play.onclick=async()=>{if(playing){playing=false;tell();return;}if(step>=ids.length+2)return;playing=true;tell();while(playing&&!disposed&&step<ids.length+2){await advance();if(step===ids.length+1&&playing)await new Promise(resolve=>setTimeout(resolve,1100));}playing=false;tell();};
 panel.querySelector('#slice-reset').onclick=()=>{if(busy)return;window.setupIndexAnimation(c);};note.textContent='先读取第一个行编号，观察对应行如何复制到中间结果。';tell();
};
})();
