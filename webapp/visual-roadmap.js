(function(root){
  'use strict';
  const labels={todo:'待做',in_progress:'制作中',review:'已实现 · 待你验收',accepted:'已验收'};
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function progress(roadmap){
    const items=roadmap?.items||[];
    return {total:items.length,pending:items.filter(i=>i.status==='todo'||i.status==='in_progress').length,review:items.filter(i=>i.status==='review').length,accepted:items.filter(i=>i.status==='accepted').length};
  }
  function mount(ctx){
    const data=ctx.book.visual_roadmap,shortcut=document.getElementById('visualRoadmapShortcut');
    if(!data?.items?.length)return;
    const p=progress(data),next=data.items.find(i=>i.batch==='first'&&(i.status==='todo'||i.status==='in_progress'))||data.items.find(i=>i.status==='todo'||i.status==='in_progress');
    let panel=document.getElementById('visualRoadmap');
    if(!panel){panel=document.createElement('dialog');panel.className='visual-roadmap roadmap-dialog';panel.id='visualRoadmap';document.body.append(panel);}
    panel.setAttribute('aria-labelledby','roadmapTitle');
    panel.innerHTML=`<div class="roadmap-heading"><h2 id="roadmapTitle">课程完善清单</h2><button type="button" data-roadmap-close aria-label="关闭课程完善清单">关闭 ×</button></div><p class="roadmap-count">${p.pending} 项待补 · ${p.review} 项待验收 · ${p.accepted} 项已验收</p><p class="small">已实现的项目仍保留，验收后再标为已验收；与代码测评成绩分开。</p>${next?`<p class="roadmap-next">下次继续：<strong>${escape(next.title)}</strong></p>`:''}${data.batches.map(batch=>`<section class="roadmap-batch"><h3>${escape(batch.title)}</h3>${data.items.filter(i=>i.batch===batch.id).map(item=>`<details class="roadmap-item ${item.lessons.includes(ctx.current)?'roadmap-current':''}"><summary><span class="roadmap-status ${escape(item.status)}">${escape(labels[item.status]||item.status)}</span><span>${escape(item.title)}${item.lessons.includes(ctx.current)?' · 本节':''}</span></summary><div class="roadmap-body"><ul>${item.criteria.map(c=>`<li>${escape(c)}</li>`).join('')}</ul><div class="controls">${item.lessons.map(id=>`<button type="button" data-roadmap-lesson="${escape(id)}">${escape(ctx.book.lessons[id]?.title||id)} ↗</button>`).join('')}</div>${item.verification?`<p class="small">已检查：${escape(item.verification)}</p>`:''}</div></details>`).join('')}</section>`).join('')}<p class="small">项目文件长期保留待办。更新：${escape(data.updated)}。</p>`;
    panel.querySelector('[data-roadmap-close]').onclick=()=>panel.close();
    panel.querySelectorAll('[data-roadmap-lesson]').forEach(button=>button.onclick=()=>{panel.close();ctx.go(button.dataset.roadmapLesson);});
    if(shortcut){
      shortcut.innerHTML=`<button type="button" aria-controls="visualRoadmap" aria-haspopup="dialog">课程完善清单 <strong>${p.pending} 项待补</strong><small>${p.review} 项待验收 · ${p.accepted} 项已验收</small></button>`;
      shortcut.querySelector('button').onclick=()=>{panel.showModal();panel.scrollTop=0;};
    }
  }
  root.VisualRoadmap={mount,progress};
  if(typeof module!=='undefined')module.exports={progress};
})(typeof window==='undefined'?globalThis:window);
