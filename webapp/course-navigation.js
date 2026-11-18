/* 顶部课程目录。只读取学习进度，不写入练习或成绩。 */
(function(root){'use strict';
 const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const expanded=new Map();
 const trainingLab={id:'training_lab',title:'训练实验室'};
 const progress=(tasks,status,paused=false)=>{const done=tasks.filter(id=>status(id).kind==='success').length;return {done,percent:!paused&&tasks.length?100*done/tasks.length:0};};
 const mainStages=book=>book.stages.filter(s=>!s.optional);
 const stageTasks=stage=>stage.modules.flatMap(module=>module.tasks);
 function outline(book,current,status,explanationState){
  const explanation=id=>explanationState(id)==='accepted'?['explanation-accepted','讲解已验收']:explanationState(id)==='review'?['explanation-review','讲解待验收']:['','讲解待调整'];
  function stageMarkup(s,si){
   const tasks=stageTasks(s),active=tasks.includes(current),key='stage-'+si;
   const modules=s.modules.map((x,mi)=>{
    const mk=key+'-module-'+mi,selected=x.tasks.includes(current),{done,percent}=progress(x.tasks,status,s.paused);
    const pending=x.tasks.filter(id=>explanationState(id)==='todo').length;
    const state=pending?['',`讲解待调整 · ${pending} 项`]:x.tasks.every(id=>explanationState(id)==='accepted')?['explanation-accepted','讲解已验收']:['explanation-review','讲解待验收'];
    return `<details class="outline-module" data-branch="${mk}" ${expanded.get(mk)??selected?'open':''}><summary class="outline-progress" style="--outline-progress:${percent}%" data-progress="${percent}"><span>${escape(x.name)}</span><small>${done}/${x.tasks.length} 完成</small><small class="explanation-tag ${state[0]}">${state[1]}</small></summary><div class="outline-tasks">${x.tasks.map((id,i)=>{
     const result=status(id),[color,label]=explanation(id);
     return `<button type="button" class="outline-task ${id===current?'active':''}" data-outline-task="${escape(id)}" ${id===current?'aria-current="page"':''}><span class="outline-task-title">${i+1}. ${escape(book.lessons[id].title)}</span><span class="outline-task-meta"><small class="${result.kind}">${escape(result.text)}</small><small class="explanation-tag ${color}">${label}</small></span></button>`;
    }).join('')}</div></details>`;
   }).join('');
   const {done,percent}=progress(tasks,status,s.paused);
   return `<details class="outline-stage" data-branch="${key}" ${expanded.get(key)??active?'open':''}><summary class="outline-progress" style="--outline-progress:${percent}%" data-progress="${percent}" ${s.paused?'data-progress-paused="true"':''}><span>${escape(s.title)}</span><small>${s.paused?'暂缓':`${done}/${tasks.length} 完成`}</small></summary>${s.paused?`<p class="small">${escape(s.description)}</p>`:modules}</details>`;
  }
  const core=book.stages.flatMap((stage,index)=>stage.optional?[]:[stageMarkup(stage,index)]);
  const lab=`<div class="outline-stage outline-lab"><button type="button" class="outline-task ${current===trainingLab.id?'active':''}" data-outline-task="${trainingLab.id}" ${current===trainingLab.id?'aria-current="page"':''}><strong>${trainingLab.title}</strong></button></div>`;
  return core.join('')+lab;
 }
 function mount(ctx){
  const dialog=document.getElementById('courseOutline'),button=document.getElementById('outlineToggle'),nav=document.getElementById('nav');
  const {book,current,status,go,explanationState}=ctx;
  const toolsMenu=document.getElementById('courseToolMenu');
  if(toolsMenu)toolsMenu.onclick=e=>{if(e.target.closest('button'))toolsMenu.open=false;};
  const isLab=current===trainingLab.id;
  const stage=book.stages.find(s=>stageTasks(s).includes(current)),module=stage?.modules.find(x=>x.tasks.includes(current));
  const location=document.getElementById('courseLocation');
  location.textContent=isLab?trainingLab.title:`${stage.title} / ${module.name} / ${book.lessons[current].title}`;
  location.title=location.textContent;
  nav.innerHTML=outline(book,current,status,explanationState);
  nav.querySelectorAll('details[data-branch]').forEach(el=>el.ontoggle=()=>expanded.set(el.dataset.branch,el.open));
  nav.querySelectorAll('[data-outline-task]').forEach(el=>el.onclick=()=>{dialog.close();go(el.dataset.outlineTask);});
  button.onclick=()=>{
   if(dialog.open){dialog.close();return;}
   dialog.showModal();button.setAttribute('aria-expanded','true');
   const selected=nav.querySelector('[aria-current="page"]');
   if(selected){let parent=selected.parentElement;while(parent&&parent!==nav){if(parent.tagName==='DETAILS')parent.open=true;parent=parent.parentElement;}selected.scrollIntoView({block:'center'});selected.focus({preventScroll:true});}
  };
  dialog.onclose=()=>button.setAttribute('aria-expanded','false');
  document.getElementById('outlineClose').onclick=()=>dialog.close();
  dialog.onclick=e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}};
  const pager=document.getElementById('lessonPager');
  if(!pager||isLab)return;
  const sequence=stage.optional?stageTasks(stage):mainStages(book).filter(s=>!s.paused).flatMap(stageTasks);
  const index=sequence.indexOf(current),previous=sequence[index-1],next=sequence[index+1];
  let onward='<span class="small">已到本组最后一项</span>';
  if(next)onward=`<button type="button" class="pager-next primary" data-completion="${status(current).kind==='success'?'passed':'pending'}" data-page-task="${next}"><small>下一任务 →</small><span>${escape(book.lessons[next].title)}</span></button>`;
  if(!stage.optional&&index===sequence.length-1)onward=`<button type="button" class="pager-next primary" data-page-task="${trainingLab.id}"><small>开始实践 →</small><span>${trainingLab.title}</span></button>`;
  pager.innerHTML=`${previous?`<button type="button" class="pager-previous" data-page-task="${previous}"><small>← 上一任务</small><span>${escape(book.lessons[previous].title)}</span></button>`:'<span></span>'}${onward}`;
  pager.querySelectorAll('[data-page-task]').forEach(el=>el.onclick=()=>go(el.dataset.pageTask));
 }
 function createReadingHistory({history,capture,navigate,valid}){
  let active,serial=0;const snapshots=new Map();
  const key=()=>Date.now().toString(36)+'-'+(++serial);
  function save(){
   if(!active)return;
   const reading=capture();snapshots.set(active.key,reading);active={...active,reading};
   history.replaceState({...history.state,courseReader:active},'');
  }
  function start(fallback){
   const saved=history.state?.courseReader;
   active=saved&&valid(saved.lesson)?saved:{key:key(),lesson:fallback,depth:0,from:null};
   history.replaceState({...history.state,courseReader:active},'');return active;
  }
  function go(lesson,{review=false}={}){
   if(!valid(lesson)||lesson===active.lesson)return;
   save();const next={key:key(),lesson,depth:active.depth+1,from:review?active.lesson:null,review};
   history.pushState({courseReader:next},'');active=next;navigate(lesson,null);
  }
  function restore(state){
   const next=state?.courseReader;if(!next||!valid(next.lesson))return false;
   snapshots.set(active.key,capture());active=next;
   navigate(next.lesson,snapshots.get(next.key)||next.reading||null);return true;
  }
  return {start,go,restore,save,entry:()=>active,back:()=>{if(active?.depth>0)history.back();}};
 }
 root.CourseNavigation={mount,createReadingHistory,trainingLab};
 if(typeof module!=='undefined')module.exports={outline,createReadingHistory,trainingLab};
})(typeof window==='undefined'?globalThis:window);
