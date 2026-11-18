const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {outline,createReadingHistory,trainingLab}=require('../webapp/course-navigation.js');
const book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
const status=id=>({kind:id==='infrastructure'?'success':'',text:id==='infrastructure'?'✓ 已通过 · 4/4':'○ 未测评'});
const explanation=id=>['infrastructure','initialization'].includes(id)?'accepted':['linear','layer_scale','layer_linear'].includes(id)?'review':'todo';

function readerHarness(){
 const states=[{existing:true}];let index=0,reader,view={scrollTop:0,folds:[]},lesson='block';
 const history={get state(){return states[index];},replaceState(s){states[index]=structuredClone(s);},pushState(s){states.splice(index+1);states.push(structuredClone(s));index++;},back(){if(index){index--;reader.restore(states[index]);}}};
 const options={history,capture:()=>structuredClone(view),navigate:(id,saved)=>{lesson=id;view=saved||{scrollTop:0,folds:[]};},valid:id=>['block','branch','norm'].includes(id)};
 reader=createReadingHistory(options);reader.start('block');
 return {reader,history,options,setView:v=>view=v,get view(){return view;},get lesson(){return lesson;},get count(){return states.length;},forward(){if(index<states.length-1){index++;reader.restore(states[index]);}}};
}

test('前置任务连续跳转后可逐层返回，恢复原滚动位置和折叠区',()=>{
 const h=readerHarness();h.setView({scrollTop:1450,folds:[false,true,true]});h.reader.go('branch',{review:true});
 assert.equal(h.reader.entry().from,'block');assert.equal(h.lesson,'branch');assert.equal(h.view.scrollTop,0);
 h.setView({scrollTop:630,folds:[true,false]});h.reader.go('norm',{review:true});
 h.reader.back();assert.equal(h.lesson,'branch');assert.deepEqual(h.view,{scrollTop:630,folds:[true,false]});
 h.reader.back();assert.equal(h.lesson,'block');assert.deepEqual(h.view,{scrollTop:1450,folds:[false,true,true]});
});

test('浏览器前进保留离开时的阅读状态，同页或无效跳转不产生假历史',()=>{
 const h=readerHarness();h.reader.go('branch',{review:true});h.setView({scrollTop:888,folds:[true]});h.reader.back();
 h.forward();assert.equal(h.lesson,'branch');assert.deepEqual(h.view,{scrollTop:888,folds:[true]});
 const count=h.count;h.reader.go('branch',{review:true});h.reader.go('missing');assert.equal(h.count,count);
 assert.equal(h.reader.restore({}),false);assert.equal(h.lesson,'branch');
});

test('刷新后读取当前历史项，保存阅读状态不需要练习或成绩存储',()=>{
 const h=readerHarness();assert.equal(h.history.state.existing,true);
 h.reader.go('branch',{review:true});h.setView({scrollTop:321,folds:[false,true]});h.reader.save();
 const fresh=createReadingHistory(h.options),entry=fresh.start('block');
 assert.equal(entry.lesson,'branch');assert.equal(entry.from,'block');assert.equal(entry.depth,1);
 assert.deepEqual(entry.reading,{scrollTop:321,folds:[false,true]});
});

test('主线任务仅出现一次；当前任务与讲解状态独立显示',()=>{
 const markup=outline(book,'layer_linear',status,explanation);
 const ids=[...markup.matchAll(/data-outline-task="([^"]+)"/g)].map(m=>m[1]).filter(id=>id!==trainingLab.id);
 const mainIds=book.stages.filter(s=>!s.optional&&!s.paused).flatMap(s=>s.modules.flatMap(m=>m.tasks));
 assert.deepEqual(new Set(ids),new Set(mainIds));
 assert.equal(ids.length,mainIds.length);
 assert.equal((markup.match(/aria-current="page"/g)||[]).length,1);
 assert.match(markup,/data-outline-task="layer_linear" aria-current="page"/);
 assert.match(markup,/讲解已验收/);assert.match(markup,/讲解待验收/);assert.match(markup,/讲解待调整/);
});

test('目录只展示主线和训练实验室，选修直达仍可使用',()=>{
 const markup=outline(book,'rebuild_gpt',status,explanation);
 assert.doesNotMatch(markup,/回收站|recycle-bin|outlineCommonGuide/);
 for(const stage of book.stages)for(const module of stage.modules)for(const id of module.tasks){
  const entry=new RegExp(`data-outline-task="${id}"`);
  if(stage.optional)assert.doesNotMatch(markup,entry);
  else if(!stage.paused)assert.match(markup,entry);
 }
});

test('目录打开定位当前任务、选择后关闭、页末切换与边界',()=>{
 const nodes={};
 function element(){return {innerHTML:'',textContent:'',attrs:{},setAttribute(k,v){this.attrs[k]=v;},querySelectorAll(selector){
  const attr=selector==='[data-outline-task]'?'outline-task':selector==='[data-page-task]'?'page-task':null;
  if(!attr)return [];
  const property=attr==='outline-task'?'outlineTask':'pageTask';
  return [...this.innerHTML.matchAll(new RegExp(`data-${attr}="([^"]+)"`,'g'))].map(m=>({dataset:{[property]:m[1]}}));
 },querySelector(){return selected;}};}
 let scrolled=false,focused=false;
 const parent={tagName:'DETAILS',open:false,parentElement:null};
 const selected={parentElement:parent,scrollIntoView(){scrolled=true;},focus(){focused=true;}};
 for(const id of ['courseOutline','outlineToggle','nav','courseLocation','outlineClose','lessonPager'])nodes[id]=element();
 nodes.courseOutline.showModal=function(){this.open=true;};
 nodes.courseOutline.close=function(){this.open=false;this.onclose();};
 global.document={getElementById:id=>nodes[id]};
 let destination=null;
 const ctx={book,current:'layer_linear',status,explanationState:explanation,go:id=>destination=id};
 global.CourseNavigation.mount(ctx);
 nodes.outlineToggle.onclick();
 assert.equal(nodes.courseOutline.open,true);assert.equal(nodes.outlineToggle.attrs['aria-expanded'],'true');
 assert.ok(scrolled&&focused&&parent.open);
 assert.equal(nodes.courseOutline.open,true);
 // 重新生成的按钮对象需保留监听器，代替 querySelectorAll 的无状态模拟。
 const old=nodes.nav.querySelectorAll;
 let buttons;nodes.nav.querySelectorAll=function(selector){const result=old.call(this,selector);if(selector==='[data-outline-task]')buttons=result;return result;};
 global.CourseNavigation.mount(ctx);
 buttons.find(b=>b.dataset.outlineTask==='linear').onclick();
 assert.equal(destination,'linear');assert.equal(nodes.courseOutline.open,false);assert.equal(nodes.outlineToggle.attrs['aria-expanded'],'false');
 assert.match(nodes.lessonPager.innerHTML,/data-page-task="linear"/);
 assert.match(nodes.lessonPager.innerHTML,/class="pager-previous"/);
 assert.match(nodes.lessonPager.innerHTML,/class="pager-next primary"/);
 global.CourseNavigation.mount({...ctx,status:()=>({kind:'success',text:'已通过'})});
 assert.match(nodes.lessonPager.innerHTML,/data-completion="passed"/);
 global.CourseNavigation.mount({...ctx,status:()=>({kind:'warn',text:'已修改，待重测'})});
 assert.match(nodes.lessonPager.innerHTML,/data-completion="pending"/);
 const main=book.stages.filter(s=>!s.optional&&!s.paused).flatMap(s=>s.modules.flatMap(x=>x.tasks));
 global.CourseNavigation.mount({...ctx,current:main[0]});
 assert.equal((nodes.lessonPager.innerHTML.match(/data-page-task=/g)||[]).length,1);
 assert.match(nodes.lessonPager.innerHTML,new RegExp(`data-page-task="${main[1]}"`));
 const pagerQuery=nodes.lessonPager.querySelectorAll;
 let pagerButtons;
 nodes.lessonPager.querySelectorAll=function(selector){return pagerButtons=pagerQuery.call(this,selector);};
 global.CourseNavigation.mount({...ctx,current:main.at(-1)});
 assert.match(nodes.lessonPager.innerHTML,new RegExp(`data-page-task="${trainingLab.id}"><small>开始实践`));
 assert.doesNotMatch(nodes.lessonPager.innerHTML,/data-completion|下一任务/);
 pagerButtons.find(b=>b.dataset.pageTask===trainingLab.id).onclick();
 assert.equal(destination,trainingLab.id);
 global.CourseNavigation.mount({...ctx,current:'rebuild_gpt'});
 assert.doesNotMatch(nodes.courseLocation.textContent,/回收站/);
 assert.doesNotMatch(nodes.lessonPager.innerHTML,/data-page-task=/);
 global.CourseNavigation.mount({...ctx,current:trainingLab.id});
 assert.equal(nodes.courseLocation.textContent,trainingLab.title);
 assert.match(nodes.nav.innerHTML,new RegExp(`data-outline-task="${trainingLab.id}" aria-current="page"`));
 buttons.find(b=>b.dataset.outlineTask===trainingLab.id).onclick();
 assert.equal(destination,trainingLab.id);
 delete global.document;
 delete global.LearningTools;
});

test('目录对课程标题做 HTML 转义',()=>{
 const b={stages:[{title:'<stage>',modules:[{name:'<module>',tasks:['a']}]}],lessons:{a:{title:'<script>" &'}}};
 const markup=outline(b,'a',status,()=> 'todo');
 assert.ok(!markup.includes('<script>'));assert.match(markup,/&lt;script&gt;&quot; &amp;/);
});

test('整块章节背景按有效成绩填充，待重测和暂缓不算完成',()=>{
 const b={stages:[{title:'chapter',modules:[{name:'module',tasks:['a','b']}]}],lessons:{a:{title:'a'},b:{title:'b'}}};
 for(const [kinds,percent] of [[['',''],0],[['success','warn'],50],[['success','success'],100]]){
  const markup=outline(b,'a',id=>({kind:kinds[id==='a'?0:1],text:''}),()=> 'accepted');
  assert.match(markup,new RegExp(`--outline-progress:${percent}%`));
 }
 b.stages[0].paused=true;
 const paused=outline(b,'a',()=>({kind:'success',text:''}),()=> 'accepted');
 assert.match(paused,/data-progress="0" data-progress-paused="true"/);
});

 test('普通任务切换保留历史但不显示回看来源',()=>{
 const h=readerHarness();h.reader.go('branch');assert.equal(h.reader.entry().from,null);assert.equal(h.reader.entry().review,false);h.reader.go('norm',{review:true});assert.equal(h.reader.entry().from,'branch');h.reader.back();assert.equal(h.reader.entry().review,false);
});
