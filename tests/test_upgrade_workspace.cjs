const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const source=fs.readFileSync('webapp/app.js','utf8');
function workspace(){
 const saved=new Map(),requests=[],nodes={};let picker;
 const node=()=>({textContent:'',hidden:false,disabled:false,value:''});
 for(const id of ['code','run','save','load','restore','editorFrame','saveNote','localSource'])nodes[id]=node();
 const label=node();nodes.run.closest=()=>({querySelector:s=>s==='.section-label'?label:null});nodes.editorFrame.before=p=>{picker=p;};
 const context={book:{lessons:{upgrade:{integration_only:true,cases:[{label:'check'}],integration_files:['a','b'],dependencies:['a','b'],local_code:'driver'},a:{file:'a.py',local_code:'local A'},b:{file:'b.py',local_code:'local B'}}},current:'upgrade',window:{},$:id=>nodes[id],get:k=>saved.has(k)?saved.get(k):null,put:(k,v)=>saved.set(k,v),isLegacyAnswers:()=>false,preserveLegacy:()=>{},preservePriorAssessment:()=>{},refresh:()=>{},paintCases:()=>{},running:new Set(),version:()=>1,esc:String};
 context.m=()=>context.book.lessons[context.current];context.document={createElement:()=>{
  const p={buttons:[],current:node(),querySelectorAll:()=>p.buttons,querySelector:()=>p.current};
  Object.defineProperty(p,'innerHTML',{set:s=>{p.buttons=[...s.matchAll(/data-module-file="([^"]+)"/g)].map(v=>({dataset:{moduleFile:v[1]},setAttribute:()=>{}}));}});return p;
 }};
 context.fetch=async(url,options)=>{requests.push({url,payload:options?JSON.parse(options.body):null});return {ok:true,json:async()=>url==='/api/catalog'?{lessons:{upgrade:{local_code:'driver'},a:{local_code:'disk A refreshed'},b:{local_code:'disk B refreshed'}}}:url==='/api/save'?{code:JSON.parse(options.body).code}:{rows:[]}};};
 vm.createContext(context);
 vm.runInContext(source.slice(source.indexOf('function draft('),source.indexOf('function explainCall('))+source.slice(source.indexOf('function dependencyCode('),source.indexOf('function status('))+source.slice(source.indexOf('function setupIntegrationTask('),source.indexOf('async function save(')),context);
 vm.runInContext('setupIntegrationTask()',context);
 return {context,saved,nodes,requests,get picker(){return picker;}};
}
test('切换原模块保留各自草稿，不写入检查入口草稿',()=>{
 const w=workspace();assert.equal(w.nodes.code.value,'local A');
 w.nodes.code.value='my A';w.nodes.code.oninput();w.picker.buttons[1].onclick();assert.equal(w.nodes.code.value,'local B');
 w.nodes.code.value='my B';w.nodes.code.oninput();w.picker.buttons[0].onclick();assert.equal(w.nodes.code.value,'my A');
 assert.equal(w.saved.get('pilot-b'),'my B');assert.equal(w.saved.has('pilot-upgrade'),false);assert.equal(w.requests.length,0);
});
test('保存绑定点击时的原文件，切换文件不会改变保存目标',async()=>{
 const w=workspace();w.nodes.code.value='edited A';w.nodes.code.oninput();
 const pending=w.nodes.save.onclick();w.picker.buttons[1].onclick();await pending;
 assert.equal(w.requests[0].payload.lesson,'a');assert.equal(w.requests[0].payload.code,'edited A');assert.equal(w.context.book.lessons.b.local_code,'local B');
});
test('测评使用当前各文件草稿，磁盘刷新不会覆盖未保存的编辑',async()=>{
 const w=workspace();w.nodes.code.value='draft A';w.nodes.code.oninput();w.picker.buttons[1].onclick();w.nodes.code.value='draft B';w.nodes.code.oninput();
 await vm.runInContext('run()',w.context);const request=w.requests.find(r=>r.url==='/api/check');
 assert.equal(request.payload.code,'driver');assert.equal(request.payload.dependencies.a,'draft A');assert.equal(request.payload.dependencies.b,'draft B');
 assert.equal(w.saved.get('pilot-a'),'draft A');assert.equal(w.context.book.lessons.a.local_code,'disk A refreshed');
});
test('提交请求启用通过后保存，并保留返回的本地保存结果',async()=>{
 const w=workspace(),original=w.context.fetch;
 w.context.fetch=async(url,options)=>url==='/api/check'?{ok:true,json:async()=>({rows:[{passed:true}],auto_save:{ok:true,updated:[],local_verified:true}})}:original(url,options);
 await vm.runInContext('run()',w.context);
 const record=JSON.parse(w.saved.get('pilot-result-upgrade'));
 assert.equal(record.data.auto_save.ok,true);assert.match(w.nodes.saveNote.textContent,/本地运行验证/);
 assert.equal(w.saved.has('pilot-a'),false);
});
