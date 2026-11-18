const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {test}=require('node:test');
const path=require('node:path');
function editor(value,cursor=value.length,end=cursor){
 const element=()=>({classList:{add(){}},setAttribute(){},append(){}});
 const scope={window:{PythonHighlighter:{highlight:code=>({value:code})}},document:{createElement:element},Event:class{constructor(type){this.type=type;}}};
 vm.createContext(scope);vm.runInContext(fs.readFileSync(path.join(__dirname,'../webapp/python-editor.js'),'utf8'),scope);
 const listeners={},textarea={value,selectionStart:cursor,selectionEnd:end,parentElement:{prepend(){},classList:{add(){}}},scrollTop:0,scrollLeft:0,
  addEventListener(type,fn){listeners[type]=fn;},inputs:0,
  setRangeText(text,start,end){this.value=this.value.slice(0,start)+text+this.value.slice(end);this.selectionStart=this.selectionEnd=start+text.length;},
  setSelectionRange(start,end){this.selectionStart=start;this.selectionEnd=end;},
  dispatchEvent(event){if(event.type==='input')this.inputs++;}
 };
 scope.window.PythonEditor.mount(textarea);
 return {textarea,key(key,options={}){const event={key,defaultPrevented:false,preventDefault(){this.defaultPrevented=true;},...options};listeners.keydown(event);return event;}};
}
test('Enter adds a level; each Backspace removes one level and emits input',()=>{
 const e=editor('    if ready:');e.key('Enter');assert.equal(e.textarea.value,'    if ready:\n        ');
 e.key('Backspace');assert.equal(e.textarea.value,'    if ready:\n    ');
 e.key('Backspace');assert.equal(e.textarea.value,'    if ready:\n');assert.equal(e.textarea.inputs,3);
 assert.equal(e.key('Backspace').defaultPrevented,false);
});
test('irregular indentation returns to the preceding four-column stop',()=>{
 for(const [before,after] of [['      ','    '],['  ',''],['\t\t','\t'],[' \t ',' \t']]){
  const e=editor(before);assert(e.key('Backspace').defaultPrevented);assert.equal(e.textarea.value,after);
 }
});
test('leading indentation is removed without deleting the following code',()=>{
 const e=editor('        return value',8);e.key('Backspace');assert.equal(e.textarea.value,'    return value');assert.equal(e.textarea.selectionStart,4);
});
test('ordinary text, interior spaces, selections and modified keys use native Backspace',()=>{
 for(const [value,start,end,options] of [['text',4,4,{}],['x    ',5,5,{}],['    x',0,4,{}],['    ',4,4,{ctrlKey:true}],['    ',4,4,{isComposing:true}]]){
  const e=editor(value,start,end);assert.equal(e.key('Backspace',options).defaultPrevented,false);assert.equal(e.textarea.inputs,0);
 }
});
test('paired brackets still delete together',()=>{
 const e=editor('()',1);assert(e.key('Backspace').defaultPrevented);assert.equal(e.textarea.value,'');assert.equal(e.textarea.inputs,1);
});
