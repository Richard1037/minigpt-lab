(function(root){
 'use strict';
 function insideStringOrComment(text,end){
  let quote='',triple=false,comment=false;
  for(let i=0;i<end;i++){
   const c=text[i];
   if(comment){if(c==='\n')comment=false;continue;}
   if(quote){if(c==='\\'){i++;continue;}if(triple&&text.slice(i,i+3)===quote.repeat(3)){quote='';triple=false;i+=2;}else if(!triple&&c===quote)quote='';continue;}
   if(c==='#'){comment=true;continue;}
   if(c==='"'||c==="'"){quote=c;triple=text.slice(i,i+3)===c.repeat(3);if(triple)i+=2;}
  }
  return !!quote||comment;
 }
 function mount(textarea){
  const frame=textarea.parentElement,pre=document.createElement('pre'),code=document.createElement('code');
  pre.className='editor-highlight';pre.setAttribute('aria-hidden','true');pre.append(code);frame.prepend(pre);
  const update=()=>{code.innerHTML=root.PythonHighlighter.highlight(textarea.value,{language:'python',ignoreIllegals:true}).value+'\n';pre.scrollTop=textarea.scrollTop;pre.scrollLeft=textarea.scrollLeft;};
  const pairs={'(':')','[':']','{':'}','"':'"',"'":"'"};
  textarea.addEventListener('scroll',()=>{pre.scrollTop=textarea.scrollTop;pre.scrollLeft=textarea.scrollLeft;});
  textarea.addEventListener('keydown',e=>{
   if(e.defaultPrevented||e.ctrlKey||e.metaKey||e.altKey||e.isComposing)return;
   const start=textarea.selectionStart,end=textarea.selectionEnd,value=textarea.value,key=e.key;
   if(key==='Enter'&&!e.shiftKey){
    const lineStart=value.lastIndexOf('\n',start-1)+1,line=value.slice(lineStart,start);
    const indent=(line.match(/^[\t ]*/)||[''])[0];
    let comment=line.length;
    for(let i=0;i<line.length;i++)if(line[i]==='#'&&!insideStringOrComment(value,lineStart+i)){comment=i;break;}
    let last=comment-1;
    while(last>=0&&/\s/.test(line[last]))last--;
    const colon=last>=0&&line[last]===':'&&!insideStringOrComment(value,lineStart+last);
    e.preventDefault();textarea.setRangeText('\n'+indent+(colon?'    ':''),start,end,'end');
    textarea.dispatchEvent(new Event('input',{bubbles:true}));return;
   }
   if(key==='Backspace'&&start===end&&start>0){
    const lineStart=value.lastIndexOf('\n',start-1)+1,prefix=value.slice(lineStart,start);
    if(/^[\t ]+$/.test(prefix)){
     // 与四空格自动缩进、tab-size:4 一致，退回上一个缩进位置。
     const columns=[0];for(const ch of prefix){const col=columns[columns.length-1];columns.push(ch==='\t'?col+4-col%4:col+1);}
     const target=Math.floor((columns[columns.length-1]-1)/4)*4;
     let keep=prefix.length;while(columns[keep]>target)keep--;
     e.preventDefault();textarea.setRangeText('',lineStart+keep,start,'end');textarea.dispatchEvent(new Event('input',{bubbles:true}));return;
    }
    if(pairs[value[start-1]]&&pairs[value[start-1]]===value[start]){e.preventDefault();textarea.setRangeText('',start-1,start+1,'end');textarea.dispatchEvent(new Event('input',{bubbles:true}));return;}
   }
   if(start===end&&[')',']','}',"'",'"'].includes(key)&&value[start]===key){e.preventDefault();textarea.setSelectionRange(start+1,start+1);return;}
   if(!pairs[key]||(start===end&&insideStringOrComment(value,start)))return;
   if(start===end&&(key==='"'||key==="'")&&/[\w]/.test(value[start-1]||''))return;
   e.preventDefault();const selected=value.slice(start,end);textarea.setRangeText(key+selected+pairs[key],start,end,'end');textarea.setSelectionRange(start+1,start+1+selected.length);textarea.dispatchEvent(new Event('input',{bubbles:true}));
  });
  root.editorHighlight=update;update();frame.classList.add('has-highlighting');
 }
 function highlightReadonly(scope){
  if(!root.PythonHighlighter)return;
  scope.querySelectorAll('.reference pre,.lesson-code pre,#localSource,pre[data-language="python"]').forEach(pre=>{
   const source=pre.textContent;pre.classList.add('python-readonly');
   pre.innerHTML='<code>'+root.PythonHighlighter.highlight(source,{language:'python',ignoreIllegals:true}).value+'</code>';
  });
 }
 root.PythonEditor={mount,insideStringOrComment,highlightReadonly};
 if(typeof module!=='undefined')module.exports={insideStringOrComment};
})(typeof window==='undefined'?globalThis:window);
