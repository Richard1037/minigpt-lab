(function(root){
  'use strict';
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const rendered=new WeakMap();
  const gradientNotation=s=>String(s)
    .replace(/\\mathrm\{d([A-Zwbxqkv])\}/g,'\\nabla $1')
    .replace(/\\mathrm\s*(?:\{d\}|d)\s*([A-Zwbxqkv])/g,'\\nabla $1')
    .replace(/\\nabla_\{?([A-Zwbxqkv])\}?\s*L/g,'\\nabla $1')
    .replace(/(?<![A-Za-z])d([A-Zwbxqkv])(?=[^A-Za-z]|$)/g,'\\nabla $1');
  const options={throwOnError:true,strict:'error',trust:false,output:'htmlAndMathml',maxExpand:1000};
  function render(el){
    if(!el||!root.katex||el.closest('pre,code,textarea'))return;
    const tex=el.dataset.tex===undefined?undefined:gradientNotation(el.dataset.tex);
    if(tex===undefined||(rendered.get(el)===tex&&el.querySelector('.katex')))return;
    try{
      root.katex.render(tex,el,{...options,displayMode:!el.classList.contains('math-inline')});
      el.classList.remove('math-error');rendered.set(el,tex);
    }catch(error){el.textContent=tex;el.classList.add('math-error');el.title=error.message;}
  }
  function renderAll(scope){
    if(scope.matches?.('[data-tex]'))render(scope);
    scope.querySelectorAll?.('[data-tex]').forEach(render);
  }
  function set(el,tex,inline=false){el.classList.toggle('math-inline',inline);el.classList.toggle('math-block',!inline);el.dataset.tex=tex;render(el);}
  const block=(tex,inline=false)=>`<${inline?'span':'div'} class="math-${inline?'inline':'block'}" data-tex="${escape(gradientNotation(tex))}">${escape(gradientNotation(tex))}</${inline?'span':'div'}>`;
  function observe(scope){
    renderAll(scope);
    const observer=new MutationObserver(changes=>changes.forEach(change=>{
      if(change.type==='attributes')render(change.target);
      else change.addedNodes.forEach(node=>{if(node.nodeType===1)renderAll(node);});
    }));
    observer.observe(scope,{childList:true,subtree:true,attributes:true,attributeFilter:['data-tex']});
    return ()=>observer.disconnect();
  }
  root.CourseMath={set,block,renderAll,observe,options};
})(typeof window==='undefined'?globalThis:window);
