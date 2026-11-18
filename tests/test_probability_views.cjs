const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {probabilityData,probabilityRelation}=require('../webapp/module-views.js');
const book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);
test('Softmax保留六位置，单一分数对同行所有概率的导数与数值扰动一致',()=>{
 const c=book.lessons.t_softmax.cases[0],d=probabilityData(c);assert.equal(d.N,6);assert.deepEqual(d.positions.at(-1),[1,2]);
 for(let n=0;n<d.N;n++)for(let j=0;j<d.V;j++){
  const r=probabilityRelation(d,n,j),h=1e-5;
  const p=x=>{const e=x.map(Math.exp),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
  const a=d.rows[n].slice(),b=a.slice();a[j]+=h;b[j]-=h;
  p(a).forEach((v,i)=>close(r.coefficients[i],(v-p(b)[i])/(2*h)));
  close(r.coefficients.reduce((a,b)=>a+b,0),0);
  close(r.dx,c.expected.dX[n===0?0:Math.floor(n/3)][n%3][j]);
 }
});
test('交叉熵每行的正确类别、全位置平均和正负零标量上游一致',()=>{
 const c=book.lessons.t_loss.cases[0],d=probabilityData(c);assert.equal(d.N,6);
 for(let n=0;n<d.N;n++)for(const g of [-2,0,1]){
  const r=probabilityRelation(d,n,0,true,g);close(r.loss,c.expected.loss);
  r.dx.forEach((v,j)=>close(v,g*c.expected.dLogits[Math.floor(n/3)][n%3][j]));
  close(r.dx.reduce((a,b)=>a+b,0),0);
 }
});

test('默认整体视图覆盖全部位置，四种整体流程公式严格有效',()=>{
 const {probabilityOverview}=require('../webapp/module-views.js'),katex=require('../webapp/vendor/katex/katex.min.js');
 const formulas=[];global.CourseMath={block:f=>{formulas.push(f);return '<math></math>';}};
 for(const loss of [false,true])for(const backward of [false,true]){
  const d=probabilityData(book.lessons[loss?'t_loss':'t_softmax'].cases[0]);
  const html=probabilityOverview(d,loss,backward);
  assert.equal((html.match(/data-pv-position=/g)||[]).length,6);
  assert.match(html,/data-pv-position="1,2"/);
 }
 for(const f of formulas){assert.ok(!/[\u0000-\u0008\u000b\u000c]/.test(f));assert.doesNotThrow(()=>katex.renderToString(f,{throwOnError:true,strict:'error'}));}
 delete global.CourseMath;
});

test('标签选择直接展示源矩阵到全部选取结果，不再以符号表充当示意图',()=>{
 const {probabilityOverview}=require('../webapp/module-views.js');global.CourseMath={block:()=>'<math></math>'};
 const d=probabilityData(book.lessons.t_loss.cases[0]),html=probabilityOverview(d,true,false);
 assert.ok(!html.includes('<table'));
 const picked=[...html.matchAll(/class="gather-source gather-selected" data-gather-row="(\d+)">([^<]+)/g)];
 assert.equal(picked.length,6);
 picked.forEach((m,n)=>{assert.equal(+m[1],n);assert.equal(m[2],d.logp[n][d.target[n]].toFixed(2));});
 assert.equal((html.match(/class="gather-source /g)||[]).length,24);
 delete global.CourseMath;
});
