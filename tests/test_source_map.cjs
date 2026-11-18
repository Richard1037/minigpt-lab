const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {model,relation,linearSelection,geluDetails}=require('../webapp/source-map.js');
const katex=require('../webapp/vendor/katex/katex.min.js');
const book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
const sample=id=>{const t=book.lessons[id];return t.cases[t.sample_index||0];};
const transpose=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const flatten=x=>x.flat();
const total=x=>x.reduce((a,b)=>a+b,0);
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);
test('GELU曲线、切线与两条导数路径一致，覆盖反向/缩小/放大区间',()=>{
 for(let i=-80;i<=80;i++){
  const x=i/20,d=geluDetails(x),h=1e-5;
  close(d.slope,(geluDetails(x+h).y-geluDetails(x-h).y)/(2*h));
  close(d.slope,d.direct+d.indirect);
 }
 assert.ok(geluDetails(-1.5).slope<0);
 assert.equal(geluDetails(0).slope,.5);
 assert.ok(geluDetails(1.5).slope>1);
});

test('Linear 每个目标的整行/整列来源对应真实样例与三种反向',()=>{
 const c=sample('layer_linear'),m=model('layer_linear',c),X=flatten(c.X),G=flatten(c.G),W=c.parameters.W,b=c.parameters.b;
 assert.deepEqual(m.prefix,[2,3]);assert.equal(m.N,6);
 assert.deepEqual(m.positions,[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]);
 const arrays={X2:X,G2:G,W,WT:transpose(W),XT:transpose(X),b:[b]};
 for(const mode of m.modes){
  const out=relation(m,mode).target.dimensions;
  for(let r=0;r<out[0];r++)for(let j=0;j<out[1];j++){
   const p=relation(m,mode,[r,j]);
   for(const source of p.sources)for(const [i,k] of source.selected){assert.ok(i<source.dimensions[0]&&k<source.dimensions[1]);}
   const values=p.sources.map(source=>source.selected.map(([i,k])=>arrays[source.id][i][k]));
   const value=values.length===1?total(values[0]):total(values[0].map((v,k)=>v*values[1][k]))+(values[2]?.[0]||0);
   const expected=mode==='forward'?c.expected.Y.flat()[r][j]:mode==='dX'?c.expected.dX.flat()[r][j]:mode==='dW'?c.expected['W.grad'][r][j]:c.expected['b.grad'][j];
   close(value,expected);
   for(const formula of [p.formula,p.focus])assert.doesNotThrow(()=>katex.renderToString(formula,{throwOnError:true,strict:'error'}));
   if(mode==='dX'){assert.equal(p.sources[1].id,'WT');assert.equal(p.sources[0].selected.length,m.D);}
   if(mode==='dW'){assert.equal(p.sources[0].id,'XT');assert.equal(p.sources[0].selected.length,m.N);}
  }
 }
});
test('前向练习不提前显示反向，无偏置模块不虚构偏置梯度',()=>{
 assert.deepEqual(model('linear',sample('linear')).modes,['forward']);
 const c=structuredClone(sample('layer_linear'));c.init[2]=false;
 const m=model('layer_linear',c);assert.equal(m.bias,false);assert.ok(!m.modes.includes('db'));
 assert.equal(relation(m,'forward').sources.length,2);assert.ok(!relation(m,'forward').formula.includes('+b'));
});
test('Embedding 整体查行和重复编号合流保留两个batch的每个来源',()=>{
 const c=sample('layer_embedding'),m=model('layer_embedding',c);
 for(let n=0;n<m.N;n++){
  const p=relation(m,'forward',[n,1]),v=m.ids[n];
  assert.equal(p.sources[0].selected[0][0],n);
  assert.deepEqual(p.sources[1].selected,Array.from({length:m.C},(_,j)=>[v,j]));
 }
 const p=relation(m,'dE',[1,0]);
 assert.deepEqual(p.sources[0].selected,[[0,0],[2,0],[4,0]]);
 assert.deepEqual(p.routes.map(r=>r.position),[[0,0],[0,2],[1,1]]);
 const G=c.G.flat();
 for(let j=0;j<m.C;j++)close(total(p.sources[1].selected.filter(co=>co[1]===j).map(([n,k])=>G[n][k])),c.expected['weight.grad'][1][j]);
 const missing=structuredClone(c);missing.X=[[1,1],[1,1]];missing.G=missing.G.map(batch=>batch.slice(0,2));
 const zero=relation(model('layer_embedding',missing),'dE',[0,0]);assert.equal(zero.routes.length,0);assert.equal(zero.sources[1].selected.length,0);
});
test('Scale 参数梯度汇总所有位置与特征，输入梯度保留逐格对应',()=>{
 const c=sample('layer_scale'),m=model('layer_scale',c),p=relation(m,'dw');
 assert.equal(p.sources[0].selected.length,m.N*m.C);assert.equal(p.sources[1].selected.length,m.N*m.C);
 assert.deepEqual(p.target.dimensions,[1,1]);
 const X=c.X.flat(),G=c.G.flat();
 close(total(p.sources[0].selected.map(([n,j])=>G[n][j]*X[n][j])),c.expected['w.grad']);
 const input=relation(m,'dX',[m.N-1,m.C-1]);assert.deepEqual(input.sources[0].selected,[[m.N-1,m.C-1]]);
 assert.deepEqual(input.sources[1].selected,[[0,0]]);
 for(const mode of m.modes){const p=relation(m,mode);for(const f of [p.formula,p.focus])assert.doesNotThrow(()=>katex.renderToString(f,{throwOnError:true,strict:'error'}));}
});

test('固定矩阵视图：每个权重的前向影响与整列上游对应，全部位置保留',()=>{
 const c=sample('layer_linear'),m=model('layer_linear',c),X=c.X.flat(),G=c.G.flat();
 const whole=linearSelection(m,{kind:'dW'});
 assert.equal(whole.marks.dW.length,m.C*m.D);
 assert.equal(whole.marks.X2.length,m.N*m.C);
 for(let i=0;i<m.C;i++)for(let j=0;j<m.D;j++){
  const s=linearSelection(m,{matrix:'W',r:i,j});
  assert.deepEqual(s.marks.dW,[[i,j]]);
  assert.deepEqual(s.marks.Y2,Array.from({length:m.N},(_,n)=>[n,j]));
  close(total(s.marks.X2.map(([n,k])=>X[n][k]*G[n][j])),c.expected['W.grad'][i][j]);
 }
});
test('固定矩阵视图：输入、偏置和单个上游对应正确的整行整列梯度',()=>{
 const m=model('layer_linear',sample('layer_linear'));
 const x=linearSelection(m,{matrix:'X2',r:4,j:2});
 assert.equal(x.kind,'dX');assert.equal(x.marks.W.length,m.D);
 assert.deepEqual(x.marks.G2,Array.from({length:m.D},(_,j)=>[4,j]));
 const bias=linearSelection(m,{matrix:'db',r:0,j:3});
 assert.equal(bias.marks.G2.length,m.N);assert.deepEqual(bias.marks.b,[[0,3]]);
 const g=linearSelection(m,{matrix:'G2',r:4,j:3});
 assert.equal(g.marks.dW.length,m.C);assert.equal(g.marks.dX2.length,m.C);
 assert.deepEqual(g.marks.db,[[0,3]]);
 const c=structuredClone(sample('layer_linear'));c.init[2]=false;
 assert.ok(!linearSelection(model('layer_linear',c),{matrix:'G2',r:0,j:0}).marks.db);
});

test('独立Linear复用完整来源：四维位置全部保留，无偏置只显示真实梯度',()=>{
 const c=sample('rebuild_linear'),m=model('rebuild_linear',c);
 assert.deepEqual(m.prefix,[2,3]);assert.equal(m.N,6);
 assert.equal(linearSelection(m,{kind:'dW'}).marks.X2.length,24);
 const c4=book.lessons.rebuild_linear.cases.find(c=>c.X&&Array.isArray(c.X[0][0][0]));
 assert.ok(c4,'公开测评包含四维输入');
 const high=model('rebuild_linear',c4);
 assert.equal(high.prefix.length,3);assert.equal(high.positions.length,high.N);
 assert.deepEqual(high.positions.at(-1),high.prefix.map(d=>d-1));
 const nb=structuredClone(c4);nb.init[2]=false;
 const noBias=model('rebuild_linear',nb);
 assert.ok(!noBias.modes.includes('db'));
 assert.ok(!linearSelection(noBias,{matrix:'G2',r:noBias.N-1,j:0}).marks.db);
 for(const direction of noBias.modes){const p=relation(noBias,direction);for(const f of [p.formula,p.focus])assert.doesNotThrow(()=>katex.renderToString(f,{throwOnError:true}));}
});
