const test=require('node:test'),assert=require('node:assert/strict');
const {attentionSlice,selection,headMapping,residualFlow,normData}=require('../webapp/module-views.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);
test('完整四切片：因果权重、加权读取与全部输入有限差分',()=>{
 for(let b=0;b<2;b++)for(let h=0;h<2;h++){
  const d=attentionSlice(b,h);assert.equal(d.A.length,3);d.A.forEach((row,i)=>{close(row.reduce((a,v)=>a+v,0),1);row.forEach((v,j)=>{if(j>i)close(v,0);});});
  for(let i=0;i<3;i++)for(let c=0;c<2;c++)close(d.Y[i][c],d.A[i].reduce((n,v,j)=>n+v*d.V[j][c],0));
  const loss=()=>{let sum=0;for(let i=0;i<3;i++){const s=d.K.map(k=>d.Q[i].reduce((v,q,c)=>v+q*k[c],0)/Math.sqrt(2)),max=Math.max(...s.slice(0,i+1)),e=s.map((v,j)=>j<=i?Math.exp(v-max):0),z=e.reduce((a,v)=>a+v,0);for(let c=0;c<2;c++)sum+=d.G[i][c]*e.reduce((v,w,j)=>v+w/z*d.V[j][c],0);}return sum;};
  for(const n of ['Q','K','V'])for(let i=0;i<3;i++)for(let c=0;c<2;c++){const old=d[n][i][c];d[n][i][c]=old+1e-5;const a=loss();d[n][i][c]=old-1e-5;const v=loss();d[n][i][c]=old;close(d['d'+n][i][c],(a-v)/2e-5);}
 }
});
test('来源轴：dV收query、dQ收key、dK收query，屏蔽单独处理',()=>{
 let r=selection('value','dV',1,0);assert.ok(!r.A(0,1)&&r.A(2,1)&&!r.A(1,0));assert.ok(r.G(2,0)&&!r.G(0,0)&&!r.G(2,1));
 r=selection('query','dQ',2,1);assert.ok(r.dS(2,0)&&r.K(0,1)&&!r.dS(1,0));
 r=selection('key','dK',2,1);assert.ok(!r.dS(0,2)&&!r.Q(0,1)&&r.Q(2,1)&&!r.dS(2,0));
 r=selection('probabilities','S',0,2);assert.ok(r.A(0,2)&&!r.A(0,0));
});
test('拆合头保留每个batch/token/head及特征身份',()=>{
 for(let b=0;b<2;b++)for(let t=0;t<3;t++)for(let h=0;h<2;h++)for(let c=0;c<2;c++)assert.deepEqual(headMapping(b,t,h,c).target,[b,t,h*2+c]);
 assert.deepEqual(residualFlow(2,3,5),{g1:5,dx:10});
});
test('归一化源向量真实去均值和统一缩放，不以重复符号表代替过程',()=>{
 for(let b=0;b<2;b++)for(let t=0;t<3;t++){const d=normData(b,t);close(d.Z.reduce((a,v)=>a+v),0);d.X.forEach((v,c)=>{close(d.Z[c],v-d.mu);close(d.H[c],d.Z[c]*d.R);});}
});
