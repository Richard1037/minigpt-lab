const test=require('node:test'),assert=require('node:assert/strict');
const {rmsRelation,mlpFlow}=require('../webapp/module-views.js');
test('单个上游仅直接影响同格，统计量路径扩散到同token整行',()=>{
 const s=rmsRelation('input',{id:'G',r:3,j:0},6,4);
 for(const id of ['G','u','direct'])assert.deepEqual(s[id].direct,[[3,0]]);
 assert.deepEqual(s.stat.indirect,[[3,0],[3,1],[3,2],[3,3]]);
 assert.equal(s.dX.indirect.length,4);assert.ok(!s.dw);
});
test('参数梯度：一个上游只贡献一项；选择dw则汇总全部batch同特征',()=>{
 const one=rmsRelation('parameter',{id:'G',r:3,j:0},6,4);
 assert.deepEqual(one.dw.direct,[[0,0]]);assert.deepEqual(one.product.direct,[[3,0]]);
 const all=rmsRelation('parameter',{id:'dw',r:0,j:2},6,4);
 assert.deepEqual(all.G.direct,[[0,2],[1,2],[2,2],[3,2],[4,2],[5,2]]);
 assert.deepEqual(all.xh.direct,all.G.direct);
});
test('全六位置的线性梯度响应与标示区域一致',()=>{
 const N=6,C=4,r=3,j=0,x=Array.from({length:N},(_,n)=>Array.from({length:C},(_,c)=>.2+n+c)),w=[1,2,.5,1.5];
 const dx=x.map((row,n)=>{const rr=1/Math.sqrt(row.reduce((s,v)=>s+v*v,0)/C+.01),xh=row.map(v=>v*rr),u=row.map((_,c)=>n===r&&c===j?w[c]:0),mean=u.reduce((s,v,c)=>s+v*xh[c],0)/C;return u.map((v,c)=>rr*(v-xh[c]*mean));});
 assert.ok(dx[r].every(v=>Math.abs(v)>0));assert.ok(dx.every((row,n)=>n===r||row.every(v=>v===0)));
});
test('MLP形状链保留非单位batch和token',()=>{
 const x=Array.from({length:2},()=>Array.from({length:3},()=>Array(4).fill(0)));
 assert.deepEqual(mlpFlow({X:x,init:[4,6]}),{input:[2,3,4],mid:[2,3,6],output:[2,3,4]});
});
