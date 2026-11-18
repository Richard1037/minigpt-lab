/* 原理动画 · 第 E 批：反向传播里三处最卡的推导，做成"能拖、能点、当场算出数字"。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和前几批一样：能算的都真的算 —— 矩阵乘/转置、softmax、softmax 反向、
   CrossEntropy 的前后向全部在 JS 里按 float64 逐格跑，readout 里的每个数字都可以
   直接抄进 numpy 对（R.__e_math 导出了这些纯函数，node/python 对拍用的就是同一份）。

   本批三个组件：matmulbwd / softmaxbwd / cebwd。
   读者不会矩阵微分，所以三张图共用同一个核心动作：
   **点结果里的一格 → 看它是哪几个数乘出来的 → 逐项算式写在下面**，公式不用背。

   共同约定：
     · 纯计算全部集中在文件上半部的 E 对象里（组件只负责画和取数），并以
       R.__e_math 导出，做法照 figs_b.js 的 R.__b_math。
     · 不引任何外部库：canvas 2D 手画格子和柱状。
     · mount 时不起定时器、也不起 rAF —— 这批没有"播放"，所有交互都是同步重画；
       事件只挂在自己创建的 canvas / 按钮 / 滑杆上，不碰 window/document。
     · 错法一律做成可开关的对照：打开时把错的结果画红，并给出它与真值的最大绝对偏差。
       正文只讲唯一正确的那一种写法，错法不并列罗列。
     · 组件内文案不用 `$…$`（MathJax 不排版动态插入的内容），公式写成 unicode 明文。
     · 自带样式全部以 `.lane-e ` 前缀注入，不碰 manual.css。

   一处形状口径要先说清楚：讲义 s03a §2.5/§6.2 和 solutions/minigpt/nn.py 里 W 一律
   存成 (out,in)，于是 y = x @ W.T、dW = g.T @ x、db = g.sum(axis=0)、dx = g @ W。
   matmulbwd 用 in=3、out=2，所以 W 是 (2,3)、dW 也是 (2,3)（和 W 同形）、dx 是 (4,3)。
   图里按这套真实代码画，不按"W 存成 (in,out)"的另一种排法 —— 两种排法混着讲正是
   §6.1 点名的那个坑（形状全对但梯度永远错）。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  if (!F || !R) return;
  const { el, fmt } = F;

  /* ------------------------------------------------------------------ 公共件 */

  let styled = false;
  function injectStyle() {
    if (styled || typeof document === 'undefined' || !document.head) return;
    styled = true;
    const s = el('style'); s.id = 'lane-e-style';
    s.textContent = [
      '.lane-e canvas{display:block;max-width:100%;margin:6px 0 0;touch-action:none;',
      '  border:1px solid var(--line);border-radius:8px;background:#101722}',
      '.lane-e canvas.pick{cursor:pointer}',
      '.lane-e .lane-e-btns{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 2px;align-items:center}',
      '.lane-e button.bad.on{border-color:var(--bad);color:var(--bad);background:#2a1a1c}',
      '.lane-e .lane-e-terms{font:11.5px/1.7 var(--mono);color:#cfe0ff;background:var(--code);',
      '  border:1px solid var(--line);border-radius:8px;padding:8px 11px;margin-top:8px;',
      '  white-space:pre-wrap;overflow-x:auto}',
      '.lane-e .lane-e-note{font:11.5px/1.6 var(--mono);color:var(--dim);margin-top:0}',
    ].join('\n');
    document.head.appendChild(s);
  }

  const ACC = '#4aa3ff', OKC = '#3ecf8e', BADC = '#ff6b6b', WARN = '#f5b445';
  const DIM = '#8d99ac', FG = '#e6ebf3', BG = '#101722';
  const CX = '#7fb3e8', CWC = '#b9a7e0', CG = '#8fd6b0', CY = '#f0c07a', COUT = '#9fd8b8';
  const MONO = 'ui-monospace,Consolas';

  /* 格子里写得下的紧凑数字：去掉多余尾零，但不超过 d 位。 */
  function trim(v, d) {
    if (!isFinite(v)) return String(v);
    let s = v.toFixed(d);
    if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
    return s === '-0' ? '0' : s;
  }
  /* 对拍用的满位数：够长到能一位位核对，又不会把 readout 撑爆。 */
  const full = v => isFinite(v) ? (v === 0 ? '0' : v.toPrecision(12)) : String(v);
  const sci = (v, n) => v === 0 ? '0' : v.toExponential(n == null ? 2 : n);
  /* 逐项算式里的求和串：负项用 − 接上，才读得出"这一项是往下拽的"。 */
  function sumStr(parts, d) {
    return parts.map((v, i) => (i === 0
      ? (v < 0 ? '−' : '') + trim(Math.abs(v), d)
      : (v < 0 ? ' − ' : ' + ') + trim(Math.abs(v), d))).join('');
  }

  function mix(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const c = k => Math.round(((pa >> k) & 255) + (((pb >> k) & 255) - ((pa >> k) & 255)) * t);
    return `rgb(${c(16)},${c(8)},${c(0)})`;
  }
  const dimmed = col => mix(col, BG, 0.72);
  const dimSoft = col => mix(col, BG, 0.42);      // 只"退后一步"，不暗到看不清数

  function mkCanvas(host, w, h) {
    const cv = el('canvas');
    cv.width = w * 2; cv.height = h * 2;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    host.appendChild(cv);
    return { cv, g: cv.getContext('2d') };
  }
  function reset(g, cv) {
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, cv.width, cv.height);
    g.scale(2, 2);
  }

  /* 画一块矩阵，返回每格的矩形（点击命中用）。
     o.color/o.ring/o.text 都收 (i,j)；选中态的暗化由调用方算好再传进来。 */
  function paintMat(g, M, o) {
    const cw = o.cw, ch = o.ch, x0 = o.x, y0 = o.y;
    const rows = M.length, cols = M[0].length;
    const rects = [];
    g.save();
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    if (o.title) { g.font = `700 11px ${MONO}`; g.fillStyle = o.titleColor || '#cfe0ff'; g.fillText(o.title, x0, y0 - 22); }
    if (o.sub) { g.font = `10px ${MONO}`; g.fillStyle = o.subColor || DIM; g.fillText(o.sub, x0, y0 - 10); }
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const x = x0 + j * cw, y = y0 + i * ch;
      g.fillStyle = o.color ? o.color(i, j) : CX;
      g.fillRect(x + 0.5, y + 0.5, cw - 1.5, ch - 1.5);
      const ring = o.ring ? o.ring(i, j) : null;
      if (ring) { g.strokeStyle = ring; g.lineWidth = 2; g.strokeRect(x + 1.5, y + 1.5, cw - 3.5, ch - 3.5); }
      g.font = `${o.fs || 10.5}px ${MONO}`;
      g.fillStyle = o.textColor ? o.textColor(i, j) : '#0b0e13';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(o.text ? o.text(i, j) : trim(M[i][j], 3), x + cw / 2, y + ch / 2 + 0.5);
      rects.push({ x, y, w: cw, h: ch, i, j });
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    }
    if (o.idx) {                                           // 列号写在块顶、行号写在块左
      g.font = `9px ${MONO}`; g.fillStyle = DIM; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
      for (let j = 0; j < cols; j++) g.fillText(o.colIdx ? o.colIdx(j) : String(j), x0 + j * cw + cw / 2, y0 - 1);
      g.textAlign = 'right'; g.textBaseline = 'middle';
      for (let i = 0; i < rows; i++) g.fillText(o.rowIdx ? o.rowIdx(i) : String(i), x0 - 5, y0 + i * ch + ch / 2);
    }
    g.restore();
    return rects;
  }

  /* 一排带符号的柱子：零线在中间，正数向上、负数向下。scale = 每单位数值的像素。
     o.below：标签一律排在柱子下方三行（索引 / 数值 / 附加），行间不会互相压字。
     o.hollow：只画轮廓 —— 用来把"真值"垫在"错值"底下做对照。 */
  function paintBars(g, vals, o) {
    const { bw, gap, x0, y0, half, scale } = o;
    const rects = [];
    g.save();
    g.strokeStyle = '#4a5568'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0 - 8, y0); g.lineTo(x0 + vals.length * (bw + gap) - gap + 8, y0); g.stroke();
    g.textAlign = 'center';
    for (let k = 0; k < vals.length; k++) {
      const x = x0 + k * (bw + gap);
      const raw = vals[k] * scale;
      const h = Math.max(-half, Math.min(half, raw));
      const col = o.color ? o.color(k, vals[k]) : (vals[k] >= 0 ? OKC : BADC);
      if (o.hollow) {
        const hh = Math.abs(h) < 1.2 ? (h < 0 ? -1.2 : 1.2) : h;
        g.strokeStyle = col; g.lineWidth = 1.4;
        g.strokeRect(x + 0.5, hh >= 0 ? y0 - hh : y0, bw - 1, Math.abs(hh));
      } else {
        g.fillStyle = col;
        g.fillRect(x, h >= 0 ? y0 - h : y0, bw, Math.max(Math.abs(h), 1.2));
      }
      g.font = `10px ${MONO}`;
      if (o.below) {
        g.textBaseline = 'alphabetic';
        g.fillStyle = DIM; g.fillText(o.idxLabel ? o.idxLabel(k) : String(k), x + bw / 2, y0 + half + 12);
        g.fillStyle = col === OKC || col === BADC ? col : FG;
        g.fillText(o.label ? o.label(k, vals[k]) : trim(vals[k], 3), x + bw / 2, y0 + half + 24);
        if (o.extra) { g.fillStyle = DIM; g.font = `9px ${MONO}`; g.fillText(o.extra(k), x + bw / 2, y0 + half + 35); }
      } else {
        g.fillStyle = FG;
        /* 负柱的标签要落在柱子下沿之外：柱底是 y0−h（h 为负），写成 y0+h 会压进柱子里 */
        g.textBaseline = h >= 0 ? 'bottom' : 'top';
        g.fillText(o.label ? o.label(k, vals[k]) : trim(vals[k], 3), x + bw / 2, h >= 0 ? y0 - h - 3 : y0 - h + 3);
        g.fillStyle = DIM; g.textBaseline = 'alphabetic';
        g.fillText(o.idxLabel ? o.idxLabel(k) : String(k), x + bw / 2, y0 + half + 12);
      }
      rects.push({ x, y: y0 - half, w: bw, h: half * 2, k });
    }
    g.restore();
    return rects;
  }

  /* ====================================================== 纯数学（R.__e_math）
     下面这一整块不碰 DOM：node 里把它抠出来和 numpy 对拍，用的就是同一份代码。 */

  const E = {};

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t = (t ^ (t + Math.imul(t ^ (t >>> 7), t | 61))) >>> 0;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  /* 格子里显示什么就用什么：先按位数取整再参与计算，读者抄进 numpy 才一位不差。 */
  const round = (v, d) => { const p = Math.pow(10, d); return Math.round(v * p) / p; };

  function transpose(A) {
    const out = [];
    for (let j = 0; j < A[0].length; j++) {
      const row = [];
      for (let i = 0; i < A.length; i++) row.push(A[i][j]);
      out.push(row);
    }
    return out;
  }
  /* A @ B；内维不等时返回 null —— 那就是 numpy 当场抛 ValueError 的那种错。 */
  function matmul(A, B) {
    const n = A.length, k = A[0].length, m = B[0].length;
    if (k !== B.length) return null;
    const C = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(m);
      for (let j = 0; j < m; j++) { let s = 0; for (let q = 0; q < k; q++) s += A[i][q] * B[q][j]; row[j] = s; }
      C.push(row);
    }
    return C;
  }
  /* Linear 的正向 + 反向三件产出，逐行对应 solutions/minigpt/nn.py 的 forward/backward。
     W 是 (out,in)、x 是摊平后的 (N,in)、g 是 (N,out)。 */
  function linearFwdBwd(x, W, b, g) {
    const y0 = matmul(x, transpose(W));
    const y = b ? y0.map(row => row.map((v, j) => v + b[j])) : y0;
    const dW = matmul(transpose(g), x);                    // g2.T @ flat_x → (out,in)
    const db = [];
    for (let o = 0; o < g[0].length; o++) { let s = 0; for (let n = 0; n < g.length; n++) s += g[n][o]; db.push(s); }
    const dx = matmul(g, W);                               // g2 @ W → (N,in)
    return { y, dW, db, dx };
  }
  /* 不摊平的写法：numpy 的 3D @ 按前两轴 batch，(B,T,out)ᵀ @ (B,T,in) → (B,out,in)。
     形状合法、不报错，但每个 b 各算一份 dW；真 dW = Σ_b out[b]。 */
  function batchedDw(g3, x3) {
    const B = g3.length, T = g3[0].length, O = g3[0][0].length, C = x3[0][0].length;
    const out = [];
    for (let b = 0; b < B; b++) {
      const M = [];
      for (let o = 0; o < O; o++) {
        const row = new Array(C).fill(0);
        for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) row[c] += g3[b][t][o] * x3[b][t][c];
        M.push(row);
      }
      out.push(M);
    }
    return out;
  }
  function softmax(z) {
    let m = -Infinity; for (const v of z) if (v > m) m = v;
    const e = z.map(v => Math.exp(v - m));
    let Z = 0; for (const v of e) Z += v;
    return e.map(v => v / Z);
  }
  /* log_probs = z − m − log Z —— nn.py CrossEntropy.forward 第 3 步，全程不对概率取 log。 */
  function logSoftmax(z) {
    let m = -Infinity; for (const v of z) if (v > m) m = v;
    const e = z.map(v => Math.exp(v - m));
    let Z = 0; for (const v of e) Z += v;
    const lz = Math.log(Z);
    return z.map(v => v - m - lz);
  }
  /* 不减最大值的版本：大尺度下 Σexp(z) 先变 inf。softshift 那张图讲的是正向，
     这里只用它演示"分开写 Softmax 层 + NLL 层"时反向会跟着一起烂。 */
  function logSoftmaxNaive(z) {
    let S = 0; for (const v of z) S += Math.exp(v);
    const l = Math.log(S);
    return z.map(v => v - l);
  }
  /* ds = p ⊙ (dp − Σ p·dp)，求和沿最后一维 —— solutions/minigpt/attention.py TODO-14。 */
  function softmaxBwd(p, dp) {
    let s = 0; for (let i = 0; i < p.length; i++) s += dp[i] * p[i];
    return { s, ds: p.map((pi, i) => pi * (dp[i] - s)) };
  }
  /* 有限差分：数组输出要先标量化，gradcheck.py 的做法是配一个余向量 cot；
     这里 dp 就是那个 cot，于是 f(z) = Σ softmax(z)·dp，ds_k = ∂f/∂z_k。 */
  function fdSoftmaxBwd(z, dp, eps) {
    const f = zz => { const p = softmax(zz); let s = 0; for (let i = 0; i < p.length; i++) s += p[i] * dp[i]; return s; };
    const num = [];
    for (let k = 0; k < z.length; k++) {
      const zp = z.slice(), zm = z.slice();
      zp[k] += eps; zm[k] -= eps;
      num.push((f(zp) - f(zm)) / (2 * eps));
    }
    return num;
  }
  /* gradcheck.py compare() 的判定式：rel = |a−b| / max(1e-8, |a|, |b|)。 */
  function relErr(a, b) { return Math.abs(a - b) / Math.max(1e-8, Math.abs(a), Math.abs(b)); }
  /* CrossEntropy 的前向 + 反向（融合写法），对应 nn.py 的 forward/backward，grad_out=1.0。 */
  function ceFwdBwd(logits, targets) {
    const N = logits.length, V = logits[0].length;
    const p = [], lp = [], nll = [];
    let acc = 0;
    for (let n = 0; n < N; n++) {
      lp.push(logSoftmax(logits[n]));
      p.push(softmax(logits[n]));
      const v = -lp[n][targets[n]];
      nll.push(v); acc += v;
    }
    const inv = 1.0 / N;                                   // nn.py 写的是 * (grad_out/N)，不是 / N
    const dlogits = p.map((row, n) => row.map((pv, v) => (pv - (v === targets[n] ? 1 : 0)) * inv));
    return { p, lp, nll, loss: acc / N, dlogits, N, V };     // np.mean(nll)
  }
  /* 朴素写法：p = exp(z)/Σexp(z)、loss = −log(p[target])，梯度照样 (p−onehot)/N。
     大尺度下 exp 先溢出成 inf，于是 inf/inf = nan，前向反向一起变 nan。 */
  function ceNaive(logits, targets) {
    const N = logits.length;
    const p = [], nll = [];
    let acc = 0;
    for (let n = 0; n < N; n++) {
      let S = 0; const e = [];
      for (const v of logits[n]) { const a = Math.exp(v); e.push(a); S += a; }
      const row = e.map(a => a / S);
      p.push(row);
      const v = -Math.log(row[targets[n]]);
      nll.push(v); acc += v;
    }
    const inv = 1.0 / N;
    const dlogits = p.map((row, n) => row.map((pv, v) => (pv - (v === targets[n] ? 1 : 0)) * inv));
    return { p, nll, loss: acc / N, dlogits, N };
  }
  function maxAbsDev(A, B) {                                // 两块同形数组的最大绝对偏差
    let m = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) {
      const d = Math.abs(A[i][j] - B[i][j]);
      if (!(d <= m)) m = d;                                 // nan 也要冒出来，不能被 max 吞掉
    }
    return m;
  }

  E.mulberry32 = mulberry32; E.gauss = gauss; E.round = round;
  E.transpose = transpose; E.matmul = matmul; E.linearFwdBwd = linearFwdBwd; E.batchedDw = batchedDw;
  E.softmax = softmax; E.logSoftmax = logSoftmax; E.logSoftmaxNaive = logSoftmaxNaive;
  E.softmaxBwd = softmaxBwd; E.fdSoftmaxBwd = fdSoftmaxBwd; E.relErr = relErr;
  E.ceFwdBwd = ceFwdBwd; E.ceNaive = ceNaive; E.maxAbsDev = maxAbsDev;

  /* ============================================================== 1. matmulbwd
     s03a §6：dW = gᵀ @ x、dx = g @ W、db = g.sum(axis=0)。
     固定 in=3、out=2、N=B·T=4（B=2,T=2），小到每个数都能显示在格子里。
     点结果里任一格 → 高亮"是哪几列/哪几行乘出来的"，并把每一项与总和写全。 */

  const MB = { N: 4, IN: 3, OUT: 2, B: 2, T: 2, CW: 48, CH: 26, W: 820, H0: 386 };

  R.matmulbwd = function (box) {
    injectStyle();
    box.classList.add('lane-e');
    F.title(box, 'Linear 的反向：点结果里任意一格，看它是谁乘出来的 —— `dW = gᵀ @ x`、`dx = g @ W`、`db = g.sum(axis=0)`');
    F.caption(box,
      '`x` 是摊平后的 `(N=4, in=3)`（`B=2,T=2` 乘成一根轴，§2.2），`W` 存成 `(out=2, in=3)`，' +
      '`y = x @ W.T` 得 `(4,2)`；`g` 是上游传下来的、和 `y` 同形的梯度。' +
      '第二排 `dW / db / dx` 里的每个数都是当场按 float64 算出来的。' +
      '`点其中任意一格`：参与它的那一列（或那一行）会被染成橙色，' +
      '下面的算式框把每一项和总和逐行写全 —— 不需要会矩阵微分，只需要看清"压掉的是哪根轴"。' +
      '再打开底部三个错法开关，看错的写法会得到什么形状、和真值差多少。');

    let xSeed = 7, gSeed = 11, gScale = 1, sel = null;
    const wrong = [false, false, false];

    /* ---- 数据：格子里显示的那份就是参与计算的那份（先取整再算，读者才能一位位复现） ---- */
    function build() {
      const rx = mulberry32(xSeed), rg = mulberry32(gSeed);
      const x = [], x3 = [], W = [], g = [], g3 = [];
      for (let n = 0; n < MB.N; n++) {
        const row = [];
        for (let c = 0; c < MB.IN; c++) row.push(round(gauss(rx), 3));
        x.push(row);
      }
      for (let o = 0; o < MB.OUT; o++) {
        const row = [];
        for (let c = 0; c < MB.IN; c++) row.push(round(gauss(rx) * 0.6, 3));
        W.push(row);
      }
      for (let n = 0; n < MB.N; n++) {
        const row = [];
        for (let o = 0; o < MB.OUT; o++) row.push(round(round(gauss(rg), 3) * gScale, 4));
        g.push(row);
      }
      for (let b = 0; b < MB.B; b++) {
        x3.push([]); g3.push([]);
        for (let t = 0; t < MB.T; t++) { x3[b].push(x[b * MB.T + t]); g3[b].push(g[b * MB.T + t]); }
      }
      return { x, x3, W, b: [0, 0], g, g3 };                // b 不影响三件产出，取 0 免得占格子
    }

    const wrap = el('div'); box.appendChild(wrap);
    const { cv, g: ctx } = mkCanvas(wrap, MB.W, MB.H0);
    cv.className = 'pick';
    const terms = el('div', 'lane-e-terms'); box.appendChild(terms);
    const ro = F.readout(box);

    /* ---- 控件 ---- */
    const btns = el('div', 'lane-e-btns'); box.appendChild(btns);
    const bNew = el('button', '', '⟳ 换一组 g（换种子）');
    const bNewX = el('button', '', '⟳ 换一组 x / W');
    const bClear = el('button', '', '取消选中');
    btns.append(bNew, bNewX, bClear);
    bNew.onclick = () => { gSeed = (gSeed * 31 + 17) % 100003; sel = null; draw(); };
    bNewX.onclick = () => { xSeed = (xSeed * 37 + 11) % 100003; sel = null; draw(); };
    bClear.onclick = () => { sel = null; draw(); };

    const sl = F.sliders(box);
    F.slider(sl, 'g 的整体尺度', 0.2, 3, 0.05, gScale, v => { gScale = v; draw(); });

    const wbar = el('div', 'lane-e-btns'); box.appendChild(wbar);
    wbar.appendChild(el('span', 'lane-e-note', '错法对照：'));
    const WNAME = ['① dW = g @ x（忘转置）', '② dx = g @ W.T（转置错方向）', '③ 不摊平，对 (B,T,C) 逐位置乘'];
    WNAME.forEach((t, k) => {
      const b = el('button', 'bad', t);
      b.onclick = () => { wrong[k] = !wrong[k]; b.classList.toggle('on', wrong[k]); draw(); };
      wbar.appendChild(b);
    });

    /* ---- 命中测试：只挂在这个 canvas 上 ---- */
    let hits = [];
    cv.addEventListener('click', e => {
      const r = cv.getBoundingClientRect();
      const px = (e.clientX - r.left) * (MB.W / r.width);
      const py = (e.clientY - r.top) * (cv.height / 2 / r.height);
      for (const h of hits) {
        if (px >= h.x && px <= h.x + h.w && py >= h.y && py <= h.y + h.h) { sel = { kind: h.kind, i: h.i, j: h.j }; draw(); return; }
      }
    });

    /* ---- 选中一格 → 逐项算式 ---- */
    function breakdown(D, L) {
      if (!sel) {
        terms.textContent =
          '点上面第二排任意一个结果格试试：\n' +
          '  · 点 dW[o,c] → 高亮 g 的第 o 列 + x 的第 c 列，算式 = Σₙ g[n,o]·x[n,c]（压掉 n）\n' +
          '  · 点 dx[n,c] → 高亮 g 的第 n 行 + W 的第 c 列，算式 = Σₒ g[n,o]·W[o,c]（压掉 o）\n' +
          '  · 点 db[c]   → 高亮 g 的整列 c，算式 = Σₙ g[n,c]（这就是 sum(axis=0)）\n' +
          '三个都是"配对的轴求和后消失"（§2.4），一次矩阵微分都不用。';
        return;
      }
      const out = [];
      if (sel.kind === 'dW') {
        const o = sel.i, c = sel.j;
        out.push(`dW[${o},${c}] = Σₙ g[n,${o}] · x[n,${c}]        ← 压掉 n（样本轴），留下 out=${o}、in=${c}`);
        out.push(`          = ${L.g.map((_, n) => `g[${n},${o}]·x[${n},${c}]`).join(' + ')}`);
        out.push(`          = ${L.g.map((row, n) => `(${trim(row[o], 4)})·(${trim(L.x[n][c], 3)})`).join(' + ')}`);
        out.push(`          = ${sumStr(L.g.map((row, n) => row[o] * L.x[n][c]), 6)}`);
        out.push(`          = ${full(D.dW[o][c])}        ← g.T @ x 的第 [${o},${c}] 格，形状和 W 一样`);
      } else if (sel.kind === 'dx') {
        const n = sel.i, c = sel.j;
        out.push(`dx[${n},${c}] = Σₒ g[${n},o] · W[o,${c}]        ← 压掉 o（输出轴），留下 n=${n}、in=${c}`);
        out.push(`          = ${L.W.map((_, o) => `g[${n},${o}]·W[${o},${c}]`).join(' + ')}`);
        out.push(`          = ${L.W.map((row, o) => `(${trim(L.g[n][o], 4)})·(${trim(row[c], 3)})`).join(' + ')}`);
        out.push(`          = ${sumStr(L.W.map((row, o) => L.g[n][o] * row[c]), 6)}`);
        out.push(`          = ${full(D.dx[n][c])}        ← g @ W 的第 [${n},${c}] 格，形状和 x 一样`);
      } else {
        const c = sel.j;
        out.push(`db[${c}] = Σₙ g[n,${c}]                     ← 压掉 n；b 只有一维，所以是 g.sum(axis=0)`);
        out.push(`       = ${L.g.map((_, n) => `g[${n},${c}]`).join(' + ')}`);
        out.push(`       = ${L.g.map((row, n) => `(${trim(row[c], 4)})`).join(' + ')}`);
        out.push(`       = ${sumStr(L.g.map(row => row[c]), 6)}`);
        out.push(`       = ${full(D.db[c])}`);
      }
      if (wrong.some(Boolean)) {
        out.push('');
        out.push('打开着的错法会得到什么形状、和真值差多少，见画布底部红框。');
      }
      terms.textContent = out.join('\n');
    }

    /* ---- 错法对照：三种并排画在画布底部，各自一栏 ---- */
    function drawWrong(g2, L, D, top) {
      g2.save();
      g2.textAlign = 'left'; g2.textBaseline = 'alphabetic';
      const COLW = 264;
      for (let k = 0; k < 3; k++) if (wrong[k]) {
        const px = 14 + k * COLW;
        let dy = top + 16;
        const line = (txt, col, font) => { g2.font = font || `10px ${MONO}`; g2.fillStyle = col || FG; g2.fillText(txt, px + 8, dy); dy += 13; };
        g2.strokeStyle = 'rgba(255,107,107,0.5)'; g2.lineWidth = 1;
        g2.strokeRect(px + 0.5, top + 0.5, COLW - 20, 168);
        if (k === 0) {
          line('错法①  dW = g @ x', BADC, `700 11px ${MONO}`); dy += 2;
          line('g 是 (4,2)、x 是 (4,3)');
          line('(4,2) @ (4,3)：内维 2 ≠ 4', BADC);
          line('numpy 当场 ValueError，');
          line('backward 根本跑不完。');
          line('');
          line('抓住它的是解释器，不是 gradcheck。', DIM);
        } else if (k === 1) {
          line('错法②  dx = g @ W.T', BADC, `700 11px ${MONO}`); dy += 2;
          line('g 是 (4,2)、W.T 是 (3,2)');
          line('(4,2) @ (3,2)：内维 2 ≠ 3', BADC);
          line('dx 要和 x 同形 (4,3)，');
          line('只有 g @ W 拼得出来。');
          line('');
          line('同样是 ValueError，不是 gradcheck。', DIM);
        } else {
          const b3 = E.batchedDw(L.g3, L.x3);              // (B,out,in) = (2,2,3)
          line('错法③  不摊平（§2.4）', BADC, `700 11px ${MONO}`); dy += 2;
          line('g.transpose(0,2,1) @ x');
          line('→ (2,2,3)：每个 b 各一份', BADC);
          g2.font = `9px ${MONO}`; g2.fillStyle = DIM;
          for (let b = 0; b < MB.B; b++) g2.fillText(`b=${b}`, px + 8 + b * 104, dy + 9);
          dy += 13;
          for (let b = 0; b < MB.B; b++) {
            paintMat(g2, b3[b], {
              x: px + 8 + b * 104, y: dy, cw: 30, ch: 20, fs: 9,
              color: () => '#e08b8b', text: (i, j) => trim(b3[b][i][j], 2)
            });
          }
          dy += 2 * 20 + 10;
          const mean = b3[0].map((row, o) => row.map((v, c) => (v + b3[1][o][c]) / 2));
          line(`真 dW = b=0 那份 + b=1 那份`, OKC);
          line('直接 += 也会 ValueError；');
          line('改成 .mean(axis=0) 就"能跑"：');
          line(`形状对了、少了 B 倍`, BADC);
          line(`max|错−真| = ${sci(E.maxAbsDev(mean, D.dW), 4)}`, BADC);
        }
      }
      g2.restore();
    }

    /* ---- 绘制 ---- */
    function draw() {
      const L = build();
      const D = E.linearFwdBwd(L.x, L.W, L.b, L.g);
      const anyW = wrong.some(Boolean);
      const H = MB.H0 + (anyW ? 186 : 0);
      cv.height = H * 2; cv.style.height = H + 'px';
      reset(ctx, cv);
      const g2 = ctx;
      hits = [];

      /* 选中态：谁被高亮。key 用 "i,j" 存进 Set，画格子时查一下就行。 */
      const hotG = new Set(), hotX = new Set(), hotW = new Set();
      if (sel) {
        if (sel.kind === 'dW') for (let n = 0; n < MB.N; n++) { hotG.add(n + ',' + sel.i); hotX.add(n + ',' + sel.j); }
        if (sel.kind === 'dx') { for (let o = 0; o < MB.OUT; o++) hotG.add(sel.i + ',' + o); for (let o = 0; o < MB.OUT; o++) hotW.add(o + ',' + sel.j); }
        if (sel.kind === 'db') for (let n = 0; n < MB.N; n++) hotG.add(n + ',' + sel.j);
      }
      const anySel = !!sel;
      const fillOf = (base, set, i, j) => !anySel ? base : (set.has(i + ',' + j) ? WARN : dimmed(base));
      const ringOf = set => (i, j) => set.has(i + ',' + j) ? '#ffe0a8' : null;
      const selRing = kind => (i, j) => (sel && sel.kind === kind && sel.i === i && sel.j === j) ? ACC : null;

      g2.save();
      g2.font = `10px ${MONO}`; g2.fillStyle = DIM; g2.textAlign = 'left';
      g2.fillText('第一排是输入：每块顶上写列号、左边写行号。g 和 y 同形，这就是"上游梯度"。', 26, 24);
      g2.restore();

      /* 第一排：g / x / W / y */
      const Ay = 62;
      paintMat(g2, L.g, {
        x: 26, y: Ay, cw: MB.CW, ch: MB.CH, idx: true, title: 'g（上游梯度）', sub: '(N=4, out=2)',
        color: (i, j) => fillOf(CG, hotG, i, j), ring: ringOf(hotG),
        text: (i, j) => trim(L.g[i][j], 4)
      });
      paintMat(g2, L.x, {
        x: 158, y: Ay, cw: MB.CW, ch: MB.CH, idx: true, title: 'x（摊平后）', sub: '(N=4, in=3)',
        color: (i, j) => fillOf(CX, hotX, i, j), ring: ringOf(hotX)
      });
      paintMat(g2, L.W, {
        x: 330, y: Ay, cw: MB.CW, ch: MB.CH, idx: true, title: 'W（存成 out×in）', sub: '(out=2, in=3)',
        color: (i, j) => fillOf(CWC, hotW, i, j), ring: ringOf(hotW)
      });
      paintMat(g2, D.y, {
        x: 512, y: Ay, cw: MB.CW, ch: MB.CH, idx: true, title: 'y = x @ W.T', sub: '(4, 2)',
        color: () => anySel ? dimmed(CY) : CY
      });

      /* 第二排：三件产出（可点） */
      const By = 262;
      g2.save();
      g2.font = `10px ${MONO}`; g2.fillStyle = OKC; g2.textAlign = 'left';
      g2.fillText('第二排是 backward 要交出的三件 —— 点任意一格看它是谁乘出来的。', 26, By - 44);
      g2.restore();
      const rW = paintMat(g2, D.dW, {
        x: 26, y: By, cw: MB.CW, ch: MB.CH, idx: true, title: 'dW = g.T @ x', sub: '(out=2, in=3) 和 W 同形',
        titleColor: OKC, color: () => (anySel && sel.kind !== 'dW') ? dimmed(COUT) : COUT, ring: selRing('dW')
      });
      const rB = paintMat(g2, [D.db], {
        x: 232, y: By, cw: MB.CW, ch: MB.CH, idx: true, title: 'db = g.sum(axis=0)', sub: '(out=2,) 和 b 同形',
        titleColor: OKC, color: () => (anySel && sel.kind !== 'db') ? dimmed(COUT) : COUT,
        ring: (i, j) => (sel && sel.kind === 'db' && sel.j === j) ? ACC : null
      });
      const rX = paintMat(g2, D.dx, {
        x: 438, y: By, cw: MB.CW, ch: MB.CH, idx: true, title: 'dx = g @ W', sub: '(N=4, in=3) 和 x 同形',
        titleColor: OKC, color: () => (anySel && sel.kind !== 'dx') ? dimmed(COUT) : COUT, ring: selRing('dx')
      });
      rW.forEach(r => hits.push({ x: r.x, y: r.y, w: r.w, h: r.h, i: r.i, j: r.j, kind: 'dW' }));
      rB.forEach(r => hits.push({ x: r.x, y: r.y, w: r.w, h: r.h, i: 0, j: r.j, kind: 'db' }));
      rX.forEach(r => hits.push({ x: r.x, y: r.y, w: r.w, h: r.h, i: r.i, j: r.j, kind: 'dx' }));

      if (anyW) {
        drawWrong(g2, L, D, MB.H0 - 6);
      } else {
        /* 没开错法时，右下角那块空位别浪费：写一句"接下来该干什么" */
        g2.save();
        g2.font = `10.5px ${MONO}`; g2.fillStyle = DIM; g2.textAlign = 'left';
        const hint = [
          '右边这块空位是给错法留的：',
          '打开下面任一开关，这里会画出',
          '那种写法真会得到的形状和数字，',
          '以及它和真值的最大绝对偏差。',
          '',
          '①② 会直接写"numpy ValueError"，',
          '③ 会画出 (2,2,3) 那两份逐位置结果。',
        ];
        hint.forEach((t, i) => g2.fillText(t, 636, 250 + i * 15));
        g2.restore();
      }
      breakdown(D, L);

      /* 底部读数：可以直接抄进 numpy 的几个数 */
      ro.innerHTML =
        `<span>抄这几个数去对拍：<b>dW[0,0] = ${full(D.dW[0][0])}</b>　<b>dW[1,2] = ${full(D.dW[1][2])}</b>　` +
        `<b>db[1] = ${full(D.db[1])}</b>　<b>dx[0,0] = ${full(D.dx[0][0])}</b>　<b>dx[3,2] = ${full(D.dx[3][2])}</b></span>` +
        `<span>y[0,0] = <b>${full(D.y[0][0])}</b>　种子：x/W=${xSeed}，g=${gSeed}，g 尺度=${gScale}</span>` +
        `<span>Σₙ 的 n 只跑 0..3 —— 这就是"摊平"换来的：B、T 合成一根轴，反向只剩一种消失方式。</span>`;
    }

    draw();

    F.caption(box,
      '这三条错法，哪条归 gradcheck 管？ ①② 是形状不合法，numpy 当场抛 `ValueError`，' +
      '连一次完整的 backward 都跑不完 —— 抓住它们的是解释器，不是 gradcheck。' +
      '③ 才是 gradcheck 的活：`(2,2,3)` 直接 `W.grad +=` 同样抛异常，' +
      '但如果你为了塞回去而 `.mean(axis=0)`，形状就成了 `(2,3)`、代码一路绿灯，' +
      '只是每个数都少了 `B` 倍。gradcheck 报的相对误差恰好是 `1 − 1/B = 0.500` —— ' +
      '和 s03c §4.3 那条规律是同一件事：少一个常数因子，误差就等于那个因子的相对偏差。' +
      '这就是 §2.4 说"反向必须摊平"的真正理由：摊平之后 `n` 这根轴只有一种消失方式，' +
      '而三件产出的形状各自只有一种拼法能拼出来（§2.5 那张表不用背）。');
  };

  /* ============================================================= 2. softmaxbwd
     s04 §5：ds = p ⊙ (dp − Σ p·dp)。V=4，dp 四个分量各一个滑杆。
     "每行和为零"是 softmax 平移不变性的对偶（s03c §2.1），所以 Σds 常驻显示。 */

  const SM = { V: 4, W: 800, BW: 62, GAP: 40, X0: 148, HALF: 30 };

  R.softmaxbwd = function (box) {
    injectStyle();
    box.classList.add('lane-e');
    F.title(box, 'softmax 的反向：`ds = p ⊙ (dp − Σ p·dp)` —— 拖 dp 的四个分量，当场看每一项怎么变');
    F.caption(box,
      '`z` 是 4 个分数，`p = softmax(z)`；`dp` 是上游对 `p` 的梯度（4 个滑杆，范围 −1..1）。' +
      '三排柱子从上到下是 `dp`（输入）/ `p`（权重）/ `ds`（输出），绿色为正、红色为负，' +
      '柱子上标的就是真算出来的数。中间那个 `s = Σ p·dp` 是唯一的耦合项：' +
      '它把"每行和恒为 1"这个约束摊到整行上 —— 动任何一格都会改分母，所以整行都得参与修正。' +
      '常驻读数 `Σ ds` 在正确实现下永远是 1e-17 量级，这正是 s03c §2.1 那句' +
      '"softmax 对整体平移不变"的对偶：沿常数方向扰动 `z` 不改变 `p`，' +
      '所以梯度必须垂直于常数方向，也就是每行和为 0。');

    const z0 = [1.2, -0.4, 0.7, -1.5];                      // 固定的一组 logits，锐度滑杆整体缩放它
    let sharp = 1.0;
    const dp = [0.6, -0.35, 0.9, 0.1];
    const wrong = [false, false, false];
    let fdOn = false, lgEps = -6;

    const wrap = el('div'); box.appendChild(wrap);
    const { cv, g: ctx } = mkCanvas(wrap, SM.W, 320);
    const terms = el('div', 'lane-e-terms'); box.appendChild(terms);
    const ro = F.readout(box);

    const sl = F.sliders(box);
    F.slider(sl, '锐度（缩放 z）', 0.1, 6, 0.1, sharp, v => { sharp = v; draw(); });
    for (let k = 0; k < SM.V; k++) {
      F.slider(sl, `dp[${k}]`, -1, 1, 0.01, dp[k], v => { dp[k] = v; draw(); });
    }

    const wbar = el('div', 'lane-e-btns'); box.appendChild(wbar);
    wbar.appendChild(el('span', 'lane-e-note', '错法对照（一次只看一种，柱子才对得上同一把尺）：'));
    const WN = ['① ds = p ⊙ dp（漏减 s）', '② ds = dp − s（漏乘 p）', '③ s = Σ dp（没加权）'];
    const wbs = WN.map((t, k) => {
      const b = el('button', 'bad', t);
      b.onclick = () => {
        for (let q = 0; q < 3; q++) { wrong[q] = q === k ? !wrong[k] : false; wbs[q].classList.toggle('on', wrong[q]); }
        draw();
      };
      wbar.appendChild(b); return b;
    });
    const bFd = el('button', '', '✓ 用有限差分验一次');
    bFd.onclick = () => { fdOn = !fdOn; bFd.classList.toggle('on', fdOn); draw(); };
    wbar.appendChild(bFd);

    const sl2 = F.sliders(box);
    F.slider(sl2, 'lg ε（1e-3 .. 1e-8）', -8, -3, 0.25, lgEps, v => {
      lgEps = v;
      if (!fdOn) { fdOn = true; bFd.classList.add('on'); }
      draw();
    });

    /* 三种错法各自的结果（和真值同形，方便算偏差） */
    function wrongDs(k, p, A) {
      if (k === 0) return p.map((pi, i) => pi * dp[i]);
      if (k === 1) return dp.map(d => d - A.s);
      let s2 = 0; for (const d of dp) s2 += d;
      return p.map((pi, i) => pi * (dp[i] - s2));
    }

    function draw() {
      const z = z0.map(v => v * sharp);
      const p = E.softmax(z);
      const A = E.softmaxBwd(p, dp);
      const wk = wrong.findIndex(Boolean);
      const H = 320 + (wk >= 0 ? 104 : 0);
      cv.height = H * 2; cv.style.height = H + 'px';
      reset(ctx, cv);
      const g2 = ctx;

      g2.save();
      g2.font = `10px ${MONO}`; g2.fillStyle = DIM; g2.textAlign = 'left';
      g2.fillText(`z = [${z.map(v => trim(v, 3)).join(', ')}]　（锐度 ${trim(sharp, 1)} 倍）　V = 4`, 16, 20);
      g2.restore();

      const rows = [
        { name: 'dp（输入）', sub: '上游给 p 的梯度', vals: dp, y: 66, col: null },
        { name: 'p = softmax(z)', sub: '权重，Σp = 1', vals: p, y: 162, col: () => ACC },
        { name: 'ds = p ⊙ (dp − s)', sub: '输出，Σds = 0', vals: A.ds, y: 258, col: null },
      ];
      rows.forEach(r => {
        let mx = 1e-9; for (const v of r.vals) mx = Math.max(mx, Math.abs(v));
        g2.save();
        g2.textAlign = 'right'; g2.textBaseline = 'middle';
        g2.font = `700 11px ${MONO}`; g2.fillStyle = '#cfe0ff';
        g2.fillText(r.name, SM.X0 - 22, r.y - 8);
        g2.font = `9.5px ${MONO}`; g2.fillStyle = DIM;
        g2.fillText(r.sub, SM.X0 - 22, r.y + 8);
        g2.restore();
        paintBars(g2, r.vals, {
          bw: SM.BW, gap: SM.GAP, x0: SM.X0, y0: r.y, half: SM.HALF,
          scale: SM.HALF / mx, color: r.col, label: (k, v) => trim(v, 4)
        });
      });

      /* 错法排：真值画空心垫底、错值画实心红，共用一把尺子才看得出偏差 */
      if (wk >= 0) {
        const wv = wrongDs(wk, p, A);
        const yw = 362;
        let mx = 1e-9;
        for (const v of wv) mx = Math.max(mx, Math.abs(v));
        for (const v of A.ds) mx = Math.max(mx, Math.abs(v));
        g2.save();
        g2.textAlign = 'right'; g2.textBaseline = 'middle';
        g2.font = `700 11px ${MONO}`; g2.fillStyle = BADC;
        g2.fillText(WN[wk].slice(2), SM.X0 - 22, yw - 8);
        g2.font = `9.5px ${MONO}`; g2.fillStyle = DIM;
        g2.fillText('红=错值，空心=真值', SM.X0 - 22, yw + 8);
        g2.restore();
        const sc = SM.HALF / mx;
        paintBars(g2, A.ds, { bw: SM.BW, gap: SM.GAP, x0: SM.X0, y0: yw, half: SM.HALF, scale: sc, hollow: true, label: () => '' });
        paintBars(g2, wv, {
          bw: SM.BW, gap: SM.GAP, x0: SM.X0, y0: yw, half: SM.HALF, scale: sc,
          color: () => BADC, label: (k, v) => trim(v, 4)
        });
      }

      /* ---- 算式框 ---- */
      const out = [];
      out.push('s = Σᵢ p[i]·dp[i]        ← 唯一的耦合项：dp 在 p 下的期望');
      out.push(`  = ${p.map((_, i) => `p[${i}]·dp[${i}]`).join(' + ')}`);
      out.push(`  = ${p.map((pi, i) => `(${trim(pi, 6)})·(${trim(dp[i], 2)})`).join(' + ')}`);
      out.push(`  = ${sumStr(p.map((pi, i) => pi * dp[i]), 6)}`);
      out.push(`  = ${full(A.s)}`);
      out.push('');
      for (let k = 0; k < SM.V; k++) {
        out.push(`ds[${k}] = p[${k}]·(dp[${k}] − s) = (${trim(p[k], 6)})·(${trim(dp[k], 2)} − (${trim(A.s, 6)})) = ${full(A.ds[k])}`);
      }
      const sumDs = A.ds.reduce((a, b) => a + b, 0);
      out.push('');
      out.push(`Σ ds = ${sci(sumDs, 3)}     ← 行和为零。这不是巧合，是"平移不变"的对偶：`);
      out.push(`        给整行 z 加同一个常数 c，p 一点不变，所以梯度必须垂直于常数方向。`);

      if (wk >= 0) {
        const wv = wrongDs(wk, p, A);
        const dev = E.maxAbsDev(wv.map(v => [v]), A.ds.map(v => [v]));
        const rel = Math.max(...wv.map((v, k) => E.relErr(v, A.ds[k])));
        const sw = wv.reduce((a, b) => a + b, 0);
        out.push('');
        if (wk === 0) {
          out.push(`错法① ds = p ⊙ dp（漏减 s）：Σ ds = ${sci(sw, 4)} ≠ 0。`);
          out.push(`  给整行 z 加同一个常数 c，p 不变、loss 不变，可这个"梯度"说每格都该动 ——`);
          out.push(`  平移不变性当场被破坏，于是它必然错。`);
        } else if (wk === 1) {
          out.push(`错法② ds = dp − s（漏乘 p）：Σ ds = ${sci(sw, 4)}。`);
          out.push(`  p 极小的那一格本该几乎不动，这里却给它一个和 dp 同量级的梯度 ——`);
          out.push(`  权重 p 就是"这一格对输出贡献多少"，省不掉。`);
        } else {
          out.push(`错法③ s = Σ dp（没加权）：Σ ds = ${sci(sw, 4)} ≠ 0，行和同样不为零。`);
          out.push(`  正确的 s 是 dp 在 p 下的加权平均，不是算术平均。`);
        }
        out.push(`  与真值的最大绝对偏差 = ${sci(dev, 4)}　gradcheck 口径的最大相对误差 = ${sci(rel, 3)}（阈值 1e-4，必红）`);
      }

      if (fdOn) {
        const eps = Math.pow(10, lgEps);
        const num = E.fdSoftmaxBwd(z, dp, eps);
        const rel = num.map((v, k) => E.relErr(v, A.ds[k]));
        const abs = num.map((v, k) => Math.abs(v - A.ds[k]));
        const mr = Math.max(...rel);
        /* 谷底现扫：ε 从 1e-3 到 1e-8，21 个点，每个点真做一次中心差分。
           不写死"谷底在 1e-6" —— 谷底位置跟 z、dp 有关，现场扫出来才算数。 */
        let vE = lgEps, vR = Infinity;
        for (let i = 0; i <= 20; i++) {
          const le = -8 + i * 5 / 20;
          const nn = E.fdSoftmaxBwd(z, dp, Math.pow(10, le));
          const rr = Math.max(...nn.map((v, k) => E.relErr(v, A.ds[k])));
          if (rr < vR) { vR = rr; vE = le; }
        }
        out.push('');
        out.push(`有限差分现场验证：f(z) = Σ softmax(z)·dp（dp 就是 gradcheck 给数组输出配的余向量）`);
        out.push(`  ε = ${sci(eps, 0)}，中心差分 (f(z+εeₖ) − f(z−εeₖ)) / (2ε)`);
        for (let k = 0; k < SM.V; k++) {
          out.push(`  ds[${k}]：数值 ${full(num[k])}　解析 ${full(A.ds[k])}　相对差 ${sci(rel[k], 2)}`);
        }
        out.push(`  → 最大绝对偏差 ${sci(Math.max(...abs), 3)}，最大相对误差 ${sci(mr, 3)}`);
        out.push(`  现场扫出的谷底：ε = 1e${vE.toFixed(2)}（最大相对误差 ${sci(vR, 2)}）；判据里用的 eps=1e-6 就在谷底一带。`);
        out.push(mr < 1e-4
          ? `  当前 ε 的相对误差在阈值 1e-4 之内：解析公式是对的，这就是现场证据。`
          : `  当前 ε 的相对误差超了 1e-4 —— 偏差来自差分本身（太大是截断、太小是舍入），不是公式错；U 形曲线见 epsnoise。`);
      }
      terms.textContent = out.join('\n');

      ro.innerHTML =
        `<span>s = <b>${full(A.s)}</b></span>` +
        `<span>Σ ds = <b style="color:${Math.abs(sumDs) < 1e-12 ? OKC : BADC}">${sci(sumDs, 3)}</b>（正确实现就该是 1e-17 量级）</span>` +
        `<span>Σ p = <b>${full(p.reduce((a, b) => a + b, 0))}</b></span>` +
        `<span>z = [${z.map(v => trim(v, 3)).join(', ')}]</span>`;
    }

    draw();

    F.caption(box,
      '两个不用等 gradcheck 的自查（s04 §5 原话，这里能当场做）：把四个 `dp` 拖成同一个数，' +
      '`ds` 立刻全变 0 —— 因为 `s` 就是那个数，`dp − s = 0`；' +
      '这正是"给一行所有分数加同一个常数不改变 softmax"，也是阶段 4 里 k-bias 是死参数的同一个性质。' +
      '把锐度拖到接近 0，`p` 变成均匀的 1/4，`ds` 就长成 `¼·(dp − dp 的均值)` —— ' +
      'softmax 在最平坦处的反向就是个 Center 操作。' +
      '最后点「用有限差分验一次」：那是在 JS 里对 softmax 真做中心差分，不是拿解析结果自己对自己；' +
      'ε 从 1e-3 推到 1e-8，偏差先降后升，算式框会现场扫出谷底（通常在 1e-5 ~ 1e-6 一带）—— ' +
      '判据里 `eps=1e-6` 就是这么来的。');
  };

  /* ================================================================== 3. cebwd
     s03c §3：dlogits = (p − onehot)/N。N 行 × V=5，target 可点。
     N 滑杆真的复制/裁剪行数（不是只改除数），于是能看见整块梯度按 1/N 缩。 */

  const CE = { V: 5, CW: 52, CH: 22, GX: 62, BX: 548, BW: 36, BGAP: 10, STRIP: 94, BHALF: 26, W: 800 };

  R.cebwd = function (box) {
    injectStyle();
    box.classList.add('lane-e');
    F.title(box, 'CrossEntropy 的反向就一行：`dlogits = (p − onehot) / N` —— 点一格换 target，看那一根柱子怎么翻负');
    F.caption(box,
      '左边是 `N` 行 × `V=5` 的 logits，`点任意一格就把那一行的 target 设成它`（橙色那格是 target，蓝框那行是当前讲解的行）。' +
      '右边每行一条柱状图画的是 `dlogits[n,v]`：target 那一根 = `p_t − 1` 再除 `N`，永远为负（把正确词往上推），' +
      '其余各根 = `p_v / N`，永远为正（把别的词往下压）。柱子下面三行分别是索引、`dlogits`、`p`，全是当场算的。' +
      '两条常驻读数——每行 `Σ dlogits = 0`、整块 `Σ = 0`——就是 s03c §3.3 那个"秒查"。' +
      '拖 `N` 滑杆时行数是真的在增减，整块柱子按 `1/N` 一起变矮（尺子固定，满格 = 1）。');

    /* 8 行底料；N 滑杆取前 N 行 —— 复制/裁剪，不是"只改除数"。 */
    const base = (() => {
      const r = mulberry32(2024);
      const A = [];
      for (let n = 0; n < 8; n++) { const row = []; for (let v = 0; v < CE.V; v++) row.push(round(gauss(r) * 0.9, 3)); A.push(row); }
      return A;
    })();
    const targets = [2, 0, 4, 1, 3, 0, 2, 1];
    let N = 3, scale = 1, selRow = 0;
    const wrong = [false, false, false];

    const wrap = el('div'); box.appendChild(wrap);
    const { cv, g: ctx } = mkCanvas(wrap, CE.W, 320);
    cv.className = 'pick';
    const terms = el('div', 'lane-e-terms'); box.appendChild(terms);
    const ro = F.readout(box);

    const sl = F.sliders(box);
    F.slider(sl, 'N（几行）', 1, 8, 1, N, v => { N = v; if (selRow >= N) selRow = 0; draw(); });
    F.slider(sl, 'logits 尺度', 1, 800, 1, scale, v => { scale = v; draw(); });

    const wbar = el('div', 'lane-e-btns'); box.appendChild(wbar);
    wbar.appendChild(el('span', 'lane-e-note', '错法对照：'));
    const WN = ['① 忘除 N（loss 用 mean、梯度按 sum）', '② 符号写反（onehot − p）', '③ 对 log_softmax 的输出再套一次 exp 求 p'];
    const wbs = WN.map((t, k) => {
      const b = el('button', 'bad', t);
      b.onclick = () => {
        for (let q = 0; q < 3; q++) { wrong[q] = q === k ? !wrong[k] : false; wbs[q].classList.toggle('on', wrong[q]); }
        draw();
      };
      wbar.appendChild(b); return b;
    });

    let hits = [];
    cv.addEventListener('click', e => {
      const r = cv.getBoundingClientRect();
      const px = (e.clientX - r.left) * (CE.W / r.width);
      const py = (e.clientY - r.top) * (cv.height / 2 / r.height);
      for (const h of hits) if (px >= h.x && px <= h.x + h.w && py >= h.y && py <= h.y + h.h) { targets[h.i] = h.j; selRow = h.i; draw(); return; }
    });

    function draw() {
      const logits = [], tg = [];
      for (let n = 0; n < N; n++) { logits.push(base[n].map(v => v * scale)); tg.push(targets[n]); }
      const A = E.ceFwdBwd(logits, tg);
      const NV = E.ceNaive(logits, tg);

      const H = 62 + N * CE.STRIP + 16;
      cv.height = H * 2; cv.style.height = H + 'px';
      reset(ctx, cv);
      const g2 = ctx;
      hits = [];

      g2.save();
      g2.font = `700 11px ${MONO}`; g2.fillStyle = '#cfe0ff'; g2.textAlign = 'left';
      g2.fillText(`logits（点一格换 target）　尺度 ×${scale}`, CE.GX, 20);
      g2.fillText('dlogits（尺子固定：满格 = 1）', CE.BX, 20);
      g2.font = `9px ${MONO}`; g2.fillStyle = DIM; g2.textAlign = 'center';
      for (let v = 0; v < CE.V; v++) g2.fillText('v=' + v, CE.GX + v * CE.CW + CE.CW / 2, 36);
      g2.restore();

      for (let n = 0; n < N; n++) {
        const ty = 44 + n * CE.STRIP;
        const rects = paintMat(g2, [logits[n]], {
          x: CE.GX, y: ty, cw: CE.CW, ch: CE.CH,
          color: (i, j) => j === tg[n] ? WARN : (n === selRow ? '#a9c6e8' : dimSoft(CX)),
          ring: (i, j) => j === tg[n] ? '#ffe0a8' : null,
          text: (i, j) => Math.abs(logits[n][j]) >= 10000 ? sci(logits[n][j], 1) : trim(logits[n][j], 3)
        });
        rects.forEach(r => hits.push({ x: r.x, y: r.y, w: r.w, h: r.h, i: n, j: r.j }));
        g2.save();
        g2.textBaseline = 'middle'; g2.font = `9.5px ${MONO}`;
        g2.fillStyle = DIM; g2.textAlign = 'right';
        g2.fillText(`n=${n}`, CE.GX - 8, ty + CE.CH / 2);
        g2.textAlign = 'left'; g2.fillStyle = n === selRow ? FG : DIM;
        g2.fillText(`target=${tg[n]}`, CE.GX + CE.V * CE.CW + 12, ty + CE.CH / 2 - 6);
        g2.fillText(`loss_n=${isFinite(A.nll[n]) ? trim(A.nll[n], 4) : String(A.nll[n])}`, CE.GX + CE.V * CE.CW + 12, ty + CE.CH / 2 + 7);
        g2.restore();

        paintBars(g2, A.dlogits[n], {
          bw: CE.BW, gap: CE.BGAP, x0: CE.BX, y0: ty + 4 + CE.BHALF, half: CE.BHALF, scale: CE.BHALF,
          below: true, color: (k) => k === tg[n] ? BADC : OKC,
          label: (k, v) => trim(v, 4),
          idxLabel: k => (k === tg[n] ? '★' + k : 'v' + k),
          extra: k => 'p=' + trim(A.p[n][k], 3)
        });
      }

      /* ---- 算式框 ---- */
      const out = [];
      const n = selRow, t = tg[n];
      out.push(`第 ${n} 行（target = ${t}，N = ${N}，1/N = ${full(1 / N)}）—— 逐根写全：`);
      for (let v = 0; v < CE.V; v++) {
        out.push(v === t
          ? `  dlogits[${n},${v}] = (p − 1)/N = (${trim(A.p[n][v], 6)} − 1) × ${trim(1 / N, 4)} = ${full(A.dlogits[n][v])}   ← 负，把正确词往上推`
          : `  dlogits[${n},${v}] = p/N      = ${trim(A.p[n][v], 6)} × ${trim(1 / N, 4)} = ${full(A.dlogits[n][v])}`);
      }
      out.push('');
      out.push(`loss = −Σₙ log p[n,target] / N = (${A.nll.map(v => isFinite(v) ? trim(v, 4) : String(v)).join(' + ')}) / ${N}`);
      out.push(`     = ${isFinite(A.loss) ? full(A.loss) : String(A.loss)}　　（ln V = ln 5 = ${trim(Math.log(CE.V), 6)} 是随机初始化的基准线）`);
      const rowSums = A.dlogits.map(r => r.reduce((a, b) => a + b, 0));
      const allSum = rowSums.reduce((a, b) => a + b, 0);
      out.push('');
      out.push(`每行 Σ dlogits = [${rowSums.map(v => sci(v, 1)).join(', ')}]`);
      out.push(`  → 最大 ${sci(Math.max(...rowSums.map(Math.abs)), 1)}；整块 Σ = ${sci(allSum, 1)}`);
      out.push(`  这就是 s03c §3.3 的秒查：Σᵥ(p − onehot) = 1 − 1 = 0，写完后 np.abs(g.sum(-1)).max() 一查就知道。`);
      if (!isFinite(A.loss)) out.push('  （融合写法在任何尺度下都不该出 nan —— 出 nan 说明 log_probs 那步没减最大值）');

      const wk = wrong.findIndex(Boolean);
      if (wk >= 0) {
        out.push('');
        if (wk === 0) {
          const w = A.p.map((row, i) => row.map((pv, v) => pv - (v === tg[i] ? 1 : 0)));
          const rel = Math.max(...w.map((row, i) => Math.max(...row.map((pv, v) => E.relErr(pv, A.dlogits[i][v])))));
          out.push(`错法① 忘除 N：dlogits = p − onehot，和真值差整整 N 倍（N = ${N}）。`);
          out.push(`  max|错−真| = ${sci(E.maxAbsDev(w, A.dlogits), 4)}　gradcheck 相对误差 = ${trim(rel, 4)} = (N−1)/N`);
          out.push(`  判据里"梯度量级"那条会红。更坏的是它在阶段 3 不炸，到阶段 5 才显形：等效学习率被放大 B·T 倍。`);
        } else if (wk === 1) {
          const w = A.dlogits.map(row => row.map(v => -v));
          out.push(`错法② 符号写反：onehot − p = −(p − onehot)。`);
          out.push(`  max|错−真| = ${sci(E.maxAbsDev(w, A.dlogits), 4)}　每根柱子都翻了号。`);
          out.push(`  秒查一眼看穿：真值里 target 那一根必须为负，这里变正了 —— 那一定是 one-hot 放错了地方。`);
        } else {
          const p3 = logits.map(z => E.logSoftmaxNaive(z).map(v => Math.exp(v)));
          const w = p3.map((row, i) => row.map((pv, v) => (pv - (v === tg[i] ? 1 : 0)) * (1 / N)));
          const dev = E.maxAbsDev(w, A.dlogits);
          const s3 = p3.map(r => r.reduce((a, b) => a + b, 0));
          out.push(`错法③ p = exp(log_softmax(z))，而且 log_softmax 是"分开写两层"那种不减最大值的：`);
          out.push(`  Σ p = [${s3.map(v => isFinite(v) ? trim(v, 6) : String(v)).join(', ')}]`);
          out.push(`  max|错−真| = ${isFinite(dev) ? sci(dev, 4) : String(dev)}　朴素 loss = ${isFinite(NV.loss) ? full(NV.loss) : String(NV.loss)}`);
          out.push(`  小尺度下它只差 1e-16 量级（gradcheck 都可能放过）；`);
          out.push(`  把"尺度"推到 710 以上：Σexp(z) 变 inf → log 出来 inf → exp(inf−inf) = nan，前向反向一起烂。`);
        }
      }
      if (scale > 400) {
        out.push('');
        out.push(`当前尺度 ×${scale}（exp 的 float64 天花板在 709 附近）：`);
        out.push(`  朴素写法 loss = ${String(NV.loss)}，dlogits[0][0] = ${String(NV.dlogits[0][0])}`);
        out.push(`  融合写法 loss = ${isFinite(A.loss) ? full(A.loss) : String(A.loss)}，dlogits[0][0] = ${full(A.dlogits[0][0])}`);
        out.push(`  反向为什么也干净：(p − onehot)/N 里的 p = e/Z、e = exp(z−m) ≤ 1，`);
        out.push(`  全程没有 exp(大数)、也没有 log(下溢成 0 的小数)。`);
      }
      terms.textContent = out.join('\n');

      ro.innerHTML =
        `<span>loss = <b style="color:${isFinite(A.loss) ? OKC : BADC}">${isFinite(A.loss) ? fmt(A.loss, 6) : String(A.loss)}</b></span>` +
        `<span>朴素写法 loss = <b style="color:${isFinite(NV.loss) ? DIM : BADC}">${isFinite(NV.loss) ? fmt(NV.loss, 6) : String(NV.loss)}</b></span>` +
        `<span>每行 Σ dlogits 最大 = <b>${sci(Math.max(...rowSums.map(Math.abs)), 1)}</b></span>` +
        `<span>整块 Σ = <b>${sci(allSum, 1)}</b></span>` +
        `<span>dlogits[0][0] = <b>${full(A.dlogits[0][0])}</b>　dlogits[${N - 1}][${tg[N - 1]}] = <b>${full(A.dlogits[N - 1][tg[N - 1]])}</b></span>`;
    }

    draw();

    F.caption(box,
      '把「logits 尺度」推到 800：朴素写法（`p = exp(z)/Σexp(z)`，也就是分成 Softmax 层 + NLL 层那种）' +
      '的 loss 直接变 `nan`，`(p − onehot)/N` 跟着变 `nan`；融合写法一位没动，反向同样干净 —— ' +
      '这里看的不是正向会不会溢出（那是 softshift 那张图讲的事），而是 `dlogits` 会不会跟着一起烂。' +
      '再把 `N` 从 1 拖到 8：柱子整体变矮，因为 `mean` 和"除以 N"是同一件事的两面（§3.4）。' +
      '最后随手点几格换 target：target 那一根立刻从正翻到负，而每行的和始终是 0 到 1e-17 量级 —— ' +
      '这条不变量比任何公式都好记，也是判据里 `np.abs(g.sum(-1)).max() = 2.26e-17` 那个数的来历。');
  };

  /* ------------------------------------------------- 4. 广播/求和 的反向（动画）
     例卡 6 那两句话的动画版：左边看"b 被复制成 N 行 → 反向沿那根轴求和"，
     右边看"N 个数合成 1 个 → 反向每人分回同一份"。两半的贡献都画成真的在流的点：
     左边每个 g 格吐一颗、落进同列的 db 格（落进去才计入，所以能看见"加起来"）；
     右边 g_m 裂成 N 颗一样大的、各回各的 dy 格。均值/求和一切换，右边每颗点的大小
     从 g_m/N 变成 g_m —— `/N` 的全部含义就在这一切换里。 */
  R.bcastbwd = function (box) {
    injectStyle();
    box.classList.add('lane-e');
    F.title(box, '广播与求和是一对反着走的操作：贡献真的在流，落进哪格、加多少，都当场算');

    const CW = 56, CH = 30, GAP = 8;
    const W = 880, H = 300;
    const FR = 116;                                   // 一轮动画的总帧数
    let gScale = 1, gm = 6, meanMode = true, seed = 7;
    let t = 0, raf = null;

    const { cv, g } = mkCanvas(box, W, H);

    const btns = el('div', 'lane-e-btns'); box.appendChild(btns);
    const sl = el('div', 'sliders'); box.appendChild(sl);
    const terms = el('div', 'lane-e-terms'); box.appendChild(terms);
    const ro = F.readout(box);

    function data() {
      const rnd = E.mulberry32(seed);
      const x = [0, 1].map(() => [0, 1, 2].map(() => Math.round((rnd() * 8 - 4) * 10) / 10));
      const b = [0, 1, 2].map(() => Math.round((rnd() * 6 - 3) * 10) / 10);
      const g0 = [0, 1].map(() => [0, 1, 2].map(() => Math.round((rnd() * 8 - 4) * 10) / 10));
      const gg = g0.map(r => r.map(v => Math.round(v * gScale * 100) / 100));
      const A = E.bcastBwd(gg, gm, meanMode);
      return { x, b, g: gg, ...A };
    }

    const ease = p => p <= 0 ? 0 : p >= 1 ? 1 : p * p * (3 - 2 * p);
    /* 第 n 行贡献的出发时刻错开，才看得出"先落一颗、再落一颗、加起来"。 */
    const depL = n => 8 + n * 16, DUR = 34;
    const depR = i => 8 + i * 4;

    function cell(g2, x, y, v, col, o) {
      o = o || {};
      g2.save();
      g2.fillStyle = o.fill || '#151d2b';
      g2.strokeStyle = o.ring || col; g2.lineWidth = o.ring ? 2 : 1;
      g2.beginPath(); g2.roundRect(x, y, CW, CH, 5); g2.fill(); g2.stroke();
      g2.fillStyle = o.dim ? DIM : FG; g2.font = `12px ${MONO}`;
      g2.textAlign = 'center'; g2.textBaseline = 'middle';
      g2.fillText(v == null ? '' : String(v), x + CW / 2, y + CH / 2 + 1);
      g2.restore();
    }
    /* chip=true 时在字后面垫一块底色：虚线箭头从标签底下过也不会划到字。 */
    const lab = (g2, x, y, s, col, chip) => {
      g2.save(); g2.font = `700 11px ${MONO}`; g2.textAlign = 'left'; g2.textBaseline = 'alphabetic';
      if (chip) { g2.fillStyle = BG; g2.fillRect(x - 3, y - 11, g2.measureText(s).width + 6, 14); }
      g2.fillStyle = col || '#cfe0ff'; g2.fillText(s, x, y); g2.restore();
    };
    const dot = (g2, x, y, col, v) => {
      g2.save(); g2.fillStyle = col; g2.beginPath(); g2.arc(x, y, 4.5, 0, 7); g2.fill();
      g2.fillStyle = FG; g2.font = `10px ${MONO}`; g2.textAlign = 'left'; g2.textBaseline = 'middle';
      g2.fillText(trim(v, 2), x + 7, y); g2.restore();
    };
    const lerp = (a, b, p) => [a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p];

    function stepAt(tt) {
      const D = data();
      reset(g, cv);
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';

      /* ---------- 左半：广播 ---------- */
      lab(g, 24, 20, '正向 y = x + b：b 只有 3 个数，被复制成 2 行去相加');
      const bx = 24, by = 40, xx = 24, xy = 92, gx = 258, gy = 92, dx = 258, dy2 = 214;
      lab(g, bx, by - 6, 'b (3,)', CWC);
      D.b.forEach((v, j) => cell(g, bx + j * (CW + GAP), by, trim(v, 2), CWC));
      /* 复制的虚线箭头：b 的第 j 个数 → 两行的第 j 列。先画箭头再画 x 的标签，
         标签垫了底色，箭头从它底下过就不会划到字。 */
      g.save(); g.strokeStyle = dimSoft(CWC); g.setLineDash([3, 3]); g.lineWidth = 1;
      for (let j = 0; j < 3; j++) for (let n = 0; n < 2; n++) {
        g.beginPath();
        g.moveTo(bx + j * (CW + GAP) + CW / 2, by + CH);
        g.lineTo(xx + j * (CW + GAP) + CW / 2, xy + n * (CH + 4) - 2);
        g.stroke();
      }
      g.restore();
      lab(g, xx, xy - 6, 'x (2,3)', CX, true);
      D.x.forEach((r, n) => r.forEach((v, j) => cell(g, xx + j * (CW + GAP), xy + n * (CH + 4), trim(v, 2), CX)));
      lab(g, gx, gy - 6, 'g（上游梯度，(2,3)）', CG);
      D.g.forEach((r, n) => r.forEach((v, j) => cell(g, gx + j * (CW + GAP), gy + n * (CH + 4), trim(v, 2), CG)));
      /* 结果行的名字放在行**下面**：落点的走廊里不留字，飞行的点才不会压到标题。 */
      lab(g, dx, dy2 + CH + 16, 'db (3,)：同列两颗落进来相加', COUT);
      for (let j = 0; j < 3; j++) {
        let acc = 0, arrived = 0;
        for (let n = 0; n < 2; n++) {
          const p = ease((tt - depL(n)) / DUR);
          if (p >= 1) { acc += D.g[n][j]; arrived++; }
          const from = [gx + j * (CW + GAP) + CW / 2, gy + n * (CH + 4) + CH];
          const to = [dx + j * (CW + GAP) + CW / 2, dy2 + 2];
          if (p > 0 && p < 1) { const q = lerp(from, to, p); dot(g, q[0], q[1], WARN, D.g[n][j]); }
        }
        cell(g, dx + j * (CW + GAP), dy2, arrived ? trim(acc, 3) : '·', COUT, { ring: arrived === 2 ? OKC : null });
      }

      /* ---------- 右半：求和 / 均值 ---------- */
      const rx = 470, ry = 92, mx = 664, my = 34, gmx = 664, gmy = 150, dyy = 232;
      lab(g, rx, 20, meanMode ? '正向 m = (1/6)·Σ yᵢ：6 个数合成 1 个' : '正向 s = Σ yᵢ：6 个数合成 1 个（不除 N）');
      lab(g, rx, ry - 6, 'y (6,)', CX);
      D.y.forEach((v, i) => cell(g, rx + i * (CW + GAP) - (i > 2 ? 8 : 0), ry, trim(v, 2), CX));
      cell(g, mx, my, meanMode ? trim(D.m, 2) : trim(D.s, 2), CY, { ring: CY });
      lab(g, mx + CW + 8, my + 20, meanMode ? 'm' : 's', CY);
      g.save(); g.strokeStyle = dimSoft(CX); g.setLineDash([3, 3]);
      for (let i = 0; i < 6; i++) { g.beginPath(); g.moveTo(rx + i * (CW + GAP) - (i > 2 ? 8 : 0) + CW / 2, ry); g.lineTo(mx + CW / 2, my + CH); g.stroke(); }
      g.restore();
      cell(g, gmx, gmy, trim(gm, 2), CG, { ring: CG });
      lab(g, gmx + CW + 8, gmy + 20, 'g_m = ∂L/' + (meanMode ? 'm' : 's'), CG);
      lab(g, rx, dyy + CH + 16, meanMode ? 'dy (6,)：每人分回 g_m/6' : 'dy (6,)：每人分回整个 g_m', COUT);
      for (let i = 0; i < 6; i++) {
        const p = ease((tt - depR(i)) / DUR);
        const cx0 = rx + i * (CW + GAP) - (i > 2 ? 8 : 0);
        const from = [gmx + CW / 2, gmy + CH];
        const to = [cx0 + CW / 2, dyy + 2];
        if (p > 0 && p < 1) { const q = lerp(from, to, p); dot(g, q[0], q[1], WARN, D.dy[i]); }
        cell(g, cx0, dyy, p >= 1 ? trim(D.dy[i], 3) : '·', COUT, { ring: p >= 1 ? OKC : null });
      }
      return D;
    }

    function draw() {
      const D = stepAt(t);
      const dbStr = '[' + D.db.map(v => trim(v, 4)).join(', ') + ']';
      const per = meanMode ? gm / 6 : gm;
      terms.textContent =
        '左（广播的反向是求和）：∂L/b_j = Σₙ g[n,j] = g[0,j] + g[1,j]\n' +
        D.db.map((v, j) => `  db[${j}] = ${sumStr([D.g[0][j], D.g[1][j]], 3)} = ${trim(v, 4)}`).join('\n') +
        `\n  → db = ${dbStr}，形状 (3,) 和 b 同形；numpy 一行就是 g.sum(axis=0)。\n` +
        (meanMode
          ? '右（均值的反向是广播再除 N）：∂L/∂yᵢ = g_m / 6 = ' + trim(gm, 3) + ' / 6 = ' + trim(per, 4) + '\n'
            + '  六格都是同一个数，形状和 y 同形；Σ dy = ' + trim(D.dySum, 4) + ' = g_m（贡献守恒，一颗不多一颗不少）。'
          : '右（求和的反向是广播，不除 N）：∂L/∂yᵢ = g_m = ' + trim(per, 4) + '\n'
            + '  和均值那条只差一个 /6：loss = sum/N 里的 /N 就是"均值那条"多出来的那一步。');
      ro.innerHTML =
        `<span>db = <b>${dbStr}</b></span>` +
        `<span>每颗点的大小：左 = g[n,j]，右 = <b>${trim(per, 4)}</b>${meanMode ? '（g_m/6）' : '（g_m）'}</span>` +
        `<span>Σ dy = <b>${trim(D.dySum, 4)}</b>${meanMode ? ' = g_m ✓' : ` = 6·g_m ✓`}</span>` +
        `<span>帧 <b>${t}</b> / ${FR}</span>`;
    }

    function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }
    function play() {
      stop();
      const tick = () => { t = t >= FR ? 0 : t + 1; draw(); raf = requestAnimationFrame(tick); };
      raf = requestAnimationFrame(tick);
    }
    const mk = (txt, fn) => { const b = el('button', null, txt); b.onclick = fn; btns.appendChild(b); return b; };
    mk('▶ 播放', play);
    mk('⏭ 单步', () => { stop(); t = Math.min(FR, t + 1); draw(); });
    mk('⏸ 暂停', stop);
    mk('↺ 重置', () => { stop(); t = 0; draw(); });
    mk(meanMode ? '当前：均值（/N）' : '当前：求和（不除 N）', function () {
      meanMode = !meanMode;
      this.textContent = meanMode ? '当前：均值（/N）' : '当前：求和（不除 N）';
      draw();
    });
    mk('换一组 g / x / b', () => { seed = (seed * 31 + 17) % 100003; draw(); });
    F.slider(sl, 'g 尺度', 0.2, 3, 0.1, gScale, v => { gScale = v; draw(); });
    F.slider(sl, 'g_m', 1, 12, 1, gm, v => { gm = v; draw(); });

    draw();
    F.caption(box,
      '这张图只讲一件事：**一个数被用了几次，反向就收几份；几个数合成一个数，反向就每人分一份。**' +
      '左边 b 的第 j 个数被两行用过，所以 db[j] 收两颗点（g[0,j] 和 g[1,j]），加起来就是 `g.sum(axis=0)`；' +
      '右边 6 个 y 合成一个 m，所以 g_m 裂成 6 颗一样大的点各回各家。' +
      '点「均值/求和」切换看右边每颗点从 g_m/6 变成 g_m —— `CrossEntropy` 里那个 `/N`、' +
      '`RMSNorm` 里那个 `s`，全是这一颗点的大小问题。所有数字当场算，readout 里可以直接抄去对拍。');
  };

  /* 广播/求和反向的纯计算：供 node 与 numpy 对拍。 */
  E.bcastBwd = function (g, gm, meanMode) {
    const N = g.length, K = g[0].length;
    const db = [];
    for (let j = 0; j < K; j++) { let s = 0; for (let n = 0; n < N; n++) s += g[n][j]; db.push(s); }
    const y = [];
    for (let n = 0; n < N; n++) for (let j = 0; j < K; j++) y.push(g[n][j] * 0 + n * K + j + 1);
    const s = y.reduce((a, b) => a + b, 0);
    const per = meanMode ? gm / y.length : gm;
    const dy = y.map(() => per);
    return { db, y, s, m: s / y.length, dy, dySum: dy.reduce((a, b) => a + b, 0) };
  };

  R.__e_math = E;
})();
