/* 原理动画 · 第 G 批：非线性与逐层反向的动线（阶段 2 / 3b）。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和前几批一样：能算的都真的算 —— tanh/1−t²、L 层梯度连乘、erf/Φ、
   GELU 两种式子及其导数、MLP 正向三步反向六步的每个数值矩阵全部在 JS 里
   当场跑，readout 里的数字可以直接拿去和 numpy / math.erf 对。

   本批三个组件：tanhsat / geludx / mlpbwd。
     tanhsat —— s02 §6（TODO-4 tanh）：饱和与"深层梯度消失"是同一个 1−t²。
                和 figs.js 的 steps（台阶拟合 sin，讲容量）、figs_c.js 的 epsnoise
                （有限差分 ε 选择，讲数值噪声）都不同：这里只讲深度方向的连乘。
     geludx  —— s03b §3（TODO-9 GELU）：精确式 x·Φ(x) 与 solutions/minigpt/nn.py
                真实使用的 tanh 近似式，值与导数四条曲线 + 三种错法对照。
     mlpbwd  —— s03b §6（TODO-10 MLP）：正向三块 + 反向按 solutions 里
                MLP.backward 的真实顺序（fc2 → act → fc1）逐步单放。

   共同约定：
     · 公式与 solutions/minigpt/nn.py、scalar.py 一字不差：tanh 反向用 1−t²
       （t 是正向缓存的值）；GELU 是 tanh 近似 y = 0.5x(1+tanh(K(x+Ax³)))，
       K=√(2/π)、A=0.044715，导数 0.5(1+t) + 0.5x(1−t²)K(1+3Ax²)；
       Linear.backward 的顺序是 dW = gᵀ@x_flat、db = g.sum(axis=0)、return g@W。
     · erf 自己写：|z|<3 用 Maclaurin 级数，|z|≥3 用 erfc 连分数（Gauss CF，
       自底向上 300 层）。Φ(x) = 0.5(1+erf(x/√2))。级数在大 |z| 有抵消，
       所以两条路径在组件里当场互检，离线与 Python math.erf 对过账。
     · 每个组件都有 ▶ 播放 / ⏭ 单步 / ⏸ 暂停 / ↺ 重置；任意一帧的状态由纯函数
       stepAt(t) 从头重算（单步不等 rAF，无头浏览器能截到确定的第 t 帧）；
       rAF 只在播放期间存在，mount 时不起任何定时器。
     · 事件只挂在自己创建的元素上（滑杆 / 按钮 / canvas），不碰 window/document。
     · 自带样式全部以 `.lane-g ` 前缀注入（<style id="lane-g-style">），不碰 manual.css。
     · 组件内文案不用 $…$（MathJax 不排版动态内容），公式一律 unicode 明文。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  if (!F || !R) return;
  const el = F.el, fmt = F.fmt;

  /* ---------------------------------------------------------------- 公共工具 */

  (function injectStyle() {
    if (typeof document === 'undefined' || !document.getElementById) return;
    if (document.getElementById('lane-g-style')) return;
    const s = el('style'); s.id = 'lane-g-style';
    s.textContent = [
      '.lane-g canvas{display:block;max-width:100%;margin:6px 0 0;',
      '  border:1px solid var(--line);border-radius:8px;background:#101722}',
      '.lane-g .lane-g-btns{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 2px;align-items:center}',
      '.lane-g .lane-g-legend{display:flex;flex-wrap:wrap;gap:3px 16px;font:11.5px/1.7 var(--mono);color:var(--dim)}',
      '.lane-g .lane-g-legend i{display:inline-block;width:18px;height:0;border-top:2.4px solid currentColor;margin-right:5px;vertical-align:middle}',
      '.lane-g .lane-g-legend i.dash{border-top-style:dashed}',
      '.lane-g .lane-g-note{font:11.5px/1.5 var(--mono);color:var(--dim);margin-top:4px}',
      '.lane-g .lane-g-chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 2px}',
      '.lane-g .lane-g-outs{font:12px/1.8 var(--mono);margin-top:8px;border-top:1px dashed var(--line);padding-top:6px}',
      '.lane-g .lane-g-outs .pending{opacity:.45}'
    ].join('\n');
    (document.head || document.documentElement).appendChild(s);
  })();

  const C = { blue: '#4aa3ff', green: '#3ecf8e', gold: '#f5b445', red: '#ff6b6b',
              purple: '#b9a7e0', dim: '#8d99ac', fg: '#e6ebf3' };

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  /* 播放控制（同 figs_b/d 的约定，自己一份）：▶ / ⏭ / ⏸ / ↺。
     redraw(t) 必须是纯的 —— 状态从 t 重算，单步按钮不依赖 rAF。 */
  function playback(box, getMax, redraw, intervalMs) {
    const row = el('div', 'lane-g-btns');
    const bPlay = el('button', '', '▶ 播放');
    const bStep = el('button', '', '⏭ 单步');
    const bPause = el('button', '', '⏸ 暂停');
    const bReset = el('button', '', '↺ 重置');
    row.append(bPlay, bStep, bPause, bReset);
    box.appendChild(row);
    const iv = intervalMs || 700;
    let t = 0, raf = 0, playing = false, last = 0;
    function stop() {
      playing = false;
      if (raf) cancelAnimationFrame(raf);
      raf = 0; bPlay.classList.remove('on');
    }
    function frame(ts) {
      if (!playing) return;
      if (ts - last >= iv) {
        last = ts;
        if (t < getMax()) { t++; redraw(t); }
        else { stop(); return; }
      }
      raf = requestAnimationFrame(frame);
    }
    bPlay.onclick = () => {
      if (playing) return;
      if (t >= getMax()) t = 0;
      playing = true; bPlay.classList.add('on');
      last = (window.performance && performance.now) ? performance.now() : Date.now();
      raf = requestAnimationFrame(frame);
    };
    bPause.onclick = stop;
    bStep.onclick = () => { stop(); if (t < getMax()) { t++; redraw(t); } };
    bReset.onclick = () => { stop(); t = 0; redraw(0); };
    return {
      get t() { return t; }, set t(v) { t = v; redraw(t); }, stop
    };
  }

  /* 画布宽度：跟随讲义栏宽，canvas 本身 max-width:100% 兜底。 */
  function plotW(box, max) { return Math.max(340, Math.min(max || 870, (box.clientWidth || (max || 870)) - 16)); }

  function legend(host, items) {
    const row = el('div', 'lane-g-legend');
    for (const [color, dash, text] of items) {
      const sp = el('span');
      sp.innerHTML = `<i class="${dash ? 'dash' : ''}" style="color:${color}"></i>${F.md(text)}`;
      row.appendChild(sp);
    }
    host.appendChild(row);
    return row;
  }

  /* ================================================================ 数学核心
     全部纯函数，node 自测脚本经 R.__g_math 调用，与 python/math.erf 对账。 */

  /* 与自测脚本共用的确定性 rng（同 figs_b：mulberry32 + Box-Muller）。 */
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

  /* ---------------- erf：小 |z| 级数、大 |z| 连分数 ----------------
     Maclaurin 级数 erf(z) = (2/√π) Σ (−1)ⁿ z^{2n+1}/(n!(2n+1)) 处处收敛，
     但 |z|>3 后最大项 ~z^{2n+1}/n! 先到 1e4 量级再掉下来，float64 抵消会把
     绝对误差抬到 1e-11 —— 所以大 |z| 换 erfc 的 Gauss 连分数：
       erfc(z) = e^{−z²}/√π ÷ (z + (1/2)/(z + 1/(z + (3/2)/(z + …))))
     系数 kₙ = n/2，自底向上截 300 层。z≥3 时它几十层就到机器精度。 */
  function erfSeries(z) {
    let term = z, sum = z, n = 0;
    for (;;) {
      n++;
      term *= -z * z / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) <= 1e-18 * (Math.abs(sum) || 1e-300) || n > 400) break;
    }
    return sum * 2 / Math.sqrt(Math.PI);
  }
  function erfcCF(z) {
    let t = 0;
    for (let n = 300; n >= 1; n--) t = (n / 2) / (z + t);
    return Math.exp(-z * z) / Math.sqrt(Math.PI) / (z + t);
  }
  function erf(z) {
    const s = z < 0 ? -1 : 1, a = Math.abs(z);
    if (a < 3) return s * erfSeries(a);
    return s * (1 - erfcCF(a));
  }
  const SQRT2 = Math.sqrt(2), SQRT2PI = Math.sqrt(2 * Math.PI);
  /* Φ 用 0.5(1+erf)：大负 x 处 1+erf 有抵消，但 erfc 分支保住了绝对误差 ~1e-16，
     而 Φ(−6)≈1e-9 本身要的就是绝对精度，导数 x·φ(x) 同理。 */
  function normCdf(x) { return 0.5 * (1 + erf(x / SQRT2)); }
  function normPdf(x) { return Math.exp(-x * x / 2) / SQRT2PI; }

  /* ---------------- GELU：两种式子、两个导数 ----------------
     tanh 近似式 = solutions/minigpt/nn.py 的 GELU 一字不差（K、A 是类属性）；
     精确式 x·Φ(x) 只作对照。导数：
       近似（backward 原文）：0.5(1+t) + 0.5x(1−t²)·K(1+3Ax²)，t=tanh(K(x+Ax³))
       精确（乘积法则）    ：Φ(x) + x·φ(x) */
  const GK = Math.sqrt(2 / Math.PI), GA = 0.044715;
  function geluTanh(x) {
    const inner = GK * (x + GA * x * x * x);
    return 0.5 * x * (1 + Math.tanh(inner));
  }
  function geluTanhDx(x) {
    const inner = GK * (x + GA * x * x * x);
    const t = Math.tanh(inner);
    const du_dx = GK * (1 + 3 * GA * x * x);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du_dx;
  }
  function geluExact(x) { return x * normCdf(x); }
  function geluExactDx(x) { return normCdf(x) + x * normPdf(x); }

  /* ---------------- 深度连乘 ----------------
     模型（讲义口径）：每层停在同一个工作点 x，每层"先乘权重 w 再过激活"，
     回传时梯度每过一层乘一次 w·act′(x)，L 层就是 [w·act′(x)]^L = w^L·Π(act′)。
     返回逐层的 log10 倍数（用 log 累加而不是直接乘：16 层 1e-3 早就下溢出 0，
     log 域才画得出曲线）。dead = 导数为 0（ReLU 死区 / GELU 深负区饱和）。 */
  const F32_MIN = 1.1754943508222875e-38;               // float32 最小正规数
  const LOG10_F32MIN = Math.log10(F32_MIN);             // ≈ −37.93
  function actPair(act, x) {                            // [激活值, 导数]
    if (act === 'relu') return [x > 0 ? x : 0, x > 0 ? 1 : 0];
    if (act === 'gelu') return [geluTanh(x), geluTanhDx(x)];
    const t = Math.tanh(x);
    return [t, 1 - t * t];                              // scalar.py TODO-4：用 t 自己
  }
  function deepChain(act, x, w, L) {
    const [, d] = actPair(act, x);
    const m = w * d;                                    // 每层的梯度倍数
    const dead = !(m > 0);                              // 0 或数值上塌成 0
    const lm = dead ? -Infinity : Math.log10(m);
    const logs = [];
    for (let l = 1; l <= L; l++) logs.push(dead ? -Infinity : l * lm);
    // 第几层塌到 float32 下限：l·lm ≤ −37.93 → l ≥ 37.93/(−lm)
    const n32 = dead ? 1 : (lm < 0 ? Math.ceil(LOG10_F32MIN / lm) : null);
    return { d, m, lm, logs, dead, n32 };
  }

  /* ---------------- MLP：小尺寸真算，两条独立路径 ----------------
     尺寸照任务书：x (2,3) → Linear1 (3→4) → GELU → Linear2 (4→2)，上游 g (2,2)。
     路径 A（stepwise）：每一步都按公式语义写显式 Σ 循环 —— 相当于学生手算，
       顺序严格照 solutions MLP.backward：fc2（dW2/db2 → return dh）→ act → fc1。
     路径 B（oneshot）：通用 matmul/transpose/colsum 一口气组合 —— numpy 式直算。
     两条路径独立实现，最后一帧对拍，最大偏差应在 1e-16 量级或恰好 0。 */
  function makeMLPData(seed) {
    const rnd = mulberry32(seed == null ? 42 : seed);
    const rn = () => gauss(rnd);
    const fill = (r, c, s) => {
      const M = [];
      for (let i = 0; i < r; i++) { const row = []; for (let j = 0; j < c; j++) row.push(rn() * (s == null ? 1 : s)); M.push(row); }
      return M;
    };
    return {
      x: fill(2, 3),
      // Linear 的 fan-in 缩放初始化（solutions/nn.py：std·√(2/in_f)，这里 std=1 让数字可读）；
      // b 和答案实现一样初始化为全 0 —— db 仍然非零，偏置梯度 = 各行梯度求和。
      W1: fill(4, 3, Math.sqrt(2 / 3)), b1: [0, 0, 0, 0],
      W2: fill(2, 4, Math.sqrt(2 / 4)), b2: [0, 0],
      g: fill(2, 2)
    };
  }

  function mlpStepwise(D) {
    const { x, W1, b1, W2, b2, g } = D;
    const rows = x.length, cin = x[0].length, hid = W1.length, cout = W2.length;
    const z1 = [], h = [], y = [];
    for (let i = 0; i < rows; i++) {
      const zr = [], hr = [], yr = [];
      for (let j = 0; j < hid; j++) {
        let s = b1[j];
        for (let k = 0; k < cin; k++) s += x[i][k] * W1[j][k];       // z1 = x @ W1ᵀ + b1
        zr.push(s); hr.push(geluTanh(s));                            // h = gelu(z1)
      }
      for (let o = 0; o < cout; o++) {
        let s = b2[o];
        for (let j = 0; j < hid; j++) s += hr[j] * W2[o][j];         // y = h @ W2ᵀ + b2
        yr.push(s);
      }
      z1.push(zr); h.push(hr); y.push(yr);
    }
    const dy = g;                                                    // 反向① dy = g
    const dW2 = [], db2 = [];                                        // 反向② fc2 存参数
    for (let o = 0; o < cout; o++) {
      const wr = []; let bs = 0;
      for (let j = 0; j < hid; j++) {
        let s = 0;
        for (let i = 0; i < rows; i++) s += dy[i][o] * h[i][j];      // dW2 = dyᵀ @ h
        wr.push(s);
      }
      for (let i = 0; i < rows; i++) bs += dy[i][o];                 // db2 = dy.sum(axis=0)
      dW2.push(wr); db2.push(bs);
    }
    const dh = [];                                                   // 反向③ fc2 的返回值
    for (let i = 0; i < rows; i++) {
      const r = [];
      for (let j = 0; j < hid; j++) {
        let s = 0;
        for (let o = 0; o < cout; o++) s += dy[i][o] * W2[o][j];     // dh = dy @ W2
        r.push(s);
      }
      dh.push(r);
    }
    const dz1 = dh.map((r, i) => r.map((v, j) => v * geluTanhDx(z1[i][j])));  // 反向④ ⊙ gelu′(z1)
    const dW1 = [], db1 = [];                                        // 反向⑤ fc1 存参数
    for (let j = 0; j < hid; j++) {
      const wr = []; let bs = 0;
      for (let k = 0; k < cin; k++) {
        let s = 0;
        for (let i = 0; i < rows; i++) s += dz1[i][j] * x[i][k];     // dW1 = dz1ᵀ @ x
        wr.push(s);
      }
      for (let i = 0; i < rows; i++) bs += dz1[i][j];                // db1 = dz1.sum(axis=0)
      dW1.push(wr); db1.push(bs);
    }
    const dx = [];                                                   // 反向⑥ fc1 的返回值
    for (let i = 0; i < rows; i++) {
      const r = [];
      for (let k = 0; k < cin; k++) {
        let s = 0;
        for (let j = 0; j < hid; j++) s += dz1[i][j] * W1[j][k];     // dx = dz1 @ W1
        r.push(s);
      }
      dx.push(r);
    }
    return { z1, h, y, dy, dW2, db2: [db2], dh, dz1, dW1, db1: [db1], dx };
  }

  /* 路径 B 的通用件（numpy 式）：矩阵一律 row-major 二维数组，向量是 1×n 矩阵。 */
  function transpose(A) { return A[0].map((_, j) => A.map(r => r[j])); }
  function matmul(A, B) {
    const m = A.length, k = B.length, n = B[0].length, out = [];
    for (let i = 0; i < m; i++) {
      const r = [];
      for (let j = 0; j < n; j++) { let s = 0; for (let q = 0; q < k; q++) s += A[i][q] * B[q][j]; r.push(s); }
      out.push(r);
    }
    return out;
  }
  function addRowBias(A, b) { return A.map(r => r.map((v, j) => v + b[j])); }
  function elmul(A, B) { return A.map((r, i) => r.map((v, j) => v * B[i][j])); }
  function colsum(A) { return [A[0].map((_, j) => A.reduce((s, r) => s + r[j], 0))]; }
  function geluMat(A) { return A.map(r => r.map(geluTanh)); }
  function geluDxMat(A) { return A.map(r => r.map(geluTanhDx)); }
  function mlpOneshot(D) {
    const { x, W1, b1, W2, b2, g } = D;
    const z1 = addRowBias(matmul(x, transpose(W1)), b1);
    const h = geluMat(z1);
    const y = addRowBias(matmul(h, transpose(W2)), b2);
    const dy = g;
    const dW2 = matmul(transpose(dy), h), db2 = colsum(dy), dh = matmul(dy, W2);
    const dz1 = elmul(dh, geluDxMat(z1));
    const dW1 = matmul(transpose(dz1), x), db1 = colsum(dz1), dx = matmul(dz1, W1);
    return { z1, h, y, dy, dW2, db2, dh, dz1, dW1, db1, dx };
  }

  function maxAbsDiffMat(A, B) {
    let m = 0;
    const walk = (u, v) => {
      if (typeof u === 'number') { m = Math.max(m, Math.abs(u - v)); return; }
      for (let i = 0; i < u.length; i++) walk(u[i], v[i]);
    };
    for (const k of ['z1', 'h', 'y', 'dW2', 'db2', 'dh', 'dz1', 'dW1', 'db1', 'dx']) walk(A[k], B[k]);
    return m;
  }
  function fro(M) {
    let s = 0;
    const walk = u => { if (typeof u === 'number') { s += u * u; return; } u.forEach(walk); };
    walk(M);
    return Math.sqrt(s);
  }

  /* 步骤表：顺序 = solutions MLP.backward（fc2 → act → fc1）逐步拆开。
     blocks = 这一步点亮的块；outs = 到这一步为止"三件产出"里已出现的。 */
  const MLP_STEPS = [
    { tag: '初始', name: '牌都摊开', io: 'x (2,3) · W1 (4,3) · b1 (4,) · W2 (2,4) · b2 (2,) · g (2,2)',
      formula: '正向将算 y = W2·GELU(W1x + b1) + b2；反向将从 dy = g 倒着走回来',
      blocks: [], outs: [] },
    { tag: '正向①', name: 'z1 = x @ W1ᵀ + b1', io: 'x (2,3) @ W1ᵀ (3,4) + b1 → z1 (2,4)',
      formula: 'z1[i,j] = Σk x[i,k]·W1[j,k] + b1[j]（Linear.forward：flat @ W.T + b）',
      blocks: ['z1'], outs: [] },
    { tag: '正向②', name: 'h = GELU(z1)', io: 'z1 (2,4) → h (2,4)，逐元素',
      formula: 'h = 0.5·z1·(1 + tanh(K(z1 + A·z1³)))，K=√(2/π)，A=0.044715',
      blocks: ['h', 'gelu'], outs: [] },
    { tag: '正向③', name: 'y = h @ W2ᵀ + b2', io: 'h (2,4) @ W2ᵀ (4,2) + b2 → y (2,2)',
      formula: 'y[i,o] = Σj h[i,j]·W2[o,j] + b2[o]',
      blocks: ['y'], outs: [] },
    { tag: '反向①', name: 'dy = g', io: '上游梯度 g (2,2) 从最右端进来',
      formula: 'dy 就是 fc2.backward(g) 收到的 g —— 反向 = 正向的镜像，从出口开始',
      blocks: ['g'], outs: [] },
    { tag: '反向②', name: 'fc2 存参数：dW2 / db2', io: 'dy (2,2), h (2,4) → dW2 (2,4), db2 (2,)',
      formula: 'dW2 = dyᵀ @ h（(2,2)ᵀ@(2,4)）；db2 = dy.sum(axis=0) —— Linear.backward 原文',
      blocks: ['dW2', 'db2'], outs: ['dW2', 'db2'] },
    { tag: '反向③', name: 'fc2 的返回值：dh = dy @ W2', io: 'dy (2,2) @ W2 (2,4) → dh (2,4)',
      formula: '每步的返回值是下一步的入参 —— g_h = self.fc2.backward(g) 这一行拿到的就是它',
      blocks: ['dh'], outs: [] },
    { tag: '反向④', name: 'act.backward：dz1 = dh ⊙ gelu′(z1)', io: 'dh (2,4) ⊙ gelu′(z1) (2,4) → dz1 (2,4)',
      formula: 'gelu′(z) = 0.5(1+t) + 0.5z(1−t²)K(1+3Az²)，t 用的是正向缓存 —— GELU 没有参数，只交返回值',
      blocks: ['dz1'], outs: [] },
    { tag: '反向⑤', name: 'fc1 存参数：dW1 / db1', io: 'dz1 (2,4), x (2,3) → dW1 (4,3), db1 (4,)',
      formula: 'dW1 = dz1ᵀ @ x（(4,2)@(2,3)）；db1 = dz1.sum(axis=0)',
      blocks: ['dW1', 'db1'], outs: ['dW1', 'db1'] },
    { tag: '反向⑥', name: 'fc1 的返回值：dx = dz1 @ W1', io: 'dz1 (2,4) @ W1 (4,3) → dx (2,3)',
      formula: 'return 一路交回上游（残差那一路就等着它）；随后与一次性直算对拍',
      blocks: ['dx'], outs: ['dx'] }
  ];
  /* 每个块从第几步开始"已算出"（数字才显示）。 */
  const MLP_REVEAL = { x: 0, W1: 0, b1: 0, W2: 0, b2: 0, g: 0, z1: 1, h: 2, y: 3,
                       dW2: 5, db2: 5, dh: 6, dz1: 7, dW1: 8, db1: 8, dx: 9 };

  /* stepAt(t)：任意一帧从头重算（纯函数，单步不等 rAF）。 */
  function mlpStepAt(D, t) {
    const A = mlpStepwise(D), B = mlpOneshot(D);
    const step = MLP_STEPS[t];
    const revealed = {}, current = {};
    for (const k in MLP_REVEAL) revealed[k] = MLP_REVEAL[k] <= t;
    for (const k of step.blocks) current[k] = true;
    const seen = {};
    for (let s = 0; s <= t; s++) for (const k of MLP_STEPS[s].outs) seen[k] = true;
    const outs = [
      { key: 'dW2', label: 'fc2.W.grad += dW2', shape: '(2,4)', mat: A.dW2, from: 5 },
      { key: 'db2', label: 'fc2.b.grad += db2', shape: '(2,)', mat: A.db2, from: 5 },
      { key: 'dW1', label: 'fc1.W.grad += dW1', shape: '(4,3)', mat: A.dW1, from: 8 },
      { key: 'db1', label: 'fc1.b.grad += db1', shape: '(4,)', mat: A.db1, from: 8 },
      { key: 'dx', label: 'return dx（给上游）', shape: '(2,3)', mat: A.dx, from: 9 }
    ].map(o => ({ ...o, done: !!seen[o.key], norm: fro(o.mat) }));
    return { t, step, A, B, revealed, current, outs, maxDev: maxAbsDiffMat(A, B) };
  }

  /* ================================================================ 1. tanhsat
     s02 §6：饱和与深层梯度消失是同一个 1−t²。上排画 tanh(x) 与导数 1−t²，
     拖 x 时两条曲线上各有一个点同步走；下排把"每层乘一次 w·(1−t²)"连乘 L 层，
     在 log₁₀ 坐标上画衰减直线，ReLU / GELU 可开关对照。
     注意与已有组件的分工：figs.js steps 讲拟合容量、figs_c.js epsnoise 讲
     有限差分 ε —— 这张图只讲深度方向的连乘。 */

  R.tanhsat = function (box) {
    box.classList.add('lane-g');
    F.title(box, 'tanh 饱和 = 每层把梯度乘小一次：`1−t²` 沿深度连乘是怎么消失的');
    F.caption(box,
      '上半张：`tanh(x)` 与它的导数 `1−t²`（就是 scalar.py TODO-4 反向里那一项，用 t 自己不用 x）。' +
      '拖动 x，两条曲线上各有一个点同步走，读数给出 t、导数、以及"这一层把上游梯度乘了多少倍"。' +
      '把 x 拖过 ±4：导数塌到 1e-3 量级、再拖到 ±6 附近是 1e-5 —— 这就是"饱和"。' +
      '下半张是这张图的重点：假设 L 层都停在同一个工作点 x，每层先乘权重尺度 w 再过激活，' +
      '回传时梯度每过一层乘一次 `w·act′(x)`，L 层连乘就是 `w^L·Π(1−t²)`（纵轴取 log₁₀，所以是直线：' +
      '每层固定掉一个数量级）。读数给出第 L 层的最终量级、以及多少层就塌穿 float32 最小正规数 1.18e-38。' +
      '打开 ReLU / GELU 对照：ReLU 正区导数恒 1（不消失，但 x<0 时整条链被一刀切成 0 —— 死区）；' +
      'GELU 是"软"的，深负区也趋 0 但不是一刀切。' +
      '这张图解释的就是 s02 判据 `MLP 最终 MSE ≤ 6e-3`（训得动）背后的机制，以及 §3.2 两分钟实验里 ' +
      '`scale2=5.0 → loss 几乎不动` 的病因：隐藏单元全进饱和区，1−t²≈0，梯度还没走到第一层就没了。' +
      '它不重复 steps 图（那讲台阶拼曲线的容量）也不重复 epsnoise 图（那讲查导数时的数值噪声）。');

    const st = { x: 0.8, L: 8, w: 1, cmp: { relu: true, gelu: true } };
    const Wpx = plotW(box, 870);

    const host1 = el('div'); box.appendChild(host1);
    const p1 = F.plot(host1, Wpx, 210, -6, 6, -1.25, 1.25);
    legend(box, [[C.blue, false, '`tanh(x)`（值）'], [C.red, true, '`1 − tanh²(x)`（导数）']]);

    const sl = F.sliders(box);
    F.slider(sl, 'x（每层的工作点）', -6, 6, 0.01, st.x, v => { st.x = v; drawCurves(); drawDecay(); });
    const ro1 = F.readout(box);

    const btnRow = el('div', 'lane-g-btns'); box.appendChild(btnRow);
    const bRelu = el('button', 'on', 'ReLU 对照');
    const bGelu = el('button', 'on', 'GELU 对照');
    bRelu.onclick = () => { st.cmp.relu = !st.cmp.relu; bRelu.classList.toggle('on', st.cmp.relu); drawDecay(); };
    bGelu.onclick = () => { st.cmp.gelu = !st.cmp.gelu; bGelu.classList.toggle('on', st.cmp.gelu); drawDecay(); };
    btnRow.append(bRelu, bGelu);

    const sl2 = F.sliders(box);
    F.slider(sl2, '层数 L', 1, 16, 1, st.L, v => { st.L = v; drawDecay(); });
    F.slider(sl2, '每层权重尺度 w', 0.5, 2, 0.01, st.w, v => { st.w = v; drawDecay(); });

    const host2 = el('div'); box.appendChild(host2);
    const p2 = F.plot(host2, Wpx, 250, 0.5, 16.5, -40, 8);
    legend(box, [
      [C.red, false, 'tanh：log₁₀ 倍数 = L·log₁₀(w(1−t²))'],
      [C.green, false, 'ReLU：正区 ×w，负区一刀切成 0'],
      [C.gold, false, 'GELU（tanh 近似式，同 solutions）'],
      [C.dim, true, 'float32 最小正规数 1.18e-38（log₁₀ ≈ −37.9）']
    ]);
    const ro2 = F.readout(box);
    const note = el('div', 'lane-g-note'); box.appendChild(note);

    function drawCurves() {
      const x = st.x, t = Math.tanh(x), d = 1 - t * t;
      p1.clear();
      p1.line(u => Math.tanh(u), C.blue, null, true);
      p1.line(u => { const s = Math.tanh(u); return 1 - s * s; }, C.red, [5, 4], true);
      p1.dots([x], u => Math.tanh(u), C.blue);
      p1.dots([x], u => { const s = Math.tanh(u); return 1 - s * s; }, C.red);
      let sat;
      if (d === 0) sat = '导数在 float64 下已经是 0';
      else if (d < 1e-2) sat = `已饱和：导数塌到 <b style="color:${C.red}">1e${Math.floor(Math.log10(d))}</b> 量级 —— 上游梯度过这一层只剩这么多`;
      else sat = '还在线性区附近，导数没塌';
      ro1.innerHTML =
        `<span>t = tanh(x) = <b>${fmt(t, 4)}</b></span>` +
        `<span>导数 1−t² = <b>${d === 0 ? '0' : d.toExponential(2)}</b></span>` +
        `<span>${sat}</span>` +
        `<span>这一层把上游梯度乘了 w·(1−t²) = <b>${fmt(st.w * d, 4)}</b> 倍（w=${st.w}）</span>`;
    }

    function drawDecay() {
      const { x, L, w } = st;
      const acts = [['tanh', C.red, true]];
      if (st.cmp.relu) acts.push(['relu', C.green, false]);
      if (st.cmp.gelu) acts.push(['gelu', C.gold, false]);
      const chains = acts.map(([a, col]) => [a, col, deepChain(a, x, w, L)]);
      p2.clear();
      p2.line(() => LOG10_F32MIN, C.dim, [4, 4]);
      for (const [, col, ch] of chains) {
        if (ch.dead) continue;
        p2.line(l => (l <= L ? l * ch.lm : NaN), col, null, true);
        p2.dots([L], () => L * ch.lm, col);
      }
      const tn = chains.find(c => c[0] === 'tanh')[2];
      let s = `<span>层数 L = <b>${L}</b>，w = <b>${w}</b>，工作点 x = <b>${fmt(x, 2)}</b></span>`;
      if (tn.dead) {
        s += `<span style="color:${C.red}">tanh 链：1−t² = 0，第 1 层起梯度恒 0</span>`;
      } else if (tn.lm === 0) {
        s += `<span>tanh 链：每层倍数恰好 ×1，不缩不涨</span>`;
      } else {
        const vL = L * tn.lm;
        s += `<span>每层倍数 w·(1−t²) = <b>${tn.m.toExponential(2)}</b>（log₁₀ = ${fmt(tn.lm, 2)}）</span>` +
             `<span>tanh 链第 ${L} 层：<b style="color:${tn.lm < 0 ? C.red : C.green}">${Math.pow(10, vL).toExponential(1)}</b>` +
             `（10^${fmt(vL, 1)}）</span>`;
        if (tn.n32 != null) {
          s += `<span>第 <b>${tn.n32}</b> 层塌穿 float32 下限 1.18e-38` +
               `${tn.n32 > L ? '（超过当前 L，往下加层就能看到穿过虚线）' : '（虚线以下）'}</span>`;
        } else {
          s += `<span style="color:${C.gold}">倍数 ≥ 1：不消失 —— 这是另一头的病（爆炸），残差/归一化两头都治</span>`;
        }
      }
      const rn = chains.find(c => c[0] === 'relu');
      if (rn) {
        s += rn[2].dead
          ? `<span style="color:${C.green}">ReLU：x<0 → 导数恒 0，梯度被一刀切（死区），不是慢慢消失</span>`
          : `<span style="color:${C.green}">ReLU：正区导数恒 1，链 = w^L，不消失；代价是 x<0 就死</span>`;
      }
      const gn = chains.find(c => c[0] === 'gelu');
      if (gn) {
        s += gn[2].dead
          ? `<span style="color:${C.gold}">GELU：深负区导数在 float64 下也塌成 0（饱和比 tanh 晚、但同样存在）</span>`
          : `<span style="color:${C.gold}">GELU：每层倍数 ${gn[2].m.toExponential(2)}，第 ${L} 层 ${Math.pow(10, L * gn[2].lm).toExponential(1)}</span>`;
      }
      ro2.innerHTML = s;
      note.textContent = '连乘在 log₁₀ 域累加（直接乘会在 16 层前就下溢出成 0，画不出曲线）；' +
        '纵轴每格 12 个数量级，横轴是层号 l = 1…L。';
    }

    drawCurves();
    drawDecay();
  };

  /* ================================================================ 2. geludx
     s03b §3：GELU 的值与导数。四条曲线：精确式 x·Φ(x)（erf 自己实现）与
     tanh 近似式（solutions/minigpt/nn.py 真实用的那条）各一条值、一条导数。
     三种错法开关：① ReLU 代 GELU ② 导数漏 x·φ(x) 项 ③ 近似值配精确导数。 */

  R.geludx = function (box) {
    box.classList.add('lane-g');
    F.title(box, 'GELU 的两张脸：精确式 `x·Φ(x)` 与代码里真实用的 tanh 近似式，值和导数一起看');
    F.caption(box,
      'solutions/minigpt/nn.py 的 GELU 用的是 **tanh 近似式** `y = 0.5x(1+tanh(K(x+Ax³)))`，' +
      'K=√(2/π)≈0.7978846、A=0.044715；精确式是 `y = x·Φ(x)`（Φ 是标准正态分布函数，' +
      '这里自己写了 erf：|z|<3 用 Maclaurin 级数、|z|≥3 用连分数，读数里当场给出两条路径的互检差；' +
      '离线与 Python `math.erf` 对账最大偏差 5.1e-15，见本批自检）。' +
      '把 x 拖到 −2 能复现讲义 §3.2 的四个数：近似 −0.04540 / 导数 −0.0861；' +
      '精确式那一列（−0.04550 / −0.0852）在 §11.1 的对照表里。' +
      '导数曲线用的是 backward 原文 `0.5(1+t) + 0.5x(1−t²)K(1+3Ax²)`；中心差分（eps 可拖）现场验一遍。' +
      '负区那条小凹陷是 GELU 和 ReLU 的本质区别：值本身最低探到 ≈ −0.17（x≈−0.75 附近），' +
      '斜率最低 ≈ −0.13（x≈−1.42，精确式恰好在 −√2），负区仍然放一点点信息过去' +
      '（读数给出真扫出来的极小点位置）。' +
      '三个错法开关对应 §3.3：漏链式项 → gradcheck 相对误差 1e0 量级当场被抓；' +
      '混用（近似值配精确导数）→ x=−2 处 1e-2、全域最大 3e-1，都过不了 1e-4 阈值；' +
      'ReLU 代 GELU → 值本身最多差 ~0.17。');

    const st = { x: -2, epsL: -6, wrong: 0 };
    const Wpx = plotW(box, 870);

    const host = el('div'); box.appendChild(host);
    const p = F.plot(host, Wpx, 260, -4, 4, -0.7, 4.3);
    legend(box, [
      [C.blue, false, '精确值 `x·Φ(x)`'], [C.blue, true, '精确导数 `Φ(x) + x·φ(x)`'],
      [C.green, false, '近似值（代码用的）'], [C.green, true, '近似导数（backward 原文）']
    ]);

    const sl = F.sliders(box);
    F.slider(sl, 'x', -4, 4, 0.01, st.x, v => { st.x = v; draw(); });
    F.slider(sl, 'log₁₀ eps（中心差分）', -9, -1, 0.5, st.epsL, v => { st.epsL = v; draw(); });

    const btnRow = el('div', 'lane-g-btns'); box.appendChild(btnRow);
    const wrongDefs = ['不开错法', '① 用 ReLU 代替 GELU', '② 导数只写 Φ(x)（漏 x·φ(x) 项）', '③ 近似值却按精确式求导'];
    const wrongBtns = wrongDefs.map((txt, k) => {
      const b = el('button', k === 0 ? 'on' : '', txt);
      b.onclick = () => { st.wrong = k; wrongBtns.forEach((bb, j) => bb.classList.toggle('on', j === k)); draw(); };
      btnRow.appendChild(b); return b;
    });

    const ro = F.readout(box);
    const ro2 = F.readout(box);
    const note = el('div', 'lane-g-note'); box.appendChild(note);

    /* 扫描网格：极值/最大差都当场扫，不写死。 */
    const GX = []; for (let i = 0; i <= 800; i++) GX.push(-4 + i * 8 / 800);
    const GXS = []; for (let i = 0; i <= 8000; i++) GXS.push(-4 + i * 4 / 8000);   // 负区细扫
    const centralD = (f, x, e) => (f(x + e) - f(x - e)) / (2 * e);
    /* gradcheck.py compare() 同款相对误差：|num−ana| / max(1e-8,|num|,|ana|) */
    const gcrel = (num, ana) => Math.abs(num - ana) / Math.max(1e-8, Math.abs(num), Math.abs(ana));

    /* 负区导数极小点（细扫），一次算好缓存 —— 只依赖公式本身，不依赖滑杆。 */
    function dMin(f) {
      let bx = GXS[0], bv = f(bx);
      for (const x of GXS) { const v = f(x); if (v < bv) { bv = v; bx = x; } }
      return { x: bx, v: bv };
    }
    const minA = dMin(geluTanhDx), minE = dMin(geluExactDx);

    function draw() {
      const x = st.x, eps = Math.pow(10, st.epsL);
      const ve = geluExact(x), va = geluTanh(x);
      const de = geluExactDx(x), da = geluTanhDx(x);
      const numA = centralD(geluTanh, x, eps);
      p.clear();
      p.line(geluExact, C.blue, null, true);
      p.line(u => geluExactDx(u), C.blue, [5, 4]);
      p.line(geluTanh, C.green, null, true);
      p.line(u => geluTanhDx(u), C.green, [5, 4]);
      /* 负斜率区：把 gelu′<0 的那段在 y=0 轴上点红，再标出极小点 */
      const negXs = GX.filter(u => geluTanhDx(u) < 0);
      p.dots(negXs, () => 0, 'rgba(255,107,107,0.55)');
      p.dots([minA.x], () => minA.v, C.red);
      if (st.wrong === 1) p.line(u => Math.max(0, u), C.purple, [3, 3], true);
      if (st.wrong === 2) p.line(u => normCdf(u), C.red, [3, 3], true);
      p.dots([x], geluExact, C.blue);
      p.dots([x], geluTanh, C.green);
      p.dots([x], geluExactDx, C.blue);
      p.dots([x], geluTanhDx, C.green);

      /* erf 互检：级数 vs 连分数，两条独立路径当场对（z≥1 才有连分数的用武之地，
         x 太小时固定在对检点 z=1 上做，说明写在 note 里）。 */
      const z = Math.abs(x) / SQRT2 >= 1 ? Math.abs(x) / SQRT2 : 1;
      const erfDiff = Math.abs(erfSeries(z) - (1 - erfcCF(z)));

      const valMaxDiff = GX.reduce((m, u) => Math.max(m, Math.abs(geluExact(u) - geluTanh(u))), 0);
      ro.innerHTML =
        `<span>x = <b>${fmt(x, 2)}</b></span>` +
        `<span>精确值 x·Φ(x) = <b>${fmt(ve, 5)}</b></span>` +
        `<span>近似值（tanh 式） = <b>${fmt(va, 5)}</b></span>` +
        `<span>两者之差 = <b>${Math.abs(ve - va).toExponential(1)}</b>（全域最大 ${valMaxDiff.toExponential(1)}，讲义 §11.1 表里 y 那一行就是它）</span>` +
        `<span>精确导数 Φ+xφ = <b>${fmt(de, 5)}</b></span>` +
        `<span>近似导数（backward 原文） = <b>${fmt(da, 5)}</b></span>` +
        `<span>中心差分(eps=${eps.toExponential(0)}) 验近似导数 = <b>${fmt(numA, 5)}</b>，` +
        `gradcheck 相对误差 = <b style="color:${gcrel(numA, da) < 1e-4 ? C.green : C.red}">${gcrel(numA, da).toExponential(1)}</b></span>` +
        `<span>Φ 两条算法互检 |级数−连分数|(z=${fmt(z, 2)}) = <b>${erfDiff.toExponential(1)}</b></span>`;

      let s2 = `<span>负斜率区（y=0 轴上的红点段）：近似式导数极小点在 x ≈ <b>${fmt(minA.x, 3)}</b>，` +
        `值 <b style="color:${C.red}">${fmt(minA.v, 4)}</b>；精确式在 x ≈ ${fmt(minE.x, 3)}（−√2），值 ${fmt(minE.v, 4)}。` +
        `GELU 不是 ReLU：负区仍然放一点点信息过去</span>`;
      if (st.wrong === 1) {
        const md1 = GX.reduce((m, u) => Math.max(m, Math.abs(Math.max(0, u) - geluTanh(u))), 0);
        s2 += `<span style="color:${C.purple}">① ReLU 代 GELU：当前 x 处差 ${Math.abs(Math.max(0, x) - va).toExponential(1)}，` +
          `全域最大绝对差 <b>${fmt(md1, 3)}</b>（在值极小点附近）；而且 x<0 时 ReLU 导数恒 0 —— 负区信息直接掐死</span>`;
      } else if (st.wrong === 2) {
        const numE = centralD(geluExact, x, eps);
        const relNow = gcrel(numE, normCdf(x));
        const relMax = GX.reduce((m, u) => Math.max(m, gcrel(centralD(geluExact, u, 1e-6), normCdf(u))), 0);
        s2 += `<span style="color:${C.red}">② 导数只写 Φ(x)（漏掉 x·φ(x) 那一项，§3.3 的原话错误）：` +
          `当前 x 处 gradcheck 相对误差 = <b>${relNow.toExponential(1)}</b>，全域最大 <b>${relMax.toExponential(1)}</b> —— ` +
          `1e0 量级，阈值 1e-4，立刻被抓</span>`;
      } else if (st.wrong === 3) {
        const relNow = gcrel(numA, geluExactDx(x));
        const relMax = GX.reduce((m, u) => Math.max(m, gcrel(centralD(geluTanh, u, 1e-6), geluExactDx(u))), 0);
        s2 += `<span style="color:${C.gold}">③ 正向用近似式、backward 却按精确式 Φ+xφ 求导（混用）：` +
          `当前 x 处相对误差 <b>${relNow.toExponential(1)}</b>，全域最大 <b>${relMax.toExponential(1)}</b>` +
          `（深负端两个导数都趋 0，分母小、相对误差反而更大）—— 都远超 1e-4 阈值，照样红</span>`;
      }
      ro2.innerHTML = s2;
      note.textContent = '错法只是对照演示，唯一正确的写法是：正向近似式 + backward 用它的导数' +
        '（0.5(1+t) + 0.5x(1−t²)K(1+3Ax²)，t 用正向缓存）。erf 互检在 |x|/√2 < 1 时固定取 z=1：' +
        'z 太小时连分数收敛慢，级数本来就是那个区间的主算法。';
    }

    draw();
  };

  /* ================================================================ 3. mlpbwd
     s03b §6：MLP 正向三块 + 反向六步的动线。小尺寸真算：
     x (2,3) → Linear1 (3→4) → GELU → Linear2 (4→2)，上游 g (2,2)，seed=42。
     反向顺序严格照 solutions MLP.backward：fc2（dW2/db2 → return dh）→ act（dz1）
     → fc1（dW1/db1 → return dx）。最后一帧把逐步路径与一次性 numpy 式直算对拍。 */

  const MB_CW = 30, MB_CH = 17;                 // 单元格尺寸
  function fmtCell(v) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 5e-3) return v.toExponential(0);
    return v.toFixed(2);
  }

  R.mlpbwd = function (box) {
    box.classList.add('lane-g');
    F.title(box, 'MLP 的反向动线：正向三块，反向按 `fc2 → act → fc1` 一步步倒回来（每格都是真数）');
    F.caption(box,
      '一个巴掌大的 MLP 当场真算：`x (2,3)` → `Linear1 (3→4)` → `GELU` → `Linear2 (4→2)`，' +
      '上游梯度 `g (2,2)`。权重用 seed=42 生成（mulberry32 + Box-Muller，fan-in 缩放 √(2/in_f)，' +
      'b 与 solutions 的 Linear 一样初始化为 0 —— 但 db 不是 0：偏置梯度 = 各行梯度求和）。' +
      '点 ⏭ 单步或 ▶ 播放：正向三步（z1 → h → y），反向严格照 solutions `MLP.backward` 的顺序 ' +
      '`fc2.backward`（先存 dW2/db2，再 return dh）→ `act.backward`（dz1 = dh ⊙ gelu′(z1)）→ ' +
      '`fc1.backward`（存 dW1/db1，return dx）。每步点亮的块就是那一步的产出，数字逐步浮现；' +
      '下侧"三件产出"清单随步骤累积（MLP 自己一行 `.grad` 都不写，参数梯度由子模块累加 —— §6.2）。' +
      '最后一步把逐步路径（每步显式 Σ 循环，相当于手算）与另一条一次性 numpy 式直算路径对拍，' +
      '最大偏差应为 0 或 1e-16 量级 —— 这就是"动线走对了"的现场证据。' +
      '顺序写反/漏步的后果见 §6.3：漏掉 act.backward 那种形状全对的错，gradcheck 实测相对误差 0.82；' +
      '答案版是 1.72e-07（s03b 判据第 3 条，阈值 1e-4）。任意一帧由纯函数 stepAt(t) 从头重算，单步不等动画。');

    const D = makeMLPData(42);
    const Wpx = 880, Hpx = 478;
    const cvh = el('div'); box.appendChild(cvh);
    const cv = el('canvas');
    cv.width = Wpx * 2; cv.height = Hpx * 2;
    cv.style.width = Wpx + 'px'; cv.style.height = Hpx + 'px';
    cvh.appendChild(cv);
    const g = cv.getContext('2d');

    const headLine = F.head(box, '');
    const chipRow = el('div', 'lane-g-chips'); box.appendChild(chipRow);
    const outs = el('div', 'lane-g-outs'); box.appendChild(outs);
    const verdict = el('div', 'lane-g-note'); box.appendChild(verdict);

    /* ---- 块的版面（画布坐标）----
       正向：权重排在上、数据排在下，W 竖直箭头汇到水平数据流上；
       反向镜像：数据梯度从右往左流，dW/db 分支向下。 */
    const KIND = { data: ['#16283c', '#cfe3ff'], w: ['#251f36', '#d9cdf2'], grad: ['#2e2118', '#f0d9b5'] };
    const BLOCKS = {
      // 正向 · 权重排（y=26）
      W1: { x: 170, y: 26, title: 'W1 (4,3)', kind: 'w', get: S => D.W1 },
      b1: { x: 274, y: 40, title: 'b1 (4,)', kind: 'w', get: S => [D.b1] },
      W2: { x: 560, y: 26, title: 'W2 (2,4)', kind: 'w', get: S => D.W2 },
      b2: { x: 694, y: 40, title: 'b2 (2,)', kind: 'w', get: S => [D.b2] },
      // 正向 · 数据排（y=112）
      x:  { x: 30,  y: 112, title: 'x (2,3)', kind: 'data', get: S => D.x },
      z1: { x: 282, y: 112, title: 'z1 (2,4)', kind: 'data', get: S => S.A.z1 },
      h:  { x: 438, y: 112, title: 'h (2,4)', kind: 'data', get: S => S.A.h },
      y:  { x: 750, y: 112, title: 'y (2,2)', kind: 'data', get: S => S.A.y },
      // 反向 · 数据梯度排（y=300），从右往左
      g:   { x: 750, y: 300, title: 'dy = g (2,2)', kind: 'grad', get: S => S.A.dy },
      dh:  { x: 438, y: 300, title: 'dh (2,4)', kind: 'grad', get: S => S.A.dh },
      dz1: { x: 282, y: 300, title: 'dz1 (2,4)', kind: 'grad', get: S => S.A.dz1 },
      dx:  { x: 30,  y: 300, title: 'dx (2,3)', kind: 'grad', get: S => S.A.dx },
      // 反向 · 参数梯度排（y=388），从对应数据梯度竖直分支下来
      dW2: { x: 560, y: 388, title: 'dW2 (2,4)', kind: 'grad', get: S => S.A.dW2 },
      db2: { x: 694, y: 402, title: 'db2 (2,)', kind: 'grad', get: S => S.A.db2 },
      dW1: { x: 170, y: 388, title: 'dW1 (4,3)', kind: 'grad', get: S => S.A.dW1 },
      db1: { x: 274, y: 402, title: 'db1 (4,)', kind: 'grad', get: S => S.A.db1 }
    };

    function matW(M) { return M[0].length * MB_CW; }
    function matH(M) { return M.length * MB_CH; }

    function drawBlock(g2, key, S) {
      const B = BLOCKS[key], M = B.get(S);
      const [bg, fg] = KIND[B.kind];
      const rev = S.revealed[key], cur = !!S.current[key];
      const w = matW(M), h = matH(M);
      g2.save();
      g2.globalAlpha = rev ? 1 : 0.22;
      g2.font = '10px ui-monospace,Consolas';
      g2.fillStyle = cur ? C.blue : C.dim;
      g2.textAlign = 'left'; g2.textBaseline = 'alphabetic';
      g2.fillText(B.title, B.x, B.y - 5);
      for (let i = 0; i < M.length; i++) for (let j = 0; j < M[0].length; j++) {
        const px = B.x + j * MB_CW, py = B.y + i * MB_CH;
        g2.fillStyle = bg; g2.fillRect(px, py, MB_CW - 1, MB_CH - 1);
        if (rev) {
          g2.fillStyle = fg;
          g2.font = '9px ui-monospace,Consolas';
          g2.textAlign = 'center'; g2.textBaseline = 'middle';
          g2.fillText(fmtCell(M[i][j]), px + (MB_CW - 1) / 2, py + MB_CH / 2);
        }
      }
      g2.strokeStyle = cur ? C.blue : '#3a4657';
      g2.lineWidth = cur ? 2 : 1;
      g2.strokeRect(B.x - 1, B.y - 1, w + 1, h + 1);
      g2.restore();
      return { w, h };
    }

    function arrow(g2, x1, y1, x2, y2, color, dash) {
      g2.save();
      g2.strokeStyle = color; g2.fillStyle = color; g2.lineWidth = 1.4;
      if (dash) g2.setLineDash(dash);
      g2.beginPath(); g2.moveTo(x1, y1); g2.lineTo(x2, y2); g2.stroke();
      const a = Math.atan2(y2 - y1, x2 - x1), L = 6;
      g2.setLineDash([]);
      g2.beginPath();
      g2.moveTo(x2, y2);
      g2.lineTo(x2 - L * Math.cos(a - 0.4), y2 - L * Math.sin(a - 0.4));
      g2.lineTo(x2 - L * Math.cos(a + 0.4), y2 - L * Math.sin(a + 0.4));
      g2.closePath(); g2.fill();
      g2.restore();
    }
    function opLabel(g2, x, y, text, on) {
      g2.save();
      g2.font = '10px ui-monospace,Consolas';
      g2.fillStyle = on ? C.fg : 'rgba(141,153,172,0.4)';
      g2.textAlign = 'center'; g2.textBaseline = 'alphabetic';
      g2.fillText(text, x, y);
      g2.restore();
    }
    /* 折线箭头：参数梯度的分支要绕过别的块，一段直线画不了 */
    function arrowPath(g2, pts, color) {
      g2.save();
      g2.strokeStyle = color; g2.fillStyle = color; g2.lineWidth = 1.4;
      g2.beginPath(); g2.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length - 1; i++) g2.lineTo(pts[i][0], pts[i][1]);
      const n = pts.length, x1 = pts[n - 2][0], y1 = pts[n - 2][1], x2 = pts[n - 1][0], y2 = pts[n - 1][1];
      g2.lineTo(x2, y2); g2.stroke();
      const a = Math.atan2(y2 - y1, x2 - x1), L = 6;
      g2.beginPath();
      g2.moveTo(x2, y2);
      g2.lineTo(x2 - L * Math.cos(a - 0.4), y2 - L * Math.sin(a - 0.4));
      g2.lineTo(x2 - L * Math.cos(a + 0.4), y2 - L * Math.sin(a + 0.4));
      g2.closePath(); g2.fill();
      g2.restore();
    }

    function draw(t) {
      const S = mlpStepAt(D, t);
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.scale(2, 2);

      g.font = '700 11px ui-monospace,Consolas'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.fillStyle = C.dim;
      g.fillText('正向 →', 12, 20);
      g.fillText('反向 ←', 12, 292);

      /* 箭头与公式标签：先画线再画块，块盖在线上 */
      const on = k => S.revealed[k];
      const ac = k => S.current[k] ? C.blue : '#4a5568';
      // 正向数据流（y=129 中线）；公式标签放在数据排下方 y=162，避开块与竖直箭头
      arrow(g, 120, 129, 278, 129, ac('z1'));
      opLabel(g, 199, 162, 'z1 = x @ W1ᵀ + b1', on('z1'));
      arrow(g, 402, 129, 434, 129, ac('h'));
      opLabel(g, 418, 104, 'GELU', on('h'));
      arrow(g, 558, 129, 746, 129, ac('y'));
      opLabel(g, 652, 162, 'y = h @ W2ᵀ + b2', on('y'));
      // 权重竖直汇入
      arrow(g, 215, 94, 215, 125, on('z1') ? '#4a5568' : 'rgba(74,85,104,0.35)');
      arrow(g, 620, 60, 620, 125, on('y') ? '#4a5568' : 'rgba(74,85,104,0.35)');
      // y → g（dy = g，t=4）
      arrow(g, 780, 146, 780, 296, ac('g'), [4, 4]);
      opLabel(g, 688, 230, '反向从出口进：dy = g', on('g') && t >= 4);
      // 反向数据流（从右往左，中线 y=317）；⊙ 那步的标签抬到标题行之上避免挤压
      arrow(g, 746, 317, 562, 317, ac('dh'));
      opLabel(g, 654, 310, 'dh = dy @ W2', on('dh'));
      arrow(g, 434, 317, 406, 317, ac('dz1'));
      opLabel(g, 420, 272, 'dz1 = dh ⊙ gelu′(z1)', on('dz1'));
      arrow(g, 278, 317, 124, 317, ac('dx'));
      opLabel(g, 201, 310, 'dx = dz1 @ W1', on('dx'));
      // 参数梯度分支：折线从"谁乘谁"的源头绕到 dW 块顶上
      arrowPath(g, [[780, 334], [780, 372], [620, 372], [620, 384]], ac('dW2'));
      opLabel(g, 640, 364, 'dW2 = dyᵀ @ h · db2 = dy.sum(0)', on('dW2'));
      arrowPath(g, [[342, 334], [342, 372], [215, 372], [215, 384]], ac('dW1'));
      opLabel(g, 250, 356, 'dW1 = dz1ᵀ @ x · db1 = dz1.sum(0)', on('dW1'));
      // 缓存连线：反向用到了正向缓存的 z1 与 h（§6.4：缓存归子模块自己）
      g.save(); g.strokeStyle = 'rgba(141,153,172,0.4)'; g.setLineDash([2, 4]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(342, 146); g.lineTo(342, 296); g.stroke();
      g.beginPath(); g.moveTo(498, 146); g.lineTo(498, 296); g.stroke();
      g.restore();
      g.save(); g.font = '9px ui-monospace,Consolas'; g.fillStyle = 'rgba(141,153,172,0.75)';
      g.textAlign = 'left';
      g.fillText('z1 缓存（gelu′要用）', 348, 224);
      g.fillText('h 缓存（dW2 要用）', 504, 240);
      g.restore();

      for (const key in BLOCKS) drawBlock(g, key, S);

      /* 步骤头 + 三件产出 + 对拍 */
      headLine.innerHTML = F.md(
        `第 ${t}/${MLP_STEPS.length - 1} 步 · ${S.step.tag} ${S.step.name}` +
        ` · 形状：${S.step.io}`);
      outs.innerHTML =
        `<b style="color:#cfe3ff">三件产出清单（随步骤累积；MLP 自己不写 .grad，由子模块累加）</b><br>` +
        S.outs.map(o =>
          `<span class="${o.done ? '' : 'pending'}">${o.done ? '✓' : '·'} ${o.label} ` +
          `${o.shape}${o.done ? ` ‖·‖F = <b>${fmt(o.norm, 4)}</b>` : '（还没走到）'}</span>`
        ).join('<br>') +
        `<br><span class="${t >= 9 ? '' : 'pending'}">对拍：逐步 Σ 循环 vs 一次性 numpy 式直算，` +
        `全部 10 个张量的最大绝对偏差 = <b style="color:${C.green}">${S.maxDev === 0 ? '0' : S.maxDev.toExponential(1)}</b></span>`;
      verdict.textContent = t >= 9
        ? '动线走对了：两条独立实现逐位吻合（0 或 1e-16 量级 = float64 求和顺序差）。这就是 §6.3 说的顺序不能反、步骤不能漏的现场验证版。'
        : '走到最后一步（反向⑥）会做两条独立路径的对拍。';
      chips.forEach((b, k) => b.classList.toggle('on', k === t));
    }

    const chips = MLP_STEPS.map((s, k) => {
      const b = el('button', '', `${k} ${s.tag}`);
      b.onclick = () => { pb.stop(); pb.t = k; };
      chipRow.appendChild(b); return b;
    });
    const pb = playback(box, () => MLP_STEPS.length - 1, draw, 800);
    draw(0);
  };

  /* 数值自测钩子：node 里 stub 掉 window/document 后可以直接调这些纯函数，
     与独立 python 脚本（math.erf / numpy）按同一公式复算对账。不注册成图。 */
  R.__g_math = { mulberry32, gauss, erf, erfSeries, erfcCF, normCdf, normPdf,
                 geluExact, geluExactDx, geluTanh, geluTanhDx, GK, GA,
                 actPair, deepChain, F32_MIN, LOG10_F32MIN,
                 makeMLPData, mlpStepwise, mlpOneshot, mlpStepAt, maxAbsDiffMat, fro,
                 transpose, matmul, MLP_STEPS };
})();
