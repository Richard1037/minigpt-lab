/* 原理动画 · 第 D 批：3D 原理可视化（形状盒 / loss 曲面）。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和前几批一样：能算的都真的算 —— sum(axis=k)、@ 收缩、41×41 loss 网格、
   优化器递推全部在 JS 里当场跑一遍，readout 里的数字可以直接拿去和 numpy 对。

   本批两个组件：shapebox / losssurf。
   共同约定：
     · 不引任何外部 3D 库：canvas 2D 手写旋转投影 + 画家算法（按视深排序逐面画）。
     · 每个组件都有 ▶ 播放 / ⏭ 单步 / ⏸ 暂停 / ↺ 重置；任意一帧的状态由纯函数
       stepAt(t) 从头重算（单步不等 rAF，无头浏览器能截到确定的第 t 帧）；
       rAF 只在播放期间存在，mount 时不起任何定时器。
     · 拖动旋转的 pointerdown/move/up 只挂在 canvas 元素上，不碰 window/document。
     · losssurf 的三条递推与 solutions/minigpt/optim.py 逐行对应：
       Momentum 是 SGD 类（buf = μ·buf + g，μ=0.9）；AdamW 是 β₁=0.9、β₂=0.95、
       eps=1e-8、wd=0.1，次序为"先解耦衰减，再用 backward 存好的梯度更新矩"，
       m̂=m/(1−β₁ᵗ)、w −= lr·m̂/(√v̂+eps)。
     · 自带样式全部以 `.lane-d ` 前缀注入，不碰 manual.css。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  const el = F.el, fmt = F.fmt;

  /* ---------------------------------------------------------------- 公共工具 */

  (function injectStyle() {
    if (typeof document === 'undefined' || document.getElementById('lane-d-style')) return;
    const s = el('style'); s.id = 'lane-d-style';
    s.textContent = [
      '.lane-d canvas{display:block;max-width:100%;margin:6px 0 0;touch-action:none;cursor:grab;',
      '  border:1px solid var(--line);border-radius:8px;background:#101722}',
      '.lane-d canvas:active{cursor:grabbing}',
      '.lane-d .lane-d-btns{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 2px;align-items:center}',
      '.lane-d .lane-d-note{font:11.5px/1.5 var(--mono);color:var(--dim);margin-top:4px}',
      '.lane-d select{background:#151d2b;color:var(--fg);border:1px solid var(--line);',
      '  border-radius:7px;padding:5px 8px;font:12px/1 var(--mono);cursor:pointer}',
      '.lane-d .lane-d-side{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}'
    ].join('\n');
    (document.head || document.documentElement).appendChild(s);
  })();

  const PAL = ['#4aa3ff', '#3ecf8e', '#f5b445', '#ff6b6b', '#b9a7e0'];
  const OKC = '#3ecf8e', BADC = '#ff6b6b', DIMC = '#8d99ac', FGC = '#e6ebf3';

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  /* '#rrggbb' 或 'rgb(r,g,b)' → [r,g,b]（lerpColor 的产物还要能喂回 shade） */
  function parseColor(c) {
    if (c[0] === '#') return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
    const m = /(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.exec(c);
    return m ? [+m[1], +m[2], +m[3]] : [255, 255, 255];
  }
  /* 颜色 × 亮度系数 → 'rgb(...)'；系数只用于面 shading。 */
  function shade(col, f) {
    const [r, g, b] = parseColor(col);
    return `rgb(${Math.round(clamp(r * f, 0, 255))},${Math.round(clamp(g * f, 0, 255))},${Math.round(clamp(b * f, 0, 255))})`;
  }
  function lerpColor(a, b, t) {
    const A = parseColor(a), B = parseColor(b);
    return `rgb(${Math.round(A[0] + (B[0] - A[0]) * t)},${Math.round(A[1] + (B[1] - A[1]) * t)},${Math.round(A[2] + (B[2] - A[2]) * t)})`;
  }

  /* ---------------- 手写 3D：yaw 绕 z 轴、elev 抬角，正交投影 + 画家算法 ----------------
     世界坐标：X 向右、Y 向画面深处、Z 向上。
     proj 返回 {x, y, d}：x/y 是旋转后的画面坐标（y 向上为正，落屏时取负），
     d 是沿视线方向的深度 —— d 越大离相机越远，画家算法按 d 从大到小画。 */
  function proj(cam, P) {
    const ca = Math.cos(cam.yaw), sa = Math.sin(cam.yaw);
    const x1 = P[0] * ca - P[1] * sa;
    const y1 = P[0] * sa + P[1] * ca;
    const z1 = P[2];
    const ce = Math.cos(cam.elev), se = Math.sin(cam.elev);
    return { x: x1, y: y1 * se + z1 * ce, d: y1 * ce - z1 * se };
  }

  /* 画一个轴对齐的小立方块（可以压扁：half 三个分量独立）。
     center/half 是世界坐标，S 是缩放（px/世界单位），cx/cy 是画面中心。
     只画法线朝向相机的面；三个可见面用不同亮度做明暗；label 画在最靠前的那个面上。 */
  const CUBE_FACES = [                       // [角点索引×4（CCW）, 法线, 亮度]
    [[4, 5, 7, 6], [0, 0, 1], 1.00],         // 顶面 z+
    [[0, 4, 6, 2], [-1, 0, 0], 0.62],        // 左面 x−
    [[1, 5, 7, 3], [1, 0, 0], 0.62],         // 右面 x+
    [[0, 1, 5, 4], [0, -1, 0], 0.80],        // 前面 y−
    [[2, 3, 7, 6], [0, 1, 0], 0.80],         // 后面 y+
    [[0, 1, 3, 2], [0, 0, -1], 0.45]         // 底面 z−
  ];
  /* face：'top'（默认）把数字写顶面——同层平铺时顶面不会被邻居挡（前面那格会盖住
     后一格的正面，label 就整片消失了，ids 行就是这么丢的）；'front' 给竖直堆叠用
     （顶面被上一层压住，只有正面看得见）；false = 面积最大的面。 */
  function drawCube(g, cam, S, cx, cy, center, half, color, label, alpha, face) {
    const cs = [];
    for (let k = 0; k < 8; k++) {
      cs.push(proj(cam, [
        center[0] + (k & 1 ? half[0] : -half[0]),
        center[1] + (k & 2 ? half[1] : -half[1]),
        center[2] + (k & 4 ? half[2] : -half[2])
      ]));
    }
    const ca = Math.cos(cam.yaw), sa = Math.sin(cam.yaw);
    const ce = Math.cos(cam.elev), se = Math.sin(cam.elev);
    let best = null, bestArea = 0, bestLum = 1;
    let top = null, topArea = 0, front = null, frontArea = 0;
    g.save();
    if (alpha != null && alpha < 1) g.globalAlpha = alpha;
    for (const [idx, n, lum] of CUBE_FACES) {
      const nx = n[0] * ca - n[1] * sa, ny = n[0] * sa + n[1] * ca, nz = n[2];
      if (ny * ce - nz * se >= 0) continue;    // 背向相机（n·f ≥ 0）
      let mx = 0, my = 0;
      const px4 = [], py4 = [];
      g.beginPath();
      for (let q = 0; q < 4; q++) {
        const p = cs[idx[q]];
        const px = cx + S * p.x, py = cy - S * p.y;
        px4.push(px); py4.push(py);
        q ? g.lineTo(px, py) : g.moveTo(px, py);
        mx += px; my += py;
      }
      g.closePath();
      g.fillStyle = shade(color, lum);
      g.fill();
      g.strokeStyle = 'rgba(8,12,18,0.55)'; g.lineWidth = 0.6; g.stroke();
      if (label != null) {
        const area = Math.abs((px4[1] - px4[0]) * (py4[2] - py4[0]) - (px4[2] - px4[0]) * (py4[1] - py4[0])) / 2;
        if (area > bestArea) { bestArea = area; best = [mx / 4, my / 4]; bestLum = lum; }
        if (n[2] === 1) { top = [mx / 4, my / 4]; topArea = area; }
        if (n[1] === -1) { front = [mx / 4, my / 4]; frontArea = area; }
      }
    }
    const use = face === 'front' ? { at: front, area: frontArea, lum: 0.80 }
      : face !== false && top && topArea >= 70 ? { at: top, area: topArea, lum: 1 }
      : { at: best, area: bestArea, lum: bestLum };
    if (use.at && use.area >= 70) {
      const fs = clamp(Math.sqrt(use.area) * 0.42, 7, 13);
      g.font = `700 ${fs.toFixed(0)}px ui-monospace,Consolas`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillStyle = use.lum >= 0.78 ? '#0d131d' : '#eef3fa';
      g.fillText(label, use.at[0], use.at[1]);
    }
    g.restore();
  }

  /* 播放控制（同 figs_b 的约定，自己一份）：▶ / ⏭ / ⏸ / ↺，状态由 redraw(t) 纯算。 */
  function playback(box, getMax, redraw, intervalMs) {
    const row = el('div', 'lane-d-btns');
    const bPlay = el('button', '', '▶ 播放');
    const bStep = el('button', '', '⏭ 单步');
    const bPause = el('button', '', '⏸ 暂停');
    const bReset = el('button', '', '↺ 重置');
    row.append(bPlay, bStep, bPause, bReset);
    box.appendChild(row);
    const iv = intervalMs || 90;
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

  /* canvas 拖动旋转：pointer 事件只挂在这个 canvas 上。 */
  function dragRotate(cv, cam, redraw) {
    let drag = null;
    cv.addEventListener('pointerdown', e => {
      drag = { x0: e.clientX, y0: e.clientY, yaw: cam.yaw, elev: cam.elev };
      if (cv.setPointerCapture) { try { cv.setPointerCapture(e.pointerId); } catch (err) { /* 无头环境可能没有 */ } }
    });
    cv.addEventListener('pointermove', e => {
      if (!drag) return;
      cam.yaw = drag.yaw + (e.clientX - drag.x0) * 0.009;
      cam.elev = clamp(drag.elev + (e.clientY - drag.y0) * 0.007, 0.08, 1.5);
      redraw();
    });
    cv.addEventListener('pointerup', () => { drag = null; });
    cv.addEventListener('pointercancel', () => { drag = null; });
  }

  function mkCanvas(host, w, h) {
    const cv = el('canvas');
    cv.width = w * 2; cv.height = h * 2;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    host.appendChild(cv);
    const g = cv.getContext('2d');
    g.scale(2, 2);
    return { cv, g, w, h };
  }

  /* ================================================================ 1. shapebox
     把 (B,T,C) 画成可拖旋转的 3D 立方格：每格一个小立方块，块上是真数值
     （x = arange(B·T·C).reshape(B,T,C)）。点一根轴 → 沿那根轴的切片按坐标染成
     同色一片；播放 → 切片当场压扁，压完的每格显示真算的和（axis=k 就是被消失
     的那根轴，s03a §2.3）。'@ 收缩' 模式并排画第二个盒子 (C,out)，共享轴 C 同色，
     播放时"左边最后一轴 × 右边倒数第二轴"双双消失，剩 (B,T,out)（s03a §2.4）。 */

  const SB_OUT = 3;                                  // @ 模式的 out 维（固定 3，够看清）
  const SB_TSUM = 24, SB_TMM1 = 12, SB_TMM = 36;     // 压扁动画帧数

  /* W[c,o] = (3c+5o) mod 7 − 3 —— 确定性小整数，readout 里给公式，可拿 numpy 对。 */
  function sbW(c, o) { return ((3 * c + 5 * o) % 7) - 3; }

  R.shapebox = function (box) {
    box.classList.add('lane-d');
    F.title(box, '形状盒：把 (B,T,C) 拿在手里转 —— `sum(axis=k)` 压掉第 k 根轴，`@` 收缩掉共享轴');
    F.caption(box,
      '每个小立方块是数组里的一个数：`x = arange(B·T·C).reshape(B,T,C)`，块上写的就是它的值。' +
      '在图上按住拖动可以旋转。点 `axis 0/1/2` 选一根轴，沿那根轴的切片会被染成同色一片 —— ' +
      'axis=k 就是"被消失的那根轴"：再按播放，同色切片当场压扁成一片面，压完每格显示的是那一摞数真算的和，' +
      'readout 同时给出结果形状和几个真值，你可以拿 `np.arange(24).reshape(2,3,4).sum(axis=1)` 逐位对。' +
      '切到 `@ 收缩` 模式：右边并排画第二个盒子 `W(C,out)`（`W[c,o] = (3c+5o) mod 7 − 3`），' +
      '共享轴 C 在两个盒子里染成同色；播放时左边的最后一轴 × 右边的倒数第二轴双双消失，' +
      '剩下 `(B,T,out)` 的新盒子，每格 = Σc x[b,t,c]·W[c,o] —— 这就是 `x @ W` 看得见的样子。');

    const st = { B: 2, T: 3, C: 4, mode: 'sum', axis: 1, t: 0 };
    const cam = { yaw: -0.62, elev: 0.52 };

    const Wpx = Math.max(340, Math.min(880, (box.clientWidth || 880) - 16));
    const Hpx = 400;
    const cvh = el('div'); box.appendChild(cvh);
    const { cv, g } = mkCanvas(cvh, Wpx, Hpx);
    const ro = F.readout(box);
    const note = el('div', 'lane-d-note'); box.appendChild(note);

    /* ---- 控件 ---- */
    const btnRow = el('div', 'lane-d-btns'); box.appendChild(btnRow);
    const bModeSum = el('button', 'on', 'sum(axis=k) 模式');
    const bModeMM = el('button', '', '@ 收缩 模式');
    btnRow.append(bModeSum, bModeMM);
    const axisRow = el('div', 'lane-d-btns'); box.appendChild(axisRow);
    const axisBtns = [0, 1, 2].map(k => {
      const b = el('button', st.axis === k ? 'on' : '', `axis ${k}`);
      b.onclick = () => { st.axis = k; st.t = 0; syncAxisBtns(); draw(); };
      axisRow.appendChild(b); return b;
    });
    function syncAxisBtns() {
      axisBtns.forEach((b, k) => b.classList.toggle('on', st.axis === k));
      axisRow.style.display = st.mode === 'sum' ? '' : 'none';
      bModeSum.classList.toggle('on', st.mode === 'sum');
      bModeMM.classList.toggle('on', st.mode === 'mm');
    }
    bModeSum.onclick = () => { if (st.mode !== 'sum') { st.mode = 'sum'; st.t = 0; pb.stop(); syncAxisBtns(); draw(); } };
    bModeMM.onclick = () => { if (st.mode !== 'mm') { st.mode = 'mm'; st.t = 0; pb.stop(); syncAxisBtns(); draw(); } };

    const sl = F.sliders(box);
    F.slider(sl, 'B（几条序列）', 2, 5, 1, st.B, v => { st.B = v; draw(); });
    F.slider(sl, 'T（几个 token）', 2, 5, 1, st.T, v => { st.T = v; draw(); });
    F.slider(sl, 'C（几维特征）', 2, 5, 1, st.C, v => { st.C = v; draw(); });

    /* ---- 数据：全部当场真算 ---- */
    function xVal(b, t, c) { return b * st.T * st.C + t * st.C + c; }        // arange reshape
    function sumResult(k) {                                                  // x.sum(axis=k)
      const { B, T, C } = st;
      const dims = [B, T, C];
      const out = [];
      const sh = dims.filter((_, i) => i !== k);
      for (let a = 0; a < sh[0]; a++) for (let bb = 0; bb < sh[1]; bb++) {
        let s = 0;
        for (let m = 0; m < dims[k]; m++) {
          const idx = [0, 0, 0];
          idx[k] = m;
          const rem = [a, bb];
          let q = 0;
          for (let i = 0; i < 3; i++) if (i !== k) idx[i] = rem[q++];
          s += xVal(idx[0], idx[1], idx[2]);
        }
        out.push(s);
      }
      return { sh, out };
    }
    function mmResult() {                                                    // x @ W → (B,T,out)
      const { B, T, C } = st;
      const y = [];
      for (let b = 0; b < B; b++) { y.push([]); for (let t = 0; t < T; t++) { let row = []; for (let o = 0; o < SB_OUT; o++) { let s = 0; for (let c = 0; c < C; c++) s += xVal(b, t, c) * sbW(c, o); row.push(s); } y[b].push(row); } }
      return y;
    }

    /* ---- 绘制 ---- */
    function draw() {
      const { B, T, C, mode, axis, t } = st;
      const dims = [B, T, C];
      const tmax = mode === 'sum' ? SB_TSUM : SB_TMM;
      const f = clamp(t / tmax, 0, 1);
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.scale(2, 2);
      const cx = Wpx / 2, cy = Hpx / 2;

      /* 视野缩放：按旋转后的包围盒算（尺寸只随 B/T/C/模式变、不随 t 变，避免动画中途缩放跳变） */
      const ca = Math.cos(cam.yaw), sa = Math.sin(cam.yaw);
      const ce = Math.cos(cam.elev), se = Math.sin(cam.elev);
      let hx = dims[0] / 2 + 0.5, hy = dims[1] / 2 + 0.5, hz = dims[2] / 2 + 0.5;
      if (mode === 'mm') {
        hx = B / 2 + 3.0 + SB_OUT / 2 + 0.5;           // 最宽 = 右盒还在远处时
        hz = Math.max(C, SB_OUT) / 2 + 0.5;
      }
      const exX = hx * Math.abs(ca) + hy * Math.abs(sa);
      const exY = (hx * Math.abs(sa) + hy * Math.abs(ca)) * se + hz * ce;
      const S = Math.min((Wpx / 2 - 78) / Math.max(exX, 0.5), (Hpx / 2 - 28) / Math.max(exY, 0.5));

      const cells = [];       // {c, half, color, label, alpha}
      const pos = (i, n, squash) => (i - (n - 1) / 2) * squash;

      if (mode === 'sum') {
        const k = axis;
        const done = t >= tmax;
        if (done) {
          /* 压扁完成：只画结果面（薄板），每格是真算的和 */
          const res = sumResult(k);
          const sh = res.sh;
          let q = 0;
          const ax = [0, 1, 2].filter(i => i !== k);
          for (let a = 0; a < sh[0]; a++) for (let b2 = 0; b2 < sh[1]; b2++) {
            const p = [0, 0, 0];
            p[ax[0]] = pos(a, sh[0], 1); p[ax[1]] = pos(b2, sh[1], 1); p[k] = 0;
            const half = [0.44, 0.44, 0.44]; half[k] = 0.09;
            cells.push({ c: p, half, color: OKC, label: String(res.out[q++]), alpha: 1 });
          }
        } else {
          for (let i = 0; i < B; i++) for (let j = 0; j < T; j++) for (let m = 0; m < C; m++) {
            const p = [pos(i, B, 1), pos(j, T, 1), pos(m, C, 1)];
            p[k] *= (1 - f);                                   // 沿 axis k 压扁
            const half = [0.42, 0.42, 0.42];
            half[k] = 0.42 * (1 - f) + 0.03;
            const coord = [i, j, m][k];
            cells.push({
              c: p, half,
              color: lerpColor(PAL[coord % 5], OKC, f * f),
              label: String(xVal(i, j, m)), alpha: 1
            });
          }
        }
      } else {
        /* @ 收缩：左盒 (B,T,C)，右盒 (C,out)；共享轴 C（竖直）同色 */
        const f1 = clamp(t / SB_TMM1, 0, 1);                    // 靠拢
        const f2 = clamp((t - SB_TMM1) / (SB_TMM - SB_TMM1), 0, 1); // C 轴消失
        const gap = 3.0 + (0.85 - 3.0) * f1;
        const off = (B / 2 + gap + SB_OUT / 2) * (1 - f2 * 0.92);
        const done = t >= SB_TMM;
        if (!done) {
          for (let b = 0; b < B; b++) for (let tt = 0; tt < T; tt++) for (let c = 0; c < C; c++) {
            const p = [pos(b, B, 1), pos(tt, T, 1), pos(c, C, 1) * (1 - f2)];
            const half = [0.42, 0.42, 0.42 * (1 - f2) + 0.03];
            cells.push({ c: p, half, color: PAL[c % 5], label: String(xVal(b, tt, c)), alpha: 1 });
          }
          for (let c = 0; c < C; c++) for (let o = 0; o < SB_OUT; o++) {
            const p = [off + pos(o, SB_OUT, 1), 0, pos(c, C, 1) * (1 - f2)];
            const half = [0.42, 0.42 * (1 - f2) + 0.03, 0.42 * (1 - f2) + 0.03];
            cells.push({ c: p, half, color: PAL[c % 5], label: String(sbW(c, o)), alpha: 1 - f2 * 0.55 });
          }
        }
        if (f2 > 0.02) {                                          // 结果盒 (B,T,out) 淡入
          const y = mmResult();
          for (let b = 0; b < B; b++) for (let tt = 0; tt < T; tt++) for (let o = 0; o < SB_OUT; o++) {
            cells.push({
              c: [pos(b, B, 1), pos(tt, T, 1), pos(o, SB_OUT, 1)],
              half: [0.42, 0.42, 0.42], color: OKC,
              label: String(y[b][tt][o]), alpha: done ? 1 : f2
            });
          }
        }
      }

      /* 轴框架：从盒子角上伸出三条轴（线在立方块之前画，字在之后画，免得被挡住） */
      const axLab = [`axis 0 · B=${B}`, `axis 1 · T=${T}`, mode === 'mm' ? `axis 2 · C=${C}（共享）` : `axis 2 · C=${C}`];
      const base = [-dims[0] / 2 - 0.35, -dims[1] / 2 - 0.35, -dims[2] / 2 - 0.35];
      const axLabels = [];
      g.save();
      for (let k = 0; k < 3; k++) {
        const p0 = base.slice();
        const p1 = base.slice(); p1[k] += dims[k] + 0.7;
        const a = proj(cam, p0), bq = proj(cam, p1);
        const hot = mode === 'sum' && st.axis === k;
        g.strokeStyle = hot ? PAL[k % 5] : 'rgba(141,153,172,0.55)';
        g.lineWidth = hot ? 2.2 : 1.2;
        g.beginPath(); g.moveTo(cx + S * a.x, cy - S * a.y); g.lineTo(cx + S * bq.x, cy - S * bq.y); g.stroke();
        axLabels.push([cx + S * bq.x + (k === 1 ? 10 : 6), cy - S * bq.y + (k === 1 ? 14 : 0), hot ? PAL[k % 5] : DIMC, axLab[k]]);
      }
      g.restore();

      /* 画家算法：按立方块中心的视深从远到近画 */
      for (const cell of cells) {
        const p = proj(cam, cell.c);
        cell._d = p.d;
      }
      cells.sort((a, b) => b._d - a._d);
      for (const cell of cells) {
        drawCube(g, cam, S, cx, cy, cell.c, cell.half, cell.color, cell.label, cell.alpha, false);
      }
      g.save();
      g.font = '11px ui-monospace,Consolas'; g.textAlign = 'left'; g.textBaseline = 'middle';
      for (const [lx, ly, lc, lt] of axLabels) { g.fillStyle = lc; g.fillText(lt, lx, ly); }
      g.restore();

      /* readout：形状 + 真值（可以直接拿 numpy 对） */
      const N = B * T * C;
      if (mode === 'sum') {
        const k = st.axis;
        const res = sumResult(k);
        const sh = res.sh;
        const picks = [[0, 0], [sh[0] - 1, 0], [sh[0] - 1, sh[1] - 1]];
        const vals = picks.map(([a, b2]) => `[${a},${b2}]=${res.out[a * sh[1] + b2]}`).join('  ');
        ro.innerHTML = `<span>x = arange(${N}).reshape(${B},${T},${C})；` +
          `<b>x.sum(axis=${k})</b> → 形状 <b>(${sh[0]},${sh[1]})</b>（axis=${k} 被消失）` +
          `　真值：${vals}</span>`;
        note.textContent = st.t === 0
          ? `当前步 0/${SB_TSUM} —— 已选 axis=${k}：沿这根轴的 ${dims[k]} 片切片染成同色；按 ▶ 播放 / ⏭ 单步 把它们压扁。`
          : t >= SB_TSUM
            ? `当前步 ${st.t}/${SB_TSUM} —— 压扁完成：同色的 ${dims[k]} 层切片各自摞成一格，每格 = 那一摞的真和。`
            : `当前步 ${st.t}/${SB_TSUM} —— axis=${k} 方向的 ${dims[k]} 片同色切片正在压扁（第 k 轴长度 → 1 再消失）。`;
      } else {
        const y = mmResult();
        ro.innerHTML = `<span>x (${B},${T},${C}) @ W (${C},${SB_OUT}) → <b>(${B},${T},${SB_OUT})</b>` +
          `　左边最后一轴 C × 右边倒数第二轴 C，双双消失` +
          `　真值：[0,0,0]=${y[0][0][0]}  [1,1,0]=${y[B - 1][T - 1][0]}  [0,${T - 1},${SB_OUT - 1}]=${y[0][T - 1][SB_OUT - 1]}</span>`;
        note.textContent = st.t === 0
          ? `当前步 0/${SB_TMM} —— 右盒 W(C,out) 在右边待命：竖直方向同色 = 共享的 C 轴；按 ▶ 播放 开始靠拢与收缩。`
          : st.t < SB_TMM1
            ? `当前步 ${st.t}/${SB_TMM} —— 右盒 W(C,out) 正在靠近：竖直方向同色 = 共享的 C 轴。`
            : t >= SB_TMM
              ? `当前步 ${st.t}/${SB_TMM} —— 收缩完成：共享轴 C 消失，剩下 (B,T,out)=(${B},${T},${SB_OUT})，每格 = Σc x[b,t,c]·W[c,o]。`
              : `当前步 ${st.t}/${SB_TMM} —— 共享轴 C 正在被收缩掉：同色的 C 摞逐格乘起来再加总，汇进左边的 (B,T,out)。`;
      }
    }

    const pb = playback(box, () => (st.mode === 'sum' ? SB_TSUM : SB_TMM), t => { st.t = t; draw(); }, 80);
    dragRotate(cv, cam, draw);
    syncAxisBtns();
    draw();
  };

  /* ================================================================ 2. losssurf
     二参数线性模型 ŷ = w₁x + w₂ 的均方误差曲面：w₁,w₂ ∈ [−2,2] 上真算 41×41
     网格，高度着色 + 线框，可拖旋转。球 = 当前参数，▶/⏭ 让它按 SGD / Momentum /
     AdamW 之一在曲面上爬（递推以 solutions/minigpt/optim.py 为准）。
     右下 2D 等高线小插图：同一份网格的等值线 + 轨迹投影，"lr 太大在谷里来回跳"
     在这里看得最清楚。 */

  const LS = {
    XS: [-1.5, -0.7, 0.2, 0.9, 1.6],          // 5 个数据点，写死种子
    YS: [-1.25, 0.24, 0.26, 1.47, 1.28],      // ≈ 0.8x+0.3 加固定噪声
    W0: [-1.6, -1.2],                          // 球的起点
    N: 41, LO: -2, HI: 2, T: 300,
    ZS: 1.4,                                   // 高度的世界尺度
    b1: 0.9, b2: 0.95, eps: 1e-8, wd: 0.1, mu: 0.9   // 与 optim.py 默认一致
  };
  LS.step = (LS.HI - LS.LO) / (LS.N - 1);

  function lsLoss(w1, w2) {                    // mean((w1·xi + w2 − yi)²)
    let s = 0;
    for (let i = 0; i < 5; i++) {
      const r = w1 * LS.XS[i] + w2 - LS.YS[i];
      s += r * r;
    }
    return s / 5;
  }
  function lsGrad(w1, w2) {                    // d/dw1, d/dw2（和 numpy 同序累加）
    let s1 = 0, s2 = 0;
    for (let i = 0; i < 5; i++) {
      const r = w1 * LS.XS[i] + w2 - LS.YS[i];
      s1 += r * LS.XS[i];
      s2 += r;
    }
    return [2 / 5 * s1, 2 / 5 * s2];
  }

  /* 高度色带（低=蓝紫，高=深红），t ∈ [0,1] */
  const RAMP = [[0, '#313695'], [0.25, '#4575b4'], [0.45, '#74add1'], [0.62, '#abd9e9'],
                [0.78, '#fee090'], [0.9, '#f46d43'], [1, '#a50026']];
  function ramp(t) {
    t = clamp(t, 0, 1);
    for (let i = 1; i < RAMP.length; i++) {
      if (t <= RAMP[i][0]) {
        const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i];
        return lerpColor(c0, c1, (t - t0) / (t1 - t0));
      }
    }
    return RAMP[RAMP.length - 1][1];
  }

  R.losssurf = function (box) {
    box.classList.add('lane-d');
    F.title(box, 'loss 曲面：两个参数、一张真算的地形，看 SGD / Momentum / AdamW 怎么下山');
    F.caption(box,
      '模型是 `ŷ = w₁x + w₂`，loss 是 5 个固定数据点（(−1.5,−1.25) (−0.7,0.24) (0.2,0.26) (0.9,1.47) (1.6,1.28)）上的均方误差。' +
      '曲面不是示意：在 w₁,w₂ ∈ [−2,2] 的 41×41 网格上逐点真算 loss，按高度着色加线框，按住拖动可旋转。' +
      '球 = 当前参数 (w₁,w₂)，按 ▶ 播放 / ⏭ 单步，它按选中的优化器递推一步步爬 —— ' +
      'SGD 就是 `w −= lr·g`；Momentum 是 optim.py 的 SGD 类（`buf = 0.9·buf + g`，`w −= lr·buf`，等效步长约放大 10 倍）；' +
      'AdamW 逐行照抄 optim.py（β₁=0.9、β₂=0.95、先解耦衰减 `w *= (1−lr·0.1)`，再偏差修正 `m̂=m/(1−β₁ᵗ)`、`w −= lr·m̂/(√v̂+eps)`）。' +
      '把 lr 拖大再看右下等高线小图：梯度下降在谷底附近是"跨过去 → 梯度反向 → 再跨回来"，' +
      '轨迹会在谷里来回跳 —— 这正是阶段 2 §2.11 说的"固定 lr 到不了最低点"。readout 每一步都是真数，可对 numpy。');

    /* ---- 网格（mount 时真算一次，41×41） ---- */
    const N = LS.N;
    const gv = new Float64Array(N);
    for (let i = 0; i < N; i++) gv[i] = LS.LO + i * LS.step;
    const L = [];
    let Lmin = Infinity, Lmax = -Infinity, iMin = 0, jMin = 0;
    for (let i = 0; i < N; i++) {
      L.push(new Float64Array(N));
      for (let j = 0; j < N; j++) {
        const v = lsLoss(gv[i], gv[j]);
        L[i][j] = v;
        if (v < Lmin) { Lmin = v; iMin = i; jMin = j; }
        if (v > Lmax) Lmax = v;
      }
    }
    /* 解析最小二乘最优 w*（闭式真算，用来对照球最终停在哪） */
    let Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (let i = 0; i < 5; i++) { Sx += LS.XS[i]; Sy += LS.YS[i]; Sxx += LS.XS[i] * LS.XS[i]; Sxy += LS.XS[i] * LS.YS[i]; }
    const w1s = (5 * Sxy - Sx * Sy) / (5 * Sxx - Sx * Sx);
    const w2s = (Sy - w1s * Sx) / 5;

    const st = { lr: 0.1, opt: 'sgd', t: 0 };
    const cam = { yaw: -0.68, elev: 0.62 };

    /* ---- 纯函数：第 t 步的完整状态（从头重算，无头也能截确定帧） ---- */
    function stepAt(t) {
      let w1 = LS.W0[0], w2 = LS.W0[1];
      let bA = 0, bB = 0;                      // Momentum 缓冲
      let mA = 0, mB = 0, vA = 0, vB = 0;      // AdamW 一/二阶矩
      const path = [[w1, w2]];
      for (let s = 1; s <= t; s++) {
        const g = lsGrad(w1, w2);              // backward() 在 step() 之前，用的是衰减前的参数
        if (st.opt === 'sgd') {
          w1 -= st.lr * g[0]; w2 -= st.lr * g[1];
        } else if (st.opt === 'mom') {
          bA = LS.mu * bA + g[0]; bB = LS.mu * bB + g[1];
          w1 -= st.lr * bA; w2 -= st.lr * bB;
        } else {                               // adamw：optim.py AdamW.step 的次序
          w1 *= (1 - st.lr * LS.wd); w2 *= (1 - st.lr * LS.wd);   // _decay 先行
          mA = LS.b1 * mA + (1 - LS.b1) * g[0];
          mB = LS.b1 * mB + (1 - LS.b1) * g[1];
          vA = LS.b2 * vA + (1 - LS.b2) * g[0] * g[0];
          vB = LS.b2 * vB + (1 - LS.b2) * g[1] * g[1];
          const c1 = 1 - Math.pow(LS.b1, s), c2 = 1 - Math.pow(LS.b2, s);
          w1 -= st.lr * (mA / c1) / (Math.sqrt(vA / c2) + LS.eps);
          w2 -= st.lr * (mB / c1) / (Math.sqrt(vB / c2) + LS.eps);
        }
        path.push([w1, w2]);
      }
      return { w1, w2, loss: lsLoss(w1, w2), path };
    }

    /* ---- 布局 ---- */
    const side = el('div', 'lane-d-side'); box.appendChild(side);
    const mainCol = el('div'); side.appendChild(mainCol);
    const Wpx = Math.max(360, Math.min(620, (box.clientWidth || 880) - 300));
    const Hpx = 420;
    const { cv, g } = mkCanvas(mainCol, Wpx, Hpx);
    const sideCol = el('div'); sideCol.style.flex = '0 0 252px'; side.appendChild(sideCol);
    const IW = 244, IH = 224;
    const { cv: icv, g: ig } = mkCanvas(sideCol, IW, IH);
    const legend = el('div', 'lane-d-note'); sideCol.appendChild(legend);
    legend.innerHTML =
      '右下小图：同一份 41×41 网格画的等值线（w₁ 横轴、w₂ 纵轴）+ 轨迹的俯视投影。' +
      '<br>绿 × = 最小二乘闭式解 w*；白线 = 球走过的路。等高线从内到外 loss 递增。';

    const ro = F.readout(box);
    const note = el('div', 'lane-d-note'); box.appendChild(note);

    /* ---- 控件 ---- */
    const ctl = el('div', 'lane-d-btns'); box.appendChild(ctl);
    const lab = el('label', null);
    lab.innerHTML = '<b>优化器</b>';
    const sel = el('select');
    for (const [v, txt] of [['sgd', 'SGD（纯梯度）'], ['mom', 'Momentum μ=0.9'], ['adamw', 'AdamW β₂=0.95 wd=0.1']]) {
      const op = el('option'); op.value = v; op.textContent = txt; sel.appendChild(op);
    }
    sel.onchange = () => { st.opt = sel.value; pb.stop(); pb.t = 0; };
    lab.append(sel);
    ctl.append(lab);

    const sl = F.sliders(box);
    F.slider(sl, 'log₁₀ lr', -2, 0, 0.05, -1, v => { st.lr = Math.pow(10, v); pb.stop(); pb.t = 0; });

    /* ---- 3D 投影小工具 ---- */
    const w2x = w1 => w1 / 2, w2y = w2 => w2 / 2;       // [−2,2] → 世界 [−1,1]
    const l2z = l => (l - Lmin) / (Lmax - Lmin) * LS.ZS; // 高度归一（线性、真值）
    function sp(P) { return proj(cam, P); }
    function sx(p) { return Wpx / 2 + Scur * p.x; }
    function sy(p) { return Hpx / 2 - Scur * p.y; }
    let Scur = 100;

    /* ---- 主画布重绘 ---- */
    function draw3D(stt) {
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.scale(2, 2);
      const ce = Math.cos(cam.elev), se = Math.sin(cam.elev);
      /* 缩放：包围盒 2×2×ZS */
      const ca = Math.cos(cam.yaw), sa = Math.sin(cam.yaw);
      const exX = 1 * Math.abs(ca) + 1 * Math.abs(sa);
      const exY = (1 * Math.abs(sa) + 1 * Math.abs(ca)) * se + LS.ZS * ce;
      Scur = Math.min((Wpx / 2 - 44) / exX, (Hpx / 2 - 44) / exY);

      /* 底面坐标框（线先画，字等曲面画完再画，免得被山脊挡住） */
      g.save();
      g.strokeStyle = 'rgba(141,153,172,0.5)'; g.lineWidth = 1;
      const fr = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
      g.beginPath();
      fr.forEach((P, i) => { const p = sp(P); i ? g.lineTo(sx(p), sy(p)) : g.moveTo(sx(p), sy(p)); });
      g.closePath(); g.stroke();
      g.restore();
      const frameLabels = [];
      {
        let p = sp([1.16, -1, 0]); frameLabels.push([sx(p), sy(p), 'w₁']);
        p = sp([-1, 1.16, 0]); frameLabels.push([sx(p), sy(p), 'w₂']);
        p = sp([-1, -1, LS.ZS + 0.12]); frameLabels.push([sx(p) - 14, sy(p), 'loss ↑']);
      }

      /* 曲面：1600 个四边形，按视深排序（画家算法） */
      const quads = [];
      for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) {
        const c00 = sp([w2x(gv[i]), w2y(gv[j]), l2z(L[i][j])]);
        const c10 = sp([w2x(gv[i + 1]), w2y(gv[j]), l2z(L[i + 1][j])]);
        const c11 = sp([w2x(gv[i + 1]), w2y(gv[j + 1]), l2z(L[i + 1][j + 1])]);
        const c01 = sp([w2x(gv[i]), w2y(gv[j + 1]), l2z(L[i][j + 1])]);
        const d = (c00.d + c10.d + c11.d + c01.d) / 4;
        const lv = (L[i][j] + L[i + 1][j] + L[i + 1][j + 1] + L[i][j + 1]) / 4;
        quads.push({ d, pts: [c00, c10, c11, c01], lv });
      }
      quads.sort((a, b) => b.d - a.d);

      /* 球与轨迹（球按深度插进画家序列，会被更近的山脊正确遮住） */
      const bw1c = clamp(stt.w1, LS.LO, LS.HI), bw2c = clamp(stt.w2, LS.LO, LS.HI);
      const ballL = lsLoss(stt.w1, stt.w2);
      const ballZ = clamp((ballL - Lmin) / (Lmax - Lmin), 0, 1.06) * LS.ZS + 0.045;
      const ball = sp([w2x(bw1c), w2y(bw2c), ballZ]);
      function drawPath() {
        g.save();
        g.strokeStyle = 'rgba(255,255,255,0.9)'; g.lineWidth = 1.5;
        g.beginPath();
        stt.path.forEach(([a, b], i) => {
          const ac = clamp(a, LS.LO, LS.HI), bc = clamp(b, LS.LO, LS.HI);
          const zz = clamp((lsLoss(a, b) - Lmin) / (Lmax - Lmin), 0, 1.06) * LS.ZS + 0.05;
          const q = sp([w2x(ac), w2y(bc), zz]);
          i ? g.lineTo(sx(q), sy(q)) : g.moveTo(sx(q), sy(q));
        });
        g.stroke();
        const p0 = stt.path[0];
        let q = sp([w2x(clamp(p0[0], -2, 2)), w2y(clamp(p0[1], -2, 2)),
                    clamp((lsLoss(p0[0], p0[1]) - Lmin) / (Lmax - Lmin), 0, 1.06) * LS.ZS + 0.05]);
        g.fillStyle = DIMC; g.beginPath(); g.arc(sx(q), sy(q), 3, 0, 7); g.fill();
        g.restore();
      }
      function drawBall() {
        const px = sx(ball), py = sy(ball);
        g.save();
        g.fillStyle = BADC; g.beginPath(); g.arc(px, py, 7, 0, 7); g.fill();
        g.strokeStyle = '#5c1f1f'; g.lineWidth = 1.4; g.stroke();
        g.fillStyle = 'rgba(255,255,255,0.75)'; g.beginPath(); g.arc(px - 2.2, py - 2.4, 2, 0, 7); g.fill();
        g.restore();
      }

      let ballDrawn = false;
      for (const qd of quads) {
        if (!ballDrawn && qd.d < ball.d) { drawPath(); drawBall(); ballDrawn = true; }
        const t = Math.sqrt(clamp((qd.lv - Lmin) / (Lmax - Lmin), 0, 1));   // √ 让谷底附近层次更清楚
        g.beginPath();
        qd.pts.forEach((pt, i) => { i ? g.lineTo(sx(pt), sy(pt)) : g.moveTo(sx(pt), sy(pt)); });
        g.closePath();
        g.fillStyle = ramp(t); g.fill();
        g.strokeStyle = 'rgba(10,14,22,0.35)'; g.lineWidth = 0.5; g.stroke();
      }
      if (!ballDrawn) { drawPath(); drawBall(); }
      g.save();
      g.font = '700 12px ui-monospace,Consolas'; g.fillStyle = DIMC;
      for (const [lx, ly, lt] of frameLabels) g.fillText(lt, lx, ly);
      g.restore();
    }

    /* ---- 2D 等高线小插图（marching squares，真算同一份网格） ---- */
    const IP = { l: 26, r: IW - 8, t: 10, b: IH - 24 };
    const IX = w1 => IP.l + (w1 - LS.LO) / (LS.HI - LS.LO) * (IP.r - IP.l);
    const IY = w2 => IP.b - (w2 - LS.LO) / (LS.HI - LS.LO) * (IP.b - IP.t);
    function drawInset(stt) {
      ig.setTransform(1, 0, 0, 1, 0, 0); ig.clearRect(0, 0, icv.width, icv.height); ig.scale(2, 2);
      ig.fillStyle = '#101722'; ig.fillRect(0, 0, IW, IH);
      /* 8 条几何间隔的等值线（谷底密、山上疏） */
      const NL = 8;
      for (let q = 0; q < NL; q++) {
        const lv = Lmin * Math.pow(Lmax / Lmin, q / (NL - 1));
        ig.strokeStyle = ramp(Math.sqrt(q / (NL - 1)));
        ig.globalAlpha = 0.9; ig.lineWidth = 1.1;
        ig.beginPath();
        for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) {
          const v00 = L[i][j], v10 = L[i + 1][j], v01 = L[i][j + 1], v11 = L[i + 1][j + 1];
          const idx = (v00 > lv ? 1 : 0) | (v10 > lv ? 2 : 0) | (v11 > lv ? 4 : 0) | (v01 > lv ? 8 : 0);
          if (idx === 0 || idx === 15) continue;
          const p00 = [IX(gv[i]), IY(gv[j])], p10 = [IX(gv[i + 1]), IY(gv[j])];
          const p11 = [IX(gv[i + 1]), IY(gv[j + 1])], p01 = [IX(gv[i]), IY(gv[j + 1])];
          const ip = (a, b, pa, pb) => { const u = (lv - a) / (b - a); return [pa[0] + (pb[0] - pa[0]) * u, pa[1] + (pb[1] - pa[1]) * u]; };
          const eB = () => ip(v00, v10, p00, p10), eR = () => ip(v10, v11, p10, p11);
          const eT = () => ip(v01, v11, p01, p11), eL = () => ip(v00, v01, p00, p01);
          const seg = (a, b) => { ig.moveTo(a[0], a[1]); ig.lineTo(b[0], b[1]); };
          switch (idx) {
            case 1: case 14: seg(eB(), eL()); break;
            case 2: case 13: seg(eB(), eR()); break;
            case 3: case 12: seg(eL(), eR()); break;
            case 4: case 11: seg(eT(), eR()); break;
            case 6: case 9: seg(eB(), eT()); break;
            case 7: case 8: seg(eT(), eL()); break;
            case 5: seg(eB(), eL()); seg(eT(), eR()); break;
            case 10: seg(eB(), eR()); seg(eL(), eT()); break;
          }
        }
        ig.stroke(); ig.globalAlpha = 1;
      }
      /* 轨迹投影 */
      ig.strokeStyle = 'rgba(255,255,255,0.95)'; ig.lineWidth = 1.4;
      ig.beginPath();
      stt.path.forEach(([a, b], i) => {
        const px = IX(clamp(a, LS.LO, LS.HI)), py = IY(clamp(b, LS.LO, LS.HI));
        i ? ig.lineTo(px, py) : ig.moveTo(px, py);
      });
      ig.stroke();
      /* 起点 / w* / 当前球 */
      ig.fillStyle = DIMC; ig.beginPath(); ig.arc(IX(LS.W0[0]), IY(LS.W0[1]), 3, 0, 7); ig.fill();
      ig.strokeStyle = OKC; ig.lineWidth = 2;
      const gx = IX(w1s), gy = IY(w2s);
      ig.beginPath(); ig.moveTo(gx - 4, gy - 4); ig.lineTo(gx + 4, gy + 4);
      ig.moveTo(gx + 4, gy - 4); ig.lineTo(gx - 4, gy + 4); ig.stroke();
      ig.fillStyle = BADC; ig.beginPath();
      ig.arc(IX(clamp(stt.w1, LS.LO, LS.HI)), IY(clamp(stt.w2, LS.LO, LS.HI)), 4.5, 0, 7); ig.fill();
      /* 边框与轴标 */
      ig.strokeStyle = 'rgba(141,153,172,0.5)'; ig.lineWidth = 1;
      ig.strokeRect(IP.l, IP.t, IP.r - IP.l, IP.b - IP.t);
      ig.fillStyle = DIMC; ig.font = '11px ui-monospace,Consolas'; ig.textAlign = 'center';
      for (const v of [-2, -1, 0, 1, 2]) ig.fillText(String(v), IX(v), IH - 8);
      ig.textAlign = 'right';
      for (const v of [-2, 0, 2]) ig.fillText(String(v), IP.l - 5, IY(v) + 3);
      ig.textAlign = 'left';
      ig.fillText('w₁', IW - 22, IH - 8);
      ig.fillText('w₂', 4, IP.t + 26);
    }

    /* ---- 总重绘 ---- */
    function draw() {
      const stt = stepAt(st.t);
      draw3D(stt);
      drawInset(stt);
      const oob = stt.w1 < LS.LO || stt.w1 > LS.HI || stt.w2 < LS.LO || stt.w2 > LS.HI;
      const optName = st.opt === 'sgd' ? 'SGD' : st.opt === 'mom' ? 'Momentum(μ=0.9)' : 'AdamW(β₁=0.9, β₂=0.95, wd=0.1)';
      ro.innerHTML =
        `<span>当前步 <b>${st.t}/${LS.T}</b></span>` +
        `<span>${optName}　lr=<b>${st.lr.toFixed(4)}</b></span>` +
        `<span>w₁=<b>${stt.w1.toFixed(6)}</b>　w₂=<b>${stt.w2.toFixed(6)}</b></span>` +
        `<span>loss=<b>${stt.loss.toFixed(6)}</b></span>` +
        `<span>闭式最优 w*=(${w1s.toFixed(6)}, ${w2s.toFixed(6)})，loss*=${lsLoss(w1s, w2s).toFixed(6)}</span>` +
        `<span>网格最低点 (${gv[iMin].toFixed(6)}, ${gv[jMin].toFixed(6)})，loss=${Lmin.toFixed(6)}（41×41 真算）</span>` +
        (oob ? `<span style="color:${BADC}">参数越出 [−2,2] 网格（球按边缘截断显示，数字仍是真值）</span>` : '');
      note.textContent = st.t === 0
        ? '当前步 0：球在起点 w₀=(−1.6, −1.2)。按 ▶ 播放 或 ⏭ 单步 开始下山；每一步的数字都由递推真算。'
        : `当前步 ${st.t}：球从 w₀ 走了 ${st.t} 步；等高线小图里的白线是完整轨迹的俯视投影。`;
    }

    const pb = playback(box, () => LS.T, t => { st.t = t; draw(); }, 70);
    dragRotate(cv, cam, draw);
    draw();
  };

  /* ================================================================== 3. shapes3d
     s03a §2 动线的 3D 版：每一步画**张量本身**，而不是形状数字。
     查表把一格整数撑成一列浮点数、线性层把深度轴拉长、拆头把板子沿特征轴拉开 H 片、
     两次 @ 把配对的轴收缩消失（小矩阵的点积/softmax/加权和全部当场真算）、×L 沿竖直堆叠。
     显示封顶：B≤2、T≤6、特征轴≤8；块旁边标真实长度，封顶内的数字是真的、可抄去对拍
     （纯计算导出在 R.__d_math，node 与 numpy 走同一份）。 */

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
  const s3rd = r => Math.round((r() * 2 - 1) * 100) / 100;
  function s3mm(A, B) {
    if (A[0].length !== B.length) return null;
    const C = [];
    for (let i = 0; i < A.length; i++) {
      const row = [];
      for (let j = 0; j < B[0].length; j++) { let s = 0; for (let q = 0; q < A[0].length; q++) s += A[i][q] * B[q][j]; row.push(s); }
      C.push(row);
    }
    return C;
  }
  function s3T(A) {
    const out = [];
    for (let j = 0; j < A[0].length; j++) { const row = []; for (let i = 0; i < A.length; i++) row.push(A[i][j]); out.push(row); }
    return out;
  }
  function s3SoftmaxRows(M) {
    return M.map(row => {
      let m = -Infinity; for (const v of row) if (v > m) m = v;
      const e = row.map(v => Math.exp(v - m));
      let Z = 0; for (const v of e) Z += v;
      return e.map(v => v / Z);
    });
  }

  /* 纯数据：封顶小张量 + 真算的 scores/softmax/attn@v。node 对拍调的就是它。 */
  function s3Data(P) {
    const r = mulberry32(991);
    const capB = Math.min(P.B, 2), capT = Math.min(P.T, 6), capD = Math.min(P.d, 8);
    const capH = Math.min(P.H, 4), capDH = Math.min(P.d / P.H, 4);
    const ids = [], emb = [];
    for (let b = 0; b < capB; b++) {
      const ri = [], re = [];
      for (let t = 0; t < capT; t++) {
        ri.push(Math.floor(r() * P.V));
        re.push(Array.from({ length: capD }, () => s3rd(r)));
      }
      ids.push(ri); emb.push(re);
    }
    const w = Array.from({ length: capD }, () => Math.round((0.8 + r() * 0.6) * 100) / 100);
    const mat = (n, m) => Array.from({ length: n }, () => Array.from({ length: m }, () => s3rd(r)));
    const q = mat(capT, capDH), k = mat(capT, capDH), v = mat(capT, capDH);
    const scores = s3mm(q, s3T(k));
    const causal = scores.map((row, i) => row.map((x, j) => (j <= i ? x : -1e9)));
    const sm = s3SoftmaxRows(causal);
    const ao = s3mm(sm, v);
    const xr = emb[0][0];
    const rms = Math.sqrt(xr.reduce((s, x) => s + x * x, 0) / capD);
    const xn = xr.map((x, i) => x / rms * w[i]);
    return { capB, capT, capD, capH, capDH, ids, emb, w, q, k, v, scores, sm, ao, xr, rms, xn };
  }

  const S3 = { FR: 100, DUR: 62 };
  const S3NAMES = ['ids', 'Embedding', 'RMSNorm', 'qkv 线性', '拆头', 'transpose', 'scores',
                   '掩码+softmax', 'attn@v', '合头+o_proj', 'fc1', 'GELU', 'fc2', '×L 堆叠', 'head'];
  const S3BADGE = ['起点：一批整数编号', '查表：一格整数撑成一列浮点数', '沿最后一轴逐行缩放，形状不动',
                   'Linear：只把深度轴拉长 d→3d', 'reshape：沿特征轴切成 H 片', 'transpose(0,2,1)：H 提到 T 前面',
                   'q @ kᵀ：特征轴配对点积后 Σ 消失', '上三角变 −∞→0，每行 softmax 归一',
                   'attn @ v：键轴消失，值轴活下来', '片合回一根深度轴，o_proj + 残差',
                   'Linear：d→4d', 'GELU 逐元素', 'Linear：4d→d', '整套 Block 竖直堆 L 遍',
                   'head：d→V（权重与 Embedding 共享）'];

  R.shapes3d = function (box) {
    box.classList.add('lane-d');
    F.title(box, '形状动线 · 3D 版：每一步画的是**张量本身**——撑开、切开、收缩、堆叠，看得见"干了什么"');
    const P = { B: 2, T: 32, d: 64, H: 4, V: 2000, L: 4 };
    const Wpx = 880, Hpx = 470;
    const { cv, g } = mkCanvas(box, Wpx, Hpx);
    const cam = { yaw: -0.62, elev: 0.48 };
    const chipRow = el('div', 'lane-d-btns'); chipRow.style.flexWrap = 'wrap'; box.appendChild(chipRow);
    const sl = el('div', 'sliders'); box.appendChild(sl);
    const note = el('div', 'lane-d-note'); note.style.whiteSpace = 'pre-wrap'; box.appendChild(note);
    const ro = F.readout(box);
    let D = s3Data(P);

    const ease = q => q <= 0 ? 0 : q >= 1 ? 1 : q * q * (3 - 2 * q);
    const lerp = (a, b, q) => a + (b - a) * q;
    const n2 = v => String(Math.round(v * 100) / 100);
    const CB = '#7fb3e8', CG = '#8fd6b0', CV = '#b9a7e0', CGold = '#f0c07a', CT = '#5fc9c0';

    function sc(pos, cx, cy, S) { const q = proj(cam, pos); return [cx + S * q.x, cy - S * q.y]; }
    function cubes(L, cx, cy, S, o) {
      const face = o && o.face !== undefined ? o.face : 'top';
      const ord = L.map((c, i) => [proj(cam, c.pos).d, i]).sort((a, b) => b[0] - a[0]);
      for (const e of ord) { const c = L[e[1]]; drawCube(g, cam, S, cx, cy, c.pos, c.half, c.col, c.val, c.alpha, face); }
    }
    /* 生成一格立方体阵列：c→X（右）、t→Y（深处）、b→Z（上）。o 收偏移/间隔/取值函数。 */
    function grid(nb, nt, nc, o) {
      o = o || {};
      const L = [], sp = o.sp || 1;
      for (let b = 0; b < nb; b++) for (let t = 0; t < nt; t++) for (let c = 0; c < nc; c++) {
        L.push({
          pos: [(c - (nc - 1) / 2) * sp + (o.dx || 0),
                (t - (nt - 1) / 2) * sp + (o.dy || 0),
                (b - (nb - 1) / 2) * ((nt * sp) + (o.gz || 1.4)) + (o.dz || 0)],
          val: o.val ? o.val(b, t, c) : null,
          col: o.col ? o.col(b, t, c) : (o.color || CB),
          alpha: o.alpha == null ? 1 : (typeof o.alpha === 'function' ? o.alpha(b, t, c) : o.alpha),
          half: o.half || [0.42, 0.42, 0.42]
        });
      }
      return L;
    }
    const txt = (x, y, s, col, fs, al) => {
      g.save(); g.font = `700 ${fs || 11}px ui-monospace,Consolas`; g.fillStyle = col || '#cfe0ff';
      g.textAlign = al || 'left'; g.textBaseline = 'alphabetic'; g.fillText(s, x, y); g.restore();
    };
    const tag = (x, y, s) => txt(x, y, s, DIMC, 11, 'center');
    function arrow(a, b, col) {
      g.save(); g.strokeStyle = col || '#f5b445'; g.lineWidth = 1.6; g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.restore();
    }
    function heat(M, o) {
      let lo = Infinity, hi = -Infinity;
      for (const row of M) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
      return grid(1, M.length, M[0].length, {
        sp: o.sp || 1, dx: o.dx, dy: o.dy, dz: o.dz,
        half: [0.47, 0.47, 0.14],
        col: (b, i, j) => ramp(clamp((M[i][j] - lo) / (hi - lo || 1), 0, 1)),
        val: (b, i, j) => o.val !== false ? n2(M[i][j]) : null,
        alpha: o.alpha
      });
    }

    function stepAt(t) {
      const K = Math.min(14, Math.floor(t / S3.FR));
      const p = ease(Math.min(1, (t - K * S3.FR) / S3.DUR));
      g.setTransform(2, 0, 0, 2, 0, 0); g.clearRect(0, 0, Wpx, Hpx);
      txt(16, 24, `第 ${K + 1}/15 步 · ${S3NAMES[K]}`, '#cfe0ff', 12);
      txt(16, 42, S3BADGE[K], DIMC, 11.5);
      const C = { B: P.B, T: P.T, d: P.d, H: P.H, V: P.V, L: P.L, dh: P.d / P.H };
      const ap = a => Math.min(1, a * 2);
      const shapeStr = {
        0: `(${C.B},${C.T})`, 1: `(${C.B},${C.T},${C.d})`, 2: `(${C.B},${C.T},${C.d})`,
        3: `(${C.B},${C.T},${3 * C.d})`, 4: `(${C.B},${C.T},${C.H},${C.dh})`, 5: `(${C.B},${C.H},${C.T},${C.dh})`,
        6: `(${C.B},${C.H},${C.T},${C.T})`, 7: `(${C.B},${C.H},${C.T},${C.T})`, 8: `(${C.B},${C.H},${C.T},${C.dh})`,
        9: `(${C.B},${C.T},${C.d})`, 10: `(${C.B},${C.T},${4 * C.d})`, 11: `(${C.B},${C.T},${4 * C.d})`,
        12: `(${C.B},${C.T},${C.d})`, 13: `(${C.B},${C.T},${C.d})`, 14: `(${C.B},${C.T},${C.V})`
      }[K];
      const show = {
        0: 'ids 是 (' + C.B + ',' + C.T + ') 的整数表：' + C.B + ' 行（几条序列）× ' + C.T + ' 列（每条几个 token），每格一个 0..' + (C.V - 1) + ' 的词号。整数不可导 —— 梯度只能传到查表取出来的浮点数。',
        1: '查表：每个词号去 V·d 表里换一行 d=' + C.d + ' 个数。图里画的是前 ' + D.capT + ' 列、每列前 ' + D.capD + ' 个数（封顶显示），数字当场生成。参数 V·d = ' + (C.V * C.d) + '。',
        2: 'RMSNorm：沿最后一轴把每行除以 rms=√(Σx²/d) 再乘 w。形状一动不动。示例行（图里那 8 个数）：rms=' + n2(D.rms) + '，下面列出 before → after。',
        3: 'Linear 只改最后一轴：' + C.d + '→' + (3 * C.d) + '（q、k、v 三份拼在一起）。前面的 (' + C.B + ',' + C.T + ') 原样带着走 —— 这就是"逐位置"。',
        4: '拆头：' + C.d + ' = ' + C.H + ' × ' + C.dh + '。沿特征轴切成 ' + C.H + ' 片、每片 (' + C.T + ',' + C.dh + ') —— reshape 不搬数，只重新划线。',
        5: 'transpose(0,2,1)：' + C.H + ' 片重排成"每个头一个 batch"→ ' + shapeStr + '。还是那些数，只是换顺序数。',
        6: 'scores = q @ kᵀ：特征轴 ' + C.dh + ' 配对点积再 Σ，消失（红字）；两根 ' + C.T + ' 都活下来。图里 ' + D.capT + '×' + D.capT + ' 每个格子 = Σ q[i,c]·k[j,c]，用画出来的数真算。',
        7: '掩码加在 softmax 之前：上三角（看未来）变 −∞ → exp 精确 0（变暗）；softmax 沿最后一轴，每行权重和=1。readout 里第 0 行的和是当场加的。',
        8: 'attn @ v：这次消失的是"键"轴 ' + C.T + '（加权求和），值轴 ' + C.dh + ' 活下来。每行 = ' + C.T + ' 个 v 向量的加权和，权重和为 1 —— 全网络唯一的 token 间信息交换。',
        9: '合头：' + C.H + ' 片挤回一根深度轴 → ' + shapeStr + '，再过 o_proj（d→d），残差加回主干。和拆头是同一件事反着做。',
        10: 'fc1 又是"只改最后一轴"：' + C.d + '→' + (4 * C.d) + '。参数 4d²+4d = ' + (4 * C.d * C.d + 4 * C.d) + '。',
        11: 'GELU 逐元素：形状一动不动，没有参数。',
        12: 'fc2：' + (4 * C.d) + '→' + C.d + '。参数 4d²+d = ' + (4 * C.d * C.d + C.d) + '。残差再加 —— 一个 Block 结束，进出都是 (' + C.B + ',' + C.T + ',' + C.d + ')。',
        13: '×L：整套 Block 竖直堆 ' + C.L + ' 遍。能堆的唯一前提是"进和出的形状相等"。每层参数 12d²+10d = ' + (12 * C.d * C.d + 10 * C.d) + '。',
        14: 'head：最后一轴 ' + C.d + '→' + C.V + '，每个位置给整张词表打分。E 和 Embedding 共享（tie），不新增参数。'
      }[K];
      note.textContent = show;

      const midY = 250;
      if (K === 0) {
        cubes(grid(D.capB, D.capT, 1, { val: (b, t) => String(D.ids[b][t]), color: CB }), 440, midY, 26);
        const rowGap = D.capT + 1.4;
        for (let b = 0; b < D.capB; b++) {
          const z = (b - (D.capB - 1) / 2) * rowGap;
          const p0 = sc([-1.15, -(D.capT - 1) / 2, z], 440, midY, 26);
          txt(p0[0], p0[1] + 4, `b=${b}`, '#8d99ac', 11, 'right');
          if (C.T > D.capT) {
            const p1 = sc([0, (D.capT - 1) / 2 + 1.02, z], 440, midY, 26);
            txt(p1[0], p1[1] + 3, '…', '#cfe0ff', 16, 'center');
          }
        }
        txt(440, 86, `B=${C.B} 条序列 × 每条 T=${C.T} 个 token（每格一个词号）；图里画 ${D.capB} 行 × ${D.capT} 列`, '#cfe0ff', 11.5, 'center');
        txt(440, 104, C.T > D.capT ? `图里封顶只画 ${D.capT} 列；"…" 后面还有 ${C.T - D.capT} 个 token` : `图里画全了：${D.capB} 行 × ${D.capT} 列`, DIMC, 11, 'center');
        tag(440, 448, `真实形状 (${C.B},${C.T})；画出来的是 ${D.capB} 行 × ${D.capT} 格`);
      } else if (K === 1) {
        cubes(grid(D.capB, D.capT, 1, { val: (b, t) => String(D.ids[b][t]), color: CB }), 230, midY, 15);
        cubes(grid(D.capB, D.capT, D.capD, {
          color: (b, t) => (b === 0 && t === 0) ? '#f5b445' : CT,
          alpha: (b, t) => (b === 0 && t === 0 ? ap(p + 0.4) : ap(p))
        }), 650, midY, 15);
        const z0 = -(D.capB - 1) / 2 * (D.capT + 1.4);
        arrow(sc([0, -(D.capT - 1) / 2, z0], 230, midY, 15),
              sc([0, -(D.capT - 1) / 2, z0], 650, midY, 15), '#f5b445');
        tag(230, 448, `(${C.B},${C.T})`); tag(650, 448, `(${C.B},${C.T},${C.d})`);
        txt(440, 76, '词号 ' + D.ids[0][0] + ' → 表里第 ' + D.ids[0][0] + ' 行：[' + D.emb[0][0].slice(0, 4).map(n2).join(', ') + ', …]', '#f5b445', 11, 'center');
      } else if (K === 2) {
        cubes(grid(D.capB, D.capT, D.capD, {
          color: (b, t) => (b === 0 && t === 0) ? '#f5b445' : CT, alpha: 0.92
        }), 440, midY, 19);
        txt(440, 92, '示例行 before [' + D.xr.slice(0, 5).map(n2).join(', ') + ', …]', DIMC, 11, 'center');
        txt(440, 110, '÷ rms=' + n2(D.rms) + ' × w → [' + D.xn.slice(0, 5).map(n2).join(', ') + ', …]', '#f5b445', 11, 'center');
        tag(440, 448, `(${C.B},${C.T},${C.d}) 进、(${C.B},${C.T},${C.d}) 出 —— 一根轴都没变`);
      } else if (K === 3 || K === 10 || K === 12 || K === 14) {
        const nc1 = K === 12 ? Math.min(4 * D.capD, 12) : D.capD;
        const nc2 = K === 3 ? Math.min(3 * D.capD, 12) : K === 10 ? Math.min(4 * D.capD, 12) : K === 12 ? D.capD : 8;
        const ind = K === 12 ? 4 * C.d : C.d;
        const outd = K === 3 ? 3 * C.d : K === 10 ? 4 * C.d : K === 12 ? C.d : C.V;
        const col2 = K === 3 ? CT : K === 10 ? CGold : K === 12 ? CG : '#f5b445';
        cubes(grid(D.capB, D.capT, nc1, { color: CB, val: false }), 230, midY, 15);
        cubes(grid(D.capB, D.capT, nc2, { color: col2, val: false, alpha: ap(p) }), 650, midY, 15);
        tag(230, 448, `(${C.B},${C.T},${ind})`);
        tag(650, 448, `(${C.B},${C.T},${outd})${K === 14 ? '（画 8 格，共 ' + C.V + '）' : ''}`);
        txt(440, 78, `只换最后一轴：${ind}→${outd}，前面 (${C.B},${C.T}) 原样带着走`, '#cfe0ff', 11.5, 'center');
      } else if (K === 4 || K === 9) {
        const gap = (K === 4 ? p : 1 - p) * 1.3;
        const L = [];
        for (let h = 0; h < D.capH; h++) {
          const dx = (h - (D.capH - 1) / 2) * (D.capDH + gap + 0.15);
          for (let b = 0; b < D.capB; b++) for (let t = 0; t < D.capT; t++) for (let c = 0; c < D.capDH; c++) {
            L.push({
              pos: [c - (D.capDH - 1) / 2 + dx, t - (D.capT - 1) / 2, (b - (D.capB - 1) / 2) * (D.capT + 1.3)],
              col: PAL[h % PAL.length], alpha: 0.95, val: null, half: [0.42, 0.42, 0.42]
            });
          }
        }
        cubes(L, 440, midY, 17);
        tag(440, 448, K === 4 ? `每片是 (T, d/H)=(${C.T},${C.dh})；图里画前 ${D.capH} 片，共 ${C.H} 片` : `合回 (${C.B},${C.T},${C.d})，再 o_proj d→d + 残差`);
      } else if (K === 5) {
        const L = [];
        const nTot = D.capB * D.capH;
        for (let b = 0; b < D.capB; b++) for (let h = 0; h < D.capH; h++) for (let t = 0; t < D.capT; t++) for (let c = 0; c < D.capDH; c++) {
          const fx = (h - (D.capH - 1) / 2) * (D.capDH + 1.45);
          const fz = (b - (D.capB - 1) / 2) * (D.capT + 1.3);
          const tz = ((b * D.capH + h) - (nTot - 1) / 2) * 1.35;
          L.push({
            pos: [c - (D.capDH - 1) / 2 + lerp(fx, 0, p), t - (D.capT - 1) / 2, lerp(fz, tz, p)],
            col: PAL[h % PAL.length], alpha: 0.95, val: null, half: [0.42, 0.42, 0.42]
          });
        }
        cubes(L, 440, midY, 16);
        tag(440, 448, `(${C.B},${C.T},${C.H},${C.dh}) → (${C.B},${C.H},${C.T},${C.dh})：每个头变成一个"小 batch"`);
      } else if (K === 6) {
        cubes(grid(1, D.capT, D.capDH, { color: CG, val: (b, i, j) => n2(D.q[i][j]) }), 170, midY, 15);
        cubes(grid(1, D.capT, D.capDH, { color: CV, val: (b, i, j) => n2(D.k[i][j]) }), 355, midY, 15);
        cubes(heat(D.scores, { alpha: ap(p) }), 645, midY, 15);
        txt(440, 78, '公共轴 d/H=' + C.dh + '：配对相乘再 Σ，然后消失', BADC, 11.5, 'center');
        tag(170, 448, `q (${C.T},${C.dh})`); tag(355, 448, `k (${C.T},${C.dh})`); tag(645, 448, `scores (${C.T},${C.T})（画前 ${D.capT}×${D.capT}）`);
      } else if (K === 7) {
        cubes(heat(D.sm, { sp: 1.15 }), 440, midY, 24);
        const sp = 1.15;
        for (let i = 0; i < D.capT; i++) for (let j = i + 1; j < D.capT; j++) {
          const q0 = proj(cam, [(j - (D.capT - 1) / 2) * sp, (i - (D.capT - 1) / 2) * sp, 0]);
          g.save(); g.globalAlpha = p * 0.6; g.fillStyle = '#0b0f16';
          g.beginPath(); g.arc(440 + 24 * q0.x, midY - 24 * q0.y, 8, 0, 7); g.fill(); g.restore();
        }
        tag(440, 448, `(${C.B},${C.H},${C.T},${C.T}) 形状不动；上三角 = 未来，softmax 后精确 0`);
      } else if (K === 8) {
        cubes(heat(D.sm, { val: false }), 185, midY, 14);
        cubes(grid(1, D.capT, D.capDH, { color: CV, val: (b, i, j) => n2(D.v[i][j]) }), 400, midY, 14);
        cubes(grid(1, D.capT, D.capDH, { color: CG, val: (b, i, j) => n2(D.ao[i][j]), alpha: ap(p) }), 655, midY, 14);
        txt(440, 78, '键轴 ' + C.T + ' 消失（每行加权求和），值轴 ' + C.dh + ' 活下来', BADC, 11.5, 'center');
        tag(185, 448, `权重 (${C.T},${C.T})`); tag(400, 448, `v (${C.T},${C.dh})`); tag(655, 448, `(${C.T},${C.dh})`);
      } else if (K === 11) {
        cubes(grid(D.capB, D.capT, Math.min(4 * D.capD, 12), {
          color: lerpColor(CGold, '#f5b445', 0.5 + 0.5 * Math.sin(p * Math.PI)), val: false
        }), 440, midY, 15);
        tag(440, 448, `(${C.B},${C.T},${4 * C.d}) 进、出 —— 逐元素，一根轴都不动`);
      } else {
        const L = [];
        const nL = Math.min(P.L, 6);
        for (let i = 0; i < nL; i++) {
          const a = clamp(p * (nL + 1) - i, 0, 1);
          if (a <= 0) continue;
          L.push({ pos: [0, 0, (i - (nL - 1) / 2) * 1.5], col: PAL[i % PAL.length], alpha: a, val: `Block ${i + 1}`, half: [3.4, 2.3, 0.32] });
        }
        cubes(L, 440, midY, 26, { face: 'front' });
        txt(120, midY, '残差：x = x + attn(x)\nx = x + mlp(x)', DIMC, 11);
        tag(440, 448, `L=${C.L} 遍，进出都是 (${C.B},${C.T},${C.d}) —— 形状相等才堆得上`);
      }
      const row0 = D.sm[0].reduce((a, b) => a + b, 0);
      const perLayer = 12 * C.d * C.d + 10 * C.d;
      ro.innerHTML =
        `<span>真实形状 <b>${shapeStr}</b></span>`
        + `<span>画面：批次画前 ${D.capB} 条、序列画前 ${D.capT} 个 token、特征画前 ${D.capD} 维（封顶省略），数字都真算</span>`
        + (K >= 6 && K <= 8 ? `<span>第 0 行权重和 = <b>${row0.toFixed(6)}</b></span>` : '')
        + `<span>每层 <b>${perLayer}</b>　+词表 V·d=<b>${C.V * C.d}</b>　合计 <b>${((perLayer * C.L + C.V * C.d) / 1e6).toFixed(2)} M</b></span>`
        + `<span>步 <b>${K + 1}</b>/15</span>`;
      chips.forEach((b, k) => { b.style.borderColor = k === K ? '#4aa3ff' : ''; });
    }

    const MAXT = 14 * S3.FR + S3.DUR;
    let t = 0, raf = 0, mode = 'idle', target = 0;
    function stop() { mode = 'idle'; if (raf) { cancelAnimationFrame(raf); raf = 0; } }
    function tick() {
      if (mode === 'idle') { raf = 0; return; }
      if (mode === 'play') t = t >= MAXT ? 0 : t + 1;
      else if (t >= target) { stop(); stepAt(t); return; }
      else t++;
      stepAt(t);
      raf = requestAnimationFrame(tick);
    }
    function go(to) {
      stop(); mode = 'to'; target = Math.min(MAXT, to);
      if (target <= t) { mode = 'idle'; stepAt(t); return; }
      raf = requestAnimationFrame(tick);
    }
    const btnRow = el('div', 'lane-d-btns'); box.appendChild(btnRow);
    const mk = (txt2, fn) => { const b = el('button', '', txt2); b.onclick = fn; btnRow.appendChild(b); return b; };
    mk('▶ 播放', () => { if (mode === 'play') return; stop(); mode = 'play'; raf = requestAnimationFrame(tick); });
    mk('⏭ 下一步', () => go((Math.min(14, Math.floor(t / S3.FR)) + 1) * S3.FR + S3.DUR));
    mk('⏸ 暂停', stop);
    mk('↺ 重置', () => { stop(); t = 0; stepAt(0); });

    const chips = S3NAMES.map((s, k) => {
      const b = el('button', '', `${k + 1} ${s}`);
      b.onclick = () => { stop(); t = k * S3.FR + S3.DUR; stepAt(t); };
      chipRow.appendChild(b); return b;
    });

    function rebuild() { D = s3Data(P); stepAt(t); }
    F.slider(sl, 'B（批次）', 1, 4, 1, P.B, v => { P.B = v; rebuild(); });
    F.slider(sl, 'T（序列长）', 8, 64, 8, P.T, v => { P.T = v; rebuild(); });
    F.slider(sl, 'd（模型宽）', 32, 256, 32, P.d, v => { P.d = v; rebuild(); });
    F.slider(sl, 'H（头数）', 1, 8, 1, P.H, v => {
      P.H = [1, 2, 4, 8].reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a); rebuild();
    });
    F.slider(sl, 'V（词表）', 1000, 32000, 1000, P.V, v => { P.V = v; rebuild(); });
    F.slider(sl, 'L（层数）', 1, 8, 1, P.L, v => { P.L = v; rebuild(); });

    dragRotate(cv, cam, () => stepAt(t));

    F.caption(box,
      '每一步只在做四种动作之一：**查表撑开**（一格整数 → 一行浮点数，多出一根特征轴）、' +
      '**拉长/缩回**（线性层只换最后一轴，前面的 (B,T) 原样带着走）、**切/拼**（reshape 沿特征轴划线或合拢，一个数都不搬）、' +
      '**收缩**（两次 @：配对的那根轴点积求和后消失，其余轴都活）。其余步（RMSNorm、GELU、softmax、掩码、残差、×L）形状一动不动。' +
      '整条动线里轴只消失两次，恰好是仅有的两处 token 之间交换信息的地方 —— 把一根轴求和掉，就是让每个位置去看另一根轴上的所有位置。' +
      '图可以拖动转视角；封顶显示的规则：画不下的地方用 "…" 标出并写真实长度（ids 那步一行 = 一条序列 b=0/b=1，一列 = 一个 token），' +
      '画出来的每个数字都当场算、可抄进 numpy 对拍。');
    box.__stepAt = stepAt;   /* 测试钩子：无头截图要定帧到变形中途。 */
    stepAt(0);
  };

  /* node 对拍入口：与 shapes3d 显示用的同一份纯计算。 */
  R.__d_math = { mulberry32, s3Data, s3mm, s3T, s3SoftmaxRows };
})();
