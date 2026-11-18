/* 原理动画 · 第 F 批：训练循环里三处"看代码看不出来、一错就全错"的机制。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和前几批一样：能算的都真的算 —— readout 里每个数字都是 JS 当场算的，
   拿 numpy 能逐位对上；组件抛错只是"这张图没出来"，不连累正文。

   本批三个组件：batcher（s05 §10 · TODO-21）/ clipnorm（s05 §12 · TODO-23）/
   samplewheel（s07 §7 · TODO-31 的最后一步"按概率抽样"本身；温度/top-k/top-p
   那条流水线 figs.js 的 sampling 组件已经画过，这里刻意不重复）。
   共同约定：
     · 口径逐行对齐参考实现：
       - batcher 对齐 solutions/minigpt/dataset.py 的 Batcher：窗口长 ctx+1，
         x = win[:-1]、y = win[1:]，合法起点 0..N−ctx−1（max_start = N−ctx−1，
         写成 N−ctx 是 off-by-one 重灾区）。真实 Batcher 是**有放回随机**起点；
         图里为了能播放、能对拍，起点取 (base + b·stride) mod (max_start+1)，
         base 由 ▶/⏭ 控制，任意一帧由纯函数 stepAt(t) 从头重算。
       - clipnorm 对齐 solutions/minigpt/optim.py 的 clip_grad_norm：
         total = sqrt(Σ 每个元素的平方)；**只有** total > max_norm 且 total > 0
         才乘同一个 scale = max_norm/total —— 参考实现里**没有 eps**，防除零
         靠的是条件本身（total=0 时根本不进分支）。返回裁剪**前**的 total。
       - samplewheel 对齐 numpy Generator.choice 的内部做法：
         cdf = p.cumsum(); cdf /= cdf[-1]; idx = cdf.searchsorted(u, 'right')，
         即"第一个满足 u < cdf[k] 的 k"。JS 侧 rng 用 mulberry32（与 figs_b
         同一颗，python 侧可逐位复现 u 序列；numpy 自己的 PCG64 序列不同，
         所以对拍的是"同一串 u → 同一串 idx"与频率收敛，见 __f_math）。
     · rAF 只在播放期间存在，mount 时不起任何定时器；事件只挂自己创建的元素。
     · 自带样式全部以 `.lane-f ` 前缀注入（<style id="lane-f-style">），
       不碰 manual.css。
     · 纯计算函数经 R.__f_math 导出，node 里 stub 掉 window/document 后可直接
       调用，与临时 python/numpy 脚本对账。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  const el = F.el, fmt = F.fmt;

  /* ---------------------------------------------------------------- 公共工具 */

  (function injectStyle() {
    if (typeof document === 'undefined' || document.getElementById('lane-f-style')) return;
    const s = el('style'); s.id = 'lane-f-style';
    s.textContent = [
      '.lane-f canvas{display:block;max-width:100%;margin:6px 0 0;touch-action:none;',
      '  border:1px solid var(--line);border-radius:8px;background:#101722}',
      '.lane-f .lane-f-btns{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 2px;align-items:center}',
      '.lane-f .lane-f-note{font:11.5px/1.5 var(--mono);color:var(--dim);margin-top:4px}',
      '.lane-f select{background:#151d2b;color:var(--fg);border:1px solid var(--line);',
      '  border-radius:7px;padding:5px 8px;font:12px/1 var(--mono);cursor:pointer}',
      '.lane-f .lane-f-side{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}'
    ].join('\n');
    (document.head || document.documentElement).appendChild(s);
  })();

  const PAL = ['#4aa3ff', '#3ecf8e', '#f5b445', '#ff6b6b', '#b9a7e0', '#5fd0d8'];
  const OKC = '#3ecf8e', BADC = '#ff6b6b', WARNC = '#f5b445', DIMC = '#8d99ac', FGC = '#e6ebf3';

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  /* ‖v‖ 不用 Math.hypot：hypot 内部有缩放技巧，与 numpy 的 sqrt(x·x+y·y) 差一个 ulp；
     对拍要逐位一致，就按 numpy 的口径逐字面写。 */
  function norm2(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1]); }

  /* 与 figs_b 同一颗确定性 rng：mulberry32。python 侧逐位对得上 u 序列。 */
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

  /* 播放控制（同 figs_b/d 的约定，自己一份）：▶ / ⏭ / ⏸ / ↺，状态由 redraw(t) 纯算。 */
  function playback(box, getMax, redraw, intervalMs) {
    const row = el('div', 'lane-f-btns');
    const bPlay = el('button', '', '▶ 播放');
    const bStep = el('button', '', '⏭ 单步');
    const bPause = el('button', '', '⏸ 暂停');
    const bReset = el('button', '', '↺ 重置');
    row.append(bPlay, bStep, bPause, bReset);
    box.appendChild(row);
    const iv = intervalMs || 120;
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

  function mkCanvas(host, w, h) {
    const cv = el('canvas');
    cv.width = w * 2; cv.height = h * 2;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    host.appendChild(cv);
    const g = cv.getContext('2d');
    g.scale(2, 2);
    return { cv, g, w, h };
  }

  function arrow(g, x0, y0, x1, y1, color, dash, wid) {
    g.save();
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = wid || 2;
    if (dash) g.setLineDash(dash);
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0), L = 7;
    g.setLineDash([]);
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - L * Math.cos(a - 0.42), y1 - L * Math.sin(a - 0.42));
    g.lineTo(x1 - L * Math.cos(a + 0.42), y1 - L * Math.sin(a + 0.42));
    g.closePath(); g.fill();
    g.restore();
  }

  /* ---------------------------------------------------------------- 纯计算 */

  /* data/corpus.txt 的前 64 个字符，原样抄录（第 56 号是换行）。
     不 fetch 文件：讲义是静态页，相对路径在测评台/独立窗口下不可靠。 */
  const CORPUS64 = 'the small town sits between two hills and a slow river .\nin the ';

  /* Batcher 的可视化版：起点 = (base + b·stride) mod (max_start+1)。
     窗口恒为完整 T+1 格：x = 流[s:s+T]，y = 流[s+1:s+T+1]（solutions/dataset.py
     的 win[:, :-1] / win[:, 1:]），末尾**不补齐也不丢弃** —— 最后 T 个字符当不了
     起点，但仍会作为更早窗口的目标出现。 */
  function batchWindows(stream, T, B, stride, base) {
    const N = stream.length;
    const maxStart = N - T - 1;
    if (maxStart < 0) return null;
    const starts = [];
    for (let b = 0; b < B; b++) starts.push((base + b * stride) % (maxStart + 1));
    const xs = starts.map(s => stream.slice(s, s + T));
    const ys = starts.map(s => stream.slice(s + 1, s + T + 1));
    return { N, maxStart, starts, xs, ys };
  }

  /* 错法②：把 batch 摊平成一维再整体错位一格。
     flat = x₀x₁…x_{B−1}（B·T 格），y[b·T+k] = flat[b·T+k+1]。
     行内没错，行尾第 T−1 格抓到的是**下一个样本**的第 0 格；最后一个样本
     越界（这里演示成回绕到 flat[0]，真实代码是 IndexError 或跨批串味）。 */
  function flatShiftY(win) {
    const B = win.xs.length, T = win.xs[0].length;
    const flat = win.xs.join('');
    const yb = [];
    for (let b = 0; b < B; b++) {
      let row = '';
      for (let k = 0; k < T; k++) {
        const gi = b * T + k + 1;
        row += gi < flat.length ? flat[gi] : flat[0];
      }
      yb.push(row);
    }
    const mism = [];
    for (let b = 0; b < B; b++) for (let k = 0; k < T; k++) if (yb[b][k] !== win.ys[b][k]) mism.push([b, k]);
    return { yb, mism };
  }

  /* "复读机"命中率：拿 x[k] 直接当 y[k] 的答案能对多少。
     正确移位下 = 语料里相邻重复字符的比例（很低）；y=x 下恒 100%。 */
  function copyAcc(xs, ys) {
    let hit = 0, tot = 0;
    for (let b = 0; b < xs.length; b++) for (let k = 0; k < xs[b].length; k++) {
      tot++;
      if (xs[b][k] === ys[b][k]) hit++;
    }
    return tot ? hit / tot : 0;
  }

  /* clip_grad_norm 的 2D 演示版：gs 是 [[x,y]…]（每个"参数张量"压成一个向量，
     只为让"方向不变"看得见；范数/夹角都按这些数真算）。
     mode: 'ok' = 参考实现；'pertensor' = 逐张量各自裁；'order' = 先裁再算 total
     （返回值变成裁后的）；'uncond' = 无条件乘 scale（小梯度被放大）。
     注意：参考实现没有 eps，防除零靠 total > max_norm && total > 0 这两个条件。 */
  function angDeg(a, b) {
    const na = norm2(a), nb = norm2(b);
    if (na === 0 || nb === 0) return 0;
    const c = (a[0] * b[0] + a[1] * b[1]) / (na * nb);
    return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
  }
  function clipApply(gs, maxNorm, mode) {
    const mags = gs.map(norm2);
    let ss = 0;
    for (const v of gs) ss += v[0] * v[0] + v[1] * v[1];
    const total = Math.sqrt(ss);                       // sqrt(Σ‖g_i‖²)，跨张量全局
    let scale = 1, out = gs.map(v => v.slice()), returned = total;
    if (mode === 'pertensor') {
      out = gs.map((v, i) => {
        const s = (mags[i] > maxNorm && mags[i] > 0) ? maxNorm / mags[i] : 1;
        return [v[0] * s, v[1] * s];
      });
    } else if (mode === 'uncond') {
      if (total > 0) {
        scale = maxNorm / total;                        // 少了 total > max_norm 这个条件
        out = gs.map(v => [v[0] * scale, v[1] * scale]);
      }
    } else {                                            // 'ok' 与 'order' 的梯度本身一样
      if (total > maxNorm && total > 0) {
        scale = maxNorm / total;
        out = gs.map(v => [v[0] * scale, v[1] * scale]);
      }
      if (mode === 'order') {
        let sa = 0;
        for (const v of out) sa += v[0] * v[0] + v[1] * v[1];
        returned = Math.sqrt(sa);                       // 裁后才算 total → 返回错值
      }
    }
    let after = 0;
    for (const v of out) after += v[0] * v[0] + v[1] * v[1];
    after = Math.sqrt(after);
    const angles = gs.map((v, i) => angDeg(v, out[i]));
    const scales = gs.map((v, i) => mags[i] > 0 ? norm2(out[i]) / mags[i] : 1);
    return { total, mags, scale, scales, out, after, returned, angles };
  }

  /* sampler.py 的 _softmax(z / temperature)：减最大值、exp、除以和。 */
  function softmaxT(z, T) {
    const zz = z.map(v => v / T);
    const m = Math.max.apply(null, zz);
    const e = zz.map(v => Math.exp(v - m));
    let S = 0;
    for (const v of e) S += v;
    return e.map(v => v / S);
  }
  /* numpy Generator.choice 的内部：cdf = p.cumsum(); cdf /= cdf[-1];
     idx = searchsorted(cdf, u, 'right') = 第一个满足 u < cdf[k] 的 k。 */
  function cdfOf(p) {
    const cum = [];
    let s = 0;
    for (const q of p) { s += q; cum.push(s); }
    const last = cum[cum.length - 1];
    return { cum, norm: cum.map(v => v / last), S: last };
  }
  function pickIdx(cdfn, u) {
    for (let k = 0; k < cdfn.length; k++) if (u < cdfn[k]) return k;
    return cdfn.length;                                 // 理论上到不了（Σp=1 时）
  }

  /* mode: 'ok' = 正确累积区间；'nocum' = 拿 u 和**单个** p[k] 比（不累积）；
     'unnorm' = p 没归一（最后一个词概率被清零却没把质量摊回去，Σp<1 → 缺口）；
     'argmax' = 用 argmax 代替抽样（rng 根本不被碰）。
     状态是纯的：给定 (p, n, seed, mode)，从头重放 n 次，us 保留前 20 个供对拍。 */
  function sampleRun(p, n, seed, mode) {
    const V = p.length;
    const counts = new Array(V).fill(0);
    let gaps = 0, lastU = null, lastIdx = null, argmax = 0;
    let cmp = null;                                     // 实际用来比较的累积数组
    if (mode === 'ok') cmp = cdfOf(p).norm;
    else if (mode === 'unnorm') {
      const q = p.slice(); q[V - 1] = 0;                // top-p 砍掉尾巴却没重新归一
      cmp = cdfOf(q).cum;                               // 不归一：最后一格 = Σp < 1
    }
    if (mode === 'argmax') for (let i = 1; i < V; i++) if (p[i] > p[argmax]) argmax = i;
    const rng = mulberry32(seed);
    const us = [];
    for (let i = 0; i < n; i++) {
      let idx;
      if (mode === 'argmax') {
        idx = argmax;
      } else {
        const u = rng();
        lastU = u;
        if (us.length < 20) us.push(u);
        if (mode === 'nocum') {
          idx = V - 1;                                  // 全落空时兜底最后一个词
          for (let k = 0; k < V; k++) if (u < p[k]) { idx = k; break; }
        } else {
          idx = pickIdx(cmp, u);
          if (idx >= V) gaps++;                         // u 落进缺口（numpy 会直接抛错）
        }
      }
      lastIdx = idx;
      if (idx < V) counts[idx]++;
    }
    return { counts, gaps, lastU, lastIdx, us, n };
  }

  /* ================================================================ 1. batcher
     把"x 和 y 错位一格"画在真实字节流上：流是 data/corpus.txt 的前 64 个字符，
     每个样本 x = 流[s:s+T]、y = 流[s+1:s+T+1]，箭头逐格连 x[k]→y[k]，
     y 的最后一格（来自流的下一格、不在 x 里）高亮。播放时窗口沿流一格一格滑，
     每帧由 stepAt(t) 从头重算。 */

  const dispCh = c => c === '\n' ? '⏎' : c === ' ' ? '·' : c;
  const showStr = s => F.esc1(s.replace(/ /g, '␣').replace(/\n/g, '⏎'));

  R.batcher = function (box) {
    box.classList.add('lane-f');
    F.title(box, 'Batcher：`x = 流[i:i+T]`、`y = 流[i+1:i+T+1]` —— 错位一格到底切出了什么');
    F.caption(box,
      '下面这排格子是<b>真数据</b>：`data/corpus.txt` 的前 64 个字符（`·` = 空格，`⏎` = 换行，都是流里真实的一格）。' +
      '每个样本从流里取一段长 `T+1` 的窗口：去掉最后一格是输入 `x`，去掉第一格是目标 `y` —— ' +
      '于是 `y[k] = 流[s+k+1] = x[k+1]`，模型在第 k 步看到 `x[k]` 时，要预测的其实是<b>流的第 s+k+1 个字符</b>；' +
      '`y` 的最后一格（黄框）来自窗口第 T+1 个字符，<b>不在 x 里</b>，这就是窗口必须切 `T+1` 而不是 `T` 的原因。' +
      '真实 Batcher 的起点是有放回随机抽的；图里为了能播放，起点取 `(base + b·stride) mod (max_start+1)`，' +
      'base 由 ▶/⏭ 控制，每帧都从头真算。拖 `T`、`batch`，切 `stride=T`（连续切）/ `stride=1`（滑动一格），' +
      '再打开两个错法开关看：`y=x` 会让判据 `移位关系 y[:, :-1] == x[:, 1:]` 变红、模型退化成复读机；' +
      '摊平再错位会让每个样本的最后一格抓到<b>隔壁样本</b>的第一个字符。');

    const st = { T: 8, B: 2, strideMode: 'T', wrong: 'none', t: 0 };
    const stream = CORPUS64;

    const Wpx = Math.max(360, Math.min(880, (box.clientWidth || 880) - 16));
    const BLK = 100, TOP0 = 74;
    function canvasH() { return TOP0 + st.B * BLK + 14; }
    const cvh = el('div'); box.appendChild(cvh);
    const { cv, g } = mkCanvas(cvh, Wpx, canvasH());
    const ro = F.readout(box);
    const note = el('div', 'lane-f-note'); box.appendChild(note);

    /* ---- 控件 ---- */
    const btnRow = el('div', 'lane-f-btns'); box.appendChild(btnRow);
    const bStrideT = el('button', 'on', '连续切（stride=T）');
    const bStride1 = el('button', '', '滑动一格（stride=1）');
    btnRow.append(bStrideT, bStride1);
    const wrongRow = el('div', 'lane-f-btns'); box.appendChild(wrongRow);
    const wrongRowLab = el('span', null); wrongRowLab.style.cssText = 'font:11.5px/1 var(--mono);color:var(--dim)';
    wrongRowLab.textContent = '错法对照：';
    wrongRow.appendChild(wrongRowLab);
    const bW0 = el('button', 'on', '正确（win[:, 1:]）');
    const bW1 = el('button', '', '① y = x（不错位）');
    const bW2 = el('button', '', '② 摊平后再错位（跨样本串味）');
    wrongRow.append(bW0, bW1, bW2);
    function syncBtns() {
      bStrideT.classList.toggle('on', st.strideMode === 'T');
      bStride1.classList.toggle('on', st.strideMode === '1');
      bW0.classList.toggle('on', st.wrong === 'none');
      bW1.classList.toggle('on', st.wrong === 'yx');
      bW2.classList.toggle('on', st.wrong === 'flat');
    }
    bStrideT.onclick = () => { st.strideMode = 'T'; syncBtns(); draw(); };
    bStride1.onclick = () => { st.strideMode = '1'; syncBtns(); draw(); };
    bW0.onclick = () => { st.wrong = 'none'; syncBtns(); draw(); };
    bW1.onclick = () => { st.wrong = 'yx'; syncBtns(); draw(); };
    bW2.onclick = () => { st.wrong = 'flat'; syncBtns(); draw(); };

    const sl = F.sliders(box);
    F.slider(sl, 'T（块长 ctx）', 4, 16, 1, st.T, v => { st.T = v; resize(); draw(); });
    F.slider(sl, 'batch（几个样本）', 1, 4, 1, st.B, v => { st.B = v; resize(); draw(); });

    function resize() {
      const h = canvasH();
      cv.height = h * 2; cv.style.height = h + 'px';
    }

    /* ---- 纯函数：第 t 帧的完整状态（从头重算） ---- */
    function stepAt(t) {
      const T = st.T, B = st.B;
      const stride = st.strideMode === 'T' ? T : 1;
      const maxStart = stream.length - T - 1;
      const base = ((t % (maxStart + 1)) + (maxStart + 1)) % (maxStart + 1);
      const win = batchWindows(stream, T, B, stride, base);
      const flat = st.wrong === 'flat' ? flatShiftY(win) : null;
      return { win, flat, copy: copyAcc(win.xs, win.ys), maxStart, stride, base };
    }

    /* ---- 绘制 ---- */
    function draw() {
      const Hpx = canvasH();
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, cv.width, cv.height);
      g.scale(2, 2);
      const s = stepAt(st.t);
      const { win } = s;
      const T = st.T, B = st.B, N = stream.length;
      const cw = (Wpx - 20) / N, ch = 20, y0 = 24;

      /* 流：64 个格子 */
      g.font = '10px ui-monospace,Consolas';
      g.fillStyle = DIMC; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.fillText('data/corpus.txt 前 64 个字符（· = 空格，⏎ = 换行）', 10, 14);
      for (let i = 0; i < N; i++) {
        const x = 10 + i * cw;
        g.fillStyle = '#1b2330';
        g.fillRect(x, y0, cw - 1, ch);
        g.fillStyle = stream[i] === ' ' ? '#6d7f96' : FGC;
        g.textAlign = 'center';
        g.fillText(dispCh(stream[i]), x + (cw - 1) / 2, y0 + 14);
      }
      g.fillStyle = '#54617a'; g.textAlign = 'center';
      for (let i = 0; i <= N; i += 8) {
        const x = 10 + Math.min(i, N - 1) * cw + (i === N ? cw : 0);
        g.fillText(String(i), clamp(x, 14, Wpx - 14), y0 + ch + 11);
      }
      /* 每个样本的窗口：x 段底色 + y 多出的那一格黄框 */
      for (let b = 0; b < B; b++) {
        const st0 = win.starts[b], col = PAL[b % PAL.length];
        g.save();
        g.globalAlpha = 0.16; g.fillStyle = col;
        g.fillRect(10 + st0 * cw, y0 - 3, T * cw - 1, ch + 6);
        g.restore();
        g.strokeStyle = col; g.lineWidth = 1.4;
        g.strokeRect(10 + st0 * cw - 0.5, y0 - 3.5, T * cw, ch + 7);
        if (st0 + T < N) {
          g.save();
          g.strokeStyle = WARNC; g.setLineDash([3, 2]); g.lineWidth = 1.6;
          g.strokeRect(10 + (st0 + T) * cw - 0.5, y0 - 3.5, cw, ch + 7);
          g.restore();
        }
      }

      /* 样本块 */
      const csW = Math.min(40, (Wpx - 150) / T), chh = 24, x0 = 128;
      for (let b = 0; b < B; b++) {
        const top = TOP0 + b * BLK;
        const xY = top + 14, yY = top + 60;
        const st0 = win.starts[b], col = PAL[b % PAL.length];
        g.font = '11px ui-monospace,Consolas'; g.textAlign = 'left';
        g.fillStyle = col;
        g.fillText(`样本 ${b} · 起点 s=${st0}`, 10, top + 10);
        g.fillStyle = DIMC;
        g.fillText('x', x0 - 16, xY + 16);
        g.fillText(st.wrong === 'yx' ? 'y=x' : 'y', x0 - 16, yY + 16);
        if (st.wrong === 'yx') { g.fillStyle = BADC; g.fillText('（错：不错位）', x0 - 16 + 26, yY + 16); }

        const yRow = st.wrong === 'yx' ? win.xs[b]
          : st.wrong === 'flat' ? s.flat.yb[b]
            : win.ys[b];
        for (let k = 0; k < T; k++) {
          const x = x0 + k * csW;
          /* x 行 */
          g.fillStyle = '#22384f';
          g.fillRect(x, xY, csW - 2, chh);
          g.strokeStyle = col; g.lineWidth = 1;
          g.strokeRect(x + 0.5, xY + 0.5, csW - 3, chh - 1);
          g.fillStyle = FGC; g.textAlign = 'center'; g.font = '12px ui-monospace,Consolas';
          g.fillText(dispCh(win.xs[b][k]), x + (csW - 2) / 2, xY + 16);
          /* y 行 */
          const leaked = st.wrong === 'flat' && k === T - 1 && s.flat.mism.some(mm => mm[0] === b && mm[1] === k);
          const extra = st.wrong === 'none' && k === T - 1;
          g.fillStyle = leaked ? '#5c2525' : st.wrong === 'yx' ? '#4a2530' : extra ? '#4a3a1d' : '#1d3326';
          g.fillRect(x, yY, csW - 2, chh);
          g.strokeStyle = leaked || st.wrong === 'yx' ? BADC : extra ? WARNC : OKC;
          g.lineWidth = extra || leaked ? 1.8 : 1;
          g.strokeRect(x + 0.5, yY + 0.5, csW - 3, chh - 1);
          g.fillStyle = FGC;
          g.fillText(dispCh(yRow[k]), x + (csW - 2) / 2, yY + 16);
          /* 箭头 x[k] → y[k]：同一列，但内容差一格 */
          const cx = x + (csW - 2) / 2;
          arrow(g, cx, xY + chh + 1, cx, yY - 3,
            leaked || st.wrong === 'yx' ? BADC : extra ? WARNC : '#8d99ac', null, 1.4);
        }
        /* 摊平错法：把"串味"的那一格连到隔壁样本的第 0 格 */
        if (st.wrong === 'flat' && B >= 1) {
          const nb = (b + 1) % B;
          const lx = x0 + (T - 1) * csW + (csW - 2) / 2, ly = yY + chh;
          const tx = x0 + (csW - 2) / 2, ty = TOP0 + nb * BLK + 14;
          g.save();
          g.strokeStyle = BADC; g.setLineDash([4, 3]); g.lineWidth = 1.6;
          g.beginPath();
          g.moveTo(lx, ly);
          g.bezierCurveTo(lx + 70, ly + 26, tx + 90, ty - 30, tx, ty - 4);
          g.stroke();
          g.restore();
          arrow(g, tx + 6, ty - 14, tx, ty - 3, BADC, null, 1.4);
          g.fillStyle = BADC; g.font = '10px ui-monospace,Consolas'; g.textAlign = 'left';
          g.fillText(nb === 0 && b === B - 1 ? '越界回绕到样本 0 的第 0 格' : `抓到样本 ${nb} 的第 0 格`,
            Math.min(lx + 8, Wpx - 175), ly + 16);
        }
        /* 正确模式下标出 y 最后一格的来源 */
        if (st.wrong === 'none') {
          g.fillStyle = WARNC; g.font = '10px ui-monospace,Consolas'; g.textAlign = 'left';
          g.fillText(`← 流[${st0 + T}]，不在 x 里（窗口第 T+1 格）`,
            Math.min(x0 + T * csW + 8, Wpx - 215), yY + 16);
        }
      }

      /* ---- readout ---- */
      const lines = [];
      lines.push(`<span>N=${N} · T=${T} · batch=${B} · stride=${s.stride} · <b>max_start = N−T−1 = ${win.maxStart}</b>（合法起点 ${win.maxStart + 1} 个，当前 base=${s.base}）</span>`);
      for (let b = 0; b < B; b++) {
        const st0 = win.starts[b];
        lines.push(`<span>样本${b} s=${st0}：x=<code>${showStr(win.xs[b])}</code>　y=<code>${showStr(st.wrong === 'yx' ? win.xs[b] : st.wrong === 'flat' ? s.flat.yb[b] : win.ys[b])}</code></span>`);
        lines.push(`<span>　第 0 步预测流的第 ${st0 + 1} 个 <code>${showStr(stream[st0 + 1])}</code>；第 ${T - 1} 步预测流的第 ${st0 + T} 个 <code>${showStr(stream[st0 + T])}</code>（模型在第 k 步预测的其实是流的第 ${st0}+k+1 个字符）</span>`);
      }
      lines.push(`<span>复读机命中率（拿 x[k] 当 y[k] 的答案）：<b>${(100 * s.copy).toFixed(1)}%</b>（当前窗口真算；正确答案在 x 的下一格，抄输入几乎全错）</span>`);
      lines.push(`<span>末尾不够一块怎么办：<b>不补齐也不丢弃</b> —— Batcher 的起点上界是 max_start=${win.maxStart}，窗口恒为完整 T+1=${T + 1} 格；最后 ${T} 个字符当不了起点，但仍作为更早窗口的目标被用到</span>`);
      if (st.wrong === 'yx') {
        lines.push(`<span style="color:${BADC}">错法①：y=x 时复读机命中率恒 <b>100%</b> —— 最优解就是恒等复制。训练 loss 会掉到 0.001 量级，但那是抄输入抄出来的，学不到任何语言；判据 <b>移位关系 y[:, :-1] == x[:, 1:]</b> 与 <b>随机数据学不动（反作弊判据）</b> 双双变红（随机数据上 loss 也照样掉，因为它还是在抄）</span>`);
      } else if (st.wrong === 'flat') {
        lines.push(`<span style="color:${BADC}">错法②：摊平后错位，串味格 <b>${s.flat.mism.length}</b> 个（真算）：${s.flat.mism.map(mm => `样本${mm[0]}的y[${mm[1]}]应为<code>${showStr(stream[win.starts[mm[0]] + T])}</code>，实际是<code>${showStr(s.flat.yb[mm[0]][mm[1]])}</code>`).join('；')}</span>`);
        lines.push(`<span style="color:${BADC}">阴险之处：判据 <b>移位关系 y[:, :-1] == x[:, 1:]</b> 只查行内列，这个错法<b>不会</b>让它变红 —— 只有每行最后一格跨界。所以参考答案是按窗口切 win[:, 1:]，绝不摊平整条流再错位</span>`);
      }
      ro.innerHTML = lines.join('');
      note.textContent = st.wrong === 'none'
        ? `播放 = 窗口沿流一格一格滑（base ${st.t}/${win.maxStart}）。黄色格/黄框都是同一件事：y 的最后一格来自流的下一格，x 里没有它 —— 这就是窗口取 T+1 的原因。`
        : '错法对照开着：红色格 = 和正确答案不同的格。切回"正确"看真实行为。';
    }

    const pb = playback(box, () => stream.length - st.T - 1, t => { st.t = t; draw(); }, 140);
    syncBtns();
    draw();
  };

  /* ================================================================ 2. clipnorm
     clip_grad_norm 的全部要点：total 是跨张量的全局 L2（sqrt(Σ‖g_i‖²)），
     超阈值时**所有**张量乘同一个 scale = max_norm/total —— 每个方向一点不动。
     6 个"参数张量"各压成一个 2D 向量：2D 只是让"方向不变"看得见的示意维度，
     范数、scale、夹角全部按这些数真算，公式与高维一字不差。 */

  const CN = { n: 6, seed: 4242 };
  CN.dir = (function () {                             // 固定随机方向（写死种子）
    const r = mulberry32(CN.seed), out = [];
    for (let i = 0; i < CN.n; i++) {
      const a = r() * 2 * Math.PI;
      out.push([Math.cos(a), Math.sin(a)]);
    }
    return out;
  })();
  CN.mags0 = (function () {
    const r = mulberry32(CN.seed + 1), out = [];
    for (let i = 0; i < CN.n; i++) out.push(Math.round((0.35 + r() * 2.1) * 100) / 100);
    return out;
  })();
  CN.names = ['embed', 'blk0.attn', 'blk0.ffn', 'blk1.attn', 'blk1.ffn', 'ln/head'];

  R.clipnorm = function (box) {
    box.classList.add('lane-f');
    F.title(box, 'clip_grad_norm：整体等比缩，方向一点不动 —— total = √(Σ‖g_i‖²)，同一个 scale = max_norm/total');
    F.caption(box,
      '6 个"参数张量"的梯度各压成一个 2D 向量（<b>2D 只是示意维度</b>，为的是让"方向不变"看得见；' +
      '真实梯度是上万维，但公式一字不差）。每个张量一根长度滑杆 = ‖g_i‖，方向固定（种子写死）。' +
      '全局范数 `total = sqrt(Σ ‖g_i‖²)` 真算；若 `total > max_norm` <b>且</b> `total > 0`，所有张量乘<b>同一个</b> ' +
      '`scale = max_norm / total` —— 参考实现里<b>没有 eps</b>，防除零靠的就是这两个条件（total=0 直接不进分支），' +
      '返回值是裁剪<b>前</b>的 total。中间面板：虚线箭头 = 裁前，实线 = 裁后；右边每层长度条同步缩。' +
      '重点看 readout 里每个张量裁前/裁后的<b>夹角（真算 arccos）</b>：无论怎么拖，都是 0.000° —— 裁剪只缩步幅，不拐方向。' +
      '三个错法开关各自对应 `stages/s05_char_lm.py` 里的一条判据。');

    const st = { mags: CN.mags0.slice(), maxNorm: 1.0, mode: 'ok' };

    const Wpx = Math.max(420, Math.min(880, (box.clientWidth || 880) - 16));
    const Hpx = 400;
    const cvh = el('div'); box.appendChild(cvh);
    const { cv, g } = mkCanvas(cvh, Wpx, Hpx);
    const ro = F.readout(box);
    const note = el('div', 'lane-f-note'); box.appendChild(note);

    /* ---- 控件 ---- */
    const btnRow = el('div', 'lane-f-btns'); box.appendChild(btnRow);
    const modes = [
      ['ok', '正确（全局同一个 scale）'],
      ['pertensor', '① 逐张量各自裁'],
      ['order', '② 先裁再算 total（返回值错）'],
      ['uncond', '③ 无条件乘 scale']
    ];
    const modeBtns = modes.map(([m, txt]) => {
      const b = el('button', st.mode === m ? 'on' : '', txt);
      b.onclick = () => { st.mode = m; modeBtns.forEach((bb, i) => bb.classList.toggle('on', modes[i][0] === m)); draw(); };
      btnRow.appendChild(b); return b;
    });

    const sl = F.sliders(box);
    for (let i = 0; i < CN.n; i++) {
      F.slider(sl, `‖g${i}‖ ${CN.names[i]}`, 0, 3, 0.01, st.mags[i], v => { st.mags[i] = v; draw(); });
    }
    F.slider(sl, 'max_norm', 0.1, 4, 0.05, st.maxNorm, v => { st.maxNorm = v; draw(); });

    function gsOf() {
      return CN.dir.map((d, i) => [st.mags[i] * d[0], st.mags[i] * d[1]]);
    }

    /* ---- 绘制 ---- */
    function draw() {
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.scale(2, 2);
      const gs = gsOf();
      const r = clipApply(gs, st.maxNorm, st.mode);
      const maxMag = Math.max.apply(null, r.mags.concat([1e-9]));
      const maxVal = Math.max(r.total, r.after, st.maxNorm) * 1.15;

      /* Zone A：总范数条 + 阈值线 */
      const baseY = Hpx - 46, unit1 = (Hpx - 120) / maxVal;
      g.font = '10px ui-monospace,Consolas'; g.textAlign = 'center';
      g.fillStyle = DIMC;
      g.fillText('全局 total', 62, 20);
      g.strokeStyle = '#28313f'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(14, baseY); g.lineTo(118, baseY); g.stroke();
      /* 裁前（空心虚线） */
      g.save();
      g.setLineDash([4, 3]); g.strokeStyle = BADC; g.lineWidth = 1.6;
      g.strokeRect(20.5, baseY - r.total * unit1, 34, r.total * unit1);
      g.restore();
      /* 裁后（实心） */
      g.fillStyle = r.after > st.maxNorm + 1e-12 ? BADC : OKC;
      g.fillRect(64, baseY - r.after * unit1, 34, r.after * unit1);
      /* 阈值线（只画在总范数条这一区，别伸进箭头区被误读成半径） */
      g.save();
      g.strokeStyle = WARNC; g.setLineDash([5, 3]); g.lineWidth = 1.6;
      const thY = baseY - st.maxNorm * unit1;
      g.beginPath(); g.moveTo(12, thY); g.lineTo(120, thY); g.stroke();
      g.restore();
      g.fillStyle = WARNC; g.textAlign = 'left';
      g.fillText(`max_norm=${st.maxNorm.toFixed(2)}`, 124, thY + 4);
      /* 数值写在各自条的顶上，避免两个标签挤在一起 */
      g.fillStyle = DIMC; g.textAlign = 'center';
      g.fillText(`前 ${r.total.toFixed(3)}`, 37, baseY - r.total * unit1 - 5);
      g.fillText(`后 ${r.after.toFixed(3)}`, 81, baseY - r.after * unit1 - 5);

      /* Zone B：2D 箭头（裁前虚线、裁后实线叠画） */
      const zx0 = 150, zx1 = Wpx - 250;
      const ox = (zx0 + zx1) / 2, oy = (Hpx + 20) / 2;
      const rad = Math.min((zx1 - zx0) / 2 - 20, Hpx / 2 - 52);
      const unit2 = rad / (maxMag * 1.18);
      g.strokeStyle = '#28313f'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(ox - rad - 12, oy); g.lineTo(ox + rad + 12, oy); g.stroke();
      g.beginPath(); g.moveTo(ox, oy - rad - 12); g.lineTo(ox, oy + rad + 12); g.stroke();
      for (let i = 0; i < CN.n; i++) {
        const col = PAL[i % PAL.length];
        const px = gs[i][0] * unit2, py = -gs[i][1] * unit2;
        const qx = r.out[i][0] * unit2, qy = -r.out[i][1] * unit2;
        arrow(g, ox, oy, ox + px, oy + py, col, [4, 3], 1.3);          // 裁前
        arrow(g, ox, oy, ox + qx, oy + qy, col, null, 2.2);             // 裁后（没裁时与虚线重合）
        g.fillStyle = col; g.font = '10px ui-monospace,Consolas'; g.textAlign = 'left';
        g.fillText(`g${i}`, ox + px + (px >= 0 ? 5 : -20), oy + py + (py >= 0 ? 12 : -5));
      }
      g.fillStyle = DIMC; g.textAlign = 'center';
      g.fillText('虚线 = 裁前 · 实线 = 裁后（等比缩，夹角 = 0）', (zx0 + zx1) / 2, Hpx - 10);

      /* Zone C：每层长度条 */
      const cx0 = Wpx - 236, barMax = 168, unit3 = barMax / (maxMag * 1.18);
      g.textAlign = 'left';
      for (let i = 0; i < CN.n; i++) {
        const col = PAL[i % PAL.length];
        const y = 30 + i * 56;
        g.fillStyle = DIMC; g.font = '10px ui-monospace,Consolas';
        g.fillText(`g${i} ${CN.names[i]}`, cx0, y);
        const mw = r.mags[i] * unit3, aw = norm2(r.out[i]) * unit3;
        g.save();
        g.setLineDash([3, 2]); g.strokeStyle = col; g.lineWidth = 1.2;
        g.strokeRect(cx0 + 0.5, y + 6.5, Math.max(mw, 1), 11);
        g.restore();
        g.fillStyle = col;
        g.fillRect(cx0, y + 6, Math.max(aw, 1), 12);
        g.fillStyle = FGC;
        g.fillText(`${r.mags[i].toFixed(2)} → ${norm2(r.out[i]).toFixed(2)}（×${r.scales[i].toFixed(3)}）`, cx0, y + 32);
      }

      /* ---- readout ---- */
      const clipped = r.total > st.maxNorm && r.total > 0;
      const lines = [];
      lines.push(`<span>total（裁前，真算 √(Σ‖g_i‖²)）= <b>${r.total.toFixed(6)}</b>　裁剪后全局范数 = <b>${r.after.toFixed(6)}</b></span>`);
      lines.push(`<span>scale = <b>${r.scale === 1 ? '1（没超上限，一点都不动）' : r.scale.toFixed(6)}</b>${clipped ? `（= max_norm/total = ${st.maxNorm}/${r.total.toFixed(4)}）` : ''}</span>`);
      lines.push(`<span>函数返回值 = <b style="color:${st.mode === 'order' ? BADC : FGC}">${r.returned.toFixed(6)}</b>（契约：裁剪<b>前</b>的范数，训练日志靠它看尖刺）</span>`);
      lines.push(`<span>每个张量裁前/裁后的夹角（真算 arccos）：${r.angles.map((a, i) => `g${i} <b style="color:${a < 1e-9 ? OKC : BADC}">${a.toFixed(3)}°</b>`).join('　')}</span>`);
      if (st.mode === 'ok') {
        lines.push(`<span>四条判据全绿的口径：<b>返回裁前范数</b> / <b>裁后全局范数 = 上限</b>（${r.after.toFixed(6)}${clipped ? ' = max_norm' : ' ≤ max_norm，未裁'}）/ <b>所有张量同一个系数</b> / <b>没超上限时一点都不动</b></span>`);
      } else if (st.mode === 'pertensor') {
        lines.push(`<span style="color:${BADC}">错法①逐张量各自裁：每层 scale 不同（${r.scales.map((s, i) => `g${i}×${s.toFixed(3)}`).join(' ')}），裁后全局范数 = <b>${r.after.toFixed(4)}</b>，仍超上限 <b>${Math.max(0, r.after - st.maxNorm).toFixed(4)}</b> —— 判据 <b>裁剪后全局范数 = 上限</b> 与 <b>裁剪对所有张量乘同一个系数</b> 变红。方向倒是没拐（夹角全 0），错的是各层步长配比：参数量大的层被偷偷限了相对权重</span>`);
      } else if (st.mode === 'order') {
        lines.push(`<span style="color:${BADC}">错法②先裁再算 total：梯度本身和正确写法一样，但返回的是裁后范数 <b>${r.returned.toFixed(6)}</b> 而不是裁前 <b>${r.total.toFixed(6)}</b> —— 判据 <b>clip_grad_norm 返回裁剪前的范数</b> 变红。症状很安静：日志里 grad_norm 永远 ≤ max_norm，发散的尖刺你永远看不见</span>`);
      } else {
        if (clipped) {
          lines.push(`<span>错法③在 total &gt; max_norm 时和正确写法数值相同 —— 它只在 total ≤ max_norm 时露馅，把 max_norm 拖大（或把 ‖g_i‖ 拖小）看下面那行</span>`);
        } else {
          lines.push(`<span style="color:${BADC}">错法③无条件乘 scale：total=${r.total.toFixed(4)} ≤ max_norm=${st.maxNorm.toFixed(2)}，本该一动不动，却被放大 ×<b>${r.scale.toFixed(4)}</b>（裁后 ${r.after.toFixed(4)}）—— 判据 <b>范数没超上限时一点都不动</b> 变红。小梯度被硬拉到上限，训练在小梯度段变成恒定大步长，收敛直接被毁</span>`);
        }
      }
      ro.innerHTML = lines.join('');
      note.textContent = '方向固定、长度可拖：total 是所有张量拼起来的 L2（先扫一遍求和开根号，再扫一遍乘同一个 scale）。' +
        '把任一 ‖g_i‖ 拖到 0 试试：total 变小但不为 0 时照常裁；全为 0 时 total=0，条件 total > 0 挡住 0/0，一点都不动。';
    }

    draw();
  };

  /* ================================================================ 3. samplewheel
     sample_next 的最后一步：给定概率分布 p，rng.choice(V, p) 怎么"抽"出一个词。
     温度→softmax 现算 p（口径 = sampler.py 的 _softmax(z/T)）；转盘 = 累积分布，
     扇区面积 = 概率；每次抽一个 u ∈ [0,1)，落进"第一个满足 u < cdf[k] 的区间"——
     与 numpy Generator.choice 的 cdf.searchsorted(u, 'right') 逐位一致。
     温度/top-k/top-p 流水线见 figs.js 的 sampling 组件，这里不重复。 */

  const SW = {
    Z: [2.0, 1.5, 1.0, 0.5, 0.0, -0.5],       // 固定 logits（词表 V=6 的假想切片）
    WORDS: ['e', '␣', 't', 'h', 's', '⏎'],     // 显示用：字符级词表
    SEED: 987654321
  };

  R.samplewheel = function (box) {
    box.classList.add('lane-f');
    F.title(box, 'sample_next 的最后一步：`rng.choice(V, p)` 到底怎么"抽"出一个词');
    F.caption(box,
      '温度/top-k/top-p 那条流水线在 §2.6 的图里已经画过，这张只讲<b>最后一下</b>：拿到概率 p 之后怎么抽。' +
      '固定 V=6 的字符 logits（`[2.0, 1.5, 1.0, 0.5, 0.0, -0.5]`），温度滑杆现算 `p = softmax(z/T)`（口径同 sampler.py：减最大值再 exp）。' +
      '左边是<b>累积分布转盘</b>：每个扇区面积 = 概率，边界 = 累积区间。点"抽一次"：带种子的 rng 吐一个 u ∈ [0,1)，' +
      '落进"第一个满足 u < cdf[k] 的区间"就选中词 k —— 这和 numpy `Generator.choice` 内部的 `cdf.searchsorted(u, side=\'right\')` 逐位一致。' +
      '右边直方图累积经验频率，白色横线 = 理论 p，竖须 = ±2σ = ±2·√(p(1−p)/N)（真算）：' +
      '<b>抽 100 次差 0.05 这个量级是正常的抽样波动，不是你的代码错了</b>；右上角小图是 1/√N 律的包络曲线与当前 N 的位置。');

    const st = { T: 1.0, nPer: 10, drawn: 0, mode: 'ok' };

    const side = el('div', 'lane-f-side'); box.appendChild(side);
    const wheelCol = el('div'); side.appendChild(wheelCol);
    const { cv: wcv, g: wg } = mkCanvas(wheelCol, 320, 330);
    const histCol = el('div'); histCol.style.flex = '1 1 340px'; histCol.style.minWidth = '0'; side.appendChild(histCol);
    const HW = Math.max(320, Math.min(520, (box.clientWidth || 880) - 370));
    const { cv: hcv, g: hg } = mkCanvas(histCol, HW, 330);

    const ro = F.readout(box);
    const note = el('div', 'lane-f-note'); box.appendChild(note);

    /* ---- 控件 ---- */
    const btnRow = el('div', 'lane-f-btns'); box.appendChild(btnRow);
    const bDraw = el('button', '', '🎲 抽一次');
    const bReset = el('button', '', '↺ 重置');
    const labN = el('label');
    labN.innerHTML = '<b>每次抽 N 个</b>';
    const selN = el('select');
    for (const v of [1, 10, 100, 1000]) {
      const op = el('option'); op.value = v; op.textContent = String(v);
      if (v === 10) op.selected = true;
      selN.appendChild(op);
    }
    selN.onchange = () => { st.nPer = +selN.value; };
    labN.append(selN);
    btnRow.append(bDraw, bReset, labN);

    const modeRow = el('div', 'lane-f-btns'); box.appendChild(modeRow);
    const modeLab = el('span'); modeLab.style.cssText = 'font:11.5px/1 var(--mono);color:var(--dim)';
    modeLab.textContent = '错法对照：';
    modeRow.appendChild(modeLab);
    const modes = [
      ['ok', '正确（累积区间）'],
      ['nocum', '① 区间判错：拿 u 和单个 p[k] 比'],
      ['unnorm', '② p 没归一（Σp < 1）'],
      ['argmax', '③ 用 argmax 代替抽样']
    ];
    const modeBtns = modes.map(([m, txt]) => {
      const b = el('button', st.mode === m ? 'on' : '', txt);
      b.onclick = () => { st.mode = m; modeBtns.forEach((bb, i) => bb.classList.toggle('on', modes[i][0] === m)); draw(); };
      modeRow.appendChild(b); return b;
    });

    const sl = F.sliders(box);
    F.slider(sl, '温度 T', 0.2, 3, 0.05, st.T, v => { st.T = v; draw(); });

    bDraw.onclick = () => { st.drawn += st.nPer; draw(); };
    bReset.onclick = () => { st.drawn = 0; draw(); };

    function pOf() { return softmaxT(SW.Z, st.T); }

    /* ---- 绘制 ---- */
    function draw() {
      const p = pOf();
      const run = sampleRun(p, st.drawn, SW.SEED, st.mode);
      drawWheel(p, run);
      drawHist(p, run);

      /* ---- readout ---- */
      const V = p.length;
      const N = st.drawn;
      const lines = [];
      let S = 0;
      for (const q of p) S += q;
      lines.push(`<span>T=${st.T.toFixed(2)} · Σp=<b>${S.toFixed(6)}</b> · 种子=${SW.SEED} · 已抽 N=<b>${N}</b></span>`);
      for (let k = 0; k < V; k++) {
        const f = N ? run.counts[k] / N : 0;
        lines.push(`<span><code>${SW.WORDS[k]}</code>：p=<b>${p[k].toFixed(4)}</b>　频率=${N ? f.toFixed(4) : '—'}　差=${N ? (f - p[k] >= 0 ? '+' : '') + (f - p[k]).toFixed(4) : '—'}</span>`);
      }
      if (N > 0) {
        let maxd = 0, max2s = 0;
        for (let k = 0; k < V; k++) {
          maxd = Math.max(maxd, Math.abs(run.counts[k] / N - p[k]));
          max2s = Math.max(max2s, 2 * Math.sqrt(p[k] * (1 - p[k]) / N));
        }
        const p0 = Math.max.apply(null, p);
        const s100 = 2 * Math.sqrt(p0 * (1 - p0) / 100);
        lines.push(`<span>max|频率−p| = <b>${maxd.toFixed(4)}</b>，当前 N 下 ±2σ 包络 = ±<b>${max2s.toFixed(4)}</b>（真算 2·√(p(1−p)/N)）—— 差值在包络内就是正常波动；N=100 时包络 ±${s100.toFixed(3)}，所以"抽 100 次差 0.05"完全正常</span>`);
      } else {
        lines.push(`<span>还没抽：点 🎲 抽一次（每次 ${st.nPer} 个）。同一种子下抽多少次都逐位可复现 —— 状态就是 (p, N, seed, mode)，直方图每次从头重放</span>`);
      }
      /* 错法说明：各配一个 2000 次的纯函数实测，不改用户当前状态 */
      if (st.mode === 'nocum') {
        const demo = sampleRun(p, 2000, SW.SEED, 'nocum');
        lines.push(`<span style="color:${BADC}">错法①：判区间时拿 u 和<b>单个</b> p[k] 比而不是累积上限 —— u ≥ p[0] 之后，后面的词永远轮不到（p 递减，u<p[k] 更不可能），全落空就兜底最后一个词。2000 次实测：${demo.counts.map((c, k) => `<code>${SW.WORDS[k]}</code>=${c}`).join(' ')} —— 中间几个词频率恒为 0，最后一个词吞掉所有剩余质量。判据 <b>4000 次抽样经验分布与 softmax 的 L1 ≤ 0.14</b> 直接爆掉。（顺带说清：单纯把边界的 &gt; 写成 &gt;= 对连续 u 没有任何影响 —— u 恰好砸在边界点上的概率是 0；真能让某个词频率恒为 0 的，是拿 u 跟错的量比，也就是这里演示的"没累积"）</span>`);
      } else if (st.mode === 'unnorm') {
        const demo = sampleRun(p, 2000, SW.SEED, 'unnorm');
        let S2 = 0;
        for (let k = 0; k < V - 1; k++) S2 += p[k];
        lines.push(`<span style="color:${BADC}">错法②：p 被砍掉最后一个词却没把质量摊回去（top-p 之后忘了重新归一就是它）：Σp=<b>${S2.toFixed(4)}</b> &lt; 1，转盘出现宽 ${(1 - S2).toFixed(4)} 的灰色缺口，<code>${SW.WORDS[V - 1]}</code> 的区间宽度为 0、永远抽不到。2000 次实测缺口命中 ${demo.gaps} 次 —— 真代码里 numpy 的 rng.choice 会直接抛 ValueError（p 的和必须为 1），这里演示的是它若静默跑会怎样</span>`);
      } else if (st.mode === 'argmax') {
        lines.push(`<span style="color:${BADC}">错法③：用 argmax 代替抽样，直方图只剩一根柱子（每次都返回 <code>${SW.WORDS[run.lastIdx]}</code>，rng 根本不被碰）。这就是 temperature=0 的贪心岔口本身 —— 在 sample_next 里它是<b>合法出口</b>（temperature ≤ 0 → int(np.argmax(z))，判据"temperature=0 时两个不同种子输出完全相同"钉的就是它）；但 T&gt;0 时写成 argmax 等于把随机性杀了，生成永远复读同一条路径</span>`);
      }
      ro.innerHTML = lines.join('');
      note.textContent = N > 0 && st.mode === 'ok' && run.lastU != null
        ? `最近一次：u = ${run.lastU.toFixed(6)}，落进 [${(run.lastIdx > 0 ? p.slice(0, run.lastIdx).reduce((a, b) => a + b, 0) : 0).toFixed(4)}, ${(run.lastIdx != null && run.lastIdx < V ? p.slice(0, run.lastIdx + 1).reduce((a, b) => a + b, 0) : 1).toFixed(4)}) → 选中 "${SW.WORDS[run.lastIdx] ?? '?'}"（转盘上的白点就是它）`
        : '转盘扇区面积 = 概率；拖温度看分布变平/变尖，直方图会用同一种子从头重放，频率随 p 连续变化。';
    }

    function drawWheel(p, run) {
      wg.setTransform(1, 0, 0, 1, 0, 0); wg.clearRect(0, 0, wcv.width, wcv.height); wg.scale(2, 2);
      const V = p.length, cx = 160, cy = 158, r = 116;
      const A0 = -Math.PI / 2;
      /* 扇区边界：mode='unnorm' 时最后一个词概率清零、不归一 → 有缺口 */
      const q = p.slice();
      if (st.mode === 'unnorm') q[V - 1] = 0;
      let S = 0;
      for (const v of q) S += v;
      const cum = [];
      let acc = 0;
      for (const v of q) { acc += v; cum.push(acc); }
      for (let k = 0; k < V; k++) {
        const lo = (k ? cum[k - 1] : 0) * 2 * Math.PI + A0;
        const hi = cum[k] * 2 * Math.PI + A0;
        if (hi - lo < 1e-9) continue;
        const hit = run.lastIdx === k;
        wg.beginPath();
        wg.moveTo(cx, cy);
        wg.arc(cx, cy, r, lo, hi);
        wg.closePath();
        wg.save();
        wg.globalAlpha = hit ? 0.95 : 0.62;
        wg.fillStyle = PAL[k % PAL.length];
        wg.fill();
        wg.restore();
        wg.strokeStyle = hit ? '#ffffff' : 'rgba(10,14,22,0.7)';
        wg.lineWidth = hit ? 3 : 1;
        wg.stroke();
        /* 标签：词 + p 值 */
        const mid = (lo + hi) / 2;
        const lr = r * (p[k] > 0.06 ? 0.62 : 0.86);
        wg.fillStyle = p[k] > 0.28 ? '#0d131d' : FGC;
        wg.font = '700 13px ui-monospace,Consolas';
        wg.textAlign = 'center'; wg.textBaseline = 'middle';
        wg.fillText(SW.WORDS[k], cx + lr * Math.cos(mid), cy + lr * Math.sin(mid) - 6);
        wg.font = '9.5px ui-monospace,Consolas';
        wg.fillText((st.mode === 'unnorm' && k === V - 1 ? '0' : q[k].toFixed(3)), cx + lr * Math.cos(mid), cy + lr * Math.sin(mid) + 7);
        /* 累积边界刻度 */
        wg.fillStyle = DIMC; wg.font = '9px ui-monospace,Consolas';
        wg.fillText(cum[k].toFixed(2), cx + (r + 14) * Math.cos(hi), cy + (r + 14) * Math.sin(hi));
      }
      /* 缺口（Σp < 1）：灰色扇区，u 落这里 = numpy 抛 ValueError 的位置 */
      if (S < 1 - 1e-9) {
        const lo = S * 2 * Math.PI + A0, hi = 2 * Math.PI + A0;
        wg.beginPath(); wg.moveTo(cx, cy); wg.arc(cx, cy, r, lo, hi); wg.closePath();
        wg.fillStyle = 'rgba(141,153,172,0.18)'; wg.fill();
        wg.save(); wg.setLineDash([4, 3]); wg.strokeStyle = DIMC; wg.lineWidth = 1.4; wg.stroke(); wg.restore();
        const mid = (lo + hi) / 2;
        wg.fillStyle = DIMC; wg.font = '9.5px ui-monospace,Consolas';
        wg.fillText('缺口', cx + r * 0.6 * Math.cos(mid), cy + r * 0.6 * Math.sin(mid) - 5);
        wg.fillText(`宽 ${(1 - S).toFixed(3)}`, cx + r * 0.6 * Math.cos(mid), cy + r * 0.6 * Math.sin(mid) + 7);
      }
      /* 最近一次的 u：白点 + 半径线 */
      if (run.lastU != null && st.mode !== 'argmax') {
        const a = run.lastU * 2 * Math.PI + A0;
        const inGap = run.lastIdx >= V;
        wg.strokeStyle = inGap ? BADC : 'rgba(255,255,255,0.85)';
        wg.lineWidth = 1.2;
        wg.beginPath(); wg.moveTo(cx, cy); wg.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)); wg.stroke();
        wg.fillStyle = inGap ? BADC : '#ffffff';
        wg.beginPath(); wg.arc(cx + r * 0.92 * Math.cos(a), cy + r * 0.92 * Math.sin(a), 4.5, 0, 7); wg.fill();
        wg.fillStyle = inGap ? BADC : FGC; wg.font = '10px ui-monospace,Consolas'; wg.textAlign = 'center';
        wg.fillText(`u=${run.lastU.toFixed(4)}${inGap ? '（落进缺口！）' : ''}`, cx, 322);
      } else if (st.mode === 'argmax') {
        wg.fillStyle = WARNC; wg.font = '10px ui-monospace,Consolas'; wg.textAlign = 'center';
        wg.fillText('argmax：不产生 u，永远选中最大扇区', cx, 322);
      } else {
        wg.fillStyle = DIMC; wg.font = '10px ui-monospace,Consolas'; wg.textAlign = 'center';
        wg.fillText('点"抽一次"看 u 落在哪个扇区', cx, 322);
      }
      wg.fillStyle = DIMC; wg.font = '10px ui-monospace,Consolas'; wg.textAlign = 'left';
      wg.fillText('累积分布转盘：面积=概率，边界=累积值', 8, 14);
    }

    function drawHist(p, run) {
      hg.setTransform(1, 0, 0, 1, 0, 0); hg.clearRect(0, 0, hcv.width, hcv.height); hg.scale(2, 2);
      const V = p.length, N = st.drawn;
      const padL = 36, padB = 26, padT = 16;
      const plotW = HW - padL - 12, plotH = 330 - padT - padB - 78;   // 下面留 78px 给 1/√N 小图
      const freqs = p.map((_, k) => N ? run.counts[k] / N : 0);
      let ymax = 0.15;
      for (let k = 0; k < V; k++) {
        ymax = Math.max(ymax, freqs[k] * 1.15, p[k] + 2 * Math.sqrt(p[k] * (1 - p[k]) / Math.max(N, 1)) + 0.04);
      }
      const Y = v => padT + plotH - (v / ymax) * plotH;
      const slot = plotW / V, bw = slot * 0.52;
      hg.strokeStyle = '#28313f'; hg.lineWidth = 1;
      hg.beginPath(); hg.moveTo(padL, padT); hg.lineTo(padL, padT + plotH); hg.lineTo(padL + plotW, padT + plotH); hg.stroke();
      hg.fillStyle = DIMC; hg.font = '9px ui-monospace,Consolas'; hg.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const v = ymax * i / 4;
        hg.fillText(v.toFixed(2), padL - 4, Y(v) + 3);
        hg.save(); hg.strokeStyle = '#1c2534'; hg.beginPath(); hg.moveTo(padL, Y(v)); hg.lineTo(padL + plotW, Y(v)); hg.stroke(); hg.restore();
      }
      for (let k = 0; k < V; k++) {
        const x = padL + k * slot + (slot - bw) / 2;
        const col = PAL[k % PAL.length];
        /* 经验频率柱 */
        if (N > 0 && freqs[k] > 0) {
          hg.fillStyle = col;
          hg.fillRect(x, Y(freqs[k]), bw, padT + plotH - Y(freqs[k]));
        } else {
          hg.save(); hg.setLineDash([3, 3]); hg.strokeStyle = col; hg.lineWidth = 1;
          hg.strokeRect(x + 0.5, padT + plotH - 3.5, bw - 1, 3);   // 0 频也留个底座
          hg.restore();
        }
        /* 理论 p：白横线；±2σ 须 */
        hg.strokeStyle = '#ffffff'; hg.lineWidth = 2;
        hg.beginPath(); hg.moveTo(x - 3, Y(p[k])); hg.lineTo(x + bw + 3, Y(p[k])); hg.stroke();
        if (N > 0) {
          const s2 = 2 * Math.sqrt(p[k] * (1 - p[k]) / N);
          hg.save(); hg.setLineDash([3, 2]); hg.strokeStyle = 'rgba(230,235,243,0.75)'; hg.lineWidth = 1.2;
          hg.beginPath(); hg.moveTo(x + bw / 2, Y(Math.max(0, p[k] - s2))); hg.lineTo(x + bw / 2, Y(p[k] + s2)); hg.stroke();
          hg.setLineDash([]);
          hg.beginPath();
          hg.moveTo(x + bw / 2 - 4, Y(Math.max(0, p[k] - s2))); hg.lineTo(x + bw / 2 + 4, Y(Math.max(0, p[k] - s2)));
          hg.moveTo(x + bw / 2 - 4, Y(p[k] + s2)); hg.lineTo(x + bw / 2 + 4, Y(p[k] + s2));
          hg.stroke();
          hg.restore();
        }
        hg.fillStyle = FGC; hg.font = '11px ui-monospace,Consolas'; hg.textAlign = 'center';
        hg.fillText(SW.WORDS[k], x + bw / 2, padT + plotH + 14);
        if (N > 0) {
          hg.fillStyle = col; hg.font = '9px ui-monospace,Consolas';
          hg.fillText(freqs[k].toFixed(3), x + bw / 2, Y(freqs[k]) - 3);
        }
      }
      hg.fillStyle = DIMC; hg.font = '10px ui-monospace,Consolas'; hg.textAlign = 'left';
      hg.fillText(`经验频率（N=${N}）· 白线=理论 p · 竖须=±2σ`, padL, 330 - 62);

      /* 1/√N 包络小图（右上角，带底色，避免和柱子/须搅在一起）：
         E(N) = 2·√(p_max(1−p_max)/N)，真算曲线 + 当前 N 的点 */
      const iw = Math.min(240, HW - padL - 20), ih = 52;
      const ix0 = padL + plotW - iw - 4, iy0 = padT + ih + 12;
      hg.fillStyle = 'rgba(16,23,34,0.92)';
      hg.fillRect(ix0 - 4, iy0 - ih - 12, iw + 8, ih + 24);
      hg.strokeStyle = '#28313f'; hg.lineWidth = 1;
      hg.strokeRect(ix0, iy0 - ih, iw, ih);
      const pmax = Math.max.apply(null, p);
      const E = n => 2 * Math.sqrt(pmax * (1 - pmax) / n);
      const lgLo = 0, lgHi = 5.3, eMax = E(10) * 1.2;
      const IX = n => ix0 + (Math.log10(Math.max(n, 1)) - lgLo) / (lgHi - lgLo) * iw;
      const IY = v => iy0 - (v / eMax) * ih;
      hg.strokeStyle = WARNC; hg.lineWidth = 1.6;
      hg.beginPath();
      for (let px = 0; px <= iw; px++) {
        const n = Math.pow(10, lgLo + px / iw * (lgHi - lgLo));
        const y = IY(E(n));
        px ? hg.lineTo(ix0 + px, y) : hg.moveTo(ix0 + px, y);
      }
      hg.stroke();
      if (N > 0) {
        hg.fillStyle = '#ffffff';
        hg.beginPath(); hg.arc(IX(N), IY(E(N)), 3.4, 0, 7); hg.fill();
      }
      hg.fillStyle = DIMC; hg.font = '9px ui-monospace,Consolas'; hg.textAlign = 'left';
      hg.fillText('1/√N 包络：±2σ = 2·√(p(1−p)/N)（p 取最大者）', ix0 + 4, iy0 - ih - 4);
      hg.textAlign = 'center';
      for (const n of [10, 100, 1000, 100000]) hg.fillText(String(n), IX(n), iy0 + 10);
    }

    draw();
  };

  /* 数值自测钩子：node 里 stub 掉 window/document 后可以直接调这些纯函数，
     与独立 python/numpy 脚本对账（切批字符串、total/scale/夹角、softmax、
     "同一串 u → 同一串 idx"）。不注册成图，页面上不会出现。 */
  R.__f_math = {
    mulberry32, CORPUS64,
    batchWindows, flatShiftY, copyAcc,
    clipApply, angDeg, CN,
    softmaxT, cdfOf, pickIdx, sampleRun, SW
  };
})();
