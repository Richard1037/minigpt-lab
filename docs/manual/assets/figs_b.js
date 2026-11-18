/* 原理动画 · 第 B 批：训练循环、优化器与检查点（阶段 5 / 6）。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和 figs.js 一样：能算的都真的算，不放示意数据；组件抛错只是"这张图没出来"。

   本批四个组件：adamw / resume / lrsched / lnv。
   共同约定：
     · 每个组件都有 ▶ 播放 / ⏭ 单步 / ⏸ 暂停 / ↺ 重置，播放用 requestAnimationFrame，
       但任意一帧的完整状态都由纯函数 stepAt(t) 从头重算出来 —— 单步不等 rAF，
       无头浏览器也能截到确定的第 t 帧；rAF 只在播放期间存在，mount 时不起任何定时器。
     · 递推式与 solutions/minigpt/optim.py 逐行对应（先解耦衰减、再更新矩、
       m̂=m/(1−β₁ᵗ)、w −= lr·m̂/(√v̂+eps)）；lr_at 也是原式（final_frac=0.1）。
     · 自带样式全部以 `.lane-b ` 前缀注入，不碰 manual.css。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  const el = F.el, fmt = F.fmt;

  /* ---------------------------------------------------------------- 公共工具 */

  /* 注入样式：只加 manual.css 里没有的布局件，选择器全部 .lane-b 前缀。 */
  (function injectStyle() {
    if (typeof document === 'undefined' || document.getElementById('lane-b-style')) return;
    const s = el('style'); s.id = 'lane-b-style';
    s.textContent = [
      '.lane-b canvas{display:block;max-width:100%;margin:6px 0 0}',
      '.lane-b .lane-b-btns{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 2px}',
      '.lane-b .lane-b-legend{display:flex;flex-wrap:wrap;gap:6px 16px;font:11.5px/1.6 var(--mono);color:var(--dim)}',
      '.lane-b .lane-b-legend i{display:inline-block;width:16px;height:0;border-top:2.4px solid currentColor;margin-right:5px;vertical-align:middle}',
      '.lane-b .lane-b-plot{position:relative}',
      '.lane-b .lane-b-vline{position:absolute;top:14px;bottom:24px;border-left:1px dashed var(--dim);opacity:.7;pointer-events:none}',
      '.lane-b .lane-b-vlab{position:absolute;top:12px;font:10px/1.3 var(--mono);color:var(--dim);white-space:nowrap;pointer-events:none}',
      '.lane-b .lane-b-note{font:11.5px/1.5 var(--mono);color:var(--dim);margin-top:4px}'
    ].join('\n');
    (document.head || document.documentElement).appendChild(s);
  })();

  /* 与自测脚本共用的确定性 rng：mulberry32 + Box-Muller（python 侧逐位对得上）。 */
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

  /* 播放控制：▶ / ⏭ / ⏸ / ↺。getMax 每次现取（总步数可能被滑杆改掉）。 */
  function playback(box, getMax, redraw, intervalMs) {
    const row = el('div', 'lane-b-btns');
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
    return { get t() { return t; }, set t(v) { t = v; redraw(t); }, stop };
  }

  /* 开关按钮（当勾选框用）：.on 是按下态。 */
  function toggle(host, label, init, fn) {
    const b = el('button', init ? 'on' : '', label);
    b.onclick = () => {
      const on = !b.classList.contains('on');
      b.classList.toggle('on', on); fn(on);
    };
    host.appendChild(b); return b;
  }

  function legend(host, items) {
    const d = el('div', 'lane-b-legend');
    d.innerHTML = items.map(([color, txt]) =>
      `<span><i style="color:${color}"></i>${F.md(txt)}</span>`).join('');
    host.appendChild(d); return d;
  }

  const C = { accent: '#4aa3ff', ok: '#3ecf8e', bad: '#ff6b6b', warn: '#f5b445',
              dim: '#8d99ac', fg: '#e6ebf3', purple: '#b9a7e0' };

  function plotW(box) { return Math.max(320, Math.min(860, (box.clientWidth || 880) - 16)); }

  /* ================================================================ 1. adamw
     一维二次目标 f(w)=½·a·w²+b·w+c 上真算 SGD / Momentum / AdamW 三条轨迹，
     叠画开关演示三件事：偏差修正、解耦衰减 vs "λw 进梯度"、bias/norm 免衰减。 */

  const AW = { a: 3, b: -2.4, c: 1, w0: -1.5, u0: -0.8, uStar: 0.5, T: 200, momMu: 0.9 };
  AW.wStar = -AW.b / AW.a;
  AW.f = w => 0.5 * AW.a * w * w + AW.b * w + AW.c;
  AW.g = w => AW.a * w + AW.b;

  /* 纯函数：给定超参，把六条递推全部算到 T，返回逐步历史。 */
  function adamwSim(p) {
    const lr = p.lr, b1 = p.b1, b2 = p.b2, wd = p.wd, eps = p.eps, T = AW.T;
    const H = { sgd: [AW.w0], mom: [AW.w0], adam: [AW.w0], nobc: [AW.w0], l2: [AW.w0] };
    const det = [null];                                   // AdamW 每步的细节（t 从 1 起）
    const uEx = [AW.u0], uDe = [AW.u0];                   // bias/norm 类参数：豁免 vs 也衰减
    let sgd = AW.w0, mom = AW.w0, buf = 0;
    let ad = AW.w0, m = 0, v = 0;
    let nb = AW.w0, nbm = 0, nbv = 0;
    let l2 = AW.w0, l2m = 0, l2v = 0;
    let ux = AW.u0, uxm = 0, uxv = 0;
    let ud = AW.u0, udm = 0, udv = 0;
    for (let t = 1; t <= T; t++) {
      const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
      /* SGD（无动量） */
      if (wd) sgd *= (1 - lr * wd);
      sgd -= lr * AW.g(sgd);
      /* Momentum（μ=0.9，optim.py 的 SGD 类：buf=μ·buf+g） */
      if (wd) mom *= (1 - lr * wd);
      buf = AW.momMu * buf + AW.g(mom);
      mom -= lr * buf;
      /* AdamW（解耦衰减 + 偏差修正）—— 与 optim.py AdamW.step 逐行对应 */
      let decayed = 0;
      if (wd) { decayed = lr * wd * Math.abs(ad); ad *= (1 - lr * wd); }
      let g = AW.g(ad);
      m = b1 * m + (1 - b1) * g;
      v = b2 * v + (1 - b2) * g * g;
      const mh = m / c1, vh = v / c2;
      const dw = lr * mh / (Math.sqrt(vh) + eps);
      ad -= dw;
      /* AdamW 漏掉偏差修正 */
      if (wd) nb *= (1 - lr * wd);
      let gn = AW.g(nb);
      nbm = b1 * nbm + (1 - b1) * gn;
      nbv = b2 * nbv + (1 - b2) * gn * gn;
      nb -= lr * nbm / (Math.sqrt(nbv) + eps);
      /* 老写法：wd·w 加进梯度（会被 √v̂ 缩放掉），不做解耦衰减 */
      let g2 = AW.g(l2);
      if (wd) g2 += wd * l2;
      l2m = b1 * l2m + (1 - b1) * g2;
      l2v = b2 * l2v + (1 - b2) * g2 * g2;
      l2 -= lr * (l2m / c1) / (Math.sqrt(l2v / c2) + eps);
      /* bias/norm 类参数 u：目标 u*=0.5，梯度 3(u−u*)；豁免 vs 不豁免 */
      uxm = b1 * uxm + (1 - b1) * AW.a * (ux - AW.uStar);
      uxv = b2 * uxv + (1 - b2) * Math.pow(AW.a * (ux - AW.uStar), 2);
      ux -= lr * (uxm / c1) / (Math.sqrt(uxv / c2) + eps);
      if (wd) ud *= (1 - lr * wd);                        // 错法：对 bias/norm 也衰减
      const gd = AW.a * (ud - AW.uStar);
      udm = b1 * udm + (1 - b1) * gd;
      udv = b2 * udv + (1 - b2) * gd * gd;
      ud -= lr * (udm / c1) / (Math.sqrt(udv / c2) + eps);

      H.sgd.push(sgd); H.mom.push(mom); H.adam.push(ad); H.nobc.push(nb); H.l2.push(l2);
      uEx.push(ux); uDe.push(ud);
      det.push({ w: ad, m, v, mh, vh, c1, c2, dw, decayed, g });
    }
    return { H, det, uEx, uDe };
  }

  /* 纯函数 stepAt：第 t 帧能看到的全部状态（历史截到 t，细节取第 t 步）。 */
  function adamwStepAt(t, p) {
    const s = adamwSim(p);
    const cut = arr => arr.slice(0, t + 1);
    return { t, H: { sgd: cut(s.H.sgd), mom: cut(s.H.mom), adam: cut(s.H.adam),
                     nobc: cut(s.H.nobc), l2: cut(s.H.l2) },
             full: s.H, det: s.det[Math.max(1, Math.min(t, AW.T))], t0: t === 0,
             uEx: s.uEx[Math.min(t, AW.T)], uDe: s.uDe[Math.min(t, AW.T)] };
  }

  R.adamw = function (box) {
    box.classList.add('lane-b');
    F.title(box, '§2.5 / §11 · SGD vs Momentum vs AdamW：三条轨迹当场真算，'
      + '看清偏差修正、解耦衰减、wd 豁免三件事');
    const hd = F.head(box, '');
    const btnRow = el('div', 'lane-b-btns'); box.appendChild(btnRow);
    let showNoBC = false, showL2 = false;
    toggle(btnRow, '叠画：AdamW 漏掉偏差修正', false, v => { showNoBC = v; play.redraw(); });
    toggle(btnRow, '叠画：老写法 wd·w 加进梯度（L2）', false, v => { showL2 = v; play.redraw(); });
    legend(box, [[C.warn, 'SGD'], [C.ok, 'Momentum μ=0.9'], [C.accent, 'AdamW（optim.py 原式）'],
                 [C.bad, 'AdamW 漏修正'], [C.purple, 'wd 进梯度'], [C.dim, '最优 w*']]);
    const wrap = el('div', 'lane-b-plot'); box.appendChild(wrap);
    const rd = F.readout(box);
    const sl = F.sliders(box);

    const p = { lr: 0.05, b1: 0.9, b2: 0.95, wd: 0.1, eps: 1e-8 };
    F.slider(sl, 'lr', 0.005, 0.25, 0.005, p.lr, v => { p.lr = v; play.redraw(); });
    F.slider(sl, 'β₁', 0, 0.99, 0.01, p.b1, v => { p.b1 = v; play.redraw(); });
    F.slider(sl, 'β₂', 0.5, 0.999, 0.001, p.b2, v => { p.b2 = v; play.redraw(); });
    F.slider(sl, 'weight_decay', 0, 1, 0.01, p.wd, v => { p.wd = v; play.redraw(); });
    F.slider(sl, 'eps=10^e', -10, -2, 1, -8, v => { p.eps = Math.pow(10, v); play.redraw(); });

    function draw(t) {
      const st = adamwStepAt(t, p);
      const W = plotW(box), Hh = 250;
      /* y 范围：按全部曲线在整段 [0,T] 上的取值算（与 t 无关，画面不跳） */
      let lo = Infinity, hi = -Infinity;
      const keys = ['sgd', 'mom', 'adam'].concat(showNoBC ? ['nobc'] : [], showL2 ? ['l2'] : []);
      keys.forEach(k => st.full[k].forEach(w => { if (w < lo) lo = w; if (w > hi) hi = w; }));
      lo = Math.max(-8, lo - 0.15); hi = Math.min(8, hi + 0.15);
      if (!(lo < hi)) { lo = -2; hi = 2; }
      const g = F.plot(wrap, W, Hh, 0, AW.T, lo, hi);
      const upto = (arr, extra) => x => {
        const i = Math.round(x);
        return i <= st.t && i < arr.length ? (extra ? extra(arr[i]) : arr[i]) : NaN;
      };
      g.line(() => AW.wStar, C.dim, [2, 4]);
      g.line(upto(st.H.sgd), C.warn);
      g.line(upto(st.H.mom), C.ok);
      g.line(upto(st.H.adam), C.accent, null, true);
      if (showNoBC) g.line(upto(st.H.nobc), C.bad, [5, 4]);
      if (showL2) g.line(upto(st.H.l2), C.purple, [5, 4]);
      g.dots([st.t], upto(st.H.sgd), C.warn);
      g.dots([st.t], upto(st.H.mom), C.ok);
      g.dots([st.t], upto(st.H.adam), C.accent);

      const d = st.det;
      hd.innerHTML = st.t0
        ? `第 <b>0</b>/${AW.T} 步：还没走。三条轨迹都从 w₀=${AW.w0} 出发，最优 w*=${fmt(AW.wStar)}（f(w)=½·${AW.a}w²${AW.b}w+${AW.c}）`
        : `第 <b>${st.t}</b>/${AW.T} 步 · SGD w=<b>${fmt(st.H.sgd[st.t])}</b>`
          + ` · Momentum w=<b>${fmt(st.H.mom[st.t])}</b>`
          + ` · AdamW w=<b>${fmt(st.H.adam[st.t])}</b>（w*=${fmt(AW.wStar)}）`;

      const firstStepRatio = (1 - p.b1) / Math.sqrt(1 - p.b2);   // §2.5 手算的 0.447（β₂=0.95 时）
      const shrink = Math.pow(1 - p.lr * p.wd, AW.T);
      const l2Eff = st.t0 ? 0 : p.lr * p.wd * Math.abs(d.w) / (Math.sqrt(d.vh) + p.eps);
      rd.innerHTML = st.t0 ? '<span>按 ▶ 播放或 ⏭ 单步开始迭代</span>'
        : `<span>① 偏差修正 t=${st.t}：1−β₁ᵗ=<b>${fmt(d.c1, 4)}</b>（m̂ 是 m 的 <b>${fmt(1 / d.c1, 2)}</b> 倍），`
          + `1−β₂ᵗ=<b>${fmt(d.c2, 4)}</b>（v̂ 放大 <b>${fmt(1 / d.c2, 2)}</b> 倍）</span>`
        + `<span>恒梯度第一步：有修正正好走 1.000·lr，没修正只走 <b>${fmt(firstStepRatio)}</b>·lr`
          + `（β₁=0.9、β₂=0.95 时就是 §2.5 手算的 0.447）</span>`
        + `<span>② 解耦衰减：每步 w×(1−lr·wd)=${fmt(1 - p.lr * p.wd, 5)}，`
          + `${AW.T} 步后 ∏(1−lr·wd)=<b>${shrink.toExponential(2)}</b>；`
          + `本步衰减量 <b>${d.decayed.toExponential(2)}</b>，`
          + `同样 wd·w 若加进梯度、再被 (√v̂+eps) 除，变成 <b>${l2Eff.toExponential(2)}</b>（由梯度大小决定，不再按 ∏ 计划）</span>`
        + `<span>③ wd 豁免：bias/norm 类参数 u（目标 u*=${AW.uStar}）`
          + `豁免=<b>${fmt(st.uEx)}</b> vs 也衰减=<b style="color:${C.bad}">${fmt(st.uDe)}</b>`
          + `（不豁免会被 wd 一路拖向 0，模型变笨）</span>`;
    }
    const play = playback(box, () => AW.T, draw);
    draw(0);
    F.caption(box, '三条轨迹都是当场用同一份式子迭代出来的（`f(w)=½·3w²−2.4w+1`，w₀=−1.5，'
      + 'AdamW 递推与 `solutions/minigpt/optim.py` 逐行对应：先解耦衰减 `w×(1−lr·wd)`，'
      + '再 `m=β₁m+(1−β₁)g`、`v=β₂v+(1−β₂)g²`、`m̂=m/(1−β₁ᵗ)`、`v̂=v/(1−β₂ᵗ)`、'
      + '`w −= lr·m̂/(√v̂+eps)`）。<b>①</b> 勾"漏掉偏差修正"：前几十步整体偏慢 —— 判据 '
      + '`AdamW 单步幅度 = lr` 抓的就是这个（恒梯度下正确实现第一步恰好走 lr，漏修正只走 0.447·lr）。'
      + '<b>②</b> 勾"老写法"：`wd·w` 进梯度后衰减量变成 lr·wd·w/√v̂ —— 被二阶矩接管，'
      + '前期 √v̂≈|g| 很大时衰减被压缩、后期梯度变小又反过来放大，全程不按 ∏(1−lr·wd) 的计划走，'
      + '两条轨迹（蓝 vs 紫）第 50 步就分家 —— '
      + '解耦与 L2 在 Adam 里<b>不等价</b>；纯衰减下权重范数按 ∏(1−lr·wd)ᵗ 指数收缩。'
      + '<b>③</b> 第三行读数是一对 bias/norm 类参数的真算对照：豁免的收敛到 u*=0.5，'
      + '不豁免的被 wd 拖着走 —— 这就是 `Parameter.wd_exempt` 存在的理由（§3.2、§11）。');
  };

  /* ================================================================ 2. resume
     "断了能续"：150 步直训 vs 100 步存档→新建模型读回→续 50 步。
     四个勾选框对应四元组 (θ, m, v, t)，勾上任何一个，续训段立刻跳。 */

  const RS = { a: 6, b: -3.6, w0: -1.5, T: 150, SAVE: 100, seed: 12345,
               b1: 0.9, b2: 0.95, eps: 1e-8 };
  RS.wStar = -RS.b / RS.a;
  RS.f = w => 0.5 * RS.a * w * w + RS.b * w;
  RS.g = w => RS.a * w + RS.b;
  RS.xi = (function () {                    // 固定的"批次噪声"序列（种子 12345）
    const r = mulberry32(RS.seed), out = [];
    for (let i = 0; i < RS.T; i++) out.push(gauss(r));
    return out;
  })();

  /* 纯函数：drops ⊆ {w,m,v,t}，返回 A/B 两条 loss 曲线与恢复段逐点差。 */
  function resumeSim(p, drops) {
    const lr = p.lr, sig = p.sig, b1 = RS.b1, b2 = RS.b2, eps = RS.eps;
    const T = RS.T, S = RS.SAVE, has = k => drops.indexOf(k) < 0;
    let w = RS.w0, m = 0, v = 0;
    const lossA = [];
    let sw = 0, sm = 0, sv = 0, stp = 0;
    for (let i = 0; i < T; i++) {
      lossA.push(RS.f(w));
      const g = RS.g(w) + sig * RS.xi[i];
      m = b1 * m + (1 - b1) * g;
      v = b2 * v + (1 - b2) * g * g;
      const t = i + 1;
      w -= lr * (m / (1 - Math.pow(b1, t))) / (Math.sqrt(v / (1 - Math.pow(b2, t))) + eps);
      if (t === S) { sw = w; sm = m; sv = v; stp = t; }   // save_train_state
    }
    /* "新建一台模型 + 新建优化器"，然后按勾选决定读回什么（load_train_state） */
    let wB = has('w') ? sw : RS.w0;
    let mB = has('m') ? sm : 0;
    let vB = has('v') ? sv : 0;
    let tB = has('t') ? stp : 0;
    const lossB = lossA.slice(0, S);
    for (let i = S; i < T; i++) {
      lossB.push(RS.f(wB));
      const g = RS.g(wB) + sig * RS.xi[i];
      mB = b1 * mB + (1 - b1) * g;
      vB = b2 * vB + (1 - b2) * g * g;
      tB += 1;
      wB -= lr * (mB / (1 - Math.pow(b1, tB))) / (Math.sqrt(vB / (1 - Math.pow(b2, tB))) + eps);
    }
    let stepwise = 0;
    for (let i = S; i < T; i++) stepwise = Math.max(stepwise, Math.abs(lossA[i] - lossB[i]));
    return { lossA, lossB, stepwise };
  }

  function resumeStepAt(t, p, drops) {
    const s = resumeSim(p, drops);
    return { t, lossA: s.lossA.slice(0, t + 1), lossB: s.lossB.slice(0, t + 1),
             stepwise: s.stepwise, full: s,
             segMax: (function () {          // 只统计已经播到的恢复段
               let mx = 0;
               for (let i = RS.SAVE; i <= Math.min(t, RS.T - 1); i++)
                 mx = Math.max(mx, Math.abs(s.lossA[i] - s.lossB[i]));
               return mx;
             })() };
  }

  R.resume = function (box) {
    box.classList.add('lane-b');
    F.title(box, '§2.4 / §7 · 断了能续：四元组 (θ, m, v, t) 缺一不可 —— 勾掉任何一个，续训段立刻跳');
    const hd = F.head(box, '');
    const chk = el('div', 'lane-b-btns'); box.appendChild(chk);
    const drops = { w: false, m: false, v: false, t: false };
    const SYMPTOM = {
      w: 'θ 没恢复：新模型带着随机初始化开跑，loss 直接跳回起点',
      m: 'm 没恢复：动量从 0 重攒，续训前几十步等于偷偷重来一次 warmup',
      v: 'v 没恢复：√v̂ 从 (1−β₂)g² 起步、偏小，头几步步子被放大',
      t: 't 没恢复：c₁=1−β₁¹=0.1 而 m 装着 100 步的量 → 第一步 m̂ 放大 10 倍、v̂ 放大 20 倍'
    };
    [['w', '不恢复 θ（权重）'], ['m', '不恢复 m（一阶矩）'],
     ['v', '不恢复 v（二阶矩）'], ['t', '不恢复 t（步数）']].forEach(([k, lab]) =>
      toggle(chk, lab, false, on => { drops[k] = on; play.redraw(); }));
    legend(box, [[C.accent, 'A：一口气训 150 步'], [C.ok, 'B：训 100 步存档 → 新建模型读回 → 续 50 步（四元组全恢复时两条重合）']]);
    const wrapA = el('div', 'lane-b-plot'); box.appendChild(wrapA);
    const wrapD = el('div', 'lane-b-plot'); box.appendChild(wrapD);
    const note = el('div', 'lane-b-note', '下图：恢复段（第 100~150 步）逐点差 |lossA−lossB|，纵轴是 log₁₀，黄色虚线是判据阈值 1e-4');
    box.appendChild(note);
    const rd = F.readout(box);
    const sl = F.sliders(box);
    const p = { lr: 0.15, sig: 2.0 };
    F.slider(sl, 'lr', 0.02, 0.3, 0.005, p.lr, v => { p.lr = v; play.redraw(); });
    F.slider(sl, 'σ（批次梯度噪声）', 0, 4, 0.1, p.sig, v => { p.sig = v; play.redraw(); });

    const dropList = () => Object.keys(drops).filter(k => drops[k]);

    function draw(t) {
      const dl = dropList();
      const st = resumeStepAt(t, p, dl);
      const W = plotW(box);
      /* 主图：两条 loss 曲线 */
      let lo = Infinity, hi = -Infinity;
      st.full.lossA.concat(st.full.lossB).forEach(y => { if (y < lo) lo = y; if (y > hi) hi = y; });
      lo -= 0.2; hi += 0.2;
      const gA = F.plot(wrapA, W, 240, 0, RS.T, lo, hi);
      const cut = arr => x => { const i = Math.round(x); return i <= st.t ? arr[i] : NaN; };
      const broken = st.stepwise > 1e-4;
      gA.line(cut(st.full.lossA), C.accent, null, true);
      gA.line(cut(st.full.lossB), broken ? C.bad : C.ok, null, true);
      gA.dots([st.t], cut(st.full.lossA), C.accent);
      gA.dots([st.t], cut(st.full.lossB), broken ? C.bad : C.ok);
      /* 存档点竖线（叠加的绝对定位 div，不动 canvas） */
      let vl = wrapA.querySelector('.lane-b-vline'), vlab = wrapA.querySelector('.lane-b-vlab');
      if (!vl) { vl = el('div', 'lane-b-vline'); vlab = el('div', 'lane-b-vlab', '第 100 步：存档 → 新建模型 → 读回'); wrapA.append(vl, vlab); }
      vl.style.left = gA.X(RS.SAVE) + 'px'; vlab.style.left = (gA.X(RS.SAVE) + 5) + 'px';
      /* 副图：逐点差（log10） */
      const gD = F.plot(wrapD, W, 150, RS.SAVE, RS.T, -17, 2);
      gD.line(() => -4, C.warn, [5, 4]);
      if (t > RS.SAVE) {
        const diff = x => {
          const i = Math.round(x);
          if (i <= st.t && i >= RS.SAVE) return Math.log10(Math.max(Math.abs(st.full.lossA[i] - st.full.lossB[i]), 1e-17));
          return NaN;
        };
        gD.line(diff, broken ? C.bad : C.ok, null, true);
      }
      const passed = st.stepwise <= 1e-4;
      hd.innerHTML = st.t <= RS.SAVE
        ? `第 <b>${st.t}</b>/${RS.T} 步 · 前 ${RS.SAVE} 步 A、B 同轨（B 还没存档）· loss=<b>${fmt(st.full.lossA[Math.min(st.t, RS.T - 1)], 4)}</b>`
        : `第 <b>${st.t}</b>/${RS.T} 步（续训段）· 恢复段逐点最大差 <b>${st.segMax.toExponential(2)}</b>`
          + ` · 判据 ≤1e-4：<b style="color:${passed ? C.ok : C.bad}">${passed ? '通过' : '不通过'}</b>`;
      rd.innerHTML = `<span>四元组恢复情况：θ <b>${drops.w ? '✗' : '✓'}</b> m <b>${drops.m ? '✗' : '✓'}</b>`
        + ` v <b>${drops.v ? '✗' : '✓'}</b> t <b>${drops.t ? '✗' : '✓'}</b></span>`
        + `<span>整段（100~150 步）逐点最大差 = <b style="color:${passed ? C.ok : C.bad}">${st.stepwise.toExponential(2)}</b>`
        + `（判据「续训与直训在恢复段逐步吻合」的阈值是 1e-4；全恢复时是浮点噪声级）</span>`
        + (dl.length ? dl.map(k => `<span style="color:${C.bad}">${SYMPTOM[k]}</span>`).join('')
                      : '<span>什么都不勾：B 与 A 逐点一致 —— 这就是"断了能续"</span>');
    }
    const play = playback(box, () => RS.T, draw);
    draw(0);
    F.caption(box, '递推与 `adamw` 那张图（阶段 5 §11）完全同一份（AdamW，β₁=0.9、β₂=0.95，无 wd），'
      + '只是梯度加了固定的批次噪声 σ·ξᵢ（种子 12345，A/B 共用同一串 —— 对应判据脚本里'
      + '"同一个 rng、同一串随机窗口"）。B 在第 100 步把 (θ, m, v, t) 存进状态文件，'
      + '然后<b>另建一台模型、另建一个优化器</b>再读回 —— 判据 `stages/s06_train.py` 就是这么设计的'
      + '（新建时用别的种子、别的 lr，逼你证明"续训"靠的是存档而不是巧合），阈值：恢复段'
      + '<b>逐点</b>最大差 ≤1e-4。逐点而不是末段均值，才能抓住"只错一步"的 bug（比如 t 少恢复一步）。'
      + '注意 lr 调度不在四元组里：`lr_at(i, 总步数, …)` 由训练脚本现算，且总步数必须是原计划的'
      + '150（§7 的 `horizon`）。文件字节布局与四种拒绝见本章前面 `ckpt` 那张图，这张只讲状态。');
  };

  /* ================================================================ 3. lrsched
     warmup + cosine 的 lr_at(i) 原式；叠一条前 30 步的 ‖Δw‖，
     再并排画"有 warmup vs 满 lr 直接上"的 loss —— 翻车翻在哪儿。 */

  const LS = { a: 40, b: -24, w0: -1.0, N: 30, sig0: 3.0, grow: 0.35, seed: 777,
               b1: 0.9, b2: 0.95, eps: 1e-8, finalFrac: 0.1 };
  LS.wStar = -LS.b / LS.a;
  LS.f = w => 0.5 * LS.a * w * w + LS.b * w;
  LS.g = w => LS.a * w + LS.b;
  LS.xi = (function () {
    const r = mulberry32(LS.seed), out = [];
    for (let i = 0; i < LS.N; i++) out.push(gauss(r));
    return out;
  })();

  /* optim.py 的 lr_at 原式（final_frac=0.1） */
  function lrAt(step, totalSteps, peak, warmup, finalFrac) {
    finalFrac = finalFrac == null ? LS.finalFrac : finalFrac;
    if (step < warmup) return peak * (step + 1) / Math.max(1, warmup);
    if (totalSteps <= warmup) return peak;
    let frac = (step - warmup) / Math.max(1, totalSteps - warmup);
    frac = Math.min(1, Math.max(0, frac));
    return peak * (finalFrac + (1 - finalFrac) * 0.5 * (1 + Math.cos(Math.PI * frac)));
  }

  /* 真算一台 AdamW 在前 N 步的 ‖Δw‖ 与 loss：mode='warm' 用调度，'full' 恒 peak。 */
  function schedRun(p, mode) {
    const dws = [], losses = [];
    let w = LS.w0, m = 0, v = 0;
    for (let i = 0; i < LS.N; i++) {
      const lr = mode === 'warm' ? lrAt(i, p.total, p.peak, p.warmup) : p.peak;
      losses.push(LS.f(w));
      const sig = LS.sig0 * (1 + LS.grow * i);
      const g = LS.g(w) + sig * LS.xi[i];
      m = LS.b1 * m + (1 - LS.b1) * g;
      v = LS.b2 * v + (1 - LS.b2) * g * g;
      const t = i + 1;
      const dw = lr * (m / (1 - Math.pow(LS.b1, t))) / (Math.sqrt(v / (1 - Math.pow(LS.b2, t))) + LS.eps);
      w -= dw;
      dws.push(Math.abs(dw));
    }
    return { dws, losses, wEnd: w };
  }

  function lrschedStepAt(t, p) {
    return { t, lr: t < p.total ? lrAt(t, p.total, p.peak, p.warmup) : lrAt(p.total - 1, p.total, p.peak, p.warmup),
             warm: schedRun(p, 'warm'), full: schedRun(p, 'full') };
  }

  R.lrsched = function (box) {
    box.classList.add('lane-b');
    F.title(box, '§13 · lr_at：线性 warmup → cosine 衰减到 peak·0.1，'
      + '以及"一上来就给满 lr"在前 30 步翻给谁看');
    const hd = F.head(box, '');
    legend(box, [[C.accent, 'lr_at(i)（左轴：学习率）'],
                 [C.ok, '‖Δw‖ 有 warmup（×k 缩放到同轴，见读数）'],
                 [C.bad, '‖Δw‖ 满 lr 直接上（×k）']]);
    const wrap1 = el('div', 'lane-b-plot'); box.appendChild(wrap1);
    const note = el('div', 'lane-b-note', '下图：前 30 步的 loss（同一台 AdamW、同一串梯度噪声）—— 翻车的证据在这里');
    const wrap2 = el('div', 'lane-b-plot'); box.appendChild(wrap2);
    box.appendChild(note);
    const rd = F.readout(box);
    const sl = F.sliders(box);
    const p = { warmup: 8, peak: 0.2, total: 120 };
    F.slider(sl, 'warmup 步数', 0, 30, 1, p.warmup, v => { p.warmup = v; play.redraw(); });
    F.slider(sl, '峰值 lr', 0.02, 0.4, 0.005, p.peak, v => { p.peak = v; play.redraw(); });
    F.slider(sl, '总步数', 40, 200, 5, p.total, v => { p.total = v; play.redraw(); });

    function draw(t) {
      const st = lrschedStepAt(t, p);
      const W = plotW(box);
      const k = p.peak / Math.max(1e-12, Math.max.apply(null, st.warm.dws.concat(st.full.dws)));
      /* 主图：lr 曲线 + 两条缩放过来的 ‖Δw‖ */
      const g1 = F.plot(wrap1, W, 230, 0, p.total, 0, p.peak * 1.18);
      const inWin = i => i < LS.N;
      g1.line(x => { const i = Math.round(x); return i <= st.t ? lrAt(i, p.total, p.peak, p.warmup) : NaN; },
              C.accent, null, true);
      g1.line(x => { const i = Math.round(x); return i <= st.t && inWin(i) ? st.warm.dws[i] * k : NaN; }, C.ok, [5, 4]);
      g1.line(x => { const i = Math.round(x); return i <= st.t && inWin(i) ? st.full.dws[i] * k : NaN; }, C.bad, [5, 4]);
      g1.dots([st.t], x => { const i = Math.round(x); return i <= st.t ? lrAt(i, p.total, p.peak, p.warmup) : NaN; }, C.accent);
      /* 副图：loss，前 30 步 */
      const allL = st.warm.losses.concat(st.full.losses);
      let lo = Math.min.apply(null, allL), hi = Math.max.apply(null, allL);
      const pad = (hi - lo) * 0.08 + 0.05;
      const g2 = F.plot(wrap2, W, 170, 0, LS.N, lo - pad, hi + pad);
      g2.line(x => { const i = Math.round(x); return i <= st.t && inWin(i) ? st.warm.losses[i] : NaN; }, C.ok, null, true);
      g2.line(x => { const i = Math.round(x); return i <= st.t && inWin(i) ? st.full.losses[i] : NaN; }, C.bad, null, true);
      hd.innerHTML = `第 <b>${st.t}</b>/${p.total} 步 · lr_at(${st.t})=<b>${fmt(st.lr, 5)}</b>`
        + (st.t === 0 ? ' · 还没走第一步'
          : st.t <= LS.N
          ? ` · 第 ${st.t} 步 ‖Δw‖：warmup 版 <b>${fmt(st.warm.dws[st.t - 1], 4)}</b>，满 lr 版 <b>${fmt(st.full.dws[st.t - 1], 4)}</b>`
          : ' · ‖Δw‖ 窗口（前 30 步）已走完，看下面 loss 的差别');
      const mxW = Math.max.apply(null, st.warm.dws), mxF = Math.max.apply(null, st.full.dws);
      const l30W = st.warm.losses[LS.N - 1], l30F = st.full.losses[LS.N - 1];
      rd.innerHTML = `<span>warmup 段第 i 步：lr = peak·(i+1)/warmup；之后 cosine 衰减到 peak·0.1（尾端不归零）</span>`
        + `<span>前 30 步 max‖Δw‖：warmup 版 <b>${fmt(mxW, 4)}</b> vs 满 lr 版 <b>${fmt(mxF, 4)}</b>`
        + `（主图里 ‖Δw‖ 按 ×k=<b>${fmt(k, 2)}</b> 缩到 lr 轴上）</span>`
        + `<span>第 30 步 loss：warmup 版 <b style="color:${C.ok}">${fmt(l30W, 3)}</b>`
        + ` vs 满 lr 版 <b style="color:${C.bad}">${fmt(l30F, 3)}</b>（差 <b>${fmt(Math.abs(l30W - l30F), 3)}</b>）</span>`
        + `<span>第 30 步离最优的距离 |w−w*|：warmup 版 <b>${fmt(Math.abs(st.warm.wEnd - LS.wStar), 3)}</b>`
        + ` vs 满 lr 版 <b>${fmt(Math.abs(st.full.wEnd - LS.wStar), 3)}</b></span>`;
    }
    const play = playback(box, () => p.total, draw);
    draw(0);
    F.caption(box, '曲线就是 `optim.py::lr_at` 的原式：i<warmup 时 `peak·(i+1)/warmup` 线性爬升，'
      + '之后 `peak·(0.1+0.9·½(1+cos(π·frac)))`，`frac=(i−warmup)/(total−warmup)` —— '
      + '尾端留 10% 不归零，后期还要一点步长微调。‖Δw‖ 与 loss 都是当场真算：玩具目标 '
      + '`f(w)=½·40w²−24w`（w*=0.6，w₀=−1），梯度带<b>逐渐变大</b>的噪声 `σᵢ=3·(1+0.35i)·ξᵢ`'
      + '（固定种子），AdamW β=(0.9, 0.95)。满 lr 版第一步 ‖Δw‖ 就是 peak 全额 —— 可这时 v̂ '
      + '只见过一两个梯度样本，方向纯属噪声，前几步的位移是随机游走；warmup 版同样方向不准，'
      + '但每步只挪 peak 的几分之一，等 m̂/v̂ 成形了才放开步子。把 warmup 拖到 0：'
      + '两条 ‖Δw‖ 重合，loss 图里红线的震荡就是"翻车"。这也是自查题那条'
      + '"warmup 和偏差修正在解决同一个问题"的图版：都在对付<b>前几十步的矩估计不可信</b>。');
  };

  /* ================================================================ 4. lnv
     ln V 基准线：随机初始化 + 正确移位时交叉熵的期望 = ln V（多批采样取平均），
     "目标不错位（看答案）"一开，CE 立刻塌到 0 附近 —— 反作弊判据的图版。 */

  const LV = { batches: 12, npos: 32, sigma: 0.05, cheatA: 12, eMin: 1, eMax: 13 };
  const lvCache = new Map();               // (cheat?1e6:0)+e*100+j → 该批平均 CE
  function ceBatch(V, j, cheat) {
    const key = (cheat ? 1000000 : 0) + V * 100 + j;
    if (lvCache.has(key)) return lvCache.get(key);
    const rnd = mulberry32(1000 + j * 7919);
    let tot = 0;
    const z = new Float64Array(V);
    for (let k = 0; k < LV.npos; k++) {
      const tgt = Math.floor(rnd() * V) % V;
      let mx = -Infinity;
      for (let i = 0; i < V; i++) {
        let zz = LV.sigma * gauss(rnd);
        if (cheat && i === tgt) zz += LV.cheatA;         // 目标不错位 = 输入里就有答案
        z[i] = zz; if (zz > mx) mx = zz;
      }
      let s = 0;
      for (let i = 0; i < V; i++) s += Math.exp(z[i] - mx);
      tot += (mx + Math.log(s)) - z[tgt];                 // logsumexp − z_tgt，减最大值写法
    }
    const mean = tot / LV.npos;
    lvCache.set(key, mean);
    return mean;
  }
  function lnvStepAt(t, cheatOn) {         // t = 已平均的批数
    const e = [];
    for (let x = LV.eMin; x <= LV.eMax; x++) {
      let hs = 0, cs = 0;
      for (let j = 0; j < t; j++) {
        hs += ceBatch(Math.pow(2, x), j, false);
        if (cheatOn) cs += ceBatch(Math.pow(2, x), j, true);
      }
      e.push({ x, V: Math.pow(2, x), honest: t ? hs / t : NaN, cheat: (cheatOn && t) ? cs / t : NaN });
    }
    return { t, e };
  }

  R.lnv = function (box) {
    box.classList.add('lane-b');
    F.title(box, '§2.4 · ln V 基准线：随机初始化的交叉熵期望就是 ln V；'
      + '目标一"不错位"（看答案），立刻塌下去 —— 反作弊判据的图版');
    const hd = F.head(box, '');
    const btnRow = el('div', 'lane-b-btns'); box.appendChild(btnRow);
    let cheat = false;
    toggle(btnRow, '目标不错位（等于看答案）', false, v => { cheat = v; play.redraw(); });
    legend(box, [[C.dim, 'ln V（理论基准线）'], [C.accent, '实测：随机初始化 + 正确移位（多批平均）'],
                 [C.bad, '实测：目标不错位（作弊）']]);
    const wrap = el('div', 'lane-b-plot'); box.appendChild(wrap);
    const rd = F.readout(box);
    const sl = F.sliders(box);
    let Ve = 5;                            // V = 2^Ve
    F.slider(sl, 'log₂V（V=2^刻度）', LV.eMin, LV.eMax, 1, Ve, v => { Ve = v; play.redraw(); });

    function draw(t) {
      const st = lnvStepAt(t, cheat);
      const W = plotW(box);
      const g = F.plot(wrap, W, 260, LV.eMin, LV.eMax, 0, LV.eMax * Math.LN2 * 1.06);
      g.line(x => x * Math.LN2, C.dim, [6, 4], true);
      const at = (arr, key) => x => {
        const i = Math.round(x) - LV.eMin;
        return i >= 0 && i < arr.length ? arr[i][key] : NaN;
      };
      if (t > 0) {
        g.line(at(st.e, 'honest'), C.accent, null, true);
        g.dots(st.e.map(o => o.x), at(st.e, 'honest'), C.accent);
        if (cheat) {
          g.line(at(st.e, 'cheat'), C.bad, null, true);
          g.dots(st.e.map(o => o.x), at(st.e, 'cheat'), C.bad);
        }
        g.dots([Ve], at(st.e, 'honest'), C.fg);
      }
      const V = Math.pow(2, Ve), lnV = Math.log(V);
      const cur = st.e[Ve - LV.eMin];
      hd.innerHTML = `第 <b>${t}</b>/${LV.batches} 批平均（每批 ${LV.npos} 个位置）`
        + ` · V=<b>${V}</b> · ln V=<b>${fmt(lnV, 4)}</b>`
        + (t ? ` · 实测 CE=<b>${fmt(cur.honest, 4)}</b>` : ' · 还没采样：按 ▶ 播放');
      if (!t) { rd.innerHTML = '<span>播放后每帧多平均一批：点从散到稳，收敛到 ln V 上</span>'; }
      else {
        const d = Math.abs(cur.honest - lnV);
        rd.innerHTML = `<span>ln V = <b>${fmt(lnV, 4)}</b>（V=${V}）</span>`
          + `<span>实测均值（${t} 批 × ${LV.npos} 位置，logits~N(0, ${LV.sigma}²)，目标均匀随机）= <b>${fmt(cur.honest, 4)}</b></span>`
          + `<span>|实测 − ln V| = <b>${fmt(d, 4)}</b>（判据「初始 loss ≈ ln V」容差 0.2 → `
          + `<b style="color:${d <= 0.2 ? C.ok : C.bad}">${d <= 0.2 ? '通过' : '不过'}</b>；`
          + `单批 t=1 时散点约 ±0.1，多批平均才收敛）</span>`
          + (cheat ? `<span>看答案（目标不错位）：CE ≈ <b style="color:${C.bad}">${cur.cheat.toExponential(2)}</b>`
            + ` —— 远低于 ln V。随机数据上 loss 掉下去的唯一解释就是作弊（预测自己）</span>` : '');
      }
    }
    const play = playback(box, () => LV.batches, draw);
    draw(0);
    F.caption(box, '每个点都是当场真算：抽 `V` 个 logits（随机初始化 → 小随机数，σ=0.05），'
      + '目标均匀随机，按稳定写法 `logsumexp(z) − z_tgt` 算交叉熵，每批 32 个位置、'
      + '最多 12 批取平均（固定种子，逐批可复现）。随机小 logits 过 softmax 接近均匀分布，'
      + '所以 E[CE] = ln V —— s05 §2.4 的 `V=25 → ln V=3.219`、s06 §2.5 实测初始 loss 3.345 vs '
      + 'ln 27=3.296，说的都是这条线；判据 `初始 loss ≈ ln V`（容差 0.2）量的就是它。'
      + '勾上"目标不错位"：相当于移位写错、模型在第 t 位看见的正是第 t 位的答案'
      + '（`y[:, :-1] == x[:, 1:]` 被破坏成"预测自己"），CE 立刻塌到 ln V 之下几个数量级 —— '
      + '这就是判据 `随机数据学不动（反作弊判据）` 的逻辑：<b>不可预测的数据上 loss 降不下去才是对的</b>，'
      + '降得动一定是抄了答案。拖动 log₂V 看 V=2 到 8192：作弊曲线的缺口随 V 变大反而更刺眼。');
  };

  /* 数值自测钩子：node 里 stub 掉 window/document 后可以直接调这些纯函数，
     与独立 python 脚本按 optim.py 的式子复算对账。不注册成图，页面上不会出现。 */
  R.__b_math = { mulberry32, gauss, adamwSim, adamwStepAt, AW,
                 resumeSim, resumeStepAt, RS, lrAt, schedRun, lrschedStepAt, LS,
                 ceBatch, lnvStepAt, LV };
})();
