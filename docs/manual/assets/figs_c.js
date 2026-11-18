/* 原理动画 · 第 C 批：数值、有限差分与缓存纪律（阶段 1 / 3）。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和 figs.js 一样：能算的都真的算，不放示意数据；组件抛错只是"这张图没出来"。

   本批三个组件都围绕"梯度校验是唯一的裁判"：
     epsnoise  —— 有限差分的 U 形误差曲线：舍入噪声(∝1/ε) vs 截断误差(∝ε²)，
                  以及 gradcheck.py 里 eps=1e-6 / 1e-5、atol=1e-9 这些真实取值的来历。
     topobwd   —— backward() 为什么必须按逆拓扑序：逐步摊梯度 + 错误顺序当场对拍。
     cachepair —— "一次 forward 配一次 backward"：三种调用排法下缓存的命运，
                  用 gradcheck.py 同款公式真算最大相对误差（只有 ① 是 1e-10 量级）。

   动画约定：▶播放 / ⏭单步 / ⏸暂停，每一步的状态由纯函数 stepAt(t)（这里是
   topoStateAt / cpRun 那类"从空状态重放前 t 个动作"）决定 —— 无头浏览器截图能拍到
   确定帧，单步按钮不依赖 requestAnimationFrame（隐藏页里 rAF 不跑）。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  if (!F || !R) return;
  const { el, sv, fmt, md } = F;

  /* ------------------------------------------------------------------ 公共件 */

  let styled = false;
  function injectStyle() {
    if (styled || typeof document === 'undefined' || !document.head) return;
    styled = true;
    document.head.appendChild(el('style', null, [
      '.lane-c .cell.w2{width:76px}',
      '.lane-c .lane-seq{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0 2px}',
      '.lane-c .lane-seq .op{font:12px/1 var(--mono);padding:7px 10px;border-radius:7px;',
      '  border:1px solid var(--line);background:var(--panel2);color:var(--dim);white-space:nowrap}',
      '.lane-c .lane-seq .op.done{color:#cfe0ff;border-color:#3a4657}',
      '.lane-c .lane-seq .op.now{border-color:var(--accent);color:var(--accent);background:#152232}',
      '.lane-c .lane-abs{position:absolute;font:11px/1.4 var(--mono);color:var(--dim)}',
      '.lane-c .lane-verdict{font:12px/1.6 var(--mono);margin-top:8px}',
    ].join('\n')));
  }

  function lg(v) {                                        // 1e-6 / 1e-6.25 这种标签
    const e = Math.log10(Math.abs(v) || 1e-300);
    return '1e' + (Math.abs(e - Math.round(e)) < 0.051 ? String(Math.round(e)) : e.toFixed(2));
  }
  function sci(v, n) { return v === 0 ? '0' : v.toExponential(n == null ? 2 : n); }

  /* 动画驱动条：render(t) 负责画第 t 步并返回"当前步"文字（状态必须能从 t 重放出来）。
     getN() 每帧现取步数 —— 换模式/排法后总步数会变。host 让调用方控制按钮条插在 DOM 哪里。 */
  function animBar(host, getN, render, dur) {
    const bar = el('div', 'readout');
    const bPlay = el('button', '', '▶ 播放');
    const bStep = el('button', '', '⏭ 单步');
    const bPause = el('button', '', '⏸ 暂停');
    bar.append(bPlay, bStep, bPause);
    host.appendChild(bar);
    const line = F.head(host, '');
    let t = 0, raf = 0, last = 0;
    const D = dur || 800;
    function show(nt) { t = Math.max(0, Math.min(getN() - 1, nt | 0)); line.innerHTML = md(render(t)); }
    function tick(ts) {
      raf = 0;
      if (!last) last = ts;
      if (ts - last >= D) {
        last = ts;
        if (t < getN() - 1) { show(t + 1); raf = requestAnimationFrame(tick); }
      } else raf = requestAnimationFrame(tick);
    }
    const stop = () => { if (raf) cancelAnimationFrame(raf); raf = 0; last = 0; };
    bPlay.onclick = () => { if (raf) return; if (t >= getN() - 1) show(0); last = 0; raf = requestAnimationFrame(tick); };
    bPause.onclick = stop;
    bStep.onclick = () => { stop(); show(t >= getN() - 1 ? 0 : t + 1); };
    show(0);
    return { show, stop, get t() { return t; } };
  }

  /* ================================================================== 1. epsnoise
     有限差分为什么长成 (f(x+ε)−f(x−ε))/(2ε)、ε 为什么不能太小也不能太大。
     误差公式与 solutions/minigpt/gradcheck.py 的 compare() 逐字对应：
       rel = |num − ana| / max(1e-8, |num|, |ana|)
     曲线全部在 float64（JS number 就是 float64）下逐点真算。 */

  const EPS_FS = {
    sq: {
      f: x => x * x,
      d: x => 2 * x,
      note: 'x² 的三阶导数是 0 —— 中心差分对二次函数没有截断误差，右支不翘、一路单调下降。U 形要切到复合函数才看全。',
    },
    comp: {
      f: x => Math.tanh(1.3 * x - 0.7),
      d: x => { const t = Math.tanh(1.3 * x - 0.7); return 1.3 * (1 - t * t); },
      note: '两层复合（Linear→tanh）三阶导数不为 0：右支是标准的截断误差 ∝ ε²。讲义的 eps=1e-5 / 1e-6 正落在谷底附近。',
    },
  };

  function epsErrs(FS, x, e) {
    const num = (FS.f(x + e) - FS.f(x - e)) / (2 * e);
    const ana = FS.d(x);
    const abs = Math.abs(num - ana);
    const rel = abs / Math.max(1e-8, Math.abs(num), Math.abs(ana));
    return { num, ana, abs, rel };
  }
  /* 谷底：相对误差在 ε 网格（1e-9..1e-1，201 点）上的最小点 */
  function epsValley(FS, x) {
    let ve = -9, vr = Infinity;
    for (let i = 0; i <= 200; i++) {
      const e = -9 + i * 8 / 200;
      const r = epsErrs(FS, x, Math.pow(10, e)).rel;
      if (r < vr) { vr = r; ve = e; }
    }
    return { ve, vr };
  }

  R.epsnoise = function (box) {
    injectStyle();
    box.classList.add('lane-c');
    F.title(box, 'ε 扫描：有限差分的 U 形误差曲线（float64 下逐点真算，误差公式与 `gradcheck.py` 的 compare() 相同）');

    let key = 'comp';
    let eCur = -6;                                        // lg ε，初值 = gradcheck.py 的默认 eps=1e-6
    let xExp = 0;                                         // lg|x|
    const xOf = () => Math.pow(10, xExp);

    /* 函数切换按钮 */
    const fbar = el('div', 'readout');
    const bSq = el('button', '', 'f = x²');
    const bCp = el('button', 'on', 'f = tanh(1.3x − 0.7)');
    fbar.append(bSq, bCp); box.appendChild(fbar);
    const ctl = el('div'); box.appendChild(ctl);          // 播放条的位置先占住

    /* SVG 图（log-log） */
    const wrap = el('div'); box.appendChild(wrap);
    const W = 660, H = 330, L = 52, Rr = 12, T = 16, B = 30;
    const X = e => L + (e + 9) / 8 * (W - L - Rr);
    const Y = l => T + (0 - l) / 16.5 * (H - T - B);      // l = lg(误差)，0 在顶、−16.5 在底
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: W, role: 'img' });
    wrap.appendChild(svg);

    /* 静态坐标轴 + 讲义真实取值的标注线（只画一次） */
    for (let e = -9; e <= -1; e++) {
      svg.appendChild(sv('line', { x1: X(e), y1: T, x2: X(e), y2: H - B, stroke: '#232a35' }));
      const t = sv('text', { x: X(e) - 12, y: H - 12, fill: '#8d99ac', 'font-size': 10 });
      t.textContent = '1e' + e; svg.appendChild(t);
    }
    for (let l = 0; l >= -16; l -= 2) {
      svg.appendChild(sv('line', { x1: L, y1: Y(l), x2: W - Rr, y2: Y(l), stroke: '#232a35' }));
      const t = sv('text', { x: 6, y: Y(l) + 3, fill: '#8d99ac', 'font-size': 10 });
      t.textContent = '1e' + l; svg.appendChild(t);
    }
    let tx = sv('text', { x: W / 2 - 30, y: H - 1, fill: '#8d99ac', 'font-size': 10.5 });
    tx.textContent = 'ε（对数轴）'; svg.appendChild(tx);
    tx = sv('text', { x: 4, y: T - 4, fill: '#8d99ac', 'font-size': 10.5 });
    tx.textContent = '误差（对数轴）'; svg.appendChild(tx);
    /* eps=1e-5（scalar.py gradient_check_graph、阶段 4 起判据）、eps=1e-6（gradcheck.py 默认） */
    [[-5, '#f5b445', 'eps=1e-5'], [-6, '#4aa3ff', 'eps=1e-6']].forEach(([e, c, lab]) => {
      svg.appendChild(sv('line', { x1: X(e), y1: T, x2: X(e), y2: H - B, stroke: c, 'stroke-dasharray': '5 4', opacity: 0.65 }));
      const t = sv('text', { x: X(e) + 4, y: T + 11, fill: c, 'font-size': 10.5 });
      t.textContent = lab; svg.appendChild(t);
    });
    /* tol=1e-4 管实线（相对误差），atol=1e-9 管虚线（绝对误差） */
    [[-4, '#ff6b6b', 'tol=1e-4（相对误差阈值）'], [-9, '#3ecf8e', 'atol=1e-9（绝对误差阈值，管虚线）']].forEach(([l, c, lab], i) => {
      svg.appendChild(sv('line', { x1: L, y1: Y(l), x2: W - Rr, y2: Y(l), stroke: c, 'stroke-dasharray': '5 4', opacity: 0.55 }));
      const t = sv('text', { x: L + 6, y: i === 0 ? Y(l) - 4 : Y(l) + 12, fill: c, 'font-size': 10.5 });
      t.textContent = lab; svg.appendChild(t);
    });
    tx = sv('text', { x: W - 262, y: H - B - 26, fill: '#4aa3ff', 'font-size': 10.5 });
    tx.textContent = '—— 相对误差 |num−ana|/max(1e-8,|num|,|ana|)'; svg.appendChild(tx);
    tx = sv('text', { x: W - 262, y: H - B - 12, fill: '#8d99ac', 'font-size': 10.5 });
    tx.textContent = '- - 绝对误差 |num−ana|'; svg.appendChild(tx);
    const dyn = sv('g', {});                              // 曲线层（每次重画）
    const mark = sv('g', {});                             // 当前 ε 标记层
    svg.appendChild(dyn); svg.appendChild(mark);

    const ro = F.readout(box);
    const sl = F.sliders(box);

    function curve(FS, x, which, color, dash) {
      const pts = [];
      for (let i = 0; i <= 200; i++) {
        const e = -9 + i * 8 / 200;
        const r = epsErrs(FS, x, Math.pow(10, e));
        const v = which === 'rel' ? r.rel : r.abs;
        const l = Math.max(-16.4, Math.min(0.3, Math.log10(Math.max(v, 1e-300))));
        pts.push(`${X(e).toFixed(1)},${Y(l).toFixed(1)}`);
      }
      const p = sv('polyline', { points: pts.join(' '), fill: 'none', stroke: color, 'stroke-width': dash ? 1.4 : 2 });
      if (dash) p.setAttribute('stroke-dasharray', dash);
      return p;
    }

    function verdict(r) {
      if (r.rel <= 1e-4) return '<b style="color:#3ecf8e">rel ≤ tol=1e-4 → 过</b>';
      if (r.abs <= 1e-9) return '<b style="color:#3ecf8e">rel 超标但 abs ≤ atol=1e-9 → 噪声底，仍判过</b>';
      return '<b style="color:#ff6b6b">rel &gt; tol 且 abs &gt; atol → gradcheck 判不过</b>';
    }

    function redraw() {
      const FS = EPS_FS[key], x = xOf();
      dyn.textContent = ''; mark.textContent = '';
      dyn.appendChild(curve(FS, x, 'rel', '#4aa3ff', null));
      dyn.appendChild(curve(FS, x, 'abs', '#8d99ac', '4 3'));
      const { ve, vr } = epsValley(FS, x);
      dyn.appendChild(sv('circle', { cx: X(ve), cy: Y(Math.max(-16.4, Math.log10(vr))), r: 3.4, fill: '#3ecf8e' }));
      const vt = sv('text', { x: Math.min(X(ve) + 6, W - 210), y: Y(Math.max(-16.4, Math.log10(vr))) - 5, fill: '#3ecf8e', 'font-size': 10.5 });
      vt.textContent = `谷底 ε≈${lg(Math.pow(10, ve))}，误差≈${lg(vr)}`;
      dyn.appendChild(vt);
      const rc = epsErrs(FS, x, Math.pow(10, eCur));
      mark.appendChild(sv('line', { x1: X(eCur), y1: T, x2: X(eCur), y2: H - B, stroke: '#f5b445', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }));
      mark.appendChild(sv('circle', { cx: X(eCur), cy: Y(Math.max(-16.4, Math.min(0.3, Math.log10(rc.rel)))), r: 4, fill: '#f5b445' }));
      ro.innerHTML =
        `<span>ε = <b>${lg(Math.pow(10, eCur))}</b></span>`
        + `<span>|x| = <b>${lg(x)}</b></span>`
        + `<span>解析 f′(x) = <b>${sci(rc.ana)}</b></span>`
        + `<span>数值 (f(x+ε)−f(x−ε))/(2ε) = <b>${sci(rc.num)}</b></span>`
        + `<span>绝对误差 = <b>${sci(rc.abs)}</b></span>`
        + `<span>相对误差 = <b>${sci(rc.rel)}</b></span>`
        + `<span>谷底 ε ≈ <b>${lg(Math.pow(10, ve))}</b>（误差 ${lg(vr)}）</span>`
        + `<span>${verdict(rc)}</span>`
        + `<span>${FS.note}</span>`;
    }

    function stepText(t) {
      const FS = EPS_FS[key], x = xOf();
      const rc = epsErrs(FS, x, Math.pow(10, eCur));
      const { ve } = epsValley(FS, x);
      const zone = eCur < ve - 1.0 ? '左支 · 舍入噪声区：f(x±ε) 相减把 float64 的末位噪声放大 ulp(f)/ε 倍，ε 每小 10 倍误差大 10 倍'
        : eCur > ve + 1.0 ? (key === 'sq' ? '右支 · x² 的三阶导数为 0，没有截断误差，误差随 ε 变大继续变小（扫到边界为止）'
          : '右支 · 截断误差区：泰勒余项 ∝ ε²·f‴ 开始主导，ε 每大 10 倍误差大 100 倍')
          : '谷底附近：舍入与截断打平 —— 讲义取 eps=1e-5 / 1e-6 就是为了站在这里';
      return `第 ${t + 1}/33 步 · ε = ${lg(Math.pow(10, eCur))}：数值 ${sci(rc.num)} vs 解析 ${sci(rc.ana)}，`
        + `rel=${sci(rc.rel)}，abs=${sci(rc.abs)}。${zone}。`;
    }

    const slEps = F.slider(sl, 'lg ε', -9, -1, 0.05, -6, v => { anim.stop(); eCur = v; redraw(); });
    F.slider(sl, 'lg|x|', -8, 2, 0.5, 0, v => { xExp = v; anim.show(anim.t); });

    bSq.onclick = () => { key = 'sq'; bSq.classList.add('on'); bCp.classList.remove('on'); anim.stop(); anim.show(anim.t); };
    bCp.onclick = () => { key = 'comp'; bCp.classList.add('on'); bSq.classList.remove('on'); anim.stop(); anim.show(anim.t); };

    /* 动画：33 步把 ε 从 1e-9 扫到 1e-1；stepAt(t) 就是 eAt(t) → 重画 */
    const eAt = t => -9 + t * 0.25;
    const anim = animBar(ctl, () => 33, t => {
      eCur = eAt(t); slEps.set(eCur.toFixed(2)); redraw();
      return stepText(t);
    }, 700);
    anim.show(12);                                        // 初始停在 ε=1e-6（gradcheck.py 默认值）

    F.caption(box, '怎么读这张图：<b>左支</b>是舍入噪声 —— f(x+ε) 与 f(x−ε) 各自带着 float64 的末位误差（约 ulp(f)），'
      + '相减后再除以 2ε，噪声被放大成 ulp(f)/ε，所以 ε 越小误差越大（∝ 1/ε）。'
      + '<b>右支</b>是截断误差 —— 中心差分的泰勒余项 ∝ ε²·f‴，ε 越大偏得越多（x² 例外：f‴=0，右支不翘）。'
      + '两支夹出的<b>谷底</b>就是讲义里 `eps=1e-6`（gradcheck.py 默认）和 `eps=1e-5`（gradient_check_graph、阶段 4 起判据）'
      + '的来历：站在这里，本页小函数的相对误差能压到 1e-11~1e-13；目标值更大的真实网络噪声底约在 1e-10 上下'
      + '（s03c §4.3：1e-9~1e-7 都算正常）—— 所以"<b>误差 1e-10 是正常</b>"。'
      + '反过来，0.5 是漏了系数 2、1e0 是漏了整项、1e-2 是真错（s03c §4.3 那张量级表）—— '
      + '它们离噪声底差 7 个数量级以上，所以"<b>1e-2 一定是错</b>"……但有一个例外，正是 `atol=1e-9` 存在的理由：'
      + '切到复合函数、把 lg|x| 拉到 1（x=10，tanh 饱和，f′ 只剩约 1e-10），ε 回到 1e-9 —— '
      + '你会看到相对误差约 1.1e-2"像真错"，<b>绝对</b>误差却只有约 1.1e-10 ≤ atol，gradcheck.py 照样判过'
      + '（compare() 的注释原文：梯度本身只有 1e-8 量级时，"相对误差看着像 1e-2，其实是噪声，不是你写错了"；'
      + '大梯度看相对、小梯度看绝对）。');
  };

  /* ================================================================== 2. topobwd
     backward() 为什么必须按逆拓扑序：用阶段 1 的 L=(a·x+b)² 六节点图逐步摊梯度，
     并用 scalar.py gradient_check_graph 的同款公式（ε=1e-5，rel=|num−ana|/max(1e-8,…,…)）
     当场对拍每种播放顺序。 */

  const TOPO_PARAMS = { a: 2.0, x: 4.0, b: 3.0 };         // p=a·x=8, u=p+b=11, L=u²=121
  /* 节点表：kids 按 scalar.py _prev 的遍历顺序固定下来，topo=[a,x,p,b,u,L] 才是确定的 */
  function topoNodes(pa) {
    return [
      { id: 'a', label: 'a', data: pa.a, kids: [], op: 'leaf' },
      { id: 'x', label: 'x', data: pa.x, kids: [], op: 'leaf' },
      { id: 'p', label: 'p = a·x', data: pa.a * pa.x, kids: ['a', 'x'], op: '*' },
      { id: 'b', label: 'b', data: pa.b, kids: [], op: 'leaf' },
      { id: 'u', label: 'u = p + b', data: pa.a * pa.x + pa.b, kids: ['p', 'b'], op: '+' },
      { id: 'L', label: 'L = u²', data: Math.pow(pa.a * pa.x + pa.b, 2), kids: ['u'], op: '**2' },
    ];
  }
  const TOPO_POS = { a: [30, 52], x: [30, 132], p: [215, 92], b: [30, 212], u: [400, 152], L: [540, 152] };

  const TOPO_MODES = [
    { key: 'rev', name: '✓ 逆拓扑序（backward() 的正确循环）', seed: true, order: ['L', 'u', 'b', 'p', 'x', 'a'], passes: 1 },
    { key: 'fwd', name: '✗ 正向序（for v in topo，忘了 reversed）', seed: true, order: ['a', 'x', 'p', 'b', 'u', 'L'], passes: 1 },
    { key: 'noseed', name: '✗ 少一遍：不设 loss.grad = 1.0', seed: false, order: ['L', 'u', 'b', 'p', 'x', 'a'], passes: 1 },
    { key: 'twice', name: '✗ 跑两遍不清零（+= 会累加）', seed: true, order: ['L', 'u', 'b', 'p', 'x', 'a'], passes: 2 },
  ];

  /* 纯函数：给定模式，把动作序列展平。t=0 是起点，最后一步是对拍。 */
  function topoPlan(mode) {
    const acts = [{ kind: mode.seed ? 'seed' : 'skipseed' }];
    for (let pass = 0; pass < mode.passes; pass++)
      for (const id of mode.order) acts.push({ kind: 'rule', id, pass });
    acts.push({ kind: 'check' });
    return acts;
  }
  /* stepAt(t)：从全 0 的 grad 重放前 t+1 个动作 —— 任何一帧都由 t 唯一决定 */
  function topoStateAt(mode, t) {
    const nodes = topoNodes(TOPO_PARAMS);
    const byId = {}; nodes.forEach(n => byId[n.id] = n);
    const g = { a: 0, x: 0, p: 0, b: 0, u: 0, L: 0 };
    const acts = topoPlan(mode);
    const ti = Math.max(0, Math.min(t, acts.length - 1));
    let fired = null;
    for (let i = 0; i <= ti; i++) {
      const a = acts[i];
      if (a.kind === 'seed') { g.L = 1.0; fired = 'L'; }
      else if (a.kind === 'skipseed') fired = null;
      else if (a.kind === 'rule') {
        fired = a.id;
        const n = byId[a.id];
        if (n.op === '*') { g.a += byId.x.data * g.p; g.x += byId.a.data * g.p; }
        else if (n.op === '+') { g.p += g.u; g.b += g.u; }
        else if (n.op === '**2') { g.u += 2 * byId.u.data * g.L; }
      } else fired = null;
    }
    return { nodes, byId, g, acts, ti, fired };
  }
  /* 对拍：解析梯度（播完全部规则的 g）vs 有限差分（每次重建图，ε=1e-5，同 gradient_check_graph） */
  function topoCheck(mode) {
    const full = topoStateAt(mode, topoPlan(mode).length - 2);   // 播完所有规则、还没到 check 步
    const rows = [];
    let worst = 0;
    for (const leaf of ['a', 'b', 'x']) {
      const pp = Object.assign({}, TOPO_PARAMS);
      const mk = () => { const u = pp.a * pp.x + pp.b; return u * u; };
      const ana = full.g[leaf];
      pp[leaf] = TOPO_PARAMS[leaf] + 1e-5; const hi = mk();
      pp[leaf] = TOPO_PARAMS[leaf] - 1e-5; const lo = mk();
      pp[leaf] = TOPO_PARAMS[leaf];
      const num = (hi - lo) / (2e-5);
      const rel = Math.abs(num - ana) / Math.max(1e-8, Math.abs(num), Math.abs(ana));
      worst = Math.max(worst, rel);
      rows.push({ leaf, ana, num, rel });
    }
    return { rows, worst };
  }

  R.topobwd = function (box) {
    injectStyle();
    box.classList.add('lane-c');
    F.title(box, 'backward() 的播放顺序：L=(a·x+b)²，六个节点，逐步把梯度摊到操作数上（a=2, x=4, b=3 → L=121）');

    let mode = TOPO_MODES[0];

    /* 模式按钮 */
    const mbar = el('div', 'readout');
    const mbs = TOPO_MODES.map(m => {
      const b = el('button', m === mode ? 'on' : '', m.name);
      b.onclick = () => { mode = m; mbs.forEach((o, i) => o.classList.toggle('on', TOPO_MODES[i] === m)); anim.stop(); anim.show(0); };
      mbar.appendChild(b); return b;
    });
    box.appendChild(mbar);
    const ctl = el('div'); box.appendChild(ctl);          // 播放条的位置先占住

    /* SVG 图 */
    const W = 680, H = 270;
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: W, role: 'img' });
    box.appendChild(svg);
    [['a', 'p'], ['x', 'p'], ['p', 'u'], ['b', 'u'], ['u', 'L']].forEach(([f, t]) => {
      const [x1, y1] = TOPO_POS[f], [x2, y2] = TOPO_POS[t];
      svg.appendChild(sv('line', { x1: x1 + 118, y1: y1 + 26, x2: x2 + 4, y2: y2 + 26, stroke: '#3a4657', 'stroke-width': 1.6 }));
      const op = { p: '·', u: '+', L: '²' }[t] || '';
      const lab = sv('text', { x: (x1 + 118 + x2) / 2 - 4, y: (y1 + y2) / 2 + 20, fill: '#8d99ac', 'font-size': 12 });
      lab.textContent = op; svg.appendChild(lab);
    });
    const nodeEls = {};
    topoNodes(TOPO_PARAMS).forEach(n => {
      const [px, py] = TOPO_POS[n.id];
      const gEl = sv('g', {});
      const rect = sv('rect', { x: px, y: py, width: 118, height: 52, rx: 9, fill: '#1a212b', stroke: '#3a4657', 'stroke-width': 1.4 });
      const t1 = sv('text', { x: px + 9, y: py + 16, fill: '#cfe0ff', 'font-size': 11.5 });
      t1.textContent = n.label;
      const t2 = sv('text', { x: px + 9, y: py + 32, fill: '#e6ebf3', 'font-size': 11 });
      t2.textContent = 'data=' + fmt(n.data, 1);
      const t3 = sv('text', { x: px + 9, y: py + 46, fill: '#3ecf8e', 'font-size': 11 });
      t3.textContent = 'grad=0';
      gEl.append(rect, t1, t2, t3); svg.appendChild(gEl);
      nodeEls[n.id] = { rect, grad: t3 };
    });
    const flowDot = sv('circle', { r: 4.5, fill: '#f5b445', opacity: 0 });
    svg.appendChild(flowDot);

    /* 对拍表 + 汇总行（汇总常驻：mount 时就把四种模式的结局真算出来） */
    const chk = el('div', 'lane-verdict'); box.appendChild(chk);
    const summary = el('div', 'lane-verdict'); box.appendChild(summary);

    function draw(t) {
      const st = topoStateAt(mode, t);
      const a = st.acts[st.ti];
      for (const id in nodeEls) {
        nodeEls[id].grad.textContent = 'grad=' + fmt(st.g[id], 2);
        nodeEls[id].grad.setAttribute('opacity', st.g[id] === 0 ? 0.35 : 1);
        const hot = st.fired === id;
        nodeEls[id].rect.setAttribute('stroke', hot ? '#f5b445' : '#3a4657');
        nodeEls[id].rect.setAttribute('stroke-width', hot ? 2.4 : 1.4);
      }
      if (a && a.kind === 'rule' && st.byId[a.id].kids.length) {
        const [x1, y1] = TOPO_POS[a.id];
        flowDot.setAttribute('opacity', 1);
        flowDot.setAttribute('cx', x1); flowDot.setAttribute('cy', y1 + 26);
      } else flowDot.setAttribute('opacity', 0);
      if (a && a.kind === 'check') {
        const c = topoCheck(mode);
        chk.innerHTML = '<b>对拍（gradient_check_graph 同款：ε=1e-5，每次扰动都重建整张图）：</b>'
          + c.rows.map(r => `<span style="color:${r.rel <= 1e-4 ? '#3ecf8e' : '#ff6b6b'}">${r.leaf}：解析 ${sci(r.ana, 4)} vs 数值 ${sci(r.num, 4)}，rel=${sci(r.rel)}</span>`).join('')
          + ` <b>最大相对误差 ${sci(c.worst)}</b>`
          + (c.worst <= 1e-4 ? ' <b style="color:#3ecf8e">→ 对上了</b>' : ' <b style="color:#ff6b6b">→ 对不上（而且不会报错，只有梯度校验能抓）</b>');
      } else chk.innerHTML = '<span style="color:#8d99ac">播到最后一步做有限差分对拍。真值：a.grad=88、x.grad=44、b.grad=22。</span>';
    }

    function desc(t) {
      const st = topoStateAt(mode, t);
      const a = st.acts[st.ti], n = st.acts.length;
      const prev = st.ti > 0 ? topoStateAt(mode, st.ti - 1).g : { a: 0, x: 0, p: 0, b: 0, u: 0, L: 0 };
      const gstr = `此刻：a.grad=${fmt(st.g.a, 1)}，x.grad=${fmt(st.g.x, 1)}，b.grad=${fmt(st.g.b, 1)}，`
        + `p.grad=${fmt(st.g.p, 1)}，u.grad=${fmt(st.g.u, 1)}，L.grad=${fmt(st.g.L, 1)}`;
      if (a.kind === 'seed')
        return `第 ${st.ti + 1}/${n} 步 · 起点：loss.grad = 1.0（∂L/∂L=1，整条链的源头）。topo=[a,x,p,b,u,L]，倒过来播：L→u→b→p→x→a。`;
      if (a.kind === 'skipseed')
        return `第 ${st.ti + 1}/${n} 步 · 这个模式<b>跳过了 loss.grad = 1.0</b>：规则照常一条条执行，但每条读到的 out.grad 都是 0。`;
      if (a.kind === 'check') {
        const c = topoCheck(mode);
        return `第 ${n}/${n} 步 · 对拍：最大相对误差 ${sci(c.worst)} → ${c.worst <= 1e-4 ? '通过（tol=1e-4）' : '不通过'}。${gstr}`;
      }
      const nd = st.byId[a.id];
      let what;
      if (nd.op === 'leaf') what = `${a.id} 是叶子，_backward 是空规则（lambda: None），什么都不摊`;
      else if (nd.op === '*') what = `p 的 _backward()（规则挂在结果 p 身上，读的是 p.grad=${fmt(prev.p, 1)}）：a.grad += x.data·p.grad = 4×${fmt(prev.p, 1)}，x.grad += a.data·p.grad = 2×${fmt(prev.p, 1)}`;
      else if (nd.op === '+') what = `u 的 _backward()（读 u.grad=${fmt(prev.u, 1)}）：p.grad += ${fmt(prev.u, 1)}，b.grad += ${fmt(prev.u, 1)}`;
      else what = `L 的 _backward()（读 L.grad=${fmt(prev.L, 1)}）：u.grad += 2·u.data·L.grad = 2×11×${fmt(prev.L, 1)}`;
      if (a.kind === 'rule' && mode.key === 'fwd') what += '。<b>正向序的问题：此刻它读到的 out.grad 还是 0</b> —— 上游还没轮到';
      const passNote = a.pass === 1 ? '（第二遍！不清零，+= 继续累加 —— 注意上游 grad 已经变大，下游跟着<b>复利</b>，不是简单翻倍）' : '';
      return `第 ${st.ti + 1}/${n} 步 · ${what}${passNote}。${gstr}`;
    }

    const anim = animBar(ctl, () => topoPlan(mode).length, t => { draw(t); return desc(t); }, 900);

    summary.innerHTML = '<b>四种播放顺序的结局（挂载时就真算好，不是示意）：</b>' + TOPO_MODES.map(m => {
      const c = topoCheck(m);
      const col = c.worst <= 1e-4 ? '#3ecf8e' : '#ff6b6b';
      return `<span style="color:${col}">${m.name}：a.grad=${fmt(topoStateAt(m, topoPlan(m).length - 2).g.a, 1)}，最大 rel=${sci(c.worst)}</span>`;
    }).join(' ｜ ');

    F.caption(box, '三件事在这里看得见：<b>① `_backward` 挂在结果身上</b> —— 摊梯度的是 p、u、L 这些"结果节点"的规则，'
      + '叶子（a、x、b）的规则是空的；<b>② 起点 loss.grad = 1.0</b> —— 少这一行，所有规则照常跑、读到的 out.grad 全是 0，'
      + '全部梯度为 0 而且不报错（"少一遍"模式）；<b>③ 规则里是 `+=` 不是 `=`</b> —— 跑两遍不清零，累加还会沿链<b>复利</b>：'
      + '本页实测 a.grad 变成 352（真值 88 的 4 倍）、b.grad 变成 66（3 倍），对拍最大 rel=7.50e-01。'
      + '（"同一份梯度被原样加两遍"才是恰好 5.00e-01 = |2g−g|/|2g|，见 cachepair 的排法③。）'
      + '正向序最难发现：每个节点在还没收到上游梯度时就被摊派，叶子全漏成 0，loss 依然会降、只是降得慢 —— '
      + '只有有限差分对拍能抓（阶段 1 §6）。');
  };

  /* ================================================================== 3. cachepair
     "一次 forward 配一次 backward"：两层小网络（Linear(3→4) → CrossEntropy），
     三种调用排法下缓存（_x_flat / _probs）的命运。解析 vs 数值全部真算，
     公式与 gradcheck.py 相同（rel=|num−ana|/max(1e-8,|num|,|ana|)，ε=1e-5 = 阶段 4 起判据的取值）。
     W、b、x①、x②、targets 是固定输入（同 figs.js 里 TRAINED 常量的做法），
     但 loss、缓存、梯度、数值对拍全部当场算。 */

  const CP = {
    W: [[0.35, -0.62, 0.18], [-0.44, 0.27, 0.81], [0.12, 0.53, -0.36], [-0.29, -0.15, 0.47]],   // (4,3)
    b: [0.10, -0.20, 0.05, 0.15],
    x1: [[0.70, -1.20, 0.40], [-0.30, 0.90, -1.50]],
    x2: [[1.90, 0.20, -0.80], [-1.10, 0.50, 1.30]],
    targets: [2, 0],
  };

  function cpForward(st, x, tag) {                        // Linear.forward + CrossEntropy.forward
    st.lin._x_flat = x; st.lin.tag = tag;                 // 缓存写入（第二次 forward 会覆盖）
    const R = x.length, V = st.W.length;
    const logits = [];
    for (let r = 0; r < R; r++) {
      const row = [];
      for (let k = 0; k < V; k++) {
        let s = st.b[k];
        for (let j = 0; j < 3; j++) s += x[r][j] * st.W[k][j];
        row.push(s);
      }
      logits.push(row);
    }
    const m = logits.map(row => Math.max.apply(null, row));
    const e = logits.map((row, r) => row.map(v => Math.exp(v - m[r])));
    const Z = e.map(row => row.reduce((s, v) => s + v, 0));
    st.ce._probs = e.map((row, r) => row.map(v => v / Z[r]));
    st.ce.tag = tag; st.ce._count = R;                    // N = targets.size = B·T
    let nll = 0;
    for (let r = 0; r < R; r++) nll += -(logits[r][CP.targets[r]] - m[r] - Math.log(Z[r]));
    return nll / R;                                       // float(np.mean(nll))
  }
  function cpBackward(st) {                               // 用的永远是"当前缓存"，不管它是谁的
    const P = st.ce._probs, R = P.length, V = P[0].length, N = st.ce._count;
    const g = P.map((row, r) => row.map((p, k) => (p - (k === CP.targets[r] ? 1 : 0)) / N));
    for (let k = 0; k < V; k++) {
      for (let j = 0; j < 3; j++) {
        let s = 0; for (let r = 0; r < R; r++) s += g[r][k] * st.lin._x_flat[r][j];
        st.gW[k][j] += s;                                 // Parameter.grad 是 +=，累加
      }
      let sb = 0; for (let r = 0; r < R; r++) sb += g[r][k];
      st.gb[k] += sb;
    }
  }
  function cpLossOf(Wm, bm, x) {                          // 纯函数版 forward（给有限差分扰动用）
    const R = x.length, V = Wm.length;
    let nll = 0;
    for (let r = 0; r < R; r++) {
      const row = [];
      for (let k = 0; k < V; k++) {
        let s = bm[k];
        for (let j = 0; j < 3; j++) s += x[r][j] * Wm[k][j];
        row.push(s);
      }
      const m = Math.max.apply(null, row);
      const e = row.map(v => Math.exp(v - m));
      const Z = e.reduce((s, v) => s + v, 0);
      nll += -(row[CP.targets[r]] - m - Math.log(Z));
    }
    return nll / R;
  }
  function cpNumeric() {                                  // ∂L(x①)/∂θ，中心差分 ε=1e-5（阶段 4 起判据的取值，公式同 gradcheck.py）
    const eps = 1e-5;
    const nW = CP.W.map(r => r.slice()), nb = CP.b.slice();
    const gW = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], gb = [0, 0, 0, 0];
    for (let k = 0; k < 4; k++) for (let j = 0; j < 3; j++) {
      const s = nW[k][j];
      nW[k][j] = s + eps; const hi = cpLossOf(nW, nb, CP.x1);
      nW[k][j] = s - eps; const lo = cpLossOf(nW, nb, CP.x1);
      nW[k][j] = s; gW[k][j] = (hi - lo) / (2 * eps);
    }
    for (let k = 0; k < 4; k++) {
      const s = nb[k];
      nb[k] = s + eps; const hi = cpLossOf(nW, nb, CP.x1);
      nb[k] = s - eps; const lo = cpLossOf(nW, nb, CP.x1);
      nb[k] = s; gb[k] = (hi - lo) / (2 * eps);
    }
    return { gW, gb };
  }
  function freshState() {
    return {
      W: CP.W.map(r => r.slice()), b: CP.b.slice(),
      lin: { _x_flat: null, tag: 0 }, ce: { _probs: null, tag: 0, _count: 2 },
      gW: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], gb: [0, 0, 0, 0],
    };
  }
  const CP_SCHEDULES = [
    { key: 's1', name: '① fwd(x①) → bwd', ops: [['fwd', 1], ['bwd', 0], ['check', 0]], verdict: '对：缓存就是 x① 的，梯度对得上' },
    { key: 's2', name: '② fwd(x①) → fwd(x②) → bwd', ops: [['fwd', 1], ['fwd', 2], ['bwd', 0], ['check', 0]], verdict: '错：第二次 forward 把 _x_flat / _probs 覆盖成 x② 的，反向算的是<b>另一条数据</b>的梯度 —— 形状全对、不报错' },
    { key: 's3', name: '③ fwd(x①) → bwd → bwd', ops: [['fwd', 1], ['bwd', 0], ['bwd', 0], ['check', 0]], verdict: '错：第二次反向把用过的缓存又摊了一遍，+= 让梯度精确翻倍' },
  ];
  function cpRun(sch) {                                   // 播完全部 op 后的状态 + 对拍（纯函数）
    const st = freshState();
    for (const [op, tag] of sch.ops) {
      if (op === 'fwd') cpForward(st, tag === 1 ? CP.x1 : CP.x2, tag);
      else if (op === 'bwd') cpBackward(st);
    }
    const nm = cpNumeric();
    let worst = 0, worstAt = '';
    for (let k = 0; k < 4; k++) for (let j = 0; j < 3; j++) {
      const rel = Math.abs(nm.gW[k][j] - st.gW[k][j]) / Math.max(1e-8, Math.abs(nm.gW[k][j]), Math.abs(st.gW[k][j]));
      if (rel > worst) { worst = rel; worstAt = `W.grad[${k}][${j}]`; }
    }
    for (let k = 0; k < 4; k++) {
      const rel = Math.abs(nm.gb[k] - st.gb[k]) / Math.max(1e-8, Math.abs(nm.gb[k]), Math.abs(st.gb[k]));
      if (rel > worst) { worst = rel; worstAt = `b.grad[${k}]`; }
    }
    return { st, nm, worst, worstAt };
  }

  R.cachepair = function (box) {
    injectStyle();
    box.classList.add('lane-c');
    F.title(box, '一次 forward 配一次 backward：三种调用排法下缓存的命运（Linear(3→4) → CrossEntropy，loss 与梯度全部真算）');

    let sch = CP_SCHEDULES[0];
    const TAGCOL = { 0: '#7fb3e8', 1: '#8fd6b0', 2: '#f0c07a' };
    const TAGNAME = { 0: '（空）', 1: 'x①', 2: 'x②' };

    /* 排法按钮 */
    const sbar = el('div', 'readout');
    const sbs = CP_SCHEDULES.map(s => {
      const b = el('button', s === sch ? 'on' : '', s.name);
      b.onclick = () => { sch = s; sbs.forEach((o, i) => o.classList.toggle('on', CP_SCHEDULES[i] === s)); paintSeq(); anim.stop(); anim.show(0); };
      sbar.appendChild(b); return b;
    });
    box.appendChild(sbar);

    /* 调用时间线（格子 = 一次调用，高亮 = 当前步） */
    const seq = el('div', 'lane-seq'); box.appendChild(seq);
    const opEls = [];
    function paintSeq() {
      seq.textContent = ''; opEls.length = 0;
      seq.appendChild(el('span', 'lane-tag', '调用顺序 →'));
      sch.ops.forEach(([op, tag]) => {
        const o = el('span', 'op', op === 'fwd' ? `forward(${TAGNAME[tag]})` : op === 'bwd' ? 'backward()' : '对拍 gradcheck');
        seq.appendChild(o); opEls.push(o);
      });
    }
    const ctl = el('div'); box.appendChild(ctl);          // 播放条的位置先占住

    /* 缓存/梯度舞台 */
    const stg = F.stage(box, 176);
    const cacheCells = [], gradCells = [];
    [[16, 'Linear._x_flat（第0行）'], [70, 'CE._probs（第0行）'], [124, 'W.grad 第0行 · b.grad[0]']].forEach(([y, txt]) => {
      const s = el('div', 'lane-abs', txt); s.style.top = y + 'px'; s.style.left = '0px'; s.style.minWidth = '130px';
      stg.appendChild(s);
    });
    for (let j = 0; j < 3; j++) cacheCells.push(F.cellEl(stg, '', 'w2'));
    for (let k = 0; k < 4; k++) cacheCells.push(F.cellEl(stg, '', 'w2'));
    for (let i = 0; i < 4; i++) gradCells.push(F.cellEl(stg, '', 'w2'));

    const verdict = el('div', 'lane-verdict'); box.appendChild(verdict);
    const summary = el('div', 'lane-verdict'); box.appendChild(summary);

    /* stepAt(t)：从空状态重放前 t+1 个调用 */
    function replay(t) {
      const st = freshState();
      for (let i = 0; i <= t && i < sch.ops.length; i++) {
        const [op, tag] = sch.ops[i];
        if (op === 'fwd') cpForward(st, tag === 1 ? CP.x1 : CP.x2, tag);
        else if (op === 'bwd') cpBackward(st);
      }
      return st;
    }

    function draw(t) {
      const n = sch.ops.length, ti = Math.max(0, Math.min(t, n - 1));
      opEls.forEach((o, i) => { o.className = 'op' + (i < ti ? ' done' : '') + (i === ti ? ' now' : ''); });
      const st = replay(ti);
      const xf = st.lin._x_flat, pb = st.ce._probs;
      for (let j = 0; j < 3; j++) {
        const c = cacheCells[j];
        c.textContent = xf ? fmt(xf[0][j], 2) : '—';
        c.style.background = TAGCOL[st.lin.tag];
        c.className = 'cell w2' + (xf && st.lin.tag === 2 ? ' hot' : '');
        F.move(c, 162 + j * 82, 10);
      }
      for (let k = 0; k < 4; k++) {
        const c = cacheCells[3 + k];
        c.textContent = pb ? fmt(pb[0][k], 3) : '—';
        c.style.background = TAGCOL[st.ce.tag];
        c.className = 'cell w2' + (pb && st.ce.tag === 2 ? ' hot' : '');
        F.move(c, 162 + k * 82, 64);
      }
      for (let i = 0; i < 4; i++) {
        const c = gradCells[i];
        const v = i < 3 ? st.gW[0][i] : st.gb[0];
        c.textContent = v === 0 ? '0' : fmt(v, 3);
        c.className = 'cell w2' + (v !== 0 ? ' summed' : ' off');
        F.move(c, 162 + i * 82, 118);
      }
      if (sch.ops[ti][0] === 'check') {
        const r = cpRun(sch);
        verdict.innerHTML = '<b>对拍（gradcheck.py 同款公式，ε=1e-5；数值梯度按 ∂L(x①)/∂θ 逐元素真算）：</b>'
          + `<span>最大相对误差 <b style="color:${r.worst <= 1e-4 ? '#3ecf8e' : '#ff6b6b'}">${sci(r.worst)}</b>（在 ${r.worstAt}）`
          + ` → ${r.worst <= 1e-4 ? '<b style="color:#3ecf8e">通过（tol=1e-4）</b>' : '<b style="color:#ff6b6b">对不上 —— 但没有任何异常抛出</b>'}</span>`
          + `<span>${sch.verdict}</span>`;
      } else verdict.innerHTML = '<span style="color:#8d99ac">格子颜色 = 这份缓存/梯度来自哪次调用：绿是 x①、橙是 x②（橙格描边 = 刚被覆盖）。</span>';
    }

    function desc(t) {
      const n = sch.ops.length, ti = Math.max(0, Math.min(t, n - 1));
      const [op, tag] = sch.ops[ti];
      if (op === 'fwd') {
        return `第 ${ti + 1}/${n} 步 · forward(${TAGNAME[tag]})：Linear 把 _x_flat 写成 ${TAGNAME[tag]} 的行，CE 把 _probs 写成 ${TAGNAME[tag]} 的 softmax`
          + (tag === 2 ? ' —— <b>x① 的缓存被覆盖，悄无声息，没有任何报错</b>' : '');
      }
      if (op === 'bwd') {
        const st = replay(ti);
        const reused = ti > 0 && sch.ops[ti - 1][0] === 'bwd';
        return `第 ${ti + 1}/${n} 步 · backward()：CE 用 _probs(${TAGNAME[st.ce.tag]}) 摊 (p−onehot)/N，Linear 用 _x_flat(${TAGNAME[st.lin.tag]}) 算 W.grad += gᵀ·x_flat`
          + (st.lin.tag === 2 ? ' —— 反向读到的已经是 <b>x②</b> 的中间量，梯度算到了另一条数据头上'
            : reused ? ' —— 用的是<b>已经摊过一遍</b>的缓存，+= 让 W.grad 精确翻倍' : '');
      }
      const r = cpRun(sch);
      return `第 ${n}/${n} 步 · 对拍：解析梯度 vs ∂L(x①)/∂θ 的数值梯度，最大相对误差 ${sci(r.worst)}（${r.worst <= 1e-4 ? '通过' : '对不上'}）。`;
    }

    paintSeq();
    const anim = animBar(ctl, () => sch.ops.length, t => { draw(t); return desc(t); }, 1000);

    summary.innerHTML = '<b>三种排法的结局（挂载时就真算好：解析 vs 数值的最大相对误差）：</b>'
      + CP_SCHEDULES.map(s => {
        const r = cpRun(s);
        return `<span style="color:${r.worst <= 1e-4 ? '#3ecf8e' : '#ff6b6b'}">${s.name}：${sci(r.worst)}</span>`;
      }).join(' ｜ ');

    const w1 = sci(cpRun(CP_SCHEDULES[0]).worst), w2 = sci(cpRun(CP_SCHEDULES[1]).worst), w3 = sci(cpRun(CP_SCHEDULES[2]).worst);
    F.caption(box, '纪律原文在 s03a §4.1 / §4.6：backward 需要的中间量存在 `self._x_flat`、`self._probs`、`self._r` 上，'
      + '<b>中间插入第二次 forward，缓存就被覆盖</b> —— 排法②的梯度算到了 x② 头上，形状全对、loss 照降、没有任何报错，'
      + '本页实测最大相对误差 ' + w2 + '。排法③是"重复反向"：`Parameter.grad` 是 `+=`，同一份缓存摊两遍，梯度精确翻倍，'
      + '对拍给出恰好 ' + w3 + '（=|2g−g|/|2g|，同 s03c §4.3 的"常数因子"规则）。只有排法①是干净的 ' + w1 + ' 量级 —— '
      + 'float64 下 eps=1e-5 的噪声底（≈ulp(L)/ε，见 gradcheck.py compare() 的注释），正是阶段 4 起判据 '
      + '`eps=1e-5, atol=1e-9` 那组取值。阶段 7 的梯度累积必须 '
      + 'fwd→bwd 严格交替、攒完再 optimizer.step()，就是这个原因。');
  };
})();
