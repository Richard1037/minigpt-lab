/* 讲义的可视化组件。八个，全部零依赖、纯前端、当场算 —— 不加载任何图表库，
   也不放"示意数据"冒充测量：能算的都真的算（梯度下降、掩码、拆头、台阶叠加），
   算不了的（gradcheck 的误差量级）在图注里写明是本项目实测的数。

   挂载方式：md_to_html.py 在对应小节末尾留一个 <div class="fig" data-fig="名字">，
   这里按名字找组件。任何一个组件抛错都只是"这张图没出来"，正文照常可读 ——
   这是这套讲义的底线：图是辅助，判据才是真相。 */

(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (t, cls, html) => { const e = document.createElement(t); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const sv = (t, at) => { const e = document.createElementNS(NS, t); for (const k in at) e.setAttribute(k, at[k]); return e; };
  const fmt = (v, n = 3) => Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(1) : v.toFixed(n);

  /* 图注/标题走 innerHTML，而组件是在 MathJax 排版**之后**才插进 DOM 的 ——
     所以组件文案里：markdown 反引号转成 <code>，公式一律写成 unicode 明文
     （`$…$` 在这里不会被排版，只会露出反斜杠）。 */
  const md = s => String(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  function title(box, text) { box.appendChild(el('div', 'fig-title', md(text))); }
  function caption(box, text) { box.appendChild(el('div', 'cap', md(text))); }

  /* 画布：把 x∈[lo,hi]、y∈[ylo,yhi] 映射到像素。返回的对象可以反复 clear + 重画，
     这样 playground 拖动滑块时不用重建 canvas（重建会闪，也会丢事件）。 */
  function plot(box, w, h, xlo, xhi, ylo, yhi) {
    let c = box.querySelector('canvas');
    if (!c) { c = el('canvas'); box.insertBefore(c, box.firstChild); }
    c.width = w * 2; c.height = h * 2; c.style.width = w + 'px'; c.style.height = h + 'px';
    const g = c.getContext('2d');
    const X = x => (x - xlo) / (xhi - xlo) * (w - 46) + 40;
    const Y = y => h - 22 - (y - ylo) / (yhi - ylo) * (h - 34);
    const api = {
      clear() {
        g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height); g.scale(2, 2);
        g.strokeStyle = '#28313f'; g.lineWidth = 1; g.font = '10px ui-monospace,Consolas'; g.fillStyle = '#8d99ac';
        for (let i = 0; i <= 4; i++) {
          const xv = xlo + (xhi - xlo) * i / 4, yv = ylo + (yhi - ylo) * i / 4;
          g.beginPath(); g.moveTo(X(xv), 8); g.lineTo(X(xv), h - 22); g.stroke();
          g.beginPath(); g.moveTo(40, Y(yv)); g.lineTo(w - 6, Y(yv)); g.stroke();
          g.fillText(xv.toFixed(1), X(xv) - 9, h - 9);
          g.fillText(fmt(yv, 2), 4, Y(yv) + 3);
        }
        g.strokeStyle = '#4a5568'; g.beginPath(); g.moveTo(40, Y(0)); g.lineTo(w - 6, Y(0)); g.stroke();
      },
      line(fn, color, dash, wide) {
        g.save(); g.strokeStyle = color; g.lineWidth = wide ? 2.4 : 1.6;
        if (dash) g.setLineDash(dash); g.beginPath();
        let started = false;
        for (let px = 40; px <= w - 6; px++) {
          const x = xlo + (px - 40) / (w - 46) * (xhi - xlo), y = fn(x);
          if (!isFinite(y)) { started = false; continue; }
          const yy = Math.max(-1e4, Math.min(1e4, Y(y)));
          started ? g.lineTo(px, yy) : (g.moveTo(px, yy), started = true);
        }
        g.stroke(); g.restore();
      },
      dots(xs, fn, color) {
        g.save(); g.fillStyle = color;
        // beginPath() 返回 undefined，不能链式 .arc() —— 那样每个点都抛 TypeError，
        // 整个组件被 mountAll 的 try/catch 吞掉，你只看到"这张图没渲染出来"。
        for (const x of xs) { g.beginPath(); g.arc(X(x), Y(fn(x)), 3, 0, 7); g.fill(); }
        g.restore();
      }
    };
    api.clear();
    return api;
  }

  // 同一份数据、同一条更新式，JS 里再跑一遍：图里的曲线就是你代码会跑出来的东西
  function s01data() {
    const n = 100, xs = [], ys = [];
    for (let i = 0; i < n; i++) { const x = -1 + 2 * (i / (n - 1)); xs.push(x); ys.push(3 * x + 1); }
    return { xs, ys, n };
  }
  function fitLine(xs, ys, lr, iters) {                    // y = a·x + b 的批量梯度下降
    let a = 0, b = 0; const curve = [];
    for (let it = 0; it < iters; it++) {
      let e2 = 0, ga = 0, gb = 0;
      for (let i = 0; i < xs.length; i++) {
        const e = a * xs[i] + b - ys[i];
        e2 += e * e; ga += 2 * e * xs[i]; gb += 2 * e;
      }
      curve.push(e2 / xs.length); a -= lr * ga / xs.length; b -= lr * gb / xs.length;
    }
    return { a, b, curve };
  }

  /* ---------------------------------------------------------------- 1. 计算图 */
  function graph(box) {
    title(box, '§6 · 一张图走完正反两趟：(a·b)² 的四个节点');
    const a = 2, b = 3, p = a * b, L = p * p;
    const rows = [
      { id: 'a', x: 40, y: 120, v: a, g: 36, label: 'a' },
      { id: 'b', x: 40, y: 210, v: b, g: 24, label: 'b' },
      { id: 'p', x: 210, y: 165, v: p, g: 12, label: 'p = a·b' },
      { id: 'L', x: 380, y: 165, v: L, g: 1, label: 'L = p²' }];
    const svg = sv('svg', { viewBox: '0 0 520 300', width: 520 });
    const edge = (x1, y1, x2, y2, txt) => {
      svg.appendChild(sv('line', { x1, y1, x2, y2, stroke: '#3a4657', 'stroke-width': 1.6, class: 'edge' }));
      // appendChild 返回的是"被挂上去的那个节点"，不是父节点 ——
      // 所以 .lastChild 在一个刚建好的空 <text> 上是 null，写 textContent 就炸。
      const t = sv('text', { x: (x1 + x2) / 2 - 14, y: (y1 + y2) / 2 - 4, fill: '#8d99ac', 'font-size': 11 });
      t.textContent = txt;
      svg.appendChild(t);
    };
    edge(78, 120, 178, 158, '·'); edge(78, 210, 178, 172, '·'); edge(248, 165, 330, 165, '²');
    rows.forEach(r => {
      const g = sv('g', { class: 'node', 'data-id': r.id });
      g.appendChild(sv('rect', { x: r.x, y: r.y - 26, width: 84, height: 52, rx: 9,
        fill: '#1a212b', stroke: '#3a4657' }));
      const t1 = sv('text', { x: r.x + 8, y: r.y - 6, fill: '#cfe0ff', 'font-size': 12 });
      t1.textContent = r.label; g.appendChild(t1);
      const t2 = sv('text', { x: r.x + 8, y: r.y + 14, fill: '#e6ebf3', 'font-size': 12, class: 'val' });
      t2.textContent = 'data=' + fmt(r.v, 2); g.appendChild(t2);
      const t3 = sv('text', { x: r.x + 8, y: r.y + 30, fill: '#3ecf8e', 'font-size': 11, class: 'grd', opacity: 0 });
      t3.textContent = 'grad=' + fmt(r.g, 2); g.appendChild(t3);
      svg.appendChild(g);
    });
    box.appendChild(svg);
    const bar = el('div', 'readout');
    const b1 = el('button', 'on', '① 正向搭图'); const b2 = el('button', '', '② 反向摊梯度');
    bar.append(b1, b2); box.appendChild(bar);
    const setMode = fwd => {
      b1.classList.toggle('on', fwd); b2.classList.toggle('on', !fwd);
      svg.querySelectorAll('.grd').forEach(t => t.setAttribute('opacity', fwd ? 0 : 1));
      svg.querySelectorAll('.val').forEach(t => t.setAttribute('opacity', fwd ? 1 : 0.45));
      svg.querySelectorAll('.node rect').forEach((r, i) => {
        r.setAttribute('stroke', fwd ? '#3a4657' : ['#3ecf8e', '#3ecf8e', '#4aa3ff', '#f5b445'][i]);
      });
    };
    b1.onclick = () => setMode(true); b2.onclick = () => setMode(false); setMode(true);
    caption(box, '反向那一趟的顺序是 L → p → {a,b}（逆拓扑）。起点 L.grad=1 漏掉的话，'
      + '后面每个 grad 都是 0，而且不报错。a 的梯度 36 = 2p·b，b 的 24 = 2p·a —— '
      + '和 §2.4 手算的数字完全一致，写完 TODO-3 就拿这两个数对。');
  }

  /* ---------------------------------------------------------------- 2. loss 曲线 */
  function losscurve(box) {
    title(box, '§2 · 同一个问题、四种学习率，各跑 3000 步（JS 里现算）');
    const { xs, ys } = s01data();
    const w = 620, h = 240;
    const runs = [[0.001, '#8d99ac', 'lr=0.001 太小'], [0.2, '#3ecf8e', 'lr=0.2 刚好'],
                  [0.55, '#f5b445', 'lr=0.55 临界'], [1.5, '#ff6b6b', 'lr=1.5 发散']];
    const data = runs.map(([lr]) => fitLine(xs, ys, lr, 3000).curve);
    const top = Math.min(12, Math.max(...data.flat().slice(0, 200).map(v => Math.min(v, 12))));
    const g = plot(box, w, h, 0, 3000, 0, top);
    data.forEach((c, i) => g.line(x => Math.min(c[Math.floor(x / 3000 * (c.length - 1))] || c[c.length - 1], top),
      runs[i][1], runs[i][0] === 1.5 ? [5, 4] : null));
    const leg = el('div', 'readout');
    runs.forEach(([lr, color, name], i) => {
      const s = el('span'); s.innerHTML = `<b style="color:${color}">●</b> ${name} · 末 loss `
        + `<b>${fmt(data[i][data[i].length - 1])}</b> · a=<b>${fmt(fitLine(xs, ys, lr, 3000).a, 3)}</b>`;
      leg.appendChild(s);
    });
    box.appendChild(leg);
    caption(box, '判据里"loss 下降倍数 ≥ 1e3"就是这条绿线的性质。红线不是"训得慢"，是发散的——'
      + '它一开始就往上炸，症状是 loss 变成 3.9e+186 这种数。稳定上限由数据的 mean(2x²) 量级决定，'
      + '不是某个魔法数字。');
  }

  /* ---------------------------------------------------------------- 3. 台阶 playground */
  const TRAINED = {  // seed=2, n_hidden=8, iters=6000, lr=0.1, decay=3.0，用 solutions/ 跑出来的
    W1: [0.21883, -0.84178, -0.21876, -2.31745, 1.20237, 0.84987, -0.21878, 0.84368],
    W2: [-1.272, -1.09953, 1.69937, 0.5541, 0.38966, 0.92108, 1.53506, 1.02134],
    b1: [0, 0, 0, 0, 0, 0, 0, 0], b2: 0, mse: 0.004483 };
  const LIN_MSE = 0.216869;                                  // 12 个点上最小二乘直线的 MSE（容量墙）

  function steps(box) {
    title(box, '§2.2 · 8 个 S 形台阶能不能拼出 sin(x)？拖一拖就知道（这就是"容量"）');
    const xs = [], ys = {};
    for (let i = 0; i < 12; i++) { const x = -3 + 6 * i / 11; xs.push(x); ys[x] = Math.sin(x); }
    const yOf = x => Math.sin(x);
    const P = [];                                            // 每个台阶：拐点 c、高度 h、陡度 k
    for (let j = 0; j < 8; j++) P.push({ c: -2.6 + j * 0.75, h: 0.5, k: 3 });
    const model = x => P.reduce((s, p) => s + p.h * Math.tanh(p.k * (x - p.c)), 0);
    const mse = () => xs.reduce((s, x) => s + (model(x) - yOf(x)) ** 2, 0) / xs.length;

    const wrap = el('div', 'row');
    const left = el('div', 'col'), right = el('div', 'col');
    wrap.append(left, right); box.appendChild(wrap);
    const g = plot(left, 430, 260, -3.4, 3.4, -1.6, 1.6);
    const out = el('div', 'readout'); left.appendChild(out);
    const sliders = el('div', 'sliders'); right.appendChild(sliders);

    function redraw() {
      g.clear();
      P.forEach(p => g.line(x => p.h * Math.tanh(p.k * (x - p.c)), 'rgba(74,163,255,.35)'));
      g.line(yOf, '#f5b445', [3, 3]);                       // 真 sin
      g.line(x => model(x), '#3ecf8e', null, true);         // 8 个台阶之和
      g.dots(xs, yOf, '#e6ebf3');
      const m = mse();
      out.innerHTML = `<span>当前 MSE <b>${fmt(m, 4)}</b></span>`
        + `<span>线性最佳 MSE <b style="color:#ff6b6b">${fmt(LIN_MSE, 4)}</b>（容量墙）</span>`
        + `<span>训练结果 <b style="color:#3ecf8e">${fmt(TRAINED.mse, 4)}</b></span>`
        + `<span>${m < LIN_MSE ? '<b style="color:#3ecf8e">已经翻过墙了</b>'
            : '<b style="color:#ff6b6b">还不如一条直线</b>'}</span>`;
    }

    P.forEach((p, j) => {
      const lab = el('label');
      lab.innerHTML = `<b>台阶${j}</b>`;
      const sc = el('input'); sc.type = 'range'; sc.min = -3; sc.max = 3; sc.step = 0.05; sc.value = p.c;
      const sh = el('input'); sh.type = 'range'; sh.min = -2; sh.max = 2; sh.step = 0.05; sh.value = p.h;
      sc.oninput = () => { p.c = +sc.value; redraw(); };
      sh.oninput = () => { p.h = +sh.value; redraw(); };
      lab.append(sc, el('span', null, '位置'), sh, el('span', null, '高度'));
      sliders.appendChild(lab);
      p._sc = sc; p._sh = sh;
    });

    const btns = el('div', 'readout');
    const bLoad = el('button', '', '载入训练结果（seed=2）');
    const bReset = el('button', '', '归零重来');
    bLoad.onclick = () => {
      P.forEach((p, j) => { p.c = -TRAINED.b1[j] / TRAINED.W1[j]; p.h = TRAINED.W2[j]; p.k = Math.abs(TRAINED.W1[j]) * 2; });
      sync(); redraw();
    };
    bReset.onclick = () => { P.forEach(p => { p.c = 0; p.h = 0; }); sync(); redraw(); };
    btns.append(bLoad, bReset); box.appendChild(btns);
    function sync() { P.forEach(p => { p._sc.value = p.c; p._sh.value = p.h; }); }

    redraw();
    caption(box, '绿线是你的 8 个台阶之和，黄虚线是真 sin，灰蓝细线是单个台阶。'
      + '把 8 个台阶全归零 —— 那就是"没有非线性"的样子：什么都表达不了。'
      + '「载入训练结果」是 solutions 里那个 25 参数 MLP 真跑 6000 步之后的权重；'
      + '你会看到 8 个拐点全在 x=0 —— 因为 sin 是奇函数、数据又对称，b1 的梯度两两抵消，'
      + '网络发现"不用挪台阶，只用调高度"就够了。这个反直觉的现象，判据里看不到，图里看得到。');
  }

  /* ---------------------------------------------------------------- 4. 容量 vs lr */
  function capacity(box) {
    title(box, '§2.10 · 三条曲线分清"容量不够"和"lr 没调好"（JS 里真跑）');
    const xs = [], ys = [];
    for (let i = 0; i < 12; i++) { const x = -3 + 6 * i / 11; xs.push(x); ys.push(Math.sin(x)); }
    function train(steps, lr, hidden) {                     // hidden=0 → 线性；8 → 台阶模型
      let a = 0, b = 0, P = [];
      if (hidden) for (let j = 0; j < hidden; j++) P.push({ c: -2.6 + j * 0.75, h: 0.3, k: 3 });
      const curve = [];
      for (let it = 0; it < steps; it++) {
        let L = 0, grads = [];
        if (!hidden) {
          let ga = 0, gb = 0;
          for (let i = 0; i < xs.length; i++) {
            const e = a * xs[i] + b - ys[i]; L += e * e; ga += 2 * e * xs[i]; gb += 2 * e;
          }
          L /= xs.length; a -= lr * ga / xs.length; b -= lr * gb / xs.length;
        } else {
          const acc = P.map(() => ({ c: 0, h: 0 })); let d = 0;
          for (let i = 0; i < xs.length; i++) {
            const x = xs[i], t = P.map(p => Math.tanh(p.k * (x - p.c)));
            const y = t.reduce((s, v, j) => s + P[j].h * v, 0);
            const e = y - ys[i]; L += e * e; d += e * e;
            t.forEach((v, j) => {
              acc[j].h += 2 * e * v;
              acc[j].c += 2 * e * P[j].h * (1 - v * v) * (-P[j].k);
            });
          }
          L /= xs.length;
          P.forEach((p, j) => { p.h -= lr * acc[j].h / xs.length; p.c -= lr * acc[j].c / xs.length; });
        }
        curve.push(L);
      }
      return curve;
    }
    const A = train(1500, 0.1, 0), B = train(1500, 0.1, 8), C = train(1500, 0.9, 8);
    const g = plot(box, 620, 250, 0, 1500, 0, 0.45);
    const at = c => x => Math.min(c[Math.max(0, Math.min(1499, Math.floor(x / 1500 * 1499)))] ?? 0.45, 0.45);
    g.line(at(A), '#ff6b6b', null, true);
    g.line(at(B), '#3ecf8e', null, true);
    g.line(at(C), '#f5b445', [5, 4]);
    g.line(() => 0.216869, '#8d99ac', [2, 4]);
    const leg = el('div', 'readout');
    leg.innerHTML = `<span><b style="color:#ff6b6b">●</b> 线性模型 lr=0.1 → 末 <b>${fmt(A[1499], 4)}</b>（平在墙上）</span>`
      + `<span><b style="color:#3ecf8e">●</b> 8 台阶 lr=0.1 → 末 <b>${fmt(B[1499], 4)}</b></span>`
      + `<span><b style="color:#f5b445">●</b> 8 台阶 lr=0.9 → 末 <b>${fmt(C[1499], 4)}</b>（震荡）</span>`
      + `<span>灰虚线 = 最佳直线 MSE <b>${fmt(0.216869, 4)}</b></span>`;
    box.appendChild(leg);
    caption(box, '红和绿用的是<b>同一个 lr</b>，唯一差别是有没有 tanh —— 所以红线那条"平在 0.217 不动"是墙，'
      + '不是"还没训够"。黄线证明另一件事：非线性够了之后，lr 太大照样到不了底。'
      + '以后 loss 卡住，先问"我这条曲线长得像哪条"。');
  }

  /* ---------------------------------------------------------------- 5. 形状数据流 */
  function shapes(box) {
    title(box, '§2 · 一个 token 序列穿过这些层时，形状怎么变（d=64, H=4, L=4, V=2000）');
    const d = 64, H = 4, L = 4, V = 2000, B = 2, T = 32, dh = d / H;
    const rows = [
      ['ids', `(${B}, ${T})`, '整数，不可导'],
      ['Embedding', `(${B}, ${T}, ${d})`, `V·d = ${V * d}`],
      ['RMSNorm ×1', `(${B}, ${T}, ${d})`, `d = ${d}`],
      ['attn q/k/v', `(${B}, ${H}, ${T}, ${dh})`, `4d²+3d = ${4 * d * d + 3 * d}`],
      ['att 分数', `(${B}, ${H}, ${T}, ${T})`, '掩码加在这里（softmax 之前）'],
      ['MLP fc1', `(${B}, ${T}, ${4 * d})`, `4d²+4d`],
      ['MLP fc2', `(${B}, ${T}, ${d})`, `4d²+d`],
      ['RMSNorm + head', `(${B}, ${T}, ${V})`, '权重和 Embedding 共享'],
    ];
    const svg = sv('svg', { viewBox: `0 0 640 ${rows.length * 34 + 20}`, width: 640 });
    rows.forEach((r, i) => {
      const y = 18 + i * 34;
      svg.appendChild(sv('rect', { x: 8, y: y - 13, width: 150, height: 26, rx: 6, fill: '#1a212b', stroke: '#3a4657' }));
      const t = sv('text', { x: 18, y: y + 5, fill: '#cfe0ff', 'font-size': 12 }); t.textContent = r[0]; svg.appendChild(t);
      const s = sv('text', { x: 172, y: y + 5, fill: '#e6ebf3', 'font-size': 12, 'font-family': 'ui-monospace,Consolas' });
      s.textContent = r[1]; svg.appendChild(s);
      const p = sv('text', { x: 330, y: y + 5, fill: '#8d99ac', 'font-size': 11 }); p.textContent = r[2]; svg.appendChild(p);
      if (i < rows.length - 1) svg.appendChild(sv('line', { x1: 83, y1: y + 14, x2: 83, y2: y + 21, stroke: '#3a4657' }));
    });
    box.appendChild(svg);
    const perLayer = 12 * d * d + 10 * d;
    const total = 12 * L * d * d + V * d;
    const read = el('div', 'readout');
    read.innerHTML = `<span>每层 <b>12d²+10d = ${perLayer}</b></span>`
      + `<span>${L} 层 <b>${perLayer * L}</b></span>`
      + `<span>词表 <b>V·d = ${V * d}</b></span>`
      + `<span>合计 <b>${((perLayer * L + V * d) / 1e6).toFixed(2)} M</b></span>`
      + `<span>训练时优化器还要 <b>${(16 * (perLayer * L + V * d) / 1e6).toFixed(1)} MB</b>（16 B/参数）</span>`;
    box.appendChild(read);
    caption(box, '形状是这套代码里最强的约束：想不出下一步的形状，就说明公式没推对。'
      + '注意 (B,T) 之后所有层都保持三维，只有 att 分数那一行是四维的 (B,H,T,T) —— '
      + '那是全网络唯一"token 之间交换信息"的地方，其余层都是逐位置独立的。');
  }

  /* ---------------------------------------------------------------- 6. gradcheck 量级 */
  function gradcheck(box) {
    title(box, '§3 · 有限差分对不上时，误差落在哪个量级（本项目实测）');
    const rows = [
      ['写对了', 1e-10, '#3ecf8e'],
      ['梯度用 = 而不是 +=', 1e0, '#ff6b6b'],
      ['规则里读 self.grad 而非 out.grad', 1e0, '#ff6b6b'],
      ['拓扑序没倒过来', 1e-1, '#f5b445'],
      ['topo 没去重（梯度翻倍）', 1e-1, '#f5b445'],
      ['RMSNorm 少 −r·x̂·s 项', 1e-1, '#f5b445'],
      ['softmax 反向少减均值项', 1e-1, '#f5b445'],
      ['loss 用 sum 不用 mean', 1e0, '#ff6b6b'],
    ];
    const svg = sv('svg', { viewBox: `0 0 640 ${rows.length * 30 + 46}`, width: 640 });
    const X = e => 250 + (Math.log10(Math.max(e, 1e-12)) + 12) / 12 * 360;   // 1e-12 → 250px，1e0 → 610px
    [-12, -10, -8, -6, -4, -2, 0].forEach(p => {
      svg.appendChild(sv('line', { x1: X(10 ** p), y1: 18, x2: X(10 ** p), y2: rows.length * 30 + 20,
        stroke: '#232a35' }));
      const t = sv('text', { x: X(10 ** p) - 12, y: rows.length * 30 + 36, fill: '#8d99ac', 'font-size': 10 });
      t.textContent = '1e' + p; svg.appendChild(t);
    });
    rows.forEach((r, i) => {
      const y = 26 + i * 30;
      const lab = sv('text', { x: 8, y: y + 11, fill: '#c9d6e8', 'font-size': 11.5 });
      lab.textContent = r[0]; svg.appendChild(lab);
      svg.appendChild(sv('rect', { x: 250, y, width: Math.max(2, X(r[1]) - 250), height: 16,
        rx: 3, fill: r[2], opacity: 0.75 }));
      const v = sv('text', { x: Math.min(600, X(r[1]) + 6), y: y + 12, fill: r[2], 'font-size': 11 });
      v.textContent = r[1] >= 1 ? '≫ 阈值' : '1e' + Math.round(Math.log10(r[1])); svg.appendChild(v);
    });
    svg.appendChild(sv('line', { x1: X(1e-4), y1: 12, x2: X(1e-4), y2: rows.length * 30 + 22,
      stroke: '#4aa3ff', 'stroke-dasharray': '4 3' }));
    const th = sv('text', { x: X(1e-4) - 66, y: 12, fill: '#4aa3ff', 'font-size': 11 });
    th.textContent = '判定阈值 1e-4'; svg.appendChild(th);
    box.appendChild(svg);
    caption(box, '绿条和蓝虚线之间就是"过"。左边第一条（1e-10）是参考实现的实际水平；'
      + '注意 <b>1e0 那一类不是"差一点"，是完全没对上</b> —— 梯度全 0 或全错时有限差分必然给出这个量级。'
      + '还有一类错它抓不住：某个参数根本没进图，数值梯度和解析梯度都是 0，"对上了"。'
      + '所以判据里另外有一条"无死参数"。');
  }

  /* ---------------------------------------------------------------- 7. 因果掩码 */
  function mask(box) {
    title(box, '§11.3 · 点一行，看第 i 个 token 被允许看谁');
    const T = 6;
    const grid = el('div', 'grid'); grid.style.gridTemplateColumns = `20px repeat(${T}, 22px)`;
    const corner = el('div', 'axis'); grid.appendChild(corner);
    for (let j = 0; j < T; j++) { const a = el('div', 'axis'); a.textContent = j; grid.appendChild(a); }
    const cells = [];
    for (let i = 0; i < T; i++) {
      const lab = el('div', 'axis'); lab.textContent = i; grid.appendChild(lab);
      for (let j = 0; j < T; j++) {
        const c = el('i'); c.dataset.i = i; c.dataset.j = j; grid.appendChild(c); cells.push(c);
      }
    }
    const read = el('div', 'readout');
    const wrong = el('button', '', '看看"掩码加在 softmax 之后"会怎样');
    let leak = false;
    function paint(sel) {
      cells.forEach(c => {
        const i = +c.dataset.i, j = +c.dataset.j;
        const allowed = j <= i;
        c.className = (allowed ? 'allow' : 'block') + (sel === i ? ' hot' : '');
        if (leak && sel === i) c.className = 'allow' + (sel === i ? ' hot' : '');
      });
      const row = sel == null ? null : Array.from({ length: T }, (_, j) =>
        leak ? '1/' + T : (j <= sel ? '可看' : '0'));
      read.innerHTML = sel == null ? '<span>点左边任意一行</span>'
        : `<span>query i=<b>${sel}</b> 能看到的 key：<b>${row.join(' ')}</b></span>`
        + `<span>${leak ? '<b style="color:#ff6b6b">未来 token 全暴露 —— 模型会抄答案</b>'
            : '被挡住的格子在 softmax 里是 exp(-1e18−max)=<b>精确 0</b>'}</span>`;
    }
    let cur = 2; paint(cur);
    grid.addEventListener('click', e => { if (e.target.tagName === 'I') { cur = +e.target.dataset.i; paint(cur); } });
    wrong.onclick = () => { leak = !leak; wrong.classList.toggle('on', leak); paint(cur); };
    const wrap = el('div', 'row'); const col = el('div', 'col');
    col.appendChild(grid); col.appendChild(read); wrap.appendChild(col);
    box.appendChild(wrap);
    caption(box, '掩码必须加在 softmax <b>之前</b>：加在后面等于没掩（概率已经算出来了，你只是把'
      + '某些格子乘 0，但那些格子的值早就参与了分母）。用 -1e18 而不是 -inf，是因为反向里 '
      + '0×inf=nan 会毁掉整条链。判据里"改未来 token 不影响过去（逐位相同）"那条测的就是这张图。');
  }

  /* ---------------------------------------------------------------- 8. 拆头/合头 */
  function heads(box) {
    title(box, '§11.4 ① · (B,T,C) → 拆头 (B,H,T,hd) → 合头：顺序反了就全错（T=4, C=8, H=4, hd=2）');
    const T = 4, C = 8, H = 4, hd = 2;
    const flat = Array.from({ length: T * C }, (_, k) => k);         // 内存里就是这个顺序：行主序
    const split = [];                                                  // 拆头：先按 (t,h,d) 摊，再换轴
    for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let d = 0; d < hd; d++)
      split.push(t * C + h * hd + d);
    const svg = sv('svg', { viewBox: `0 0 660 150`, width: 660 });
    const cell = (x, y, v, color) => {
      svg.appendChild(sv('rect', { x, y, width: 15, height: 15, rx: 2, fill: color, stroke: '#0f1115' }));
      const t = sv('text', { x: x + 2.5, y: y + 11, 'font-size': 9, fill: '#0b0e13' });
      t.textContent = v; svg.appendChild(t);
    };
    const hue = k => `hsl(${(k % C) * 45}, 62%, ${Math.floor(k / C) % 2 ? 62 : 78}%)`;
    let merged = true;
    function draw() {
      svg.innerHTML = '';
      const order = merged ? flat : split;
      const lab1 = sv('text', { x: 8, y: 22, fill: '#8d99ac', 'font-size': 11 });
      lab1.textContent = merged ? '内存顺序 (B,T,C)：每行 8 个 = 一个 token' : '拆头后 (B,H,T,hd)：同一 token 的 8 个数被分给 4 个头';
      svg.appendChild(lab1);
      order.forEach((k, idx) => cell(8 + (idx % 22) * 20, 34 + Math.floor(idx / 22) * 20, k, hue(k)));
      const note = sv('text', { x: 8, y: 132, fill: '#c9d6e8', 'font-size': 11 });
      note.textContent = merged
        ? 'y.reshape(B,T,H,hd).transpose(0,2,1,3) —— 先 reshape 再 transpose'
        : '合头是逆操作：transpose(0,2,1,3) 再 reshape(B,T,C)，数字必须回到原位';
      svg.appendChild(note);
    }
    const btn = el('button', '', '切换：合头 ⇄ 拆头');
    btn.onclick = () => { merged = !merged; draw(); };
    box.appendChild(svg); box.appendChild(btn);
    draw();
    caption(box, 'GIV-2 那条检查测的就是"拆了再合回去必须逐元素相等"。'
      + '顺序写成 transpose 在前、reshape 在后，形状照样对、数字全错位 —— '
      + '而 loss 依然会降（模型只是把 token 内部维度读乱了），只有这条可逆性检查能抓。');
  }

  /* ==========================================================================
     下面五个是"板子系"组件：把 reshape / sum(axis) / scatter-add / 归一化耦合 /
     减最大值 这几件在纸上很难想的事，做成能点、能拖、当场算的动画。
     共同底线：所有数字都在浏览器里真算（JS 的 number 就是 float64，溢出/下溢
     和 numpy 行为一致），不放示意值。                                     */
  const CW = 40, CH = 26;                                  // 一格 38×24 + 2px 间隙
  function cellEl(stage, txt, cls) {
    const c = el('div', 'cell' + (cls ? ' ' + cls : ''));
    c.textContent = txt; stage.appendChild(c); return c;
  }
  function move(c, x, y) { c.style.transform = `translate(${x}px,${y}px)`; }

  /* ---------------------------------------------------------------- 9. 摊平动画 */
  function flatten(box) {
    title(box, '§2.2 · 两块 3×4 的板子拼成一块 6×4：reshape 只是把前两轴乘成一根');
    const B = 2, T = 3, C = 4, O = 2;
    const W = [[1, 1, 0, 1], [0, 1, 1, 0]];                // (out,in) —— 和 nn.py 的存法一致
    const rows = [];
    for (let n = 0; n < B * T; n++) rows.push(Array.from({ length: C }, (_, k) => n * C + k));
    const mul = r => W.map(w => r.reduce((s, v, k) => s + v * w[k], 0));
    const y = rows.map(mul);                              // 拼起来一次乘
    const perBoard = [];                                  // 你脑子里那个算法：逐块乘再摞起来
    for (let b = 0; b < B; b++) perBoard.push(...rows.slice(b * T, b * T + T).map(mul));
    const same = Math.max(...y.map((r, n) => Math.max(...r.map((v, j) => Math.abs(v - perBoard[n][j])))));

    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); box.appendChild(stage);
    const xc = [], yc = [], wc = [];
    rows.forEach((r, n) => r.forEach((v, k) => {
      const c = cellEl(stage, v, n < T ? '' : 'b1');
      c.n = n; c.k = k; c.style.cursor = 'pointer';
      c.onclick = () => { hot = (hot === n ? null : n); draw(); };
      xc.push(c);
    }));
    y.forEach((r, n) => r.forEach((v, j) => { const c = cellEl(stage, v, 'out'); c.n = n; c.k = j; yc.push(c); }));
    W.forEach((r, o) => r.forEach(v => { const c = cellEl(stage, v, 'w'); wc.push(c); }));
    wc.forEach((c, idx) => { c.o = (idx / C) | 0; c.k = idx % C; });

    const read = el('div', 'readout'); box.appendChild(read);
    const lb0 = el('div', 'slab'), lb1 = el('div', 'slab'), lbW = el('div', 'slab');
    stage.append(lb0, lb1, lbW);
    const btns = el('div', 'readout'); box.appendChild(btns);
    let st = 0, hot = null;
    const NAMES = ['x 还是两块板 (2,3,4)', '拼成 (6,4) —— 前 3 行是 b=0，后 3 行是 b=1',
                   '×Wᵀ → (6,2)', '拆回 (2,3,2)', '非法切法：reshape 成 (4,6)'];

    function draw() {
      head.innerHTML = `<b>${NAMES[st]}</b>`;
      xc.forEach(c => {
        const n = +c.n, k = +c.k, b = (n / T) | 0, r = n % T, f = n * C + k;
        if (st === 0) move(c, 16 + b * (C * CW + 56) + k * CW, 40 + r * CH);
        else if (st === 4) move(c, 16 + (f % 6) * CW, 40 + ((f / 6) | 0) * CH);
        else move(c, 16 + k * CW, 40 + n * CH);
        c.classList.toggle('gone', st === 3);
        c.classList.toggle('hot', hot === n);
      });
      yc.forEach(c => {
        const n = +c.n, j = +c.k, b = (n / T) | 0;
        c.classList.toggle('gone', st !== 2 && st !== 3);
        move(c, st === 3 ? 250 + b * (O * CW + 56) + j * CW : 250 + j * CW,
             40 + (st === 3 ? (n % T) : n) * CH);
        c.classList.toggle('hot', hot === n);
      });
      wc.forEach(c => {
        c.classList.toggle('gone', st !== 2);
        move(c, 520 + c.k * CW, 40 + c.o * CH);
      });
      lb0.textContent = st === 3 ? '拆回 b=0' : '板子 b=0';
      move(lb0, 16 + (st === 3 ? 250 : 0), 20);
      lb0.classList.toggle('gone', st !== 0 && st !== 3);
      lb1.textContent = st === 3 ? '拆回 b=1' : '板子 b=1';
      move(lb1, (st === 3 ? 250 + O * CW + 56 : C * CW + 56 + 16), 20);
      lb1.classList.toggle('gone', st !== 0 && st !== 3);
      lbW.textContent = 'W (out,in)=(2,4) —— 乘的是它的转置 Wᵀ (4,2)';
      move(lbW, 520, 20); lbW.classList.toggle('gone', st !== 2);
      const bits = [`<span>内存里始终是这 24 个数，行主序：<b>0,1,2,…,23</b></span>`];
      if (hot != null) {
        const r = rows[hot];
        bits.push(`<span>第 <b>${hot}</b> 行 <code>[${r.join(', ')}]</code> → <code>[${r[0]}+${r[1]}+${r[3]}, ${r[1]}+${r[2]}]`
          + ` = <b>[${y[hot].join(', ')}]</b></span>`);
      } else bits.push('<span>点任意一行，右边写出它的算式</span>');
      if (st === 2 || st === 3) bits.push(`<span>逐块分别乘 ⇄ 拼起来乘：最大逐位差 <b style="color:#3ecf8e">${same}</b>`
        + `（不是"近似"，是同一个数）</span>`);
      if (st === 4) bits.push('<span>第 0 行拿到的是 <b>前 6 个数</b> [0,1,2,3,4,5] —— 那是 b=0 的 1.5 个 token，'
        + '<b style="color:#ff6b6b">不是</b>"每个位置的第 0 维"。要那个得 transpose，reshape 做不到</span>');
      read.innerHTML = bits.join('');
      btns.querySelectorAll('button').forEach((b, i) => b.classList.toggle('on', i === st));
    }
    NAMES.forEach((t, i) => {
      const b = el('button', '', ['① 两块板 (2,3,4)', '② 拼成 (6,4)', '③ ×W → (6,2)',
                                 '④ 拆回 (2,3,2)', '⑤ 试试 reshape 成 (4,6)'][i]);
      b.onclick = () => { st = i; draw(); }; btns.appendChild(b);
    });
    box.appendChild(btns);
    draw();
    caption(box, '这就是 §6.1 里 `flat = x.reshape(-1, in_f)` 那一步的全部含义。矩阵乘只吃二维，'
      + '而 `W` 对 B·T 个位置共享，所以"拼起来"不改变任何数学事实：逐块乘再摞起来 = 拼成一块乘，'
      + '逐位相同（图里那个差是真的算出来的 0）。唯一要守的规矩是**行序**：拼的时候前 3 行属于 b=0，'
      + '拆的时候按同一边界切回去，所以你不需要记住"哪行属于谁" —— 内存顺序替你记着。'
      + '反向是这台机器倒放：`g (2,3,out)` 也摊成 `(6,out)`，`dW=gᵀ@x`、`db=g.sum(axis=0)`、'
      + '`dx=(g@W)` 再拆回 `(2,3,in)`。');
  }

  /* ---------------------------------------------------------------- 10. 压掉哪根轴 */
  function axes(box) {
    title(box, '§2.3 / §2.5 · `sum(axis=k)` 把第 k 根轴压掉；`@` 收缩掉的是"左最后一轴 × 右倒数第二轴"');
    const B = 2, T = 3, C = 4;
    const x = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, r) => Array.from({ length: C }, (_, k) => b * T * C + r * C + k)));
    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); box.appendChild(stage);
    const cells = [];
    x.forEach((slab, b) => slab.forEach((row, r) => row.forEach((v, k) => {
      const c = cellEl(stage, v, b ? 'b1' : '');
      c.b = b; c.r = r; c.k = k; move(c, 16 + b * (C * CW + 56) + k * CW, 40 + r * CH);
      cells.push(c);
    })));
    const outCells = [];
    const read = el('div', 'readout'); box.appendChild(read);
    const btns = el('div', 'readout'); box.appendChild(btns);
    let ax = 1, pick = null;

    const SHAPE = { 0: [T, C], 1: [B, C], 2: [B, T] };
    const NAME = ['axis=0（b：哪条序列）', 'axis=1（t：第几个 token）', 'axis=2（c：第几维特征）'];
    const SUM = { 0: (r, k) => x[0][r][k] + x[1][r][k], 1: (b, k) => x[b][0][k] + x[b][1][k] + x[b][2][k],
                  2: (b, r) => x[b][r][0] + x[b][r][1] + x[b][r][2] + x[b][r][3] };

    function draw() {
      const [R, K] = SHAPE[ax];
      head.innerHTML = `<b>sum(x, axis=${ax}) → 形状 (${R}, ${K})</b> —— 压掉的是 ${NAME[ax]}；`
        + `剩下两根轴决定结果里每个格子的位置`;
      outCells.splice(0).forEach(c => c.remove());
      for (let r = 0; r < R; r++) for (let k = 0; k < K; k++) {
        const c = cellEl(stage, SUM[ax](r, k), 'out');
        c.r = r; c.k = k; c.style.cursor = 'pointer';
        move(c, 420 + k * CW, 40 + r * CH);
        c.onclick = () => { pick = (pick === c ? null : c); draw2(); };
        outCells.push(c);
      }
      draw2();
      btns.querySelectorAll('button').forEach((b, i) => b.classList.toggle('on', i === ax));
    }
    function feeds(c) {                                    // 哪个输入格子喂了这个结果格子
      return cells.filter(i => ax === 0 ? (i.r === c.r && i.k === c.k)
                      : ax === 1 ? (i.b === c.r && i.k === c.k)
                                 : (i.b === c.r && i.r === c.k));
    }
    function draw2() {
      cells.forEach(c => c.classList.remove('summed', 'hot'));
      if (pick) {
        feeds(pick).forEach(c => c.classList.add('summed'));
        pick.classList.add('hot');
        const f = feeds(pick).map(c => c.textContent);
        read.innerHTML = `<span>结果格 <b>[${pick.r},${pick.k}]</b> = <code>${f.join(' + ')}</code>`
          + ` = <b>${pick.textContent}</b></span>`
          + `<span>被加起来的这几个格子，<b>除了被压掉的那根轴以外坐标都相同</b></span>`;
      } else {
        read.innerHTML = '<span>点右边任意一个结果格，左边会亮出"是哪几个数加成了它"</span>'
          + '<span>Linear 的 <code>db = g2.sum(axis=0)</code> 就是这个动作：bias 形状 (out,)，'
          '被压掉的是 N=B·T 那根轴</span>';
      }
    }
    ['axis=0 压掉 b', 'axis=1 压掉 t', 'axis=2 压掉 c'].forEach((t, i) => {
      const b = el('button', '', t);
      b.onclick = () => { ax = i; pick = null; draw(); }; btns.appendChild(b);
    });
    box.appendChild(btns);
    draw();
    caption(box, '三条规律，够用一辈子：① `sum(axis=k)` 之后第 k 根轴没了，结果形状就是原形状删掉那一项；'
      + '② `@` 收缩掉的是左边最后一轴和右边倒数第二轴，两根轴必须等长，剩下的轴拼成结果形状；'
      + '③ 想知道梯度公式对不对，先看形状能不能凑出来 —— 3a §2.5 那张速查表就是把这三条连用三次。');
  }

  /* ---------------------------------------------------------------- 11. 散射累加 */
  function scatter(box) {
    title(box, '§5.3 · `ids=[3,3,3,0,7]`、g 全 1：同一个 id 写三次，覆盖留 4、累加得 12');
    const ids = [3, 3, 3, 0, 7], C = 4, V = 11;
    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); stage.style.height = '190px'; box.appendChild(stage);
    const read = el('div', 'readout'); box.appendChild(read);
    const btns = el('div', 'readout'); box.appendChild(btns);
    let mode = 'add', step = 0;                            // add = np.add.at，set = fancy 赋值
    const rowsShown = [0, 3, 7];

    function val(row) {                                    // 前 step 次写入之后，这一行的每个格子是多少
      const hits = ids.slice(0, step).filter(i => i === row).length;
      if (!hits) return 0;
      return mode === 'add' ? hits : 1;                    // 覆盖：只留最后一次，每格 1
    }
    function draw() {
      stage.innerHTML = '';
      ids.forEach((v, n) => {
        const c = cellEl(stage, 'id' + v, n < step ? 'out' : '');
        c.style.cursor = 'pointer'; move(c, 16 + n * (CW + 6), 8);
        c.onclick = () => { step = n + 1; draw(); };
      });
      const lab = el('div', 'slab', 'g 的每一行都是 [1,1,1,1]（点格子=写到第几次）');
      move(lab, 16, 8 + CH + 4); stage.appendChild(lab);
      rowsShown.forEach((row, i) => {
        for (let k = 0; k < C; k++) {
          const c = cellEl(stage, val(row), row === 3 ? 'b1' : '');
          move(c, 16 + k * CW, 66 + i * CH);
        }
        const l = el('div', 'slab', `weight.grad[${row}]`); move(l, 16 + C * CW + 12, 66 + i * CH); stage.appendChild(l);
      });
      const l2 = el('div', 'slab', '其余 8 行恒为 0（没出现的行梯度必须是 0，判据第 6 条）');
      move(l2, 16, 66 + 3 * CH + 2); stage.appendChild(l2);
      const sum3 = val(3) * C;
      head.innerHTML = `<b>${mode === 'add' ? 'np.add.at(gW, ids, vals) —— 逐条累加'
        : 'gW[ids] = vals —— fancy 赋值，重复索引覆盖'}</b> · 已写入 ${step}/5 次`;
      read.innerHTML = `<span>第 3 行四个格子之和 = <b style="color:${mode === 'add' ? '#3ecf8e' : '#ff6b6b'}">${sum3}</b></span>`
        + `<span>判据「重复 id 梯度累加」期望 <b>${3 * C}</b></span>`
        + `<span>${step >= 5 && mode === 'set' ? '<b style="color:#ff6b6b">差的那 8 就是被覆盖掉的两次写入</b>'
            : mode === 'add' ? '同一条规矩还管 weight tying 和梯度累积：+= 不是 =' : '点格子往回退一步也行'}</span>`;
      btns.querySelectorAll('button').forEach(b => b.classList.toggle('on',
        (b.__m && b.__m === mode) || (b.__s != null && b.__s === step)));
    }
    const bAdd = el('button', '', 'np.add.at（累加）'); bAdd.__m = 'add';
    bAdd.onclick = () => { mode = 'add'; draw(); };
    const bSet = el('button', '', 'gW[ids] = vals（覆盖）'); bSet.__m = 'set';
    bSet.onclick = () => { mode = 'set'; draw(); };
    const bAll = el('button', '', '全部写完'); bAll.onclick = () => { step = 5; draw(); };
    const bRst = el('button', '', '↺ 清空'); bRst.onclick = () => { step = 0; draw(); };
    btns.append(bAdd, bSet, bAll, bRst); box.appendChild(btns);
    draw();
    caption(box, '关键不在 numpy 语法，而在"两次写同一格会发生什么"。fancy 赋值把 `ids` 当成一组'
      + '互不相关的地址，同一格写第二次就把第一次盖掉；`np.add.at` 才是"按索引逐条加"。'
      + '麻烦的地方在于：<code>gradcheck</code> 抓不住它 —— 它抽到的 ids 恰好没有重复时，两种写法'
      + '给出一样的结果（3b §5.3 实测 1.1e-11，全绿）。所以判据必须自己构造重复 id。');
  }

  /* ---------------------------------------------------------------- 12. 归一化的耦合项 */
  function rmscouple(box) {
    title(box, '§4.1–4.2 · 一行 C 个数是耦合的：只写 r·u 会漏掉哪一项（下面全部当场有限差分）');
    const C = 4;
    let x = [1, 2, 3, 4], g = [1, 0, 0, 0], w = 1;
    const yOf = v => { const r = 1 / Math.sqrt(v.reduce((s, a) => s + a * a, 0) / C); return v.map(a => a * r * w); };
    const L = v => yOf(v).reduce((s, a, i) => s + a * g[i], 0);
    function parts(v) {
      const r = 1 / Math.sqrt(v.reduce((s, a) => s + a * a, 0) / C);
      const xh = v.map(a => a * r), u = g.map(a => a * w);
      const s = xh.reduce((t, a, i) => t + a * u[i], 0) / C;
      return { r, xh, u, s, good: u.map((a, i) => r * a - r * xh[i] * s), bad: u.map(a => r * a) };
    }
    const num = (v, j, eps = 1e-6) => (L(v.map((a, i) => a + (i === j ? eps : 0)))
      - L(v.map((a, i) => a - (i === j ? eps : 0)))) / (2 * eps);

    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); stage.style.height = '176px'; box.appendChild(stage);
    const read = el('div', 'readout'); box.appendChild(read);
    const sl = el('div', 'sliders'); box.appendChild(sl);
    let show = 'good';

    function draw() {
      stage.innerHTML = '';
      const p = parts(x), rel = f => Math.max(...p[f].map((a, j) => {
        const d = num(x, j), n = Math.max(Math.abs(d), 1e-12);
        return Math.abs(a - d) / n;
      }));
      const rows = [['x', x], ['x̂ = x·r', p.xh], ['u = g·w', p.u],
                    ['只写 r·u（漏项）', p.bad], ['完整：r·u − r·x̂·s', p.good],
                    ['有限差分（真值）', x.map((_, j) => num(x, j))]];
      rows.forEach((rw, i) => {
        rw[1].forEach((v, k) => {
          const c = cellEl(stage, fmt(v, 4), i === 3 ? '' : i === 4 ? 'b1' : i === 5 ? 'out' : 'w');
          c.style.width = '62px'; c.style.fontSize = '10.5px';
          c.classList.toggle('off', i >= 3 && (i === 3) !== (show === 'bad'));
          move(c, 120 + k * 66, 8 + i * 27);
        });
        const l = el('div', 'slab', rw[0]); move(l, 4, 8 + i * 27 + 4); stage.appendChild(l);
      });
      head.innerHTML = `<b>x=[${x.map(v => fmt(v, 2)).join(', ')}]，g=[${g.join(', ')}]，w=${w}</b>`
        + ` · r=${fmt(p.r, 5)}，s=mean(u·x̂)=${fmt(p.s, 5)}`;
      const eg = rel('good'), eb = rel('bad');
      read.innerHTML = `<span>完整公式 vs 有限差分：最大相对误差 <b style="color:#3ecf8e">${eg.toExponential(1)}</b></span>`
        + `<span>只写 r·u：<b style="color:#ff6b6b">${eb.toExponential(1)}</b>（差的不是小系数，是整整一项）</span>`
        + `<span>s 的物理意义：沿 x̂ 方向的扰动只改分母、不改输出，所以那一份"被禁止的方向"必须减掉</span>`;
    }
    const bG = el('button', '', '看完整公式'); bG.onclick = () => { show = 'good'; draw(); };
    const bB = el('button', '', '看漏掉修正项'); bB.onclick = () => { show = 'bad'; draw(); };
    const btns = el('div', 'readout'); btns.append(bG, bB); box.appendChild(btns);
    for (let j = 0; j < C; j++) {
      const lab = el('label'); lab.innerHTML = `<b>x[${j}]</b>`;
      const s = el('input'); s.type = 'range'; s.min = -4; s.max = 4; s.step = 0.1; s.value = x[j];
      s.oninput = () => { x[j] = +s.value; draw(); }; lab.append(s); sl.appendChild(lab);
    }
    draw();
    caption(box, '这一行的 C 个分量被归一化绑在一起：动任何一个，分母 rms 就变，整行输出都跟着变。'
      + '所以 ∂L/∂x 不能只写"逐元素那一项" r·u，必须再减去沿 x̂ 方向的那一份。图里的真值是浏览器'
      + '当场做的中心差分（步长 1e-6），和判据用的是同一种招 —— 拖动滑块，看漏项那一行什么时候追上真值：'
      + '只有 s≈0（g 与 x̂ 正交）时两者才相等，那不是"公式简化了"，是那一行恰好没耦合。');
  }

  /* ---------------------------------------------------------------- 13. 减最大值 */
  function softshift(box) {
    title(box, '§2.2–2.3 · 减最大值不改答案，只救 exp；log 放在除之前才不下溢');
    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); stage.style.height = '150px'; box.appendChild(stage);
    const read = el('div', 'readout'); box.appendChild(read);
    const sl = el('div', 'sliders'); box.appendChild(sl);
    let base = 0, gap = 1;                                 // z=[base, base+1, base-2]，gap 控制 log(p) 下溢那一组
    const V = 3, TY = 1;                                   // 真值取第 1 个词

    function row(z, stable) {
      const m = stable ? Math.max(...z) : 0;
      const e = z.map(v => Math.exp(v - m));
      const Z = e.reduce((s, a) => s + a, 0);
      const sumExp = z.reduce((s, v) => s + Math.exp(v), 0);          // 不减最大值的分母，会先炸在这里
      const logp = stable ? z.map(v => v - m - Math.log(Z))
                          : z.map(v => v - Math.log(sumExp));
      return { p: e.map(a => a / Z), logp, loss: -logp[TY] };
    }
    function draw() {
      stage.innerHTML = '';
      const z = [base, base + 1, base - 2];
      const A = row(z, true), Naive = row(z, false);
      const pDiff = Math.max(...A.p.map((v, i) => Math.abs(v - row(z, false).p[i])));
      const big = [0, 0, gap];                             // log(p) vs z-m-logZ 的下溢演示
      const mB = Math.max(...big), eB = big.map(v => Math.exp(v - mB)), ZB = eB.reduce((s, a) => s + a, 0);
      const pB = eB.map(v => v / ZB);
      const viaLog = Math.log(pB[0]), viaZ = big[0] - mB - Math.log(ZB);
      const cells = [
        ['logits z', z, 'w'],
        ['减最大值后 z−m', z.map(v => v - Math.max(...z)), 'b1'],
        ['exp(z) 不减', z.map(v => Math.exp(v)), ''],
        ['exp(z−m)', z.map(v => Math.exp(v - Math.max(...z))), ''],
        ['概率 p（两种写法）', A.p, 'out'],
      ];
      cells.forEach((rw, i) => {
        rw[1].forEach((v, k) => {
          const c = cellEl(stage, Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(1) : fmt(v, 4), rw[2]);
          c.style.width = '74px'; c.style.fontSize = '10.5px';
          move(c, 150 + k * 78, 6 + i * 28);
        });
        const l = el('div', 'slab', rw[0]); move(l, 4, 10 + i * 28); stage.appendChild(l);
      });
      const bad = !isFinite(Naive.loss) || Number.isNaN(Naive.loss);
      head.innerHTML = `<b>z=[${z.join(', ')}]，真值=第 ${TY} 个词</b>`;
      read.innerHTML = `<span>稳定写法 loss=<b>${isFinite(A.loss) ? fmt(A.loss, 6) : String(A.loss)}</b></span>`
        + `<span>不减最大值 loss=<b style="color:${bad ? '#ff6b6b' : '#3ecf8e'}">${bad ? String(Naive.loss) : fmt(Naive.loss, 6)}</b></span>`
        + `<span>两者概率最大差 <b>${isFinite(pDiff) ? pDiff.toExponential(1) : 'nan（朴素写法已经 inf/inf）'}</b></span>`
        + `<span>下溢演示 gap=${gap}：log(p₀)=<b style="color:${isFinite(viaLog) ? '#3ecf8e' : '#ff6b6b'}">${viaLog}</b>，`
        + `z−m−logZ=<b style="color:#3ecf8e">${fmt(viaZ, 1)}</b></span>`;
    }
    [['base（整体平移）', 0, 900, () => base, v => base = v],
     ['gap（第二行那组的间距）', 0, 820, () => gap, v => gap = v]].forEach(([t, lo, hi, get, set]) => {
      const lab = el('label'); lab.innerHTML = `<b>${t.split('（')[0]}</b>`;
      const s = el('input'); s.type = 'range'; s.min = lo; s.max = hi; s.step = 1; s.value = get();
      s.oninput = () => { set(+s.value); draw(); }; lab.append(s, el('span', null, t)); sl.appendChild(lab);
    });
    box.appendChild(sl);
    draw();
    caption(box, '拖 base 到 710 附近：`exp(709)≈8.2e307` 是 float64 的顶，再大就是 `inf`，'
      + '于是朴素写法给出 `inf` 或 `nan`，而稳定写法的答案一位没动 —— 因为 softmax 对整体平移不变，'
      + '分子分母的 e^c 相消。拖 gap 到 800 看第二件事：`p` 下溢成恰好 0，`log(0)=-inf`，'
      + '而 `z−m−log Z` 照样给出 -800。这就是参考答案里 `log_probs` 一路用 z 表示、'
      + '从不写 `np.log(概率)` 的原因。判据里 `logits=900 不溢出` 那条同时挡掉 `inf` 和 `nan` 两种坏法。');
  }

  /* ---------------------------------------------------------------- 14. 采样流水线 */
  function sampling(box) {
    title(box, '§2.6 · 一条流水线：惩罚 → 贪心岔口 → 温度 → top-k → softmax → top-p → 抽样');
    const z0 = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5];            // V=6，两个"已出现过"的词是 1 和 4
    const seen = [1, 4];
    const MASK = -1e18;
    let rho = 1, temp = 1, topk = 0, topp = 1, wrong = false;

    const head = el('div', 'shape'); box.appendChild(head);
    const bars = el('div', 'stage'); bars.style.height = '176px'; box.appendChild(bars);
    const read = el('div', 'readout'); box.appendChild(read);
    const sl = el('div', 'sliders'); box.appendChild(sl);
    const btns = el('div', 'readout'); box.appendChild(btns);

    const softmax = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), Z = e.reduce((s, a) => s + a, 0); return e.map(a => a / Z); };
    function draw() {
      bars.innerHTML = '';
      // ① 惩罚（正分除、负分乘 —— 两个不对称）
      const zp = z0.map((v, i) => !seen.includes(i) ? v : (v > 0 ? v / rho : v * rho));
      // ② 温度
      const greedy = temp <= 0;
      const zt = greedy ? zp : zp.map(v => v / temp);
      // ④ top-k（在分数上）
      const idx = zp.map((_, i) => i).sort((a, b) => zt[b] - zt[a]);
      const zk = topk > 0 ? zt.map((v, i) => idx.slice(0, topk).includes(i) ? v : MASK) : zt.slice();
      // ⑤ softmax → ⑥ top-p（在概率上）
      let p = softmax(zk);
      if (!wrong && topp < 1) {
        const order = p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
        let acc = 0; const keep = [];
        for (const i of order) { acc += p[i]; keep.push(i); if (acc >= topp) break; }
        p = softmax(zk.map((v, i) => keep.includes(i) ? v : MASK));
      }
      if (wrong && topp < 1) {                                // 错法：拿分数当概率排序
        const order = zk.map((_, i) => i).sort((a, b) => zk[b] - zk[a]);
        const raw = zk.map(v => Math.max(0, v));              // 假"概率"：分数截负、不归一
        const tot = raw.reduce((s, a) => s + a, 0) || 1;
        let acc = 0; const keep = [];
        for (const i of order) { acc += raw[i] / tot; keep.push(i); if (acc >= topp) break; }
        p = softmax(zk.map((v, i) => keep.includes(i) ? v : MASK));
      }
      const maxP = Math.max(...p, 1e-9);
      p.forEach((v, i) => {
        const w = Math.max(2, v / maxP * 420);
        const b = el('div'); b.style.cssText = `position:absolute;left:96px;top:${8 + i * 27}px;width:${w}px;height:22px;`
          + `border-radius:3px;background:${v > 1e-6 ? (seen.includes(i) ? '#8fd6b0' : '#7fb3e8') : '#2a3340'}`;
        bars.appendChild(b);
        const l = el('div', 'slab', `词${i}${seen.includes(i) ? ' ←见过' : ''}`); move(l, 2, 10 + i * 27); bars.appendChild(l);
        const t = el('div', 'slab', (greedy && i === idx[0] ? '★ ' : '') + (v < 1e-6 ? '0' : v.toFixed(3)));
        move(t, 100 + w, 10 + i * 27); bars.appendChild(t);
      });
      const ent = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
      head.innerHTML = `<b>惩罚 ρ=${rho} · 温度 T=${temp}${greedy ? '（贪心：直接 argmax，不经过任何随机）' : ''} · top-k=${topk || '关'} · top-p=${topp}</b>`;
      read.innerHTML = `<span>词1 的分数 <b>${z0[1]} → ${fmt(zp[1], 3)}</b>，词4（负分）<b>${z0[4]} → ${fmt(zp[4], 3)}</b></span>`
        + `<span>存活 <b>${p.filter(v => v > 1e-6).length}</b> 个词 · 熵 <b>${fmt(ent, 4)}</b>（ln6=${fmt(Math.log(6), 3)} 是均匀上限）</span>`
        + `<span>${wrong ? '<b style="color:#ff6b6b">错序演示</b>：top-p 需要概率，拿截负后的分数当概率，'
            + '阈值和词表尺度耦合 —— 同一批参数会留下完全不同的词' : '顺序换了会出事的只有三处：惩罚必须在贪心岔口之前、'
            + '温度必须在 softmax 之前、top-p 必须在 softmax 之后'}</span>`;
      btns.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.__w === wrong));
    }
    [['ρ 惩罚', 1, 6, 0.1, () => rho, v => rho = v],
     ['T 温度', 0, 2, 0.05, () => temp, v => temp = v],
     ['k top-k', 0, 6, 1, () => topk, v => topk = v],
     ['p top-p', 0.05, 1, 0.05, () => topp, v => topp = v]].forEach(([t, lo, hi, st, get, set]) => {
      const lab = el('label'); lab.innerHTML = `<b>${t}</b>`;
      const s = el('input'); s.type = 'range'; s.min = lo; s.max = hi; s.step = st; s.value = get();
      s.oninput = () => { set(+s.value); draw(); }; lab.append(s); sl.appendChild(lab);
    });
    box.appendChild(sl);
    const bR = el('button', '', 'T=0（贪心）'); bR.onclick = () => { temp = 0; draw(); };
    const bW = el('button', '', '把 top-p 挪到 softmax 之前（错）'); bW.__w = true;
    bW.onclick = () => { wrong = !wrong; draw(); };
    const bC = el('button', '', '还原默认'); bC.onclick = () => { rho = 1; temp = 1; topk = 0; topp = 1; wrong = false; draw(); };
    btns.append(bR, bW, bC); box.appendChild(btns);
    draw();
    caption(box, '温度可单独拖到 0 看贪心岔口：那时 argmax 直接取走，ρ 和 k 都不起作用 —— '
      + '所以惩罚必须放在这个岔口**之前**。top-k 和温度可交换（除以 T 是单调缩放，谁大还是那些）；'
      + 'top-p 不行：它要的是"累计到 0.9 需要几个词"，那是概率意义上的数，'
      + '在分数上做阈值等于把阈值和词表尺度绑在一起。');
  }

  /* ---------------------------------------------------------------- 15. 残差方差 */
  function resid(box) {
    title(box, '§3.3 · 残差每层加两路子层输出：pre-norm 下按 1+2cL 线性长，出口乘 1/√(2L) 后与深度无关');
    let L = 6, c = 1, model = 'pre';
    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); stage.style.height = '168px'; box.appendChild(stage);
    const read = el('div', 'readout'); box.appendChild(read);
    const sl = el('div', 'sliders'); box.appendChild(sl);
    const btns = el('div', 'readout'); box.appendChild(btns);

    // 两种"每层加进来多少"的假设：
    //   pre-norm  —— 子层看到的是 RMSNorm(x)，尺度与主干无关 ⇒ 每层加的是**常数** c ⇒ 线性 1+2cL
    //   post-norm —— 子层输出正比于主干 ⇒ 每层乘 (1+2c) ⇒ 指数
    // 本项目是 pre-norm，所以判据钉的那个 1/√(2L) 在两种假设下都够用（指数的话甚至不够）。
    const at = (l, scaled) => model === 'pre'
      ? 1 + (scaled ? c * l / L : 2 * c * l)
      : Math.pow(1 + (scaled ? c / L : 2 * c), l);

    function draw() {
      stage.innerHTML = '';
      const raw = [], scaled = [];
      for (let l = 1; l <= 12; l++) { raw.push(at(l, false)); scaled.push(at(l, true)); }
      const top = Math.log10(Math.max(...raw, 10));
      for (let l = 1; l <= 12; l++) {
        const h1 = Math.min(130, 130 * Math.log10(raw[l - 1]) / top);
        const b1 = el('div'); b1.style.cssText = `position:absolute;left:${16 + (l - 1) * 52}px;top:${140 - h1}px;`
          + `width:20px;height:${Math.max(2, h1)}px;background:#ff6b6b;border-radius:2px 2px 0 0`; stage.appendChild(b1);
        const h2 = Math.min(130, 130 * Math.log10(Math.max(scaled[l - 1], 1)) / top);
        const b2 = el('div'); b2.style.cssText = `position:absolute;left:${38 + (l - 1) * 52}px;top:${140 - h2}px;`
          + `width:20px;height:${Math.max(2, h2)}px;background:#3ecf8e;border-radius:2px 2px 0 0`; stage.appendChild(b2);
      }
      const lab = el('div', 'slab', '红=不缩放 · 绿=出口乘 1/√(2L) · 纵轴 log₁₀ 方差比（顶格 = '
        + Math.pow(10, Math.round(top)).toLocaleString() + '）');
      move(lab, 4, 4); stage.appendChild(lab);
      head.innerHTML = `<b>${model === 'pre' ? 'pre-norm（本项目的做法）：每层加的是常数 → 线性'
        : 'post-norm：每层按主干比例加 → 指数'}</b> · L=${L}，支路系数 c=${c}`;
      read.innerHTML = `<span>缩放因子 <b>1/√(2L)=${(1 / Math.sqrt(2 * L)).toFixed(4)}</b>，只作用在残差出口（`
        + `attn.proj.W、mlp.fc2.W）</span>`
        + `<span>不缩放：第 ${L} 层方差比 <b>${at(L, false).toExponential(2)}</b>；缩放后 <b>${at(L, true).toFixed(3)}</b></span>`
        + `<span>绿柱几乎一样高 —— 这就是"与深度无关"。判据 <code>GIV-3</code> 查的是权重本身：`
        + `出口矩阵的范数应当等于不缩放时的 1/√(2L)（L=3 时两者比值正好 √6=2.4495），且别的张量一位不动</span>`;
      btns.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.__m === model));
    }
    [['L 层数', 1, 12, 1, () => L, v => L = v],
     ['c 支路系数', 0.2, 2, 0.1, () => c, v => c = v]].forEach(([t, lo, hi, st, get, set]) => {
      const lab = el('label'); lab.innerHTML = `<b>${t}</b>`;
      const s = el('input'); s.type = 'range'; s.min = lo; s.max = hi; s.step = st; s.value = get();
      s.oninput = () => { set(+s.value); draw(); }; lab.append(s); sl.appendChild(lab);
    });
    box.appendChild(sl);
    const bPre = el('button', '', 'pre-norm（本项目）'); bPre.__m = 'pre';
    bPre.onclick = () => { model = 'pre'; draw(); };
    const bPost = el('button', '', 'post-norm（对照）'); bPost.__m = 'post';
    bPost.onclick = () => { model = 'post'; draw(); };
    btns.append(bPre, bPost); box.appendChild(btns);
    draw();
    caption(box, '判据里"残差出口置零后 Block 是恒等映射"查的是另一半：把 fc2.W、attn.proj.W 清零，'
      + '整层必须原样吐出输入 —— 那证明你两条残差真的加了、梯度也原样透传。两件事合起来才是 '
      + 'init_resid_scale 的完整理由。切到 post-norm 那一栏是为了让你看见"指数"长什么样：'
      + '本项目用 pre-norm，所以不缩放的代价是线性起飞（末端数值尺度随深度漂），不是爆炸。');
  }

  /* ---------------------------------------------------------------- 16. 检查点的字节账 */
  function ckpt(box) {
    title(box, '§2.2 · 那个文件为什么正好是 417,005 字节：四段加起来，一段都不能多');
    // 阶段 6 那台模型：2 层、d_model=64、ctx=40、vocab=27、29 个张量、103,936 个参数
    const N = 103936, HDR = 12;                             // 29 个张量：187 + 1062 字节元信息
    let wide = false;
    const head = el('div', 'shape'); box.appendChild(head);
    const stage = el('div', 'stage'); stage.style.height = '150px'; box.appendChild(stage);
    const read = el('div', 'readout'); box.appendChild(read);
    const btns = el('div', 'readout'); box.appendChild(btns);
    let pick = 0;

    const segs = () => [
      ['文件头', HDR, 'magic(4) + version(4) + n(4) —— 先读 12 个字节才知道后面有多长'],
      ['头 JSON', 187, '超参 + dtype + 张量名顺序。删掉长度前缀 n，读侧就只能"读到报错为止"'],
      ['张量元信息', 1062, `29 个张量：名字字节数 + 维数 + 每维形状。名字进文件是为了 C++ 侧按名取`],
      ['权重数据', wide ? N * 8 : N * 4, `${N.toLocaleString()} 个参数 × ${wide ? 8 : 4} 字节`
        + (wide ? ' —— 这就是"写成 float64"的那个世界' : '（fp32 是公开格式的一部分）')],
    ];
    function draw() {
      stage.innerHTML = '';
      const S = segs(), tot = S.reduce((s, r) => s + r[1], 0);
      S.forEach((r, i) => {
        const w = Math.max(3, 470 * r[1] / tot);
        const b = el('div'); b.style.cssText = `position:absolute;left:150px;top:${8 + i * 32}px;width:${w}px;height:26px;`
          + `border-radius:3px;background:${i === pick ? '#f0c07a' : (i === 3 ? '#7fb3e8' : '#3d4a5c')}`;
        b.style.cursor = 'pointer'; b.onclick = () => { pick = i; draw(); }; stage.appendChild(b);
        const l = el('div', 'slab', r[0]); move(l, 4, 12 + i * 32); stage.appendChild(l);
        const t = el('div', 'slab', r[1].toLocaleString() + ' B'); move(t, 156 + w, 12 + i * 32); stage.appendChild(t);
      });
      head.innerHTML = `<b>合计 ${tot.toLocaleString()} 字节</b> = ${S.map(r => r[1].toLocaleString()).join(' + ')}`
        + `　（点某一段看它是怎么来的）`;
      read.innerHTML = `<span><b>${segs()[pick][0]}</b>：${segs()[pick][2]}</span>`
        + `<span>${wide ? '<b style="color:#ff6b6b">按 8 字节写完、再按 8 字节读回来 —— 自己测永远绿</b>；'
            + 'C++ 侧照公开的表读 fp32，第一个张量之后全部错位，读出一堆看起来合理的垃圾'
            : '判据用一个不认你代码的解析器（`parse_ckpt`，二十行）照着这张表推进 —— 多写一个 padding 都会红'}</span>`
        + `<span>对照：同一台模型的状态文件要存 $\theta,m,v$ 三份，其中两份是 float64 —— 所以它比权重文件大得多</span>`;
      btns.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.__w === wide));
    }
    const b32 = el('button', '', 'float32（公开格式）'); b32.__w = false;
    b32.onclick = () => { wide = false; draw(); };
    const b64 = el('button', '', '写成 float64（错）'); b64.__w = true;
    b64.onclick = () => { wide = true; draw(); };
    btns.append(b32, b64); box.appendChild(btns);
    draw();
    caption(box, '这笔账的价值在于它是**可反驳的**：417,005 这个数由"格式"唯一决定，'
      + '所以任何多余的一个字节（padding、把名字写两遍、把 tied 的输出头再存一份）都会当场红。'
      + '判据 `磁盘上是 float32` 就是在数数据段是不是恰好 4 字节 × 参数总数；'
      + '`张量名与数量 == named_parameters` 数的是元信息段是不是 29 项而不是 30 项。');
  }

  const REG = { graph, losscurve, steps, capacity, shapes, gradcheck, mask, heads,
                flatten, axes, scatter, rmscouple, softshift, sampling, resid, ckpt };

  window.Figs = {
    mountAll(root) {
      root.querySelectorAll('.fig[data-fig]').forEach(box => {
        const fn = REG[box.dataset.fig];
        if (!fn || box.__mounted) return;
        box.__mounted = true;
        try { fn(box); } catch (err) {                        // 图坏了不能连累正文
          box.innerHTML = '<div class="cap">这张图没渲染出来：' + esc1(String(err)) + '（正文与判据不受影响）</div>';
        }
      });
    }
  };
  function esc1(s) { return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
})();
