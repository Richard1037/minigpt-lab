/* 原理动画 · 第 A 批：注意力与整机装配（阶段 4 / 5）。
   注册方式与工具函数见 figs2.js 顶部；挂载点在讲义 md 里写 `@@fig 名字@@`。
   底线和 figs.js 一样：能算的都真的算，不放示意数据；组件抛错只是"这张图没出来"。

   本批四个组件：attnflow / causalfix / shiftdir / tiedgrad。
   公共约定：
     · 每个组件都有 ▶播放 / ⏭单步 / ⏸暂停，"当前步"由纯函数 render(t) 算出 ——
       单步不依赖 rAF，无头浏览器停在任意一帧都是确定的。
     · 随机数用 mulberry32 + Box-Muller（正态），同一颗种子在 JS / Python 里
       逐位可复现（除 sin/cos/log 的最后一两个 ulp），所以讲义判据的数值
       可以用独立脚本按 solutions/ 的式子复算核对。
     · 需要的新样式全部注入在 .lane-a 前缀下，不动 manual.css。 */
(function () {
  const F = window.MGPT_FIG, R = window.MGPT_FIGS;
  if (!F || !R) return;

  /* ------------------------------------------------------------ 注入样式 */
  if (!document.getElementById('lane-a-css')) {
    const css = document.createElement('style');
    css.id = 'lane-a-css';
    css.textContent = [
      '.lane-a .tp{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0 2px}',
      '.lane-a .tp .now{flex:1 1 300px;min-width:0;font:12px/1.5 var(--mono);color:#cfe0ff}',
      '.lane-a .tp .now b{font-weight:700}',
      '.lane-a .gx{display:grid;gap:2px;width:max-content;margin:4px 0}',
      '.lane-a .gx i{display:flex;align-items:center;justify-content:center;border-radius:3px;',
      '  background:#1b2330;font:400 10px/1 var(--mono);font-style:normal;color:#cfe0ff;cursor:default}',
      '.lane-a .gx i.ax{background:transparent;color:var(--dim);font-size:9.5px}',
      '.lane-a .gx i.click{cursor:pointer}',
      '.lane-a .gx i.sel{outline:2px solid var(--accent);outline-offset:-2px}',
      '.lane-a .gx i.hot{outline:2px solid var(--warn);outline-offset:-2px}',
      '.lane-a .gx i.summed{outline:2px solid var(--accent);outline-offset:-2px}',
      '.lane-a .gx i.off{opacity:.14}',
      '.lane-a .glab{font:11.5px/1.4 var(--mono);color:var(--dim);margin:6px 0 0}',
      '.lane-a .glab b{color:#cfe0ff;font-weight:700}'
    ].join('\n');
    document.head.appendChild(css);
  }

  /* ------------------------------------------- 可复现随机数（Python 同构） */
  function mulberry32(seed) {
    let a = seed | 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normRng(seed) {                       // Box-Muller，spare 缓存
    const u = mulberry32(seed);
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      const u1 = u(), u2 = u();
      const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
  }
  function randnVec(n, rng, std) { const v = []; for (let i = 0; i < n; i++) v.push(rng() * std); return v; }
  function randnMat(r, c, rng, std) { const m = []; for (let i = 0; i < r; i++) m.push(randnVec(c, rng, std)); return m; }

  /* ------------------------------------------------------------- 数学工具 */
  const NEG = -1e18;                              // 和 solutions 的 NEG_INF 一致
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function softmaxRow(s) {                        // 减最大值防溢出，同 _softmax_rows
    let m = -Infinity;
    for (const v of s) if (v > m) m = v;
    const e = []; let Z = 0;
    for (const v of s) { const x = Math.exp(v - m); e.push(x); Z += x; }
    return e.map(x => x / Z);
  }
  function variance(xs) {
    let s = 0; for (const x of xs) s += x;
    const mu = s / xs.length;
    let q = 0; for (const x of xs) q += (x - mu) * (x - mu);
    return q / xs.length;
  }

  /* ------------------------------------------------------------- UI 工具 */
  /* 播放/单步/暂停 + 当前步行。render(t) 必须是纯函数（只由 t 与控件值决定）。 */
  function transport(box, N, render, interval) {
    const tp = F.el('div', 'tp');
    const bPlay = F.el('button', null, '▶ 播放');
    const bStep = F.el('button', null, '⏭ 单步');
    const bPause = F.el('button', null, '⏸ 暂停');
    const now = F.el('div', 'now');
    tp.append(bPlay, bStep, bPause, now);
    box.appendChild(tp);
    let t = 0, raf = 0, last = 0;
    function draw() {
      let desc = '';
      try { desc = render(t) || ''; } catch (err) { desc = '这一步算错了：' + F.esc1(String(err)); }
      now.innerHTML = '<b>第 ' + (t + 1) + '/' + N + ' 步</b> · ' + F.md(desc);
    }
    function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; last = 0; bPlay.classList.remove('on'); }
    function tick(ts) {
      if (!last) last = ts;
      if (ts - last >= interval) { last = ts; t = (t + 1) % N; draw(); }
      raf = requestAnimationFrame(tick);
    }
    bPlay.onclick = () => { if (raf) return; bPlay.classList.add('on'); last = 0; raf = requestAnimationFrame(tick); };
    bPause.onclick = stop;
    bStep.onclick = () => { stop(); t = (t + 1) % N; draw(); };
    draw();
    return { draw, setT(v) { t = ((v % N) + N) % N; draw(); } };
  }

  /* 数值矩阵格子：rows×cols 的 <i> 网格（含可选的首行/首列轴标签）。 */
  function gx(host, rows, cols, cw, ch) {
    const g = F.el('div', 'gx');
    g.style.gridTemplateColumns = 'repeat(' + cols + ',' + cw + 'px)';
    const cells = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        const c = F.el('i');
        c.style.width = cw + 'px'; c.style.height = ch + 'px';
        g.appendChild(c); row.push(c);
      }
      cells.push(row);
    }
    host.appendChild(g);
    return {
      g, cells,
      at(i, j) { return cells[i][j]; },
      ax(i, j, txt) { const c = cells[i][j]; c.className = 'ax'; c.textContent = txt; },
      put(i, j, txt) { const c = cells[i][j]; c.className = ''; c.style.opacity = ''; c.style.background = ''; c.style.color = ''; c.textContent = txt; return c; }
    };
  }

  /* 按 |v|/scale 上色；接近 0 的格子压暗。rgb 形如 '74,163,255'。 */
  function paint(cell, v, scale, rgb) {
    const a = Math.min(1, Math.abs(v) / scale);
    if (Math.abs(v) < 5e-4) {
      cell.style.background = '#1b2330'; cell.style.color = '#5c6b80';
    } else {
      cell.style.background = 'rgba(' + rgb + ',' + (0.14 + 0.8 * a).toFixed(3) + ')';
      cell.style.color = a > 0.55 ? '#0b0e13' : '#e6ebf3';
    }
  }

  /* ============================================================ attnflow */
  R.attnflow = function (box) {
    box.classList.add('lane-a');
    const T = 5;
    F.title(box, 'attnflow · 一行注意力权重的诞生：q·k/√d → 因果掩码 → 逐行 softmax → 按权重收集 v');
    const head = F.head(box, '');

    const row = F.el('div', 'row'); box.appendChild(row);
    const colL = F.el('div', 'col'); colL.style.flex = '0 0 auto';
    const colR = F.el('div', 'col');
    row.append(colL, colR);

    const labL = F.el('div', 'glab'); colL.appendChild(labL);
    const mat = gx(colL, T + 1, T + 1, 46, 24);
    mat.ax(0, 0, 'att');
    for (let j = 0; j < T; j++) mat.ax(0, j + 1, 'k' + j);
    for (let i = 0; i < T; i++) mat.ax(i + 1, 0, 'q' + i);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) mat.cells[i + 1][j + 1].classList.add('click');

    const labR = F.el('div', 'glab'); colR.appendChild(labR);
    const det = gx(colR, 1, T + 1, 46, 24);
    det.ax(0, 0, '行 i');
    const outLine = F.el('div', 'glab'); colR.appendChild(outLine);
    const plotWrap = F.el('div'); colR.appendChild(plotWrap);
    const legend = F.el('div', 'glab', '选中行 i 的概率：蓝实线 = ÷√d，黄虚线 = 不除'); colR.appendChild(legend);

    const ro = F.readout(box);
    const sl = F.sliders(box);
    const sdh = F.slider(sl, 'd_head', 2, 64, 1, 16, () => tp.draw());

    const NAMES = ['① s = q·k（原始分数）', '② ÷√d_head', '③ + 因果掩码（j>i 处 −1e18）', '④ 逐行 softmax → a'];

    function render(t) {
      const i = Math.floor(t / 4) % T, ph = t % 4;
      const d = +sdh.input.value, rt = Math.sqrt(d);
      const rng = normRng(7);                     // 固定种子：q、k 随 d_head 重新生成
      const q = [], k = [], v = [];
      for (let a = 0; a < T; a++) q.push(randnVec(d, rng, 1));
      for (let a = 0; a < T; a++) k.push(randnVec(d, rng, 1));
      for (let a = 0; a < T; a++) v.push(randnVec(3, rng, 1));

      const sRaw = [], sS = [], P = [], P0 = [], rawPool = [], sPool = [];
      for (let a = 0; a < T; a++) {
        sRaw.push([]); sS.push([]); P.push([]); P0.push([]);
        const mS = [], mR = [];
        for (let b = 0; b < T; b++) {
          const r0 = dot(q[a], k[b]);
          sRaw[a].push(r0); sS[a].push(r0 / rt);
          rawPool.push(r0); sPool.push(r0 / rt);   // 方差统计用全部 T² 个分数
          const ok = b <= a;
          mS.push(ok ? r0 / rt : NEG);
          mR.push(ok ? r0 : NEG);
        }
        P[a] = softmaxRow(mS); P0[a] = softmaxRow(mR);
      }
      const out = [];
      for (let a = 0; a < T; a++) {
        const o = [0, 0, 0];
        for (let b = 0; b < T; b++) for (let c = 0; c < 3; c++) o[c] += P[a][b] * v[b][c];
        out.push(o);
      }
      const varRaw = variance(rawPool), varS = variance(sPool);

      head.innerHTML = F.md('q,k：(T=' + T + ', d_head=' + d + ') · att：(' + T + ',' + T + ') 单头 · 掩码允许 j ≤ i');
      labL.innerHTML = '<b>' + NAMES[ph] + '</b>（点矩阵任意行可直接跳到该行的 ④）';

      /* 左：整块矩阵，按阶段换内容 */
      for (let a = 0; a < T; a++) for (let b = 0; b < T; b++) {
        const c = mat.cells[a + 1][b + 1];
        c.className = 'click' + (a === i ? ' sel' : '');
        c.style.opacity = ''; c.style.background = ''; c.style.color = '';
        const ok = b <= a;
        if (ph === 0) { c.textContent = F.fmt(sRaw[a][b], 1); }
        else if (ph === 1) { c.textContent = F.fmt(sS[a][b], 1); }
        else if (ph === 2) {
          if (ok) c.textContent = F.fmt(sS[a][b], 1);
          else { c.textContent = '−1e18'; c.style.opacity = '.3'; }
        } else {
          if (ok) { c.textContent = F.fmt(P[a][b], 2); paint(c, P[a][b], 1, '74,163,255'); }
          else { c.textContent = '0'; c.style.opacity = '.18'; }
        }
      }

      /* 右：选中行的当前阶段值 + out + 概率对照图 */
      labR.innerHTML = '行 i=' + i + ' 的 ' + (ph === 3 ? '注意力权重 a（本行和 = 1）' : '分数');
      det.at(0, 0).textContent = 'q' + i;
      for (let b = 0; b < T; b++) {
        const c = det.cells[0][b + 1];
        c.className = b <= i ? '' : 'off';
        c.style.background = ''; c.style.color = '';
        const val = ph === 0 ? sRaw[i][b] : ph === 1 || ph === 2 ? sS[i][b] : P[i][b];
        c.textContent = b <= i ? F.fmt(val, ph === 3 ? 3 : 1) : (ph === 2 ? '−1e18' : '0');
        if (ph === 3 && b <= i) paint(c, P[i][b], 1, '74,163,255');
      }
      const rowSum = P[i].reduce((s, x) => s + x, 0);
      outLine.innerHTML = ph === 3
        ? 'out = Σ a·v = (' + out[i].map(x => F.fmt(x, 3)).join(', ') + ')　行和 = ' + F.fmt(rowSum, 6)
        : 'out = Σ a·v（等 ④ softmax 之后才算）';

      const g = F.plot(plotWrap, 300, 150, -0.4, T - 0.6, 0, 1.08);
      const xs = [0, 1, 2, 3, 4].filter(j => j <= i);
      const prof = P_ => x => {                   // 只在允许段 [0, i] 上线性插值
        if (x < 0 || x > i) return NaN;
        const j0 = Math.floor(x), fr = x - j0;
        const a = P_[j0], b = j0 + 1 <= i ? P_[j0 + 1] : a;
        return a + (b - a) * fr;
      };
      g.line(prof(P[i]), '#4aa3ff', null, true);
      g.line(prof(P0[i]), '#f5b445', [4, 3]);
      g.dots(xs, j => P[i][j], '#4aa3ff');
      g.dots(xs, j => P0[i][j], '#f5b445');

      const pmax = Math.max(...P[i]), p0max = Math.max(...P0[i]);
      ro.innerHTML =
        '<span>分数方差 Var(q·k) = <b>' + F.fmt(varRaw, 2) + '</b>（全部 ' + T * T + ' 个分数；理论 ≈ d_head = ' + d + '，s04 §11.2）</span>' +
        '<span>÷√d 后 Var = <b>' + F.fmt(varS, 2) + '</b>（≈ 1）</span>' +
        '<span>行 ' + i + ' 最大概率：÷√d <b>' + F.fmt(pmax, 3) + '</b>｜不除 <b>' + F.fmt(p0max, 3) + '</b></span>';

      for (let a = 0; a < T; a++) for (let b = 0; b < T; b++) {
        const ii = a;
        mat.cells[a + 1][b + 1].onclick = () => tp.setT(ii * 4 + 3);
      }

      if (ph === 0) return '行 i=' + i + '：s(i,j) = q' + i + '·k(j)，' + T + ' 个原始分数（还没缩放）';
      if (ph === 1) return '÷√d_head = ÷' + F.fmt(rt, 2) + '：分数方差从 ' + F.fmt(varRaw, 1) + ' 拉回 ' + F.fmt(varS, 2) + '。不除的话 d_head 越大 softmax 越"非一即零"（看右图黄虚线）';
      if (ph === 2) return '加因果掩码：j>i 处 + (−1e18)，必须加在 softmax <b>之前</b>（s04 §11.3；加在之后分母就被污染了）';
      return '逐行 softmax：行和 = ' + F.fmt(rowSum, 6) + '；out = Σ a·v = (' + out[i].map(x => F.fmt(x, 2)).join(', ') + ')。本行把 1.0 分给了 j ≤ ' + i + ' 的 key，上三角精确为 0';
    }

    const tp = transport(box, T * 4, render, 900);
    F.caption(box, '判据对照（tests/test_attention.py + webapp/checks.py TODO-12）：`_probs` 每行和 = 1、上三角概率全为 0。' +
      '所有分数、softmax、加权和都在本页现算；换 d_head 会用固定种子重新生成 q、k。');
  };

  /* ============================================================ causalfix */
  R.causalfix = function (box) {
    box.classList.add('lane-a');
    const T = 6, D = 8, VOC = 6, EPS = 1e-5;
    F.title(box, 'causalfix · 因果性是可证伪的：改一个未来 token，过去的输出必须一个比特都不变');
    const head = F.head(box, '单层玩具模型：x = tok_emb[id] + pos_emb[t] → rmsnorm → 单头注意力(d=8) → +残差；权重随机、种子固定');

    /* 固定权重（种子 3）：与 Python 复算脚本逐位同构 */
    const rng = normRng(3);
    const tokE = randnMat(VOC, D, rng, 0.5);
    const posE = randnMat(T, D, rng, 0.5);
    const Wq = randnMat(D, D, rng, 0.5), bq = randnVec(D, rng, 0.1);
    const Wk = randnMat(D, D, rng, 0.5);
    const Wv = randnMat(D, D, rng, 0.5), bv = randnVec(D, rng, 0.1);
    const Wo = randnMat(D, D, rng, 0.5);
    const ids0 = [3, 1, 4, 0, 5, 2];

    function lin(x, W, b) {                       // y = x @ Wᵀ + b（W 存 (out,in)）
      const y = [];
      for (let o = 0; o < W.length; o++) {
        let s = dot(x, W[o]);
        if (b) s += b[o];
        y.push(s);
      }
      return y;
    }
    function fwd(ids, causal) {
      const x = [];
      for (let t = 0; t < T; t++) x.push(tokE[ids[t]].map((e, c) => e + posE[t][c]));
      const q = [], k = [], v = [];
      for (let t = 0; t < T; t++) {
        let ms = 0; for (const e of x[t]) ms += e * e;
        const r = 1 / Math.sqrt(ms / D + EPS);    // rmsnorm，w = 1
        const h = x[t].map(e => e * r);
        q.push(lin(h, Wq, bq)); k.push(lin(h, Wk, null)); v.push(lin(h, Wv, bv));
      }
      const out = [];
      for (let i = 0; i < T; i++) {
        const s = [];
        for (let j = 0; j < T; j++) s.push(!causal || j <= i ? dot(q[i], k[j]) / Math.sqrt(D) : NEG);
        const p = softmaxRow(s);
        const y = [];
        for (let c = 0; c < D; c++) { let acc = 0; for (let j = 0; j < T; j++) acc += p[j] * v[j][c]; y.push(acc); }
        const a = lin(y, Wo, null);
        out.push(x[i].map((e, c) => e + a[c]));   // 残差
      }
      return out;
    }

    const baseC = fwd(ids0, true), baseP = fwd(ids0, false);

    /* ids 行 + 两块 Δ 网格 */
    const idsG = gx(box, 1, T + 1, 40, 22);
    idsG.ax(0, 0, 'ids');
    const row2 = F.el('div', 'row'); box.appendChild(row2);
    const colA = F.el('div', 'col'); colA.style.flex = '0 0 auto';
    const colB = F.el('div', 'col'); colB.style.flex = '0 0 auto';
    row2.append(colA, colB);
    function mkGrid(host, name) {
      const lab = F.el('div', 'glab', '<b>' + name + '</b>'); host.appendChild(lab);
      const g = gx(host, T + 1, D + 2, 40, 22);
      g.ax(0, 0, '位置');
      for (let c = 0; c < D; c++) g.ax(0, c + 1, 'd' + c);
      g.ax(0, D + 1, 'max|Δ|');
      for (let i = 0; i < T; i++) g.ax(i + 1, 0, 't' + i);
      return g;
    }
    const gC = mkGrid(colA, '因果掩码（j ≤ i）：|out′ − out|');
    const gP = mkGrid(colB, '允许看未来（无掩码）：|out′ − out|');

    const ro = F.readout(box);
    const sl = F.sliders(box);
    const scp = F.slider(sl, '改动位置', 1, 5, 1, 5, () => tp.draw());

    function showOut(g, out) {
      for (let i = 0; i < T; i++) {
        let mx = 0;
        for (let c = 0; c < D; c++) {
          const cell = g.put(i + 1, c + 1, F.fmt(out[i][c], 2));
          paint(cell, out[i][c], 3, '74,163,255');
          mx = Math.max(mx, Math.abs(out[i][c]));
        }
        g.put(i + 1, D + 1, F.fmt(mx, 2)).style.color = '#8d99ac';
      }
    }
    function showDelta(g, A, B, hl) {
      let pastMax = 0, cp = +scp.input.value;
      for (let i = 0; i < T; i++) {
        let mx = 0;
        for (let c = 0; c < D; c++) {
          const dv = Math.abs(A[i][c] - B[i][c]);
          mx = Math.max(mx, dv);
          const cell = g.put(i + 1, c + 1, dv === 0 ? '0' : F.fmt(dv, 2));
          if (dv === 0) { cell.style.background = '#16211c'; cell.style.color = '#3ecf8e'; }
          else paint(cell, dv, 1.5, '255,107,107');
        }
        if (i < cp) pastMax = Math.max(pastMax, mx);
        const mc = g.put(i + 1, D + 1, F.fmt(mx, 3));
        mc.style.color = mx === 0 ? '#3ecf8e' : '#ff6b6b';
        mc.className = i === hl ? 'hot' : '';
      }
      return pastMax;
    }

    function render(t) {
      const cp = +scp.input.value;
      const idsAlt = ids0.slice();
      idsAlt[cp] = (idsAlt[cp] + 1) % VOC;
      for (let i = 0; i < T; i++) {
        const c = idsG.put(0, i + 1, String(ids0[i]));
        if (t >= 1 && i === cp) { c.textContent = ids0[i] + '→' + idsAlt[i]; c.className = 'hot'; }
      }

      if (t === 0) {
        showOut(gC, baseC); showOut(gP, baseP);
        ro.innerHTML = '<span>两块网格是同一份 ids 在两种掩码下的输出（还没改动）</span>';
        return '前向（真算）：随机权重 + 因果掩码，记下 6 个位置 × 8 维的输出。右边对照：把掩码整个去掉（允许看未来）';
      }
      if (t === 1) {
        showOut(gC, baseC); showOut(gP, baseP);
        ro.innerHTML = '<span>即将改动：ids[' + cp + '] ' + ids0[cp] + ' → ' + idsAlt[cp] + '（check_causality 同款操作）</span>';
        return '改掉一个<b>未来</b> token：位置 ' + cp + ' 的 id ' + ids0[cp] + ' → ' + idsAlt[cp] + '。问题只有一个：位置 0..' + (cp - 1) + ' 的输出会不会变？';
      }
      const outC = fwd(idsAlt, true), outP = fwd(idsAlt, false);
      if (t === 2 || t === 3) {
        const pastC = showDelta(gC, outC, baseC, -1);
        let lastC = 0;
        for (let i = cp; i < T; i++) for (let c = 0; c < D; c++) lastC = Math.max(lastC, Math.abs(outC[i][c] - baseC[i][c]));
        if (t === 2) {
          gP.g.style.opacity = '.35';
          ro.innerHTML = '<span>因果掩码：过去位置(0..' + (cp - 1) + ') max|Δ| = <b>' + F.fmt(pastC, 6) + '</b></span>' +
            '<span>位置 ' + cp + '..' + (T - 1) + ' max|Δ| = <b>' + F.fmt(lastC, 4) + '</b>（模型不是常数 → 上一条才有意义）</span>';
          return '因果掩码下重算、逐位比较：过去位置 max|Δ| = ' + F.fmt(pastC, 6) + '（<b>逐位相同</b>，不是"差不多"）；位置 ' + cp + ' 自己变了 ' + F.fmt(lastC, 3) + ' —— 这正是判据"改未来 token 不影响过去"+"改当前 token 确实改变当前输出"那一对';
        }
        gP.g.style.opacity = '';
        const pastP = showDelta(gP, outP, baseP, -1);
        ro.innerHTML = '<span>因果掩码：过去 max|Δ| = <b>' + F.fmt(pastC, 6) + '</b></span>' +
          '<span>允许看未来：过去 max|Δ| = <b>' + F.fmt(pastP, 4) + '</b>（≠ 0，判据当场红）</span>';
        return '同一个改动，把掩码写成"允许看未来"（漏加掩码、或加在 softmax 之后）：过去位置 max|Δ| = ' + F.fmt(pastP, 4) + ' ≠ 0 —— 位置 0 都"看见"了未来的改动，因果性被破坏';
      }
      /* t = 4..9：逐位置扫 */
      const p = t - 4;
      const pastC = showDelta(gC, outC, baseC, p);
      const pastP = showDelta(gP, outP, baseP, p);
      gP.g.style.opacity = '';
      let mx = 0; for (let c = 0; c < D; c++) mx = Math.max(mx, Math.abs(outC[p][c] - baseC[p][c]));
      let mp = 0; for (let c = 0; c < D; c++) mp = Math.max(mp, Math.abs(outP[p][c] - baseP[p][c]));
      ro.innerHTML = '<span>位置 ' + p + '：因果掩码 max|Δ| = <b>' + F.fmt(mx, 6) + '</b></span>' +
        '<span>允许看未来 max|Δ| = <b>' + F.fmt(mp, 4) + '</b></span>' +
        '<span>过去位置(0..' + (cp - 1) + ')汇总：因果 <b>' + F.fmt(pastC, 6) + '</b>｜偷看 <b>' + F.fmt(pastP, 4) + '</b></span>';
      const verdict = p < cp
        ? (mx === 0 ? '在改动位置之前：因果掩码下 Δ = 0；偷看掩码下 Δ ≠ 0 —— 差异就出在这里' : '在改动位置之前但 Δ ≠ 0？检查你的掩码')
        : (p === cp ? '改动位置本身：两种掩码下都必须变（否则模型是常数函数）' : '在改动位置之后：两种掩码下都变 —— 影响只向<b>后</b>传播，这是对的');
      return '逐位扫描 t' + p + '：' + verdict;
    }

    const tp = transport(box, 4 + T, render, 950);
    F.caption(box, '判据原文：tests/test_attention.py `改未来 token 不影响过去`（前 T−1 个位置<b>逐位相同</b>）、' +
      '`改当前 token 确实改变当前输出`（反面对照）、`影响只向后传播`。' +
      '本页的两次前向都是现算的；"允许看未来"= 掩码全去掉，对应 s04 §8 症状表"因果性判据挂（前缀变了）"。');
  };

  /* ============================================================= shiftdir */
  R.shiftdir = function (box) {
    box.classList.add('lane-a');
    const V = 5, T = 6;
    const ids = [2, 0, 4, 1, 3, 0];               // 相邻不重复，避免"蒙对"
    F.title(box, 'shiftdir · 语言模型唯一的秘密：预测下一个 token —— 错位一格，loss 就假得离谱');
    const head = F.head(box, 'V=5、T=6 的玩具 logits（固定种子）：这个模型把"预测自己"练到了收敛 —— 每行在 ids[t] 处有一根 +6 的大 logit');

    const rng = normRng(6);
    const logits = [];
    for (let t = 0; t < T; t++) {
      const r = randnVec(V, rng, 0.6);
      r[ids[t]] += 6.0;                           // "背下自己"的模型会长成这样
      logits.push(r);
    }
    const lnV = Math.log(V);

    function lossAt(k) {                          // 真算交叉熵：logits[t] 配 ids[t+k]
      let s = 0, n = 0;
      for (let t = 0; t < T; t++) {
        const u = t + k;
        if (u < 0 || u >= T) continue;
        const p = softmaxRow(logits[t])[ids[u]];
        s += -Math.log(p); n++;
      }
      return s / n;
    }
    const L = { '-1': lossAt(-1), '0': lossAt(0), '1': lossAt(1) };

    const pairG = gx(box, 3, T + 1, 64, 24);
    pairG.ax(0, 0, 't');
    pairG.ax(1, 0, '目标');
    pairG.ax(2, 0, '−ln p');
    for (let t = 0; t < T; t++) pairG.ax(0, t + 1, 'logits[' + t + ']');

    const plotWrap = F.el('div'); box.appendChild(plotWrap);
    const ro = F.readout(box);
    const sl = F.sliders(box);
    const sk = F.slider(sl, '错位量 k', -1, 1, 1, 1, () => tp.draw());

    const KNAME = {
      '1': 'k=+1：`logits[:, :-1]` 配 `ids[:, 1:]` —— <b>对</b>（s05 §8.1）',
      '0': 'k=0：`logits` 配 `ids` —— <b>错</b>，"预测自己"，等于看答案',
      '-1': 'k=−1：配<b>上一个</b> token —— 也是错（判据抓的移位错法之一）'
    };

    function render(t) {
      const k = +sk.input.value;
      let sumShown = 0, nShown = 0;
      for (let u = 0; u < T; u++) {
        const tgt = u + k;
        const valid = tgt >= 0 && tgt < T;
        const cT = pairG.put(1, u + 1, valid ? 'ids[' + tgt + ']=' + ids[tgt] : '×');
        const cL = pairG.put(2, u + 1, '');
        cT.className = valid ? '' : 'off';
        if (!valid) { cL.className = 'off'; cL.textContent = '—'; continue; }
        const p = softmaxRow(logits[u])[ids[tgt]];
        const nlp = -Math.log(p);
        cL.textContent = F.fmt(nlp, 2);
        paint(cL, Math.min(nlp, 6) / 6, 1, nlp < 0.5 ? '62,207,142' : '245,180,69');
        if (u === t) { cT.className = 'hot'; cL.className = 'hot'; }
        if (u <= t) { sumShown += nlp; nShown++; }
      }

      const g = F.plot(plotWrap, 620, 150, -1.45, 1.45, 0, Math.max(2, L['-1'], L['0'], L['1']) * 1.15);
      g.line(() => lnV, '#8d99ac', [5, 4]);
      g.line(x => (Math.abs(x - k) < 0.35 ? L[String(k)] : NaN), '#4aa3ff', null, true);
      g.dots([-1, 0, 1], kk => L[String(kk)], '#4aa3ff');

      const partial = nShown ? sumShown / nShown : 0;
      ro.innerHTML =
        '<span>' + KNAME[String(k)] + '</span>' +
        '<span>loss(k=+1) = <b>' + F.fmt(L['1'], 4) + '</b></span>' +
        '<span>loss(k=0) = <b>' + F.fmt(L['0'], 4) + '</b>（假）</span>' +
        '<span>loss(k=−1) = <b>' + F.fmt(L['-1'], 4) + '</b></span>' +
        '<span>ln V = <b>' + F.fmt(lnV, 3) + '</b>（灰色虚线）</span>' +
        (nShown ? '<span>动画已扫过的 ' + nShown + ' 个位置均值 = <b>' + F.fmt(partial, 4) + '</b></span>' : '');

      const tgt = t + k;
      if (tgt < 0 || tgt >= T) return 't=' + t + '：目标 ids[' + tgt + '] 越界 —— 这个位置不参与 loss（k=+1 时末位、k=−1 时首位就是这么被丢掉的）';
      const p = softmaxRow(logits[t])[ids[tgt]];
      return 't=' + t + '：用 logits[' + t + '] 预测 ids[' + tgt + ']=' + ids[tgt] +
        '；p = ' + F.fmt(p, 4) + '，−ln p = ' + F.fmt(-Math.log(p), 3) +
        (k === 0 ? '（目标就是自己的输入 → 大 logit 命中 → 接近 0，<b>这是作弊</b>）' : '');
    }

    const tp = transport(box, T, render, 900);
    F.caption(box, '判据原文：tests/test_model.py `正确移位：新数据上学不动`（loss 必须停在 ln V 附近）与 ' +
      '`错误移位：预测自己在任何数据上都秒收敛`；stages/s05_char_lm.py 的 `随机数据学不动（反作弊判据）`。' +
      '本页 logits 是"把预测自己练到收敛"的模型会长成的样子（固定种子），三种错位的交叉熵全部现算：' +
      'k=0 掉到 ≈0.02 看着"很健康"，k=+1 才暴露它根本不会预测下一个。真实训练里正确移位的 loss 只能降到 ln V = 1.609 附近 —— 学不动才是对的。');
  };

  /* ============================================================= tiedgrad */
  R.tiedgrad = function (box) {
    box.classList.add('lane-a');
    const V = 4, C = 4, T = 3;
    const ids = [2, 0, 2];                        // token 2 出现两次：散射必须累加
    F.title(box, 'tiedgrad · tied 权重的两路梯度必须 += 累加，写成 = 就是"谁后跑谁赢"');
    F.head(box, '小例子：tok.weight (V=4, C=4)，ids = [2,0,2]，x_flat (3,4)、g_logits (3,4) 固定种子随机');

    const rng = normRng(9);
    const xFlat = randnMat(T, C, rng, 0.8);
    const gLog = randnMat(T, V, rng, 0.3);

    /* ① head 路径：g_head = g_logitsᵀ @ x_flat   (V,C) */
    const gHead = [];
    for (let o = 0; o < V; o++) {
      const r = [];
      for (let c = 0; c < C; c++) { let s = 0; for (let t = 0; t < T; t++) s += gLog[t][o] * xFlat[t][c]; r.push(s); }
      gHead.push(r);
    }
    /* ② 查表路径：散射回对应行（np.add.at 的语义：同 id 累加） */
    const gEmb = [];
    for (let o = 0; o < V; o++) gEmb.push([0, 0, 0, 0]);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) gEmb[ids[t]][c] += gLog[t][c];

    const gPlus = gHead.map((r, o) => r.map((x, c) => x + gEmb[o][c]));
    const gAssign = gEmb.map(r => r.slice());     // = 写法：查表最后跑，head 那路被抹掉
    let maxHead = 0, maxDiff = 0;
    for (let o = 0; o < V; o++) for (let c = 0; c < C; c++) {
      maxHead = Math.max(maxHead, Math.abs(gHead[o][c]));
      maxDiff = Math.max(maxDiff, Math.abs(gPlus[o][c] - gAssign[o][c]));
    }

    const row2 = F.el('div', 'row'); box.appendChild(row2);
    const colA = F.el('div', 'col'); colA.style.flex = '0 0 auto';
    const colB = F.el('div', 'col'); colB.style.flex = '0 0 auto';
    row2.append(colA, colB);
    function mkGrid(host, label) {
      const lab = F.el('div', 'glab', label); host.appendChild(lab);
      const g = gx(host, V, C + 1, 46, 22);
      for (let o = 0; o < V; o++) g.ax(o, 0, 'w' + o);
      g.lab = lab;
      return g;
    }
    const g1 = mkGrid(colA, '① TiedHead.backward（先跑）：grad += g_logitsᵀ @ x_flat');
    const g2 = mkGrid(colA, '② Embedding.backward（后跑）：散射回 ids 对应的行');
    const g3 = mkGrid(colB, '正确写法 `+=`：① + ② 两路都在');
    const g4 = mkGrid(colB, '错误写法 `=`：谁后跑谁赢 → 只剩 ②');

    function fill(gr, M, dim, hotAgainst) {
      gr.g.style.opacity = dim ? '.25' : '';
      for (let o = 0; o < V; o++) for (let c = 0; c < C; c++) {
        const cell = gr.put(o, c + 1, F.fmt(M[o][c], 2));
        paint(cell, M[o][c], 1.2, '185,167,224');
        if (hotAgainst && Math.abs(M[o][c] - hotAgainst[o][c]) > 1e-12) cell.className = 'hot';
      }
    }

    const ro = F.readout(box);

    function render(t) {
      const show1 = t >= 1, show2 = t >= 2, show3 = t >= 3, show4 = t >= 4;
      fill(g1, gHead, !show1);
      fill(g2, gEmb, !show2);
      fill(g3, gPlus, !show3);
      fill(g4, gAssign, !show4, show4 ? gPlus : null);
      for (let o = 0; o < V; o++) g2.cells[o][0].className = 'ax' + (show2 && o === 2 ? ' summed' : '');
      if (show2) for (let c = 0; c < C; c++) g2.cells[2][c + 1].classList.add('summed');

      const a = gPlus[0][0], b = gEmb[0][0], h = gHead[0][0];
      ro.innerHTML =
        '<span>grad[0,0]：① = <b>' + F.fmt(h, 4) + '</b>，② = <b>' + F.fmt(b, 4) + '</b></span>' +
        '<span>`+=` → <b>' + F.fmt(a, 4) + '</b>（= ① + ②）</span>' +
        '<span>`=` → <b>' + F.fmt(b, 4) + '</b>（只剩查表那路）</span>' +
        '<span>max|+= − =| = <b>' + F.fmt(maxDiff, 4) + '</b> = max|①| = <b>' + F.fmt(maxHead, 4) + '</b></span>';

      if (t === 0) return 'zero_grad() 之后：tok.weight.grad 全 0。这块内存有<b>两个用户</b> —— 输出头（借用，weight tying）和查表层；GPT.backward 的顺序是 head → blocks → pos → tok，所以 ① 先写、② 后写';
      if (t === 1) return '① head 路径写入：TiedHead.backward 算出 g_logitsᵀ @ x_flat (4,4)。若这里写 `=`，它先落盘，随后会被 ② 覆盖';
      if (t === 2) return '② 查表路径写入：Embedding.backward 把 g_logits 散射回 ids=[2,0,2] 对应的行 —— token 2 出现两次，行 2 是<b>累加</b>了两行（np.add.at 的语义，蓝框）';
      if (t === 3) return '`+=` 写法：grad = ① + ②。看 grad[0,0]：' + F.fmt(h, 3) + ' + ' + F.fmt(b, 3) + ' = ' + F.fmt(a, 3) + '，两路都在 —— 判据 `tied 的两路梯度是累加不是覆盖` 对的就是这个数';
      if (t === 4) return '`=` 写法：查表最后跑，谁后跑谁赢 → ① 被整个抹掉（黄框 = 与 `+=` 不同的格子，几乎全部）';
      return '丢了多少：max|+= − =| = ' + F.fmt(maxDiff, 4) + '，恰好等于 max|①| = ' + F.fmt(maxHead, 4) + ' —— 覆盖丢掉的正是输出头那一路。整机 loss 仍会降（查表那半是主体），所以只有这条判据抓得住';
    }

    const tp = transport(box, 6, render, 1000);
    F.caption(box, '对应讲义 s05 §2.3 / §6：`self.weight.grad += g2.T @ self._x_flat` 里的 `+=` 是本阶段最重要的一行；' +
      '判据：stages/s05_char_lm.py `tied 的两路梯度是累加不是覆盖`，tests/test_model.py `GPT 整机反向正确(含权重共享)`。' +
      '本页 ①② 与两种写法的最终梯度全部现算，可用 numpy 按 solutions/minigpt/gpt.py 的式子逐位复算。');
  };
})();
