/* 第二批可视化组件的"底座"。

   为什么另开一个文件，而不直接往 figs.js 里加：
     · figs.js 是八个 → 十五个组件长出来的那份，改它的 REG 表是单点冲突 ——
       多个作者（含后台任务）同时加组件就会互相覆盖，覆盖完的表现是
       "这张图安静地不出现"（mountAll 找不到名字就 return，连报错都没有）。
     · 这里把"挂载"做成可插拔：组件往 window.MGPT_FIGS 里注册即可，
       不需要任何人改动别人的表。

   加载顺序：manual.js 先调 window.Figs.mountAll()（figs.js 的表），
   本文件把 mountAll 包一层，顺带挂 MGPT_FIGS；然后再去拉同批的模块文件，
   每拉到一个就补挂一次 —— 晚到的组件不会漏挂，也不会重复挂（__mounted 挡住了）。

   工具函数（el/sv/fmt/plot/cellEl/move）在这里是**第二份拷贝**：figs.js 的那些
   关在 IIFE 里没导出。两份的取舍写在 figs.js 顶部（"能算的都真的算"），
   改动 plot 的行为时记得两边都看。 */

(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (t, cls, html) => { const e = document.createElement(t); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const sv = (t, at) => { const e = document.createElementNS(NS, t); for (const k in at) e.setAttribute(k, at[k]); return e; };
  const fmt = (v, n = 3) => Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(1) : v.toFixed(n);
  const md = s => String(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  const esc1 = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function title(box, text) { box.appendChild(el('div', 'fig-title', md(text))); }
  function caption(box, text) { box.appendChild(el('div', 'cap', md(text))); }
  function head(box, text) { const h = el('div', 'shape'); h.innerHTML = md(text); box.appendChild(h); return h; }
  function readout(box) { const r = el('div', 'readout'); box.appendChild(r); return r; }
  function stage(box, h) { const s = el('div', 'stage'); if (h) s.style.height = h + 'px'; box.appendChild(s); return s; }
  function sliders(box) { const s = el('div', 'sliders'); box.appendChild(s); return s; }

  /* 一个滑杆：label 上写名字，右边跟着当前值。fn 收到数值就重画。 */
  function slider(host, name, lo, hi, step, val, fn) {
    const lab = el('label');
    lab.innerHTML = `<b>${esc1(name)}</b>`;
    const s = el('input'); s.type = 'range'; s.min = lo; s.max = hi; s.step = step; s.value = val;
    const out = el('span', null, String(val));
    s.oninput = () => { out.textContent = s.value; fn(+s.value); };
    lab.append(s, out); host.appendChild(lab);
    return { set: v => { s.value = v; out.textContent = String(v); }, input: s };
  }

  function cellEl(stageEl, txt, cls) {
    const c = el('div', 'cell' + (cls ? ' ' + cls : ''));
    c.textContent = txt; stageEl.appendChild(c); return c;
  }
  function move(c, x, y) { c.style.transform = `translate(${x}px,${y}px)`; }

  /* 画布：x∈[lo,hi]、y∈[ylo,yhi] → 像素。clear/line/dots 可以反复调用而不重建
     canvas（重建会闪，也会丢事件）。和 figs.js 那份同源，改动要两边对照。 */
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
        for (const x of xs) { g.beginPath(); g.arc(X(x), Y(fn(x)), 3, 0, 7); g.fill(); }
        g.restore();
      },
      X, Y
    };
    api.clear();
    return api;
  }

  window.MGPT_FIG = { NS, el, sv, fmt, md, esc1, title, caption, head, readout, stage,
                     sliders, slider, cellEl, move, plot };
  window.MGPT_FIGS = window.MGPT_FIGS || {};

  /* 挂载：把 figs.js 的 mountAll 包一层。名字先查第一批的表，再查第二批。
     用 __MGPT_LANE2 挡住"同一个包装装两遍"：套两遍不会画重（__mounted 挡住了），
     但每次 mountAll 都要多跑一遍 querySelectorAll，图多的章节会明显变慢。 */
  window.Figs = window.Figs || {};
  if (!window.__MGPT_LANE2) {
    window.__MGPT_LANE2 = true;
    const prev = window.Figs.mountAll;
    window.Figs.mountAll = function (root, host) {
      if (typeof prev === 'function') prev(root, host);
      (root || document).querySelectorAll('.fig[data-fig]').forEach(box => {
        const fn = window.MGPT_FIGS[box.dataset.fig];
        if (!fn || box.__mounted) return;
        box.__mounted = true;
        try { fn(box); } catch (err) {                  // 图坏了不能连累正文
          box.innerHTML = '<div class="cap">这张图没渲染出来：' + esc1(String(err)) + '（正文与判据不受影响）</div>';
        }
      });
    };
  }

  /* 各批的模块文件（figs_a/b/c.js）由页面模板按顺序静态引入 —— 它们要用
     window.MGPT_FIG，所以必须排在 figs2.js 后面。加一批 = 改 md_to_html.py 的
     模板一行 + manual_assets.py 登记一行，别在这里动态插 <script>：动态加载
     会晚于 manual.js 的挂载，就得再补一次 mountAll，多一条时序要记。 */
})();
