/* 讲义外壳。同一个文件服务两种打开方式：

   · 独立打开（双击 s02.html / 打印）：body 上没有 in-app 类，crumb 和 TOC 都在，
     按钮退化成"跳到测评台并带上参数"的链接。
   · 嵌在测评台里（中栏的"读讲义"视图）：宿主给根节点加 .in-app 并调 Manual.mount(root, host)，
     TOC/crumb 隐藏（左栏已经有了），按钮直接叫宿主干活 —— 在讲义里点"跑这条检查"
     就该看到结果，不用先跳走。

   宿主协议只有一个对象：host = { checkTodo(id), editTodo(id), stage }。
   讲义不假设宿主长什么样，所以测评台改 UI 不会改坏讲义。 */

(function () {
  const LS = 'mgpt.manual';

  function applyFont(px) {
    document.documentElement.style.setProperty('--fs', px + 'px');
    try { localStorage.setItem(LS + '.fs', String(px)); } catch (e) {}
  }

  function fontSlider(root) {
    const el = root.querySelector('#fs');
    if (!el) return;
    let saved = 16;
    try { saved = Number(localStorage.getItem(LS + '.fs')) || 16; } catch (e) {}
    el.value = saved; applyFont(saved);
    el.addEventListener('input', () => applyFont(Number(el.value)));
  }

  /* 侧栏目录跟着滚动高亮 —— 长文里"我在第几节"必须一眼看得到 */
  function tocSpy(root) {
    const links = [...root.querySelectorAll('.toc a[href^="#"]')];
    if (!links.length || !('IntersectionObserver' in window)) return;
    const map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
    const seen = new Set();
    const io = new IntersectionObserver(es => {
      es.forEach(e => { e.isIntersecting ? seen.add(e.target.id) : seen.delete(e.target.id); });
      let top = null;
      for (const [id] of seen) {
        const el = document.getElementById(id);
        if (el && (!top || el.getBoundingClientRect().top < top.getBoundingClientRect().top)) top = el;
      }
      links.forEach(a => a.classList.remove('active'));
      if (top && map.get(top.id)) map.get(top.id).classList.add('active');
    }, { rootMargin: '-15% 0px -70% 0px' });
    root.querySelectorAll('.sec[id]').forEach(s => io.observe(s));
  }

  /* 代码块里正文说"就这三行"的那种强调：把注释里带 ← 的行钉一条亮边 */
  function highlightCode(root) {
    root.querySelectorAll('.code pre code').forEach(pre => {
      const lines = pre.innerHTML.split('\n');
      let hit = false;
      const out = lines.map(ln => {
        if (/←|←——|IMPORTANT|关键/.test(ln) && !/<span class="hl">/.test(ln)) { hit = true; return '<span class="hl">' + ln + '</span>'; }
        return ln;
      });
      if (hit) pre.innerHTML = out.join('\n');
    });
  }

  const ON_HTTP = /^https?:$/.test(location.protocol);

  /* 独立窗口（或单独打开的讲义页）自己跑单项检查。
     files 传空 = 让服务端用 src/ 里**已保存**的那份跑，所以这里要说清"已保存"：
     测评台编辑器里没提交的改动不会体现在这个窗口的结果里。 */
  async function localCheck(btn, tid, stage){
    const box = btn.closest('.sec-act') || btn.parentElement;
    let out = box.querySelector('.fig-result');
    if (!out) { out = document.createElement('span'); out.className = 'fig-result'; box.appendChild(out); }
    const old = btn.textContent;
    btn.disabled = true; btn.textContent = '测…';
    out.className = 'fig-result run'; out.textContent = tid + ' 正在跑（src/ 里已保存的版本）…';
    let r;
    try {
      r = await fetch('/api/check', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({stage: stage, id: tid, files: {}})}).then(x => x.json());
    } catch (e) {
      r = {status: 'error', line: '这个窗口连不上测评台（测评台还在跑吗？）：' + e, output: ''};
    }
    btn.disabled = false; btn.textContent = old;
    const label = {pass: '通过', fail: '不通过', blocked: '待前置', error: '出错'}[r.status] || r.status;
    out.className = 'fig-result ' + r.status;
    // 服务端出错时只有 output 没有 line（例如 KeyError），只显示 line 会是一片空白
    out.textContent = `${tid} → ${label}：` + ((r.line || r.output || '').trim().split('\n')[0] || '（无详情）').slice(0, 200);
    out.title = '跑的是 src/ 里已保存的内容；编辑器里未提交的改动不算。\n\n' + (r.output || '').slice(0, 4000);
  }

  /* 手算例子卡：一次露一行。reveal 而不是全摊开，是逼你先自己算完再看答案；
     全部露完按钮变"重来"，再点一次收回去，方便反复练。 */
  function stepExample(card){
    const lines = [...card.querySelectorAll('.exline')];
    const hid = lines.find(l => l.hidden);
    if (hid) hid.hidden = false;
    else lines.forEach((l, k) => { l.hidden = k > 0; });
    card.querySelector('.excount').textContent =
      lines.filter(l => !l.hidden).length + '/' + lines.length;
    card.querySelector('.exnext').textContent =
      lines.some(l => l.hidden) ? '下一行 ▶' : '重来 ↺';
  }

  /* 交互自查卡：判定是纯字符串比对，两种模式 ——
       accept：去掉所有空格后和几种写法之一完全相同（形状题、数值题）
       keys：  要点覆盖，; 分要点、/ 分同一要点的多种说法；全覆盖才算过
     开放题（只有 ans）不判对错，直接和答案要点对照。 */
  const qzNorm = s => (s || '').replace(/\s+/g, '').replace(/（/g, '(').replace(/）/g, ')')
    .replace(/，/g, ',').replace(/＝/g, '=').replace(/[≈~约]/g, '').toLowerCase();
  function judgeQuiz(card){
    const res = card.querySelector('.qzres');
    const val = (card.querySelector('.qzin').value || '').trim();
    res.hidden = false;
    if (!val) {
      res.className = 'qzres no';
      res.textContent = '先写下你自己的答案再判 —— 写错没关系，错了还有提示接你。';
      return;
    }
    const nv = qzNorm(val);
    const accept = (card.querySelector('.qzaccept').textContent || '').trim();
    const keys = (card.querySelector('.qzkeys').textContent || '').trim();
    if (accept) {
      const alts = accept.split('/').map(qzNorm).filter(Boolean);
      // 包含即算对：自查题不是判据，"ln65≈4.17" 里写出 4.17 就说明会了
      const ok = alts.some(a => nv === a || nv.includes(a));
      res.className = 'qzres ' + (ok ? 'ok' : 'no');
      res.textContent = ok ? '对，和参考写法一致。'
        : '对不上。这题按精确写法判，参考接受：' + alts.join(' 或 ') + '（忽略空格）。想不通就点提示。';
      return;
    }
    if (keys) {
      const groups = keys.split(';').map(g => g.split('/').map(qzNorm).filter(Boolean))
        .filter(g => g.length);
      const miss = groups.filter(g => !g.some(t => nv.includes(t)));
      res.className = 'qzres ' + (miss.length ? 'no' : 'ok');
      res.textContent = miss.length
        ? `要点覆盖 ${groups.length - miss.length}/${groups.length}，还差 ${miss.length} 个 —— 说不出来就点提示，或回讲义那一节。`
        : '要点全覆盖了。再点"看答案要点"对一遍措辞。';
      return;
    }
    res.className = 'qzres no';
    res.textContent = '这题是开放题，机器不判对错：写完直接点"看答案要点"对照。';
  }

  function actions(root, host) {
    // 阶段号在三个地方：注入进测评台时由宿主给（host.stage）；单独打开时写在 <body data-stage>；
    // .chapter 自己没有 —— 只读 root.dataset 会让独立窗口的检查带着空阶段号去请求。
    const stage = (root.dataset && root.dataset.stage) || (host && host.stage)
      || document.body.dataset.stage || '';
    root.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' && ev.target.classList && ev.target.classList.contains('qzin')) {
        ev.preventDefault();
        judgeQuiz(ev.target.closest('.quiz'));
      }
    });
    root.addEventListener('click', ev => {
      // 讲义互链：在测评台里换左栏阶段，别让整页跳走（独立打开时就是普通相对链接）
      const ch = ev.target.closest('a[href^="s0"],a[href^="s1"]');
      if (ch && host) {
        const id = ((ch.getAttribute('href') || '').match(/^(s\d\d)\.html$/) || [])[1];
        if (id) { ev.preventDefault(); host.openStage && host.openStage(id); return; }
      }
      // 指向仓库源码的链接：测评台里用弹窗看那份判据/骨架，独立打开时走相对路径
      const repo = ev.target.closest('a.repo');
      if (repo && host) {
        ev.preventDefault();
        host.openPath && host.openPath(repo.dataset.repo || '', +(repo.dataset.line || 0));
        return;
      }
      const ex = ev.target.closest('.exnext');
      if (ex) { stepExample(ex.closest('.excard')); return; }
      const qz = ev.target.closest('.qzcheck,.qzhint,.qzans');
      if (qz) {
        const card = qz.closest('.quiz');
        if (qz.classList.contains('qzhint')) {
          const hs = [...card.querySelectorAll('.qzh')];
          for (let i = 0; i < +qz.dataset.k && i < hs.length; i++) hs[i].hidden = false;
        } else if (qz.classList.contains('qzans')) {
          const a = card.querySelector('.qza');
          a.hidden = !a.hidden;
          qz.textContent = a.hidden ? '看答案要点' : '收起答案';
        } else judgeQuiz(card);
        return;
      }
      const b = ev.target.closest('button[data-check],button[data-edit],button[data-goto],a.sec-jump');
      if (!b) return;
      if (b.classList.contains('sec-jump')) {                 // 独立打开时它就是普通链接
        if (host) { ev.preventDefault(); host.openStage && host.openStage(stage); }
        return;
      }
      const tid = b.dataset.check || b.dataset.edit || b.dataset.goto || '';
      if (!tid) return;
      ev.preventDefault();
      if (host) {
        if (b.dataset.check && host.checkTodo) host.checkTodo(tid, stage);
        if ((b.dataset.edit || b.dataset.goto) && host.editTodo) host.editTodo(tid, stage);
      } else if (ON_HTTP) {
        // 独立窗口：检查就地跑，跳编辑器交给那个叫 mgpt-app 的窗口（测评台启动时给自己取的名），
        // 这样读的位置不会被抢走。
        if (b.dataset.check) localCheck(b, tid, stage);
        else window.open('/?stage=' + encodeURIComponent(stage) + '&edit=' + encodeURIComponent(tid), 'mgpt-app');
      } else {
        // 本地双击打开（file://）：没有后端可问，只能把意图交给测评台
        location.href = '/?stage=' + encodeURIComponent(stage) +
          (b.dataset.check ? '&check=' : '&edit=') + encodeURIComponent(tid);
      }
    });
  }

  /* 小地图上色：状态一律"读的时候现取"，生成时不写死 —— 否则刚跑完判据，
     点还是灰的，那条带就成了误导。两个来源：
       · 测评台里：宿主给 host.progress()（左栏刚刷过，数据是热的）
       · 独立窗口：自己拉 /api/stages，切回这个窗口时再拉一次 */
  function paintMinimap(root, prog){
    const mm = root.querySelector('.minimap');
    if (!mm) return;
    const pend = (prog && prog.todo_pending) || [];
    const last = (prog && prog.last) || {};
    let left = 0;
    mm.querySelectorAll('.mm').forEach(a => {
      const tid = a.dataset.todo;
      const st = last[tid];
      let state;
      /* 'answer' 是"拿 solutions/ 那份跑出来的 pass"，故意不在下面任何一条里 ——
         它落到 pend/done，也就是只看你自己的 src/ 写到哪一步。 */
      if (st === 'pass') state = 'pass';
      else if (st === 'fail' || st === 'error') state = 'fail';
      else if (pend.includes(tid)) { state = 'pend'; left++; }
      else state = 'done';
      a.dataset.st = state;
      a.title = tid + ' · ' + {pass: '单项检查通过', fail: '单项检查没过',
                               pend: '还没写完', done: '占位已处理，但判据才算数'}[state];
    });
    let tag = mm.querySelector('.mm-left');
    if (!tag) { tag = document.createElement('span'); tag.className = 'mm-left'; mm.appendChild(tag); }
    tag.textContent = left ? `还差 ${left} 处` : '占位都处理了 —— 去跑整卷';
  }

  function refreshMinimap(root, host){
    if (host && host.progress) { paintMinimap(root, host.progress()); return Promise.resolve(); }
    if (!ON_HTTP) return Promise.resolve();
    const stage = (root.dataset && root.dataset.stage) || document.body.dataset.stage || '';
    return fetch('/api/stages').then(x => x.json()).then(r => {
      const s = (r.stages || []).find(v => v.id === stage);
      if (s) paintMinimap(root, s.progress || {});
    }).catch(() => {});
  }

  function typesetMath(root) {
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([root]).catch(() => {});
  }

  const Manual = {
    mount(root, host) {
      if (root.__mgptMounted) { typesetMath(root); refreshMinimap(root, host); return; }
      root.__mgptMounted = true; root.__mgptHost = host || null;
      fontSlider(root); tocSpy(root); highlightCode(root); actions(root, host);
      typesetMath(root);
      refreshMinimap(root, host);
      if (window.Figs) window.Figs.mountAll(root, host);      // 可视化组件（figs.js）
    },
    /* 宿主跑完判据后叫一声，小地图立刻重涂 —— 不用等用户刷新页面 */
    refresh(root) { refreshMinimap(root || document.querySelector('.chapter'), root && root.__mgptHost); },
    _refresh: refreshMinimap
  };
  window.Manual = Manual;

  if (!window.__MGPT_APP) {                                   // 独立打开
    document.addEventListener('DOMContentLoaded', () => {
      const root = document.querySelector('.chapter') || document.body;
      Manual.mount(root, null);
      // 独立窗口没有宿主推进度：切回这个窗口时自己去拉一次
      addEventListener('focus', () => refreshMinimap(root, null));
      document.addEventListener('visibilitychange',
        () => { if (!document.hidden) refreshMinimap(root, null); });
    });
  }
})();
