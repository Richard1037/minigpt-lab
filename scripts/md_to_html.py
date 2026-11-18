"""把 docs/stage-0N-*.md 一次性转成 docs/manual/sNN.html —— 测评台的第二个视图。

只在"md → html 迁移"这一次运行；转完之后 HTML 就是唯一真相（见 README §6）。

为什么要有这个脚本而不是手抄：四篇讲义共 1938 行、140 个表格行、43 个代码块、
15 个折叠参考答案。手抄一定会丢内容，而判据全写在讲义里，丢一句"要缓存什么"
就等于埋一个 bug。所以转换 + **保真校验**是硬闸门：

    python scripts/md_to_html.py            # 生成 docs/manual/s0*.html
    python scripts/md_to_html.py --check    # 只比对文本，报告丢失/改写的行；0 丢失才许继续

生成的页面是**双用**的：
  · 测评台 fetch 它、取 <article class="chapter"> 注入中栏（和编辑器共用左栏进度）；
  · 也可以直接双击用浏览器打开读 / 打印（那时顶部 crumb 才显示，按钮变成回链）。

章节头那张"代码在哪 / 你要做什么"的表不在 md 里手写，而是从两处现取：
webapp/server.py 的 STAGES 登记表（判据命令、要交的文件、TODO 列表）和 src/ 里每个
TODO 占位符的**当前行号** —— 这样讲义不会和代码对不上。
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import re
import sys
from urllib.parse import quote

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS, OUT = ROOT / "docs", ROOT / "docs" / "manual"
sys.path.insert(0, str(ROOT / "webapp"))
from server import STAGES, JSON_MARK, _pending_in, expand_todo, progress_for   # noqa: E402  同一套判断，别写两遍
from manual_assets import SERVE, url_for, by_url             # noqa: E402  静态资源登记表
from checks import CHECKS                    # noqa: E402  ```checks 围栏的唯一真源

TODO_MARK = re.compile(r'NotImplementedError\("(TODO-[0-9A-Za-z/]+)')
SEC_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s")
TODO_IN_TITLE = re.compile(r"TODO-[0-9]+(?:/[0-9]+)*[A-Za-z]?")

# 可视化挂载点：(阶段, 节号) → 组件名。组件实现在 assets/figs.js，
# 讲义里只留一个空壳 div，所以图坏了也不影响正文可读。
#
# 键是 **h2 的节号**，图落在这一节末尾。要精确插到某个 h3 下面，在 md 里单独一行写
# `@@fig 组件名@@`（可空格分隔多个）—— 它和 `---` 一样算排版指令，不进保真校验。
# 阶段 3 拆成 3a/3b/3c 时，下面两个键还留着 `s03`，结果那两张图静默消失了一个多月：
# 挂载表按 id 精确匹配，写错不报错。改阶段 id 必须连这张表一起改（audit_pointers 查不出来，
# 因为它只查 TODO 锚点）。
FIGS = {
    ("s01", "6"): "graph",          # 计算图：正向搭图 / 反向摊梯度
    ("s01", "2"): "losscurve",      # loss 与下降倍数（数据来自判据的 sparkline）
    ("s02", "2"): "steps",          # 8 个 tanh 台阶叠加拟合 sin（可拖）
    ("s02", "10"): "capacity",      # 容量不够 vs lr 没调好：三条曲线对照
    # ("s03a", "2") 的 shapes 改成了 md 里 @@fig shapes@@ 内联挂载 ——
    # 它后面要紧跟 shapes3d（3D 张量动线），表挂载只能落在小节末尾，控制不了相邻顺序。
    ("s03a", "5"): "gradcheck",     # 有限差分误差量级：对的是 1e-10，错的是 1e-1
    ("s04", "2"): "mask",           # 因果掩码矩阵，点一行看它能看谁
    ("s04", "3"): "heads",          # 拆头/合头的形状动画
}

FIG_INLINE = re.compile(r"^\s*@@fig\s+([a-z0-9_ ]+)@@\s*$")


def esc(s: str) -> str:
    return html.escape(s, quote=False)


PY_TOKEN = re.compile(r"""
    (?P<tstr>[rbfuRBFU]{0,2}(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"))
  | (?P<str>[rbfuRBFU]{0,2}(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))
  | (?P<com>\#[^\n]*)
  | (?P<deco>^[ \t]*@[A-Za-z_][\w.]*)
  | (?P<fn>(?<=def\s)[A-Za-z_]\w*)
  | (?P<cls>(?<=class\s)[A-Za-z_]\w*)
  | (?P<num>\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?\b)
  | (?P<kw>\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del
      |elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or
      |pass|raise|return|try|while|with|yield)\b)
  | (?P<bi>\b(?:abs|all|any|bool|dict|enumerate|float|getattr|hasattr|int|isinstance|len
      |list|map|max|min|print|property|range|reversed|set|setattr|sorted|str|sum|super
      |tuple|type|zip|Exception|ValueError|RuntimeError|NotImplementedError)\b)
""", re.X | re.M)


def py_highlight(src: str) -> str:
    """Python 围栏 → 带 span 的高亮。在生成时做，页面不挂 CDN 脚本。

    正则按"字符串 → 注释 → 词"的顺序匹配：先吃掉字符串，里面的 # 和关键字
    就不会被误判；到 # 时前面的引号分支配不上，才轮到注释。
    """
    out, pose = [], 0
    for m in PY_TOKEN.finditer(src):
        out.append(esc(src[pose:m.start()]))
        out.append(f'<span class="{m.lastgroup}">{esc(m.group())}</span>')
        pose = m.end()
    out.append(esc(src[pose:]))
    return "".join(out)


def link_html(label: str, url: str) -> str:
    """md 链接 → 讲义里能用的 <a>。

    原文档里写的是 file:///D:/Transformer/... 这种绝对链接：换台机器就断，
    在测评台里点了浏览器只会去开本地文件。统一改写成两种开法都成立的地址：

      · docs/stage-0N-*.md   → sNN.html      讲义互链（站内 /manual/s02，站外同目录）
      · 其他仓库内的文件      → ../../<rel>   并留 data-repo/data-line，
        在测评台里由宿主拦下、用弹窗显示那份判据/骨架源码并滚到那一行
      · 站外 http(s) 链接     → 原样
    """
    base, _, frag = url.partition("#")
    if base.startswith("file:///"):
        base = re.sub(r"^file:///(?:[A-Za-z]:)?/", "", base)
        i = base.find("Transformer/")
        if i >= 0:
            base = base[i + len("Transformer/"):]
    from urllib.parse import unquote
    base = unquote(base)
    m = re.search(r"docs/stage-(\d\d[a-z]?)", base)
    if m and base.endswith(".md"):
        return f'<a href="s{m.group(1)}.html{("#" + frag) if frag else ""}">{label}</a>'
    if base.startswith(("docs/", "src/", "stages/", "solutions/", "webapp/")):
        line = f' data-line="{esc(frag[1:])}"' if re.fullmatch(r"L\d+", frag or "") else ""
        return f'<a class="repo" href="../../{esc(base)}" data-repo="{esc(base)}"{line}>{label}</a>'
    return f'<a href="{esc(url)}">{label}</a>'


def inline(s: str) -> str:
    """行内 markdown → HTML。`$…$` / `$$…$$` 原样留着，交给 MathJax。"""
    out = esc(s)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(^|[^*\w])\*([^*\n]+)\*(?!\*)", r"\1<i>\2</i>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: link_html(m.group(1), m.group(2)), out)
    return out


def sec_id(text: str) -> str:
    m = SEC_NUM.match(text)
    return f' id="sec-{m.group(1).replace(".", "-")}"' if m else ""


ARROW_ONLY = re.compile(r"^\s*[↓▼⬇│|]+\s*$")
TREE_PRE = re.compile(r"^([│├└─|↓▼⬇]+)\s+(.*)$")
CIRCLED = re.compile(r"^([①-⑳])\s*(.*)$")


def flow_html(buf: list[str]) -> str:
    """```flow 围栏 → 竖向动线图（卡片 + 序号 + 连接线），不是等宽文本。

    为什么要有这个：动线讲的是"接下来按这个顺序走"，用 ```text 画会把它压成
    ASCII —— 字号、留白、主次全丢，读起来像在看源码。但正文里的伪代码、
    具体数字的矩阵、命令行输出**仍然该用代码块**，那些要的是等宽对齐。

    保真校验要求 md 里的可见行一行都不能少，所以连接符（↓ ▼ │）原样留在 DOM 里，
    只是换成 <i> 由 CSS 决定长什么样；行首的树形符号也照样存在，不吞字。

    序号只给"步骤卡片"编。行首带 `│ ├─ └─ ↓ ▼` 的行算连接线/注释（例如
    `↓ 做完能停下来：判据第 1 条绿`），不占序号，否则纯步骤列表的编号会跳。
    行首写死 `①②③` 的，圆圈号直接当序号 —— 正文里如果到处用 ① 回指这几步，
    换成自动 1/2/3 就会把那些回指全指错。

    行首缩进（每 3 个空格算一层）会变成卡片的左边距 —— 树形图里"谁挂在谁下面"
    在等宽文本里靠空格表达，换成卡片后必须换个方式表达，否则 `└─` 的子行和
    父行看起来是同一级。
    """
    parts, n = [], 0
    for raw in buf:
        s = raw.strip()
        if not s:
            continue
        if ARROW_ONLY.match(s):
            parts.append(f'<i class="fa">{esc(s)}</i>')
            continue
        depth = (len(raw) - len(raw.lstrip())) // 3
        pad = f' style="margin-left:{depth * 20}px"' if depth else ""
        c = CIRCLED.match(s)
        if c:
            parts.append(f'<div class="fs"{pad}><i class="n circ">{c.group(1)}</i>'
                         f'<span>{inline(c.group(2))}</span></div>')
            continue
        m = TREE_PRE.match(s)
        if m:
            parts.append(f'<div class="fs"{pad}><i class="mk">{esc(m.group(1))}</i>'
                         f'<span>{inline(m.group(2))}</span></div>')
        else:
            n += 1
            parts.append(f'<div class="fs"{pad}><i class="n">{n}</i>'
                         f'<span>{inline(s)}</span></div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def example_html(buf: list[str]) -> str:
    """```example 围栏 → 手算例子卡：一次只露一行，点"下一行"再露一行。
    第一行默认可见（通常是"设 …"那行前提），其余 hidden；manual.js 负责揭示。"""
    lines = [inline(b.strip()) for b in buf if b.strip()]
    body = "".join(f'<div class="exline"{" hidden" if k else ""}>{t}</div>'
                   for k, t in enumerate(lines))
    return (f'<div class="excard"><div class="exhead">'
            f'<span class="extag">手算例子 · 先自己算，再点下一行</span>'
            f'<button class="exnext" type="button">下一行 ▶</button>'
            f'<span class="excount">1/{len(lines)}</span></div>'
            f'<div class="exbody">{body}</div></div>')


def quiz_html(buf: list[str]) -> str:
    """```quiz 围栏 → 交互自查卡：输入答案当场判、三级提示、讲义小节直达。

    字段（每行一个前缀，其余行续接到上一个字段）：
      q:      题干（可多行）
      accept: 精确判定的几种写法，用 / 分隔（判定时先去掉所有空格再比）
      keys:   要点关键词；要点之间用 ; 分隔，同一要点的多种说法用 / 分隔；全覆盖算过
      hint1/2/3: 三级提示，从最抽象到最具体
      link:   讲义锚点（如 sec-2-3），点提示还不会时跳回去读
      ans:    答案要点（点"看答案要点"才展开）
    分隔符不用 | 而用 /：保真门在 md 侧会把 | 剥掉，两边就对不上了。
    """
    f: dict[str, str] = {}
    cur = None
    for raw in buf:
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^(q|accept|keys|hint1|hint2|hint3|link|ans)\s*[:：]\s*(.*)$", s)
        if m:
            cur = m.group(1)
            f[cur] = m.group(2)
        elif cur:
            f[cur] = f.get(cur, "") + " " + s
    hints = [f.get(k, "") for k in ("hint1", "hint2", "hint3")]
    hints = [h for h in hints if h]
    link = f.get("link", "")
    head = (f'<div class="qzhead"><span class="qztag">自查题</span>'
            f'<span class="qzq">{inline(f.get("q", ""))}</span></div>')
    ctrl = ('<div class="qzctrl"><input class="qzin" type="text" placeholder="把你的答案写在这儿" '
            'autocomplete="off">'
            '<button class="qzcheck" type="button">判一下</button>'
            + "".join(f'<button class="qzhint" type="button" data-k="{k}">提示 {k}</button>'
                      for k in range(1, len(hints) + 1))
            + ('<button class="qzans" type="button">看答案要点</button>' if f.get("ans") else "")
            + (f'<a class="qzlink" href="#{link}">回讲义读这一节</a>' if link else "")
            + '</div>')
    body = ('<div class="qzbody">'
            + "".join(f'<div class="qzh" hidden>{inline(h)}</div>' for h in hints)
            + f'<div class="qzres" hidden></div>'
            + (f'<div class="qza" hidden>{inline(f.get("ans", ""))}</div>' if f.get("ans") else "")
            + '</div>')
    # accept/keys/link 放 hidden div 而不是 data-* 属性或锚点文字：保真门剥标签时
    # 属性里的字会丢、锚点文字和 md 那行对不上；hidden 里的字不丢（和例子卡同一道理）。
    meta = (f'<div class="qzaccept" hidden>{esc(f.get("accept", ""))}</div>'
            f'<div class="qzkeys" hidden>{esc(f.get("keys", ""))}</div>'
            f'<div class="qzlinkv" hidden>{esc(link)}</div>')
    return f'<div class="quiz">{head}{ctrl}{body}{meta}</div>'


def card_purpose(code: str) -> str:
    """从一条快测的 code 里把 card() 的第一个参数（「目的」）抠出来。

    抠而不复制：判据的「为什么存在」只写在 checks.py 里那一处，讲义和测评台读同一句。
    手写一份到 md 里，判据一改它就腐烂 —— 阶段 3 那版 §7 已经腐烂过一轮。
    """
    import ast
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "card":
            a = node.args[0]
            try:
                return str(ast.literal_eval(a))
            except Exception:
                return "".join(p.value for p in getattr(a, "values", [])
                               if isinstance(p, ast.Constant) and isinstance(p.value, str))
    return ""


_JUDGE_CACHE: dict[str, list[dict]] = {}


def judge_rows(judge: str) -> list[dict]:
    """跑一遍这一卷的判据（用 solutions/ 那份答案），把 Report 自己承认的清单拿过来。

    为什么跑而不解析源码：判据名有的是循环里的变量（Linear 那两条就是 `name`），
    静态抠不出来；而 rows 就是这张考卷的目录 —— 用它，讲义和真跑出来的结果不可能对不上。
    答案跑一次约 1~2 秒，整份讲义构建时每种 judge 只跑一次（见缓存）。
    """
    if judge in _JUDGE_CACHE:
        return _JUDGE_CACHE[judge]
    import subprocess
    rows: list[dict] = []
    env = dict(os.environ, MGPT_ANSWER="1", MGPT_JSON="1", PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run([sys.executable, str(ROOT / judge)], capture_output=True,
                           env=env, cwd=str(ROOT), timeout=600)
        for ln in p.stdout.decode("utf-8", "replace").split("\n"):
            if JSON_MARK in ln:
                rows = json.loads(ln.split(JSON_MARK, 1)[1]).get("rows", [])
                break
    except Exception as e:                                  # 跑不动就交空表，让围栏自己说明
        print(f"警告：判据 {judge} 跑不出机器可读清单（{type(e).__name__}: {e}）")
    _JUDGE_CACHE[judge] = rows
    return rows


def _table(head_cols: list[str], rows: list[str], caption: str, note: str) -> str:
    return ('<div class="checkstable"><div class="cfhead"><span class="cftag">' + caption
            + "</span></div><table><thead><tr>"
            + "".join(f"<th>{c}</th>" for c in head_cols)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            + f'<p class="cfnote">{note}</p></div>')


def checks_html(buf: list[str]) -> str:
    """```checks 围栏 → 这一卷的判据逐条表，内容全部现取，md 里一个字都不抄。

      judge: tests/test_layers_a.py  → 整卷判据（组 / 名字 / 这条查什么），来自 Report 的 rows
      stage: s03a                    → 单项快测（id / 标题 / 目的），来自 webapp/checks.py

    以前这一节是我手抄的散文，抄的正是判据里已经写着的那几句话，而且抄出来那份更弱
    （没数字、没实时结论、判据一改就腐烂）。阶段 3 那版 §7 就是这么腐烂了一轮。
    """
    field = dict(re.findall(r"^(judge|stage)\s*[:：]\s*(\S+)\s*$",
                            "\n".join(b.strip() for b in buf), re.M))
    if field.get("judge"):
        judge = field["judge"]
        rows = [r for r in judge_rows(judge) if r.get("kind") != "info"]
        if not rows:
            return _table([], [], f"判据表：{judge} 没跑出机器可读清单",
                          "在命令行加 <code>MGPT_JSON=1</code> 跑一次可以看出原因。")
        body, last = [], None
        for r in rows:
            grp = r.get("group") or ""
            gcell = f'<td class="cgrp">{inline(grp)}</td>' if grp != last else '<td class="cgrp"></td>'
            last = grp
            body.append(f"<tr>{gcell}<td>{inline(r.get('name', ''))}</td>"
                        f"<td>{inline(r.get('detail', ''))}</td></tr>")
        return _table(["组", "判据", "这条在判什么、写错了会是什么症状"], body,
                      f"判据逐条 · 本表现取：跑一遍 solutions/ 那份答案，把 {judge} 的 Report 每行原样搬来",
                      f"跑法：<code>python {judge}</code>。<b>表里的数字是答案跑出来的</b>，不是你的 —— "
                      "你的那一行在中栏「跑整卷」的结果里，长得很像但数值不会一样。")
    stage = field.get("stage", "")
    items = [c for c in CHECKS if c["stage"] == stage]
    if not items:
        return _table([], [], f"checks.py 里没有 stage={stage} 的单项快测", "")
    body = [f'<tr><td class="cid">{inline(c["id"])}</td><td>{inline(c["label"])}</td>'
            f"<td>{inline(card_purpose(c['code']) or '（这条没写 card 目的）')}</td></tr>"
            for c in items]
    return _table(["TODO", "查的是哪条", "这条到底在判什么、写错了会是什么症状"], body,
                  "单项快测逐条 · 本表由 webapp/checks.py 现取",
                  "右侧「单项快测」可以只跑其中一条，输出是它自己的目的 / 输入 / 输出 / 判定四段，带真实数字。")


# ------------------------------------------------------------------ 块级解析
def blocks(md: str) -> list[str]:
    lines, out, para = md.split("\n"), [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
            para.clear()

    i = 0
    while i < len(lines):
        ln = lines[i]

        if re.match(r"^\s*<details\b", ln, re.I):            # 折叠参考答案，内部继续转
            buf = []
            while i < len(lines) and not re.match(r"^\s*</details>", lines[i], re.I):
                buf.append(lines[i]); i += 1
            buf.append(lines[i] if i < len(lines) else "</details>"); i += 1
            text = "\n".join(buf)
            m = re.search(r"<summary>(.*?)</summary>", text, re.S)
            head, body = (m.group(1), text[m.end():]) if m else ("参考答案", text)
            body = re.sub(r"\s*</details>\s*$", "", body, flags=re.I)   # 收尾标签自己会加
            flush()
            # 写作源在 <summary> 里混了 HTML（`<code>__mul__</code>`）。整段转义会把
            # 标签变成看得见的 "&lt;code&gt;"；先把这类标签折成 markdown 反引号，
            # 再走 inline() —— 行内代码还是行内代码，别的字符照旧转义。
            head = re.sub(r"<code>(.+?)</code>", r"`\1`", head, flags=re.S)
            out.append('<details class="answer"><summary>' + inline(head) + "</summary>"
                       '<div class="answer-body">' + "\n".join(blocks(body)) + "</div></details>")
            continue

        mf = FIG_INLINE.match(ln)                             # @@fig 名字@@ → 空壳 div
        if mf:
            flush()
            out.extend(fig_div(n) for n in mf.group(1).split())
            i += 1
            continue

        if re.match(r"^\s*```", ln):                          # 代码块
            flush()
            lang = re.match(r"^\s*```\s*(\S*)", ln).group(1).lower()
            buf = []; i += 1
            while i < len(lines) and not re.match(r"^\s*```", lines[i]):
                buf.append(lines[i]); i += 1
            i += 1
            if lang == "flow":                                # 动线图，不是代码
                out.append(flow_html(buf))
                continue
            if lang == "example":                             # 手算例子卡：点一下出一行
                out.append(example_html(buf))
                continue
            if lang == "quiz":                                # 交互自查卡：输入+判定+提示
                out.append(quiz_html(buf))
                continue
            if lang == "checks":                              # 判据逐条表：内容取自 checks.py，不在 md 里手写
                out.append(checks_html(buf))
                continue
            body = chr(10).join(buf)
            code = py_highlight(body) if lang in ("python", "py") else esc(body)
            out.append('<figure class="code"><figcaption>' + esc(lang or "text") + "</figcaption>"
                       f"<pre><code>{code}</code></pre></figure>")
            continue

        if re.match(r"^\|", ln) and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush()                                            # 表格
            rows = []
            while i < len(lines) and re.match(r"^\|", lines[i]):
                rows.append(lines[i]); i += 1
            cells = lambda r: [inline(c.strip()) for c in r.strip().strip("|").split("|")]
            t = "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells(rows[0])) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in cells(r)) + "</tr>" for r in rows[2:])
            out.append(t + "</tbody></table>")
            continue

        if re.match(r"^#{1,6}\s", ln):
            flush(); out.append("@" + ln); i += 1; continue

        if re.match(r"^\s*---+\s*$", ln):
            flush(); i += 1; continue                          # 节线由 <section> 自己承担

        if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            flush()                                            # 列表（含一层缩进）
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append((len(mm.group(1)), mm.group(2), mm.group(3))); i += 1
            ordered = items[0][1][0].isdigit()
            tag, html_out, opened = ("ol" if ordered else "ul"), [], 0
            for ind, mark, txt in items:
                if ind > opened:
                    html_out.append(f"<{tag}>"); opened = ind
                elif ind < opened:
                    html_out.append(f"</{tag}>"); opened = ind
                    if not opened:
                        html_out.append(f"<{tag}>")
                html_out.append(f"<li>{inline(txt)}</li>")
            html_out.append(f"</{tag}>")
            out.append("".join(html_out))
            continue

        if not ln.strip():
            flush(); i += 1; continue

        para.append(ln); i += 1

    flush()
    return out


# ------------------------------------------------------------------ 章节切分
def sections(md: str) -> list[dict]:
    secs, cur = [], None
    for blk in blocks(md):
        if blk.startswith("@"):
            ln, title = blk[1:], blk.lstrip("@").lstrip("#").strip()
            level = len(ln) - len(ln.lstrip("#"))
            if level == 1:
                continue
            if level == 2:
                m = SEC_NUM.match(title)
                cur = dict(id=f"sec-{m.group(1).replace('.', '-')}" if m else "sec-x",
                           num=m.group(1) if m else "", title=title, body=[])
                secs.append(cur); continue
            blk = f"<h{level + 1}{sec_id(title)}>{inline(title)}</h{level + 1}>"
        if cur is None:
            cur = dict(id="sec-0", num="", title="", body=[]); secs.append(cur)
        cur["body"].append(blk)
    for s in secs:
        s["html"] = "\n".join(s.pop("body"))
    return secs


# ------------------------------------------------------------------ 代码地图
def todo_index(stage: dict) -> list[dict]:
    """每个 TODO 现在在哪个文件、第几行、写完没有。

    "写完没有"复用站点的 _pending_in（占位不见了 / 被 return 挤成死代码 / 整行注释掉
    都算已处理），免得网站说写完、讲义说没写完。
    """
    rows = []
    for text in stage["todos"]:
        # 用网站的 expand_todo 拆编号：登记表里写 "nn.py TODO-5/6 Linear"，
        # 骨架里却是两句 raise（TODO-5、TODO-6）。整条 "TODO-5/6" 去比 _pending_in
        # 的结果永远比不中，于是讲义显示"已写"而网站显示"0/7" —— 就是这类裂缝。
        for tid in expand_todo(text):
            at = None
            for rel in stage["files"]:
                p = ROOT / "src" / rel
                if not p.exists():
                    continue
                lines = p.read_text(encoding="utf-8").split("\n")
                if tid not in _pending_in(lines):
                    continue
                for n, ln in enumerate(lines, 1):
                    m = TODO_MARK.search(ln)
                    if m and m.group(1) == tid and "#" not in ln[:ln.index("NotImplementedError")]:
                        at = (rel, n); break
                if at:
                    break
            rows.append(dict(id=tid, desc=text, at=at))
    return rows


def code_panel(stage: dict) -> str:
    tr = []
    for r in todo_index(stage):
        if r["at"]:
            rel, n = r["at"]
            where = f'<code class="path">{esc(rel)}</code> : <b class="ln">{n}</b>'
            state = '<span class="pill todo">待写</span>'
        else:
            where, state = '<span class="dim">骨架里已没有占位</span>', '<span class="pill done">已写</span>'
        tr.append(f'<tr data-todo="{esc(r["id"])}"><td><code>{esc(r["id"])}</code></td>'
                  f"<td>{esc(r['desc'])}</td><td>{where}</td><td>{state}</td>"
                  f'<td><button class="mini-go" data-goto="{esc(r["id"])}">去写</button></td></tr>')
    files = "、".join(f"<code>{esc(f)}</code>" for f in stage["files"])
    return f"""<section class="panel" id="panel-task">
<h2>你要写的代码，具体在这儿</h2>
<p class="lead">判据 = <code>{esc(stage['judge'])}</code>（命令行和测评台跑的是同一个脚本）。
要交的文件：{files}。预计 {esc(stage['mins'])}。</p>
<table class="todo-table"><thead><tr><th>编号</th><th>写什么</th><th>位置</th><th>状态</th><th></th></tr></thead>
<tbody>{''.join(tr)}</tbody></table>
<p class="note">位置是<b>生成时</b>的行号（重跑 <code>python scripts/md_to_html.py</code> 就刷新）。
你在改代码，行号一定会漂 —— 真正可靠的定位方式是<b>在文件里搜编号本身</b>：
<code>grep -n "TODO-3" src/minigpt/*.py</code>。
状态只看骨架里那句占位 <code>raise NotImplementedError</code> 还在不在，<b>它不判对错</b> ——
对错只有判据说。</p></section>"""


# ------------------------------------------------------------------ 页面组装
def minimap(stage: dict, secs: list[dict]) -> str:
    """章顶那条 TODO 小地图：一个编号一个点，滚到哪都能一眼看到还差几处。

    点本身不带状态 —— 状态是**读的时候**现取的（测评台用宿主的进度，
    独立窗口自己拉 /api/stages），所以生成一次、状态跟着判据跑动。
    锚点用"标题里出现这个编号的那一节"，和正文的 §N 是同一套编号。
    """
    home = {}
    for s in secs:
        for tid in expand_todo(s["title"]):
            home.setdefault(tid, s["id"])
    items = []
    for r in todo_index(stage):
        sec = home.get(r["id"], "")
        items.append(
            f'<a class="mm" data-todo="{esc(r["id"])}"'
            + (f' href="#{sec}"' if sec else "")
            + f' title="{esc(r["desc"])}"><i></i><code>{esc(r["id"])}</code></a>')
    return ('<nav class="minimap" aria-label="TODO 小地图">'
            + "".join(items)
            + '<span class="mm-key"><b class="k-pend"></b>待写<b class="k-done"></b>已写'
            + '<b class="k-pass"></b>单项过<b class="k-fail"></b>单项挂</span></nav>')


def fig_div(name: str) -> str:
    return (f'<div class="fig" data-fig="{name}" role="img" '
            f'aria-label="可视化：{name}"><noscript>这一节的图需要浏览器脚本；'
            f"正文和判据不受影响。</noscript></div>")


def page(stage: dict, md: str) -> str:
    secs = sections(md)
    title = re.search(r"^#\s+(.+)$", md, re.M).group(1)
    toc = "".join(f'<a href="#{s["id"]}"><span>{esc(s["num"] or "·")}</span>{inline(s["title"])}</a>'
                  for s in secs if s["title"])
    body = []
    for s in secs:
        head = f"<h2>{inline(s['title'])}</h2>" if s["title"] else ""
        tids = sorted(set(TODO_IN_TITLE.findall(s["title"])))
        acts = ""
        if tids:
            acts = ('<div class="sec-act">' + "".join(
                f'<button data-check="{t}">跑 {t} 的单项检查</button>'
                f'<button data-edit="{t}">跳到编辑器</button>' for t in tids) +
                f'<a class="sec-jump" href="/?stage={stage["id"]}">在测评台里打开 →</a></div>')
        fig = FIGS.get((stage["id"], s["num"]))
        body.append(f'<section class="sec" id="{s["id"]}" data-stage="{stage["id"]}">'
                    + head + acts + s["html"] + (fig_div(fig) if fig else "") + "</section>")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<link rel="stylesheet" href="assets/manual.css">
<script src="assets/mj-config.js"></script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
        onerror="window.NO_MATHJAX=1"></script>
</head>
<body class="chapter-page" data-stage="{stage['id']}">
<article class="chapter" id="ch-{stage['id']}">
{minimap(stage, secs)}
<header class="hero">
  <div class="crumb"><a href="/?stage={stage['id']}">← 测评台</a> ·
    <a href="index.html">讲义总览</a> · 阶段 {stage['id']}</div>
  <h1>{inline(title)}</h1>
  <div class="meta"><span>判据 <code>{esc(stage['judge'])}</code></span>
    <span>预计 {esc(stage['mins'])}</span>
    <span>要交 {esc("、".join(stage['files']))}</span></div>
  {code_panel(stage)}
</header>
<nav class="toc" aria-label="目录"><div class="toc-h">目录</div>{toc}</nav>
<main>{''.join(body)}</main>
</article>
<script src="assets/manual.js"></script>
<script src="assets/figs.js"></script>
<script src="assets/figs2.js"></script>
<script src="assets/figs_a.js"></script>
<script src="assets/figs_b.js"></script>
<script src="assets/figs_c.js"></script>
<script src="assets/figs_d.js"></script>
<script src="assets/figs_e.js"></script>
<script src="assets/figs_f.js"></script>
<script src="assets/figs_g.js"></script>
</body></html>
"""


# ------------------------------------------------------------------ 讲义总览
def overview(stages: list[dict]) -> str:
    """docs/manual/index.html —— 「我现在该写哪一行」的全局视图。

    四章讲义各自有代码地图，但阶段之间跳的时候需要一张总表：还剩哪些 TODO、
    各在哪个文件第几行、这一阶段的判据跑过没有。这张表全部从 STAGES + src/ +
    webapp/state.json 现取（progress_for / todo_index 和网站共用），
    所以不会出现"网站说写完了、讲义说没写完"。

    第三块是静态资源登记表（同一份数据也供 `python webapp/manual_assets.py` 自检）：
    测评台和讲义互相引用，清单只写在一处，加文件时照着它改，别凭记忆。
    """
    live = [s for s in stages if s["available"]]

    prog = []
    for st in live:
        p = progress_for(st)
        suite = (f"{p['suite'].get('got', 0)}/{p['suite'].get('total', 0)}"
                 + ("（过期）" if p["suite_stale"] else "")) if p["suite"].get("status") else "未跑"
        prog.append(
            f'<tr><td><a href="{st["id"]}.html">{esc(st["title"])}</a></td>'
            f'<td class="num"><b>{p["todo_done"]}</b>/{p["todo_total"]}</td>'
            f'<td class="num">{p["checks_pass"]}/{p["checks_total"]}</td>'
            f'<td class="num">{suite}</td>'
            f'<td><code>{esc(st["judge"])}</code></td>'
            f'<td class="num">{esc(st["mins"])}</td></tr>')
    for st in stages:
        if st["available"]:
            continue
        prog.append(f'<tr class="unpub"><td>{esc(st["title"])}</td>'
                    f'<td class="num dim" colspan="5">骨架、讲义、判据还没发布 —— '
                    f'三样是一起发的，缺一样判据就不可信</td></tr>')

    tables = []
    for st in live:
        rows = []
        for r in todo_index(st):
            if r["at"]:
                rel, n = r["at"]
                where = f'<code class="path">{esc(rel)}</code> : <b class="ln">{n}</b>'
                state = '<span class="pill todo">待写</span>'
            else:
                where, state = '<span class="dim">骨架里已没有占位</span>', '<span class="pill done">已写</span>'
            rows.append(f'<tr data-todo="{esc(r["id"])}"><td><code>{esc(r["id"])}</code></td>'
                        f"<td>{esc(r['desc'])}</td><td>{where}</td><td>{state}</td>"
                        f'<td><a class="mini-go" href="/?stage={st["id"]}&edit={quote(r["id"])}">去写 →</a></td></tr>')
        left = sum(1 for r in todo_index(st) if r["at"])
        tables.append(
            f'<section class="sec" id="tasks-{st["id"]}" data-stage="{st["id"]}">'
            f'<h2><a href="{st["id"]}.html">{esc(st["title"])}</a>'
            f'<span class="sub">{("还有 " + str(left) + " 处待写") if left else "骨架里的占位都已处理 —— 去跑整卷判据"}</span></h2>'
            f'<table class="todo-table"><thead><tr><th>编号</th><th>写什么</th><th>位置</th>'
            f'<th>状态</th><th></th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>')

    served = by_url()
    cols = []
    for prefix, entries in SERVE.items():
        items = "".join(
            f'<li><code>{esc(rel)}</code>'
            + (f' → <code>{esc(url_for(prefix, rel))}</code>' if url_for(prefix, rel) in served
               else ' <span class="dim">（后端依赖，不挂 URL）</span>')
            + ("" if (ROOT / rel).is_file() else ' <span class="pill todo">文件缺失</span>')
            + "</li>" for rel, _ in entries)
        cols.append(f"<div><h3>{esc(prefix)}</h3><ul>{items}</ul></div>")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>讲义总览 · MiniGPT</title>
<link rel="stylesheet" href="assets/manual.css">
</head>
<body class="chapter-page" data-stage="index">
<article class="chapter" id="ch-index">
<header class="hero">
  <div class="crumb"><a href="/">← 测评台</a> · 讲义总览</div>
  <h1>讲义总览：代码在哪儿、还要写什么</h1>
  <div class="meta"><span>本页由 <code>python scripts/md_to_html.py</code> 生成</span>
    <span>进度取自 webapp/state.json，与测评台同源</span></div>
</header>
<main>
<section class="sec" id="sec-progress"><h2>1. 各阶段进度</h2>
<table class="todo-table prog"><thead><tr><th>阶段</th><th>TODO 已写</th><th>单项检查</th><th>整卷判据</th>
<th>验收命令</th><th>预计</th></tr></thead><tbody>{"".join(prog)}</tbody></table>
<p class="note">「已写」只看骨架里那句 <code>raise NotImplementedError</code> 还在不在，
<b>它不判对错</b> —— 对错只有整卷判据说。边读边测就点阶段名进测评台，中栏切「读讲义」。</p></section>

<section class="sec" id="sec-tasks"><h2>2. 每一处要写的代码，具体在哪个文件哪一行</h2>
<p class="note">行号是生成时的，改过代码就会漂。可靠定位有两条：在文件里搜编号
（<code>grep -n "TODO-3" src/minigpt/*.py</code>），或点「去写 →」让测评台把光标放到那一行。</p>
{"".join(tables)}</section>

<section class="sec" id="sec-assets"><h2>3. 静态资源登记表</h2>
<p class="note">测评台和讲义互相引用，清单只写在 <code>webapp/manual_assets.py</code> 一处。
加一个 asset 就去那儿加一行，然后跑 <code>python webapp/manual_assets.py</code> 自检
（缺文件、孤儿文件都会报，返回码非 0）。下表就是那张登记表。</p>
<div class="cols">{"".join(cols)}</div></section>

<section class="sec" id="sec-write"><h2>4. 想改讲义本身</h2>
<p>正文的写作源是 <code>docs/stage-0N-*.md</code>（公式直接写 <code>$…$</code>，
参考答案放 <code>&lt;details&gt;</code> 折叠块）。这一页和四章 HTML 都是生成的：
改完先跑 <code>python scripts/md_to_html.py --check</code> 比对可见文本，
<b>0 丢失</b>才继续生成 —— 否则「转成 HTML 了」其实等于「讲义少了几段」。</p></section>
</main>
</article>
<script src="assets/manual.js"></script>
</body></html>
"""


# ------------------------------------------------------------------ 文本预算
PROSE_LIMIT = 6          # 一段连续纯正文最多几行 md 源行
# 九份讲义都已清零，所以全部入闸：再写超标文本直接构建失败。
# 确实需要写长段落时用 `<!-- prose-ok 理由 -->` 豁免紧随其后的那一段。
MIGRATED = {s["id"] for s in STAGES}

PROSE_OK = re.compile(r"^\s*<!--\s*prose-ok\b", re.I)
MD_LIST = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
MD_HTML = re.compile(r"^\s*</?(details|summary|div|section|p)\b", re.I)


def classify(md: str):
    """逐行贴标签：(行号, 类别, 原文)。类别里只有 prose 算"话"。

    fence/table/list/math 不算话 —— 它们是同一堆信息的结构形态。
    math 单独摘出来：`$$\\begin{aligned}` 那种推导块行数不少，但一行汉字都没有。
    """
    fence = math = False
    for n, raw in enumerate(md.split("\n"), 1):
        s = raw.strip()
        if re.match(r"^\s*```", raw):
            fence = not fence
            yield n, "fence", s
            continue
        if fence:
            yield n, "fence", s
            continue
        if math:
            if s.endswith("$$"):
                math = False
            yield n, "math", s
            continue
        if PROSE_OK.match(raw):
            yield n, "ok", s
            continue
        if not s:
            yield n, "blank", s
            continue
        if s.startswith("$$"):
            one_line = s.endswith("$$") and len(s) > 4
            yield n, "math", s
            math = not one_line
            continue
        if re.match(r"^#{1,6}\s", raw):
            yield n, "head", s
        elif re.match(r"^\s*-{3,}\s*$", raw):
            yield n, "rule", s
        elif FIG_INLINE.match(raw):
            yield n, "fig", s
        elif s.startswith("|"):
            yield n, "table", s
        elif MD_LIST.match(raw):
            yield n, "list", s
        elif MD_HTML.match(raw):
            yield n, "html", s
        else:
            yield n, "prose", s


def prose_audit(md: str) -> list[dict]:
    """找出连续 prose 超过 PROSE_LIMIT 行的段落，返回 [{line, span, head}]。

    为什么要有这个闸门（不是审美，是激励）：写一段话的成本是一行 md，做一个图的成本是
    几十行 JS + 挂 script 清单 + 登记表 + 无头浏览器验证。没有约束时我一定滑向文本 ——
    九份讲义 37% 是正文，就是这么攒出来的。豁免用 `<!-- prose-ok 理由 -->`：确实只能是话的
    东西（比如"为什么这条判据存在"的一段推理）允许写，但必须当场交代为什么压不成结构。
    """
    out, run, exempt = [], [], False

    def record():
        if len(run) > PROSE_LIMIT and not run[0][2]:
            out.append(dict(line=run[0][0], span=len(run), head=run[0][1][:70]))

    for n, kind, s in classify(md):
        if kind == "ok":
            record(); run, exempt = [], True
        elif kind == "prose":
            if not run:
                head_exempt, exempt = exempt, False     # 段首决定这一段豁不豁免
            run.append((n, s, head_exempt))
        elif kind == "blank":
            record(); run = []                           # 断段，但没用掉的豁免留着
        else:
            record(); run, exempt = [], False            # 结构行断段也断豁免
    record()
    return out


def prose_share(md: str) -> tuple[int, int]:
    """(话行, 非话行)：只用于报数，让人看见每章还剩多少话没压成结构。"""
    c: dict[str, int] = {}
    for _, kind, _ in classify(md):
        c[kind] = c.get(kind, 0) + 1
    return c.get("prose", 0), sum(v for k, v in c.items() if k != "blank")


SECTION_TALK_MIN = 12      # 一节里"话"超过这么多行，就必须出现一个非文本通道
TALK = {"prose", "list"}
STRUCT = {"table", "fence", "fig", "math"}


def structure_audit(md: str) -> list[dict]:
    """整节全是话（没有表 / 图 / 代码 / 公式 / 卡）→ 报出来。

    段落预算抓不到这一类：§7「判据逐条解读」是编号列表，每条两行，没有一段超标，
    整节 40 行却全是文本。这条抓的是"这一节有没有把信息压成结构"，也就是根源本身。
    """
    out, cur, talk, struct = [], "（章首）", 0, 0
    for n, kind, s in classify(md):
        if kind == "head" and s.startswith("## "):
            if talk >= SECTION_TALK_MIN and not struct:
                out.append(dict(title=cur, talk=talk, line=n))
            cur, talk, struct = s[3:], 0, 0
        elif kind in TALK:
            talk += 1
        elif kind in STRUCT:
            struct += 1
    if talk >= SECTION_TALK_MIN and not struct:
        out.append(dict(title=cur, talk=talk, line=0))
    return out


# ------------------------------------------------------------------ 保真校验
def visible_from_md(md: str) -> list[str]:
    keep, fence = [], False
    for ln in md.split("\n"):
        if re.match(r"^\s*```", ln):
            fence = not fence; continue
        if re.match(r"^\s*-{3,}\s*$", ln):
            continue                       # 分节线是排版，不是内容：<section> 已经承担分隔
        if FIG_INLINE.match(ln):
            continue                       # @@fig 挂载点同上：它换成的是空壳 div，没有文字
        s = re.sub(r"<[^>]+>", "", ln)
        s = re.sub(r"[#*`>|]", "", s)
        s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
        s = re.sub(r"^\s*[-*]\s+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s and re.fullmatch(r"[-: ]*", s):
            continue                       # 表格分隔行 |---|---| 剥完只剩横线，也不是内容
        if s:
            keep.append(s)
    return keep


def visible_from_html(h: str) -> set:
    h = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", h, flags=re.S | re.I)
    h = re.sub(r"<[^>]+>", "\n", h)
    return {re.sub(r"\s+", " ", p).strip() for p in html.unescape(h).split("\n") if p.strip()}


def check(pairs) -> int:
    lost = 0
    for name, md, ht in pairs:
        got, want = visible_from_html(ht), [x for x in dict.fromkeys(visible_from_md(md))]
        miss = [x for x in want if not any(x in g or g in x for g in got)]
        print(f"{name}: md 去重可见行 {len(want)} / 丢失 {len(miss)}")
        for m in miss[:15]:
            print("   丢:", m[:120])
        lost += len(miss)
    return lost


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = []
    for st in STAGES:
        # 源 md 按阶段号找（stage-03-*.md），不再从登记表取 —— 因为登记表的 manual
        # 现在指向**生成出来的 html**（网站和用户读的都是它），md 退成我的写作源文件。
        cand = sorted(DOCS.glob(f"stage-{st['id'][1:]}-*.md"))
        src = cand[0] if cand else None
        if not st["available"]:
            continue                                   # 未发布：没有写作源是常态，不用喊
        if src is None:
            print(f"跳过 {st['id']}：已发布但找不到 docs/stage-{st['id'][1:]}-*.md")
            continue
        md = src.read_text(encoding="utf-8")
        out = OUT / f'{st["id"]}.html'
        out.write_text(page(st, md), encoding="utf-8")
        pairs.append((out.name, md, out.read_text(encoding="utf-8")))
        print("生成", out.relative_to(ROOT))
    (OUT / "index.html").write_text(overview(STAGES), encoding="utf-8")
    print("生成", (OUT / "index.html").relative_to(ROOT))
    bad = prose_report(pairs)
    if "--check" in sys.argv:
        print("\n--- 保真校验 ---")
        n = check(pairs)
        print("丢失行数:", n)
        return 1 if (n or bad) else 0
    return 1 if bad else 0


def prose_report(pairs) -> int:
    """文本预算：每章报「超标段落 / 正文占比」，MIGRATED 里的阶段超标就返回非零。

    没迁移的章只报数不拦（一次性清九份会让网站长时间半改），但数字必须摆在眼前 ——
    看不见欠账是上一轮文本泛滥的一半原因。
    """
    print("\n--- 文本预算（一段连续正文 >%d 行算超标；一节话满 %d 行还没有结构也算超标）---"
          % (PROSE_LIMIT, SECTION_TALK_MIN))
    bad = 0
    for name, md, _ in pairs:
        sid = name.split(".")[0]
        hits = prose_audit(md) + [dict(line=x["line"], span=x["talk"], head="整节全是话：" + x["title"])
                                  for x in structure_audit(md)]
        prose, total = prose_share(md)
        gate = "拦" if sid in MIGRATED else "只报"
        print(f'{sid}: 话 {prose}/{total} = {prose * 100 // max(total, 1)}%  '
              f'超标 {len(hits)} 处  [{gate}]')
        for h in sorted(hits, key=lambda x: x["line"]):
            print(f'   L{h["line"]:4d} 连续 {h["span"]:2d} 行  {h["head"]}')
        if sid in MIGRATED:
            bad += len(hits)
    if bad:
        print(f"\n构建失败：已清零的阶段里又写出 {bad} 处超标文本。"
              f"压成表 / 例子卡 / 图，或写 `<!-- prose-ok 理由 -->` 说明为什么这段必须是话。")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
