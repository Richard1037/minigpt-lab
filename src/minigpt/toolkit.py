"""每个阶段脚本共用的验收工具：判据打印、ASCII 画图、计时。

设计目标：你跑任何一个 stage，最后都会看到一张"哪条过了、哪条没过"的清单。
过不了就别进下一阶段 —— 这是整个学习路线的护栏。
零第三方依赖，图用 ASCII 画，不需要 matplotlib。
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager

# 网站跑判据时会带上 MGPT_JSON=1：除了给人看的文本，再吐一行机器可读的结果。
# 命令行不带这个变量，输出和以前一字不差 —— 结构化只服务渲染，不改变判据本身。
JSON_MARK = "@@MGPT-JSON@@"
_FIT_MAX = 240        # 拟合图像最多带这么多点：100 个点抽到 240 以内，payload 才不会拖垮一次请求

# ---------------------------------------------------------------- 控制台宽度
_WIDE = (
    (0x1100, 0x115F), (0x2E80, 0x303F), (0x3040, 0x33FF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xA000, 0xA4CF), (0xAC00, 0xD7A3), (0xF900, 0xFAFF),
    (0xFE30, 0xFE4F), (0xFF00, 0xFF60), (0xFFE0, 0xFFE6), (0x1F300, 0x1FAFF),
)


def dwidth(text: str) -> int:
    """按终端显示宽度计算字符串长度（中文算 2，ASCII 算 1）。"""
    w = 0
    for ch in text:
        cp = ord(ch)
        w += 2 if any(lo <= cp <= hi for lo, hi in _WIDE) else 1
    return w


def pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格，让表格在中文环境下仍然对齐。"""
    return text + " " * max(0, width - dwidth(text))


def _init_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------- 验收报告
class Report:
    """收集判据并打印成一张清单。硬性判据全过，done() 才返回 True。"""

    MUST, SOFT, INFO = "must", "soft", "info"
    created: list["Report"] = []        # 供"中途停在 TODO"时把已跑完的判据吐出来

    def __init__(self, title: str, subtitle: str = ""):
        _init_stdout()
        self.title, self.subtitle = title, subtitle
        self.rows: list[tuple[str, str, str, str, str]] = []
        self.curves: list[dict] = []
        self.fits: list[dict] = []
        self._group = ""
        self._t0 = time.perf_counter()
        Report.created.append(self)

    # ---- 记录 ----
    def group(self, name: str) -> None:
        """给后面的判据挂一个分组名（"Embedding"、"CrossEntropy"…）。

        只是**展示用**的标签：判据的条数、阈值、结论一个字都不变，
        命令行打印时多一行分组标题，网站/结果窗口按它分卡片。
        """
        self._group = name

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        """硬性判据：失败即整体验收失败。"""
        self.rows.append((self.MUST, "PASS" if ok else "FAIL", name, detail, self._group))
        return bool(ok)

    def soft(self, name: str, ok: bool, detail: str = "") -> bool:
        """软性提示：不通过不影响验收，但值得看一眼（例如过拟合预警）。"""
        self.rows.append((self.SOFT, "注意" if ok else "提示", name, detail, self._group))
        return bool(ok)

    def info(self, name: str, value: object = "") -> None:
        self.rows.append((self.INFO, "info", name, str(value), self._group))

    def curve(self, name: str, values, note: str = "") -> None:
        """登记一条曲线（通常是每轮 loss），供网站用 SVG 画。

        命令行仍然由脚本自己 print(sparkline(...)) 画 ASCII —— 两边数据同源，
        不会出现"网页上的图和终端上的图不是一条曲线"。
        """
        clean = [float(v) for v in values if v == v]        # 去 NaN
        if len(clean) >= 2:
            self.curves.append(dict(name=name, note=note, values=clean))

    def fit(self, name: str, xs, ys, pred, truth=None, cx=None, note: str = "") -> None:
        """登记一张"拟合图像"：数据点 (xs, ys) + 你的曲线 pred（画在 cx 上）。

        和 curve 一样的原则 —— **数据由判据脚本自己算**，网站只负责画，
        所以网页上那条曲线就是你代码跑出来的东西，不是前端另算一遍。

        cx 是画曲线用的密集网格（默认就用 xs）。阶段 2 只有 12 个数据点，
        拿 12 个点连线会画成折线，看着像模型不行 —— 所以散点和曲线分开给。
        点数超过 _FIT_MAX 等间隔抽样，否则一次测评的 payload 会到几百 KB。
        """
        def sample(seq, n):
            if n < 2:
                return list(range(n))
            step = max(1, -(-n // _FIT_MAX))                # 向上取整
            return list(range(0, n, step))
        grid = xs if cx is None else cx
        ix, ic = sample(xs, len(xs)), sample(grid, len(grid))
        curve = lambda seq: [round(float(seq[i]), 6) for i in ic if seq[i] == seq[i]]
        dots = [(round(float(xs[i]), 6), round(float(ys[i]), 6))
                for i in ix if xs[i] == xs[i] and ys[i] == ys[i]]   # 散点成对过滤，别让 x/y 长度错开
        self.fits.append(dict(
            name=name, note=note,
            x=[p[0] for p in dots], y=[p[1] for p in dots],
            cx=[round(float(grid[i]), 6) for i in ic],
            pred=curve(pred), truth=curve(truth) if truth is not None else None))

    # ---- 判据小工具 ----
    def close(self, name: str, got: float, want: float, tol: float, unit: str = "", note: str = "") -> bool:
        return self.check(name, abs(got - want) <= tol,
                          f"期望 {want:g}{unit} 实测 {got:.4g}{unit} 误差 {abs(got - want):.3g} ≤ {tol:g}"
                          + (f" · {note}" if note else ""))

    def below(self, name: str, got: float, limit: float, note: str = "") -> bool:
        return self.check(name, got <= limit, f"实测 {got:.4g} ≤ 上限 {limit:g}" + (f" · {note}" if note else ""))

    def above(self, name: str, got: float, limit: float, note: str = "") -> bool:
        return self.check(name, got >= limit, f"实测 {got:.4g} ≥ 下限 {limit:g}" + (f" · {note}" if note else ""))

    # ---- 打印 ----
    # ---- 机器可读通道（只有 MGPT_JSON=1 时打开，给测评网站渲染用） ----
    def _emit_json(self, n_ok: int, n_must: int, elapsed: float, partial: bool) -> None:
        if os.environ.get("MGPT_JSON") != "1":
            return
        payload = dict(title=self.title, subtitle=self.subtitle, partial=partial,
                       elapsed=round(elapsed, 3), got=n_ok, total=n_must,
                       ok=(n_ok == n_must),
                       rows=[dict(kind=k, mark=m, name=n, detail=d, group=g)
                             for k, m, n, d, g in self.rows],
                       curves=self.curves, fits=self.fits)
        # 一行、不换行、前面带标记：服务端按标记切出来，剩下的照旧当文本显示
        print("\n" + JSON_MARK + json.dumps(payload, ensure_ascii=False) + "\n")

    def _print_rows(self) -> list[str]:
        """打印所有判据行，返回失败的硬判据名字。

        两处（done / print_partial）共用一份，分组标题的逻辑只写一遍。
        组名变化时打一行加粗标题 —— 命令行里"11 条平铺"变成"三组各几条"，
        判据本身没有任何变化。
        """
        fails, last_group = [], None
        for kind, mark, name, detail, group in self.rows:
            if group != last_group:
                if group:
                    print(f"\n\x1b[1m▸ {group}\x1b[0m")
                last_group = group
            if kind == self.INFO:
                print(f"  ·  {pad(name, 26)} {detail}")
                continue
            tag = {"PASS": "\x1b[32m[OK]\x1b[0m", "FAIL": "\x1b[31m[NO]\x1b[0m",
                   "注意": "\x1b[33m[!!]\x1b[0m", "提示": "\x1b[36m[..]\x1b[0m"}[mark]
            print(f"  {tag} {pad(name, 26)} {detail}".rstrip())
            if kind == self.MUST and mark == "FAIL":
                fails.append(name)
        return fails

    def print_partial(self) -> None:
        """脚本中途停在某个 TODO 时，把已经跑完的判据先吐出来。

        没有这个方法的话，你写完 Linear 去跑 test_layers，脚本会在 GELU 那里抛
        NotImplementedError，于是连 Linear 那条的结果都看不到 —— 而"每写完一个 TODO
        就测一次"恰恰是最该有的节奏。
        """
        bar = "─" * max(28, dwidth(self.title) + 4)
        print(f"\n\x1b[1m{self.title}\x1b[0m（中途停在 TODO，以下已跑完）\n{bar}")
        self._print_rows()
        must_rows = [r for r in self.rows if r[0] == self.MUST]
        n_ok = sum(1 for r in must_rows if r[1] == "PASS")
        print(f"{bar}\n已跑完 {len(must_rows)} 条，其中 {n_ok} 条达标。")
        self._emit_json(n_ok, len(must_rows), time.perf_counter() - self._t0, partial=True)

    def done(self, must_print: bool = True) -> bool:
        elapsed = time.perf_counter() - self._t0
        bar = "─" * max(28, dwidth(self.title) + 4)
        print(f"\n\x1b[1m{self.title}\x1b[0m\n{bar}")
        if self.subtitle:
            print(self.subtitle)
        fails = self._print_rows()
        must_rows = [r for r in self.rows if r[0] == self.MUST]
        n_ok = sum(1 for r in must_rows if r[1] == "PASS")
        print(bar)
        if fails and must_print:
            print(f"\x1b[31m验收未通过\x1b[0m：{n_ok}/{len(must_rows)} 条硬判据达标，用时 {elapsed:.1f}s")
            print("卡住的项：")
            for f in fails:
                print(f"  - {f}")
            print("\n先修好它再进下一阶段。别带着这个 bug 往下写，它会伪装成别的问题。")
        else:
            print(f"\x1b[32m验收通过\x1b[0m：{n_ok}/{len(must_rows)} 条硬判据达标，用时 {elapsed:.1f}s")
            print("可以进下一阶段了。")
        self._emit_json(n_ok, len(must_rows), elapsed, partial=False)
        return not fails


# ---------------------------------------------------------------- ASCII 图
_RAMP = " .:-=+*#%@"


def sparkline(values, width: int = 56, height: int = 9, label: str = "loss") -> str:
    """把一维序列画成 ASCII 曲线（值越小越靠下）。"""
    vals = [float(v) for v in values if v == v]  # 去 NaN
    if len(vals) < 2:
        return f"{label}: 数据点不足，无法画图"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = max(1, len(vals) // width)
    sampled = vals[::step]
    grid = [[" "] * len(sampled) for _ in range(height)]
    for col, v in enumerate(sampled):
        row = height - 1 - int(round((v - lo) / span * (height - 1)))
        grid[row][col] = "*"
    head = f"{label}: 最高 {hi:.4g} → 最低 {lo:.4g}   ({len(vals)} 个点)"
    body = "\n".join("|" + "".join(r) for r in grid)
    return f"{head}\n{body}\n`-{'-' * len(sampled)}"


def scatter(xs, ys_true, ys_pred, height: int = 15, label: str = "拟合") -> str:
    """把 y_true 画成 '.'，y_pred 画成 'o'，叠在同一张 ASCII 图上。"""
    xs = [float(v) for v in xs]
    lo_y, hi_y = min(min(ys_true), min(ys_pred)), max(max(ys_true), max(ys_pred))
    lo_x, hi_x = min(xs), max(xs)
    span_y = (hi_y - lo_y) or 1.0
    span_x = (hi_x - lo_x) or 1.0
    w = 72
    grid = [[" "] * w for _ in range(height)]

    def put(px, py, ch):
        col = min(w - 1, max(0, int((px - lo_x) / span_x * (w - 1))))
        row = min(height - 1, max(0, height - 1 - int((py - lo_y) / span_y * (height - 1))))
        if grid[row][col] == " ":
            grid[row][col] = ch

    for x, y in zip(xs, ys_true):
        put(x, y, ".")
    for x, y in zip(xs, ys_pred):
        put(x, y, "o")
    out = [f"{label}: y ∈ [{lo_y:.3g}, {hi_y:.3g}]   · =目标  o =模型"]
    for i, row in enumerate(grid):
        mark = f"{lo_y + span_y * (height - 1 - i) / (height - 1):6.2f} |"
        out.append(mark + "".join(row))
    out.append("       " + "-" * w)
    out.append(f"       x ∈ [{lo_x:.3g}, {hi_x:.3g}]")
    return "\n".join(out)


def attention_row(tokens: list[str], weights, top: int = 4) -> str:
    """把一行注意力权重渲染成 '词(权重)' 的降序列表，用来肉眼检查它在看哪儿。"""
    order = sorted(range(len(weights)), key=lambda i: -weights[i])[:top]
    return " → ".join(f"{tokens[i]}({weights[i]:.2f})" for i in order)


@contextmanager
def timer(name: str = "耗时", report: Report | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if report is not None:
        report.info(name, f"{dt:.2f}s")
    else:
        print(f"  ·  {pad(name, 26)} {dt:.2f}s")


def print_progress(done: int, total: int, every: int = 200, prefix: str = "训练") -> None:
    if done % every == 0 or done == total:
        pct = done / total * 100
        filled = int(pct / 100 * 30)
        print(f"\r  {prefix} [{'#' * filled}{'.' * (30 - filled)}] {done}/{total} ({pct:4.1f}%)",
              end="", flush=True)
        if done == total:
            print()
