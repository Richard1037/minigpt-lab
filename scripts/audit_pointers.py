"""锚点审计：骨架注释 / NotImplementedError 文案里写的 `docs/manual/sNN.html#sec-K`，
和生成物里"这个 TODO 到底落在哪一节"对不对得上。

README §6 说"别手工数 §N，用生成物反查" —— 这个脚本就是那句反查本身：
`<h2>` 里含 `TODO-N` 的那一节就是它的家。三种裂缝它都能抓出来：

  · 登记表列了某个 TODO，但七章里没有任何 `<h2>` 提到它  → 小地图和右栏"讲义"按钮找不到家；
  · 阶段标了 available=True，但 HTML 还没生成           → 网站上点进去 404；
  · 骨架注释指向 #sec-4，生成物说它在 #sec-6            → 用户按提示跳过去，看到的是别的东西。

用法：先 `python scripts/md_to_html.py` 生成，再 `python scripts/audit_pointers.py`。
返回码非 0 = 有对不上的地方。
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from server import STAGES, expand_todo  # noqa: E402

H2 = re.compile(r"<h2>(.*?)</h2>", re.S)
SEC = re.compile(r'<section class="sec" id="(sec-[0-9.\-]+)"')
PTR = re.compile(r"(TODO-\d+[A-Za-z]?)[:：]?[^\n]{0,160}?docs/manual/(s\d\d)\.html#(sec-[0-9.\-]+)")
TAG = re.compile(r"<[^>]+>")


def home_of(html: str) -> dict[str, str]:
    """TODO 编号 → 讲它的那一节（按 <h2> 与 section 的出现顺序配对）。"""
    ids = SEC.findall(html)
    h2s = [TAG.sub("", m).replace("&amp;", "&").strip() for m in H2.findall(html)]
    home: dict[str, str] = {}
    for i, sec in enumerate(ids):
        for tid in expand_todo(h2s[i] if i < len(h2s) else ""):
            home.setdefault(tid, sec)
    return home


def main() -> int:
    problems: list[str] = []
    for stage in STAGES:
        if not stage["available"]:
            continue
        path = ROOT / "docs" / "manual" / f"{stage['id']}.html"
        if not path.exists():
            problems.append(f"{stage['id']}：已发布但没有 {path.relative_to(ROOT)}（网站上点进去 404）")
            continue
        home = home_of(path.read_text(encoding="utf-8"))
        orphan = [t for t in stage["todos"] for tid in expand_todo(t) if tid not in home]
        if orphan:
            problems.append(f"{stage['id']}：登记表的 {orphan} 没出现在任何 <h2> 里（小地图找不到家）")
        for rel in stage["files"]:
            src = ROOT / "src" / rel
            if not src.exists():
                continue
            for m in PTR.finditer(src.read_text(encoding="utf-8")):
                tid, ref, sec = m.groups()
                want = home.get(tid)
                if want and (ref != stage["id"] or sec != want):
                    problems.append(f"{rel}：{tid} 的注释指 {ref}#{sec}，生成物说它在 #{want}")
    for p in problems:
        print("!!", p)
    print("审计结论：", "全部对得上" if not problems else f"{len(problems)} 处对不上")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
