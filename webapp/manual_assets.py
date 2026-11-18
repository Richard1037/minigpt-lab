"""静态资源登记表 —— 谁依赖谁，一处写清，两处消费。

为什么要有这个文件：讲义和测评台互相引用（测评台 fetch 讲义、讲义回链测评台），
而它们的文件分散在 webapp/ 和 docs/manual/ 两个目录。加一个 asset、或者把
s03.html 改名叫 chapter-3.html，很容易只改一边 —— 于是页面看起来正常，
点某个按钮才发现 404。登记表把"谁用什么"变成可校验的数据：

  · webapp/server.py 读 SERVE：只把登记过的文件挂成 URL，没登记的一律 404；
  · scripts/md_to_html.py 把它嵌进讲义总览页，浏览器里能直接看到引用清单；
  · 直接跑本文件（python webapp/manual_assets.py）就是自检：缺文件、
    有孤儿（磁盘上有但登记表没写）都会报出来并返回非 0。

SERVE 的键是 URL 前缀，值是 (相对仓库根的路径, 类型)。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 测评台首页：index.html 里那一个 <style> 直接内联，没有独立 app.css
APP = (
    ("webapp/index.html", "html"),
    ("webapp/result.html", "html"),      # 「↗ 结果窗口」：判据结论 + 拟合图像 + 原始输出
    ("webapp/server.py", "py"),          # 后端：路由 + 阶段登记表 + /api/check
    ("webapp/checks.py", "py"),          # 单项判据，网站和命令行共用同一份
    ("webapp/lab.py", "py"),             # 调参试验台：旋钮定义 + 各阶段实跑
    ("webapp/manual_assets.py", "py"),   # 就是本文件：登记表自己也要被登记
)

# 讲义：总览 + 各章正文 + 四个 asset。md 是写作源，html 是产物（scripts/md_to_html.py 生成）
# 发布新章节时在 STAGE_IDS 里加一号（"03a" 这种带字母的子阶段也走这里）——
# 没登记的章节 /manual/sNN.html 一律 404，自检也会报"登记表里有、磁盘上没有"，
# 所以加号和写讲义要一起做。
STAGE_IDS = ("01", "02", "03a", "03b", "03c", "04", "05", "06", "07")
MANUAL = (("docs/manual/index.html", "html"),) \
    + tuple((f"docs/manual/s{i}.html", "html") for i in STAGE_IDS) + (
    ("docs/manual/assets/manual.css", "css"),
    ("docs/manual/assets/manual.js", "js"),
    ("docs/manual/assets/figs.js", "js"),        # 第一批可视化组件（自带 REG 表）
    ("docs/manual/assets/figs2.js", "js"),       # 第二批的底座：工具函数 + 可插拔注册表
    ("docs/manual/assets/figs_a.js", "js"),      # 注意力与整机装配的原理动画
    ("docs/manual/assets/figs_b.js", "js"),      # 训练循环、优化器、检查点的原理动画
    ("docs/manual/assets/figs_c.js", "js"),      # 数值与梯度校验的原理动画
    ("docs/manual/assets/figs_d.js", "js"),      # 3D 可视化：形状盒 / loss 曲面
    ("docs/manual/assets/figs_e.js", "js"),      # 反向传播三处硬骨头：矩阵乘 / softmax / CE
    ("docs/manual/assets/figs_f.js", "js"),      # 训练循环的三件事：切批 / 裁范数 / 抽样
    ("docs/manual/assets/figs_g.js", "js"),      # 非线性与逐层动线：饱和 / GELU / MLP 反向
    ("docs/manual/assets/mj-config.js", "js"),   # MathJax 配置
)

SERVE: dict[str, tuple[tuple[str, str], ...]] = {"app": APP, "manual": MANUAL}

# URL 前缀 → 这个前缀下的文件放在磁盘哪儿；反查"给 URL 找文件"用
PREFIX_ROOT = {"app": ROOT / "webapp", "manual": ROOT / "docs" / "manual"}


def url_for(prefix: str, rel: str) -> str:
    """磁盘相对路径 → 站点 URL。

    讲义内部引用 assets 用相对路径（同目录的 assets/manual.css），
    测评台用绝对路径（/manual/assets/manual.css），所以这里只规范 manual 前缀。
    """
    if prefix != "manual":
        return "/" + rel[len("webapp/"):] if rel.startswith("webapp/") else "/" + rel
    return "/manual/" + rel[len("docs/manual/"):]


# 只有这三类走 HTTP。登记表里还有 .py（server.py / checks.py）—— 那是"这个页面依赖哪个
# 后端文件"的说明，不是可以下载的 URL：把源码挂出去只会让 /manual/ 变成任意文件读取。
WEB_KINDS = {"html", "css", "js"}


def files(prefix: str) -> list[str]:
    return [f for f, _ in SERVE[prefix]]


def by_url() -> dict[str, pathlib.Path]:
    """URL → 磁盘路径。服务端只按这张表放行，表外的一律 404。"""
    return {url_for(p, f): ROOT / f
            for p in ("app", "manual") for f, kind in SERVE[p] if kind in WEB_KINDS}


def check() -> list[str]:
    """返回问题列表；空列表 = 登记表和磁盘对得上。"""
    problems = []
    for prefix, entries in SERVE.items():
        for rel, _kind in entries:
            if not (ROOT / rel).is_file():
                problems.append(f"登记表里有、磁盘上没有：{rel}")
    on_disk = {str(p.relative_to(ROOT)).replace("\\", "/")
               for base in (ROOT / "webapp", ROOT / "docs" / "manual" / "assets")
               for p in base.glob("*") if p.is_file()}
    listed = {f for _, e in SERVE.items() for f, _ in e}
    for orphan in sorted(on_disk - listed - {"webapp/state.json", "webapp/workspaces"}):
        if "__pycache__" in orphan or not Path_suffix_matters(orphan):
            continue
        problems.append(f"磁盘上有、登记表里没写（孤儿文件）：{orphan}")
    return problems


def Path_suffix_matters(rel: str) -> bool:
    """只有会被浏览器/工具加载的后缀才算孤儿（.png 截图之类的一次性产物不算）。"""
    return pathlib.Path(rel).suffix.lower() in {".html", ".css", ".js", ".py"}


def main() -> int:
    problems = check()
    for prefix, entries in SERVE.items():
        n_web = sum(1 for _, k in entries if k in WEB_KINDS)
        print(f"[{prefix}] {len(entries)} 项，其中 {n_web} 项有 URL")
        for rel, kind in entries:
            mark = "ok " if (ROOT / rel).is_file() else "缺失"
            where = url_for(prefix, rel) if kind in WEB_KINDS else "（依赖记录，不挂 URL）"
            print(f"  {mark} {rel:<38} → {where}")
    print("\n自检：", "登记表和磁盘对得上" if not problems else f"{len(problems)} 个问题")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
