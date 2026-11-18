"""所有 stage/test 脚本的第一行都 import 它：决定用哪一份 minigpt。

    python stages/s01_fit_line.py            → 用 src/minigpt（你自己写的）
    MGPT_ANSWER=1 python stages/s01_fit_line.py  → 用 solutions/minigpt（完整答案）

为什么用"两个同名包 + 改路径"而不是复制文件：判据（tests/ 和 stages/ 里的验收部分）
必须对两份实现完全一致 —— 同一张考卷，答案只是换个来源。这样你写的时候能确信
"过了判据就是真的对"，也能在我给的答案上先看到通过长什么样。

这个文件不含任何模型逻辑，只有路径，你可以随时读它确认没有魔法。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 三个来源，优先级从高到低：
#   MGPT_PKG=<目录>   → 用该目录下的 minigpt 包（网站的提交工作区走这条）
#   MGPT_ANSWER=1     → solutions/minigpt（完整答案）
#   都没有            → src/minigpt（你自己写的）
_env_pkg = os.environ.get("MGPT_PKG")
if _env_pkg:
    pkg_dir = Path(_env_pkg).resolve()
    USING_SOLUTION = False
else:
    IMPL = "solutions" if os.environ.get("MGPT_ANSWER") == "1" else "src"
    pkg_dir = ROOT / IMPL
    USING_SOLUTION = IMPL == "solutions"

if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def banner(extra: str = "") -> None:
    """脚本开头打印一行，告诉你现在跑的是哪份实现（避免"过了但过的是答案"）。"""
    if _env_pkg:
        tag = f"提交工作区 {_env_pkg}"
    elif USING_SOLUTION:
        tag = "答案实现 solutions/"
    else:
        tag = "你的实现 src/"
    print(f"[实现来源：{tag}]{(' ' + extra) if extra else ''}")


def run(main_fn) -> int:
    """stage/test 的统一出口：把"还没写完"和"真出错"区分开。

    你第一次跑这些脚本时一定是没写完的状态，那不该是一屏 traceback ——
    它应该只告诉你下一步去做哪个 TODO。
    """
    try:
        return int(main_fn() or 0)
    except NotImplementedError as exc:
        # 先让已经跑完的判据露面，再说下一个卡点 —— 否则"写完一个 TODO 就测一次"
        # 会什么都看不到（脚本在下一个未实现的层那里就抛了）。
        try:
            from minigpt.toolkit import Report
            if Report.created:
                Report.created[-1].print_partial()
        except Exception:                                  # noqa: BLE001
            pass
        print("")
        print(f"还没写完 → {exc}")
        print("两种走法：")
        print("  A. 按讲义把那个 TODO 写完（推荐）；")
        print("  B. 先看判据长什么样：MGPT_ANSWER=1 python <这个脚本>   （跑答案实现）")
        return 2
