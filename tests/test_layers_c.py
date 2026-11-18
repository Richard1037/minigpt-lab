r"""s03c 的判据 · CrossEntropy 与"测谎仪本身可信吗"。跑法：

    python tests/test_layers_c.py

两件事：

  1. **CrossEntropy**（TODO-11）—— 本阶段唯一输出标量的层，也是最容易写错的：
     减不减最大值、分母除以 T 还是除以 B·T、梯度为什么是 `(p - onehot)/N`。
     `loss = ln(V)` 那条基准线以后每个阶段都要用（随机初始化时模型什么都不会，
     它的 loss 就恰好是 $\ln V$，掉不下来说明没在学）。
  2. **gradcheck 自检** —— 你后面会经常拿到一句"最大相对误差 3e-2"，那时候你需要
     知道该怀疑自己还是怀疑工具。这里用两个已知答案的层钉住工具的可靠性：
     一个反向写对的，它必须说 OK；一个故意漏掉系数 2 的，它必须报错。

前提：s03a、s03b 已经绿了。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.gradcheck import gradcheck
from minigpt.nn import CrossEntropy, Module
from minigpt.toolkit import Report

rng = np.random.default_rng(13)


class _SquareOK(Module):
    r"""y = x²，反向按定义写对：$\partial y/\partial x = 2x$。gradcheck 该说 OK。"""

    def forward(self, x):
        self._x = np.asarray(x, dtype=np.float64)
        return self._x ** 2

    def backward(self, g):
        return g * (2.0 * self._x)


class _SquareWrong(Module):
    """故意把 2x 写成 x —— gradcheck 必须抓住，否则它就是个假绿灯。"""

    def forward(self, x):
        self._x = np.asarray(x, dtype=np.float64)
        return self._x ** 2

    def backward(self, g):
        return g * self._x


def main() -> int:
    rep = Report("s03c · CrossEntropy 与 gradcheck 自检", "有限差分步长 1e-6，相对误差阈值 1e-4")

    rep.group("CrossEntropy · 数值稳定性与 ln(V) 基准线")
    B, T, V = 2, 5, 9
    logits = rng.standard_normal((B, T, V)) * 2.0
    targets = rng.integers(0, V, size=(B, T))
    ce = CrossEntropy()
    ok, worst, detail = gradcheck(ce, [logits, targets], label="CrossEntropy")
    rep.check("CrossEntropy", ok, f"最大相对误差 {worst:.2e}"
              + " · 输出是标量，所以余因子是 1，有限差分照样能查")
    if not ok:
        print(detail)

    huge = CrossEntropy()
    loss = huge(np.full((1, 3, 8), 900.0), np.array([[1, 2, 3]]))
    rep.check("logits=900 不溢出", np.isfinite(loss),
              f"loss={loss:.4f}（每行 logits 全相等 → 恰为 ln(8)=2.079，且没有 inf）")

    uniform = CrossEntropy()(np.zeros((1, 4, 8)), np.array([[1, 2, 3, 0]]))
    rep.close("均匀 logits 的 loss = ln(V)", uniform, np.log(8), 1e-12,
              note="随机初始化时 loss 就该在这个值附近，后面阶段用它当基准线")

    # ---- gradcheck 自己可信吗：一个已知对、一个已知错
    rep.group("gradcheck 自检 · 怀疑工具之前先钉住工具")
    x = rng.standard_normal((2, 3, 4))
    ok_right, worst_r, _ = gradcheck(_SquareOK(), [x], label="SquareOK")
    rep.check("反向写对的层，gradcheck 说 OK", ok_right,
              f"最大相对误差 {worst_r:.2e}（阈值 1e-4）· 2x 这个导数它认得")
    ok_wrong, worst_w, _ = gradcheck(_SquareWrong(), [x], label="SquareWrong")
    rep.check("故意漏掉系数 2，gradcheck 报错", not ok_wrong,
              f"最大相对误差 {worst_w:.2e} · 它抓得住，说明你以后拿到的红是可信的红")

    passed = rep.done()
    print("\n阶段 3 到此结束。这三卷以后会一直为你服务：")
    print("  · 任何新层写完，第一件事是加进对应那卷跑一遍，而不是拿去训模型。")
    print("  · loss=ln(V) 是'模型还没开始学'的基准线，阶段 5/6 的训练曲线从它起步。")
    print("  · 觉得 gradcheck 有 bug 时，先跑这一卷的最后两条 —— 它们就是那个用途。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
