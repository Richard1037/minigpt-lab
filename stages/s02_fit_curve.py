"""阶段 2 · 拟合一条曲线：搞懂"为什么需要隐藏层和非线性"。

任务：给 12 个点，满足 y = sin(x)，x ∈ [-3, 3]。让程序自己学出这个函数。

这一步会当场演示两件非常重要的事：
  1) 单层线性模型拟合 sin 会撞到一面"降不下去的墙"（loss 卡在 0.5 左右）。
     这不是训练不够久，是模型**没那个表达能力** —— 这就是"模型容量"。
  2) 把两层线性叠起来但中间不加激活函数，墙一模一样高：
     线性∘线性 还是线性。激活函数（tanh/ReLU/GELU）才是让网络变强的东西。

跑法：
    python stages/s02_fit_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.exercises import ScalarMLP, predict, train_mlp   # ← TODO-2A/2B 在那里写
from minigpt.scalar import gradient_check_graph
from minigpt.toolkit import Report, print_progress, scatter, sparkline

N_POINTS = 12
SEED = 1


def make_data():
    xs = np.linspace(-3.0, 3.0, N_POINTS)
    ys = np.sin(xs)
    return xs, ys


def main() -> int:
    rep = Report("阶段 2 · 拟合 sin(x)", f"{N_POINTS} 个点，x∈[-3,3]，目标 y = sin(x)")
    xs, ys = make_data()

    # ---------------- 对照组：纯线性（n_hidden=1 且不加激活，等价于 a·x+b）
    print("\n第 1 步：先用一个没有隐藏层的线性模型试试（预期：撞墙）")
    linear = ScalarMLP(n_hidden=1, activate=False)
    c_linear = train_mlp(linear, xs, ys, iters=2000, lr=0.1, log_every=0, prefix="线性模型")
    mse_linear = c_linear[-1]
    rep.curve("线性模型 loss", c_linear, note="对照：容量不足，降不下去")
    print(f"\n线性模型最终 MSE = {mse_linear:.4f}   （sin 的方差约 0.5，也就是说它几乎没学到形状）")

    # ---------------- 关键对照：两层线性叠加，中间不加激活
    print("\n第 2 步：把两个线性层叠起来，但中间不加激活函数（预期：还是撞同一面墙）")
    deep_linear = ScalarMLP(n_hidden=4, activate=False)
    c_deep = train_mlp(deep_linear, xs, ys, iters=2000, lr=0.1, log_every=0, prefix="深层线性")
    rep.curve("深层线性 loss", c_deep, note="线性∘线性=线性：加深不等于加容量")
    print(f"\n两层线性最终 MSE = {c_deep[-1]:.4f}")

    # ---------------- 真正的主角：带 tanh 的 MLP
    print("\n第 3 步：同样的层数，只在中间加一个 tanh（预期：墙消失了）")
    mlp = ScalarMLP(n_hidden=8, activate=True)
    c_mlp = train_mlp(mlp, xs, ys, iters=6000, lr=0.1, decay=3.0, prefix="MLP+tanh")
    mse_mlp = c_mlp[-1]
    rep.curve("MLP+tanh loss", c_mlp, note="唯一真正学会的一条")
    print()
    print(scatter(xs, ys, predict(mlp, xs), label="sin(x) 拟合"))
    print()
    print(sparkline(c_mlp, label="loss(MLP)"))
    # 拟合图像：主角是 MLP 那条；线性对照单独一张，"容量墙"要看形状而不是只看一个 MSE 数字。
    # 曲线画在 200 点的密集网格上 —— 数据只有 12 个点，直接连线会画成折线，冤枉了模型。
    cx = np.linspace(-3.0, 3.0, 200)
    rep.fit("阶段 2 · MLP+tanh 拟合 sin(x)", xs, ys, predict(mlp, cx), truth=np.sin(cx), cx=cx,
            note=f"最终 MSE={mse_mlp:.5f}；点是 12 个数据、粗线是你的网络、细线是真实 sin")
    rep.fit("对照 · 线性模型（容量不足）", xs, ys, predict(linear, cx), truth=np.sin(cx), cx=cx,
            note=f"MSE={mse_linear:.4f}：它只能画直线，学不动不是你写错了")

    # ---------------- 验收
    rep.above("线性模型学不动", mse_linear, 0.05, note="MSE 卡在 sin 方差附近，说明容量不足而非训练不足")
    rep.soft("深层线性同样学不动", c_deep[-1] > 0.05, f"MSE={c_deep[-1]:.4f} —— 线性∘线性=线性，白叠")
    rep.below("MLP 最终 MSE", mse_mlp, 6e-3)
    rep.check("MLP 比线性好至少 20 倍", mse_linear / max(mse_mlp, 1e-12) >= 20,
              f"线性 {mse_linear:.4f} ÷ MLP {mse_mlp:.6f} = {mse_linear / max(mse_mlp, 1e-12):.0f}×")
    rep.above("MLP loss 下降倍数", c_mlp[0] / max(mse_mlp, 1e-12), 80)

    # ---------------- 反向传播校验（这次参数有 19 个，逐个查）
    probe = ScalarMLP(n_hidden=8, activate=True, seed=3)
    px, py = xs[:2], ys[:2]

    def build():
        probe.zero_grad()
        return probe.loss(px, py)

    ok, worst, detail = gradient_check_graph(build, params=probe.parameters())
    rep.check(f"梯度校验({len(probe.parameters())} 参数 MLP)", ok, f"最大相对误差 {worst:.2e}，阈值 1e-4")
    if not ok:
        print(detail)

    # 逐点误差最大的位置，帮你直观看模型在哪儿偷懒
    errs = [abs(p - y) for p, y in zip(predict(mlp, xs), ys)]
    rep.info("最大单点误差", f"{max(errs):.4f}（出现在 x={xs[int(np.argmax(errs))]:.2f} 附近，通常是端点）")

    passed = rep.done()
    print("\n这一步你确认了两件事：")
    print("  1) 训练不动有两种原因：优化没调好 vs 模型容量不够。看 loss 曲线形态能区分：")
    print("     容量不够 = 很快变平且平在一个较高的值；优化没调好 = 震荡或几乎不动。")
    print("  2) 激活函数不是可选调料，它是'能不能表达这个函数'的开关。")
    print("   后面 transformer 里的 GELU、以及为什么要有 12 层而不是 1 层，理由和这里完全一样。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
