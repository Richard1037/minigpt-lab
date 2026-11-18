"""阶段 1 · 拟合一条直线：搞懂"梯度下降"这四个字。

任务：给你 100 个 (x, y) 点，它们严格满足 y = 3x + 1。你不知道这个式子，
只让程序自己调 a 和 b，使 (a·x + b) 尽量接近 y。

这一步刻意不用任何"框架"。先看手写梯度，再看自动微分，两者结果必须一模一样 ——
这就是在告诉你：自动微分没有任何魔法，它只是替你求了偏导数。

跑法：
    python stages/s01_fit_line.py

学到什么：
  * loss 是"错得多离谱"的一个数字，训练就是让它变小
  * 梯度 = 参数变动时 loss 的变化率；往梯度的反方向走，loss 就降
  * 学习率太大 / 太小会怎样（本脚本末尾会现场演示）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.exercises import fit_by_autograd   # ← TODO-1A 在这个文件里写
from minigpt.scalar import Value, gradient_check_graph
from minigpt.toolkit import Report, print_progress, scatter, sparkline

TRUE_A, TRUE_B = 3.0, 1.0
N_POINTS = 100
SEED = 0


def make_data():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-1.0, 1.0, size=N_POINTS)
    y = TRUE_A * x + TRUE_B
    return x, y


# ---------------------------------------------------------------- 第 A 部分：手写梯度
def fit_by_hand(x, y, iters=3000, lr=0.2, log_every=500):
    """不用任何自动微分，偏导数自己推。

    loss(a,b) = mean_i (a·x_i + b − y_i)^2
    令 e_i = a·x_i + b − y_i，则
      ∂loss/∂a = mean_i 2·e_i·x_i
      ∂loss/∂b = mean_i 2·e_i
    这就是链式法则：先对平方求导(2e)，再对 e 里 a 的依赖求导(x_i)。
    """
    a, b = 0.0, 0.0
    curve = []
    for _ in range(iters):
        e = a * x + b - y
        loss = float(np.mean(e * e))
        curve.append(loss)
        ga = float(np.mean(2.0 * e * x))
        gb = float(np.mean(2.0 * e))
        a -= lr * ga
        b -= lr * gb
        if log_every:
            print_progress(len(curve), iters, every=log_every, prefix="手写梯度")
    return a, b, curve


# ---------------------------------------------------------------- 第 B 部分：自动微分
# fit_by_autograd 从 minigpt.exercises 导入 —— 那是你要写的练习（TODO-1A）。
# 本脚本只负责调用它、和第 A 部分的手推结果对账。
# ---------------------------------------------------------------- 第 C 部分：演示学习率翻车
def lr_demo(x, y, lr):
    """跑 300 步就停，返回最终 loss。用于展示 lr 过大时的发散。"""
    a, b = 0.0, 0.0
    for _ in range(300):
        e = a * x + b - y
        a -= lr * float(np.mean(2.0 * e * x))
        b -= lr * float(np.mean(2.0 * e))
    return float(np.mean((a * x + b - y) ** 2)), a, b


def _trim(v: float) -> str:
    """发散时数值会大到几百位，打印时截断一下，别把表格撑爆。"""
    return f"{v:.3f}" if abs(v) < 1e6 else "∞"


def main() -> int:
    rep = Report("阶段 1 · 拟合一条直线", f"目标函数 y = {TRUE_A}x + {TRUE_B}，{N_POINTS} 个点")
    print("\n先跑第 A 部分：完全手写的梯度下降")
    x, y = make_data()
    a_h, b_h, curve_h = fit_by_hand(x, y)
    print("\n再跑第 B 部分：用计算图自动求梯度（同一个问题）")
    a_g, b_g, curve_g = fit_by_autograd(x, y)

    print()
    print(scatter(x, y, a_h * x + b_h, label="拟合直线"))
    print()
    print(sparkline(curve_h, label="loss(手写)"))
    # 同一份数据再登记一次：命令行看上面的 ASCII，网站用 SVG 画这条（见 toolkit.Report.curve）
    rep.curve("loss(手写偏导)", curve_h, note=f"a={a_h:.4f} b={b_h:.4f}")
    rep.curve("loss(自动微分)", curve_g, note=f"a={a_g:.4f} b={b_g:.4f}")
    # 拟合图像：数据点、你的拟合线、真实直线三份同源，网页/结果窗口只是把它们画出来
    rep.fit("阶段 1 · 直线拟合", x, y, a_g * x + b_g, truth=TRUE_A * x + TRUE_B,
            note=f"自动微分跑出来的 a={a_g:.4f} b={b_g:.4f}；真实 a={TRUE_A} b={TRUE_B}")

    # ---- 判据 1：学到了正确的 a 和 b
    rep.close("学到斜率 a", a_h, TRUE_A, 0.02, unit="")
    rep.close("学到截距 b", b_h, TRUE_B, 0.02, unit="")
    # ---- 判据 2：loss 真的降到接近 0
    rep.below("最终 MSE(手写)", curve_h[-1], 1e-4)
    rep.above("loss 下降倍数", curve_h[0] / max(curve_h[-1], 1e-12), 1e3, note="第一版就有 3 个数量级的下降才算真在学")
    # ---- 判据 3：自动微分和手写结果一致 —— 证明你信得过后者
    rep.below("两种方法 a 之差", abs(a_h - a_g), 1e-6)
    rep.below("两种方法 b 之差", abs(b_h - b_g), 1e-6)
    # ---- 判据 4：反向传播本身正确（有限差分数值验证）
    ok, worst, detail = _gradcheck_line()
    rep.check("梯度校验(有限差分)", ok, f"最大相对误差 {worst:.2e}，阈值 1e-6")
    if not ok:
        print(detail)

    # ---- 学习率演示（不计入验收，纯观察）
    print("\n顺手看看学习率的影响（各跑 300 步）：")
    for lr in (0.001, 0.2, 0.55, 1.5):
        loss, a, b = lr_demo(x, y, lr)
        verdict = "太小：还在半路" if loss > 1e-3 and lr < 0.01 else (
            "刚好" if loss < 1e-30 else ("临界：还在降" if loss < 1 else "太大：炸了"))
        rep.info(f"lr={lr} 最终 loss", f"{loss:.4g}  a={_trim(a)} b={_trim(b)}   {verdict}")

    passed = rep.done()
    print("\n这一步你确认了三件事：")
    print("  1) loss 会降，且降了 3 个数量级以上；")
    print("  2) 手推偏导 = 自动微分，二者数值完全一致；")
    print("  3) 学习率有稳定区间，超出就不降反升。")
    print("   后面每一个更复杂的模型，靠的还是这三条。")
    return 0 if passed else 1


def _gradcheck_line():
    xs, ys = make_data()
    a, b = Value(0.7, label="a"), Value(-0.3, label="b")

    def build():
        a.grad = b.grad = 0.0
        loss = Value(0.0, label="loss")
        for xi, yi in zip(xs, ys):
            e = a * xi + b - yi
            loss = loss + e * e
        return loss / len(xs)

    return gradient_check_graph(build, params=[a, b])


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
