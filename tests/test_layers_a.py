r"""s03a 的判据 · 热身层 Scale 与 Linear。跑法：

    python tests/test_layers_a.py

三组，按你在讲义里的动线排：

  1. **Scale**（TODO-5A）—— 只有乘法的玩具层。它的作用是让你在还不用想形状的时候，
     先把"三件产出、累加、gradcheck 变绿"这一整套体验跑通一次。
  2. **Linear 的有限差分**（TODO-5/6）—— 有 bias、无 bias 两种。
  3. **手算对账** —— 一个 B=1,T=2,C_in=2,C_out=3 的具体例子，把 dW / db / dx 用
     最笨的双重循环逐个元素算出来，再和你的矩阵写法比。判据信的是定义，不是形状技巧。

这一卷绿了就可以进 s03b，不必等后面几个层。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.gradcheck import gradcheck
from minigpt.nn import Linear, Scale
from minigpt.toolkit import Report

rng = np.random.default_rng(7)


def naive_linear(x, W, b, g):
    r"""最笨的 Linear 反向：三重循环按定义算，不用任何矩阵技巧。

    它是讲义 §4 那张"形状推理表"的裁判 —— 你的 `g2.T @ flat_x` 到底等不等于
    $\partial L/\partial W_{oi}=\sum_n g_{no}x_{ni}$，这里说得上话。
    """
    B, T, cin = x.shape
    cout = W.shape[0]
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)
    for n in range(B * T):
        xn = x.reshape(B * T, cin)[n]
        gn = g.reshape(B * T, cout)[n]
        for o in range(cout):
            db[o] += gn[o]
            for i in range(cin):
                dW[o, i] += gn[o] * xn[i]
                dx.reshape(B * T, cin)[n, i] += W[o, i] * gn[o]
    return dx, dW, db


def main() -> int:
    rep = Report("s03a · 热身层 Scale 与 Linear", "有限差分步长 1e-6，相对误差阈值 1e-4")

    # ---- 1) Scale：三件产出的最小样本
    rep.group("热身 · Scale：只有乘法的层，先跑绿一次")
    sc = Scale(2.5)
    ok, worst, detail = gradcheck(sc, [rng.standard_normal((2, 3, 4))], label="Scale")
    rep.check("Scale(参数梯度)", ok, f"最大相对误差 {worst:.2e}"
              + "（y = w·x：dx = w·g，dw = Σ g·x，就这两件）"
              + " · 这条过了说明整套 forward/backward 协议你会用了；没过先别看后面，八成是缓存没存或梯度没累加")
    if not ok:
        print(detail)

    # 累加而不是覆盖：约定 3 的最小可见证据
    g_in = rng.standard_normal((2, 3, 4))
    sc.zero_grad()
    sc(g_in)
    sc.backward(g_in)
    first = sc.w.grad.copy()
    rep.above("第一次 backward 就有非零梯度", float(np.max(np.abs(first))), 1e-6,
              note="这条不成立的话，下面那条'翻倍'会因为两边都是 0 而假通过")
    sc.backward(g_in)
    rep.close("backward 两次后梯度翻倍（累加不是覆盖）",
              float(np.max(np.abs(sc.w.grad - 2.0 * first))), 0.0, 1e-12,
              note="写成 = 而不是 += 的话，这一条和后面的 weight tying 会一起塌")

    # ---- 2) Linear 的有限差分
    rep.group("Linear · 有限差分（有 bias / 无 bias）")
    cases = [
        ("Linear (B,T,C)", Linear(5, 3, rng=rng), [rng.standard_normal((2, 4, 5))],
         "三件产出全查：参数 W、b 和输入 x 各要给出自己的梯度"),
        ("Linear 无 bias", Linear(4, 6, use_bias=False, rng=rng), [rng.standard_normal((3, 2, 4))],
         "use_bias=False 时 self.b is None，判空那条分支在上一组里走不到，所以单独测一组"),
    ]
    for name, module, inputs, why in cases:
        ok, worst, detail = gradcheck(module, inputs, label=name)
        rep.check(name, ok, f"最大相对误差 {worst:.2e} · {why}")
        if not ok:
            print(detail)

    # ---- 3) 手算对账：具体数字，不靠形状技巧
    rep.group("Linear · 与逐元素手算对账（B=1, T=2, C_in=2, C_out=3）")
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    W = np.array([[1.0, 0.5], [-2.0, 1.0], [0.0, 3.0]])
    b = np.array([0.1, -0.2, 0.3])
    g = np.array([[[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]]])
    lin = Linear(2, 3)
    lin.W.data, lin.b.data = W.copy(), b.copy()
    want_dx, want_dW, want_db = naive_linear(x, W, b, g)
    lin.zero_grad()
    lin(x)
    got_dx = lin.backward(g)
    rep.close("dW 等于逐元素定义", float(np.max(np.abs(lin.W.grad - want_dW))), 0.0, 1e-12,
              note="手算 dW[0,0]=Σ_n g_n0·x_n0 = 1·1 + 0.5·3 = 2.5；你的矩阵写法要给出同一个数")
    rep.close("db 等于逐元素定义", float(np.max(np.abs(lin.b.grad - want_db))), 0.0, 1e-12,
              note="对样本维求和：db[o] = Σ_n g[n,o]，形状必须和 b 一样是一维")
    rep.close("dx 等于逐元素定义", float(np.max(np.abs(got_dx - want_dx))), 0.0, 1e-12,
              note="dx 要和 x 同形 (1,2,2)：先 (N,out)@(out,in) 再 reshape 回去")

    passed = rep.done()
    print("\n这一卷绿了意味着：")
    print("  · 你已经有一次 forward 配一次 backward、梯度累加、gradcheck 变绿的完整经验；")
    print("  · 你的 Linear 三种产出既过有限差分，也过逐元素定义 —— 不是碰巧形状对得上。")
    print("下一步（s03b）：Embedding / RMSNorm / GELU / MLP，四个层，每个都只是换了三件产出的算法。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
