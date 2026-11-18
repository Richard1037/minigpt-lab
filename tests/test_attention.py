"""s04 的判据 · 因果自注意力。跑法：

    python tests/test_attention.py

只依赖 `attention.py`：写完 TODO-12/13/14 这一卷就能全绿。
阶段 5 的 Block/GPT（含 tied 两路累加）在 `tests/test_model.py`，不在这里查。

这一卷要证明的四件事：
  1) softmax 反向满足两个守恒性质（行和为 0、常数余向量下为 0）；
  2) 单头与多头注意力的梯度都对（有限差分）；
  3) 掩码真的挡住了未来 —— 改未来 token，过去的输出**逐位不变**；
  4) 参数量 4d²+3d：k 没有 bias（b_k 的梯度恒为 0）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.attention import CausalSelfAttention, softmax_backward
from minigpt.config import Config
from minigpt.gradcheck import gradcheck
from minigpt.toolkit import Report

rng = np.random.default_rng(11)

# eps=1e-5、atol=1e-9：q/k 的梯度是多条大数相减的抵消结果，绝对值常见 1e-8，
# 有限差分自身的舍入噪声会把相对误差抬到 1e-2 量级。不设绝对容差就会追一个不存在的 bug。
GD = dict(eps=1e-5, tol=1e-4, atol=1e-9)


def tiny_cfg(**kw) -> Config:
    d = dict(vocab=9, d_model=8, n_layer=2, n_head=2, ctx=6)
    d.update(kw)
    return Config(**d)


def main() -> int:
    rep = Report("s04 · 因果自注意力",
                 "有限差分步长 1e-5 + 绝对容差 1e-9；因果性判据独立于梯度检查")

    # ---------------------------------------------------------------- softmax
    rep.group("softmax 反向 · 两个守恒性质")
    z = rng.standard_normal((1, 2, 4, 5))
    e = np.exp(z - z.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    gp = rng.standard_normal(p.shape)
    g = softmax_backward(p, gp)
    sum0 = float(np.max(np.abs(g.sum(-1))))
    const = float(np.max(np.abs(softmax_backward(p, np.full_like(p, 2.5)))))
    rep.check("每行梯度之和为 0", sum0 < 1e-9,
              f"max|Σg|={sum0:.2e} · softmax 每行和恒为 1，动一格整行都要修正")
    rep.check("常数余向量下梯度为 0", const < 1e-9,
              f"max|g|={const:.2e} · softmax(z+c)=softmax(z)，整行加常数不该产生梯度")

    # ---------------------------------------------------------------- 梯度
    rep.group("注意力 · 梯度（单头 / 多头）")
    cases = [
        ("注意力(单头)", CausalSelfAttention(tiny_cfg(d_model=6, n_head=1), rng=rng),
         [rng.standard_normal((2, 5, 6))],
         "H=1 时拆头/合头退化成恒等，先排除多头的轴序干扰"),
        ("注意力(多头)", CausalSelfAttention(tiny_cfg(d_model=8, n_head=4), rng=rng),
         [rng.standard_normal((2, 6, 8))],
         "H=4 才真正走拆头→算→合头，轴序错在这里现形"),
    ]
    for name, module, inputs, why in cases:
        ok, worst, detail = gradcheck(module, inputs, label=name, **GD)
        rep.check(name, ok, f"最大相对误差 {worst:.2e} · {why}")
        if not ok:
            print(detail)

    # ---------------------------------------------------------------- 形状/因果
    rep.group("注意力 · 形状、因果性与参数量")
    cfg = tiny_cfg(d_model=8, n_head=2)
    att = CausalSelfAttention(cfg, rng=rng)
    x = rng.standard_normal((2, 5, 8))
    y = att(x)
    probs = att._probs
    shape_ok = y.shape == x.shape and probs.shape == (2, 2, 5, 5)
    row_dev = float(np.max(np.abs(probs.sum(-1) - 1.0)))
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    upper_max = float(np.max(np.abs(probs * upper)))
    rep.check("形状 (2,5,8)/(2,2,5,5) 与行和", shape_ok and row_dev < 1e-9,
              f"y {y.shape} / probs {probs.shape}；行和偏差 {row_dev:.2e} · 拆头→算→合头→proj 一整圈")
    rep.check("上三角概率精确为 0", upper_max == 0.0,
              f"max|上三角|={upper_max:.2e} · 掩码必须在 softmax 之前")

    x_future = x.copy()
    x_future[0, -1] += 0.5
    y_future = att(x_future)
    rep.check("改未来 token 不影响过去", np.array_equal(y[:, :4], y_future[:, :4]),
              f"前 4 个位置逐位相同；末位差异 {np.max(np.abs(y[0, -1] - y_future[0, -1])):.3f}")
    rep.check("改当前 token 确实改变当前输出", not np.array_equal(y[:, -1], y_future[:, -1]),
              "反面对照：模型不是常数，上一条才有意义")

    x_mid = x.copy()
    x_mid[0, 2] += 0.5
    y_mid = att(x_mid)
    rep.check("影响只向后传播", np.array_equal(y[:, :2], y_mid[:, :2])
              and not np.array_equal(y[:, 3], y_mid[:, 3]),
              "位置 0-1 不变，位置 3 之后变了")

    n_par = sum(p.data.size for p in att.parameters())
    expect = 4 * 8 * 8 + 3 * 8
    rep.close("参数量 4d²+3d（k 没有 bias）", n_par, expect, 0, unit="",
              note="q/v/proj 各有 bias，k 没有 —— b_k 对整行只贡献一个常数，梯度恒为 0")

    att.zero_grad()
    att(x)
    att.backward(rng.standard_normal(x.shape))
    dead = [name for name, p in att.named_parameters() if not np.any(p.grad)]
    rep.check("无死参数", not dead,
              f"梯度恒为 0 的：{dead}" if dead else f"{len(att.parameters())} 个参数全部收到梯度")

    passed = rep.done()
    print("\n这一卷绿了意味着：")
    print("  · 注意力的正向（形状/行和/因果）与反向（probs→v/k/q）都对，单头多头一致；")
    print("  · b_k 真的被数学删掉了 —— 参数量按 4d²+3d 算才过得去。")
    print("下一步（s05）：把注意力接进残差 Block 和 GPT 整机，tied 两路梯度在那里才考。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
