"""s03b 的判据 · Embedding / RMSNorm / GELU / MLP。跑法：

    python tests/test_layers_b.py

四个层，每个层要交的还是那三件产出 —— 换的只是算法：

  · **GELU** 没有参数，只有输入梯度（返回值）；
  · **RMSNorm** 有一个不吃 weight decay 的权重；
  · **Embedding** 输入是整数 id，所以**没有输入梯度**，但参数梯度有两条额外的坑
    （重复 id 要累加、没出现的行必须恰好是 0）；
  · **MLP** 是三个层的串联，第一次真正考"缓存有没有存对、顺序有没有反"。

前提：s03a 已经绿了（`python tests/test_layers_a.py`）。这一卷不重测 Linear。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.gradcheck import gradcheck
from minigpt.nn import Embedding, GELU, MLP, RMSNorm
from minigpt.toolkit import Report

rng = np.random.default_rng(11)


def main() -> int:
    rep = Report("s03b · 四个层的反向传播", "有限差分步长 1e-6，相对误差阈值 1e-4")

    rep.group("四个层的有限差分 · GELU / RMSNorm / MLP")
    cases = [
        ("GELU", GELU(), [rng.standard_normal((2, 3, 4)) * 2.0],
         "没有参数，只查输入梯度：dy/dx 那两项少一项就是 1e0 量级"),
        ("RMSNorm", RMSNorm(8), [rng.standard_normal((2, 5, 8))],
         "同时查 dx 和 dw；漏掉耦合项 -r*xh*s，误差从 1e-9 跳到 1e-1"),
        ("MLP", MLP(6, 16, rng=rng), [rng.standard_normal((2, 3, 6))],
         "组合层：fc1/fc2 的 W、b 加输入 x 全查；漏掉 act.backward 形状照样合法"),
    ]
    for name, module, inputs, why in cases:
        ok, worst, detail = gradcheck(module, inputs, label=name)
        rep.check(name, ok, f"最大相对误差 {worst:.2e} · {why}")
        if not ok:
            print(detail)

    rep.group("Embedding · 查表、重复 id 累加、未用行必须为 0")
    emb = Embedding(11, 4, rng=rng)
    ids = rng.integers(0, 11, size=(2, 5))
    ok, worst, detail = gradcheck(emb, [ids], label="Embedding")
    rep.check("Embedding(参数梯度)", ok, f"最大相对误差 {worst:.2e}"
              + " · 输入是整数，有限差分动不了它，所以只查参数梯度")
    if not ok:
        print(detail)
    dup_ids = np.array([[3, 3, 3, 0, 7]])
    emb.zero_grad()
    emb(dup_ids)
    emb.backward(np.ones((1, 5, 4)))
    rep.close("重复 id 梯度累加", emb.weight.grad[3].sum(), 3.0 * 4, 1e-9,
              unit="", note="id=3 出现三次，每次贡献 4 个 1；覆盖写法只会得到 4")
    rep.check("未出现的行梯度为 0", np.all(emb.weight.grad[1] == 0.0), "id=1 不在序列里")

    passed = rep.done()
    print("\n这一卷绿了意味着：")
    print("  · 四个层各自的反向都对，MLP 的串联顺序与缓存也对；")
    print("  · Embedding 的散射累加是'加'而不是'赋值'—— 这是 weight tying 的同一条规矩。")
    print("下一步（s03c）：CrossEntropy，唯一输出标量的那一层，也是最容易写错的一层。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
