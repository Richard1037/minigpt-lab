"""整机校验：注意力、Block、完整 GPT。跑法：

    python tests/test_model.py

这一层要证明的四件事，缺一不可：
  1) 每一层的反向传播数值正确（有限差分）
  2) 因果掩码真的挡住了未来 —— 改最后一个 token，前面的输出必须**逐位不变**
  3) 预测目标是"下一个 token"而不是"自己"（移位错位是新手最致命且最隐蔽的 bug）
  4) 权重共享没有把同一个参数登记两次
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.attention import CausalSelfAttention
from minigpt.config import Config
from minigpt.gpt import Block, GPT
from minigpt.gradcheck import gradcheck
from minigpt.nn import CrossEntropy
from minigpt.optim import AdamW
from minigpt.toolkit import Report

rng = np.random.default_rng(11)


def tiny_cfg(**kw) -> Config:
    d = dict(vocab=9, d_model=8, n_layer=2, n_head=2, ctx=6)
    d.update(kw)
    return Config(**d)


# ------------------------------------------------------------------ 1) 梯度
# 整机/注意力用 eps=1e-5、atol=1e-9，而不是简单层的 1e-6/0：
# q、k 的梯度是多个大数相减的抵消结果，绝对值常常只有 1e-8，这时有限差分的
# 舍入噪声（约 ulp(目标)/eps ≈ 1e-11）在相对误差里占比就很大 —— 看着像 bug，
# 其实是噪声底。判据要能区分这两种情况，否则你会花一小时去查一个不存在错误。
GD = dict(eps=1e-5, tol=1e-4, atol=1e-9)


def check_grads(rep: Report) -> None:
    rep.group("梯度 · 整机与注意力（含权重共享的两路累加）")
    for label, module, inputs in [
        ("注意力(单头)", CausalSelfAttention(tiny_cfg(d_model=6, n_head=1), rng=rng),
         [rng.standard_normal((2, 5, 6))]),
        ("注意力(多头)", CausalSelfAttention(tiny_cfg(d_model=8, n_head=4), rng=rng),
         [rng.standard_normal((2, 6, 8))]),
        ("Block", Block(tiny_cfg(), rng=rng), [rng.standard_normal((2, 5, 8))]),
    ]:
        ok, worst, detail = gradcheck(module, inputs, label=label, **GD)
        rep.check(f"{label} 反向正确", ok, f"最大相对误差 {worst:.2e}")
        if not ok:
            print(detail)

    # 整机：整数输入 → 只查参数梯度，但这已经覆盖了 tied 输出头的梯度累加。
    # 如果 TiedHead.backward 漏了 weight.grad +=，tok.weight 这一项就会对不上。
    model = GPT(tiny_cfg(), rng=rng)
    ids = rng.integers(0, model.cfg.vocab, size=(2, 5))
    ok, worst, detail = gradcheck(model, [ids], label="GPT", **GD)
    rep.check("GPT 整机反向正确(含权重共享)", ok, f"最大相对误差 {worst:.2e}")
    if not ok:
        print(detail)

    # 每个参数都要真的收到非零梯度：漏接一条支路的典型症状是"某个参数梯度恒为 0"
    model.zero_grad()
    model.loss_backward(ids)
    dead = [n for n, p in model.named_parameters() if not np.any(p.grad)]
    rep.check("无死参数(所有参数都有梯度)", not dead,
              f"梯度恒为 0 的：{dead}" if dead else f"{len(model.parameters())} 个参数全部收到梯度")


# ------------------------------------------------------------------ 2) 因果性
def check_causality(rep: Report) -> None:
    rep.group("因果性 · 未来看不到过去，改动只向后传")
    cfg = tiny_cfg(n_layer=3, ctx=8)
    model = GPT(cfg, rng=np.random.default_rng(3))
    T = 7
    ids = rng.integers(0, cfg.vocab, size=(1, T))

    base = model.forward(ids)

    # 改最后一个 token：位置 0..T-2 的输出必须一个比特都不能变
    alt = ids.copy()
    alt[0, -1] = (alt[0, -1] + 1) % cfg.vocab
    changed = model.forward(alt)
    rep.check("改未来 token 不影响过去", np.array_equal(base[:, :T - 1], changed[:, :T - 1]),
              f"前 {T-1} 个位置逐位相同；末位置差异 "
              f"{np.max(np.abs(base[0, T - 1] - changed[0, T - 1])):.3f}")
    rep.check("改当前 token 确实改变当前输出", not np.array_equal(base[:, T - 1], changed[:, T - 1]),
              "否则说明模型是个常数，上面的'不影响过去'毫无意义")

    # 改中间 token：它之后的位置应该变，之前的不该变
    mid = ids.copy()
    mid[0, 2] = (mid[0, 2] + 3) % cfg.vocab
    m = model.forward(mid)
    rep.check("影响只向后传播", np.array_equal(base[:, :2], m[:, :2])
              and not np.array_equal(base[:, 3], m[:, 3]),
              "位置 0-1 不变，位置 3 之后变了")

    # 反向也要遵守因果：未来 token 的梯度不该回流到过去位置的嵌入
    model.zero_grad()
    model.loss_backward(ids)
    g = model.tok.weight.grad
    rep.info("词嵌入有梯度的行数", f"{int(np.any(g, axis=1).sum())} / {cfg.vocab}")


# ------------------------------------------------------------------ 3) 移位方向
def check_label_shift(rep: Report) -> None:
    rep.group("移位方向 · 学的是“预测下一个”而不是“预测自己”")
    """最有价值的一条：证明我们学的是"预测下一个"，而不是"预测自己"。

    造一段**完全随机**的 token。随机序列是不可预测的，所以：
      · 目标 = 下一个 token  → loss 只能降到 ln(V) 附近，降不下去才是对的
      · 目标 = 自己           → 输入嵌入里就有答案，几步就能把 loss 打到接近 0
    如果你哪天把 shift 写错，这条判据会立刻抓住它 —— 而"loss 掉得飞快"看起来很爽。
    """
    cfg = tiny_cfg(vocab=12, d_model=16, n_layer=1, n_head=2, ctx=16)
    train_data = np.random.default_rng(5).integers(0, cfg.vocab, size=(4, 16))
    # 判据看的是**没训过的新数据**。这个模型有 ~2000 个参数、只有 60 个训练样本，
    # 背下随机序列完全做得到（训练 loss 能压到 0.001）—— 那不是"学到了规律"，
    # 是过拟合。所以只看 test loss：随机的下一 token 无法泛化，"预测自己"能泛化。
    test_data = np.random.default_rng(77).integers(0, cfg.vocab, size=(4, 16))
    baseline = math.log(cfg.vocab)

    def run(target_self: bool, steps: int = 150) -> float:
        model = GPT(cfg, rng=np.random.default_rng(9))
        opt = AdamW(model, lr=8e-3, weight_decay=0.0)
        ce = CrossEntropy()
        for _ in range(steps):
            opt.zero_grad()
            if target_self:
                # 故意写错的样子：位置 t 的输出配位置 t 的 token（预测自己）
                logits = model.forward(train_data)
                g = ce(logits[:, :-1, :], train_data[:, :-1])
                g = ce.backward(1.0)
                full = np.zeros(logits.shape, dtype=np.float64)
                full[:, :-1, :] = g                        # 末位没参与 loss，梯度补 0
                model.backward(full)
            else:
                model.loss_backward(train_data)
            opt.step()
        # 换一份全新的随机数据评估
        logits = model.forward(test_data)
        if target_self:
            return float(ce(logits[:, :-1, :], test_data[:, :-1]))
        return float(ce(logits[:, :-1, :], test_data[:, 1:]))

    next_loss = run(False)
    self_loss = run(True)
    rep.check("正确移位：新数据上学不动", next_loss > 0.75 * baseline,
              f"test loss {next_loss:.3f} vs 基准 ln(V)={baseline:.3f}（随机序列无法泛化，学不动才对）")
    rep.soft("错误移位：预测自己在任何数据上都秒收敛", self_loss < 0.2 * baseline,
             f"test loss {self_loss:.4f} ≪ 基准 —— 这就是 shift 写错时长得'很健康'的样子，别被骗")


# ------------------------------------------------------------------ 4) 参数登记
def check_tying(rep: Report) -> None:
    rep.group("参数登记 · tied 权重不能登记两次")
    model = GPT(tiny_cfg(vocab=40), rng=rng)
    named = model.named_parameters()
    ids_ = [id(p) for _, p in named]
    rep.check("参数不重复登记", len(ids_) == len(set(ids_)),
              f"{len(ids_)} 个登记项 / {len(set(ids_))} 个唯一对象")
    rep.check("输出头复用的是词嵌入矩阵", model.head.weight is model.tok.weight,
              "id 相同 → 更新一次、存一份、梯度两路相加")

    V, C, L = model.cfg.vocab, model.cfg.d_model, model.cfg.n_layer
    # 每层：注意力 4d² + 3d（q/v/proj 有 bias，k 没有 —— 它的 bias 梯度恒为 0）
    #       + FFN 8d² + 5d + 两个 RMSNorm 的 w 共 2d
    per_layer = 12 * C * C + 10 * C
    expect = V * C + model.cfg.ctx * C + L * per_layer + C
    got = sum(p.data.size for _, p in named)
    rep.close("参数量与手算一致", got, expect, 0,
              note="词嵌入 + 位置嵌入 + 每层(12d²+10d) + 末端 norm；tied 头不另算")
    rep.info("配置估算 estimate_params", f"{model.cfg.estimate_params():,}（不含 bias/位置嵌入的粗略值）")

    # 优化器只更新一次：如果重复登记，tied 那份会被走两遍，等效 lr 翻倍
    model.zero_grad()
    p = model.tok.weight
    before = p.data.copy()
    opt = AdamW(model, lr=1e-2, weight_decay=0.0)
    p.grad += 1.0
    opt.step()
    moved = float(np.max(np.abs(p.data - before)))
    rep.below("tied 参数单步移动幅度", moved, 1.01 * 1e-2,
              note="Adam 单步上限就是 lr；翻倍说明被更新了两次")


# ------------------------------------------------------------------ 生成
def check_generate(rep: Report) -> None:
    rep.group("生成 · 长度、词表范围、贪心可复现")
    cfg = tiny_cfg(ctx=12, vocab=20)
    model = GPT(cfg, rng=np.random.default_rng(1))
    prompt = np.array([[1, 2, 3, 4]])
    out = model.generate(prompt, max_new_tokens=8, temperature=0.7, top_k=5, seed=0)
    rep.check("生成长度正确", len(out) == prompt.size + 8, f"得到 {len(out)} 个 token")
    rep.check("生成 id 全部在词表内", all(0 <= t < cfg.vocab for t in out),
              f"min={min(out)} max={max(out)} < vocab={cfg.vocab}")

    a = model.generate(prompt, max_new_tokens=6, temperature=0.0, seed=0)
    b = model.generate(prompt, max_new_tokens=6, temperature=0.0, seed=7)
    rep.check("temperature=0 是贪心且可复现", a == b, "同一份权重 + 贪心 → 逐 token 相同")

    # 超过 ctx 时要截断而不是崩掉
    long_prompt = np.arange(cfg.ctx + 5)[None] % cfg.vocab
    try:
        o2 = model.generate(long_prompt, max_new_tokens=4, temperature=1.0, seed=0)
        rep.check("超长 prompt 走滑窗而不是崩掉", len(o2) == cfg.ctx,
                  f"prompt {long_prompt.shape[1]} > ctx {cfg.ctx}：丢掉最老的 "
                  f"{long_prompt.size + 4 - len(o2)} 个 token，输出 {len(o2)} 个")
    except Exception as exc:                                   # noqa: BLE001
        rep.check("超长 prompt 被滑窗截断", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    rep = Report("整机校验 · 注意力 / Block / GPT",
                 "有限差分步长 1e-5 + 绝对容差 1e-9；因果性与移位判据独立于梯度检查")
    check_grads(rep)
    check_causality(rep)
    check_label_shift(rep)
    check_tying(rep)
    check_generate(rep)
    ok = rep.done()
    print("\n读一下这几条：")
    print("  · '改未来 token 不影响过去'不成立 → 掩码漏了或加在了 softmax 之后，模型能偷看答案。")
    print("  · '正确移位：随机数据学不动'反而**必须**成立：学不动说明它没在作弊。")
    print("  · '参数不重复登记'挂了 → 检查点会存两份一样的权重，优化器会对 tied 参数走两步。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
