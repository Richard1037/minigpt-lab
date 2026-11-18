"""阶段 7 · 采样与生成质量：同一台模型，换一套采样参数就是另一种性格。

训练结束以后，模型对每个位置只给出一个分数向量（logits）。**说出口的是哪一个**，
完全由这个文件里的几行决定。因为采样不参与训练，改坏了不会让 loss 变差 ——
所以判据必须直接盯着**分布**本身，而不是"输出看着还行"。

这一卷分两段：

* 前段在一组**合成分数**上做统计检验（不训模型，跑得快、结论干净）：
  温度改变熵、top-k 改变候选集、top-p 改变概率质量、重复惩罚改变频率。
  每条都用大样本重演一次，和解析算出来的理论值对账。
* 后段才上真模型：一段 8 行的自造语料训 300 步（约 2 秒），
  然后看它贪心能不能接上下句、升温以后会不会换说法、加惩罚会不会少复读。

跑法：
    python stages/s07_generate.py
    MGPT_ANSWER=1 python stages/s07_generate.py

判据里的"候选集"都指**词表下标集合**，不是分数：`{2, 5, 7}` 这种。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.config import Config
from minigpt.gpt import GPT
from minigpt.optim import AdamW, clip_grad_norm, lr_at
from minigpt.sampler import _softmax, apply_penalty, entropy, keep_top_k, keep_top_p, sample_next
from minigpt.dataset import Batcher                                   # ← 阶段 5 的 TODO-21
from minigpt.tokenizer import CharTokenizer                            # ← 阶段 5 的 TODO-20
from minigpt.toolkit import Report, print_progress

# ---------------------------------------------------------------- 合成分数
V = 12
Z = np.round(np.array([0.4, -1.2, 2.1, 0.7, -3.0, 1.5, 0.1, 2.8, -0.5, 1.1, 0.3, -2.2]) * 1.6, 3)
N_BIG = 6000            # 统计判据用的抽样次数：L1 误差 ~ sqrt(V/N) ≈ 0.045，阈值给到 3 倍
N_SMALL = 1200
SEED = 0


def empirical(logits, rng, n, **kw) -> np.ndarray:
    """按 sample_next 抽 n 次，返回每个 id 的经验频率。"""
    counts = np.zeros(logits.size)
    for _ in range(n):
        counts[sample_next(logits, rng, **kw)] += 1.0
    return counts / n


def support(probs, tol=1e-12) -> set[int]:
    return {int(i) for i in np.flatnonzero(probs > tol)}


# ------------------------------------------------------------------ 1) 温度
def check_temperature(rep: Report):
    rep.group("温度 · 分布的陡峭程度（讲义 §2.2）")
    ent = []
    for t in (0.2, 0.5, 1.0, 2.0, 5.0):
        ent.append(entropy(_softmax(keep_top_k(Z / t, 0))))
    mono = all(ent[i] < ent[i + 1] - 1e-9 for i in range(len(ent) - 1))
    rep.check("温度越高熵越大（分布越平）", mono,
              "T 从 0.2→5.0 的熵：" + " ".join(f"{e:.3f}" for e in ent)
              + f"；ln V={math.log(V):.3f} 是上限")
    rng = np.random.default_rng(SEED)
    e_low = entropy(empirical(Z, rng, N_SMALL, temperature=0.3))
    e_high = entropy(empirical(Z, rng, N_SMALL, temperature=2.5))
    rep.below("经验熵也跟着温度上升", e_low, e_high,
              f"实测：T=0.3 时 {e_low:.3f}，T=2.5 时 {e_high:.3f}（这两个数是从抽样频率算的，不是公式）")

    best = int(np.argmax(Z))
    greedy = {sample_next(Z, np.random.default_rng(s), temperature=0.0) for s in (0, 1, 7)}
    rep.check("temperature=0 就是贪心（与 seed 无关）", greedy == {best},
              f"三个不同种子都交出 argmax={best}，结果集合 {sorted(greedy)}")
    g2 = sample_next(Z, np.random.default_rng(3), temperature=1e-3)
    rep.check("温度极小但不为 0 时仍在赌最高分", g2 == best, f"T=1e-3 → {g2}（等于 argmax {best}）")

    p1 = _softmax(Z / 1.0)
    emp = empirical(Z, np.random.default_rng(11), N_BIG, temperature=1.0)
    l1 = float(np.abs(emp - p1).sum())
    rep.below("抽样频率 ≈ softmax 概率（L1 距离）", l1, 0.14,
              f"n={N_BIG}、V={V}：理论误差 sqrt(V/n)≈{math.sqrt(V / N_BIG):.3f}，"
              f"实测 L1={l1:.3f}（阈值 0.14 ≈ 3σ）")
    rep.info("理论 vs 经验（前 5 个 id）",
             " ".join(f"{p1[i]:.3f}/{emp[i]:.3f}" for i in range(5)))


# ------------------------------------------------------------------ 2) top-k
def check_top_k(rep: Report):
    rep.group("top-k · 只保留最高的 k 个候选（讲义 §2.3）")
    order = np.argsort(Z)[::-1]
    for k in (2, 3, 6):
        seen = support(empirical(Z, np.random.default_rng(21), N_SMALL, temperature=1.0, top_k=k))
        want = set(int(i) for i in order[:k])
        rep.check(f"top_k={k} 的候选集恰好是前 {k} 名", seen == want,
                  f"出现过 {sorted(seen)}，期望 {sorted(want)}")
    seen3 = support(empirical(Z, np.random.default_rng(22), N_SMALL, temperature=4.0, top_k=2))
    rep.check("k 这一刀在温度之前生效：升温不引入新候选", seen3 == set(int(i) for i in order[:2]),
              f"T=4、k=2 时仍然只有 {sorted(seen3)} —— 温度只能改变这 k 个之间的比例")
    same = np.allclose(keep_top_k(Z, 0), Z) and np.allclose(keep_top_k(Z, V + 3), Z)
    rep.check("k=0 或 k>=V 时不剪枝", bool(same), "k 是个开关：0 表示关掉 top-k")
    tie_in = keep_top_k(np.array([1.0, 1.0, 0.0]), 1)
    rep.check("并列的最高分都保留", bool(tie_in[0] > 0 and tie_in[1] > 0 and tie_in[2] < 0),
              f"两个并列 1.0 都活着：[{tie_in[0]:g}, {tie_in[1]:g}, {tie_in[2]:g}] —— "
              "写成 <= 会把并列的两个一起掩掉")


# ------------------------------------------------------------------ 3) top-p
def check_top_p(rep: Report):
    rep.group("top-p · nucleus 覆盖多少概率质量（讲义 §2.4）")
    p1 = _softmax(Z)
    for pp in (0.5, 0.8, 0.95):
        q = keep_top_p(p1, pp)
        s = support(q)
        mass = float(np.sum(q))
        orig = float(np.sum(p1[list(s)])) if s else 0.0
        ok_prefix = s == set(int(i) for i in np.argsort(p1)[::-1][:len(s)])
        rep.check(f"top_p={pp} 的核覆盖原分布 ≥ {pp} 的质量",
                  abs(mass - 1.0) < 1e-12 and orig >= pp - 1e-12 and ok_prefix,
                  f"核 = {sorted(s)} · 在原分布里占 {orig:.3f} · 归一化后和为 {mass:.6f} · "
                  f"是概率降序的前缀：{ok_prefix}")
    q_no = keep_top_p(p1, 1.0)
    rep.check("top_p=1 不改变分布", bool(np.allclose(q_no, p1, atol=1e-12)),
              f"最大差异 {float(np.max(np.abs(q_no - p1))):.2e}")
    hot = _softmax(Z / 1e-4)
    q_hot = keep_top_p(hot, 0.5)
    rep.check("极端 one-hot 时不冒出 nan", bool(np.isfinite(q_hot).all() and abs(q_hot.sum() - 1.0) < 1e-12),
              f"T=1e-4 把分布压成 one-hot（第一名概率 {hot.max():.6f} > p=0.5），"
              f"核里只剩它一个 → 下标 {int(np.argmax(q_hot))}，全程没有 nan")
    emp = empirical(Z, np.random.default_rng(31), N_SMALL, temperature=1.0, top_p=0.8)
    q = keep_top_p(p1, 0.8)
    l1 = float(np.abs(emp - q).sum())
    rep.below("按 top-p 抽样后仍与归一化核一致", l1, 0.14,
              f"n={N_SMALL}，实测 L1={l1:.3f}（核只留下 {len(support(q))} 个候选，所以抽样更集中）")


# ------------------------------------------------------------------ 4) 重复惩罚
def check_penalty(rep: Report):
    rep.group("重复惩罚 · 让出现过的词少再来一次（讲义 §2.5）")
    z = Z.copy()
    keep = z.copy()
    out = apply_penalty(z, [7, 4], 2.5)
    rep.check("惩罚不修改调用者的 logits", bool(np.array_equal(z, keep) and out is not z),
              f"入参数组逐比特没动：{np.array_equal(z, keep)}；返回的是新数组：{out is not z}"
              "（`np.asarray(...).reshape(-1)` 给的是视图，就地改会把模型刚算好的那一行毁掉）")
    rep.check("正分除以惩罚、负分乘以惩罚",
              bool(abs(out[7] - Z[7] / 2.5) < 1e-12 and abs(out[4] - Z[4] * 2.5) < 1e-12),
              f"id 7（正分 {Z[7]:.2f}）→ {out[7]:.3f}；id 4（负分 {Z[4]:.2f}）→ {out[4]:.3f} —— "
              "两种情况都是往更差的方向推；统一除会把负分抬向 0")

    top = int(np.argmax(Z))
    rng_a = np.random.default_rng(41)
    rng_b = np.random.default_rng(41)
    fa = empirical(Z, rng_a, N_SMALL, temperature=1.0)[top]
    fb = empirical(Z, rng_b, N_SMALL, temperature=1.0, seen=[top], penalty=3.0)[top]
    rep.below("被罚的那个词经验频率明显下降", fb, fa * 0.75,
              f"penalty=1 时 {fa:.3f}，penalty=3 时 {fb:.3f}（同一个 rng，同一段抽样序列）")
    noop = sample_next(Z, np.random.default_rng(2), temperature=0.0, seen=[top], penalty=1.0)
    rep.check("penalty=1 等于不惩罚", noop == top, f"返回 {noop}（还是 argmax {top}）")
    oob = apply_penalty(Z, [V + 5, -2], 3.0)
    rep.check("越界 id 被安全忽略", bool(np.allclose(oob, Z)),
              "生成时序列里可能带着旧 prompt 的 id，换过 tokenizer 就会超出词表")


# ------------------------------------------------------------------ 5) 端到端
def check_generate(rep: Report):
    rep.group("端到端 · 一台真训过的小模型开口说话（讲义 §8）")
    text = ("the cat sat on the mat .\n" * 3 + "the dog ran in the park .\n" * 3
            + "the bird sang at the dawn .\n" * 3 + "the fish swam by the rock .\n" * 3) * 3
    tok = CharTokenizer(text)
    cfg = Config(vocab=tok.vocab_size, d_model=48, n_layer=2, n_head=4, ctx=32)
    ids = tok.encode_array(text)
    bat = Batcher(ids, cfg.ctx)
    model = GPT(cfg, rng=np.random.default_rng(1))
    opt = AdamW(model, lr=6e-3, weight_decay=0.0)
    rng = np.random.default_rng(7)
    print("\n先花两秒钟训一台能读这段小诗的模型（300 步）…")
    curve = []
    for i in range(300):
        opt.zero_grad()
        x, _y = bat(4, rng)
        curve.append(model.loss_backward(x))
        clip_grad_norm(model, 1.0)
        opt.step(lr_at(i, 300, 6e-3, warmup=30))
        print_progress(i + 1, 300, every=100, prefix="阶段 7 的小模型")
    print(f"\n末段 loss {float(np.mean(curve[-20:])):.3f}（ln V = {math.log(cfg.vocab):.3f}）")
    rep.curve("阶段 7 小模型训练 loss", curve, note="只为生成演示服务，300 步")

    probe = "the cat sat on the "
    prompt = tok.encode_array(probe)
    idx = text.rfind(probe) + len(probe)              # 取语料里**最后一次**出现，真值唯一
    truth = ids[idx:idx + 10]

    greedy = model.generate(prompt, max_new_tokens=10, temperature=0.0, top_k=0, top_p=1.0, seed=0)
    got = np.asarray(greedy[len(prompt):], dtype=np.int64)
    agree = float(np.mean(got == truth)) if got.size == truth.size else 0.0
    rep.above("贪心续写与原文逐字符一致率", agree, 0.70,
              f"喂进 {probe!r}，写出 {tok.decode(got)!r}（真值 {tok.decode(truth)!r}）")

    g_a = model.generate(prompt, max_new_tokens=10, temperature=0.0, seed=0)
    g_b = model.generate(prompt, max_new_tokens=10, temperature=0.0, seed=999)
    rep.check("temperature=0 时两个不同种子输出完全相同", g_a == g_b,
              f"贪心不经过任何随机：{tok.decode(g_a[len(prompt):])!r}")

    k1 = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=1, seed=5)
    rep.check("top_k=1 与贪心等价", k1 == g_a, "只留一个候选，抽签就只有一个签")

    # 这台模型训得很熟，T=1.0 时它几乎总是选同一个词 —— 所以多样性的判据把温度开到 2.0。
    # 这本身就是个真实结论：**模型越自信，同样的温度越驯服**。
    cold = {model.generate(prompt, max_new_tokens=1, temperature=0.0, seed=s)[len(prompt)]
            for s in range(24)}
    hot = {model.generate(prompt, max_new_tokens=1, temperature=2.0, top_k=8, seed=s)[len(prompt)]
           for s in range(24)}
    rep.check("贪心：24 个种子只有 1 种可能", len(cold) == 1,
              f"T=0 → {sorted(cold)}（就是 argmax）")
    rep.above("T=2.0 时 24 个种子的不同首字符数", float(len(hot)), 3.0,
              f"出现的候选：{sorted(tok.char_at(i) for i in hot)!r} —— 温度把分差抹平之后，"
              "老二老三就有机会开口了")

    texts_hot = {tuple(model.generate(prompt, max_new_tokens=10, temperature=2.0, top_k=8,
                                      seed=s)[len(prompt):]) for s in range(10)}
    rep.above("同一段 prompt 十个种子的不同续写数（T=2.0）", float(len(texts_hot)), 5.0,
              f"{len(texts_hot)} 种 —— 温度 0 时只有 1 种")

    s1 = list(texts_hot)[0]
    rep.check("生成的 token 都能被 tokenizer 解回文本", isinstance(tok.decode(s1), str)
              and all(0 <= int(t) < cfg.vocab for t in s1),
              f"例：{tok.decode(s1)!r}")

    over = model.generate(np.arange(cfg.ctx + 7) % cfg.vocab, max_new_tokens=5,
                          temperature=1.0, top_k=6, seed=0)
    rep.check("prompt 比 ctx 长时走滑窗而不是崩掉", len(over) == cfg.ctx,
              f"prompt {cfg.ctx + 7} 个 > ctx {cfg.ctx}：每生成一个先丢掉最老的一个，"
              f"总长永远不超过 ctx —— 输出 {len(over)} 个 token")

    pen = model.generate(prompt, max_new_tokens=20, temperature=1.5, top_k=8, seed=3)
    rep_pen = model.generate(prompt, max_new_tokens=20, temperature=1.5, top_k=8, seed=3,
                             repetition_penalty=1.6)
    rep.soft("重复惩罚换来更多样的字符分布",
             len(set(rep_pen[len(prompt):])) >= len(set(pen[len(prompt):])),
             f"penalty=1.0 的 20 个字符里有 {len(set(pen[len(prompt):]))} 种不同；"
             f"penalty=1.6 → {len(set(rep_pen[len(prompt):]))} 种。样本很小，只是给你一眼方向感")
    rep.info("T=2.0 的一种续写", repr(tok.decode(s1)))
    rep.info("penalty=1.6 的同一段", repr(tok.decode(rep_pen[len(prompt):])))


def main() -> int:
    rep = Report("阶段 7 · 采样与生成质量",
                 f"前段：{V} 个固定分数上做大样本统计（n={N_BIG}/{N_SMALL}）；后段：一台真训过的小模型")
    check_temperature(rep)
    check_top_k(rep)
    check_top_p(rep)
    check_penalty(rep)
    check_generate(rep)

    passed = rep.done()
    print("\n这一步你确认了三件事：")
    print("  1) 四个开关各自的作用点：温度改**形状**、top-k 改**候选集**、top-p 改**概率质量**、"
          "惩罚改**分数**，顺序换了结论就换。")
    print("  2) 抽样频率能对上解析概率 —— 这是「真的在按分布抽」的唯一证据，看文本是看不出来的。")
    print("  3) 贪心确定、采样随机、top_k=1 等价于贪心、超长 prompt 走滑窗：接口的四条边界都在。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
