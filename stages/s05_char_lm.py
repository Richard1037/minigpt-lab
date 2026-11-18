"""阶段 5 · 完整 GPT + 训练循环：一台真模型在一段真文本上学会「下一个字符」。

前面四个阶段你写的都是零件。这一阶段把它们接成一台机器，然后**开起来**：
分词 → 切窗口 → 组装 GPT → 算 loss → AdamW 更新 → 看着 loss 从 ln V 掉下去。

这一步要现场回答三个问题，每个问题都有一条判据守着：

1) **这台机器的梯度是对的吗？** 有限差分逐个参数查，包括权重共享那条双路累加。
2) **它学的是「下一个字符」还是「自己」？** 用一份**随机数据**当对照组：
   随机序列的下一个字符不可预测，所以 loss 必须**降不下去**（停在 ln V 附近）。
   移位写错（预测自己）时随机数据也能被压到接近 0 —— 这一条就是照妖镜。
3) **优化器真的在起作用吗？** 单步幅度、梯度裁剪、学习率调度各自单独验证，
   不接受「loss 大概在降」这种模糊证据。

跑法：
    python stages/s05_char_lm.py
    MGPT_ANSWER=1 python stages/s05_char_lm.py      （先看通过长什么样）

语料是本文件里的 `TEXT`（8 行短句 ×6 = 1434 个字符、25 个不同字符）。
它不是 `data/corpus.txt`：阶段 5 要的是「几秒内能学会、学没学会一眼看得出来」，
真数据留到阶段 6。整卷在 CPU 上约 10 秒。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.config import Config
from minigpt.dataset import Batcher, evaluate, token_stream, train_val_texts   # ← TODO-21
from minigpt.gpt import Block, GPT, TiedHead                                   # ← TODO-15..19
from minigpt.gradcheck import gradcheck
from minigpt.nn import Parameter                                               # ← 阶段 3
from minigpt.optim import AdamW, clip_grad_norm, lr_at                         # ← TODO-22/23/24
from minigpt.tokenizer import CharTokenizer                                    # ← TODO-20
from minigpt.toolkit import Report, print_progress, sparkline

# ---------------------------------------------------------------- 数据与配置
LINES = [
    "the cat sat on the mat .",
    "the dog ran in the park .",
    "the bird sang at the dawn .",
    "the fish swam by the rock .",
    "a cat slept and a dog barked .",
    "a bird flew and a fish jumped .",
    "the cat saw the dog at the gate .",
    "the dog saw the bird in the tree .",
]
TEXT = "".join(ln + "\n" for ln in LINES) * 6
# train/val 泄漏那一条要一份「每行都不重复」的语料才测得出来（TEXT 里同一行出现 6 次，
# 按行切也必然重叠 —— 那是数据性质，不是切分函数的 bug）。
SPLIT_TEXT = "".join(f"row {i:02d} is a line of its own .\n" for i in range(40))

B, CTX = 4, 32
D_MODEL, N_LAYER, N_HEAD = 48, 2, 4
STEPS = 400
OVERFIT_STEPS = 250
RANDOM_STEPS = 150
LR, WARMUP = 6e-3, 40
SEED = 0

# 有限差分：简单层用 eps=1e-6/atol=0；Block、GPT 这种整机级模块用 eps=1e-5 + atol=1e-8。
# 理由和 tests/test_model.py 顶部那条注释一样：注意力里 q/k/bias 的梯度常常只有 1e-6 量级
# （多个大数相减的抵消结果），而差分的舍入噪声约 ulp(目标)/eps ≈ 1e-11，
# 相对误差看着像 1e-3、绝对误差只有 1e-9。1e-8 已经是量级上的让步 ——
# 真写错 backward 的话误差是 1e-2 到 1e-1 量级，不会被这条放过。
GD = dict(eps=1e-5, tol=1e-4, atol=1e-8)


def make_cfg(vocab: int) -> Config:
    return Config(vocab=vocab, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, ctx=CTX)


def train(model, opt, batcher, steps, rng, log_every=0, prefix="训练", fixed_batch=None, curve=None):
    """训练循环就六步，顺序写死在这里：zero_grad → 取批次 → loss_backward → 裁剪 → step。

    换了顺序不会报错，但会用上一步的梯度更新、或者把清零当成信号 —— 两种都白训。
    每步的 lr 由 `lr_at` 给（warmup + cosine），不是构造优化器时那个常数。
    """
    curve = curve if curve is not None else []
    for i in range(steps):
        opt.zero_grad()
        x = fixed_batch if fixed_batch is not None else batcher(B, rng)[0]
        curve.append(model.loss_backward(x))
        clip_grad_norm(model, 1.0)
        opt.step(lr_at(i, steps, LR, warmup=WARMUP))
        if log_every:
            print_progress(i + 1, steps, every=log_every, prefix=prefix)
    return curve


# ------------------------------------------------------------------ 1) 分词
def check_tokenizer(rep: Report, tok: CharTokenizer):
    rep.group("分词 · 往返一致与词表口径")
    ids = tok.encode_array(TEXT)
    tr, va = train_val_texts(TEXT, 0.15)
    rt_tr = tok.decode(tok.encode(tr)) == tr
    rt_va = tok.decode(tok.encode(va)) == va
    rep.check("往返一致 decode(encode(x)) == x", rt_tr and rt_va,
              f"train {len(tr)} 字符、val {len(va)} 字符，两处都要逐字符相等")
    rep.check("词表口径 vocab == 语料去重字符数", tok.vocab_size == len(set(TEXT)),
              f"vocab={tok.vocab_size} · 语料里不同字符 {len(set(TEXT))} 个 · id 范围 "
              f"[{int(ids.min())}, {int(ids.max())}]")
    rep.check("编码结果全在词表内", int(ids.min()) >= 0 and int(ids.max()) < tok.vocab_size,
              f"共 {ids.size} 个 id，最大 {int(ids.max())} < vocab {tok.vocab_size}")

    probe = TEXT + "\t" + "【"                       # 两个都不在词表里
    n_kept = len(tok.encode(probe))
    try:
        tok.encode(probe, drop_unknown=False)
        raised = False
    except KeyError:
        raised = True
    rep.check("语料外的字符按契约处理", n_kept == ids.size and raised,
              f"drop_unknown=True 时丢掉 2 个陌生字符、长度回到 {n_kept}；"
              f"drop_unknown=False 抛 KeyError：{raised}")
    rep.info("词表长这样", repr(tok.chars()))


# ------------------------------------------------------------------ 2) 数据
def check_dataset(rep: Report, tok: CharTokenizer, bat: Batcher):
    rep.group("数据 · 窗口形状、移位关系、train/val 不泄漏")
    rng = np.random.default_rng(SEED)
    x, y = bat(B, rng)
    rep.check("批次形状 (B, ctx)", x.shape == (B, CTX) and y.shape == (B, CTX),
              f"x {tuple(x.shape)} · y {tuple(y.shape)}")
    rep.check("批次 id 全在词表内", int(x.min()) >= 0 and int(x.max()) < tok.vocab_size,
              f"min {int(x.min())} · max {int(x.max())} < vocab {tok.vocab_size}")
    rep.check("移位关系 y[:, :-1] == x[:, 1:]", np.array_equal(y[:, :-1], x[:, 1:]),
              "y 的第 t 格就是 x 的第 t+1 格 —— 这句话写成数据就是这个样子；"
              "如果 y 和 x 同位置相等，那就是在预测自己")

    x2, _ = bat(B, np.random.default_rng(SEED + 1))
    x3, _ = bat(B, np.random.default_rng(SEED + 1))
    rep.check("随机窗口真的随机且可复现", (not np.array_equal(x, x2)) and np.array_equal(x2, x3),
              "换 rng 要换窗口（否则每一步吃同一批），同 rng 要同窗口（否则日志不可复现）")

    tr, va = train_val_texts(SPLIT_TEXT, 0.15)
    tr_lines = {ln for ln in tr.split("\n") if ln.strip()}
    va_lines = [ln for ln in va.split("\n") if ln.strip()]
    leak = [ln for ln in va_lines if ln in tr_lines]
    cut_ok = tr.endswith("\n") and "\n" in va
    rep.check("train/val 按行切、不重叠", (not leak) and len(va_lines) > 0 and cut_ok,
              f"{len(SPLIT_TEXT.split(chr(10)))} 行唯一文本 · train {len(tr_lines)} 行 / val {len(va_lines)} 行"
              f" · 重叠 {len(leak)} 行 · train 以换行结尾：{cut_ok}"
              "（按字符数直接砍会把一句话劈成两半，val 就成了送分题）")
    rep.info("token 流长度",
             f"train {token_stream(tr, tok).size} · val {token_stream(va, tok).size}"
             f"（都要 > ctx+1 = {CTX + 1} 才切得出窗口）")


# ------------------------------------------------------------------ 3) 整机梯度
def check_model(rep: Report, cfg: Config, rng):
    rep.group("整机梯度 · Block 残差 / 权重共享 / 参数量")
    tiny = Config(vocab=cfg.vocab, d_model=16, n_layer=2, n_head=2, ctx=8)
    x_in = rng.standard_normal((2, 6, tiny.d_model))

    blk = Block(tiny, rng=rng)
    ok, worst, detail = gradcheck(blk, [x_in], label="Block", **GD)
    rep.check("Block 反向正确", ok, f"最大相对误差 {worst:.2e}（阈值 1e-4 + 绝对容差 1e-8）")
    if not ok:
        print(detail)

    # 残差恒等：把两个「残差出口」的权重清零，这一层应当什么都不做
    probe = Block(tiny, rng=np.random.default_rng(3))
    probe.attn.proj.W.data[...] = 0.0
    probe.mlp.fc2.W.data[...] = 0.0
    xin = rng.standard_normal((2, 5, tiny.d_model))
    same_fwd = np.array_equal(probe(xin), xin)
    g = rng.standard_normal(xin.shape)
    same_bwd = np.array_equal(probe.backward(g), g)
    rep.check("残差出口置零后 Block 是恒等映射", same_fwd and same_bwd,
              f"forward 逐比特等于输入：{same_fwd} · backward 把梯度原样交出：{same_bwd}"
              "（forward 漏一次 +x，或 backward 漏一路 +g，都会在这儿露馅）")

    # tied 输出头：单独 gradcheck + 「梯度会累加」
    V, C = 7, 5
    w = Parameter(rng.standard_normal((V, C)) * 0.3, name="tok.weight")
    head = TiedHead(w)
    hx = rng.standard_normal((2, 3, C))
    ok2, worst2, detail2 = gradcheck(head, [hx], label="TiedHead", **GD)
    rep.check("tied 输出头反向正确", ok2, f"最大相对误差 {worst2:.2e}")
    if not ok2:
        print(detail2)

    head.zero_grad()
    out = head(hx)
    g_out = rng.standard_normal(out.shape)
    head.backward(g_out)
    first = w.grad.copy()
    head.backward(g_out)
    moved0 = float(np.abs(first).max())
    doubled = bool(np.allclose(w.grad, 2.0 * first))
    rep.check("tied 的两路梯度是累加不是覆盖", doubled and moved0 > 0,
              f"再跑一次 backward 后 weight.grad 翻倍：{doubled}（第一次最大梯度 {moved0:.4g}）"
              "—— 写成 = 的话，整机的 tok.weight 会少掉查表那一半")

    model = GPT(cfg, rng=rng)
    ids = rng.integers(0, cfg.vocab, size=(2, 6))
    ok3, worst3, detail3 = gradcheck(model, [ids], label="GPT", **GD)
    rep.check("GPT 整机反向正确(含权重共享)", ok3, f"最大相对误差 {worst3:.2e}")
    if not ok3:
        print(detail3)

    model.zero_grad()
    model.loss_backward(ids)
    dead = [n for n, p in model.named_parameters() if not np.any(p.grad)]
    rep.check("无死参数", not dead,
              f"{len(model.parameters())} 个张量全部收到非零梯度" if not dead
              else f"梯度恒为 0 的：{dead}")

    per_layer = 12 * cfg.d_model ** 2 + 10 * cfg.d_model
    expect = cfg.vocab * cfg.d_model + cfg.ctx * cfg.d_model + cfg.n_layer * per_layer + cfg.d_model
    got = model.n_params()
    rep.check("参数量与手算一致", got == expect,
              f"手算 {expect:,} = 词嵌入 V·d({cfg.vocab * cfg.d_model:,}) + 位置嵌入 ctx·d"
              f"({cfg.ctx * cfg.d_model:,}) + {cfg.n_layer}×(12d²+10d) + 末端 norm d · 实测 {got:,}")
    rep.check("输出头复用的是词嵌入矩阵", model.head.weight is model.tok.weight,
              "同一个 Parameter 对象：更新一次、存一份、梯度两路相加")
    rep.info("配置", cfg.describe())
    return model


# ------------------------------------------------------------------ 4) 训练
def check_training(rep: Report, cfg: Config, tok, tr_bat, va_bat):
    rep.group("训练 · 能学，而且只在能学的地方学")
    rng = np.random.default_rng(SEED)
    ln_v = math.log(cfg.vocab)

    # 4.1 初始 loss：随机初始化时必须正好落在 ln V
    probe = GPT(cfg, rng=np.random.default_rng(42))
    x0, _ = tr_bat(B, np.random.default_rng(1))
    init_loss = probe.loss_backward(x0)
    probe.zero_grad()
    rep.close("初始 loss ≈ ln V", init_loss, ln_v, 0.2,
              note=f"ln({cfg.vocab})={ln_v:.3f}；差得远说明移位或归一化有问题")

    # 4.2 单批过拟合：整个项目最重要的单条诊断
    fx, _ = tr_bat(B, np.random.default_rng(3))
    m_of = GPT(cfg, rng=np.random.default_rng(2))
    o_of = AdamW(m_of, lr=LR, weight_decay=0.0)
    print("\n第 1 步：只喂同一批数据，看它能不能背下来（过拟合测试）")
    c_of = train(m_of, o_of, tr_bat, OVERFIT_STEPS, rng, log_every=50,
                 prefix="单批过拟合", fixed_batch=fx)
    print()
    print(sparkline(c_of, label="loss(单批过拟合)"))
    rep.curve("单批过拟合 loss", c_of, note=f"同一批数据 {OVERFIT_STEPS} 步：背不下来 = 反向有 bug")
    of_final = float(np.mean(c_of[-10:]))
    rep.below("单批过拟合最终 loss", of_final, 0.05,
              note="反向、优化器、移位三件事同时成立才做得到（阶段 8 的路线图把它当第一诊断）")

    # 4.3 真训练
    print(f"\n第 2 步：随机窗口训 {STEPS} 步（这才是「训练」）")
    model = GPT(cfg, rng=np.random.default_rng(1))
    opt = AdamW(model, lr=LR, weight_decay=0.0)
    c_main = train(model, opt, tr_bat, STEPS, rng, log_every=100, prefix="模式语料")
    print()
    print(sparkline(c_main, label=f"loss(模式语料 {STEPS} 步)"))
    rep.curve("模式语料训练 loss", c_main, note=f"{STEPS} 步 · lr {LR} · warmup {WARMUP}")
    final = float(np.mean(c_main[-20:]))
    val = evaluate(model, va_bat, B, 6, np.random.default_rng(9))
    rep.below("训练 loss 掉到 ln V 的三分之一", final, ln_v / 3.0,
              note=f"末段均值 {final:.3f} vs ln V {ln_v:.3f}")
    rep.above("下降倍数 ≥ 5×", c_main[0] / max(final, 1e-9), 5.0,
              note=f"{c_main[0]:.3f} → {final:.3f}")
    rep.below("验证 loss（按行切出来、没训过的行）", val, 0.4 * ln_v,
              note=f"val {val:.3f} vs ln V {ln_v:.3f}；评测走 evaluate()，不会污染 .grad")
    rep.soft("train/val 没有明显裂缝", abs(val - final) < 0.5,
             f"train {final:.3f} · val {val:.3f}（差 {val - final:+.3f}）"
             "—— 1.4K 字符的小语料本来就背得下来，这条只是提醒你别把背下来当学会")

    # 4.4 贪心看下一字符（只用 argmax，不经过阶段 7 的采样器）
    ids_all = tok.encode_array(TEXT)
    hits = tot = 0
    for s in range(0, ids_all.size - CTX - 1, 4):
        lg = model.forward(ids_all[s:s + CTX][None])
        hits += int(int(np.argmax(lg[0, -1])) == int(ids_all[s + CTX]))
        tot += 1
    acc = hits / max(tot, 1)
    rep.above("贪心下一字符准确率", acc, 0.70,
              note=f"{hits}/{tot} 个位置；只用 argmax，所以这一步还没有依赖 sampler")

    # 4.5 对照组：随机数据学不动 —— 移位写错时这条会红
    print("\n第 3 步：对照组。同样的模型和步数，数据换成随机字符（真正不可预测）")
    rid = np.random.default_rng(5).integers(0, cfg.vocab, size=3000)
    rb = Batcher(rid, CTX)
    m_r = GPT(cfg, rng=np.random.default_rng(4))
    o_r = AdamW(m_r, lr=LR, weight_decay=0.0)
    c_r = train(m_r, o_r, rb, RANDOM_STEPS, np.random.default_rng(11), log_every=50, prefix="随机数据")
    print()
    rep.curve("随机数据 loss（对照组）", c_r, note="下一字符不可预测 → loss 必须停在 ln V 附近")
    rnd_final = float(np.mean(c_r[-10:]))
    rep.above("随机数据学不动（反作弊判据）", rnd_final, 0.9 * ln_v,
              note=f"末段均值 {rnd_final:.3f}，下限 {0.9 * ln_v:.3f}；它要是掉下去了，"
                   "说明模型在预测自己而不是下一个字符")
    return model


# ------------------------------------------------------------------ 5) 优化器
def check_optimizer(rep: Report):
    rep.group("优化器 · AdamW 单步 / 梯度裁剪 / 学习率调度")
    # 5.1 单步幅度 = lr（关掉 weight decay，才看得出 Adam 本身）
    p = Parameter(np.zeros((4, 3)) + 0.5, name="W")
    o = AdamW([p], lr=1e-2, weight_decay=0.0, beta1=0.9, beta2=0.95)
    p.grad[...] = 3.0
    before = p.data.copy()
    o.step()
    moved = float(np.max(np.abs(p.data - before)))
    rep.close("AdamW 单步幅度 = lr", moved, 1e-2, 1e-6,
              note=f"梯度恒定时 m̂/√v̂ = 1，所以第一步正好走 lr；忘了 1-β^t 修正只有 ≈0.1·lr，"
                   f"实测 {moved:.6f}")

    # 5.2 weight decay 只作用于没被豁免的参数
    w = Parameter(np.zeros((4, 3)) + 0.5, name="W2")
    b = Parameter(np.zeros((4,)) + 0.5, name="bias", wd_exempt=True)
    o2 = AdamW([w, b], lr=1e-2, weight_decay=0.1)
    w.grad[...] = 3.0
    b.grad[...] = 3.0
    bw, bb = w.data.copy(), b.data.copy()
    o2.step()
    mw = float(np.max(np.abs(w.data - bw)))
    mb = float(np.max(np.abs(b.data - bb)))
    rep.check("weight decay 只作用于非豁免参数", abs(mb - 1e-2) < 1e-9 and mw > 1e-2,
              f"bias 恰好走 lr（{mb:.6f}，没被衰减）· 权重多走了一份 lr·λ·w = 1e-2×0.1×0.5"
              f"（{mw:.6f}）")

    # 5.3 梯度为 0 时不许漂移
    idle = Parameter(np.ones((3,)) * 2.0, name="idle")
    o3 = AdamW([idle], lr=1e-2, weight_decay=1e-2)
    o3.step()
    drift = float(np.max(np.abs(idle.data - 2.0)))
    rep.check("梯度为 0 时 AdamW 只衰减不推进", abs(drift - 2.0 * 1e-2 * 1e-2) < 1e-9,
              f"m=v=0 → 更新量 0/(0+eps)=0；实测最大变化 {drift:.3e}（正好等于衰减项）")

    # 5.4 梯度裁剪
    a = Parameter(np.zeros((2, 2)), name="a")
    b2 = Parameter(np.zeros((6,)), name="b")
    a.grad[...] = 3.0
    b2.grad[...] = -4.0
    total = float(np.sqrt(4 * 9.0 + 6 * 16.0))
    got = clip_grad_norm([a, b2], 1.0)
    scale = 1.0 / total
    after = float(np.sqrt(np.sum(a.grad ** 2) + np.sum(b2.grad ** 2)))
    same_dir = bool(np.allclose(a.grad, 3.0 * scale) and np.allclose(b2.grad, -4.0 * scale))
    rep.close("clip_grad_norm 返回裁剪前的范数", got, total, 1e-12,
              note=f"4 个 3.0 和 6 个 -4.0：√(4·9+6·16) = {total:.4f}")
    rep.close("裁剪后全局范数 = 上限", after, 1.0, 1e-12, note="只缩短，不改变方向")
    rep.check("裁剪对所有张量乘同一个系数", same_dir,
              f"a → {a.grad[0, 0]:.6f}（3×{scale:.6f}）· b → {b2.grad[0]:.6f}（-4×{scale:.6f}）")
    small = Parameter(np.zeros((2,)), name="small")
    small.grad[...] = 0.1
    got2 = clip_grad_norm([small], 5.0)
    rep.check("范数没超上限时一点都不动", bool(np.allclose(small.grad, 0.1)) and abs(got2 - math.sqrt(0.02)) < 1e-12,
              f"返回 {got2:.4f}，梯度保持原样")

    # 5.5 学习率调度
    total_steps, peak, warm = 1000, 6e-3, 100
    sched = [lr_at(i, total_steps, peak, warmup=warm) for i in range(total_steps + 5)]
    up = all(sched[i] <= sched[i + 1] + 1e-15 for i in range(warm - 1))
    down = all(sched[i] >= sched[i + 1] - 1e-15 for i in range(warm, total_steps))
    rep.check("lr 调度：warmup 递增、之后递减", up and down,
              f"前 {warm} 步单调不降：{up} · 之后单调不升：{down}")
    rep.below("lr 全程不超过 peak", max(sched), peak * 1.0000001,
              note=f"最大 {max(sched):.5g}，peak {peak:.5g}")
    rep.close("warmup 结束时正好到 peak", sched[warm - 1], peak, 1e-12,
              note=f"lr_at({warm - 1}) = {sched[warm - 1]:.6g}")
    rep.close("最后一步收到 peak 的 10%", sched[total_steps], peak * 0.1, 1e-9,
              note="不归零：后期还要留一点步长继续微调")
    rep.check("第 0 步的 lr 不是 0", sched[0] > 0,
              f"lr_at(0) = {sched[0]:.3g}（写成 step/warmup 就正好是 0，白走一步）")
    rep.info("lr 曲线抽样", " ".join(f"{sched[i]:.4f}" for i in (0, 50, 100, 300, 700, 1000)))


def main() -> int:
    rep = Report("阶段 5 · 完整 GPT + 训练循环",
                 f"字符级语料 {len(TEXT)} 个字符 · B={B} ctx={CTX} d={D_MODEL} L={N_LAYER} · {STEPS} 步")
    rng = np.random.default_rng(SEED)
    tok = CharTokenizer(TEXT)
    tr, va = train_val_texts(TEXT, 0.15)
    cfg = make_cfg(tok.vocab_size)
    tr_bat = Batcher(token_stream(tr, tok), CTX)
    va_bat = Batcher(token_stream(va, tok), CTX)

    check_tokenizer(rep, tok)
    check_dataset(rep, tok, tr_bat)
    check_model(rep, cfg, rng)
    model = check_training(rep, cfg, tok, tr_bat, va_bat)
    check_optimizer(rep)

    # 最后给一眼「它写出来像什么」：只用 argmax，阶段 7 才谈采样
    pad_id = int(np.argmax(np.bincount(tok.encode(TEXT))))
    prompt = tok.encode_array("the cat sat on the ")
    win = np.concatenate([np.full((1, max(0, CTX - prompt.size)), pad_id, dtype=np.int64),
                          prompt[None].astype(np.int64)], axis=1)[:, -CTX:]
    gen = [int(np.argmax(model.forward(win)[0, -1]))]
    for _ in range(23):
        win = np.concatenate([win[:, 1:], np.array([[gen[-1]]], dtype=np.int64)], axis=1)
        gen.append(int(np.argmax(model.forward(win)[0, -1])))
    rep.info("贪心续写 24 个字符", repr("the cat sat on the " + tok.decode(gen)))

    passed = rep.done()
    print("\n这一步你确认了四件事：")
    print("  1) 整台机器的梯度逐个参数都对得上有限差分，包括权重共享那条双路累加。")
    print("  2) 同一份配置、同样的步数：模式数据学得动，随机数据学不动 —— 这才叫「预测下一个」。")
    print("  3) 单批过拟合把 loss 压到 0.05 以下：反向、优化器、移位三件事同时成立才做得到。")
    print("  4) AdamW 单步幅度、梯度裁剪、lr 调度各自符合定义，而不是「看着在降」。")
    print("\n下一步（阶段 6）：把它接到真语料 data/corpus.txt 上训，并把权重写进 MGPT 检查点。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
