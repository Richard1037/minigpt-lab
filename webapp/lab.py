"""调参试验台：把几个输入做成旋钮，跑**你写的实现**，但结论不算验收。

为什么要和判据分开：判据那几条阈值（线性模型 MSE ≥ 0.05、MLP ≤ 6e-3、好 20 倍）
是照**固定参数**定的。参数一可调，"过了判据"这句话就失去意义 ——
你可以把 n_hidden 调到 64 让 MSE 好看，也可以把 lr 调到 1.5 让一个正确的实现"失败"。
所以这里只报观察值（最终 MSE、和对照的比值、曲线形状），不打分、不写进度。

跑法（网站用 MGPT_PKG 指向你的提交副本，和判据同一套机制）：
    python webapp/lab.py s02 '{"n_hidden":8,"activate":true,"lr":0.1,"decay":3,"n_points":12,"x_max":3}'
输出：一行 @@MGPT-JSON@@{...}，字段和 toolkit.Report 的机器可读通道一致
      （rows / curves / title），网站那套卡片和曲线渲染直接复用。

新增阶段只要在这里加 KNOBS[stage] 和 RUN[stage]，前端按 KNOBS 自动长表单。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

JSON_MARK = "@@MGPT-JSON@@"          # 必须和 src/minigpt/toolkit.py 一致

# ------------------------------------------------------------------ 旋钮定义
# type: range（整数/浮点滑杆）、logrange（对数滑杆，lr 这种跨三个数量级的用它的）、bool
KNOBS: dict[str, list[dict]] = {
    "s01": [
        dict(key="lr", label="学习率 lr", type="logrange", min=0.001, max=1.5, default=0.2,
             why="每步挪多远。稳定区间由数据的 mean(2x²) 决定，不是魔法数字"),
        dict(key="n_points", label="采样点数", type="range", min=6, max=200, step=1, default=100,
             why="点越多，单步梯度越接近真实均值"),
        dict(key="x_max", label="x 区间 ±", type="range", min=0.5, max=6, step=0.5, default=1.0,
             why="区间变宽 = 梯度变大 = 等效于把 lr 调大"),
    ],
    "s02": [
        dict(key="n_hidden", label="隐藏层个数 n_hidden", type="range", min=1, max=24, step=1, default=8,
             why="tanh 的台阶数，也就是模型的容量"),
        dict(key="activate", label="中间加 tanh", type="bool", default=True,
             why="关掉它：线性∘线性还是线性，加深不涨容量"),
        dict(key="lr", label="学习率 lr", type="logrange", min=0.001, max=1.5, default=0.1,
             why="太小走不到、太大来回弹；和 decay 一起看"),
        dict(key="decay", label="lr 衰减 decay", type="range", min=0, max=10, step=0.5, default=3.0,
             why="lr/(1+decay·t/T)：末期步长变细，才能贴到最低点"),
        dict(key="n_points", label="采样点数", type="range", min=6, max=60, step=1, default=12,
             why="点太稀，sin 的弯根本看不见"),
        dict(key="x_max", label="x 区间 ±", type="range", min=1, max=6, step=0.5, default=3.0,
             why="超过 ±π 就要拟合不止一个弯，容量需求翻倍"),
    ],
    "s03a": [
        dict(key="n", label="样本数 N", type="range", min=1, max=8, step=1, default=3,
             why="x 的行数。dW、db 是这 N 行贡献的总和 —— 调 N 看哪些数字跟着变、哪些不变"),
        dict(key="dim_in", label="输入维数 in", type="range", min=1, max=6, step=1, default=3,
             why="W 的列数；x 是 (N, in)"),
        dict(key="dim_out", label="输出维数 out", type="range", min=1, max=6, step=1, default=2,
             why="W 的行数；y 是 (N, out)"),
        dict(key="seed", label="随机种子", type="range", min=0, max=30, step=1, default=0,
             why="换种子 = 换一组 x、W、g。数字全变，形状规则一条都不变"),
        dict(key="xscale", label="输入幅度 xscale", type="logrange", min=0.01, max=10, default=1.0,
             why="x 越大 y 和 dW 跟着越大：盯着数字看，别背公式"),
    ],
    "s03b": [
        dict(key="b", label="序列条数 B", type="range", min=1, max=4, step=1, default=2,
             why="几条独立序列；norm 和 MLP 都在 (B·T) 个位置上各算各的"),
        dict(key="t", label="每条 token 数 T", type="range", min=1, max=6, step=1, default=3,
             why="位置数。RMSNorm 的 r 是 (B,T,1)：每个位置一个数"),
        dict(key="c", label="通道数 C", type="range", min=1, max=6, step=1, default=4,
             why="每个位置几维；norm 沿这根轴求均方"),
        dict(key="hidden", label="MLP 隐藏维 H", type="range", min=1, max=8, step=1, default=6,
             why="中间先升维再降回 C：升维那一步的矩阵就是 fc1"),
        dict(key="seed", label="随机种子", type="range", min=0, max=30, step=1, default=0,
             why="换一组 x 和权重，规则不变"),
    ],
    "s03c": [
        dict(key="b", label="序列条数 B", type="range", min=1, max=4, step=1, default=2,
             why="几条序列；loss 是全部 B·T 个位置的平均"),
        dict(key="t", label="每条 token 数 T", type="range", min=1, max=6, step=1, default=3,
             why="位置数；每个位置独立做一次 softmax"),
        dict(key="v", label="词表大小 V", type="range", min=2, max=10, step=1, default=5,
             why="softmax 的竞争者个数。logits 全 0 时 loss 必须正好是 ln(V)"),
        dict(key="sharp", label="logit 尖锐度", type="logrange", min=0.1, max=10, default=1.0,
             why="乘在 logits 上：越大概率越集中到最大的那个词，梯度越接近 (one-hot 差)/N"),
        dict(key="seed", label="随机种子", type="range", min=0, max=30, step=1, default=0,
             why="换一组 logits 和 targets"),
    ],
    "s04": [
        dict(key="b", label="序列条数 B", type="range", min=1, max=4, step=1, default=2,
             why="注意力对每条序列独立算，B 只是把它们并排放"),
        dict(key="t", label="每条 token 数 T", type="range", min=2, max=8, step=1, default=5,
             why="概率图是 T×T：第 i 行是「第 i 个 token 在看谁」。调大它看三角掩码怎么长"),
        dict(key="c", label="通道数 C", type="range", min=4, max=12, step=2, default=8,
             why="每个 token 的向量长度，也是 q/k/v/proj 四个矩阵的边长"),
        dict(key="h", label="头数 H", type="range", min=1, max=4, step=1, default=2,
             why="C 被切成 H 份，每份 hd=C/H。H 不能整除 C 时会自动退到能整除的最大值"),
        dict(key="sharpen", label="q 的幅度（注意力变尖）", type="logrange", min=0.2, max=6.0, default=1.0,
             why="把 W_q 整体乘这个数：分数变大 → softmax 变硬 → 概率图从一片糊变成一个点"),
        dict(key="seed", label="随机种子", type="range", min=0, max=30, step=1, default=0,
             why="换一组权重和输入"),
    ],
    "s05": [
        dict(key="vocab", label="词表 V", type="range", min=4, max=16, step=1, default=9,
             why="logits 的最后一维；随机初始化时 loss 应该正好落在 ln V 附近"),
        dict(key="d_model", label="通道数 C", type="range", min=4, max=16, step=2, default=8,
             why="主干宽度，整条动线上每一层的向量长度"),
        dict(key="n_layer", label="Block 数 L", type="range", min=1, max=4, step=1, default=2,
             why="形状不变（还是 (B,T,C)），只是同一套动线重复 L 遍；resid 缩放除的是 √(2L)"),
        dict(key="n_head", label="头数 H", type="range", min=1, max=4, step=1, default=2,
             why="每个 Block 里注意力的头数，必须整除 C"),
        dict(key="ctx", label="上下文长度 ctx", type="range", min=3, max=12, step=1, default=6,
             why="位置嵌入只有 ctx 行 —— T 超过它必须被拦住"),
        dict(key="t", label="实际长度 T", type="range", min=1, max=12, step=1, default=5,
             why="超过 ctx 的那一档就是用来触发拦截的"),
        dict(key="resid_scale", label="残差出口缩放", type="bool", default=True,
             why="GPT-2 那个初始化：attn.proj 和 mlp.fc2 除以 √(2L)，深层残差主干才不会越走越飘"),
        dict(key="seed", label="随机种子", type="range", min=0, max=30, step=1, default=0,
             why="换一组权重"),
    ],
    "s06": [
        dict(key="lr", label="峰值学习率 lr", type="logrange", min=0.0003, max=0.3, default=0.02,
             why="warmup 爬到它，再 cosine 收到 frac·lr。太大发散、太小不动，是这一档最敏感的旋钮"),
        dict(key="iters", label="训练步数", type="range", min=20, max=400, step=10, default=120,
             why="步数太少还看不到收敛拐点"),
        dict(key="batch", label="批大小 B", type="range", min=1, max=16, step=1, default=8,
             why="一次喂几条窗口；梯度是这 B 条的平均，B 越小噪声越大"),
        dict(key="ctx", label="窗口长度 ctx", type="range", min=4, max=16, step=1, default=8,
             why="模型看到的 token 数，也是位置嵌入的行数"),
        dict(key="warmup", label="warmup 步数", type="range", min=0, max=60, step=2, default=10,
             why="前面线性爬到峰值：优化器刚起步时二阶矩还没攒起来，直接用大 lr 容易一脚踩空"),
        dict(key="clip", label="梯度裁剪 max_norm", type="logrange", min=0.1, max=10.0, default=1.0,
             why="全局 L2 超过它就整体等比缩小。调小 → 曲线更平但学不动；调大 → 尖峰露出来"),
        dict(key="wd", label="weight decay", type="range", min=0, max=0.3, step=0.01, default=0.1,
             why="解耦衰减，只作用在权重上（bias/norm 豁免）"),
    ],
    "s07": [
        dict(key="temperature", label="温度 τ", type="logrange", min=0.05, max=5.0, default=1.0,
             why="分数先除以 τ 再 softmax：τ→0 变成贪心，τ 大一片糊 → 用词变宽"),
        dict(key="top_k", label="top-k", type="range", min=0, max=12, step=1, default=0,
             why="只保留最高的 k 个；0 = 不剪"),
        dict(key="top_p", label="top-p（nucleus）", type="range", min=0.1, max=1.0, step=0.05, default=1.0,
             why="按概率降序累加到 p 为止，剩下的剪掉再归一化"),
        dict(key="penalty", label="重复惩罚", type="range", min=1.0, max=4.0, step=0.1, default=1.0,
             why=">1 时压掉已经出现过的词：正分除、负分乘"),
        dict(key="n_samples", label="采样次数", type="range", min=100, max=6000, step=100, default=2000,
             why="经验分布的柱子要靠次数堆出来，太少会抖"),
        dict(key="seed", label="随机种子", type="range", min=0, max=30, step=1, default=0,
             why="同一份 logits、同一个 rng 起点"),
    ],
}

# ------------------------------------------------------------------ 各阶段实跑
def run_s01(p: dict) -> dict:
    import numpy as np
    from minigpt.exercises import fit_by_autograd

    n, xm, true_a, true_b = int(p["n_points"]), float(p["x_max"]), 3.0, 1.0
    x = np.linspace(-xm, xm, n)
    y = true_a * x + true_b
    curve = fit_by_autograd(x, y, lr=float(p["lr"]), log_every=0)[2]
    rows = [
        dict(kind="info", mark="info", name="最终 MSE", detail=f"{curve[-1]:.4g}"),
        dict(kind="info", mark="info", name="下降倍数",
             detail=f"{curve[0] / max(curve[-1], 1e-300):.3g}×（判据要求 ≥ 1e3，用固定参数跑才算数）"),
        dict(kind="info", mark="info", name="发散/停滞",
             detail="loss 变 nan 或后半程不降 —— 看曲线就知道，不用等报错"),
    ]
    return _payload("阶段 1 · 拟合直线（试验台）", f"{n} 个点，x∈[±{xm}]，lr={p['lr']}",
                    rows, [dict(name="loss(自动微分)", values=curve,
                                note=f"目标 y={true_a}x+{true_b}")])


def run_s02(p: dict) -> dict:
    import numpy as np
    from minigpt.exercises import ScalarMLP, train_mlp

    n, xm = int(p["n_points"]), float(p["x_max"])
    xs, ys = np.linspace(-xm, xm, n), np.sin(np.linspace(-xm, xm, n))
    iters = 4000

    # 对照永远是"1 个隐藏层 + 不加激活"，参数不跟着你调 —— 否则比值没有意义
    ctrl_model = ScalarMLP(n_hidden=1, activate=False)
    ctrl = train_mlp(ctrl_model, xs, ys, iters=iters, lr=0.1, log_every=0)
    model = ScalarMLP(n_hidden=int(p["n_hidden"]), activate=bool(p["activate"]))
    mine = train_mlp(model, xs, ys, iters=iters, lr=float(p["lr"]),
                     decay=float(p["decay"]), log_every=0)
    mse_c, mse_m = ctrl[-1], mine[-1]
    tail = mine[len(mine) // 2:]
    n_param = len(model.parameters())        # 直接数，不背公式：n_in=1 时是 3n+1
    rows = [
        dict(kind="info", mark="info", name="这一组的最终 MSE", detail=f"{mse_m:.4g}"),
        dict(kind="info", mark="info", name="对照（线性）MSE", detail=f"{mse_c:.4g}"),
        dict(kind="info", mark="info", name="比对照好",
             detail=f"{mse_c / max(mse_m, 1e-300):.3g}×"),
        dict(kind="info", mark="info", name="后半程还在降吗",
             detail="在降 —— 继续跑或加大 lr 还有空间" if min(tail) < tail[0] * 0.98
                    else "基本平了 —— 再多种训练也救不了，这是容量/表达能力的墙"),
        dict(kind="info", mark="info", name="参数量", detail=f"{n_param} 个（n_hidden={p['n_hidden']}）"),
    ]
    return _payload("阶段 2 · 拟合 sin(x)（试验台）",
                    f"{n} 个点，x∈[±{xm}]，n_hidden={p['n_hidden']}，"
                    f"{'tanh' if p['activate'] else '无激活'}，lr={p['lr']}，decay={p['decay']}",
                    rows, [dict(name="对照：线性模型", values=ctrl, note="固定参数，不跟着调"),
                           dict(name="你选的这一组", values=mine,
                                note="和对照同一条数据、同一个循环")])


def run_s03a(p: dict) -> dict:
    """Linear 试验台：同一组 (x, W, b, g)，你的 backward 和参考算法逐格对比。

    参考算法就是讲义 §2.5 那张表的四行 numpy 直写。热力图（mats 通道）里
    你的结果和参考并排放，哪一格不对一眼就能看到；rows 里给最大差和形状。
    """
    import numpy as np
    from minigpt.nn import Linear
    from minigpt.gradcheck import gradcheck

    n, di, do = int(p["n"]), int(p["dim_in"]), int(p["dim_out"])
    seed = int(p["seed"])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, di)) * float(p["xscale"])
    g = rng.standard_normal((n, do))

    lin = Linear(di, do, rng=rng)          # W、b 由你的实现自己初始化（rng 固定 → 可复现）
    W, b = lin.W.data, lin.b.data
    lin.zero_grad()
    y = np.asarray(lin.forward(x))
    dx = np.asarray(lin.backward(g))
    dW, db = lin.W.grad, lin.b.grad

    # 参考：§2.5 速查表逐行直写
    y_ref = x @ W.T + b
    dW_ref = g.T @ x
    db_ref = g.sum(axis=0)
    dx_ref = g @ W

    def diff(a, ref):
        a = np.asarray(a, dtype=np.float64)
        if a.shape != ref.shape:
            return None
        return float(np.max(np.abs(a - ref)))

    def row(name, mine, ref, shape_txt):
        d = diff(mine, ref)
        if d is None:
            detail = (f"形状 {tuple(np.asarray(mine).shape)}，应为 {tuple(ref.shape)} —— 先看形状再看数值")
            mark = "FAIL"
        elif d < 1e-10:
            detail, mark = f"和参考完全一致（最大差 {d:.1e}），形状 {shape_txt}", "PASS"
        else:
            detail, mark = f"最大差 {d:.3g}（形状 {shape_txt} 对了，数值不对）", "FAIL"
        return dict(kind="check", mark=mark, name=name, detail=detail)

    rows = [
        row("y = x @ W.T + b", y, y_ref, f"({n},{do})"),
        row("dW = g.T @ x", dW, dW_ref, f"({do},{di}) 和 W 同形"),
        row("db = g.sum(axis=0)", db, db_ref, f"({do},) 和 b 同形"),
        row("dx = g @ W", dx, dx_ref, f"({n},{di}) 和 x 同形"),
    ]

    # gradcheck：数值扰动 vs 你的解析梯度（和判据同款调用，只是参数跟着旋钮走）
    try:
        lin2 = Linear(di, do, rng=np.random.default_rng(seed + 100))
        ok, worst, detail = gradcheck(lin2, [x], eps=1e-5, atol=1e-9, label="Linear")
        rows.append(dict(kind="check", mark="PASS" if ok else "FAIL", name="gradcheck（数值 vs 解析）",
                         detail=f"最差相对差 {worst:.2e} —— {detail}"))
    except NotImplementedError as exc:
        rows.append(dict(kind="soft", mark="注意", name="gradcheck", detail=str(exc)))

    def mat(name, arr, note):
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        return dict(name=name, note=note, shape=list(arr.shape),
                    values=[[float(v) for v in r] for r in arr])

    mats = [
        mat("x（输入）", x, f"({n},{di})"),
        mat("W（权重）", W, f"({do},{di})"),
        mat("g（上游梯度）", g, f"({n},{do})"),
        mat("y（你的 forward）", y, f"参考最大差 {diff(y, y_ref) if diff(y, y_ref) is not None else '形状不对'}"),
        mat("dW（你的 W.grad）", dW, "g.T @ x 的结果"),
        mat("db（你的 b.grad）", db, "g.sum(axis=0) 的结果"),
        mat("dx（你的返回值）", dx, "g @ W 的结果"),
    ]
    return _payload("阶段 3a · Linear 前向/反向（试验台）",
                    f"N={n}，in={di}，out={do}，seed={seed}，xscale={p['xscale']}",
                    rows, [], mats=mats)


def _gelu_ref(x):
    import numpy as np
    K = np.sqrt(2.0 / np.pi)
    A = 0.044715
    inner = K * (x + A * x ** 3)
    t = np.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    dy = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * K * (1.0 + 3.0 * A * x * x)
    return y, dy


def run_s03b(p: dict) -> dict:
    """RMSNorm + GELU + MLP 试验台：你的三层 vs 逐元素参考式，全部摊成 (B·T, ·) 画出来。"""
    import numpy as np
    from minigpt.nn import GELU, MLP, RMSNorm
    from minigpt.gradcheck import gradcheck

    B, T, C, H = int(p["b"]), int(p["t"]), int(p["c"]), int(p["hidden"])
    seed = int(p["seed"])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, T, C))
    g = rng.standard_normal((B, T, C))

    def flat(a):
        a = np.asarray(a, dtype=np.float64)
        return a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(1, -1)

    rows, mats = [], []

    # ---- RMSNorm：参考式 r = sqrt(mean(x²)+eps)，y = w·x/r
    rn = RMSNorm(C)
    w = rn.w.data
    r = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + rn.eps)
    u = x / r
    y_ref = w * u
    gw = g * w
    gx_ref = (gw - (gw * u).mean(axis=-1, keepdims=True) * u) / r
    dw_ref = (g * u).sum(axis=(0, 1))
    rn.zero_grad()
    y = np.asarray(rn.forward(x))
    gx = np.asarray(rn.backward(g))
    dw = rn.w.grad
    rows += [_cmp("RMSNorm y = w·x/r", y, y_ref, f"({B},{T},{C})"),
             _cmp("RMSNorm dx", gx, gx_ref, f"({B},{T},{C}) 和 x 同形"),
             _cmp("RMSNorm dw", dw, dw_ref, f"({C},) 和 w 同形")]
    mats += [_mat("x（输入）", flat(x), f"({B}·{T},{C})"), _mat("r（每位置的均方根）", flat(r), f"({B}·{T},1)"),
             _mat("x̂ = x/r", flat(u), "每行的均方正好是 1"),
             _mat("y = w·x̂（你的 forward）", flat(y), f"({B}·{T},{C})"),
             _mat("dx（你的 backward）", flat(gx), "减掉'长度被锁住'那项才对")]

    # ---- GELU
    ge = GELU()
    yg, dyg = _gelu_ref(x)
    y2 = np.asarray(ge.forward(x))
    gx2 = np.asarray(ge.backward(g))
    rows += [_cmp("GELU y", y2, yg, f"({B},{T},{C})"), _cmp("GELU dx", gx2, g * dyg, f"({B},{T},{C})")]

    # ---- MLP：参考式用你自己的权重直写三条串联
    mp = MLP(C, H, rng=rng)
    W1, b1 = mp.fc1.W.data, mp.fc1.b.data
    W2, b2 = mp.fc2.W.data, mp.fc2.b.data
    x2 = x.reshape(-1, C)
    g2 = g.reshape(-1, C)
    h = x2 @ W1.T + b1
    a, dgelu = _gelu_ref(h)
    y3_ref = (a @ W2.T + b2).reshape(B, T, C)
    ga = g2 @ W2
    gh = ga * dgelu
    gx3_ref = (gh @ W1).reshape(B, T, C)
    mp.zero_grad()
    y3 = np.asarray(mp.forward(x))
    gx3 = np.asarray(mp.backward(g))
    rows += [_cmp("MLP y（三条串联）", y3, y3_ref, f"({B},{T},{C})"),
             _cmp("MLP dx", gx3, gx3_ref, f"({B},{T},{C})")]
    mats += [_mat("h = fc1(x)（升维后）", h, f"({B}·{T},{H})"), _mat("GELU(h)", a, "负的那半被压扁但不归零"),
             _mat("MLP y（你的 forward）", flat(y3), f"({B}·{T},{C})"),
             _mat("MLP dx（你的 backward）", flat(gx3), "反着走：fc2 → GELU → fc1")]

    try:
        mp2 = MLP(C, H, rng=np.random.default_rng(seed + 100))
        ok, worst, detail = gradcheck(mp2, [x], eps=1e-5, atol=1e-9, label="MLP")
        rows.append(dict(kind="check", mark="PASS" if ok else "FAIL", name="gradcheck（MLP 数值 vs 解析）",
                         detail=f"最差相对差 {worst:.2e} —— {detail}"))
    except NotImplementedError as exc:
        rows.append(dict(kind="soft", mark="注意", name="gradcheck", detail=str(exc)))

    return _payload("阶段 3b · RMSNorm / GELU / MLP（试验台）",
                    f"B={B}，T={T}，C={C}，H={H}，seed={seed}", rows, [], mats=mats)


def run_s03c(p: dict) -> dict:
    """CrossEntropy 试验台：softmax 概率和 logits 梯度逐格画出来，和参考式对比。"""
    import numpy as np
    from minigpt.gradcheck import gradcheck
    from minigpt.nn import CrossEntropy

    B, T, V = int(p["b"]), int(p["t"]), int(p["v"])
    seed = int(p["seed"])
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((B, T, V)) * float(p["sharp"])
    targets = rng.integers(0, V, size=(B, T))

    z = logits.reshape(-1, V)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=-1, keepdims=True)
    idx = np.arange(z.shape[0])
    tgt = targets.reshape(-1)
    loss_ref = float(-np.log(probs[idx, tgt]).mean())
    dlogits_ref = probs.copy()
    dlogits_ref[idx, tgt] -= 1.0
    dlogits_ref /= z.shape[0]
    dlogits_ref = dlogits_ref.reshape(B, T, V)
    onehot = np.zeros_like(probs)
    onehot[idx, tgt] = 1.0

    ce = CrossEntropy()
    loss = float(ce.forward(logits, targets))
    dlogits = np.asarray(ce.backward())

    zero = CrossEntropy()
    loss_zero = float(zero.forward(np.zeros((1, 1, V)), np.zeros((1, 1), dtype=int)))

    rows = [
        _cmp("loss（标量）", np.array([[loss]]), np.array([[loss_ref]]), "全部 B·T 个位置的平均"),
        dict(kind="check", mark="PASS" if abs(loss_zero - np.log(V)) < 1e-9 else "FAIL",
             name=f"logits 全 0 时 loss = ln(V) = {np.log(V):.4f}",
             detail=f"你算出 {loss_zero:.6f}" + ("" if abs(loss_zero - np.log(V)) < 1e-9 else " —— 差太多，softmax 那步不对")),
        dict(kind="check", mark="PASS" if np.allclose(probs.sum(axis=-1), 1.0) else "FAIL",
             name="每行概率和为 1", detail="softmax 的归一化（参考侧验证，和你的实现无关）"),
        _cmp("dlogits = (probs − one-hot)/N", dlogits, dlogits_ref, f"({B},{T},{V})"),
    ]
    try:
        ce2 = CrossEntropy()
        lg = rng.standard_normal((B, T, V))
        ok, worst, detail = gradcheck(ce2, [lg, targets], eps=1e-5, atol=1e-7, label="CrossEntropy")
        rows.append(dict(kind="check", mark="PASS" if ok else "FAIL", name="gradcheck（数值 vs 解析）",
                         detail=f"最差相对差 {worst:.2e} —— {detail}"))
    except NotImplementedError as exc:
        rows.append(dict(kind="soft", mark="注意", name="gradcheck", detail=str(exc)))

    mats = [
        _mat("logits（你的输入）", logits.reshape(-1, V), f"({B}·{T},{V})，乘过尖锐度 {p['sharp']}"),
        _mat("probs = softmax(logits)", probs, "每行和为 1；最大的那个词赢"),
        _mat("one-hot(targets)", onehot, "正确词是 1，其余 0"),
        _mat("dlogits（你的 backward）", dlogits.reshape(-1, V), "(probs − one-hot)/N"),
    ]
    return _payload("阶段 3c · CrossEntropy（试验台）",
                    f"B={B}，T={T}，V={V}，尖锐度={p['sharp']}，seed={seed}", rows, [], mats=mats)


def run_s04(p: dict) -> dict:
    """注意力试验台：你的 forward 和「用给你的零件直写的参考实现」并排画出来。

    参考侧只用已经给好的 _split_heads / _causal_mask / _softmax_rows 和三个 Linear，
    所以它不替你写 TODO-12，只是把目标摆出来 —— 你还没写的时候，这张图照样能看。
    概率图按头分开画：上三角是掩码区（灰格），行和恒为 1。
    """
    import numpy as np
    from minigpt.attention import CausalSelfAttention, _softmax_rows
    from minigpt.config import Config
    from minigpt.gradcheck import gradcheck

    B, T, C = int(p["b"]), int(p["t"]), int(p["c"])
    H = max(1, int(p["h"]))
    while C % H:                                  # 头数必须整除通道数：退到能整除的最大值
        H -= 1
    hd = C // H
    seed, sharpen = int(p["seed"]), float(p["sharpen"])
    rng = np.random.default_rng(seed)
    cfg = Config(d_model=C, n_head=H, vocab=9, ctx=T)
    att = CausalSelfAttention(cfg, rng=rng)
    att.q.W.data *= sharpen                       # 只动权重幅度：分数变大 → 概率图变尖
    x = rng.standard_normal((B, T, C))

    # 参考侧：完全用已经给好的零件拼，不含任何 TODO
    q = att.q.forward(x)
    k = att.k.forward(x)
    v = att.v.forward(x)
    qh, kh, vh = (att._split_heads(z, B, T) for z in (q, k, v))
    scores = qh @ kh.transpose(0, 1, 3, 2) * att.scale + CausalSelfAttention._causal_mask(T, x.dtype)
    probs_ref = _softmax_rows(scores)
    y_ref = att.proj.forward((probs_ref @ vh).transpose(0, 2, 1, 3).reshape(B, T, C))

    blocked, y, probs = None, None, probs_ref
    try:
        y = np.asarray(att(x))
        probs = np.asarray(getattr(att, "_probs", probs_ref))
    except NotImplementedError as exc:
        blocked = str(exc)
    except Exception as exc:                      # noqa: BLE001  形状错在这里比在判据里更难看
        blocked = f"{type(exc).__name__}: {exc}"

    rows = []
    if blocked:
        rows.append(dict(kind="soft", mark="注意", name="你的 forward 还没跑通",
                         detail=blocked + " —— 下面全是参考侧的图，写完再跑一次就能并排比"))
    else:
        rows.append(_cmp("y = att(x)", y, y_ref, f"({B},{T},{C})"))
        rows.append(_cmp("probs（缓存里的 _probs）", probs, probs_ref, f"({B},{H},{T},{T})"))
        dsum = float(np.max(np.abs(probs.sum(axis=-1) - 1.0)))
        rows.append(dict(kind="check", mark="PASS" if dsum < 1e-9 else "FAIL",
                         name="每行概率和为 1", detail=f"最大偏差 {dsum:.2e}"))
        upper = np.triu(np.ones((T, T), dtype=bool), k=1)
        leak = float(np.max(np.abs(probs * upper)))
        rows.append(dict(kind="check", mark="PASS" if leak == 0.0 else "FAIL",
                         name="因果性：上三角（j>i）精确为 0",
                         detail=f"最大泄漏 {leak:.3g}" + ("" if leak == 0.0 else " —— 在偷看未来")))
        try:
            ok, worst, detail = gradcheck(CausalSelfAttention(cfg, rng=np.random.default_rng(seed + 50)),
                                          [x], eps=1e-5, atol=1e-9)
            rows.append(dict(kind="check", mark="PASS" if ok else "FAIL",
                             name="gradcheck（反向全链）", detail=f"最差相对差 {worst:.2e} —— {detail}"))
        except NotImplementedError as exc:
            rows.append(dict(kind="soft", mark="注意", name="gradcheck（反向全链）", detail=str(exc)))

    upper_cells = np.triu(np.ones((T, T), dtype=bool), k=1)
    mats = []
    for h in range(H):
        mats.append(dict(name=f"probs · 头 {h}（第 i 行 = 第 i 个 token 在看谁）",
                         note=f"行和为 1；灰格是因果掩码，j>i 不许看。画的是第 0 条序列（共 {B} 条）",
                         shape=[T, T], scale="prob",
                         mask=upper_cells.astype(int).tolist(),
                         values=[[float(z) for z in row] for row in probs[0, h]]))
    mats.append(dict(name="分数 scores · 头 0（softmax 之前）",
                     note="q·k/√hd 的原始关注度，红正蓝负；灰格是被掩码遮住的那半（画成 −1e18 会把色阶压平）",
                     shape=[T, T], mask=upper_cells.astype(int).tolist(),
                     values=[[float(scores[0, 0, i, j]) if not upper_cells[i, j] else float("nan")
                              for j in range(T)] for i in range(T)]))
    mats.append(_mat("x（输入）", x.reshape(-1, C), f"({B}·{T}, {C})"))

    flow = [
        dict(name="输入 x", shape=(B, T, C), note="每个 token 一根 C 维向量"),
        dict(name="q, k, v = 三个 Linear", shape=(B, T, C), note="同一个 x 三次投影；k 没有 bias"),
        dict(name="_split_heads 拆头", shape=(B, H, T, hd), note=f"C 切成 {H} 份，每份 {hd} 维"),
        dict(name="scores = q@kᵀ/√hd + 掩码", shape=(B, H, T, T), note="第 (i,j) 格 = i 对 j 的关注度"),
        dict(name="_softmax_rows 逐行归一", shape=(B, H, T, T), note="每行和为 1，上三角精确为 0"),
        dict(name="y = probs @ v", shape=(B, H, T, hd), note="每个位置输出 = 允许位置的 v 加权和"),
        dict(name="合头（拆头的逆）", shape=(B, T, C), note="transpose(0,2,1,3) 再 reshape，顺序反了数值就乱"),
        dict(name="proj", shape=(B, T, C), note="头之间混合；进出形状一样"),
    ]
    flow = [dict(step, shape=" × ".join(str(s) for s in step["shape"])) for step in flow]
    return _payload("阶段 4 · 因果自注意力（试验台）",
                    f"B={B}，T={T}，C={C}，H={H}（hd={hd}），q 幅度={sharpen:.3g}，seed={seed}",
                    rows, [], mats=mats, flow=flow)


def _softmax_last(a):
    """最后一维 softmax（减最大值）—— 参考侧用，不叫你的任何实现。"""
    import numpy as np
    z = np.asarray(a, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_s05(p: dict) -> dict:
    """整台 GPT 的形状动线：ids 进、logits 出，每一步的形状都摆出来。

    动线是照着模块**一步步调出来的**（tok+pos → 每个 Block 的 norm1/attn/残差/norm2/mlp/残差
    → 末端 norm → tied 头），所以 Block.forward 还没写的时候它照样画得全 ——
    这张图画的是这台机器该长什么样，不是你的实现。
    """
    import numpy as np
    from minigpt.config import Config
    from minigpt.gpt import GPT
    from minigpt.nn import CrossEntropy

    V, C, L = int(p["vocab"]), int(p["d_model"]), int(p["n_layer"])
    ctx, T = int(p["ctx"]), int(p["t"])
    H = max(1, int(p["n_head"]))
    while C % H:
        H -= 1
    seed, B = int(p["seed"]), 2
    cfg = Config(vocab=V, d_model=C, n_layer=L, n_head=H, ctx=ctx,
                 resid_scale=bool(p["resid_scale"]))
    rng = np.random.default_rng(seed)
    model = GPT(cfg, rng=rng)
    ids = rng.integers(0, V, size=(B, T))

    steps, blocked, lg_ref = [], None, None

    def push(name, arr, note):
        steps.append(dict(name=name, shape=" × ".join(str(s) for s in np.shape(arr)), note=note))

    try:
        pos = np.arange(T)[None, :].repeat(B, 0)
        push("ids（token 下标）", ids, f"取值 0..{V - 1}")
        e_tok = model.tok(ids)
        push("tok(ids) 词嵌入", e_tok, "查表：每行换成词表里那一行向量")
        e_pos = model.pos(pos)
        push("pos(positions) 位置嵌入", e_pos, f"只有 ctx={ctx} 行，按位置下标取")
        x = e_tok + e_pos
        push("x = tok + pos", x, "两个嵌入相加就进主干，没有别的入口")
        for i, blk in enumerate(model.blocks):
            n1 = blk.norm1(x)
            push(f"L{i} · norm1(x)", n1, "pre-norm：先归一化再进子层")
            a = blk.attn(n1)
            push(f"L{i} · attn(n1)", a, "拆头→打分→softmax→加权→合头→proj")
            x1 = x + a
            push(f"L{i} · 第一条残差", x1, "主干：子层算不出来时这一路原样透传")
            n2 = blk.norm2(x1)
            push(f"L{i} · norm2(x1)", n2, "吃的是 x1，不是最原始的 x —— 写错这里 loss 照样降")
            m = blk.mlp(n2)
            push(f"L{i} · mlp(n2)", m, "逐位置的两层 MLP（fc1→GELU→fc2）")
            x = x1 + m
            push(f"L{i} · 第二条残差", x, "形状始终是 (B,T,C)：主干一路不变宽")
        xf = model.norm_f(x)
        push("末端 norm_f", xf, "进头之前最后一次归一化")
        lg_ref = model.head(xf)
        push("tied 头 → logits", lg_ref, f"借 tok.weight 算 x@Wᵀ，最后一维 = V = {V}")
    except NotImplementedError as exc:
        blocked = str(exc)
    except Exception as exc:                          # noqa: BLE001
        blocked = f"{type(exc).__name__}: {exc}"

    lg = None
    try:
        lg = np.asarray(model.forward(ids))
    except NotImplementedError as exc:
        blocked = blocked or str(exc)
    except Exception as exc:                          # noqa: BLE001
        blocked = blocked or f"{type(exc).__name__}: {exc}"

    rows = []
    if blocked:
        rows.append(dict(kind="soft", mark="注意", name="你的 GPT.forward 还没跑通",
                         detail=blocked + " —— 动线是照着模块一步步调出来的，照样画得全"))
    if lg is not None:
        ce = CrossEntropy()
        rows.append(_cmp("logits = model.forward(ids)", lg, lg_ref, f"({B},{T},{V})"))
        loss = float(ce(lg[:, :-1, :], ids[:, 1:]))
        rows.append(dict(kind="check", mark="PASS" if abs(loss - math.log(V)) < 0.9 else "FAIL",
                         name=f"初始 loss 应落在 ln V = {math.log(V):.4f} 附近",
                         detail=f"实测 {loss:.4f}（移位：位置 t 预测 t+1）"))
        keep = model.pos.weight.data.copy()
        model.pos.weight.data[...] = 0.0
        pos_used = not np.array_equal(model.forward(ids), lg)
        model.pos.weight.data[...] = keep
        rows.append(dict(kind="check", mark="PASS" if pos_used else "FAIL",
                         name="位置嵌入清零后输出必须变",
                         detail="加了" if pos_used else "不变 = pos 根本没加进主干"))
        shuffled = bool(np.array_equal(model.forward(ids[:, ::-1])[:, 0], lg[:, 0]))
        rows.append(dict(kind="check", mark="PASS" if not shuffled else "FAIL",
                         name="打乱 token 顺序后输出必须变",
                         detail="变了" if not shuffled else "不变 = 因果掩码或位置嵌入没起作用"))
        try:
            model.forward(rng.integers(0, V, size=(1, ctx + 2)))
            guard = False
        except Exception:                             # noqa: BLE001
            guard = True
        rows.append(dict(kind="check", mark="PASS" if guard else "FAIL",
                         name=f"T > ctx（{ctx}）必须被拦住",
                         detail="拦住了" if guard else "没拦住：会拿越界的位置嵌入算出垃圾"))
        tied = model.head.weight is model.tok.weight
        rows.append(dict(kind="check", mark="PASS" if tied else "FAIL",
                         name="输出头借的是同一个 Parameter 对象",
                         detail="is 成立，梯度会累加到同一份数组" if tied else "只是数值相等 —— 梯度会分家"))

    src = lg if lg is not None else lg_ref
    mats = [_mat("tok.weight（词嵌入，头借的就是它）", model.tok.weight.data, f"({V}, {C})")]
    if src is not None:
        mats.insert(0, dict(name=f"下一个词的概率 · 位置 0（{B} 条序列）",
                            note="刚初始化完应该接近均匀（每格≈1/V）；训过之后才会出现尖峰",
                            shape=[B, V], scale="prob",
                            values=[[float(v) for v in row] for row in _softmax_last(src[:, 0, :])]))
        mats.insert(1, _mat("logits · 位置 0", src[:, 0, :], f"({B}, {V})"))
    return _payload("阶段 5 · 整台 GPT 的形状动线（试验台）",
                    f"B={B}，T={T}，V={V}，C={C}，L={L}，H={H}（hd={C // H}），ctx={ctx}，"
                    f"resid_scale={bool(p['resid_scale'])}，seed={seed}",
                    rows, [], mats=mats, flow=steps)


def run_s06(p: dict) -> dict:
    """训练循环试验台：真语料上短训一段，把 loss / 全局梯度范数 / 学习率 三条线一起画。

    三条必须一起看：只看 loss 你会以为 clip 没用（它只在尖峰那几步才生效），
    只看 lr 又会以为 warmup 那几十步是白花时间。
    """
    import numpy as np

    steps, B, ctx = int(p["iters"]), int(p["batch"]), int(p["ctx"])
    lr_peak, warmup = float(p["lr"]), int(p["warmup"])
    clip, wd = float(p["clip"]), float(p["wd"])
    curve, gnorms, lrs, clipped, blocked, vocab = [], [], [], 0, None, 0
    try:
        from minigpt.config import Config
        from minigpt.dataset import Batcher, token_stream, train_val_texts
        from minigpt.gpt import GPT
        from minigpt.optim import AdamW, clip_grad_norm, lr_at
        from minigpt.tokenizer import CharTokenizer

        text = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
        tr, _va = train_val_texts(text, 0.1)
        tok = CharTokenizer(tr)
        ids = token_stream(tr, tok)
        vocab = tok.vocab_size
        cfg = Config(vocab=vocab, d_model=32, n_layer=2, n_head=4, ctx=ctx)
        rng = np.random.default_rng(0)
        model = GPT(cfg, rng=rng)
        opt = AdamW(model, lr=lr_peak, weight_decay=wd)
        bat = Batcher(ids, ctx)
        for i in range(steps):
            opt.zero_grad()
            x, _y = bat(B, rng)
            curve.append(float(model.loss_backward(x)))
            g = float(clip_grad_norm(model, clip))
            lr = float(lr_at(i, steps, lr_peak, warmup=warmup))
            opt.step(lr)
            gnorms.append(g)
            lrs.append(lr)
            clipped += int(g > clip)
    except NotImplementedError as exc:
        blocked = str(exc)
    except Exception as exc:                          # noqa: BLE001
        blocked = f"{type(exc).__name__}: {exc}"

    rows = []
    if blocked:
        rows.append(dict(kind="soft", mark="注意", name="循环跑不起来",
                         detail=blocked + (" —— 已经跑完的这几步的曲线还在下面" if curve else "")))
    if curve:
        rows += [
            dict(kind="info", mark="info", name="loss 首末",
                 detail=f"{curve[0]:.4f} → {curve[-1]:.4f}，降 {curve[0] / max(curve[-1], 1e-12):.1f}×"),
            dict(kind="info", mark="info", name="ln V 基准",
                 detail=f"ln {vocab} = {math.log(vocab):.4f}（随机初始化该从这儿起步）"),
            dict(kind="info", mark="info", name="clip 触发",
                 detail=f"{clipped}/{len(curve)} 步被裁剪（max_norm={clip:g}）—— 0 次说明这个阈值根本没起作用"),
            dict(kind="info", mark="info", name="梯度范数（裁剪前）",
                 detail=f"首 {gnorms[0]:.3g} → 末 {gnorms[-1]:.3g}，最大 {max(gnorms):.3g}"),
            dict(kind="info", mark="info", name="学习率",
                 detail=f"warmup {warmup} 步爬到 {lr_peak:g}，末端收到 {lrs[-1]:.3g}"),
        ]
    if not blocked:
        try:
            import os
            import tempfile
            from minigpt.checkpoint import load_checkpoint, save_checkpoint
            path = os.path.join(tempfile.mkdtemp(), "lab.ckpt")
            save_checkpoint(path, model, cfg)
            info = load_checkpoint(path)
            ref = dict(model.named_parameters())
            bad = [n for n, t in info["tensors"].items()
                   if n not in ref or not np.allclose(np.asarray(t), ref[n].data, atol=1e-5)]
            rows.append(dict(kind="check", mark="PASS" if not bad else "FAIL",
                             name="检查点往返（存完再读回来）",
                             detail=f"{len(info['tensors'])} 个张量逐个对上（fp32 容差 1e-5）"
                                    if not bad else f"对不上：{bad[:3]}"))
        except NotImplementedError as exc:
            rows.append(dict(kind="soft", mark="注意", name="检查点往返", detail=str(exc)))
        except Exception as exc:                      # noqa: BLE001
            rows.append(dict(kind="check", mark="FAIL", name="检查点往返",
                             detail=f"{type(exc).__name__}: {exc}"))

    curves = []
    if curve:
        curves = [
            dict(name="loss（每步）", values=curve, note="ln V 是起点，掉到它的一半以下才叫学得动"),
            dict(name=f"全局梯度范数（裁剪前，参考线 {clip:g}）", values=gnorms,
                 note="尖峰就是 clip 在工作；一直很小 = clip 阈值太松"),
            dict(name="学习率 lr", values=lrs, note=f"warmup={warmup}，之后 cosine"),
        ]
    return _payload("阶段 6 · 训练循环（试验台）",
                    (f"V={vocab}，B={B}，ctx={ctx}，steps={steps}，lr={lr_peak:g}，"
                     f"warmup={warmup}，clip={clip:g}，wd={wd:g}") if curve or vocab else
                    f"B={B}，ctx={ctx}，steps={steps}，lr={lr_peak:g}，clip={clip:g}（还没跑到能读词表的那一步）",
                    rows, curves)


ALPHABET = " theand.iof,chragsmlz\n"          # 采样试验台的假词表：下标 → 看得懂的字符


def run_s07(p: dict) -> dict:
    """采样试验台：一组**固定**的分数上，温度 / top-k / top-p / 惩罚各自把分布改成什么样。

    故意不接模型：分数不变，所以柱子的差别只可能来自采样器本身。
    理论那根柱子是这里按定义直写的（不叫你的 keep_top_k / keep_top_p），
    经验那根来自你的 sample_next —— 两根不一样高，就是实现和定义没对上。
    """
    import numpy as np
    from minigpt.sampler import entropy, sample_next

    V = 12
    seed = int(p["seed"])
    tau, k, pp = float(p["temperature"]), int(p["top_k"]), float(p["top_p"])
    n, penalty = int(p["n_samples"]), float(p["penalty"])
    z = np.random.default_rng(seed).standard_normal(V) * 2.0

    def ref_dist(tau, k, pp):
        if tau <= 0:
            q = np.zeros(V)
            q[int(np.argmax(z))] = 1.0
            return q
        q = _softmax_last(z / tau)
        if 0 < k < V:
            thr = np.sort(q)[::-1][k - 1]
            q = np.where(q >= thr, q, 0.0)
        if pp < 1.0:
            idx = np.argsort(-q)
            cum = np.cumsum(q[idx])
            cut = int(np.searchsorted(cum, pp, side="left")) + 1
            keep = np.zeros(V, dtype=bool)
            keep[idx[:cut]] = True
            q = np.where(keep, q, 0.0)
        return q / q.sum()

    want = ref_dist(tau, k, pp)
    rows, counts, blocked = [], {}, None
    try:
        rng = np.random.default_rng(seed)
        for _ in range(n):
            t = int(sample_next(z, rng, temperature=tau, top_k=k, top_p=pp))
            counts[t] = counts.get(t, 0) + 1
        emp = np.array([counts.get(i, 0) for i in range(V)], dtype=float) / n
        l1 = float(np.abs(emp - want).sum())
        rows.append(dict(kind="check", mark="PASS" if l1 <= 0.14 else "FAIL",
                         name=f"经验分布 vs 定义：L1 = {l1:.3f}",
                         detail=f"采样 {n} 次；阈值 0.14（次数调大这根柱子更稳）"))
        rows.append(dict(kind="info", mark="info", name="候选集大小",
                         detail=f"定义里有 {int((want > 0).sum())} 个词，你的采样出现过 {len(counts)} 个"))
        rows.append(dict(kind="info", mark="info", name="熵",
                         detail=f"定义侧 {entropy(want):.4f} nat，τ=1 无剪枝是 {entropy(_softmax_last(z)):.4f}"
                                f"，均匀上限 ln V = {np.log(V):.4f}"))
        g0 = int(sample_next(z, np.random.default_rng(1), temperature=0))
        g1 = int(sample_next(z, np.random.default_rng(99), temperature=0))
        best = int(np.argmax(z))
        rows.append(dict(kind="check", mark="PASS" if (g0 == best == g1) else "FAIL",
                         name="τ=0 必须等于 argmax 且与 seed 无关",
                         detail=f"返回 {g0} / {g1}，argmax 是 {best}"))
        r1 = int(sample_next(z, np.random.default_rng(7), temperature=tau, top_k=k, top_p=pp))
        r2 = int(sample_next(z, np.random.default_rng(7), temperature=tau, top_k=k, top_p=pp))
        rows.append(dict(kind="check", mark="PASS" if r1 == r2 else "FAIL",
                         name="同 seed 必须同结果", detail=f"{r1} vs {r2}"))
    except NotImplementedError as exc:
        blocked = str(exc)
        rows.append(dict(kind="soft", mark="注意", name="你的 sample_next 还没写",
                         detail=blocked + " —— 下面先画定义侧那根柱子"))
    except Exception as exc:                          # noqa: BLE001
        blocked = f"{type(exc).__name__}: {exc}"
        rows.append(dict(kind="check", mark="FAIL", name="sample_next 跑不起来", detail=blocked))

    labels = [f"{ALPHABET[i] if ALPHABET[i] != ' ' else '␠'}（{i}）" for i in range(V)]
    series = [dict(name="定义（lab 按定义直算）", values=[float(v) for v in want], color="#3ecf8e")]
    if not blocked:      # 没写成就只画定义那根 —— 拿定义冒充"你的采样"是骗人
        series.append(dict(name="你的采样（经验频率）",
                           values=[counts.get(i, 0) / n for i in range(V)], color="#4aa3ff"))
    bars = [dict(name="下一个词的概率（分数固定，所以只反映采样器）",
                 note=f"τ={tau:g}，top_k={k}，top_p={pp:g}，采样 {n} 次",
                 labels=labels, series=series)]

    text = []
    if not blocked:
        rng = np.random.default_rng(seed + 3)
        seen, out = [], []
        for _ in range(120):
            t = int(sample_next(z, rng, temperature=tau, top_k=k, top_p=pp,
                               seen=seen if penalty > 1 else None, penalty=penalty))
            out.append(ALPHABET[t])
            if penalty > 1:
                seen = (seen + [t])[-24:]
        text.append(dict(name=f"生成 120 个字符（penalty={penalty:g}，seen 窗口 24）",
                         body="".join(out),
                         note=f"出现过的不同字符 {len(set(out))} 个，最长连续重复 "
                              f"{max((len(m.group(0)) for m in __import__('re').finditer(r'(.)\\1*', ''.join(out))), default=0)}"))
    return _payload("阶段 7 · 采样与生成（试验台）",
                    f"V={V}（假词表：下标→字符），τ={tau:g}，top_k={k}，top_p={pp:g}，"
                    f"penalty={penalty:g}，n={n}，seed={seed}",
                    rows, [], bars=bars, text=text)


def _cmp(name, mine, ref, shape_txt):
    import numpy as np
    mine = np.asarray(mine, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if mine.shape != ref.shape:
        return dict(kind="check", mark="FAIL", name=name,
                    detail=f"形状 {tuple(mine.shape)}，应为 {tuple(ref.shape)} —— 先看形状再看数值")
    d = float(np.max(np.abs(mine - ref)))
    if d < 1e-10:
        return dict(kind="check", mark="PASS", name=name,
                    detail=f"和参考完全一致（最大差 {d:.1e}），形状 {shape_txt}")
    return dict(kind="check", mark="FAIL", name=name,
                detail=f"最大差 {d:.3g}（形状 {shape_txt} 对了，数值不对）")


def _mat(name, arr, note):
    import numpy as np
    arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    return dict(name=name, note=note, shape=list(arr.shape),
                values=[[float(v) for v in r] for r in arr])


def _payload(title, subtitle, rows, curves, mats=None, flow=None, bars=None, text=None):
    # 发散时 loss 会是 nan/inf：留个洞给前端画"断了"比画一条乱线诚实
    clean = []
    for c in curves:
        vals = [float(v) for v in c["values"] if v == v and abs(v) != float("inf")]
        if len(vals) >= 2:
            clean.append(dict(name=c["name"], note=c.get("note", ""), values=vals))
    mats = mats or []
    # 矩阵里的 NaN/Infinity 不能进 JSON：json.dumps 会原样写出 NaN，
    # 而浏览器那边 JSON.parse 直接拒收整个响应（"is not valid JSON"），整页都画不出来。
    # 非有限值统一换成 null，前端画成空格子。
    for m in mats:
        m["values"] = [[(float(v) if v is not None and math.isfinite(float(v)) else None)
                        for v in row] for row in m.get("values", [])]
    return dict(title=title, subtitle=subtitle, partial=False, elapsed=0.0,
                got=0, total=0, ok=None, rows=rows, curves=clean, mats=mats,
                flow=flow or [], bars=bars or [], text=text or [])


RUN = {"s01": run_s01, "s02": run_s02, "s03a": run_s03a, "s03b": run_s03b, "s03c": run_s03c,
       "s04": run_s04, "s05": run_s05, "s06": run_s06, "s07": run_s07}


def main() -> int:
    stage = sys.argv[1]
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if stage not in RUN:
        print(json.dumps(dict(error=f"这个阶段还没有试验台：{stage}"), ensure_ascii=False))
        return 2
    defaults = {k["key"]: k["default"] for k in KNOBS[stage]}
    defaults.update({k: v for k, v in params.items() if k in defaults})
    try:
        payload = RUN[stage](defaults)
    except NotImplementedError as exc:                 # 你还没写完：说人话，别甩 traceback
        payload = _payload("试验台跑不起来", "", [dict(kind="soft", mark="注意",
                    name="前置还没写完", detail=str(exc))], [])
    print(JSON_MARK + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
