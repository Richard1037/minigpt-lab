"""采样：把"最后一个位置的 logits"变成"下一个 token"。

一个模型、一份权重、同一个 prompt，换一套采样参数就是另一种"性格"。
这个文件决定你读到的每一句话，而且它**不参与训练** —— 所以它是整个手册里
唯一一个"改坏了也不会让 loss 变差"的部分：判据必须直接盯着分布本身。

顺序（§2 讲为什么必须是这个顺序）：

    1) 重复惩罚 apply_penalty      —— 在**分数**上动手，出现过的词被推小
    2) 温度 temperature            —— logits / T，改变分布的陡峭程度
    3) top-k keep_top_k            —— 在**分数**上切一刀（掩成 MASK，不是删元素）
    4) softmax → 概率
    5) top-p keep_top_p            —— 在**概率**上取 nucleus，重新归一化
    6) rng.choice                  —— 按概率抽签

============================================================ 你要写的部分
    TODO-28  apply_penalty     阶段 7 · 讲义 §4
    TODO-29  keep_top_k        阶段 7 · 讲义 §5
    TODO-30  keep_top_p        阶段 7 · 讲义 §6
    TODO-31  sample_next       阶段 7 · 讲义 §7   ← 把上面三个按顺序接起来

已给你写好：`_softmax`（减最大值那套，阶段 3/4 已经见过两次）、`entropy`（诊断用）、
`temperature_sweep`（演示"温度=陡峭程度"用）。

验收：`python stages/s07_generate.py`。它不训大模型 —— 前 11 条判据直接在合成 logits 上
统计分布性质，只有最后几条需要一台真训过的小模型。
"""

from __future__ import annotations

import numpy as np

# 和 attention.NEG_INF 同一个理由：够大、但没有 inf，所以 softmax 里 exp(MASK - max)
# 精确等于 0，反向/归一化都不会冒出 nan。
MASK = -1.0e18


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    e = np.exp(logits - m)
    return e / np.sum(e)


def apply_penalty(logits: np.ndarray, seen, penalty: float = 1.0) -> np.ndarray:
    """已经在 `seen` 里出现过的 id：正分数除以 penalty，负分数乘以 penalty。

    ============ TODO-28（阶段 7 · 讲义 §4）============
    GPT-2 原文的做法，两个不对称里藏着道理：
        out[idx] > 0   →  out[idx] /= penalty      （好的被压低）
        out[idx] <= 0  →  out[idx] *= penalty      （差的被推得更差）
    两条合起来只做一件事：**把它往更小的方向推**。如果统一写成"除以 penalty"，
    负分数会被抬向 0 —— 反而更接近被选中，惩罚变成了奖励。
    三个必须做对的细节：
      · 返回**新数组**。`np.asarray(x).reshape(-1)` 在 dtype 已经是 float64 时给的是
        **视图**，就地 `/=` 会把模型刚算出来的那一行 logits 改掉。这一条判据直接查：
        调用前后原数组必须逐比特相同。
      · `seen` 先 `np.unique` 去重，再筛掉越界 id：越界是真实情况 —— 生成时序列里
        可能带着老 prompt 的 id，换过 tokenizer 就会超出词表。不筛会 IndexError。
      · `penalty == 1.0` 或 `seen` 为空时原样返回（一份拷贝都不改），
        `seen` 里全是越界 id 时也是原样返回。
    """
    raise NotImplementedError("TODO-28: 写 apply_penalty（见 docs/manual/s07.html#sec-4）")


def keep_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """只保留最高的 k 个分数，其余掩成 MASK；k<=0 或 k>=V 时原样返回。

    ============ TODO-29（阶段 7 · 讲义 §5）============
        cutoff = np.partition(z, -k)[-k]          # 第 k 大的值，O(V) 而不是 O(V log V)
        return np.where(z < cutoff, MASK, z)
    三个坑：
      · 用 `np.partition` 而不是 `argsort`：前者平均线性时间，词表上万时差一个量级；
        而且 argsort 之后还要把位置映射回去，多两步就多两个出错机会。
      · 比较用 `<`（不是 `<=`）：恰好等于 cutoff 的那一格要**保留**。写成 `<=` 时，
        如果最高的两个分数并列相等，两个都会被掩掉 → 整行都是 MASK →
        `exp(MASK - max)` 全是 1，看似没炸，其实是把并列的两个都留了/都删了，
        判据里专门备了一份"两个并列最高分"的输入。
      · 掩成 MASK 而不是删元素：删了就得记住"哪一列对应哪个 id"，
        返回的就不再是词表上的分布，`sample_next` 里的 `rng.choice(V, p=...)` 直接对不上号。
    """
    raise NotImplementedError("TODO-29: 写 keep_top_k（见 docs/manual/s07.html#sec-5）")


def keep_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """按概率从高到低累加，落在 nucleus 之外的置 0 再归一化。

    ============ TODO-30（阶段 7 · 讲义 §6）============
    nucleus 的定义是"从高到低累加、刚攒够 p 就停"，所以要看**轮到这一格之前**攒了多少：
        order = np.argsort(q)[::-1]           # 概率从大到小的 id
        cum = np.cumsum(q[order])             # 到这一格为止的累计
        before = cum - q[order]               # 轮到这一格**之前**的累计
        take = before < p                     # 还没攒够 p 就继续收，攒够了就停
        take 全空（p<=0）时只留 order[0]；其余置 0，再除以新的和
    为什么不是 `cum <= p`：那样留下的质量总是**刚不到** p（p=0.8 时可能只有 0.62），
    而所有人说"nucleus 覆盖 80% 概率"指的都是 ≥ 80%。判据就在数这个。
    为什么要那个空集兜底：p<=0 时一格都选不上 → 全 0 → 归一化 0/0 → nan，
    而 nan 传进 `rng.choice` 是 ValueError，报错位置离真正的错误已经隔了两层。
    为什么在**概率**上切而不是在分数上切：top-p 的定义就是"覆盖多少概率质量"，
    同一刀切在分数上会因为温度不同而切出完全不同的东西。
    注意这一步必须在 softmax **之后**（见 §2 的顺序），且返回的还是一个和为 1 的分布 ——
    调用者拿到的是概率，不是分数。
    """
    raise NotImplementedError("TODO-30: 写 keep_top_p（见 docs/manual/s07.html#sec-6）")


def sample_next(logits, rng, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, seen=None, penalty: float = 1.0) -> int:
    """从一个位置的 logits 抽一个 token id。

    ============ TODO-31（阶段 7 · 讲义 §7）============
    把前三个按 §2 的顺序接起来，五行：
        z = asarray(logits, float64).reshape(-1).copy()
        if penalty != 1.0 and seen: z = apply_penalty(z, seen, penalty)
        if temperature <= 0: return int(np.argmax(z))     # 贪心：不经过任何随机
        probs = _softmax(keep_top_k(z / temperature, top_k))
        return int(rng.choice(probs.size, p=keep_top_p(probs, top_p)))
    四个点：
      · **贪心在惩罚之后、在除温度之前**。T=0 的定义是"取最大的分数"；不先惩罚的话
        复读问题在贪心模式下最严重（它一定会选回同一个词）。
      · `rng` 是参数而不是内部 `default_rng()`：同一段文本要能复现，靠的是"调用方决定种子"。
        内部新建的话，同一个 seed 两次调用给出不同结果 —— 判据里那条"可复现"就在测它。
      · 温度的位置：温度必须在 **softmax 之前**（放之后就等于没放）。注意 `z/T` 是单调缩放，
        所以"先 top-k 还是先调温"切出来的是同一批候选，顺序换了不会出事；真正不可交换的是
        温度与 softmax、以及 top-p 必须在 softmax 之后（它切的是概率质量，不是分数）。
        "T 开得很大输出仍然很单调"的症状来自温度没作用在分数上，别赖 k 的先后。
      · `rng.choice(n, p=probs)` 要求 p 非负、和为 1；`keep_top_p` 交出的已经是一个归一化过的
        分布，这里**不要再自己除一次和**。多除一次不会当场炸（numpy 有 1e-8 容差），真正的
        风险是"归一化"这件事在两个地方各写一半 —— 哪天其中一处漏了，症状才是那句
        `ValueError: Probabilities do not sum to 1`，而你会先去怀疑别的地方。
    """
    raise NotImplementedError("TODO-31: 写 sample_next（见 docs/manual/s07.html#sec-7）")


def entropy(probs) -> float:
    """分布的香农熵（nat）。阶段 7 用它把"温度改变了分布形状"变成一个可比的数。"""
    q = np.asarray(probs, dtype=np.float64).reshape(-1)
    q = q[q > 0]
    return float(-np.sum(q * np.log(q)))


def temperature_sweep(logits, rng, temperatures=(0.3, 0.7, 1.0, 1.5), n=200, **kw):
    """同一份 logits 在不同温度下各抽 n 个词，返回 {温度: {id: 次数}}。"""
    out: dict[float, dict[int, int]] = {}
    for t in temperatures:
        counts: dict[int, int] = {}
        sub = np.random.default_rng(rng.integers(1 << 31))
        for _ in range(n):
            tok = sample_next(logits, sub, temperature=t, **kw)
            counts[tok] = counts.get(tok, 0) + 1
        out[float(t)] = counts
    return out
