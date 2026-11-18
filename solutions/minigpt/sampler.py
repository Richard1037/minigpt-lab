"""采样：把"最后一个位置的 logits"变成"下一个 token"。（完整答案）

顺序（讲义 §2 讲为什么是这个顺序）：

    1) 重复惩罚 apply_penalty      —— 在**分数**上动手，出现过的词被推小
    2) 温度 temperature            —— logits / T，改变分布的陡峭程度
    3) top-k keep_top_k            —— 在**分数**上切一刀（掩成 MASK，不是删元素）
    4) softmax → 概率
    5) top-p keep_top_p            —— 在**概率**上取 nucleus，重新归一化
    6) rng.choice                  —— 按概率抽签

骨架版（`src/minigpt/sampler.py`）把每个坑都写在契约里，这里只有实现。
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

    返回**新数组**，绝不改动调用者的 logits（`np.asarray(...).reshape(-1)` 在 dtype
    已经是 float64 时给的是视图，就地 `/=` 会把模型刚算出来的那一行改掉）。
    """
    out = np.array(logits, dtype=np.float64, copy=True).reshape(-1)
    if not seen or penalty == 1.0:
        return out
    idx = np.unique(np.asarray(seen, dtype=np.int64))
    idx = idx[(idx >= 0) & (idx < out.size)]
    if idx.size == 0:
        return out
    pos = out[idx] > 0
    out[idx[pos]] /= penalty
    out[idx[~pos]] *= penalty
    return out


def keep_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """只保留最高的 k 个分数，其余掩成 MASK；k<=0 或 k>=V 时原样返回。"""
    z = np.asarray(logits, dtype=np.float64)
    if not k or k <= 0 or k >= z.size:
        return z.copy()
    cutoff = np.partition(z, -int(k))[-int(k)]
    return np.where(z < cutoff, MASK, z)


def keep_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """按概率从高到低累加，落在 nucleus 之外的置 0 再归一化。"""
    q = np.array(probs, dtype=np.float64, copy=True).reshape(-1)
    if p >= 1.0 or q.size <= 1:
        s = q.sum()
        return q / s if s > 0 else q
    order = np.argsort(q)[::-1]
    cum = np.cumsum(q[order])
    before = cum - q[order]                  # 轮到这一格**之前**攒下的质量
    take = before < max(float(p), 0.0)        # 还没攒够 p 就继续收，攒够了就停
    if not take.any():
        take[0] = True                        # p<=0 也要留一个候选，否则 0/0
    mask = np.zeros_like(q)
    mask[order[take]] = 1.0
    q = q * mask
    s = q.sum()
    return q / s if s > 0 else q


def sample_next(logits, rng, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, seen=None, penalty: float = 1.0) -> int:
    """从一个位置的 logits 抽一个 token id。`rng` 必须传进来，才能复现同一段文本。"""
    z = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if penalty != 1.0 and seen:
        z = apply_penalty(z, seen, penalty)
    if temperature <= 0:
        return int(np.argmax(z))              # 贪心：完全确定性，不加任何随机
    z = z / temperature
    z = keep_top_k(z, top_k)
    probs = _softmax(z)
    probs = keep_top_p(probs, top_p)
    return int(rng.choice(probs.size, p=probs))


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
