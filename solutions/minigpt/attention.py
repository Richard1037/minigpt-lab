"""因果多头自注意力 —— transformer 里唯一"让 token 之间交换信息"的部分。

============================================================ 先理解形状
输入  x      (B, T, C)        每个 token 一个 C 维向量
拆头  q,k,v  (B, H, T, hd)    hd = C // H。同一个 token 的 C 维被切成 H 份，各给一个头
算分  att    (B, H, T, T)      att[i,j] = 第 i 个 token 对第 j 个 token 的关注度
掩码  只允许 i ≥ j             "不许偷看未来"，这是语言模型能自回归的根本原因
加权  y      (B, H, T, hd)     每个位置输出 = 所有允许位置 v 的加权和
合头  (B, T, C) → proj → 输出

多头有什么用：H 个头用 H 套不同的投影，就能同时追踪 H 种不同的关系
（一个头管"上一个词是谁"，一个头管"句子里的主语"……具体分工是学出来的，不是设计的）。
"""

from __future__ import annotations

import math

import numpy as np

from .nn import Linear, Module

# 为什么用 -1e18 而不是 -inf：softmax 的反向要乘 mask 处的梯度，
# -inf 会先产生 0*inf=nan 再毁掉整条链。用一个足够大的负数，效果一样但没有 nan。
NEG_INF = -1.0e18


class CausalSelfAttention(Module):
    """pre-norm 之后的标准实现：q/k/v 三个独立投影，注意力后接一个输出投影。

    刻意不把 q/k/v 融成一个大矩阵（很多实现会那么做，能少一次 GEMM）：
    分开写你能在检查点里看清每个矩阵是什么，出问题时也能单独校验。
    以后要提速，第一件事就是把它仨合并。
    """

    def __init__(self, cfg, rng=None):
        C, H = cfg.d_model, cfg.n_head
        self.C, self.H, self.hd = C, H, cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.hd)
        self.q = Linear(C, C, rng=rng)
        # k 故意不设 bias：b_k 会给每一行所有分数加同一个常数(q_i·b_k)，而 softmax
        # 对"整行 +常数"完全不变 → 它的梯度恒等于 0，是个纯死参数，还白占 d 个格子。
        # 这不是省字数的优化，是能用数学推出来的"这层不需要这参数"。
        self.k = Linear(C, C, use_bias=False, rng=rng)
        self.v = Linear(C, C, rng=rng)
        self.proj = Linear(C, C, rng=rng)

    # ------------------------------------------------------------------ 正向
    def forward(self, x):
        B, T, C = x.shape
        assert C == self.C, f"输入通道 {C} 与配置 d_model={self.C} 不符"
        q = self._split_heads(self.q(x), B, T)
        k = self._split_heads(self.k(x), B, T)
        v = self._split_heads(self.v(x), B, T)
        self._q, self._k, self._v = q, k, v

        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale        # (B,H,T,T)
        att = att + self._causal_mask(T, x.dtype)               # 掩码必须加在 softmax **之前**
        self._probs = _softmax_rows(att)
        y = self._probs @ v                                     # (B,H,T,hd)
        merged = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        self._merged_shape = (B, T, C)
        return self.proj(merged)

    # ------------------------------------------------------------------ 反向
    def backward(self, g):
        g_merged = self.proj.backward(g)                         # (B,T,C)
        B, T, C = self._merged_shape
        gy = g_merged.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)   # (B,H,T,hd)

        probs, v = self._probs, self._v
        g_probs = gy @ v.transpose(0, 1, 3, 2)                   # d(y=probs@v)/d(probs)
        g_v = probs.transpose(0, 1, 3, 2) @ gy                   # d/dv

        # 被掩掉的位置在正向里已经变成 exp(-1e18)=0，所以这里天然不贡献梯度，
        # 不需要（也**不能**）再拿 mask 去 where —— 常见错误是把条件写反，
        # 结果清掉了允许位置的梯度、保留了禁止位置的梯度，loss 照样会降但模型是坏的。
        g_att = softmax_backward(probs, g_probs)

        q = self._q                                              # 缓存在 forward 里
        k = self._k
        g_q = (g_att @ k) * self.scale
        g_k = g_att.transpose(0, 1, 3, 2) @ q * self.scale

        g_q_flat = g_q.transpose(0, 2, 1, 3).reshape(B, T, C)
        g_k_flat = g_k.transpose(0, 2, 1, 3).reshape(B, T, C)
        g_v_flat = g_v.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.q.backward(g_q_flat) + self.k.backward(g_k_flat) + self.v.backward(g_v_flat)

    # ------------------------------------------------------------------ 工具
    def _split_heads(self, y, B, T):
        """(B,T,C) → (B,H,T,hd)。注意顺序：先按 (T,H,hd) 摊开，再把 T 和 H 换到前面。"""
        out = y.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)
        return out

    @staticmethod
    def _causal_mask(T: int, dtype) -> np.ndarray:
        """j ≤ i 处为 0，其余为 NEG_INF，形状 (1,1,T,T) 以便对所有 B、H 广播。"""
        m = np.triu(np.ones((T, T), dtype=dtype), k=1) * NEG_INF
        return m.reshape(1, 1, T, T)


def softmax_backward(probs: np.ndarray, g_probs: np.ndarray) -> np.ndarray:
    """softmax 的逐行反向：g = p * (gp - Σ(gp·p))，求和沿最后一维。

    减掉的那一项来自"每行和恒为 1"这个约束：任何一格的扰动都会改变分母，
    所以整行都要参与。写成 gp * p 直接返回，相对误差会在 1e-1 量级 —— 这是
    本阶段最"看起来合理"的错。
    """
    s = np.sum(g_probs * probs, axis=-1, keepdims=True)
    return probs * (g_probs - s)


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    """最后一维做 softmax，先减最大值防溢出。"""
    m = np.max(a, axis=-1, keepdims=True)
    e = np.exp(a - m)
    return e / np.sum(e, axis=-1, keepdims=True)
