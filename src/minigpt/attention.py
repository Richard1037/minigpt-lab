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

============================================================ 你要写的部分
    TODO-14  softmax_backward                阶段 4 · 讲义 §5   ← 先写这个，backward 要用它
    TODO-12  CausalSelfAttention.forward     阶段 4 · 讲义 §4
    TODO-13  CausalSelfAttention.backward    阶段 4 · 讲义 §6

已给你写好：__init__（含一个刻意的决定：k 没有 bias，理由写在里面）、
_split_heads、_causal_mask、_softmax_rows。

验收：`python tests/test_model.py`。注意它不只测注意力 —— 它还拿因果性、参数量、
死参数三件事一起查你，所以别只看 gradcheck 那几条。
网站版：把 `minigpt/attention.py` 提交到「阶段 4 · 因果自注意力」。
"""

from __future__ import annotations

import math

import numpy as np

from .nn import Linear, Module

# 为什么用 -1e18 而不是 -inf：softmax 的反向会乘到掩码位置的梯度上，
# -inf 会先产生 0*inf=nan 再毁掉整条链。用一个足够大的负数，效果一样但没有 nan。
# 附带好处：exp(-1e18 − 任何有限值) 在浮点里**精确等于 0**，所以被掩掉的位置
# 概率天然是 0、梯度天然是 0，不需要任何额外的 where。
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
        # k 故意不设 bias：b_k 给每一行所有分数加同一个常数(q_i·b_k)，而 softmax
        # 对"整行 +常数"完全不变 → 它的梯度恒等于 0，是个纯死参数，还白占 C 个格子。
        # 这不是省字数的优化，是能用数学推出来的"这层不需要这个参数"。
        self.k = Linear(C, C, use_bias=False, rng=rng)
        self.v = Linear(C, C, rng=rng)
        self.proj = Linear(C, C, rng=rng)

    # ------------------------------------------------------------------ 正向
    def forward(self, x):
        """(B,T,C) → (B,T,C)。中间那步 (B,H,T,T) 是全部魔法所在。

        ============ TODO-12（阶段 4 · 讲义 §4）============
        六步，形状全程对着文件头那张表走：
            ① q,k,v = 三个 Linear 各作用一次 → (B,T,C)，再 _split_heads → (B,H,T,hd)
            ② att = q @ k.transpose(0,1,3,2) * scale                        → (B,H,T,T)
            ③ att = att + _causal_mask(T, x.dtype)                          ← 必须在 softmax 之前
            ④ probs = _softmax_rows(att)                                    → 每行和为 1
            ⑤ y = probs @ v                                                 → (B,H,T,hd)
            ⑥ 合头（_split_heads 的逆操作）→ proj → 输出
        反向要用的东西都得缓存（约定见 nn.py 头部）：q、k、v、probs，还有合头前的形状。
        两个坑：
          · transpose 只动最后两维，别写成全轴 reverse；
          · 合头是 transpose(0,2,1,3) 回 (B,T,H,hd) 再 reshape(B,T,C)，
            顺序颠倒的话形状照样对、数值是乱的 —— 只有 gradcheck 抓得住它。
        """
        raise NotImplementedError("TODO-12: 写 CausalSelfAttention.forward（见 docs/manual/s04.html#sec-4）")

    # ------------------------------------------------------------------ 反向
    def backward(self, g):
        """输入 dL/d(输出)，返回 dL/dx；同时把梯度累加进 q/k/v/proj 四个 Linear。

        ============ TODO-13（阶段 4 · 讲义 §6）============
        倒着走 §4 的六步，"正向怎么算"决定"反向怎么乘"：
            ⑥′ g_merged = proj.backward(g)                                  → (B,T,C)
            ⑤′ g_probs = gy @ vᵀ        ；   g_v = probsᵀ @ gy               （y = probs@v 的两条支路）
            ④′ g_att = softmax_backward(probs, g_probs)                      ← TODO-14
            ②′ g_q = (g_att @ k) * scale ；g_k = g_attᵀ @ q * scale
            ①′ 三个 (B,H,T,hd) 合头成 (B,T,C)，分别喂给 q/k/v 的 backward，**返回值相加**
        三个坑：
          · 别写 `np.where(mask==0, 0, g_att)` 去"清掉禁止位置"：条件一反就清掉了
            **允许**位置的梯度，loss 照样降、模型却是坏的（本项目真踩过）。
            禁止位置的概率本来就是 0，乘出来自然是 0，什么都不用做。
          · scale 要出现在 g_q 和 g_k 里：正向乘过一次，反向两条支路各要乘一次。
          · 返回 `q.backward(...) + k.backward(...) + v.backward(...)`：
            Linear.backward 返回的是"对输入的梯度"，三条支路汇到同一个 x 所以要相加；
            参数梯度已经在各自 backward 里累加完了，别在这里重复处理。
        """
        raise NotImplementedError("TODO-13: 写 CausalSelfAttention.backward（见 docs/manual/s04.html#sec-6）")

    # ------------------------------------------------------------------ 工具（已给你）
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
    """softmax 的逐行反向：已知输出 probs 和对输出的梯度，求对输入（分数）的梯度。

    ============ TODO-14（阶段 4 · 讲义 §5）============
    推导只需要一个事实：softmax 每一行的和恒等于 1，所以**任何扰动的一阶效应加起来必须是 0**。
    结论形式（自己验一遍，别背）：

        g = probs * (g_probs − Σ_j(g_probs · probs))      求和沿最后一维，keepdims=True

    两项的直觉：`probs * g_probs` 是"只看自己这一格"的直接效应，减掉的那项是
    "整行和必须守恒"带来的约束 —— 和阶段 3 的 RMSNorm 里那个 `s` 是同一类东西。
    写完先单独自查：拿 (1,1,2,3) 的小数组手算一行，或者只跑单头注意力的 gradcheck。
    """
    raise NotImplementedError("TODO-14: 写 softmax_backward（见 docs/manual/s04.html#sec-5）")


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    """最后一维做 softmax，先减最大值防溢出。"""
    m = np.max(a, axis=-1, keepdims=True)
    e = np.exp(a - m)
    return e / np.sum(e, axis=-1, keepdims=True)
