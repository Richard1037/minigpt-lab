"""完整的 GPT：词嵌入 + 位置嵌入 + N 个 block + 末端归一化 + 输出头。

架构（pre-norm，和现代 LLM 一致）：

    x = tok_emb[ids] + pos_emb[0..T)
    重复 n_layer 次： x = x + attn(rmsnorm(x))
                     x = x + mlp( rmsnorm(x))
    x = rmsnorm(x)
    logits = x @ tok_embᵀ          ← 权重共享（tied），省 vocab×C 个参数

每个 block 都是"子层输出加回主干"的残差结构，所以主干是 T 条并行的"信息高速公路"，
子层只是在路边做修改。层数再深也不会把最初的 token 信息弄丢。
"""

from __future__ import annotations

import numpy as np

from .attention import CausalSelfAttention
from .config import Config
from .nn import (MLP, Module, Parameter, RMSNorm, CrossEntropy, Embedding,
                 init_resid_scale)


class Block(Module):
    """一个 pre-norm transformer 层。顺序很重要：先归一化→进子层→加回残差。"""

    def __init__(self, cfg: Config, rng=None):
        self.norm1 = RMSNorm(cfg.d_model, cfg.eps)
        self.attn = CausalSelfAttention(cfg, rng=rng)
        self.norm2 = RMSNorm(cfg.d_model, cfg.eps)
        hidden = cfg.swiglu_hidden() if cfg.use_swiglu else 4 * cfg.d_model
        self.mlp = MLP(cfg.d_model, hidden, rng=rng)

    def forward(self, x):
        h1 = self.norm1(x)
        a = self.attn(h1)
        x1 = x + a
        h2 = self.norm2(x1)
        m = self.mlp(h2)
        return x1 + m

    def backward(self, g):
        """out = x1 + mlp(norm2(x1))，x1 = x + attn(norm1(x))。

        残差的梯度是"原样透传 + 支路回传"两路相加，这是最容易写漏的一处：
        收到 g 时，它既是给 mlp 输出的梯度，也是给主干 x1 的梯度。
        """
        g_h2 = self.mlp.backward(g)              # 支路：穿过 mlp
        g_x1 = g + self.norm2.backward(g_h2)     # 直通 + 绕道 norm2
        g_h1 = self.attn.backward(g_x1)          # 第二层残差同理
        return g_x1 + self.norm1.backward(g_h1)


class TiedHead(Module):
    """输出层：logits = x @ embedding_weightᵀ。

    它不拥有参数，只是"借用"词嵌入的权重 —— 这就是 weight tying。
    好处：省 vocab×C 个参数（小模型上是 10%~20%），且"什么词"这个概念
    在输入侧和输出侧只学一次。
    """

    def __init__(self, weight: Parameter):
        self.weight = weight

    def forward(self, x):
        self._x_shape = x.shape
        self._x_flat = x.reshape(-1, x.shape[-1])
        V = self.weight.data.shape[0]
        return (self._x_flat @ self.weight.data.T).reshape(x.shape[:-1] + (V,))

    def backward(self, g):
        # 关键：这个梯度要**累加**进词嵌入的 grad，和查表层回传的梯度叠在一起
        V = self.weight.data.shape[0]
        g2 = g.reshape(-1, V)
        self.weight.grad += g2.T @ self._x_flat
        return (g2 @ self.weight.data).reshape(self._x_shape)


class GPT(Module):
    """语言模型本体。训练只用到 forward / loss_backward / generate 三个口。"""

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None):
        self.cfg = cfg
        rng = rng if rng is not None else np.random.default_rng(0)
        self.tok = Embedding(cfg.vocab, cfg.d_model, cfg.init_std, rng=rng)
        self.pos = Embedding(cfg.ctx, cfg.d_model, cfg.init_std, rng=rng)
        self.blocks = [Block(cfg, rng=rng) for _ in range(cfg.n_layer)]
        self.norm_f = RMSNorm(cfg.d_model, cfg.eps)
        self.head = TiedHead(self.tok.weight)      # 注意：同一个 Parameter 对象
        if cfg.resid_scale:
            init_resid_scale(self, cfg.n_layer)
        self._loss_fn = CrossEntropy()

    # ------------------------------------------------------------------ 正向
    def forward(self, ids):
        """ids: (B, T) 整数 → logits: (B, T, vocab)。

        位置嵌入按 token 在序列里的**下标**取，0..T-1，和数据本身无关。
        """
        ids = np.asarray(ids)
        B, T = ids.shape
        assert T <= self.cfg.ctx, f"序列长度 {T} 超过配置 ctx={self.cfg.ctx}"
        positions = np.arange(T)[None, :].repeat(B, 0)
        x = self.tok(ids) + self.pos(positions)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm_f(x))

    # ------------------------------------------------------------------ 反向
    def backward(self, grad_logits):
        """从 logits 的梯度一路回传到所有参数（含 tied 的两路合并）。

        grad_logits 必须是 (B, T, vocab) 的**完整**形状：最后一个位置没参与 loss，
        它的梯度就是 0 —— 别在这里省掉那一行，省了形状就对不上，也会算错 head 的梯度。
        """
        g = self.head.backward(grad_logits)
        g = self.norm_f.backward(g)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        self.pos.backward(g)      # 位置嵌入的梯度散射回 pos.weight
        self.tok.backward(g)      # 词嵌入的梯度散射回 tok.weight（和 head 那份叠加）
        return None

    # ------------------------------------------------------------------ 训练用
    def loss_backward(self, ids) -> float:
        """训练循环就调这一个：给 (B,T) 的 id，返回下一个 token 的交叉熵。

        这里做了一件关键的事：**logits 去掉最后一位、targets 去掉第一位**。
        位置 t 的输出用来预测 t+1 位置的 token。移位错一次，模型学的是"预测自己"，
        loss 会瞬间掉到 0 且生成全是垃圾 —— 这是新手最容易中的致命 bug（第 3 节有判据）。
        """
        ids = np.asarray(ids)
        logits = self.forward(ids)
        loss = self._loss_fn(logits[:, :-1, :], ids[:, 1:])
        grad = self._loss_fn.backward(1.0)              # (B, T-1, V)
        full = np.zeros(logits.shape, dtype=np.float64)  # 补回最后一位：它的梯度是 0
        full[:, :-1, :] = grad
        self.backward(full)
        return loss

    # ------------------------------------------------------------------ 生成
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8,
                 top_k=40, top_p=0.9, seed=0, repetition_penalty=1.0):
        """自回归生成。阶段 7 会详细讲采样（sampler.py 的 TODO-28~31 挂在下面那一行上）；
        这里先把"能出字"的循环搭好。

        注意：完全没有 KV cache —— 每生成一个 token 都把整段重算一遍。
        正确但慢，慢得很有教育意义（你会主动想去实现 cache）。
        """
        from .sampler import sample_next

        rng = np.random.default_rng(seed)
        ids = np.asarray(prompt_ids).reshape(1, -1)
        for _ in range(max_new_tokens):
            if ids.shape[1] >= self.cfg.ctx:
                ids = ids[:, -self.cfg.ctx + 1:]
            logits = self.forward(ids)
            nxt = sample_next(logits[0, -1], rng, temperature, top_k, top_p,
                              seen=ids.reshape(-1).tolist(), penalty=repetition_penalty)
            ids = np.concatenate([ids, np.array([[nxt]], dtype=ids.dtype)], axis=1)
        return ids.reshape(-1).tolist()
