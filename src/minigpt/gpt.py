"""完整的 GPT：词嵌入 + 位置嵌入 + N 个 block + 末端归一化 + 输出头。

架构（pre-norm，和现代 LLM 一致）：

    x = tok_emb[ids] + pos_emb[0..T)
    重复 n_layer 次： x = x + attn(rmsnorm(x))
                     x = x + mlp( rmsnorm(x))
    x = rmsnorm(x)
    logits = x @ tok_embᵀ          ← 权重共享（tied），省 vocab×C 个参数

每个 block 都是"子层输出加回主干"的残差结构，所以主干是 T 条并行的"信息高速公路"，
子层只是在路边做修改。层数再深也不会把最初的 token 信息弄丢。

============================================================ 你要写的部分
    TODO-15  Block.forward              阶段 5 · 讲义 §4
    TODO-16  Block.backward             阶段 5 · 讲义 §5
    TODO-17  TiedHead.forward/backward  阶段 5 · 讲义 §6   ← 权重共享，两路梯度在这儿汇合
    TODO-18  GPT.forward                阶段 5 · 讲义 §7
    TODO-19  GPT.loss_backward          阶段 5 · 讲义 §8   ← 移位，本阶段最贵的一条

已给你写好：两个 `__init__`（含 `init_resid_scale` 的调用时机）、`GPT.backward`
（纯粹是"反着走一遍"，没有新数学，但阶段 5 的判据会替你看住它）、`GPT.generate`
（阶段 7 才写采样，这里只是把循环搭好）。

验收：`python stages/s05_char_lm.py`。写完这五条，`tests/test_model.py`（阶段 4 的考卷）
也会多出 Block / 整机 / 因果性那几条 —— 它们本来测的就是这台整机。
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
        """(B,T,C) → (B,T,C)。pre-norm 残差块，四条子表达式，顺序不许换。

        ============ TODO-15（阶段 5 · 讲义 §4）============
        要做的事只有五行，但每一行都在决定"梯度往后能不能走通"：
            h1 = norm1(x)        ← 归一化只在**进子层之前**做，主干 x 一个字都不改
            a  = attn(h1)
            x1 = x + a           ← 第一次加回主干；缓存它，backward 要用
            h2 = norm2(x1)
            return x1 + mlp(h2)
        为什么"先 norm 再进子层"（pre-norm）而不是"加完再 norm"（post-norm）：
        pre-norm 让主干成为一条不被改写的恒等路径，深度加到十几层仍然能训；
        post-norm 在深层上会把梯度压没。这是 2020 年之后所有 LLM 都用 pre-norm 的原因。
        两个坑：
          · 忘记缓存 x1 之外还要缓存"h1/h2 的输入"吗？不用 —— 每一层自己存自己的
            （RMSNorm/MLP/注意力都已经按约定缓存了）。Block 只是把它们接起来。
          · 写成 `return x + a + mlp(norm2(x))`（拿 x 而不是 x1 进第二条 norm）：
            形状全对、loss 照样降，但第二残差流断了，梯度校验会挂。
        """
        raise NotImplementedError("TODO-15: 写 Block.forward（见 docs/manual/s05.html#sec-4）")

    # ------------------------------------------------------------------ 反向
    def backward(self, g):
        """out = x1 + mlp(norm2(x1))，x1 = x + attn(norm1(x))。

        ============ TODO-16（阶段 5 · 讲义 §5）============
        倒着走 TODO-15 那五行。**最容易写漏的是"原样透传"那一路**：
        收到的 g 既要用给 mlp 的输出，也是主干 x1 的输出 —— 残差把同一个梯度抄了两份。
            g_h2 = mlp.backward(g)              # 支路：穿过 mlp（顺便把 mlp 的参数梯度累加了）
            g_x1 = g + norm2.backward(g_h2)     # 直通 + 绕道 norm2
            g_h1 = attn.backward(g_x1)
            return g_x1 + norm1.backward(g_h1)  # 同样的两路，再抄一次
        自查（不用跑判据就能看出来）：把 attn 和 mlp 两个"残差出口"的权重
        （`attn.proj.W`、`mlp.fc2.W`）全置 0 再 forward，输出必须**逐比特等于输入**；
        这时 backward 喂进去一个全 1 的 g，返回值也必须逐比特等于 g。
        少了任何一条 "+ g"，这两条都会露馅（前者查 forward，后者查 backward）。
        """
        raise NotImplementedError("TODO-16: 写 Block.backward（见 docs/manual/s05.html#sec-5）")


class TiedHead(Module):
    """输出层：logits = x @ embedding_weightᵀ。

    它不拥有参数，只是"借用"词嵌入的权重 —— 这就是 weight tying。
    好处：省 vocab×C 个参数（小模型上是 10%~20%），且"什么词"这个概念
    在输入侧和输出侧只学一次。
    """

    def __init__(self, weight: Parameter):
        self.weight = weight

    def forward(self, x):
        """logits = x @ Wᵀ，W 是借来的词嵌入 (V, C)。输入 (B,T,C) → 输出 (B,T,V)。

        ============ TODO-17a（阶段 5 · 讲义 §6）============
        和 Linear.forward 是同一件事，只有两处不同：
          · 没有 bias（有 bias 就等于给每个词加同一个偏好，会被 softmax 的
            "整行 +常数不变"吃掉一半，而且它其实已经在嵌入里了）；
          · **不拥有参数**。`self.weight` 是别人（Embedding）的对象，这里只是引用。
        缓存反向要用的东西：摊平成 (B·T, C) 的输入本身和输入形状。
        形状推导：`x` 最后一维是 C，`W` 是 (V, C)，所以要 `W.T` 才能得到 (…, V)。
        """
        raise NotImplementedError("TODO-17: 写 TiedHead.forward（见 docs/manual/s05.html#sec-6）")

    def backward(self, g):
        """返回 dL/dx（形状同输入），并把 dL/dW **累加**进借来的那块权重。

        ============ TODO-17b（阶段 5 · 讲义 §6）============
        两行数学 + 一个致命细节：
            dL/dW = g_flatᵀ @ x_flat        (V, C)   ← 和 Linear 的 gW 一模一样
            dL/dx = g_flat @ W              (N, C)   ← 还原成输入形状
            self.weight.grad += dL/dW       ← 是 +=，不是 =
        为什么这条 `+=` 是本阶段最重要的一行：`tok.weight` 同时被两处使用 ——
        查表层（TODO 在阶段 3）把梯度散射回行，输出头把 `gᵀ@x` 加回来。
        写成覆盖 `=` 的话，先后顺序决定结果，整机梯度校验挂在 `tok.weight` 上；
        更阴的是**单独测输出头时看不出问题**，所以判据里专门有一条"跑两次 backward，
        梯度必须翻倍"。
        """
        raise NotImplementedError("TODO-17: 写 TiedHead.backward（见 docs/manual/s05.html#sec-6）")


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

        ============ TODO-18（阶段 5 · 讲义 §7）============
        五步，全是"把已有的层接起来"，但位置嵌入那一步有个必须想清楚的点：
            positions = np.arange(T)[None, :].repeat(B, 0)     # (B,T)，每行都是 0..T-1
            x = self.tok(ids) + self.pos(positions)            # 两股信息相加
            for blk in self.blocks: x = blk(x)
            return self.head(self.norm_f(x))                   # 末端还要再 norm 一次
        · 位置是 token 在序列里的**下标**，和它是什么字无关 —— 所以同一个字出现在
          第 1 位和第 9 位，进到 block 之前就已经是两个不同的向量了。
        · `assert T <= ctx` 留着：位置嵌入只有 ctx 行，越界会是 IndexError，
          而 assert 的消息告诉你去改 `Config.ctx`。
        · 末端那次 `norm_f` 不是多余的：主干被加了 2·L 次，长度已经和输入不是一个量级，
          直接进 tied 头等于让 logits 的尺度随深度漂。
        自查：把同一批 token 打乱顺序，输出必须**变**（不变就说明位置嵌入没生效）。
        """
        raise NotImplementedError("TODO-18: 写 GPT.forward（见 docs/manual/s05.html#sec-7）")

    # ------------------------------------------------------------------ 反向
    def backward(self, grad_logits):
        """从 logits 的梯度一路回传到所有参数（含 tied 的两路合并）。——已给你

        grad_logits 必须是 (B, T, vocab) 的**完整**形状：最后一个位置没参与 loss，
        它的梯度就是 0 —— 别在这里省掉那一行，省了形状就对不上，也会算错 head 的梯度。
        顺序是正向的镜像：head → norm_f → blocks 反序 → pos → tok。
        最后两个都要走：pos 和 tok 都是"把梯度散射回嵌入表"，只走 tok 的话
        位置嵌入会永远是死参数（`无死参数` 那条判据就是为它准备的）。
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

        ============ TODO-19（阶段 5 · 讲义 §8）============
        四行，其中一行是整个项目最贵的 bug 藏身处：
            logits = self.forward(ids)
            loss   = self._loss_fn(logits[:, :-1, :], ids[:, 1:])   ← 移一位
            grad   = self._loss_fn.backward(1.0)                    # (B, T-1, V)
            full   = np.zeros(logits.shape); full[:, :-1, :] = grad; self.backward(full)
        · **logits 去掉最后一位、targets 去掉第一位**：位置 t 的输出预测位置 t+1 的字。
          写成 `ids[:, :-1]`（预测自己）不会报错：输入嵌入里就有答案，几步之后 loss
          打到 0，曲线漂亮得让人感动，生成出来的全是"当前这个字本身"。
          `tests/test_model.py` 的"正确移位：新数据上学不动"和
          `stages/s05_char_lm.py` 的"随机数据上学不动"两条判据专门抓它。
        · 最后一位的梯度补 0，不能直接把 (B,T-1,V) 喂给 `self.backward`：
          形状差一格；而且那一格本来就该是 0（末位没有被任何 loss 项用到）。
        · `backward(1.0)` 里那个 1.0 是"loss 对 loss 的导数"，链式法则的最上游。
        · 返回的是 `float`（CrossEntropy 已经返回 Python 标量），训练循环要拿它画曲线。
        """
        raise NotImplementedError("TODO-19: 写 GPT.loss_backward（见 docs/manual/s05.html#sec-8）")

    # ------------------------------------------------------------------ 生成
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8,
                 top_k=40, top_p=0.9, seed=0, repetition_penalty=1.0):
        """自回归生成。阶段 7 会详细讲采样（`sampler.py` 的 TODO-28~31 就挂在下面这一行上）；
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
