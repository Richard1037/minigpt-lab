"""语料 → token 流 → 随机窗口批次：喂给模型的那一步。

这一步看起来最"没技术含量"，但它是两个致命 bug 的藏身处：

**其一：训练/验证泄漏。**
如果随机切窗口时不区分 train 和 val，同一段文字会既被学过又被用来评估 ——
val loss 低得假，你以为模型学会了，其实它背过。所以 `train_val_texts`
按**行**切，切完之后两边的窗口不可能重叠到同一行内容上（同一行永远只在一边）。

**其二：移位方向。**
下一 token 预测需要一个长度 `ctx+1` 的窗口，切成
    x = win[:-1]   （模型的输入）
    y = win[1:]    （每个位置的"标准答案"）
的关系是 `y[t] = x[t+1]`。这个 `y` 训练时用不到 —— `GPT.loss_backward(x)`
内部自己会移一位（阶段 5 的 TODO-19）。那为什么还要返回它？两个理由：
① `evaluate()` 用它算验证 loss，并且**必须**和训练用同一个移位，否则 train/val
    两个数不可比，你会误判"过拟合"；
② 它把"谁是答案"写成数据，能被判据直接检查（`y[:, :-1] == x[:, 1:]`）。

随机窗口而不是顺序 epoch：语言模型的 batch 是"从语料里随手抓一段"，
不是"把语料从头到尾走一遍"。所以下游没有 epoch 这个概念，只有 step 数 ——
这也是为什么阶段 6 的学习率调度按 step 而不是按 epoch 算。

============================================================ 你要写的部分
    TODO-21  Batcher.__call__   阶段 5 · 讲义 §10

已给你写好：`train_val_texts`（按行切，防泄漏）、`token_stream`、`Batcher.__init__`
（含那个 off-by-one 的重灾区 `max_start`）、`evaluate`（评测不许污染 .grad）。

验收：`python stages/s05_char_lm.py` 的"批次形状与移位"和"随机窗口真的随机"两条。
"""

from __future__ import annotations

import numpy as np

from .nn import CrossEntropy


# ------------------------------------------------------------------ 切分
def train_val_texts(text: str, val_frac: float = 0.1) -> tuple[str, str]:
    """按行切 train/val —— **不许**按字符数直接砍。

    按字符砍会把一句话劈成两半，前半进 train、后半进 val，val 就等于送分题。
    这里找到 `val_frac` 位置之后**第一个换行符**，在它后面下刀，保证两边都是整行。
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac 要在 [0,1) 里，收到 {val_frac}")
    n = len(text)
    cut = int(n * (1.0 - val_frac))
    nl = text.find("\n", cut)
    if nl < 0:                        # 后面没有换行了，往前找，宁可让 val 短一点
        nl = text.rfind("\n", 0, cut)
    if nl < 0:
        raise ValueError("语料里一个换行都没有，没法按行切 train/val")
    return text[: nl + 1], text[nl + 1:]


def token_stream(text: str, tok) -> np.ndarray:
    """文本 → 一维 int64 token 流。空流当场报错，别让它流到 batcher 里才炸。"""
    ids = tok.encode_array(text)
    if ids.size == 0:
        raise ValueError("编码结果是空的：tokenizer 的语料和这段文本没有交集？")
    return ids


# ------------------------------------------------------------------ 批次
class Batcher:
    """从一条 token 流里随机抓 `batch` 个长度为 `ctx+1` 的窗口。"""

    def __init__(self, ids, ctx: int):
        ids = np.asarray(ids).reshape(-1)
        if ctx < 1:
            raise ValueError(f"ctx 至少是 1，收到 {ctx}")
        if ids.size <= ctx:
            raise ValueError(f"token 流只有 {ids.size} 个，切不出长度 {ctx + 1} 的窗口")
        self.ids = ids
        self.ctx = int(ctx)
        # 合法起点是 0 .. ids.size-ctx-1（含），所以可选起点共 `max_start+1` 个。
        # 这一行是 off-by-one 的重灾区：写成 `ids.size - ctx` 会让最后一个窗口只有 ctx 个 token。
        self.max_start = ids.size - ctx - 1

    def __len__(self) -> int:
        return self.max_start + 1

    def __call__(self, batch: int, rng: np.random.Generator | None = None):
        """返回 `(x, y)`，形状都是 `(batch, ctx)`，且 `y[:, :-1] == x[:, 1:]`。

        ============ TODO-21（阶段 5 · 讲义 §10）============
        要做三件事：
          1) 取 `batch` 个随机起点（**有放回**，所以 batch 可以大于窗口总数；
             `rng.integers` 的上界是**开区间**，要写 `self.max_start + 1`）
          2) 每个起点截 `ctx+1` 个 token，堆成 `(batch, ctx+1)`
             —— 列表推导 + `np.stack` 就够；别用高级索引一次取完，那会摊平成二维
                再要自己 reshape，形状对了值却是错的（阶段 3/4 反复见过这一类）。
          3) 左边掉最后一个 = 输入 x，右边掉第一个 = 目标 y
             `return win[:, :-1], win[:, 1:]` —— 两个都是 `(batch, ctx)`。
        三个坑：
          · 写成 `win[:, :-1], win[:, :-1]` 之类的对称样子：形状全对，
            y 变成了"预测自己"，和 TODO-19 的移位叠加后 loss 一样能降 —— 判据
            "批次移位关系 y[:, :-1] == x[:, 1:]" 专抓它。
          · `rng` 不传就每次都用同一个种子：批次数值上永远一样，
            "随机窗口真的随机"那条判据会红 —— 而且训练看着正常，只是每步吃同一批。
          · `batch` 传 numpy 标量时 `size=` 也收，但 `range(batch)` 会挑刺，
            所以入口处 `int(batch)` 转一下。
        """
        raise NotImplementedError("TODO-21: 写 Batcher.__call__（见 docs/manual/s05.html#sec-10）")


def evaluate(model, batcher: Batcher, batch: int = 4, n_batches: int = 8,
             rng: np.random.Generator | None = None, ce: CrossEntropy | None = None) -> float:
    """平均验证 loss（**不做反向**，所以不会污染 `.grad`）。

    为什么不用 `model.loss_backward()` 来评测：那个函数会把梯度累加进每个参数，
    而 `zero_grad()` 是训练循环负责的 —— 评测一次就等于偷偷多走了一步梯度。
    所以这里只走 `forward` + `CrossEntropy`，移位方式和 `loss_backward` 完全一致
    （`logits[:, :-1]` 配 `x[:, 1:]`），两个数才可以放在一起比。

    `n_batches` 而不是 1 个批次：单批 loss 的噪声很大（语料里有的段落就是难），
    拿一个批次当验证分，曲线会抖得像发散。8 个批次是"快"和"能看出趋势"的折中。
    """
    rng = rng if rng is not None else np.random.default_rng(20240)
    ce = ce if ce is not None else CrossEntropy()
    total = 0.0
    for _ in range(n_batches):
        x, _y = batcher(batch, rng)
        logits = model.forward(x)
        total += float(ce(logits[:, :-1, :], x[:, 1:]))
    return total / max(1, n_batches)
