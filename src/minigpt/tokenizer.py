"""字符级 tokenizer：把字符串变成整数序列，再原样变回来。

为什么阶段 5 从"一个字符 = 一个 token"开始：

* **协议先立起来**。模型只吃整数序列，它不知道也不关心这些整数是字、词还是子词。
  以后换成 BPE（阶段 8）只换这个文件的内部实现，`dataset.py`、`gpt.py`、判据都不用动。
* **词表小，基准看得懂**。本手册的语料只有 30~70 个不同字符，随机猜的交叉熵基准
  $\\ln V$ 落在 3.5~4.2 之间，你能把"loss 从 4.1 降到 1.9"和"模型学会了什么"对上号。
  词表上万的时候这个直觉就没了。
* **它能脱离模型自己验证**。`decode(encode(text)) == text` 是一条不依赖任何网络层的判据 ——
  分词写错了在训练开始之前就会被抓出来，而不是三天后在 loss 曲线里找原因。

三条写死的约定（别在别处偷偷改，改了判据就换地方红）：

1. id 从 0 起，按字符的 **Unicode 码点排序**分配。所以同一份语料永远得到同一个词表，
   重跑一次不会"词表漂了、以前存的检查点对不上"。
2. 不做大小写折叠、不切词、不压缩空白：文本里有什么字符，词表里就有什么字符。
   换行 `\\n` 也是一个正式 token —— 它恰恰是模型要学的"句子在哪儿结束"。
3. 语料外的字符由 `drop_unknown` 决定：默认丢掉（让评测/演示先把能表达的部分跑通），
   设 False 则抛 `KeyError`（数据入库时用这个，宁可炸也不要悄悄丢内容）。

============================================================ 你要写的部分
    TODO-20  CharTokenizer.encode / decode   阶段 5 · 讲义 §9

已给你写好：词表构造（`sorted(set(text))`，决定 id 的唯一地方）、`vocab_size`、
`char_at`、`chars()`、以及套在 encode 外面的 `encode_array`。
所以你要写的只有"查表"和"反查表"两件事 —— 但判据盯的是它们之间的**互逆性**。

验收：`python stages/s05_char_lm.py` 的头两条判据就是它（往返一致 + 词表口径）。
"""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    """`text ⇄ list[int]`。构造时给它一份语料，词表就是这份语料里出现过的字符。"""

    def __init__(self, text: str = ""):
        # sorted() 而不是 set() 的迭代顺序：Python 的 set 顺序会随插入历史变化，
        # 那样两次构造出来的词表可能不同，检查点里的整数就全错位了。
        self._chars: list[str] = sorted(set(text))
        self._index: dict[str, int] = {c: i for i, c in enumerate(self._chars)}

    # ---------------------------------------------------------------- 词表
    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    def __len__(self) -> int:
        return len(self._chars)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab={self.vocab_size})"

    def chars(self) -> str:
        """整个词表拼成一个字符串，第 i 个字符就是 id i。调试时最好用的一条。"""
        return "".join(self._chars)

    def char_at(self, i: int) -> str:
        return self._chars[int(i)]

    # ---------------------------------------------------------------- 编码
    def encode(self, text: str, drop_unknown: bool = True) -> list[int]:
        """字符串 → id 列表。

        ============ TODO-20a（阶段 5 · 讲义 §9）============
        逐字符查 `self._index`，三个必须想清楚的点：
          · **用 `dict.get` 还是 `in`**：注意 id 0 是合法的，`if not i:` 会把第一个字符
            当成"没找到"—— 这是这一层唯一会静默出错的地方（词表第一个字符永远编不进去，
            loss 会莫名其妙高一点，而且永远不会报错）。
          · 查不到时：`drop_unknown=True` 跳过；否则抛 KeyError，
            消息里要带**是哪个字符、在第几个位置**（调试时你只会感谢多写了这两个数）。
          · 返回 `list[int]`，不是 numpy 数组：`dataset.py` 会自己转，这里保持纯 Python，
            将来换 BPE 时"一段文本对应几个 token"是不定的，列表才是对的抽象。
        """
        raise NotImplementedError("TODO-20: 写 CharTokenizer.encode（见 docs/manual/s05.html#sec-9）")

    def decode(self, ids) -> str:
        """id 序列（list / 数组都行）→ 字符串。

        ============ TODO-20b（阶段 5 · 讲义 §9）============
        一行：`self._chars[int(i)]` 逐个取再 join。两条注意：
          · id 越界**不要**取模、不要 clamp，让它 IndexError 炸出来 ——
            生成时模型吐出一个越界 id 是"训练崩了"的现场证据，悄悄折回词表就把线索抹掉了。
          · `int(i)` 是为了 numpy 整数（np.int64 也能直接当索引，但 `list[np.int64]`
            在某些 numpy 版本会报 TypeError，显式转一下最省事）。
        写完自查：`decode(encode(原文)) == 原文`（原文只用词表内的字符）。
        这条往返在 `stages/s05_char_lm.py` 里是硬判据，`ids` 从 `generate()` 回来时也要成立。
        """
        raise NotImplementedError("TODO-20: 写 CharTokenizer.decode（见 docs/manual/s05.html#sec-9）")

    # ---------------------------------------------------------------- 便利层（已给你）
    def encode_array(self, text: str, dtype=np.int64) -> np.ndarray:
        """训练要的整数数组。单独包一层，是因为 `dataset.py` 只在两个地方碰 numpy。"""
        return np.asarray(self.encode(text), dtype=dtype)

    def counts_unknown(self, text: str) -> int:
        """有几个字符不在词表里。做数据体检用的，不参与训练。"""
        return sum(1 for ch in text if ch not in self._index)
