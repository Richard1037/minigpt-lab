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

        逐字符查 `self._index`：
          · 查到 → 把整数放进结果
          · 查不到（`dict.get` 返回 None，注意 0 是合法 id，不能用假值判断）
              - drop_unknown=True：跳过
              - 否则：抛 KeyError，并在消息里说清是哪个字符、第几个位置
        """
        out: list[int] = []
        for pos, ch in enumerate(text):
            i = self._index.get(ch)
            if i is None:
                if drop_unknown:
                    continue
                raise KeyError(f"字符 {ch!r}（第 {pos} 个）不在词表里；要么 drop_unknown=True，要么换 tokenizer")
            out.append(i)
        return out

    def decode(self, ids) -> str:
        """id 序列 → 字符串。越界**不要**取模，直接让 IndexError 炸出来。"""
        return "".join(self._chars[int(i)] for i in ids)

    # ---------------------------------------------------------------- 便利层（已给你）
    def encode_array(self, text: str, dtype=np.int64) -> np.ndarray:
        """训练要的整数数组。单独包一层，是因为 `dataset.py` 只在两个地方碰 numpy。"""
        return np.asarray(self.encode(text), dtype=dtype)

    def counts_unknown(self, text: str) -> int:
        """有几个字符不在词表里。做数据体检用的，不参与训练。"""
        return sum(1 for ch in text if ch not in self._index)
