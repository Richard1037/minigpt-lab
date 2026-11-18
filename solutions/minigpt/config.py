"""模型结构配置。

字段命名与 docs/advanced-cpp-roadmap.md §6.4 的 MGPT 检查点格式保持一致，
所以这里训出来的权重以后能被 C++ 侧直接读走，不需要转换脚本。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # ---- 结构 ----
    vocab: int = 65          # 词表大小（字符级约 65，BPE 一般 4096/8192）
    d_model: int = 128       # 每个 token 的向量维度
    n_layer: int = 4         # transformer block 层数
    n_head: int = 4          # 注意力头数，必须整除 d_model
    ctx: int = 256           # 一次看多少个 token（上下文长度）

    # ---- 数值细节 ----
    eps: float = 1e-5        # RMSNorm 防除零
    init_std: float = 0.02   # 权重初始化标准差
    resid_scale: bool = True  # 残差出口投影是否按 1/sqrt(2*n_layer) 缩放

    # ---- 备用升级位（基础阶段不启用，先把钩子留好）----
    use_rope: bool = False       # 位置编码：False=学习式嵌入，True=RoPE
    use_swiglu: bool = False     # FFN：False=GELU 两层，True=SwiGLU 三层
    dropout: float = 0.0         # 基础阶段恒为 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} 必须能被 n_head={self.n_head} 整除")
        if min(self.vocab, self.d_model, self.n_layer, self.n_head, self.ctx) < 1:
            raise ValueError("所有尺寸字段必须为正整数")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"未知配置字段：{sorted(extra)}")
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "Config":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    # ------------------------------------------------------------------ 估算
    def estimate_params(self) -> int:
        """粗略参数量，用于训练前判断"这跑得动吗"。

        每层 = 注意力 4*d^2 + FFN 8*d^2（GELU 两层为 4*d*hidden≈8*d^2）；
        输出头与输入 embedding 共享权重，故 vocab*d 只算一次。
        """
        per_layer = 12 * self.d_model ** 2
        ffn_extra = 0
        if self.use_swiglu:
            ffn_extra = self.n_layer * (3 * self.d_model * self.swiglu_hidden() - 8 * self.d_model ** 2)
        return self.n_layer * per_layer + ffn_extra + self.vocab * self.d_model + self.n_layer * self.d_model + self.d_model

    def swiglu_hidden(self, multiple: int = 64) -> int:
        h = int(self.d_model * 8 / 3)
        return -(-h // multiple) * multiple

    def describe(self) -> str:
        pos = "RoPE" if self.use_rope else "学习式位置嵌入"
        ffn = f"SwiGLU({self.swiglu_hidden()})" if self.use_swiglu else "GELU"
        return (f"{self.n_layer} 层 · d={self.d_model} · {self.n_head} 头(每头 {self.head_dim}) · "
                f"ctx={self.ctx} · vocab={self.vocab} · FFN={ffn} · 位置={pos} · "
                f"约 {self.estimate_params() / 1e6:.2f}M 参数")
