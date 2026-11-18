"""权重存档：MGPT 二进制格式 —— 一次写盘，C++ 侧能直接读走。

为什么不用 `np.save` / `torch.save`：它们要么绑死 numpy 的 pickle 格式，
要么绑死某个框架的对象图。这份手册的最终目标是把同一个模型用 C++ 重写一遍
（`docs/advanced-cpp-roadmap.md`），跨语言只有一件东西靠得住：**自己定的字节布局**。
所以这里手写一个几十行的容器，字段、顺序、字节数全都写在注释里，
两边各读各的，不需要任何转换脚本。

格式（所有整数与浮点数都是**小端**）：

    [0, 4)     magic   b"MGPT"      优化器状态用 b"MGTR"，其余完全同构
    [4, 8)     u32     version = 1
    [8, 12)    u32     n            头 JSON 的字节数
    [12,12+n)  bytes   config JSON  —— 就是 Config.to_dict()，字段名和 C++ 侧一致
    然后按张量重复，直到 EOF：
        u32 name_len | name(UTF-8) | u32 ndim | i64[ndim] shape | fp32 data（行主序、连续）

三个刻意的决定，每一个都会在某天救你一次：

* **权重要降成 float32，优化器状态留 float64。** float32 让文件正好一半大，而且 C++ 侧
  全程用 float32 推理；训练时激活值用 float64 是为了让有限差分可信，存下来的权重不需要 ——
  第 7 位有效数字对生成没有任何影响。优化器状态（Adam 的一二阶矩）反过来**只给自己读**，
  而且"断点续训"要和"一口气训完"对账，留 float64 才对得上。同一个容器、两种 dtype，
  理由完全不同。
* **张量名直接用 `model.named_parameters()` 的名字**（`"blocks.0.attn.q.W"` 这种）。
  短名（`wq.0`）省几十字节，但要求读写两侧同步维护一张映射表 —— 而 `named_parameters()`
  已经按 id 去重，**tied 的输出头天然只有一份**，名字还能和形状一起自校验。
* **读的时候校验三件事**：魔数、版本号、以及每个张量的形状。少了这三条，一个读错的
  检查点不会报错，它会安静地给出一堆看起来合理的垃圾权重。

============================================================ 你要写的部分
    TODO-25  save_checkpoint          阶段 6 · 讲义 §4
    TODO-26  load_checkpoint          阶段 6 · 讲义 §5   ← 校验与装配，这一步的真正内容
    TODO-27  save/load_train_state    阶段 6 · 讲义 §6

已给你写好：整个字节层（`_u32` / `_write_header` / `_write_tensors` / `_read_header` /
`_read_tensors`）和 `_moment_lists`。字节序、截断检测、重名检测这些**最容易出错**的部分
都在这儿了 —— 把它们留给你写，你只会花一个下午和 struct 格式串搏斗，
而这一章真正要练的是"协议两端怎么对账"。

验收：`python stages/s06_train.py`。它跑的是真语料（`data/corpus.txt`），最后一条判据
拿"存了再续训"和"一口气训完"对账 —— 那是这一章唯一说得清"存档真的有用"的检验。
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

from .config import Config

MAGIC = b"MGPT"           # 权重存档
STATE_MAGIC = b"MGTR"     # 训练状态存档（优化器矩 + step + lr + loss 曲线）
VERSION = 1


# ------------------------------------------------------------------ 字节层
def _u32(v: int) -> bytes:
    return struct.pack("<I", int(v))


def _write_header(f, magic: bytes, header: dict) -> None:
    body = json.dumps(header, sort_keys=True).encode("utf-8")
    f.write(magic)
    f.write(_u32(VERSION))
    f.write(_u32(len(body)))
    f.write(body)


def _write_tensors(f, items, dtype: str) -> int:
    for name, arr in items:
        a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
        raw = name.encode("utf-8")
        f.write(_u32(len(raw)))
        f.write(raw)
        f.write(_u32(a.ndim))
        f.write(np.asarray(a.shape, dtype="<i8").tobytes())
        f.write(a.tobytes(order="C"))
    return len(items)


def _read_header(f, expect_magic: bytes) -> dict:
    magic = f.read(4)
    if magic != expect_magic:
        raise ValueError(f"魔数不对：读到 {magic!r}，期望 {expect_magic!r}。这不是 MGPT 存档。")
    ver = struct.unpack("<I", f.read(4))[0]
    if ver != VERSION:
        raise ValueError(f"检查点版本 {ver}，这份手册只认 {VERSION}（格式变了，别硬读）")
    n = struct.unpack("<I", f.read(4))[0]
    body = f.read(n)
    if len(body) != n:
        raise ValueError("文件在头 JSON 处被截断")
    return json.loads(body.decode("utf-8"))


def _read_tensors(f, dtype: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    while True:
        nb = f.read(4)
        if not nb:
            break                                  # 正常结束：EOF 就是"没有更多张量"
        if len(nb) < 4:
            raise ValueError("文件被截断（还剩几个字节？写的时候崩在半路）")
        name = f.read(struct.unpack("<I", nb)[0]).decode("utf-8")
        nd = struct.unpack("<I", f.read(4))[0]
        shape = [int(v) for v in np.frombuffer(f.read(8 * nd), dtype="<i8")]
        cnt = int(np.prod(shape)) if shape else 1
        item = np.frombuffer(f.read(cnt * np.dtype(dtype).itemsize), dtype=dtype)
        if item.size != cnt:
            raise ValueError(f"张量 {name} 期望 {cnt} 个数，只读到 {item.size} —— 文件截断")
        if name in out:
            raise ValueError(f"张量名重复：{name}（tied 权重被登记了两次？）")
        out[name] = item.reshape(shape).copy()      # copy：frombuffer 给的是只读视图
    return out


# ------------------------------------------------------------------ 权重存档
def save_checkpoint(path: str, model, cfg: Config | None = None) -> int:
    """写权重存档（fp32），返回文件字节数。

    ============ TODO-25（阶段 6 · 讲义 §4）============
    字节层已经给你了，你要定的只有两件事：
      1) **存哪些张量、按什么顺序**：用 `model.named_parameters()`，它已经按 id 去重
         （tied 的输出头是同一个对象，只出现一次），而且顺序稳定 → 加载时能按名字对账。
         每项写成 `(名字, 参数的数据)`。
      2) **头 JSON 放什么**：`{"config": cfg.to_dict()}`；cfg 缺省时取 `model.cfg`，
         两者都没有就抛 ValueError —— 没有配置的检查点是半截存档：形状能读回来，
         但没人知道 d_model/n_layer，C++ 侧建不出模型。
    dtype 传 `"<f4"`：磁盘上是 float32（理由见文件头第一个决定）。
    返回值用 `os.path.getsize(path)`，判据拿它核对"是不是真的降到 4 字节一个数"。
    """
    raise NotImplementedError("TODO-25: 写 save_checkpoint（见 docs/manual/s06.html#sec-4）")


def load_checkpoint(path: str, model=None, strict: bool = True) -> dict:
    """读权重存档；给了 model 就按名字灌进去（原地写），返回 config/tensors/applied。

    ============ TODO-26（阶段 6 · 讲义 §5）============
    读字节给你了，你写的是**校验与装配** —— 这才是这一步真正要练的东西：
      1) `_read_header(f, MAGIC)` + `_read_tensors(f, "<f4")`；头 JSON 的 `"config"`
         交给 `Config.from_dict` 变成对象（里面没有 config 就抛错）。
         `model is None` 到这里就可以直接返回了 —— "不建模型先看看存档里有什么"是排错的
         第一条命令，也是判据里"文件格式合规"那条要用的读法。
      2) 给了 model 先比配置：`cfg != model.cfg` 就抛 ValueError，并把**具体哪几个字段
         不一样**写进消息（`Config` 是 dataclass，`!=` 会逐字段比；想列出差异就遍历
         `cfg.to_dict()`）。为什么这么狠：配置错了但形状恰好兼容时，模型照样能跑、
         loss 照样是个数 —— 只是它是一台被搬错零件的机器。
      3) 逐个参数按**名字**取张量：取不到 = 缺东西；形状不同 = 抛错并写明期望/实际；
         存档里有模型没有的名字 → `strict` 时抛错，否则只在返回值里列出来
         （这是"以后加新层但要读旧存档"的兼容钩子，也是为什么签名里有 `strict`）。
      4) 赋值要 `p.data[...] = t`，**原地写**，不要 `p.data = t`：tied 的输出头
         （`self.head.weight`）和优化器缓存的 `self.params` 拿着的还是原来那块内存，
         换了对象它们就各看各的 —— 症状是"加载没报错，生成却还是随机的"。
    """
    raise NotImplementedError("TODO-26: 写 load_checkpoint（见 docs/manual/s06.html#sec-5）")


# ------------------------------------------------------------------ 训练状态（断点续训）
def _moment_lists(opt):
    """把优化器的动量缓冲归一成 `[(前缀, 与参数同序的列表), …]`：AdamW 是 m/v，SGD 是 buf。"""
    out = []
    for attr in ("m", "v", "buf"):
        buf = getattr(opt, attr, None)
        if isinstance(buf, list):
            out.append((attr, buf))
    return out


def save_train_state(path: str, model, opt, step: int, curve=(), cfg: Config | None = None) -> int:
    """存档"训到哪儿了"：step / lr / loss 曲线 / 配置 + 优化器的矩（float64）。

    ============ TODO-27a（阶段 6 · 讲义 §6）============
    和 save_checkpoint 只有两处不同，请想清楚再写：
      1) 头 JSON 装 `{"config":…, "step":…, "lr":…, "curve":[…]}`。
         `curve` 要逐个 `float()` —— 里面可能是 numpy 标量，`json.dumps` 不认 numpy.float64。
      2) 张量是优化器的矩，命名规则固定为 `f"{attr}.{参数名}"`，例如
         `"m.blocks.0.norm1.w"`。`_moment_lists(opt)` 给你 `[(属性名, 与参数同序的列表), …]`
         （AdamW 是 m/v，SGD 是 buf），把它和 `model.named_parameters()` **zip** 起来按名字编。
         用名字而不是裸下标，是为了让 load 那侧能校验"这台优化器配的是那个模型吗"；
         顺手比一下 `len(buf) == len(named)`，不等就说明配错了。
    dtype 传 `"<f8"`（留 float64 的理由见文件头），魔数用 `STATE_MAGIC`。
    """
    raise NotImplementedError("TODO-27: 写 save_train_state（见 docs/manual/s06.html#sec-6）")


def load_train_state(path: str, model=None, opt=None) -> dict:
    """读训练状态；给了 model+opt 就按名字恢复矩和 `opt.step_count`。

    ============ TODO-27b（阶段 6 · 讲义 §6）============
    校验点是"顺序"：
      · 按 `model.named_parameters()` 的顺序重建每个列表，元素从 `tensors[f"{attr}.{name}"]`
        取；取不到就抛 ValueError 说清缺的是哪个名字 —— 拿错优化器状态最坏的下场
        不是报错，是**用别人的动量继续走**，loss 会诡异地抖。
      · 形状要和参数一致（矩永远和参数同形）。
      · `opt.step_count = 头里的 step`。漏了这一条，偏差修正里那个 t 从 1 重新开始，
        等于把刚存下来的矩又放大一遍，续训的第一步就会跳。
      · 赋值同样要**原地**（`buf[i][...] = t`）：优化器构造时保存的是这些数组的引用。
    """
    raise NotImplementedError("TODO-27: 写 load_train_state（见 docs/manual/s06.html#sec-6）")


# ------------------------------------------------------------------ 便利
def checkpoint_summary(path: str) -> str:
    """一行看看某个存档里有什么（不建模型，纯读文件）。排错用。"""
    with open(path, "rb") as f:
        head = _read_header(f, MAGIC)
        t = _read_tensors(f, "<f4")
    cfg = head.get("config", {})
    bytes_of = sum(a.size * 4 for a in t.values())
    return (f"MGPT v{VERSION} · {len(t)} 个张量 · 权重 {bytes_of / 1e6:.2f} MB(fp32) · "
            f"vocab={cfg.get('vocab')} d_model={cfg.get('d_model')} n_layer={cfg.get('n_layer')}")


def file_size(path: str) -> int:
    return os.path.getsize(path)
