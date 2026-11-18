"""权重存档：MGPT 二进制格式 —— 一次写盘，C++ 侧能直接读走。（完整答案）

格式（所有整数与浮点数都是**小端**）：

    [0, 4)     magic   b"MGPT"      优化器状态用 b"MGTR"，其余完全同构
    [4, 8)     u32     version = 1
    [8, 12)    u32     n            头 JSON 的字节数
    [12,12+n)  bytes   config JSON  —— 就是 Config.to_dict()，字段名和 C++ 侧一致
    然后按张量重复，直到 EOF：
        u32 name_len | name(UTF-8) | u32 ndim | i64[ndim] shape | fp32 data（行主序、连续）

骨架版（`src/minigpt/checkpoint.py`）把"为什么这么定"逐条讲开了，这里是实现：
权降 float32（给 C++ 读、文件减半）、优化器状态留 float64（只给自己读、要能逐位续训）、
张量名用 `named_parameters()`（天然对 tied 去重）、读时校验魔数/版本/形状。
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
    """写权重存档（fp32），返回文件字节数。"""
    cfg = cfg if cfg is not None else getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("没拿到配置：显式传 cfg=，或者用带 .cfg 属性的模型（GPT 就有）")
    items = [(name, p.data) for name, p in model.named_parameters()]
    with open(path, "wb") as f:
        _write_header(f, MAGIC, {"config": cfg.to_dict()})
        _write_tensors(f, items, "<f4")
    return os.path.getsize(path)


def load_checkpoint(path: str, model=None, strict: bool = True) -> dict:
    """读权重存档；给了 model 就按名字灌进去（原地写），返回 config/tensors/applied。"""
    with open(path, "rb") as f:
        head = _read_header(f, MAGIC)
        tensors = _read_tensors(f, "<f4")
    if "config" not in head:
        raise ValueError("头 JSON 里没有 config 字段：这份存档不知道该建多大的模型")
    cfg = Config.from_dict(head["config"])
    out = {"config": cfg, "tensors": tensors}
    if model is None:
        return out                      # 只想知道存档里有什么，不建模型也能读

    ref = getattr(model, "cfg", None)
    if ref is not None and ref != cfg:
        diff = [f"{k}: 存档 {getattr(cfg, k)} vs 模型 {getattr(ref, k)}"
                for k in cfg.to_dict() if getattr(cfg, k, None) != getattr(ref, k, None)]
        raise ValueError("检查点的配置和模型对不上：" + "；".join(diff))

    named = model.named_parameters()
    have = {n for n, _ in named}
    missing = [n for n, _ in named if n not in tensors]
    if missing:
        raise ValueError(f"存档里缺少 {len(missing)} 个参数，例如 {missing[:4]}")
    extra = sorted(set(tensors) - have)
    if extra and strict:
        raise ValueError(f"存档里有模型没有的张量：{extra}")
    for name, p in named:
        t = tensors[name]
        if tuple(t.shape) != p.shape:
            raise ValueError(f"张量 {name} 形状不符：存档 {tuple(t.shape)}，模型 {tuple(p.shape)}")
        p.data[...] = t                # 原地写：tied 输出头和优化器拿着同一个对象
    out["applied"] = len(named)
    out["extra"] = extra
    return out


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
    """存档"训到哪儿了"：step / lr / loss 曲线 / 配置 + 优化器的矩（float64）。"""
    cfg = cfg if cfg is not None else getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("没拿到配置：显式传 cfg=，或者用带 .cfg 属性的模型")
    named = model.named_parameters()
    items: list[tuple[str, object]] = []
    for attr, buf in _moment_lists(opt):
        if len(buf) != len(named):
            raise ValueError(f"{attr} 有 {len(buf)} 项，模型有 {len(named)} 个参数 —— 优化器配的不是这个模型")
        for (name, _p), arr in zip(named, buf):
            items.append((f"{attr}.{name}", arr))
    header = {"config": cfg.to_dict(), "step": int(step),
              "lr": float(getattr(opt, "lr", 0.0)), "curve": [float(v) for v in curve]}
    with open(path, "wb") as f:
        _write_header(f, STATE_MAGIC, header)
        _write_tensors(f, items, "<f8")
    return os.path.getsize(path)


def load_train_state(path: str, model=None, opt=None) -> dict:
    """读训练状态；给了 model+opt 就按名字恢复矩和 `opt.step_count`。"""
    with open(path, "rb") as f:
        head = _read_header(f, STATE_MAGIC)
        tensors = _read_tensors(f, "<f8")
    out = dict(head)
    out["config"] = Config.from_dict(head["config"])
    out["tensors"] = tensors
    if model is None or opt is None:
        return out
    named = model.named_parameters()
    for attr, buf in _moment_lists(opt):
        if len(buf) != len(named):
            raise ValueError(f"{attr} 有 {len(buf)} 项，模型有 {len(named)} 个参数：状态和模型不配套")
        for i, (name, p) in enumerate(named):
            key = f"{attr}.{name}"
            t = tensors.get(key)
            if t is None:
                raise ValueError(f"训练状态里缺少 {key} —— 这份状态是别的模型存的？")
            if tuple(t.shape) != p.shape:
                raise ValueError(f"{key} 形状不符：存档 {tuple(t.shape)}，参数 {tuple(p.shape)}")
            buf[i][...] = t               # 原地：优化器构造时存的是这些数组的引用
    opt.step_count = int(head.get("step", 0))
    out["resumed_at"] = opt.step_count
    return out


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
