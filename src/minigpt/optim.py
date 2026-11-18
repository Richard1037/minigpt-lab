"""优化器与学习率调度：把梯度变成参数更新。

为什么默认用 AdamW 而不是 SGD：transformer 的每一层尺度差别很大（注意力分数、
MLP 激活、嵌入），SGD 一个全局步长很难同时照顾好；Adam 给每个参数自己估一个
步长，对学习率的容忍度高得多，小模型上也更稳。

weight decay 为什么单独拎出来讲：把它从"梯度里加一项 λw"改成"更新时直接
w *= (1 - lr·λ)"才是 AdamW，两者在 Adam 里**不等价**（前者会被二阶矩缩放掉）。
而且 bias 和 norm 的缩放系数不该被衰减 —— 它们不是"权重矩阵"，衰减它们只会
让模型变笨。这个豁免标记在 Parameter.wd_exempt 里。

============================================================ 你要写的部分
    TODO-22  AdamW.step         阶段 5 · 讲义 §11   ← 偏差修正是这一条的全部难点
    TODO-23  clip_grad_norm     阶段 5 · 讲义 §12
    TODO-24  lr_at              阶段 5 · 讲义 §13

已给你写好：`_as_params`（Module/Parameter/列表三种入参都吃）、`Optimizer` 基类
（含 `zero_grad` 与解耦衰减 `_decay`）、`SGD`（带动量，正好是 AdamW 的暖身版本）。

验收：`python stages/s05_char_lm.py`。这三条不写，`tests/test_model.py`（阶段 4 的考卷）
里 tied 参数那条也会一起卡住 —— 它用 AdamW 走了一步才知道你有没有重复更新。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .nn import Module, Parameter


def _as_params(target) -> list[Parameter]:
    if isinstance(target, Module):
        return target.parameters()
    if isinstance(target, Parameter):
        return [target]
    params = list(target)
    if any(not isinstance(p, Parameter) for p in params):
        raise TypeError("优化器只接受 Module / Parameter / Parameter 列表")
    return params


def _zero_bias(params: Iterable[Parameter]):
    return [np.zeros_like(p.data) for p in params]


class Optimizer:
    """共同接口：step() 更新参数，zero_grad() 清梯度。

    lr 既能在构造时给定，也能每次 step 传入（配合 lr_at 做 warmup+cosine）。
    """

    def __init__(self, target, lr: float = 3e-4, weight_decay: float = 0.1):
        self.params = _as_params(target)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def _decay(self, p: Parameter, lr: float) -> None:
        if self.weight_decay and not p.wd_exempt:
            p.data *= (1.0 - lr * self.weight_decay)

    def step(self, lr: float | None = None) -> float:
        raise NotImplementedError


class SGD(Optimizer):
    """带动量的随机梯度下降。小实验里比 Adam 更好观察，因为每步就是"沿梯度走"。"""

    def __init__(self, target, lr: float = 1e-2, weight_decay: float = 0.0,
                 momentum: float = 0.9):
        super().__init__(target, lr, weight_decay)
        self.momentum = momentum
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def step(self, lr: float | None = None) -> float:
        lr = self.lr if lr is None else float(lr)
        self.step_count += 1
        for p, b in zip(self.params, self.buf):
            self._decay(p, lr)
            b *= self.momentum
            b += p.grad
            p.data -= lr * b
        return lr


class AdamW(Optimizer):
    """Adam + 解耦权重衰减。β2=0.95 而不是默认 0.999：小模型步数少，
    0.999 的二阶矩"记忆"太长，前期估计不准，0.9~0.95 收敛更快。
    """

    def __init__(self, target, lr: float = 3e-4, weight_decay: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.95, eps: float = 1e-8):
        super().__init__(target, lr, weight_decay)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = _zero_bias(self.params)
        self.v = _zero_bias(self.params)

    def step(self, lr: float | None = None) -> float:
        """一次更新：更新一二阶矩 → 偏差修正 → 走步。返回真正用到的 lr。

        ============ TODO-22（阶段 5 · 讲义 §11）============
        骨架（顺序别改，`self.step_count += 1` 必须在算 c1/c2 之前）：
            lr = self.lr if lr is None else float(lr)
            self.step_count += 1
            t  = self.step_count
            c1 = 1 - b1**t ；c2 = 1 - b2**t          # 偏差修正因子
            for p, mi, vi in zip(self.params, self.m, self.v):
                self._decay(p, lr)                   # 解耦衰减：已给你，直接调
                mi[...] = b1*mi + (1-b1)*g
                vi[...] = b2*vi + (1-b2)*g*g
                p.data -= lr * (mi/c1) / (np.sqrt(vi/c2) + eps)
        四个坑，每一个都能让 loss 看起来"还在降"：
          · **忘了 c1/c2**：m、v 初值是 0，前几步估计严重偏小。忘了修正的话第一步
            只走 lr·|g|/(|g|+eps)·(1-b1) ≈ 0.1·lr，等于把 warmup 悄悄放大又缩小，
            曲线前面一大段是平的。判据用"单步移动幅度 ≈ lr"抓它。
          · `mi[...] = ` 而不是 `mi = `：优化器构造时把数组引用存进 self.m，
            重新绑定名字等于把缓存扔掉，下一步又从零开始 —— 症状是"训不动"。
          · eps 加在**分母的根号外面**：√(v)+eps，不是 √(v+eps)。放里面对小梯度更宽容，
            数值上不一样（本项目按外面这个约定，判据也照它设阈值）。
          · 梯度为 0 时这一步必须**不动**（m=v=0 → 0/(0+eps)=0）。
            如果你在这里加了个 `+ 1e-12` 之类的东西，参数会自己漂移。
        """
        raise NotImplementedError("TODO-22: 写 AdamW.step（见 docs/manual/s05.html#sec-11）")


def clip_grad_norm(target, max_norm: float) -> float:
    """按全局 L2 范数裁剪梯度，返回裁剪前的范数。

    为什么要：语言模型偶发"某一批数据特别难"，梯度瞬间变大，一步把模型踹进
    坏区域，loss 曲线出现一根很高的尖刺。裁剪不改变方向，只限制步幅。
    注意顺序：**先算梯度、裁剪，再 optimizer.step()**；反过来等于没裁。

    ============ TODO-23（阶段 5 · 讲义 §12）============
    三件事，缺一件就不是"全局裁剪"：
      1) 范数是**所有张量拼起来**的 L2：把每个张量的每个元素的平方全加起来再开根号，
         不是逐张量各自裁。逐张量裁的话，参数量大的层（嵌入、FFN）相对权重会变。
      2) 返回**裁剪前**的范数 —— 训练日志要看它，它是发散的先行指标。
      3) 只有 $g>$ `max_norm` 时才缩放，且所有张量乘**同一个** `scale = max_norm / g`，
         所以方向完全不变，只缩短。`g` 为 0 时别除（直接不缩放）。
    自查：造两个参数、梯度范数 10、`max_norm=1`，裁完范数应该正好是 1，且两个梯度
    的比值和裁前一样。
    """
    raise NotImplementedError("TODO-23: 写 clip_grad_norm（见 docs/manual/s05.html#sec-12）")


def lr_at(step: int, total_steps: int, peak: float, warmup: int = 400,
          final_frac: float = 0.1) -> float:
    """线性 warmup → cosine 衰减到 peak·final_frac。

    warmup 的作用：Adam 的二阶矩在最初几十步还没估准，这时用大 lr 容易把
    第一层的权重打飞；先用小 lr 走几步"热身"，稳定性提升非常明显。
    cosine 尾端不归零而是留 10%，是因为后期需要一点步长继续微调，
    归零会让模型卡在离最优点还有一小段的地方。

    ============ TODO-24（阶段 5 · 讲义 §13）============
    分两段写，条件很清楚：
      · `step < warmup`：线性爬升 peak * (step+1)/warmup。
        写 `step/warmup` 的话第 0 步 lr 正好是 0，那一步白走（`+1` 是防它）。
      · 之后：peak * ( f + (1-f) * (1 + cos(pi*x))/2 )，
        其中 x = (step-warmup)/(total_steps-warmup) 要夹在 [0,1]（最后一步可能越界）。
    三个自查点：step=0 时约等于 0；step=warmup-1 时正好等于 peak；step≥total_steps 时
    等于 `peak*final_frac`，且整条曲线在 warmup 之后单调不增、从不超过 peak。
    `total_steps <= warmup` 要单独返回 peak（否则分母为 0 或负数）。
    """
    raise NotImplementedError("TODO-24: 写 lr_at（见 docs/manual/s05.html#sec-13）")
