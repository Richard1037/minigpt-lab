"""梯度校验：本项目唯一能"证明反向传播写对了"的工具。

为什么必须有它：反向传播写错**不会报错**。形状对了、代码跑通了、loss 也降了，
但它可能收敛到一个没意义的地方，或者慢得离谱。唯一可靠的检查是：

    解析梯度（你手写的 backward） vs 数值梯度（把参数挪一点点看 loss 变多少）

两者对不上 = backward 错了，没有第三种解释。误差量级：
  float64 下相对误差 < 1e-6 完全正常；1e-4 是可接受上限；更大就是有 bug。

用法（任何 Module 都能测，包括带参数的组合模块）：
    from minigpt.gradcheck import gradcheck
    ok, worst, detail = gradcheck(Linear(5, 3), [rng.standard_normal((2, 4, 5))])
"""

from __future__ import annotations

import numpy as np

_MAX_ENTRIES_PER_TENSOR = 24      # 每张量最多抽查这么多元素，避免慢到不能用


def _objective(module, inputs, cot):
    """把模块输出压成一个标量目标：标量输出直接用，数组输出与固定余向量点乘。"""
    out = module(*inputs)
    if isinstance(out, float) or (isinstance(out, np.ndarray) and out.ndim == 0):
        return float(out)
    return float(np.sum(out * cot))


def _sample_indices(size: int, limit: int) -> list[int]:
    if size <= limit:
        return list(range(size))
    return [int(i) for i in np.linspace(0, size - 1, limit).astype(int)]


def gradcheck(module, inputs, eps: float = 1e-6, tol: float = 1e-4, atol: float = 0.0,
              seed: int = 0, check_inputs: bool = True, label: str = ""):
    """返回 (是否全部通过, 最大相对误差, 明细文本)。

    module: 一个 Module（或任何实现了 forward/backward/named_parameters 的对象）。
    inputs: 传给 forward 的数组列表；整数数组（如 token id）会自动跳过输入梯度检查。
    """
    rng = np.random.default_rng(seed)
    # 必须是连续数组：否则 reshape(-1) 返回副本，扰动写不进真正的输入，校验会假通过
    inputs = [np.ascontiguousarray(np.asarray(x)) for x in inputs]
    float_inputs = [x for x in inputs if x.dtype.kind == "f"]

    # ---- 1) 解析梯度：跑一次 forward+backward，记下参数梯度和输入梯度
    probe = module(*inputs)
    is_scalar = isinstance(probe, float) or (isinstance(probe, np.ndarray) and probe.ndim == 0)
    cot = None if is_scalar else rng.standard_normal(probe.shape) * 0.7

    module.zero_grad()
    grad_in = module.backward(np.ones_like(probe) if is_scalar else cot)
    analytic_params = {id(p): p.grad.copy() for _, p in module.named_parameters()}
    # 输入梯度：标量输出（loss 型模块）同样要查，别因为"输出是标量"就跳过 ——
    # 那样 CrossEntropy 会变成"一个都没查却报通过"的假绿灯。
    ana_in: list = []
    if check_inputs and float_inputs and grad_in is not None:
        ana_in = list(grad_in) if isinstance(grad_in, (tuple, list)) else [grad_in]

    base = _objective(module, inputs, cot)
    lines, worst, ok = [], 0.0, True

    def compare(tag: str, analytic: float, numeric: float):
        """判定：相对误差 <= tol，或绝对误差 <= atol。

        为什么需要 atol：有限差分的绝对噪声约等于 ulp(目标值)/eps。目标值 ~1、
        eps=1e-6 时噪声底约 1e-10；如果被检查的那个梯度本身只有 1e-8（注意力里
        很常见，多个大数相减的抵消结果），相对误差看着像 1e-2，其实是噪声，
        不是你写错了。所以大梯度看相对误差，小梯度看绝对误差。
        注意别把 atol 调大到能盖住真 bug 的量级（1e-9 合适，1e-3 就是自欺）。
        """
        nonlocal worst, ok
        abs_err = abs(numeric - analytic)
        rel = abs_err / max(1e-8, abs(numeric), abs(analytic))
        if rel > worst:
            worst = rel
        if rel > tol and abs_err > atol:
            ok = False
            lines.append(f"    {tag}: 解析 {analytic:+.8f}  数值 {numeric:+.8f}  相对误差 {rel:.2e}  ← 对不上")
        return rel

    # ---- 2) 参数梯度
    for name, p in module.named_parameters():
        rels = []
        flat_data = p.data.reshape(-1)
        flat_ana = analytic_params[id(p)].reshape(-1)
        for idx in _sample_indices(flat_data.size, _MAX_ENTRIES_PER_TENSOR):
            saved = flat_data[idx]
            flat_data[idx] = saved + eps
            hi = _objective(module, inputs, cot)
            flat_data[idx] = saved - eps
            lo = _objective(module, inputs, cot)
            flat_data[idx] = saved
            rels.append(compare(f"{name} 第 {idx} 个元素", flat_ana[idx], (hi - lo) / (2 * eps)))
        lines.append(f"  {label or module.__class__.__name__} 参数 {name} {p.shape}: "
                     f"{len(rels)} 个抽查点，最大相对误差 {max(rels) if rels else 0:.2e}")

    # ---- 3) 输入梯度（可导输入才查）
    for i, x in enumerate(float_inputs):
        if i >= len(ana_in) or ana_in[i] is None:
            continue
        rels = []
        flat_x = x.reshape(-1)
        flat_ana = np.asarray(ana_in[i]).reshape(-1)
        for idx in _sample_indices(flat_x.size, _MAX_ENTRIES_PER_TENSOR):
            saved = flat_x[idx]
            flat_x[idx] = saved + eps
            hi = _objective(module, inputs, cot)
            flat_x[idx] = saved - eps
            lo = _objective(module, inputs, cot)
            flat_x[idx] = saved
            rels.append(compare(f"输入[{i}] 第 {idx} 个元素", flat_ana[idx], (hi - lo) / (2 * eps)))
        lines.append(f"  {label or module.__class__.__name__} 输入[{i}] {x.shape}: "
                     f"{len(rels)} 个抽查点，最大相对误差 {max(rels) if rels else 0:.2e}")

    n_checked = sum(1 for ln in lines if "抽查点" in ln)
    if n_checked == 0:
        return False, float("inf"), (f"  {label or module.__class__.__name__}: 没有任何量被检查"
                                     "（既无浮点参数也无浮点输入）—— 这不是通过，是没测")
    return ok, worst, "\n".join(lines)


def finite_diff_ok(module, inputs, **kw) -> tuple[bool, float]:
    """只要结论的简版，给阶段脚本用（不打印明细）。"""
    ok, worst, _ = gradcheck(module, inputs, **kw)
    return ok, worst
