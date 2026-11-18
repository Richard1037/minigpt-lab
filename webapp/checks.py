"""单项快测：每写完一个 TODO 就能测那一条，不用等整卷跑通。

为什么需要它：stages/ 和 tests/ 里的判据是"整卷"，阶段 3 那种七个层的卷子，
你只写完 Linear 时根本跑不到结果。这里给每个 TODO 配一条最小检查：

    needs  → 还依赖哪些 TODO（没写完时网站会直接告诉你"先写它"，而不是抛 traceback）
    code   → 一段独立可跑的小程序，打印 PASS / FAIL 开头的一行

写检查的三条原则：
1) **能不依赖未写的部分就不依赖**。例如测 `__mul__` 时不碰 `backward()`：
   直接给 `p.grad` 赋值再调用 `p._backward()` 这个闭包，就能单独验证乘法法则。
2) **一条检查只判一件事**。FAIL 时要能立刻定位到"就是这个算子的规则写错了"。
3) **结尾统一用 `card(...)` 打四段**：目的 / 输入 / 输出 / 判定。
   「目的」写这条到底在判什么、写错了会是什么症状 —— 这一句是 FAIL 时唯一有用的话，
   写在讲义里你查完就忘，写在输出里是跑的时候正对着看。PASS/FAIL 必须在行首，
   网站和命令行都靠它定位结论。

新增 TODO 时在这里加一条，同时在 server.py 的 STAGES 里登记 id（网站按钮由这两处生成）。

命令行也能跑同一段代码（PowerShell / cmd / Git Bash 都能直接粘贴，不用引号续行）：

    python webapp/checks.py                # 列出所有 id
    python webapp/checks.py TODO-1         # 只跑一条
    python webapp/checks.py ACT-1 TODO-4   # 跑几条
    $env:MGPT_ANSWER="1"; python webapp/checks.py TODO-1   # 用答案版跑（PowerShell）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 每条快测脚本共用的前置：把"这条在判什么、喂进去什么、吐出什么"说成人话的小工具。
# 规则：数组只报 形状 + dtype + 前几个数，绝不整块 dump ——
# 一个 (2,5,8) 的张量全打出来是 80 个数，你只会跳过它，等于没打印。
PRELUDE = '''

def brief(v, n=6):
    import numpy as np
    if isinstance(v, np.ndarray):
        flat = v.reshape(-1)
        head = ", ".join(f"{float(x):.4g}" for x in flat[:n])
        more = f", …（共 {v.size} 个数）" if v.size > n else ""
        return f"shape{tuple(v.shape)} {v.dtype} [{head}]{more}"
    if isinstance(v, (list, tuple)):
        head = ", ".join(brief(x, 3) for x in list(v)[:n])
        more = f", …（共 {len(v)} 项）" if len(v) > n else ""
        return f"{type(v).__name__} [{head}]{more}"
    if isinstance(v, dict):
        head = ", ".join(f"{k}: {brief(x, 3)}" for k, x in list(v.items())[:n])
        return "{ " + head + ("… }" if len(v) > n else " }")
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    shp = getattr(v, "shape", None)
    if shp is not None:                                  # Parameter 之类
        d = np.asarray(getattr(v, "data"))
        head = ", ".join(f"{float(x):.4g}" for x in d.reshape(-1)[:n])
        return f"{type(v).__name__} shape{tuple(shp)} [{head}]"
    return repr(v)[:90]


_VERDICT = []


def _section(label, items):
    """四段里的一段：第一行带标签，后面的行缩进对齐到同一列。"""
    pad = " " * 8                                  # 「  目的  」= 2 空格 + 2 个全角 + 2 空格
    for i, it in enumerate(items or []):
        head = f"  {label}  " if i == 0 else pad
        if isinstance(it, tuple):
            name, val = it[0], it[1]
            txt = f"{name} = {val if isinstance(val, str) else brief(val)}"
            print(head + (txt + f"　（{it[2]}）" if len(it) > 2 else txt))
        else:
            print(head + str(it))


def card(purpose, inputs, outputs, ok, verdict):
    """一条快测的标准四段：目的 / 输入 / 输出 / 判定。

    为什么要拆成四段：以前压成一行「输入 … → 输出 …」，看得见面包却看不出这条
    到底在判什么、错了会是什么症状 —— 而 FAIL 的时候恰恰需要那句话。

    结论行必须以 PASS / FAIL 开头（网站和命令行都按行首找它），所以前三段的行首
    别出现这两个词，要写在句子里就写「通过」「不过」。
    """
    _VERDICT.append("pass" if ok else "fail")
    _section("目的", [purpose])
    _section("输入", inputs)
    _section("输出", outputs)
    print(("PASS  " if ok else "FAIL  ") + verdict)

'''

# 每条：id（和 scalar/nn 里的 TODO 编号对齐）、label、needs、code
CHECKS: list[dict] = [
    # ------------------------------------------------------------------ 阶段 1
    dict(
        id="TODO-1", stage="s01", label="__mul__ 的乘法法则",
        needs=[],
        code=r"""
from minigpt.scalar import Value
a, b = Value(2.0, label="a"), Value(3.0, label="b")
p = a * b
p.grad = 1.0            # 假装上游传下来 1，直接触发这条规则，绕开还没写的 backward()
p._backward()
want = (3.0, 2.0)       # ∂(ab)/∂b = a, ∂(ab)/∂a = b
got = (a.grad, b.grad)
card("只判乘法法则这一条规则：∂(ab)/∂a = b、∂(ab)/∂b = a。backward() 的拓扑排序是 TODO-3，"
     "所以这里手工把上游梯度摆成 1.0，直接触发 p._backward() 那个闭包。",
     [("a", 2.0), ("b", 3.0), ("p = a*b", p.data),
      "上游梯度 p.grad = 1.0（手工赋值，绕开还没写的 backward()）"],
     [("a.grad", a.grad, "期望 3.0，就是 b"), ("b.grad", b.grad, "期望 2.0，就是 a")],
     got == want,
     f"a.grad={got[0]}、b.grad={got[1]}（期望 {want}；写成 a.grad=a 是最常见的抄错）")
""",
    ),
    dict(
        id="TODO-2", stage="s01", label="__pow__ 的幂法则（除法靠它）",
        needs=[],
        code=r"""
from minigpt.scalar import Value
x = Value(2.0, label="x")
y = x ** 3
y.grad = 1.0
y._backward()
ok_pow = abs(x.grad - 12.0) < 1e-9          # 3·2²
z = Value(4.0, label="z")
q = z / 2.0                                  # 走 z * 2**-1，检验组合出来的除法
q.grad = 1.0
q._backward()
ok_div = abs(z.grad - 0.5) < 1e-9           # ∂(z/2)/∂z = 1/2
card("幂法则 ∂(x³)/∂x = 3x²，以及靠它组合出来的除法 ∂(z/2)/∂z = 1/2。"
     "除法没有单独的 _backward，它是 z * 2⁻¹ —— 所以这条同时是「组合算子会不会互相污染」的检查。",
     [("x", 2.0), ("y = x**3", y.data), ("z", 4.0), ("q = z/2.0", q.data),
      "上游梯度都是手工摆的 1.0"],
     [("x.grad", x.grad, "期望 12 = 3·2²"), ("z.grad", z.grad, "期望 0.5")],
     ok_pow and ok_div,
     f"x**3 的梯度={x.grad}（期望 12）、z/2 的梯度={z.grad}（期望 0.5）")
""",
    ),
    dict(
        id="TODO-3", stage="s01", label="backward() 的拓扑顺序与起点",
        needs=["TODO-1"],
        code=r"""
from minigpt.scalar import Value
a, b = Value(2.0, label="a"), Value(3.0, label="b")
L = (a + b) * b          # 两条路径都汇到 b：∂L/∂b = b + (a+b) = 8
L.backward()
ok = abs(a.grad - 3.0) < 1e-9 and abs(b.grad - 8.0) < 1e-9
card("backward() 的两件事：起点 self.grad=1.0，以及沿拓扑序把梯度一条条传下去。"
     "这里故意让 b 被用到两次（(a+b) 里一次、乘数一次），∂L/∂b 必须把两条路径加起来 = 8。",
     [("a", 2.0), ("b", 3.0), ("L = (a+b)*b", L.data),
      ("L.grad", L.grad, "backward() 自己应该把它摆成 1.0")],
     [("a.grad", a.grad, "期望 3.0"), ("b.grad", b.grad, "期望 8.0 = b + (a+b)，两条路径都要算到")],
     ok,
     f"a.grad={a.grad}、b.grad={b.grad}（期望 3.0 / 8.0；全是 0 = 忘了起点 self.grad=1.0，"
     "翻倍 = 同一个节点走了两遍）")
""",
    ),
    dict(
        id="TODO-1A", stage="s01", label="fit_by_autograd 拟合直线",
        needs=["TODO-1", "TODO-2", "TODO-3"],
        code=r"""
import numpy as np
from minigpt.exercises import fit_by_autograd
x = np.linspace(-1, 1, 20)
y = 3.0 * x + 1.0
a, b, curve = fit_by_autograd(x, y, iters=400, lr=0.2, log_every=0)
ok = abs(a - 3.0) < 0.05 and abs(b - 1.0) < 0.05 and curve[-1] < 1e-3
card("第一次让整条自动微分链路真干活：用你写的 Value + backward() 拟合一条直线。"
     "数据是严格线性的 y=3x+1，没有噪声，所以答案必须回到 a≈3、b≈1；"
     "梯度方向反了会发散，忘了 zero_grad 会越跑越偏。",
     [("x", x, "20 个点，-1..1 均匀"), ("y = 3x+1", y, "真值，无噪声"),
      "迭代 iters=400，lr=0.2"],
     [("a", a, "期望 ≈3.0"), ("b", b, "期望 ≈1.0"),
      ("curve", f"{len(curve)} 项：首 {curve[0]:.4f} → 末 {curve[-1]:.3e}", "每轮的 loss，长度应等于 iters")],
     ok,
     f"a={a:.4f} b={b:.4f}；loss {curve[0]:.4f} → {curve[-1]:.3e}，下降 {curve[0]/max(curve[-1],1e-12):.1f}×"
     "（阈值：|a−3|<0.05、|b−1|<0.05、末端 <1e-3）")
""",
    ),

    # ------------------------------------------------------------------ 激活函数（测"已给你的代码"）
    dict(
        id="ACT-1", stage="s02", label="exp / relu / sigmoid 的导数与饱和（讲义 §2.3–§2.5）",
        needs=[],
        code=r"""
import math
from minigpt.scalar import Value

xs = (-3.0, -0.7, 0.4, 2.0)
bad = []
for name, want in [("exp", lambda v: math.exp(v)),
                   ("relu", lambda v: 1.0 if v > 0 else 0.0),
                   ("sigmoid", lambda v: (lambda s: s * (1 - s))(1 / (1 + math.exp(-v))))]:
    for xv in xs:
        v = Value(xv)
        y = getattr(v, name)()
        y.grad = 1.0            # 手工喂上游梯度，绕开还没写的 backward()
        y._backward()
        if abs(v.grad - want(xv)) > 1e-12:
            bad.append(f"{name}({xv}) 解析={v.grad} 期望={want(xv)}")
sat = 1 / (1 + math.exp(-8.0))
sat_ok = abs(sat * (1 - sat) - 3.35e-4) < 5e-5          # σ(8) 已经躺平：导数只有 3.35e-4
# 有限差分那一半需要你自己写完 backward()（TODO-3），没写完就先跳过，不算失败
extra = ""
try:
    from minigpt.scalar import gradient_check_graph
    node = Value(-0.7, label="x")          # 必须是**同一个**叶子：差分要扰动它的 data 再重建图
    def build():
        node.grad = 0.0
        return node.sigmoid()
    ok, worst, _ = gradient_check_graph(build, params=[node])
    extra = f"；差分对拍 {worst:.1e}" + ("通过" if ok else "不通过")
    if not ok:
        bad.append("差分对不上，backward 有问题")
except NotImplementedError:
    extra = "；差分对拍待 TODO-3（写完 backward 再点一次会更强）"
x0 = -0.7
s0 = 1 / (1 + math.exp(-x0))
card("这条测的是**已经给你的**三个激活函数的导数：exp'=exp、relu'=（x>0）、sigmoid'=σ(1−σ)。"
     "上游梯度手工摆 1.0，所以不需要你先写 backward()。另外判 σ 在 8 处已经躺平（导数只剩 3.35e-4）——"
     "那是后面「梯度消失」的源头，不是浮点噪声。",
     [("x 取样", str(xs), "四个点，正负都有"), ("上游梯度", "1.0（手工摆进 y.grad，绕开还没写的 backward()）")],
     [("x0", x0), ("exp(x0)", math.exp(x0)), ("relu(x0)", 0.0, "x0<0，所以是 0"),
      ("sigmoid(x0)", s0), ("exp'(x0)", math.exp(x0)), ("relu'(x0)", 0.0),
      ("sigmoid'(x0)", s0 * (1 - s0), "= σ(1−σ)"), ("σ(8) 的导数", sat * (1 - sat), "饱和")],
     not bad and sat_ok,
     ("三个函数 × 4 个点，解析式与手算一致" if not bad else "对不上：" + "; ".join(bad[:3]))
     + f"。σ(8)={sat:.6f} 的导数={sat*(1-sat):.3e}（饱和）" + extra)
""",
    ),
    dict(
        id="ACT-2", stage="s02", label="exp 溢出边界与「减最大值」不变性（讲义 §2.6）",
        needs=[],
        code=r"""
import math
import numpy as np

try:
    math.exp(710.0)
    overflow_ok, over = False, "e^710 居然没溢出（意外）"
except OverflowError:
    overflow_ok, over = True, "e^710 → OverflowError（float64 上限约 e^709）"

z = np.array([900.0, 880.0, 910.0])
with np.errstate(over="ignore"):
    naive = np.exp(z)                       # 不减最大值：直接爆
naive_inf = bool(np.isinf(naive).any())


def stable(v):
    e = np.exp(v - v.max())                 # 最大项变成 e^0 = 1，其余 ≤ 1
    return e / e.sum()


p = stable(z)
finite_ok = bool(np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-12)
# 真正要成立的不是某个具体数值，而是"平移不变"：softmax(z) == softmax(z − 500)
invariant = bool(np.allclose(stable(z), stable(z - 500.0), atol=1e-15))
argmax_ok = bool(int(np.argmax(p)) == int(np.argmax(z)))
ok = overflow_ok and naive_inf and finite_ok and invariant and argmax_ok
card("softmax 里「减最大值」那一步不是可选的数值技巧：z=900 时 e^900 直接 inf，减完之后概率一模一样。"
     "真正要成立的是平移不变性 softmax(z) == softmax(z−500)，这条判的就是它，"
     "顺带把 float64 的溢出边界（约 e^709）摆出来给你看。",
     [("z", z, "故意取到 900 量级，让 e^z 溢出"), ("对照", "z − 500（同一个分布整体平移）")],
     [("exp(z) 不减", naive, "inf 就是这里冒出来的"), ("softmax(z)", p),
      ("softmax(z−500)", stable(z - 500.0), "应与上一行完全相同"),
      ("argmax", int(np.argmax(p)), "平移不该改变谁最大")],
     ok,
      f"{over}；不减得到 inf：{naive_inf}；减完有限且和为 1：{finite_ok}；"
      f"平移不变：{invariant}；argmax 一致：{argmax_ok}")
""",
    ),

    # ------------------------------------------------------------------ 阶段 2
    dict(
        id="TODO-4", stage="s02", label="tanh 的导数",
        needs=[],
        code=r"""
import math
from minigpt.scalar import Value
seen = []
bad = None
for xv in (-1.0, 0.0, 0.5, 2.0):
    t = Value(xv)
    y = t.tanh()
    y.grad = 1.0
    y._backward()
    want = 1.0 - math.tanh(xv) ** 2
    seen.append((xv, float(y.data), float(t.grad), want))
    if abs(t.grad - want) > 1e-9:
        bad = f"x={xv}：解析梯度={t.grad}，期望 {want}"
        break
card("tanh 那一行导数写成 1−t²，用的是**输出** t 而不是输入 x。"
     "写成 1−x² 在 x=0 处照样对，取四个点一摆就露出来 —— 这就是为什么要多取几个点。",
     [("x 取样", "(-1.0, 0.0, 0.5, 2.0)"), ("上游梯度", "1.0（手工摆进 y.grad）")],
     [(f"x={xv:g} → tanh", f"{tv:.4f}", f"梯度={gv:.4f}，期望 1−t²={want:.4f}")
      for xv, tv, gv, want in seen],
     bad is None,
     ("四个点都等于 1−t²：" + "；".join(f"x={xv:g}→梯度 {gv:.4f}" for xv, _, gv, _ in seen)
      if bad is None else bad))
""",
    ),
    dict(
        id="TODO-2A", stage="s02", label="ScalarMLP.forward 的线性/非线性",
        needs=["TODO-1", "TODO-3", "TODO-4"],
        code=r"""
from minigpt.exercises import ScalarMLP
lin = ScalarMLP(n_hidden=4, activate=False, seed=1)
f = lambda v: lin.forward(v).data
# 线性网络必须满足 f(2x) − f(0) = 2(f(x) − f(0))
x = 0.7
resid = abs((f(2 * x) - f(0.0)) - 2 * (f(x) - f(0.0)))
is_lin = resid < 1e-9
mlp = ScalarMLP(n_hidden=4, activate=True, seed=1)
g = lambda v: mlp.forward(v).data
resid2 = abs((g(2 * x) - g(0.0)) - 2 * (g(x) - g(0.0)))
is_nonlin = resid2 > 1e-6
ok = is_lin and is_nonlin
card("forward 里 activate 那个开关到底接没接上。判据是仿射齐次性：没有激活函数时网络是 f(x)=c·x+d，"
     "必须满足 f(2x)−f(0) = 2(f(x)−f(0))，残差≈0；加上 tanh 后必须不满足，残差明显≠0。"
     "两组都≈0 = tanh 没起作用；两组都≠0 = 线性那条被写坏了。",
     [("x", x, "取 0.7，另外还算 f(0)、f(2x)"), ("网络", "ScalarMLP(n_hidden=4, seed=1)，activate 两组各一台")],
     [("f(0) / f(x) / f(2x) 无激活", f"{f(0.0):.4f} / {f(x):.4f} / {f(2 * x):.4f}",
       f"线性残差={resid:.2e}，要≈0"),
      ("g(0) / g(x) / g(2x) 带 tanh", f"{g(0.0):.4f} / {g(x):.4f} / {g(2 * x):.4f}",
       f"残差={resid2:.2e}，要≠0")],
     ok,
     f"activate=False 的线性残差={resid:.2e}（阈值 <1e-9）；加 tanh 后={resid2:.2e}（阈值 >1e-6）")
""",
    ),
    dict(
        id="TODO-2B", stage="s02", label="train_mlp 真的在降 loss",
        needs=["TODO-1", "TODO-2", "TODO-3", "TODO-4", "TODO-2A"],
        code=r"""
import numpy as np
from minigpt.exercises import ScalarMLP, train_mlp
xs = np.linspace(-3, 3, 8)
ys = np.sin(xs)
m = ScalarMLP(n_hidden=8, activate=True, seed=2)
curve = train_mlp(m, xs, ys, iters=1200, lr=0.1, decay=3.0, log_every=0)
drop = curve[0] / max(curve[-1], 1e-12)
ok = curve[-1] < 0.05 and drop > 8 and all(np.isfinite(curve))
card("训练循环是不是真的在优化：8 个 sin 采样点、1200 轮，loss 既要掉到 0.05 以下，"
     "也要至少降 8 倍（只看不掉到多少、不看降了多少倍，会被「一开始就很小」骗过去）。"
     "忘了 zero_grad、没调 backward、decay 把 lr 吃掉，症状都是「loss 抖但不降」。",
     [("xs", xs, "8 个点，-3..3"), ("ys = sin(xs)", ys),
      "n_hidden=8, iters=1200, lr=0.1, decay=3.0"],
     [("curve", f"{len(curve)} 项：首 {curve[0]:.4f} → 末 {curve[-1]:.5f}", "每轮一个 loss，必须全有限")],
     ok,
     f"首末 loss {curve[0]:.4f} → {curve[-1]:.5f}，下降 {drop:.0f}×（阈值：末端 <0.05 且降幅 >8×）")
""",
    ),

    # ------------------------------------------------------------------ 阶段 3
    dict(
        id="TODO-5A", stage="s03a", label="Scale 的两件产出与累加",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import Scale
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(0)
x = rng.standard_normal((2, 3, 4))
sc = Scale(2.5)
ok, worst, detail = gradcheck(sc, [x], eps=1e-5, atol=1e-9, label="Scale")
sc.zero_grad()
sc(x)
sc.backward(x)
one = sc.w.grad.copy()
sc.backward(x)
acc = bool(np.allclose(sc.w.grad, 2.0 * one)) and float(np.max(np.abs(one))) > 1e-9
card("最小的一个层，只判两件事：正向 y = w·x 的数值，和反向的两件产出（dx = w·g 回传、dw 累加进参数）。"
     "backward 连跑两次是为了区分 += 和 = —— 覆盖写法数值照样对、gradcheck 也照样绿，"
     "只有连跑两次才露出来。",
     [("x", x, "B=2, T=3, C=4 的随机正态"), ("w", sc.w, "Scale(2.5)，一个标量参数"),
      ("g（上游梯度）", x, "gradcheck 拿它当余因子；手工那两次也用它")],
     [("y = 2.5·x", sc(x)), ("dw 跑一次", one, "应等于 Σ g·x"),
      ("dw 跑两次", sc.w.grad, "应是跑一次的两倍，不是相等")],
     ok and acc,
     f"最大相对误差 {worst:.2e}（阈值 1e-4）；一次 backward dw={brief(one)}，两次 dw={brief(sc.w.grad)}")
if not ok:
    print(detail)
elif not acc:
    print("  ↑ 数值是对的，但梯度没累加：backward 两次应该得到两倍的量。")
""",
    ),
    dict(
        id="TODO-5/6", stage="s03a", label="Linear 的 forward + backward",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import Linear
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(0)
xin, xin2 = rng.standard_normal((2, 4, 5)), rng.standard_normal((3, 2, 4))
lin, nl = Linear(5, 3, rng=rng), Linear(4, 6, use_bias=False, rng=rng)
ok1, w1, d1 = gradcheck(lin, [xin], eps=1e-5, atol=1e-9, label="Linear")
ok2, w2, d2 = gradcheck(nl, [xin2], eps=1e-5, atol=1e-9, label="Linear 无 bias")
g = rng.standard_normal((2, 4, 3))                  # 假装的"上游梯度"，形状 = y 的形状
lin.zero_grad()
lin(xin)
dx = lin.backward(g)
ok = bool(ok1 and ok2 and dx.shape == xin.shape)
card("第一个真层。正向 y = x@Wᵀ + b；反向必须同时交出三样东西："
     "dx = g@W（返回值，形状要和 x 一模一样，不然上一层接不住）、"
     "dW = g 与 x 的外积求和（形状跟 W 同形）、db = 对样本维求和（一维，跟 b 同形）。"
     "这三样写错任何一种，代码都不报错、loss 照样降 —— 只有有限差分对拍抓得住，所以两组各跑一次 gradcheck。"
     "第二组把 bias 关掉，判的是 use_bias=False 时不许偷偷算 db。",
     [("x（第一组）", xin, "B=2, T=4, in_f=5"),
      ("lin = Linear(5→3)", f"W={tuple(lin.W.shape)}、b={tuple(lin.b.shape)}", "参数由它自己随机初始化"),
      ("g（上游梯度）", g, "形状就是 y 的形状 (2,4,3)"),
      ("x（第二组）", xin2, "B=3, T=2, in_f=4"),
      ("nl = Linear(4→6)", f"W={tuple(nl.W.shape)}、nl.b is None：{nl.b is None}", "use_bias=False 那一组")],
     [("y = lin(x)", lin(xin), "期望 (2,4,3)"), ("dx = lin.backward(g)", dx, f"期望和 x 同形 {tuple(xin.shape)}"),
      ("dW", lin.W.grad, f"期望跟 W 同形 {tuple(lin.W.shape)}"),
      ("db", lin.b.grad, f"期望跟 b 同形 {tuple(lin.b.shape)}，一维"),
      ("y = nl(x2)", nl(xin2), "期望 (3,2,6)")],
     ok,
     f"有 bias 最大相对误差 {w1:.2e}；无 bias {w2:.2e}（阈值 1e-4）；dx 形状 {tuple(dx.shape)} == x 形状：{dx.shape == xin.shape}")
if not ok:
    print(d1 if not ok1 else "", d2 if not ok2 else "")
""",
    ),
    dict(
        id="TODO-7", stage="s03b", label="Embedding 的查表与散射累加",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import Embedding
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(1)
emb = Embedding(11, 4, rng=rng)
ids = rng.integers(0, 11, size=(2, 5))
ok, worst, detail = gradcheck(emb, [ids], eps=1e-5, atol=1e-9)
dup = np.array([[3, 3, 3, 0, 7]])
emb.zero_grad()
emb(dup)
out = emb(dup)
emb.backward(np.ones((1, 5, 4)))
acc = abs(emb.weight.grad[3].sum() - 12.0) < 1e-9      # 3 次 × 4 维
clean = bool(np.all(emb.weight.grad[1] == 0.0))        # 没用到的行必须是 0
ok2 = ok and acc and clean
card("查表 + 散射累加。正向是 ids → weight 的对应行；反向把 g 按 id 加回 weight 的对应行，"
     "于是有两件事必须同时成立：出现三次的 id 要加三份（+= 不是 =），没用到的行必须精确为 0。"
     "重复 id 那一组是特意构造的 —— 随机 ids 里到底有没有重复全看运气，"
     "把覆盖写法（=）放过去一次，它就永远抓不到这个 bug。",
     [("ids", ids, "词表大小 11，形状 (2,5)"), ("emb.weight", emb.weight, "V=11 行，每行 C=4"),
      ("dup", dup, "故意让 3 出现三次"), ("g（上游梯度）", "np.ones((1,5,4))")],
     [("out = emb(dup)", out, "每行就是 weight 里对应那一行"),
      ("weight.grad[3]", emb.weight.grad[3], "三次 × 四维 = 12，逐元素是 3"),
      ("weight.grad[1]", emb.weight.grad[1], "id 1 没出现过，必须全 0")],
     ok2,
     f"gradcheck 误差 {worst:.2e}；重复 id 累加={emb.weight.grad[3].sum()} 期望 12；"
     f"未用行为 0：{clean}")
if not ok2:
    print(detail)
""",
    ),
    dict(
        id="TODO-8", stage="s03b", label="RMSNorm 的耦合项",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import RMSNorm
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(2)
xin = rng.standard_normal((2, 5, 8))
rm = RMSNorm(8)
out = rm(xin)
ok, worst, detail = gradcheck(rm, [xin], eps=1e-5, atol=1e-9)
card("RMSNorm 反向里那个容易被省掉的耦合项。因为每一行的长度被强行归成 √C，"
     "动一个元素会牵动同组其它元素，所以除了逐元素的 1/r，还得再减一项 −r·x̂·s（s = Σ g·x̂）。"
     "漏掉它：正向一模一样、形状一模一样、loss 照样降，只有 gradcheck 给到 1e-1 量级。",
     [("x", xin, "B=2, T=5, C=8 的随机正态"), ("w", rm.w, "尺度参数，初始全 1")],
     [("y = rm(x)", out, "期望 (2,5,8)，最后一维被归一化"),
      ("每行平方和的均值", float(np.mean(np.sum(out * out, -1))), "应≈8，也就是每行长度被归成 √8")],
     ok,
     f"最大相对误差 {worst:.2e}（阈值 1e-4；少了 −r·x̂·s 这一项会挂在 1e-1 量级）")
if not ok:
    print(detail)
""",
    ),
    dict(
        id="TODO-9", stage="s03b", label="GELU 的导数",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import GELU
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(3)
xin = rng.standard_normal((2, 3, 4)) * 2.0
y = GELU()(xin)
ok, worst, detail = gradcheck(GELU(), [xin], eps=1e-5, atol=1e-9)
card("GELU 的导数：tanh 近似式里套着 tanh 自己的导数，少乘一层 (1−t²) 在 x=0 附近看不出来，"
     "所以取幅度 ×2 的随机点，把正负两侧都摆进来。正向顺便确认它是「软 relu」：x=−2 几乎为 0、x=+2 ≈ x。",
     [("x", xin, "幅度 ×2，专门覆盖 ±2 那一带")],
     [("y = GELU(x)", y, "负侧压向 0，正侧贴着 x")],
     ok,
     f"最大相对误差 {worst:.2e}（判定用 相对≤1e-4 或 绝对≤1e-9，后者是为了不被浮点噪声误判）")
if not ok:
    print(detail)
""",
    ),
    dict(
        id="TODO-10", stage="s03b", label="MLP 的三层串联反向",
        needs=["TODO-5/6", "TODO-9"],
        code=r"""
import numpy as np
from minigpt.nn import MLP
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(4)
xin = rng.standard_normal((2, 3, 6))
net = MLP(6, 16, rng=rng)
ok, worst, detail = gradcheck(net, [xin], eps=1e-5, atol=1e-9)
card("MLP 是 fc1 → GELU → fc2 三层串联，反向没有新数学，只是把正向**倒着走一遍**："
     "g 先过 fc2 的导数，再过 act 的导数，最后过 fc1 的导数。"
     "最常见的错是少走一次 act.backward —— 三处形状全都合法（C→H→C 一路对得上），"
     "代码跑通、loss 也降，实测最大相对误差 0.82。这条就是收那三行学费的。",
     [("x", xin, "B=2, T=3, C=6 的随机正态"), ("net = MLP(6→16)", f"隐藏维 H=16，参数 {len(net.named_parameters())} 个张量")],
     [("y = net(x)", net(xin), "期望 (2,3,6)：进 6 出 6"),
      ("中间 h", "(2,3,16)", "forward 里缓存的那一层，反向要拿它")],
     ok,
     f"最大相对误差 {worst:.2e}（阈值 1e-4）")
if not ok:
    print(detail)
""",
    ),
    dict(
        id="TODO-11", stage="s03c", label="CrossEntropy 的稳定性与 ln(V) 基准",
        needs=[],
        code=r"""
import math
import numpy as np
from minigpt.nn import CrossEntropy
from minigpt.gradcheck import gradcheck
rng = np.random.default_rng(5)
B, T, V = 2, 5, 9
lg, tg = rng.standard_normal((B, T, V)) * 2.0, rng.integers(0, V, (B, T))
ok, worst, detail = gradcheck(CrossEntropy(), [lg, tg], eps=1e-5, atol=1e-9)
base = CrossEntropy()(np.zeros((1, 4, 8)), np.array([[1, 2, 3, 0]]))
lnV = abs(base - math.log(8)) < 1e-12
huge = CrossEntropy()(np.full((1, 3, 8), 900.0), np.array([[1, 2, 3]]))
noflow = np.isfinite(huge)
ok2 = ok and lnV and noflow
card("loss 层一次判三件事。① 数值：loss = mean(−log p[target])；梯度是 (prob − onehot) 再除以元素个数，"
     "形状回给 logits（余因子是 1，因为 loss 是标量）。"
     "② 基准线：logits 全 0 也就是均匀分布时，loss 必须正好 ln V —— 以后你看训练曲线第一眼看的就是它。"
     "③ 稳定性：logits=900 时 e^900 直接 inf，不减最大值的话 loss 会炸成 nan，"
     "而减完之后 loss 的数值一模一样 —— 只有这一条判据能证明那一步不是可选的。",
     [("logits", lg, "B=2, T=5, V=9，幅度 ×2"), ("targets", tg, "0..8 的随机整数，形状 (2,5)"),
      ("均匀对照", "zeros((1,4,8)) + targets=[[1,2,3,0]]", "loss 应正好 = ln 8"),
      ("极端对照", "full((1,3,8), 900.0)", "loss 必须是有限数")],
     [("loss", f"{float(CrossEntropy()(lg, tg)):.6f}（标量）"),
      ("dL/dlogits", f"形状 {tuple(lg.shape)}，等于 (prob − onehot)/{tg.size}"),
      ("均匀 logits 的 loss", f"{base:.6f}", f"期望 ln(8)={math.log(8):.6f}"),
      ("logits=900 的 loss", f"{huge:.4f}", "要有限，不能是 inf/nan")],
     ok2,
     f"gradcheck 误差 {worst:.2e}；ln V 基准：{lnV}；900 那组不溢出：{noflow}")
if not ok2:
    print(detail)
""",
    ),

    # ------------------------------------------------------------------ 阶段 4
    dict(
        id="TODO-14", stage="s04", label="softmax 逐行反向",
        needs=[],
        code=r"""
import numpy as np
from minigpt.attention import softmax_backward
rng = np.random.default_rng(6)
z = rng.standard_normal((1, 2, 4, 5))
m = np.max(z, axis=-1, keepdims=True)
e = np.exp(z - m)
p = e / e.sum(-1, keepdims=True)
gp = rng.standard_normal(p.shape)
g = softmax_backward(p, gp)
# 性质一：每行梯度之和为 0（"行和恒为 1"的守恒要求）
sum0 = float(np.max(np.abs(g.sum(-1))))
# 性质二：gp 是常数时梯度必须为 0（softmax 对整行 +常数不变）
const = float(np.max(np.abs(softmax_backward(p, np.ones_like(p) * 2.5))))
ok = sum0 < 1e-9 and const < 1e-9
card("softmax 反向的两个守恒性质。它们不依赖任何具体数值，所以比数值对拍更硬："
     "① 每一行的梯度之和必须为 0 —— softmax 每行输出之和恒等于 1，谁都改不动这个总和；"
     "② 上游给一个常数余向量（整行 +c）时梯度必须为 0 —— softmax(z+c) = softmax(z)。"
     "g_i = p_i·(gp_i − Σ_j p_j·gp_j) 里那个减号或那次求和写错，这两条立刻同时红。",
     [("z", z, "形状 (1,2,4,5)，最后一维才是 softmax 的那一维"),
      ("p = softmax(z)", p, "每行和为 1"), ("gp（上游对 p 的梯度）", gp, "同形状随机数")],
     [("g = softmax_backward(p, gp)", g),
      ("每行梯度之和 max|Σg|", f"{sum0:.2e}", "要 0"),
      ("常数余向量下的 max|g|", f"{const:.2e}", "也要 0")],
     ok,
     f"每行梯度和 max|Σg|={sum0:.2e}（要 0）；常数余向量下 max|g|={const:.2e}（要 0）")
""",
    ),
    dict(
        id="TODO-12", stage="s04", label="注意力 forward 的形状与因果性",
        needs=["TODO-5/6", "TODO-8", "TODO-9", "TODO-14"],
        code=r"""
import numpy as np
from minigpt.attention import CausalSelfAttention
from minigpt.config import Config
from minigpt.nn import Linear, RMSNorm, GELU   # 借用阶段 3 的层，确保它们已就位
cfg = Config(d_model=8, n_head=2, vocab=9, ctx=6)
rng = np.random.default_rng(7)
att = CausalSelfAttention(cfg, rng=rng)
x = rng.standard_normal((2, 5, 8))
y = att(x)
probs = att._probs
shape_ok = y.shape == (2, 5, 8) and probs.shape == (2, 2, 5, 5)
row_sum_ok = float(np.max(np.abs(probs.sum(-1) - 1.0))) < 1e-9
# 因果性：上三角（j>i）的概率必须精确为 0。用掩码一次算完，别逐行切片 ——
# 最后一行的切片是空数组，np.max 会炸。
T = probs.shape[-1]
upper = np.triu(np.ones((T, T), dtype=bool), k=1)
zero_ok = bool(np.max(np.abs(probs * upper)) == 0.0)
ok = shape_ok and row_sum_ok and zero_ok
card("注意力**正向**的两件事：形状与因果性。y 必须回到 (B,T,C) —— 拆头→算→合头→proj 一整圈，"
     "进出长度一样；probs 必须是 (B,H,T,T)，每头一张 T×T。每行和为 1 是 softmax 的账，"
     "上三角 j>i 精确为 0 是因果掩码的账。掩码写成下三角、或者 bias 加错了方向，"
     "正向数值照样漂亮，一旦训练就是在偷看未来。反向是 TODO-13，这条不碰。",
     [("x", x, "B=2, T=5, C=8；H=2，每头 hd=4"),
      ("att = CausalSelfAttention(cfg)", f"d_model={cfg.d_model}, n_head={cfg.n_head}, ctx={cfg.ctx}",
       "q/k/v/proj 四个 Linear 由它自己随机初始化")],
     [("y = att(x)", y, "期望 (2,5,8)"), ("att._probs", probs, "期望 (2,2,5,5)：每头一张 T×T"),
      ("每行和的偏差", f"{np.max(np.abs(probs.sum(-1) - 1)):.2e}", "要 0"),
      ("上三角最大概率", f"{float(np.max(np.abs(probs * upper))):.2e}", "要精确 0，这就是「不许偷看」")],
     ok,
     f"形状 {y.shape}/{probs.shape}；行和偏差 {np.max(np.abs(probs.sum(-1)-1)):.2e}；"
     f"上三角概率全为 0：{zero_ok}")
""",
    ),
    dict(
        id="TODO-13", stage="s04", label="注意力 backward 全链",
        needs=["TODO-5/6", "TODO-8", "TODO-9", "TODO-12", "TODO-14"],
        code=r"""
import numpy as np
from minigpt.attention import CausalSelfAttention
from minigpt.config import Config
from minigpt.gradcheck import gradcheck
cfg = Config(d_model=8, n_head=2, vocab=9, ctx=6)
rng = np.random.default_rng(8)
xin = rng.standard_normal((2, 5, 8))
mod = CausalSelfAttention(cfg, rng=rng)
ok, worst, detail = gradcheck(mod, [xin], eps=1e-5, atol=1e-9)
g = rng.standard_normal((2, 5, 8))
mod(xin)
gx = mod.backward(g)
ok = bool(ok and gx.shape == xin.shape)
card("整条注意力反向链，一段都不能少：g 先倒着过 proj，再过「合头」（把 (B,H,T,hd) 摊回 (B,T,C)），"
     "然后分头进 v / k / q 三个 Linear，中间还夹着 1/√hd 的缩放和 softmax_backward。"
     "少写最外面那一段时三处形状全都合法、loss 也降，只有有限差分对拍会红。"
     "q/k 的梯度常常只有 1e-8 量级，所以除了相对阈值 1e-4 还带 1e-9 的绝对容差，别把浮点噪声当成 bug。",
     [("x", xin, "B=2, T=5, C=8；H=2，每头 hd=4"),
      ("mod = CausalSelfAttention(cfg)", f"d_model={cfg.d_model}, n_head={cfg.n_head}",
       f"反向要累加进 {len(mod.named_parameters())} 个参数张量")],
     [("y = mod(x)", mod(xin), "期望 (2,5,8)"),
      ("gx = mod.backward(g)", gx, f"期望和 x 同形 {tuple(xin.shape)}")],
     ok,
     f"最大相对误差 {worst:.2e}（阈值 1e-4 + 绝对容差 1e-9）；gx 形状 {tuple(gx.shape)}")
if not ok:
    print(detail)
""",
    ),

    # ------------------------------------------------------------------ 阶段 5
    dict(
        id="TODO-15", stage="s05", label="Block.forward 的残差结构",
        needs=["TODO-5/6", "TODO-8", "TODO-9", "TODO-10", "TODO-12"],
        code=r"""
import numpy as np
from minigpt.config import Config
from minigpt.gpt import Block
cfg = Config(vocab=9, d_model=8, n_layer=2, n_head=2, ctx=6)
rng = np.random.default_rng(0)
x = rng.standard_normal((2, 5, 8))
blk = Block(cfg, rng=rng)
y = blk(x)
shape_ok = y.shape == x.shape and y.dtype == np.float64
finite_ok = bool(np.isfinite(y).all())
# 残差恒等：把两个「残差出口」的权重清零，这一层必须原样交出输入
probe = Block(cfg, rng=rng)
probe.attn.proj.W.data[...] = 0.0
probe.mlp.fc2.W.data[...] = 0.0
ident = np.array_equal(probe(x), x)
ok = shape_ok and finite_ok and ident
card("残差结构：x1 = x + attn(norm1(x))，x2 = x1 + mlp(norm2(x1))。"
     "判法很干净：把两个残差出口（attn.proj.W 和 mlp.fc2.W）清零，子层输出必为 0，"
     "这一层就得**逐比特**把输入原样交出来。忘了加回主干、或者第二条 norm 吃的是 x 而不是 x1，"
     "这两种错在 loss 曲线上都看不出来，这里会红。",
     [("x", x, "B=2, T=5, C=8；H=2，每头 hd=4"),
      ("blk = Block(cfg)", f"d_model={cfg.d_model}, n_head={cfg.n_head}, ctx={cfg.ctx}", "随机初始化"),
      ("probe", "同一结构，但 proj.W 和 fc2.W 被清零", "专门用来验残差恒等")],
     [("y = blk(x)", y, "期望 (2,5,8) 且全有限"), ("probe(x)", probe(x), "期望逐比特等于 x")],
     ok,
     f"形状 {tuple(y.shape)}：{shape_ok} · 全有限：{finite_ok} · "
     f"残差出口置零后 forward 逐比特等于输入：{ident}"
     "（不等于输入 = 忘了把子层加回主干，或者第二条 norm 吃的是 x 而不是 x1）")
""",
    ),
    dict(
        id="TODO-16", stage="s05", label="Block.backward 的两路残差",
        needs=["TODO-5/6", "TODO-8", "TODO-9", "TODO-10", "TODO-12", "TODO-13", "TODO-14", "TODO-15"],
        code=r"""
import numpy as np
from minigpt.config import Config
from minigpt.gpt import Block
from minigpt.gradcheck import gradcheck
cfg = Config(vocab=9, d_model=8, n_layer=2, n_head=2, ctx=6)
rng = np.random.default_rng(1)
x = rng.standard_normal((2, 5, 8))
blk = Block(cfg, rng=rng)
ok, worst, detail = gradcheck(blk, [x], eps=1e-5, tol=1e-4, atol=1e-8, label="Block")
probe = Block(cfg, rng=np.random.default_rng(2))
probe.attn.proj.W.data[...] = 0.0
probe.mlp.fc2.W.data[...] = 0.0
g = rng.standard_normal((2, 5, 8))
probe(x)                                  # 必须先 forward：子层要拿缓存
out_g = probe.backward(g)
passthrough = bool(np.array_equal(out_g, g))
n_par = len(probe.named_parameters())
card("Block 反向有两条路要合：主干那条把 g 原样加回去（+= g），支路那条过子层的 backward。"
     "判法还是把两个残差出口清零 —— 这时子层对梯度是死的，backward 必须把 g **一字不差透传**出来。"
     "少写那一路 += g，返回的就只剩支路那半，而形状依然完全正确。",
     [("x", x, "B=2, T=5, C=8"), ("blk = Block(cfg)", "用来跑 gradcheck"),
      ("probe", "同一结构，proj.W 与 fc2.W 清零", "用来跑透传"),
      ("g（上游梯度）", g, "形状 (2,5,8)")],
     [("y = blk(x)", blk(x), "期望 (2,5,8)"),
      ("gx = probe.backward(g)", out_g, f"期望逐比特等于 g，参数共 {n_par} 个张量要吃梯度")],
     ok and passthrough,
     f"最大相对误差 {worst:.2e}（阈值 1e-4，绝对容差 1e-8）；"
     f"残差出口置零时梯度原样透传：{passthrough}（少一路 +g 就只剩支路那半）")
if not ok:
    print(detail)
""",
    ),
    dict(
        id="TODO-17", stage="s05", label="TiedHead：借来的权重 + 梯度累加",
        needs=[],
        code=r"""
import numpy as np
from minigpt.gpt import TiedHead
from minigpt.gradcheck import gradcheck
from minigpt.nn import Parameter
rng = np.random.default_rng(3)
V, C = 7, 5
w = Parameter(rng.standard_normal((V, C)) * 0.3, name="tok.weight")
head = TiedHead(w)
x = rng.standard_normal((2, 3, C))
out = head(x)
ref = (x.reshape(-1, C) @ w.data.T).reshape(2, 3, V)
val_ok = bool(np.allclose(out, ref, atol=1e-12))
ok, worst, detail = gradcheck(head, [x], eps=1e-5, tol=1e-4, atol=1e-8, label="TiedHead")
head.zero_grad()
out = head(x)                             # 重新 forward，把缓存指回这份 x
g = rng.standard_normal(out.shape)
gx = head.backward(g)
first = w.grad.copy()
head.backward(g)
doubled = bool(np.allclose(w.grad, 2.0 * first)) and float(np.abs(first).max()) > 0
gx_ref = (g.reshape(-1, V) @ w.data).reshape(x.shape)
gx_ok = bool(np.allclose(gx, gx_ref, atol=1e-12))
allok = val_ok and gx_ok and ok and doubled
card("输出头不新建权重，直接借 tok.weight 算 x@Wᵀ（V×C）—— 省一份参数，也省一次过拟合。"
     "反向因此要同时交出两样：gx = g@W 回主干，以及把梯度**累加**进借来的 weight。"
     "它同时还在被词嵌入那一头用，写成 = 而不是 += 就等于把另一路的贡献抹掉；"
     "所以这里连跑两次 backward，看 weight.grad 是不是正好翻倍。",
     [("x", x, "B=2, T=3, C=5"), ("w（借来的 weight）", w, "V=7 行 × C=5，和词嵌入同一个对象"),
      ("g（上游梯度）", g, "形状 (2,3,7)，最后一维是词表")],
     [("logits = head(x)", out, "期望 (2,3,7)"), ("gx = head.backward(g)", gx, "期望 (2,3,5)"),
      ("weight.grad 跑一次", first, "对照 gᵀ@x"), ("weight.grad 跑两次", w.grad, "应是上一行的两倍")],
     allok,
     f"数值等于 x@Wᵀ：{val_ok} · gx 等于 g@W：{gx_ok} · gradcheck 最大相对误差 {worst:.2e} · "
     f"跑两次 backward 后 weight.grad 翻倍：{doubled}（写成 = 而不是 += 就不是翻倍，"
     f"第一次最大梯度 {np.abs(first).max():.4g}）")
if not allok:
    print(detail)
""",
    ),
    dict(
        id="TODO-18", stage="s05", label="GPT.forward：词嵌入 + 位置嵌入 + 末端 norm",
        needs=["TODO-5/6", "TODO-8", "TODO-9", "TODO-10", "TODO-12", "TODO-15", "TODO-17"],
        code=r"""
import numpy as np
from minigpt.config import Config
from minigpt.gpt import GPT
cfg = Config(vocab=9, d_model=8, n_layer=2, n_head=2, ctx=6)
rng = np.random.default_rng(4)
model = GPT(cfg, rng=rng)
ids = rng.integers(0, cfg.vocab, size=(2, 5))
lg = model.forward(ids)
shape_ok = lg.shape == (2, 5, cfg.vocab) and np.isfinite(lg).all()
# 位置嵌入真的参与了计算：把它清零，输出必须变
keep = model.pos.weight.data.copy()
model.pos.weight.data[...] = 0.0
zeroed = model.forward(ids)
model.pos.weight.data[...] = keep
pos_ok = not np.array_equal(zeroed, lg)
# 同一批 token 换个顺序，输出必须变（注意力有因果掩码，所以这条不是白给的）
perm = model.forward(ids[:, ::-1])
order_ok = not np.allclose(perm[0], lg[0])
# 超过 ctx 要拦住（位置嵌入只有 ctx 行）
try:
    model.forward(rng.integers(0, cfg.vocab, size=(1, cfg.ctx + 2)))
    guard = False
except (AssertionError, IndexError):
    guard = True
ok = bool(shape_ok and pos_ok and order_ok and guard)
card("整机装配：ids → tok 嵌入 + pos 嵌入 → n_layer 个 Block → 末端 RMSNorm → tied 头 → logits。"
     "四个探针各盯一件容易漏的事：形状 (B,T,V)；pos.weight 清零后输出必须变"
     "（忘了把位置嵌入加进去是最常见的装配漏项）；同一批 token 换个顺序输出必须变"
     "（不变说明 pos 或因果掩码根本没起作用）；T>ctx 必须被拦住（pos 只有 ctx 行，不拦就是静默越界）。",
     [("ids", ids, f"B=2, T=5，词表 {cfg.vocab}"),
      ("model = GPT(cfg)", f"d_model={cfg.d_model}, n_layer={cfg.n_layer}, n_head={cfg.n_head}, ctx={cfg.ctx}"),
      ("越界探针", f"ids 形状 (1, {cfg.ctx + 2})", "超过 ctx，应当被拦住而不是算出垃圾")],
     [("logits = model.forward(ids)", lg, f"期望 (2,5,{cfg.vocab})"),
      ("pos 清零后", zeroed, "期望和上一行不同"),
      ("token 反序后", perm[0], "期望和 lg[0] 不同"),
      ("T>ctx", "被拦住（AssertionError/IndexError）" if guard else "没拦住")],
     ok,
     f"形状 {tuple(lg.shape)} 且全有限：{shape_ok} · 位置嵌入清零后输出会变：{pos_ok} · "
     f"打乱顺序输出会变：{order_ok} · T>ctx 被拦住：{guard}")
""",
    ),
    dict(
        id="TODO-19", stage="s05", label="loss_backward 的移位方向（预测下一个）",
        needs=["TODO-5/6", "TODO-8", "TODO-9", "TODO-10", "TODO-11", "TODO-12", "TODO-15",
               "TODO-17", "TODO-18"],
        code=r"""
import math
import numpy as np
from minigpt.config import Config
from minigpt.gpt import GPT
from minigpt.nn import CrossEntropy
cfg = Config(vocab=9, d_model=8, n_layer=2, n_head=2, ctx=6)
rng = np.random.default_rng(5)
model = GPT(cfg, rng=rng)
ids = rng.integers(0, cfg.vocab, size=(2, 6))
lg = model.forward(ids)
ce = CrossEntropy()
want = float(ce(lg[:, :-1, :], ids[:, 1:]))          # 位置 t 预测 t+1
self_want = float(ce(lg[:, :-1, :], ids[:, :-1]))    # 位置 t 预测自己
model.zero_grad()
got = float(model.loss_backward(ids))
shift_ok = abs(got - want) < 1e-12
not_self = abs(got - self_want) > 1e-6
lnv_ok = abs(got - math.log(cfg.vocab)) < 0.8        # 随机初始化时 loss≈ln V
# 端到端有限差分：loss_backward 交出的梯度，和"把权重挪一格看 loss 变多少"对得上吗
w = model.tok.weight
ana = w.grad.copy()
errs = []
for (r, c) in ((1, 2), (4, 0), (0, 3)):
    saved = float(w.data[r, c])
    e = 1e-5
    w.data[r, c] = saved + e
    hi = float(model.loss_backward(ids))
    w.data[r, c] = saved - e
    lo = float(model.loss_backward(ids))
    w.data[r, c] = saved
    num = (hi - lo) / (2 * e)
    errs.append(abs(num - ana[r, c]) / max(1e-8, abs(num), abs(ana[r, c])))
fd_ok = max(errs) < 1e-4
ok = bool(shift_ok and not_self and lnv_ok and fd_ok)
card("这一条的全部内容就是移位方向：位置 t 的输出去预测 t+1，所以 "
     "loss = CE(logits[:, :-1, :], ids[:, 1:])。写成「预测自己」（ids[:, :-1]）时 loss 会明显偏低 —— "
     "模型学的是把下一个 token 抄进自己，训练曲线好看但生成一塌糊涂，"
     "所以除了要求它等于正确移位，还要求它**不等于**自预测。"
     "另外两件事：随机初始化时 loss 应落在 ln V 附近；loss_backward 交出的梯度要经得起端到端有限差分。",
     [("ids", ids, "B=2, T=6"), ("model", f"GPT(cfg)，词表 {cfg.vocab}", "整机 forward + 反向一起走"),
      ("有限差分探针", "tok.weight 的 (1,2)/(4,0)/(0,3) 三个元素，ε=1e-5")],
     [("loss = model.loss_backward(ids)", f"{got:.6f}（标量）", "同时把梯度写进所有参数"),
      ("正确移位 CE(logits[:,:-1], ids[:,1:])", f"{want:.6f}", "loss 应该等于它"),
      ("预测自己 CE(logits[:,:-1], ids[:,:-1])", f"{self_want:.6f}", "loss 应该**不等于**它"),
      ("ln V 基准", f"{math.log(cfg.vocab):.6f}", "初始 loss 应在它附近 ±0.8"),
      ("三个抽查点的相对误差", f"{max(errs):.2e}", "阈值 1e-4")],
     ok,
     f"loss 等于正确移位：{shift_ok} · 不等于「预测自己」：{not_self} · "
     f"初始就在 ln V 附近：{lnv_ok} · 有限差分最大相对误差 {max(errs):.2e}")
""",
    ),
    dict(
        id="TODO-20", stage="s05", label="CharTokenizer 的 encode/decode 与未知字符",
        needs=[],
        code=r"""
from minigpt.tokenizer import CharTokenizer
text = "hello world .\nbrave new day .\n" * 3
tok = CharTokenizer(text)
ids = tok.encode(text)
rt = tok.decode(ids) == text
rng_ok = min(ids) >= 0 and max(ids) < tok.vocab_size
vocab_ok = tok.vocab_size == len(set(text))
sorted_ok = tok.chars() == "".join(sorted(set(text)))
first_ok = tok.char_at(0) == min(set(text))          # id 0 = 码点最小的字符，不是"未找到"
probe = text + "\t"                                  # 制表符不在词表里
drop_ok = len(tok.encode(probe)) == len(ids)
try:
    tok.encode(probe, drop_unknown=False)
    strict_ok = False
except KeyError:
    strict_ok = True
ok = all([rt, rng_ok, vocab_ok, sorted_ok, first_ok, drop_ok, strict_ok])
card("字符词表的定义就一句话：语料里出现过的字符按码点排序，id = 下标。"
     "判六件事：往返 decode(encode(text))==text；id 全落在 [0,V)；V 等于去重字符数且真按码点排；"
     "id 0 是那个最小码点的字符、而不是「没找到」返回的假值（最阴的一种坑，"
     "char_at(0) 返回空串时前面几条照样绿）；陌生字符默认丢掉；drop_unknown=False 必须抛 KeyError，"
     "不许静默丢 —— 静默丢会让模型学到「标点不存在」。",
     [("text", f"{len(text)} 字符，去重后 {len(set(text))} 个", "重复三遍，保证每个字符都出现过"),
      ("probe", "text + 一个制表符 '\\t'", "制表符不在词表里")],
     [("ids = encode(text)", ids[:8], f"共 {len(ids)} 个"), ("chars() 词表", tok.chars()),
      ("char_at(0)", repr(tok.char_at(0)), "应是码点最小的那个字符"),
      ("encode(probe)", f"{len(tok.encode(probe))} 个（丢掉 \\t 后与原文等长）"),
      ("encode(probe, drop_unknown=False)", "抛 KeyError" if strict_ok else "没抛，静默丢了")],
     ok,
     f"往返一致：{rt} · id 全在范围内：{rng_ok} · vocab={tok.vocab_size} 且按码点排序："
     f"{vocab_ok and sorted_ok} · id 0 是 {tok.char_at(0)!r}（不是假值陷阱）：{first_ok} · "
     f"陌生字符默认丢弃：{drop_ok} · drop_unknown=False 抛 KeyError：{strict_ok}")
""",
    ),
    dict(
        id="TODO-21", stage="s05", label="Batcher 的窗口切法与 x/y 移位",
        needs=[],
        code=r"""
import numpy as np
from minigpt.dataset import Batcher
ids = np.arange(60, dtype=np.int64) % 7
bat = Batcher(ids, 8)
rng = np.random.default_rng(0)
x, y = bat(4, rng)
shape_ok = x.shape == (4, 8) and y.shape == (4, 8)
in_range = int(x.min()) >= 0 and int(x.max()) < 7
shift_ok = np.array_equal(y[:, :-1], x[:, 1:])
# 每个窗口都必须是原序列里**连续**的一段
contig = any(np.array_equal(ids[a:a + 8] % 7, x[i])
             for i in range(x.shape[0]) for a in range(len(ids) - 7))
len_ok = len(bat) == len(ids) - 8                   # 合法起点个数 = max_start+1
x2, _ = bat(4, np.random.default_rng(1))
x3, _ = bat(4, np.random.default_rng(1))
rand_ok = (not np.array_equal(x, x2)) and np.array_equal(x2, x3)
try:
    Batcher(np.arange(5), 8)
    short_ok = False
except ValueError:
    short_ok = True
ok = bool(shape_ok and in_range and shift_ok and contig and len_ok and rand_ok and short_ok)
card("切窗口：从 ids 里随机取长度 ctx 的**连续**一段当 x，y 是同一窗口右移一位"
     "（x[t] 的下一个 token 才是监督信号）。这里判七件事，每件都对应一种真实写法错误："
     "y[:, :-1]==x[:, 1:] 是移位方向；「每行都是原序列里连续的一段」是防你把窗口内打乱；"
     "窗口总数 = len(ids)−ctx；换 rng 换窗口、同 rng 必须可复现（不然断点续训没法对齐）；"
     "语料比 ctx 短要 ValueError 而不是返回空数组。",
     [("ids", ids[:10], f"共 {len(ids)} 个，0..6 循环"), ("ctx", 8), ("batch", 4),
      ("rng", "default_rng(0)；另外两批用 default_rng(1) 各取一次")],
     [("x", x, "期望 (4,8)"), ("y", y, "期望 (4,8)，且 y[:, :-1] == x[:, 1:]"),
      ("len(bat)", f"{len(bat)}（应等于 {len(ids) - 8}）")],
     ok,
     f"形状 (4,8)/(4,8)：{shape_ok} · id 在词表内：{in_range} · y[:, :-1]==x[:, 1:]：{shift_ok} · "
     f"每行都是原序列里连续的一段：{contig} · 窗口总数 {len(bat)}=={len(ids) - 8}：{len_ok} · "
     f"换 rng 换窗口且同 rng 可复现：{rand_ok} · 语料太短时报 ValueError：{short_ok}")
""",
    ),
    dict(
        id="TODO-22", stage="s05", label="AdamW.step：偏差修正 + 解耦衰减",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import Parameter
from minigpt.optim import AdamW
p = Parameter(np.zeros((3, 4)) + 0.5, name="W")
o = AdamW([p], lr=1e-2, weight_decay=0.0, beta1=0.9, beta2=0.95)
p.grad[...] = 3.0
before = p.data.copy()
o.step()
mv1 = float(np.max(np.abs(p.data - before)))          # 有偏差修正：正好 lr
before = p.data.copy()
o.step()
mv2 = float(np.max(np.abs(p.data - before)))          # 第二步还是 lr
step_ok = abs(mv1 - 1e-2) < 1e-9 and abs(mv2 - 1e-2) < 1e-9

w = Parameter(np.zeros((3, 4)) + 0.5, name="W2")
b = Parameter(np.ones(4) * 0.5, name="bias", wd_exempt=True)
o2 = AdamW([w, b], lr=1e-2, weight_decay=0.1)
w.grad[...] = 3.0
b.grad[...] = -1.5
bw, bb = w.data.copy(), b.data.copy()
ret = o2.step()
mv_w = float(np.max(np.abs(w.data - bw)))
mv_b = float(np.max(np.abs(b.data - bb)))
decay_ok = abs(mv_w - (1e-2 + 0.5 * 1e-2 * 0.1)) < 1e-9   # 权重多走一份 lr·λ·w
bias_ok = abs(mv_b - 1e-2) < 1e-9                          # 被豁免：只走 lr

idle = Parameter(np.ones(3) * 2.0, name="idle")
o3 = AdamW([idle], lr=1e-2, weight_decay=0.0)
o3.step()
zero_ok = np.array_equal(idle.data, np.full(3, 2.0))
count_ok = o3.step_count == 1 and ret == 1e-2
ok = bool(step_ok and decay_ok and bias_ok and zero_ok and count_ok)
card("优化器两个最容易写错的点。① 偏差修正 m̂=m/(1−β₁ᵗ)、v̂=v/(1−β₂ᵗ)：梯度恒定时"
     "更新量正好等于 lr，第一步就是 lr（忘了修正只有 ≈0.45·lr，第二步还是 lr）。"
     "② 解耦衰减 w ← w − lr·(m̂/√v̂ + λw)：λ 只作用在不豁免的参数上，"
     "bias 和 norm 权重（wd_exempt=True）不许吃 decay。"
     "另外梯度为 0 时参数必须一动不动（decay 也不许偷偷动它），step() 返回真正用到的 lr。",
     [("W", "(3,4) 全 0.5", "grad 恒为 3.0，lr=1e-2，β=(0.9, 0.95)，wd=0"),
      ("w / b", "wd=0.1 那一组：w=(3,4) 全 0.5，b=(4,) 全 0.5 且 wd_exempt=True", "b 的 grad=-1.5"),
      ("idle", "grad 全 0 的参数", "应当一步都不动")],
     [("第一步移动", mv1, "期望正好 0.01"), ("第二步移动", mv2, "期望还是 0.01"),
      ("wd 组 W 移动", mv_w, "期望 0.01 + 0.1×0.5×0.01"), ("wd 组 bias 移动", mv_b, "期望只有 0.01"),
      ("step() 返回", ret, "期望 1e-2"), ("step_count", o3.step_count, "每 step 一次 +1")],
     ok,
     f"两步幅度都恰好 = lr（偏差修正 1−βᵗ 写对了）：{step_ok} · "
     f"weight decay 只作用于非豁免参数：{decay_ok and bias_ok} · "
     f"梯度为 0 时参数一动不动：{zero_ok} · step_count 递增且 step() 返回用到的 lr：{count_ok}")
""",
    ),
    dict(
        id="TODO-23", stage="s05", label="clip_grad_norm：全局 L2 与方向不变",
        needs=[],
        code=r"""
import math
import numpy as np
from minigpt.nn import Parameter
from minigpt.optim import clip_grad_norm
a = Parameter(np.zeros((2, 2)), name="a")
b = Parameter(np.zeros(6), name="b")
a.grad[...] = 3.0
b.grad[...] = -4.0
total = math.sqrt(4 * 9.0 + 6 * 16.0)
got = clip_grad_norm([a, b], 1.0)
scale = 1.0 / total
after = math.sqrt(float(np.sum(a.grad ** 2) + np.sum(b.grad ** 2)))
dir_ok = bool(np.allclose(a.grad, 3.0 * scale) and np.allclose(b.grad, -4.0 * scale))
small = Parameter(np.zeros(2), name="small")
small.grad[...] = 0.1
got2 = clip_grad_norm([small], 5.0)
untouched = bool(np.allclose(small.grad, 0.1)) and abs(got2 - math.sqrt(0.02)) < 1e-12
zero = Parameter(np.zeros(3), name="zero")
try:
    got3 = clip_grad_norm([zero], 1.0)
    no_nan = np.isfinite(got3) and np.allclose(zero.grad, 0.0)
except Exception as exc:
    got3, no_nan = float("nan"), False
ok = bool(abs(got - total) < 1e-12 and abs(after - 1.0) < 1e-12 and dir_ok and untouched and no_nan)
card("梯度裁剪的三个要点。① 范数是**全局**的：把所有参数的梯度拼成一个向量算 L2，"
     "逐张量各裁各的会悄悄改变张量之间的相对比例；② 返回的是裁剪**前**的范数，"
     "训练日志要看它的趋势，裁完永远是 1.0 就没信息了；③ 裁剪是整体等比缩放，方向不许变。"
     "另外两个边界：没超限时一点都不许动，全零梯度不许除出 nan（要么加 eps 要么提前返回）。",
     [("a", "(2,2)，grad 全 3.0"), ("b", "(6,)，grad 全 −4.0"),
      ("max_norm", 1.0, f"裁剪前全局 L2 = {total:.4f}"),
      ("对照小梯度", "small=(2,) 全 0.1，max_norm=5.0", "没超限"),
      ("对照全零", "zero=(3,) 全 0", "不许炸")],
     [("返回值（裁剪前范数）", got, f"期望 {total:.4f}"), ("裁完全局范数", after, "期望 1.0"),
      ("缩放系数", scale, "两个张量必须共用同一个"), ("小梯度组返回", got2, f"期望 √0.02={math.sqrt(0.02):.4f}"),
      ("全零组返回", got3, "期望 0 且有限")],
     ok,
     f"返回裁剪**前**的范数：{abs(got - total) < 1e-12} · 裁完全局范数 = 1.0：{after:.6f} · "
     f"方向不变（同一个 scale={scale:.6f}）：{dir_ok} · 没超限时一点都不动：{untouched} · "
     f"全零梯度返回 0 且不炸：{no_nan}")
""",
    ),
    dict(
        id="TODO-24", stage="s05", label="lr_at：warmup + cosine 调度",
        needs=[],
        code=r"""
from minigpt.optim import lr_at
peak, warm, total, frac = 6e-3, 100, 1000, 0.1
s = [lr_at(i, total, peak, warmup=warm, final_frac=frac) for i in range(total + 3)]
up = all(s[i] <= s[i + 1] + 1e-18 for i in range(warm - 1))
down = all(s[i] >= s[i + 1] - 1e-18 for i in range(warm, total))
first_ok = 0 < s[0] < peak / warm * 1.000001
top_ok = abs(s[warm - 1] - peak) < 1e-15
tail_ok = abs(s[total] - peak * frac) < 1e-9
cap_ok = max(s) <= peak * 1.0000001
short = lr_at(3, 3, peak, warmup=10)              # total_steps <= warmup：不许除零
short_ok = 0 < short <= peak
ok = bool(up and down and first_ok and top_ok and tail_ok and cap_ok and short_ok)
card("学习率调度两条线：前 warmup 步线性爬到 peak，之后 cosine 收到 frac·peak。"
     "四个边界各对应一种真实写法错误：step 0 不许是 0（那一步白跑）；"
     "warmup 结束那一步必须**正好**等于 peak（差一点说明分母写成 warmup+1）；"
     "末端收到 frac·peak 而不是 0；全程不许超过 peak。"
     "短跑（total_steps ≤ warmup）不许除零 —— 这是最容易忘的那个分母。",
     [("调用", f"lr_at(step, total_steps={total}, peak={peak}, warmup={warm}, final_frac={frac})",
       "step 从 0 到 1002 全取一遍"),
      ("短跑探针", "lr_at(3, total_steps=3, warmup=10)", "warmup 比总步数还长")],
     [("s[0]", s[0], "期望 >0 且 ≈ peak/warmup"), ("s[99]", s[99], "期望正好 peak"),
      ("s[500]", s[500], "cosine 中段"), ("s[1000]", s[1000], f"期望 peak×{frac}={peak * frac:.5g}"),
      ("max(s)", max(s), f"期望 ≤ peak={peak}"), ("短跑", short, "期望 0 < short ≤ peak")],
     ok,
     f"warmup 段单调不降：{up} · 之后单调不升：{down} · step0 不为 0：{first_ok} · "
     f"warmup 结束正好 peak：{top_ok} · 末端收到 {frac}×peak：{tail_ok} · "
     f"全程不超过 peak：{cap_ok} · total_steps<=warmup 时不炸：{short_ok}")
""",
    ),

    # ------------------------------------------------------------------ 阶段 6
    dict(
        id="TODO-25", stage="s06", label="save_checkpoint 的 MGPT 字节格式",
        needs=[],
        code=r"""
import json
import os
import struct
import tempfile

import numpy as np
from minigpt.checkpoint import MAGIC, VERSION, save_checkpoint
from minigpt.config import Config
from minigpt.nn import Module, Parameter


class Tiny(Module):
    def __init__(self):
        self.a = Parameter(np.arange(6, dtype=np.float64).reshape(2, 3), name="a")
        self.b = Parameter(np.full(3, 0.25), name="b", wd_exempt=True)
        self.c = self.a                    # tied：同一个对象，只许存一份


def parse(buf):
    # 参考解析器：完全照着骨架文件头那张字节表读，不用作者自己的 load
    o = 0
    magic, ver, n = buf[o:o + 4], struct.unpack("<I", buf[o + 4:o + 8])[0], struct.unpack("<I", buf[o + 8:o + 12])[0]
    o = 12
    head = json.loads(buf[o:o + n].decode("utf-8"))
    o += n
    tensors = {}
    while o < len(buf):
        nl = struct.unpack("<I", buf[o:o + 4])[0]
        o += 4
        name = buf[o:o + nl].decode("utf-8")
        o += nl
        nd = struct.unpack("<I", buf[o:o + 4])[0]
        o += 4
        shape = [int(v) for v in np.frombuffer(buf[o:o + 8 * nd], dtype="<i8")]
        o += 8 * nd
        cnt = int(np.prod(shape))
        tensors[name] = np.frombuffer(buf[o:o + 4 * cnt], dtype="<f4").reshape(shape)
        o += 4 * cnt
    return magic, ver, head, tensors, o


cfg = Config(vocab=5, d_model=3, n_layer=1, n_head=1, ctx=4)
m = Tiny()
path = os.path.join(tempfile.mkdtemp(), "tiny.ckpt")
size = save_checkpoint(path, m, cfg)
raw = open(path, "rb").read()
magic, ver, head, tensors, consumed = parse(raw)
names_ok = set(tensors) == {"a", "b"}                      # c 是 a 的别名，不许存两遍
val_ok = (np.allclose(tensors["a"], m.a.data, atol=1e-7)
          and np.allclose(tensors["b"], m.b.data, atol=1e-7))
f32_ok = all(v.dtype == np.float32 for v in tensors.values())
cfg_ok = head.get("config") == cfg.to_dict()
size_ok = int(size) == len(raw) and consumed == len(raw)    # 读得完、不多写尾巴
payload = sum(4 + len(nm.encode()) + 4 + 8 * a.ndim + 4 * a.size for nm, a in tensors.items())
acct_ok = consumed == 12 + struct.unpack("<I", raw[8:12])[0] + payload   # 每个数正好 4 字节
ok = bool(magic == MAGIC and ver == VERSION and names_ok and val_ok and f32_ok
          and cfg_ok and size_ok and acct_ok)
card("字节格式照着骨架文件头那张表写：magic(4) | version u32 小端(4) | head_len u32(4) | JSON | "
     "然后每个张量 name_len u32 | name | ndim u32 | shape i64×nd | data f32×n。"
     "这里用一份**独立写的参考解析器**读回来（不用你自己的 load），所以格式写歪一格就会当场露出来。"
     "判五件事：魔数与版本号；头 JSON 里的 config 和 Config.to_dict() 完全相同；"
     "tied 的别名 c 只许存一份（存两遍读回来会互相覆盖）；fp64 灌进去必须降成 fp32；"
     "字节账对得上（返回值==文件大小、参考解析器正好读到 EOF、没有多余尾巴）。",
     [("模型", "Tiny：a=(2,3) 的 arange、b=(3,) 全 0.25（wd 豁免）、c 是 a 的别名", "tied 只许存一份"),
      ("cfg", f"Config(vocab={cfg.vocab}, d_model={cfg.d_model}, ctx={cfg.ctx})"),
      ("path", path, "临时目录里的 tiny.ckpt")],
     [("文件大小", f"{len(raw)} 字节"), ("解析出的张量", sorted(tensors)),
      ("a 的形状/dtype", f"{tuple(tensors['a'].shape)} / {tensors['a'].dtype}", "必须是 <f4"),
      ("返回值 size", size, "应等于文件大小"), ("头 JSON 里的 config", head.get("config"))],
     ok,
     f"魔数 {magic!r}==MGPT：{magic == MAGIC} · 版本 {ver}=={VERSION}：{ver == VERSION} · "
     f"头 JSON 的 config 与 Config.to_dict 相同：{cfg_ok} · 张量名只有 a/b（tied 不存两遍）：{names_ok} · "
     f"数值一致且 fp32：{val_ok and f32_ok} · 返回值==文件大小且正好读到 EOF：{size_ok} · "
     f"字节账对得上（4 字节一个数，没有 fp64 尾巴）：{acct_ok}")
""",
    ),
    dict(
        id="TODO-26", stage="s06", label="load_checkpoint 的校验与原地装配",
        needs=["TODO-25"],
        code=r"""
import os
import tempfile

import numpy as np
from minigpt.checkpoint import load_checkpoint, save_checkpoint
from minigpt.config import Config
from minigpt.nn import Module, Parameter


class Tiny(Module):
    def __init__(self, v=1.0, cfg=None):
        self.a = Parameter(np.full((2, 3), v), name="a")
        self.b = Parameter(np.full(3, v * 2), name="b")
        self.cfg = cfg if cfg is not None else Config(vocab=5, d_model=3, n_layer=1, n_head=1, ctx=4)


cfg = Config(vocab=5, d_model=3, n_layer=1, n_head=1, ctx=4)
src = Tiny(0.7)
path = os.path.join(tempfile.mkdtemp(), "t.ckpt")
save_checkpoint(path, src, cfg)

info = load_checkpoint(path)                                   # 不建模型的读法
reads_ok = (isinstance(info["config"], Config)
            and info["config"].to_dict() == cfg.to_dict()
            and set(info["tensors"]) == {"a", "b"}
            and info["tensors"]["a"].dtype == np.float32)

dst = Tiny(99.0)
data_id = id(dst.a.data)
r = load_checkpoint(path, model=dst)
applied_ok = (np.allclose(dst.a.data, src.a.data, atol=1e-7)
              and np.allclose(dst.b.data, src.b.data, atol=1e-7)
              and int(r.get("applied", 0)) == 2)
inplace_ok = id(dst.a.data) == data_id          # 必须原地写：tied/优化器拿着旧引用

bad = Config(vocab=7, d_model=3, n_layer=1, n_head=1, ctx=4)    # 词表不同，但张量形状碰巧兼容
path2 = os.path.join(tempfile.mkdtemp(), "bad.ckpt")
save_checkpoint(path2, Tiny(0.3), bad)
try:
    load_checkpoint(path2, model=Tiny(0.0))
    mismatch = False
except ValueError:
    mismatch = True

with open(path, "rb") as f:
    junk = f.read(4)
open(path + ".bad", "wb").write(b"XXXX" + open(path, "rb").read()[4:])
try:
    load_checkpoint(path + ".bad")
    magic_guard = False
except ValueError:
    magic_guard = True

ok = bool(reads_ok and applied_ok and inplace_ok and mismatch and magic_guard)
card("load_checkpoint 的两种用法：只读（不建模型，光看文件里有什么 —— 不传 model 就不许碰参数）"
     "和灌回（传 model= 时数值必须和存的那台一致，applied 报个数）。"
     "灌回这一半真正要判的是**原地写**：必须往 Parameter 已有的 data 数组里写，"
     "不能 self.a = Parameter(...) 换个新对象 —— tied 的别名和优化器手里攥的是旧引用，"
     "换新对象它们看不见，续训时优化器还在动一份旧数组。"
     "另外两种拒绝：config 不一致（哪怕张量形状碰巧兼容也要拦）、魔数不对。",
     [("path", path, "a=0.7 那台模型存的，6+3 个数"), ("只读", "load_checkpoint(path)", "不传 model"),
      ("dst", "Tiny(99.0)", "灌回前先全是 99，用来验数值真被覆盖"),
      ("bad.ckpt", "Config(vocab=7) 存的", "形状碰巧兼容但 config 不一致"),
      ("坏文件", "把前 4 个字节改成 XXXX", "魔数不对")],
     [("info['config']", type(info['config']).__name__, "应还原成 Config 对象而不是 dict"),
      ("info['tensors']", sorted(info['tensors'])), ("applied", r.get('applied'), "期望 2"),
      ("dst.a", dst.a.data, "期望全 0.7"), ("data 对象是否同一个", "同一个（原地写）" if inplace_ok else "被换掉了")],
     ok,
     f"只读不建模型也拿得到 config/tensors：{reads_ok} · 灌回模型数值一致：{applied_ok} · "
     f"原地写（Parameter 的 data 对象没被换掉）：{inplace_ok} · "
     f"配置不一致时抛 ValueError：{mismatch} · 魔数不对时抛 ValueError：{magic_guard}")
""",
    ),
    dict(
        id="TODO-27", stage="s06", label="训练状态存档与断点续训",
        needs=[],
        code=r"""
import io
import os
import tempfile

import numpy as np
from minigpt.checkpoint import (STATE_MAGIC, _read_header, _read_tensors, load_train_state,
                                save_train_state)
from minigpt.config import Config
from minigpt.nn import Module, Parameter
from minigpt.optim import AdamW


class Tiny(Module):
    def __init__(self):
        self.a = Parameter(np.ones((2, 2)), name="a")
        self.b = Parameter(np.ones(3), name="b", wd_exempt=True)


cfg = Config(vocab=5, d_model=2, n_layer=1, n_head=1, ctx=4)
m = Tiny()
opt = AdamW(m, lr=3e-4, weight_decay=0.1)
for i, (_, p) in enumerate(m.named_parameters()):        # 不依赖 TODO-22：手工填矩
    opt.m[i][...] = 0.1 * (i + 1) + np.arange(p.data.size).reshape(p.data.shape) * 1e-3
    opt.v[i][...] = 0.2 * (i + 2)
path = os.path.join(tempfile.mkdtemp(), "t.state")
size = save_train_state(path, m, opt, 137, [3.1, 2.9, 2.7], cfg)
raw = open(path, "rb").read()
with open(path, "rb") as f:
    hdr = _read_header(f, STATE_MAGIC)
    tt = _read_tensors(f, "<f8")            # 按 float64 读；写成 fp32 就会读到一半就断
    tail = f.read()
magic_ok = raw[:4] == STATE_MAGIC and hdr.get("step") == 137
f64_ok = (bool(tt) and all(v.dtype == np.float64 for v in tt.values()) and tail == b""
          and set(tt) == {"m.a", "v.a", "m.b", "v.b"})

m2 = Tiny()
opt2 = AdamW(m2, lr=1e-3, weight_decay=0.0)
head_id = id(opt2.m[0])
st = load_train_state(path, model=m2, opt=opt2)
mom_ok = all(np.allclose(opt2.m[i], opt.m[i], atol=1e-15) and
             np.allclose(opt2.v[i], opt.v[i], atol=1e-15)
             for i in range(2))
step_ok = opt2.step_count == 137 and st.get("step") == 137
meta_ok = st.get("lr") == 3e-4 and st.get("curve") == [3.1, 2.9, 2.7] and st["config"].to_dict() == cfg.to_dict()
inplace_ok = id(opt2.m[0]) == head_id                     # 原地：优化器存的是数组引用
ok = bool(magic_ok and f64_ok and mom_ok and step_ok and meta_ok and inplace_ok and int(size) == len(raw))
card("断点续训真正要存的不是权重，是优化器的状态：魔数是 MGTR（跟 MGPT 分开，两种文件不许互读），"
     "矩留 float64（8 字节；压成 fp32 续训不会报错，但动量会悄悄变，loss 曲线接不上），"
     "键名带参数前缀（m.a / v.a / m.b / v.b），头里存 step、lr、curve、config。"
     "读回时 opt.step_count 必须恢复成 137 —— 不然偏差修正又从 t=1 开始，续训头几步的更新量会明显偏大；"
     "数组同样要原地写，优化器手里拿的是旧引用。",
     [("模型/优化器", "Tiny 两个参数 + AdamW(lr=3e-4, wd=0.1)", "m/v 手工填成有规律的数，不依赖 TODO-22"),
      ("存的东西", "step=137、curve=[3.1, 2.9, 2.7]、config", ""),
      ("path", path, "临时目录里的 t.state")],
     [("文件大小", f"{len(raw)} 字节"), ("张量", f"{sorted(tt)}，dtype {list(tt.values())[0].dtype}", "必须 <f8"),
      ("读回 step", st.get('step'), "期望 137，且 opt2.step_count 同步"),
      ("读回 lr / curve", f"{st.get('lr')} / {st.get('curve')}"), ("m/v 是否对上", "按参数名对上且数值一致" if mom_ok else "没对上")],
     ok,
     f"魔数是 MGTR（不是 MGPT）且头里有 step：{magic_ok} · 矩留 float64 且按 8 字节读完：{f64_ok} · "
     f"m/v 按参数名对上且数值一致：{mom_ok} · opt.step_count 恢复成 137：{step_ok} · "
     f"lr/curve/config 都在头里：{meta_ok} · 原地写（数组对象没被换）：{inplace_ok}")
""",
    ),

    # ------------------------------------------------------------------ 阶段 7
    dict(
        id="TODO-28", stage="s07", label="重复惩罚：正分除、负分乘，且不改入参",
        needs=[],
        code=r"""
import numpy as np
from minigpt.sampler import apply_penalty
z = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
orig = z.copy()
out = apply_penalty(z, [0, 1, 3], 2.0)
want = np.array([1.0, -2.0, 0.5, 0.0, 3.0])     # 0 号 2/2；1 号 -1*2；3 号 0*2
vals_ok = bool(np.allclose(out, want))
no_mutate = bool(np.array_equal(z, orig)) and out is not z
noop = bool(np.array_equal(apply_penalty(z, [0], 1.0), z))
empty = bool(np.array_equal(apply_penalty(z, [], 2.0), z))
oob = bool(np.array_equal(apply_penalty(z, [99, -3], 2.0), z))
card("重复惩罚的符号：已经出现过的 token，正分**除以** penalty、负分**乘以** penalty —— "
     "两种都是把它压向 0。统一写成除法会把负分抬离 0，本来该压的反而更突出，惩罚变成奖励。"
     "另外不许改调用者的数组（返回新数组，sample_next 后面还要拿原始 logits 算分布）；"
     "penalty=1、seen 为空、下标越界都要原样返回而不是报错。",
     [("z", z, "有正有负，0/1/3 号当作「已出现过」"), ("seen", "[0, 1, 3]"), ("penalty", 2.0),
      ("对照", "penalty=1.0 / seen=[] / seen=[99,-3]", "三种都不该有变化")],
     [("out", out, f"期望 [1.0, −2.0, 0.5, 0.0, 3.0]：2/2、−1×2、0×2"),
      ("z 是否被改", "没改（返回的是新数组）" if no_mutate else "被就地改了"),
      ("三种空操作", "原样返回" if (noop and empty and oob) else "有变化")],
     vals_ok and no_mutate and noop and empty and oob,
     f"正分÷2、负分×2、未出现的不动：{vals_ok} · 不修改调用者的数组：{no_mutate} · "
     f"penalty=1 或 seen 为空/全越界时原样返回：{noop and empty and oob}")
""",
    ),
    dict(
        id="TODO-29", stage="s07", label="top-k：并列的保留，其余掩成 MASK",
        needs=[],
        code=r"""
import numpy as np
from minigpt.sampler import MASK, keep_top_k
z = np.array([0.1, 5.0, 2.0, 5.0, -3.0, 4.0])
k3 = keep_top_k(z, 3)
kept3 = set(np.flatnonzero(k3 > MASK / 2).tolist())
want3 = {1, 3, 5}
masked_ok = bool(np.all(k3[~np.isin(np.arange(6), list(want3))] == MASK))
tie = keep_top_k(np.array([1.0, 1.0, 0.0]), 1)
tie_ok = bool(tie[0] == 1.0 and tie[1] == 1.0 and tie[2] == MASK)   # 并列最高都留下
noop0 = bool(np.array_equal(keep_top_k(z, 0), z))
noopV = bool(np.array_equal(keep_top_k(z, 6), z)) and bool(np.array_equal(keep_top_k(z, 99), z))
shape_ok = keep_top_k(z, 2).shape == z.shape
card("top-k 保留最高的 k 个，其余写成 MASK（−1e9）而不是删掉 —— "
     "形状必须和输入一样，后面 softmax 那一步才接得住；真删了元素就少了，位置全错位。"
     "并列的那组是重点：[1.0, 1.0, 0.0] 取 k=1 时两个并列最高都要留下（阈值比较用 < 而不是 <=），"
     "写成 <= 会随机丢掉一个，症状是「同样的输入换个顺序结果就变」。k=0 或 k≥V 时不该剪。",
     [("z", z, "6 个数，5.0 出现两次（故意并列）"), ("k", "3 / 1 / 0 / 6 / 99 各试一遍"),
      ("并列探针", "[1.0, 1.0, 0.0] 取 k=1", "两个都该留")],
     [("keep_top_k(z, 3)", k3, f"保留下标 {sorted(kept3)}，其余 = {MASK:g}"),
      ("keep_top_k([1,1,0], 1)", tie, "期望前两个原样、第三个 MASK"),
      ("k=0 / k≥V", "原样返回" if (noop0 and noopV) else "被剪了")],
     kept3 == want3 and masked_ok and tie_ok and noop0 and noopV and shape_ok,
     f"保留的正好是最高 3 个（{sorted(kept3)}=={sorted(want3)}）：{kept3 == want3} · "
     f"被筛的写成 MASK 而不是删掉：{masked_ok} · 并列最高两个都保留（用 < 不是 <=）：{tie_ok} · "
     f"k=0 或 k>=V 不剪：{noop0 and noopV}")
""",
    ),
    dict(
        id="TODO-30", stage="s07", label="top-p：nucleus 覆盖概率质量后重新归一化",
        needs=[],
        code=r"""
import numpy as np
from minigpt.sampler import keep_top_p
p = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
q8 = keep_top_p(p, 0.8)
mass_ok = bool(np.allclose(q8, [0.625, 0.375, 0.0, 0.0, 0.0], atol=1e-12))
sum_ok = abs(float(q8.sum()) - 1.0) < 1e-12
q1 = keep_top_p(p, 1.0)
full_ok = bool(np.allclose(q1, p, atol=1e-12))
q04 = keep_top_p(p, 0.4)                              # 第一名 0.5 > 0.4：只留它
tiny_ok = bool(np.allclose(q04, [1.0, 0.0, 0.0, 0.0, 0.0])) and np.isfinite(q04).all()
hot = keep_top_p(np.array([0.97, 0.02, 0.01]), 0.9)   # 极端 one-hot，不许冒出 nan
hot_ok = bool(np.allclose(hot, [1.0, 0.0, 0.0]))
prefix = sorted(np.flatnonzero(q8 > 0).tolist()) == [0, 1]      # 保留的是排序后的前缀
card("nucleus 采样：把概率从大到小累加，覆盖到 p 的那一批留下、其余置 0，"
     "然后**重新归一化** —— 少这一步，和不为 1，后面 softmax 出来的分布就是歪的。"
     "输入已经是概率分布（不是 logits），所以这里判的是裁剪与归一化，不是 softmax 本身。"
     "两个边界最容易写错：第一名自己就超过 p 时只留它一个（别把循环条件写成严格小于，"
     "否则一个都不留、归一化 0/0）；极端 one-hot 不许冒出 nan。",
     [("p（概率分布）", p, "0.5+0.3+0.15+0.04+0.01，已归一化"),
      ("top_p", "0.8 / 1.0 / 0.4 各一遍", "0.4 时第一名 0.5 就已超过"),
      ("one-hot 探针", "[0.97, 0.02, 0.01] 取 p=0.9", "只留第一个，不许 nan")],
     [("keep_top_p(p, 0.8)", q8, "期望 [0.625, 0.375, 0, 0, 0]"),
      ("keep_top_p(p, 0.4)", q04, "期望 [1, 0, 0, 0, 0]"),
      ("keep_top_p(p, 1.0)", q1, "期望原样"),
      ("和", float(q8.sum()), "期望 1.0")],
     mass_ok and sum_ok and full_ok and tiny_ok and hot_ok and prefix,
     f"0.5+0.3 覆盖到 0.8 → 归一化成 0.625/0.375：{mass_ok} · 和为 1：{sum_ok} · "
     f"p=1 不剪：{full_ok} · 第一名超过 p 时只留一个：{tiny_ok} · "
     f"one-hot 也不 0/0：{hot_ok} · 保留的是概率降序的前缀：{prefix}")
""",
    ),
    dict(
        id="TODO-31", stage="s07", label="sample_next 的组装：贪心/可复现/分布一致",
        needs=["TODO-28", "TODO-29", "TODO-30"],
        code=r"""
import numpy as np
from minigpt.sampler import sample_next
rng0 = np.random.default_rng(0)
z = rng0.standard_normal(12) * 2.0
orig = z.copy()
best = int(np.argmax(z))
g1 = sample_next(z, np.random.default_rng(1), temperature=0)
g2 = sample_next(z, np.random.default_rng(99), temperature=0)
greedy_ok = (g1 == best == g2)
r1 = sample_next(z, np.random.default_rng(7), temperature=0.8, top_k=5, top_p=0.9)
r2 = sample_next(z, np.random.default_rng(7), temperature=0.8, top_k=5, top_p=0.9)
repro_ok = (r1 == r2)
k1_ok = all(sample_next(z, np.random.default_rng(s), temperature=1.0, top_k=1) == best
            for s in range(6))
pen = sample_next(z, np.random.default_rng(3), temperature=0, seen=[best], penalty=8.0)
pen_ok = pen != best                                  # 惩罚在贪心之前 → 换第二个
counts = {}
sub = np.random.default_rng(11)
n = 4000
for _ in range(n):
    t = sample_next(z, sub, temperature=1.0)
    counts[t] = counts.get(t, 0) + 1
e = np.exp(z - z.max())
want = e / e.sum()
emp = np.array([counts.get(i, 0) for i in range(len(z))], dtype=float) / n
l1 = float(np.abs(emp - want).sum())
in_range = all(0 <= k < len(z) for k in counts)
no_mut = np.array_equal(z, orig)
ok = bool(greedy_ok and repro_ok and k1_ok and pen_ok and l1 <= 0.14 and in_range and no_mut)
card("sample_next 是把前面几个零件装在一起，所以它真正判的是**顺序**："
     "temperature=0 直接返回 argmax（且与 seed 无关）；惩罚要在 top-k/top-p **之前**生效，"
     "写在之后就等于奖励了一个已经被剪掉的 token；同 seed 必须同结果，不然生成没法复现；"
     "top_k=1 恒等于 argmax。最硬的一条是分布：4000 次采样的经验分布要贴住 softmax 的理论分布"
     "（L1 ≤ 0.14）—— 归一化多做一次、温度除反、MASK 没减掉，数值上都是小偏差，"
     "但经验分布会明显跑偏，这条一抓一个准。",
     [("z", z, f"V={len(z)}，幅度 ×2 的随机 logits"), ("探针", "temperature=0 / 0.8+top_k=5+top_p=0.9 / top_k=1 / seen=[argmax],penalty=8", ""),
      ("采样", "temperature=1.0 连采 4000 次", "seed 固定 11")],
     [("贪心返回", g1, f"应等于 argmax={best}"), ("同 seed 两次", f"{r1} / {r2}", "必须相同"),
      ("惩罚后的贪心", pen, "应换成第二个（惩罚在剪枝之前）"),
      ("经验分布", emp, "4000 次的频率"), ("理论分布", want, "softmax(z)")],
     ok,
     f"temperature=0 == argmax 且与 seed 无关：{greedy_ok} · 同 seed 同结果：{repro_ok} · "
     f"top_k=1 恒为 argmax：{k1_ok} · 惩罚在贪心之前生效：{pen_ok} · "
     f"经验分布与 softmax 的 L1={l1:.3f}（阈值 0.14）：{l1 <= 0.14} · "
     f"返回合法 int：{in_range} · 不改入参数组：{no_mut}")
""",
    ),

    # ------------------------------------------------------------------ 已给出代码的配套检查
    dict(
        id="ACT-3", stage="s02", label="初始化尺度让方差不变（讲义 §3.1）",
        needs=[],
        code=r"""
import numpy as np
from minigpt.exercises import ScalarMLP

# activate=False 时网络是仿射的：out = c·x + const，其中 c = Σ_j W2[j]·W1[j][0]
# E[c²] = Σ_j E[W2_j²]E[W1_j] = n·(1/n)·1 = 1 —— 与 n_hidden 无关，这正是 1/√n 的作用。
res = {}
for n_hidden in (1, 8, 64):
    cs = []
    for seed in range(200):
        m = ScalarMLP(n_hidden=n_hidden, activate=False, seed=seed)
        cs.append(sum(m.W2[j].data * m.W1[j][0].data for j in range(n_hidden)))
    res[n_hidden] = float(np.mean(np.square(cs)))
bias_zero = all(abs(m.b2.data) < 1e-15 for m in [ScalarMLP(n_hidden=4, seed=i) for i in range(4)])
flat = max(abs(v - 1.0) for v in res.values()) < 0.6
ok = flat and bias_zero
card("初始化尺度为什么是 1/√n：activate=False 时网络是仿射的，"
     "对输入的等效斜率 c = Σ_j W2_j·W1_j，于是 E[c²] = n·(1/n)·1 = 1 —— 与 n_hidden 无关。"
     "不除那个 √n，c 的方差会随宽度线性涨，第一层就把输入放大几十倍，"
     "症状是 loss 一开始就爆炸或者 sigmoid/tanh 全饱和。"
     "这条跑 200 个种子 × 三种宽度（1/8/64）实测 E[c²]，都要贴着 1；bias 必须从 0 起步。",
     [("宽度", "n_hidden = 1 / 8 / 64", "每种 200 个种子，共 600 台小网络"),
      ("输入", "x = 1 的单层仿射网络", "这样等效斜率就是输出对 x 的系数")],
     [("E[c²] 按宽度", ", ".join(f"n={k}:{v:.2f}" for k, v in res.items()), "都要≈1 且不随宽度增长"),
      ("bias 起点", "全 0" if bias_zero else "非 0")],
     ok,
     "E[方差] 按宽度 " + ", ".join(f"n={k}:{v:.2f}" for k, v in res.items())
     + f"（都要≈1 且不随宽度增长）；bias 从 0 起步：{bias_zero}")
""",
    ),
    dict(
        id="GIV-1", stage="s03a", label="Parameter/Module 的约定（讲义 §4.2-§4.3）",
        needs=[],
        code=r"""
import numpy as np
from minigpt.nn import MLP, Module, Parameter, RMSNorm, Linear, init_resid_scale

p = Parameter(np.zeros((3, 4)), name="probe")
dt_ok = p.data.dtype == np.float64 and p.data.flags["C_CONTIGUOUS"]
try:
    p.grads = 1.0                     # 拼错属性名（少一个 d）
    slots_ok = False
except AttributeError:
    slots_ok = True
g_ref = id(p.grad)
p.grad += 1.0
p.zero_grad()
inplace_ok = id(p.grad) == g_ref and not p.grad.any()

lin = Linear(4, 3)
wd_ok = bool(lin.b.wd_exempt and not lin.W.wd_exempt and RMSNorm(3).w.wd_exempt)

holder = MLP(4, 8)
before = {n: v.data.copy() for n, v in holder.named_parameters()}
init_resid_scale(holder, n_layer=3)
scale = 1.0 / np.sqrt(6.0)
moved = {n: float(np.max(np.abs(v.data - before[n]))) for n, v in holder.named_parameters()}
changed = {n for n, mv in moved.items() if mv > 0}
only_fc2 = changed == {n for n in before if n.endswith("fc2.W")}
ok = dt_ok and slots_ok and inplace_ok and wd_ok and only_fc2
card("这几条判的都是**已经给你的**基础设施的约定 —— 你写每一层时都在踩它们，"
     "所以它们坏了比你自己写错更难查：Parameter 的 data 必须是 float64 且 C 连续（否则 gradcheck 的"
     "扰动会写不进原数组）；__slots__ 让拼错属性名当场报错，而不是悄悄多出一个没人读的字段；"
     "zero_grad 原地清、不换对象；bias 和 norm 的权重不吃 weight decay；"
     "init_resid_scale 只缩放残差出口 attn.proj.W / mlp.fc2.W，别的参数一点不许动。",
     [("p", Parameter(np.zeros((3, 4))), "故意用 zeros，方便看 dtype 和连续性"),
      ("lin = Linear(4,3) / RMSNorm(3)", "看 wd_exempt 标记落在谁身上"),
      ("holder = MLP(4,8)", f"缩放前 fc2.W={brief(before['fc2.W'], 3)}"),
      ("拼错属性", "p.grads = 1.0（少一个 d）", "应抛 AttributeError")],
     [("缩放后 fc2.W", holder.fc2.W.data, f"每个元素应乘 1/√6={1.0 / np.sqrt(6.0):.4f}"),
      ("被改动的参数", sorted(changed), "期望只有 fc2.W"),
      ("grad 对象是否同一个", "同一个（原地清）" if inplace_ok else "被换掉了")],
     ok,
     f"float64+连续：{dt_ok}；拼错属性会报错：{slots_ok}；zero_grad 原地：{inplace_ok}；"
     f"bias/norm 免 decay：{wd_ok}；只缩放残差出口（{sorted(changed)}）：{only_fc2}")
""",
    ),
    dict(
        id="GIV-2", stage="s04", label="拆头可逆、掩码结构与两个 bias 的命运（讲义 §3）",
        needs=[],
        code=r"""
import numpy as np
from minigpt.attention import CausalSelfAttention, NEG_INF
from minigpt.config import Config

cfg = Config(d_model=8, n_head=4, vocab=9, ctx=6)
att = CausalSelfAttention(cfg, rng=np.random.default_rng(9))
B, T, C = 2, 5, 8
y = np.random.default_rng(0).standard_normal((B, T, C))
h = att._split_heads(y, B, T)
back = h.transpose(0, 2, 1, 3).reshape(B, T, C)
merge_ok = bool(np.array_equal(y, back))                      # 严格逆操作
head_shape_ok = h.shape == (B, cfg.n_head, T, cfg.head_dim)

m = CausalSelfAttention._causal_mask(T, np.float64)[0, 0]
mask_ok = bool(np.all(m[np.triu(np.ones((T, T), bool), 1)] == NEG_INF)
               and np.all(m[np.tril(np.ones((T, T), bool))] == 0.0))

bias_choice = att.k.b is None and att.q.b is not None and att.v.b is not None

# b_q 的梯度非零需要 backward()（TODO-13），没写完就只报结构
extra = ""
try:
    x = np.random.default_rng(1).standard_normal((1, 4, C))
    att(x)
    att.backward(np.random.default_rng(2).standard_normal((1, 4, C)))
    extra = f"；跑完反向后 b_q 梯度非零：{bool(np.any(att.q.b.grad))}，b_v 梯度非零：{bool(np.any(att.v.b.grad))}"
except NotImplementedError:
    extra = "；梯度那一半待 TODO-13"
ok = merge_ok and head_shape_ok and mask_ok and bias_choice
card("注意力给你的三件东西，每件都是你写反向时躲不开的形状："
     "① 拆头 (B,T,C) → (B,H,T,d/H) 和它的严格逆操作 —— 反向要把四个头的梯度摊回 (B,T,C)，"
     "顺序错一位不会报错，只会让梯度悄悄串位，所以这里逐比特验互逆；"
     "② 因果掩码的形状：上三角是 NEG_INF、下三角是 0（注意是加在 logits 上的 bias，不是 0/1 掩码）；"
     "③ q/v 有 bias、k 没有 —— 这是 NanoGPT 沿袭下来的选择，k 的 bias 会被随后的 RMSNorm 吃掉，"
     "留着只会多一堆永远学不动的参数。",
     [("y", y, "B=2, T=5, C=8；H=4，每头 d/H=2"), ("cfg", f"d_model={cfg.d_model}, n_head={cfg.n_head}"),
      ("掩码", "_causal_mask(T, float64)", "看第 0 行和第 2 行就够看出结构")],
     [("h = _split_heads(y)", h, "期望 (2,4,5,2)"), ("back = 合头", back, "期望逐比特等于 y"),
      ("掩码第 0 行", m[0], "只有 j=0 是 0，其余 NEG_INF"), ("掩码第 2 行", m[2], "j≤2 是 0"),
      ("bias 配置", f"q.b={att.q.b is not None}、k.b={att.k.b is None}、v.b={att.v.b is not None}")],
     ok,
     f"拆头/合头互逆：{merge_ok}；头形状 {h.shape}；掩码结构：{mask_ok}；"
     f"k 无 bias、q/v 有：{bias_choice}{extra}")
""",
    ),
    dict(
        id="GIV-3", stage="s05", label="给你的那半：resid 缩放、tied 登记、模型装配（讲义 §3）",
        needs=[],
        code=r"""
import numpy as np
from minigpt.config import Config
from minigpt.gpt import Block, GPT, TiedHead
from minigpt.nn import CrossEntropy, MLP, RMSNorm

cfg = Config(vocab=9, d_model=8, n_layer=3, n_head=2, ctx=6)
scaled = GPT(cfg, rng=np.random.default_rng(0))
d = cfg.to_dict()
d["resid_scale"] = False
plain = GPT(Config(**d), rng=np.random.default_rng(0))      # 同一个 rng 序列
pn = dict(scaled.named_parameters())
qn = dict(plain.named_parameters())
exits = [n for n in pn if n.endswith("attn.proj.W") or n.endswith("mlp.fc2.W")]
target = (2.0 * cfg.n_layer) ** 0.5
ratio_ok = bool(exits) and all(
    abs(float(np.max(np.abs(qn[n].data)) / np.max(np.abs(pn[n].data))) - target) < 1e-6 for n in exits)
others_ok = all(np.array_equal(pn[n].data, qn[n].data) for n in pn if n not in exits)

blk = scaled.blocks[0]
struct_ok = (isinstance(blk.norm1, RMSNorm) and isinstance(blk.norm2, RMSNorm)
             and isinstance(blk.mlp, MLP) and blk.mlp.fc2.W.data.shape[0] == cfg.d_model)
tying_ok = scaled.head.weight is scaled.tok.weight and isinstance(scaled.head, TiedHead)
names = [n for n, _ in scaled.named_parameters()]
dedup_ok = len(names) == len(set(names)) and not any(n.startswith("head.") for n in names)
pos_ok = scaled.pos.weight.shape == (cfg.ctx, cfg.d_model)
loss_ok = isinstance(scaled._loss_fn, CrossEntropy)

extra = ""
try:
    ids = np.random.default_rng(1).integers(0, cfg.vocab, size=(2, 5))
    scaled.zero_grad()
    scaled.loss_backward(ids)
    extra = (f"；跑通整机反向后 pos.weight 有梯度：{bool(np.any(scaled.pos.weight.grad))}"
             f"、tok.weight 有梯度：{bool(np.any(scaled.tok.weight.grad))}")
except NotImplementedError as e:
    extra = f"；反向那一半待 {str(e).split(':')[0]}"
ok = bool(ratio_ok and others_ok and struct_ok and tying_ok and dedup_ok and pos_ok and loss_ok)
card("GPT 装配给你的那半，判法很省：拿同一份 Config 只把 resid_scale 关掉、rng 序列完全相同，"
     "造两台模型对拍。这样唯一该有差别的就是这 " + str(len(exits)) + " 个残差出口张量"
     "（attn.proj.W / mlp.fc2.W 要除以 √(2L)，GPT-2 那个著名初始化），"
     "其余张量必须逐比特相同 —— 一旦别的参数也变了，说明缩放写漏到了主干上。"
     "另外几件结构性的事：Block 是 norm→attn→norm→mlp；输出头借的是 tok.weight **同一个对象**"
     "（is 而不是 ==，否则梯度累加会分家）；named_parameters 不许出现 head.*（tied 不重复登记）；"
     "位置嵌入正好 ctx 行。",
     [("cfg", f"vocab={cfg.vocab}, d_model={cfg.d_model}, n_layer={cfg.n_layer}, n_head={cfg.n_head}, ctx={cfg.ctx}"),
      ("两台模型", "scaled（resid_scale=True）与 plain（False），同一个 rng 序列", "只差那一个开关"),
      ("残差出口", sorted(exits), f"该被除以 √(2L)={target:.4f}")],
     [("出口张量的缩放比", f"{target:.4f}", "只有这些张量不同"),
      ("其余张量", f"{len(names) - len(exits)} 个逐比特相同" if others_ok else "有别的参数被误改"),
      ("参数名", f"{len(names)} 个，全唯一，无 head.*"), ("head.weight is tok.weight", tying_ok)],
     ok,
     f"只有 attn.proj.W / mlp.fc2.W 被除以 √(2L)：{ratio_ok} · 别的参数一点没动：{others_ok} · "
     f"Block 结构是 norm→attn→norm→mlp：{struct_ok} · 输出头借的是 tok.weight 对象：{tying_ok} · "
     f"tied 不重复登记（{len(names)} 个名字全唯一，没有 head.*）：{dedup_ok} · "
     f"位置嵌入有 ctx 行：{pos_ok} · 模型自带 CrossEntropy：{loss_ok}{extra}")
""",
    ),
    dict(
        id="GIV-4", stage="s06", label="给你的字节层：写头、写张量、读回与四种拒绝（讲义 §3）",
        needs=[],
        code=r"""
import io
import struct

import numpy as np
from minigpt.checkpoint import (MAGIC, STATE_MAGIC, VERSION, _moment_lists, _read_header,
                                _read_tensors, _u32, _write_header, _write_tensors)
from minigpt.nn import Module, Parameter
from minigpt.optim import AdamW, SGD

arr = np.arange(6, dtype=np.float64).reshape(2, 3)
buf = io.BytesIO()
_write_header(buf, MAGIC, {"config": {"d_model": 3}})
n_written = _write_tensors(buf, [("a", arr), ("b", np.zeros(3))], "<f4")
raw = buf.getvalue()
head = _read_header(io.BytesIO(raw), MAGIC)
off = 12 + struct.unpack("<I", raw[8:12])[0]        # 跳过 magic|version|len|JSON
t = _read_tensors(io.BytesIO(raw[off:]), "<f4")
rt_ok = set(t) == {"a", "b"} and t["a"].dtype == np.float32 and np.allclose(t["a"], arr, atol=1e-7)
head_ok = head == {"config": {"d_model": 3}}
le_ok = raw[4:8] == struct.pack("<I", VERSION)      # 版本号是小端 u32

def guard(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


magic_guard = guard(lambda: _read_header(io.BytesIO(b"XXXX" + raw[4:]), MAGIC))
ver_guard = guard(lambda: _read_header(io.BytesIO(raw[:4] + struct.pack("<I", 99) + raw[8:]), MAGIC))
dup_buf = io.BytesIO()
_write_tensors(dup_buf, [("a", arr), ("a", arr)], "<f4")
dup_guard = guard(lambda: _read_tensors(io.BytesIO(dup_buf.getvalue()), "<f4"))
trunc_guard = guard(lambda: _read_tensors(io.BytesIO(raw[:len(raw) - 5]), "<f4"))


class Tiny(Module):
    def __init__(self):
        self.a = Parameter(np.ones((2, 2)), name="a")


m = Tiny()
attrs_adam = [a for a, _ in _moment_lists(AdamW(m, lr=1e-3))]
attrs_sgd = [a for a, _ in _moment_lists(SGD(m, lr=1e-3))]
mom_ok = attrs_adam == ["m", "v"] and attrs_sgd == ["buf"]
sizes_ok = all(len(lst) == len(m.named_parameters()) for _, lst in _moment_lists(AdamW(m)))
magic_distinct = MAGIC != STATE_MAGIC
ok = bool(rt_ok and head_ok and le_ok and magic_guard and ver_guard and dup_guard
          and trunc_guard and mom_ok and sizes_ok and magic_distinct and n_written == 2)
card("字节层给你的四个小函数（_write_header / _write_tensors / _read_header / _read_tensors），"
     "你写的 save/load 只是它们的调用者，所以这里直接考零件。"
     "两个 fp64 张量灌进去，读回来必须已经降成 <f4 且数值一致；头 JSON 原样回来；"
     "版本号是**小端** u32（大端写出来的文件在别的机器上读会读成一个巨大的数）。"
     "四种拒绝是重点：错魔数、错版本、重名（tied 存了两遍）、文件被截断 —— "
     "每一种都必须抛 ValueError，静默接受一个坏 checkpoint 比报错难查十倍。"
     "_moment_lists 还要认出 AdamW 的 (m, v) 和 SGD 的 (buf)，续训才知道该读几组矩。",
     [("张量", "a=(2,3) 的 arange（fp64）、b=(3,) 全 0", "dtype='<f4' 存成 fp32"),
      ("头", '{"config": {"d_model": 3}}'), ("坏输入", "改魔数 / 改版本 / 重名 / 砍掉末尾 5 字节", "四种都要拒绝"),
      ("优化器", "AdamW 与 SGD 各一台 Tiny", "看 _moment_lists 认出几组矩")],
     [("缓冲区大小", f"{len(raw)} 字节"), ("读回", f"名字 {sorted(t)}，dtype {t['a'].dtype}"),
      ("头 JSON", head), ("写出的张量数", n_written), ("矩属性名", f"AdamW={attrs_adam}、SGD={attrs_sgd}")],
     ok,
     f"写进去读回来数值一致且降到 fp32：{rt_ok} · 头 JSON 完整：{head_ok} · "
     f"版本号是小端 u32：{le_ok} · 拒绝错魔数：{magic_guard} · 拒绝错版本：{ver_guard} · "
     f"拒绝重名（tied 存了两遍）：{dup_guard} · 拒绝截断：{trunc_guard} · "
     f"_moment_lists 认 AdamW(m/v) 和 SGD(buf)：{mom_ok and sizes_ok}")
""",
    ),
    dict(
        id="GIV-5", stage="s07", label="给你的 _softmax / entropy / temperature_sweep（讲义 §3）",
        needs=[],
        code=r"""
import numpy as np
from minigpt.sampler import _softmax, entropy, temperature_sweep

z = np.array([900.0, 880.0, 910.0])
with np.errstate(over="ignore"):
    naive = np.exp(z)
p = _softmax(z)
safe_ok = bool(np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-12 and int(np.argmax(p)) == 2)
inv_ok = bool(np.allclose(_softmax(z - 500.0), p, atol=1e-15))
neg_ok = bool(np.allclose(_softmax(np.array([-50.0, -40.0, 0.0])),
                          np.exp(np.array([-50.0, -40.0, 0.0])) / np.exp(np.array([-50.0, -40.0, 0.0])).sum()))
e_uni = entropy(np.full(8, 0.125))
e_hot = entropy(np.array([1.0, 0.0, 0.0]))
e_two = entropy(np.array([0.5, 0.5]))
ent_ok = (abs(e_uni - np.log(8)) < 1e-12 and abs(e_hot) < 1e-12
          and abs(e_two - np.log(2)) < 1e-12)

extra = ""
try:
    rng = np.random.default_rng(0)
    zz = rng.standard_normal(10) * 2.0
    sw = temperature_sweep(zz, rng, temperatures=(0.2, 1.5), n=150)
    widths = {k: len(v) for k, v in sw.items()}
    sweep_ok = all(w > 0 for w in widths.values()) and widths[1.5] >= widths[0.2]
    extra = f"；sweep 用过的不同 id 数：{widths}"
except NotImplementedError as e:
    sweep_ok = True
    extra = f"；sweep 会调 sample_next，那一半待 {str(e).split(':')[0]}"
ok = bool(safe_ok and inv_ok and neg_ok and ent_ok and sweep_ok)
card("采样器给你的三个函数，你写的 sample_next 全靠它们。"
     "_softmax 必须减最大值（e^900 直接 inf），而且**平移不变**、全负分数也算得对 —— "
     "不是只在你随手测的那组数上碰巧对。entropy 用三个你手算得出的哨兵值来判：均匀分布=ln V、"
     "one-hot=0、二选一=ln 2，底数搞错或多加了 1 立刻露出来。"
     "temperature_sweep 用同一个 rng 扫两档温度，温度越高用词应该越宽（出现过的不同 id 更多）——"
     "这是「温度到底作用在哪一步」最便宜的一次验证。",
     [("z", z, "e^900 会溢出：必须靠减最大值活下来"), ("平移对照", "z − 500", "结果应完全相同"),
      ("全负对照", "[-50, -40, 0]", "不减最大值也能算，减了也要一样对"),
      ("熵的三个哨兵", "均匀 8 / one-hot / 二选一", "ln 8、0、ln 2"),
      ("sweep", "10 个分数，temperature 0.2 与 1.5 各采 150 次", "同一 rng")],
     [("softmax(z)", p, "期望有限、和为 1、argmax=2"), ("entropy(均匀 8)", e_uni, f"期望 ln 8={np.log(8):.6f}"),
      ("entropy(one-hot)", e_hot, "期望 0"), ("entropy(二选一)", e_two, f"期望 ln 2={np.log(2):.6f}"),
      ("sweep", extra.lstrip("；") or "没跑成：temperature_sweep 要调 sample_next")],
     ok,
     f"减最大值后有限且和为 1、argmax 不变：{safe_ok} · 平移不变：{inv_ok} · "
     f"全负分数也算得对：{neg_ok} · 熵：均匀=ln V、one-hot=0、二选一=ln 2：{ent_ok}"
     f" · temperature_sweep 温度越高用词越宽：{sweep_ok}{extra}")
""",
    ),
]

BY_ID = {c["id"]: c for c in CHECKS}


def checks_for(stage_id: str) -> list[dict]:
    """某个阶段页面上该显示的检查。

    `also` 让一条检查出现在多个阶段：TODO-4（tanh）写在 scalar.py 里，而 scalar.py
    是阶段 1 的文件，但 tanh 真正被用到是阶段 2 的事 —— 只在 s02 列出来的话，
    你在阶段 1 看着 TODO-4 却没有「测」按钮，等于逼你跳阶段。
    """
    return [c for c in CHECKS if c["stage"] == stage_id or stage_id in c.get("also", [])]


def home_of(check: dict, stage_id: str) -> bool:
    """这条检查是不是"主场"在该阶段（ False = 只是顺带出现在别的阶段）"""
    return check["stage"] == stage_id


# ------------------------------------------------------------------ 命令行入口
def run_one(check: dict) -> str:
    """在进程内跑一条检查的代码，返回 pass / fail / blocked。

    和网站那条走的是**同一段 code**，所以不会出现"命令行说过了但网站上红着"。
    结论取 card() 记下的那一次（而不是"没抛异常就算过"）：梯度算错是不抛异常的。
    """
    ns: dict = {}
    exec(PRELUDE, ns)
    try:
        exec(compile(check["code"], f"<{check['id']}>", "exec"), ns)
    except NotImplementedError as exc:
        print(f"BLOCKED 被前置依赖挡住：{exc}")
        return "blocked"
    except Exception as exc:                              # noqa: BLE001
        print(f"FAIL {type(exc).__name__}: {exc}")
        return "fail"
    return ns["_VERDICT"][-1] if ns["_VERDICT"] else "pass"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    pkg = "solutions" if os.environ.get("MGPT_ANSWER") == "1" else "src"
    sys.path.insert(0, str(root / pkg))
    if not argv:
        print("用法：python webapp/checks.py <id> [<id> …]  或  python webapp/checks.py all")
        for c in CHECKS:
            print(f"  {c['id']:<10} {c['stage']}  {c['label']}")
        return 0
    ids = [c["id"] for c in CHECKS] if argv == ["all"] else argv
    bad = 0
    for arg in ids:
        check = BY_ID.get(arg) or BY_ID.get(arg.upper())
        if check is None:
            print(f"没有这条检查：{arg}（不带参数运行可以看到全部 id）")
            bad += 1
            continue
        print(f"\n── {check['id']} · {check['label']}")
        if check["needs"]:
            print(f"  需要先写：{'、'.join(check['needs'])}")
        if run_one(check) != "pass":
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
