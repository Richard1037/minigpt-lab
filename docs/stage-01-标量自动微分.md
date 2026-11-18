# 阶段 1 · 标量自动微分：搞懂"梯度下降"这四个字

预计：40 行代码，1~2 小时（含读讲义）。
验收命令：`python stages/s01_fit_line.py`
要写的地方：`src/minigpt/scalar.py` 的 TODO-1/2/3，`src/minigpt/exercises.py` 的 TODO-1A。

---

## 1. 这一步在干什么

有 100 个点，严格满足 `y = 3x + 1`。程序不知道这个式子，只知道一堆 `(x, y)`。
让它自己调 `a` 和 `b`，使 `a·x + b` 尽量贴近 `y`。

脚本会跑两遍**同一个问题**：

* 第 A 部分：偏导数我手推（已经写好了，你只需读懂）；
* 第 B 部分：偏导数由你自己写的 `Value` 引擎自动算出来（TODO-1A）。

判据要求两者的 `a`、`b` 差值小于 `1e-6`。这条判据的真正作用不是"拟合直线"，
而是**证明自动微分没有魔法**：它只是替你求了偏导数，仅此而已。
以后你面对 3000 行的 transformer，敢不敢改它，取决于你现在相不相信这句话。

---

## 2. 必要理论：loss、梯度、链式法则

### 2.1 loss 就是"错得多离谱"的一个数字

$$L(a,b)=\frac{1}{N}\sum_{i=1}^{N}\bigl(a x_i + b - y_i\bigr)^{2}$$

一个标量。训练 = 让这一个数字变小。所有复杂度都来自"怎么知道该往哪边调"。

### 2.2 梯度 = 变化率

$\partial L/\partial a$ 读作"a 动一点点，L 动多少倍"。所以把 a 往梯度的**反方向**挪，loss 就降：

$$a \leftarrow a - \eta\,\frac{\partial L}{\partial a}$$

$\eta$ 是学习率（代码里叫 `lr`），决定一次挪多远。就这么一行，它就是"训练"的全部内容。

### 2.3 链式法则：把复合函数拆成局部导数相乘

对 $L=\frac1N\sum_i e_i^{2}$，其中 $e_i = a x_i + b - y_i$，链式法则给出

$$\frac{\partial L}{\partial a}=\frac1N\sum_i 2e_i\,x_i,\qquad
  \frac{\partial L}{\partial b}=\frac1N\sum_i 2e_i$$

外层的导数乘内层的导数 —— 这就是"链"。反向传播干的就是这件事，只不过它把这条链
**沿计算图自动走完**，并且是"从 loss 往参数方向"走（所以叫 back-ward）。

### 2.4 一个你待会儿会手算的例子

设 $a=2$，$b=3$，$L=(ab)^{2}$。图上有四个节点：

```
a ─┐
   ├─ mul ── p(=6) ── pow2 ── loss(=36)
b ─┘
```

$$\frac{\partial L}{\partial p}=2p=12,\qquad
  \frac{\partial p}{\partial a}=b=3,\qquad \frac{\partial p}{\partial b}=a=2$$

再乘一次链：**∂L/∂a** = 12 · 3 = **36**，**∂L/∂b** = 12 · 2 = **24**。

现在换成"自动微分怎么走"：`loss.grad` 起手是 1，`pow2` 节点把 1 · 2*p* = 12 交给 `p`
（注意：是 `p.grad += 12`，而不是"p 的梯度等于 12"这个事实本身 —— 它是**累加**到 p 上的），
`mul` 节点再把 `p.grad` 分发：`a.grad += b * p.grad` = 3 · 12 = 36，`b.grad += a * p.grad` = 24。

**关键结论**：每个节点的规则只需要知道"我上游传下来多少"（`out.grad`）和"我的局部导数"，
不需要知道整张图长什么样。这就是为什么反向传播可以一层一层写。

---

## 3. 已给你的代码：`Value` 与那条加法

```python
class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")
```

| 字段 | 作用 |
|---|---|
| `data` | 这个数现在的值 |
| `grad` | $\partial L/\partial\text{self}$，`backward()` 之后才有意义 |
| `_prev` | 由哪些 Value 算出来的（父节点集合） |
| `_op` | 用什么算的，只为打印调试 |
| `_backward` | **这个节点自己的**局部反向规则（一个闭包） |

`__add__` 是完整样例，它的三行结构就是所有算子的模板：

```python
out = Value(self.data + other.data, (self, other), "+")   # ① 正向算出结果

def _backward():
    self.grad += out.grad                                  # ② 把 out.grad 摊回操作数
    other.grad += out.grad

out._backward = _backward                                  # ③ 规则挂在结果身上
```

三个容易忽略的点，你写乘法时也会踩：

1. 规则里读的是 **`out.grad`**，不是 `self.grad`。写成 `self.grad += other.data * self.grad`
   的话所有梯度恒为 0，loss 纹丝不动，而且**不会报错**。这是本阶段第一大坑。
2. 是 `+=` 而不是 `=`。一个 Value 可以同时被多处用到（比如同一个 `a` 出现 100 次），
   梯度必须把每条路径的贡献**加起来**。用 `=` 就等于只算最后一条路径。
3. 规则是闭包，它记住了 `self`/`other`/`out` 三个对象。所以图是"节点各自带一条局部规则"，
   而不是"一个全局的反向函数"。

`__neg__ / __sub__ / __rsub__ / __truediv__` 也是给你的，注意它们是用 `*` 和 `**` **拼出来的**
（$a/b = a\,b^{-1}$）。所以你写完 `__mul__` 和 `__pow__`，减法除法自动就通了 —— 这叫
组合性，也是真实框架的做法。

---

### 3.5 文件里那四个激活函数：阶段 1 一个都用不到

`scalar.py` 里还摆着 `exp / relu / sigmoid / tanh`，拟合直线不需要它们 ——
非线性只有在你要拟合**弯的东西**时才有意义，那是阶段 2。现在只要认出一件事：
它们的写法和 §3 那条 `__add__` 是同一个模板（算出 out、写一条读 `out.grad` 的规则、挂上去）。
定义、导数、饱和、为什么 FFN 最后选 GELU，全部在
[docs/stage-02 §2.3 起](file:///D:/Transformer/docs/stage-02-%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%B8%8E%E5%AE%B9%E9%87%8F.md)。
`TODO-4`（tanh）也归属阶段 2，所以阶段 1 的页面上不会有它的「测」按钮 —— 不是漏了。

---

## 4. TODO-1：`__mul__`

**契约**：`self * other`，返回新的 `Value`，`_op="*"`，`_prev` 包含两个操作数。

要写的两件事：

* 正向：`out.data = ?`
* 规则：$\partial(ab)/\partial a=b$、$\partial(ab)/\partial b=a$，各自再乘 `out.grad`

**别忘的两件小事**：
`other` 可能是普通 float（`a * 2`），要先 `Value(other)` 包一层；
`Value(other)` 没有父节点，所以它是叶子 —— 但训练时你不会更新它，这没关系。

**自检（30 秒，比跑整个脚本快）**——在项目根目录跑这一行，PowerShell / cmd / Git Bash 都能直接粘贴：

```
python webapp/checks.py TODO-1
```

它跑的就是网站上那个「测」按钮里的同一段代码，会打印输入和输出：

```
── TODO-1 · __mul__ 的乘法法则
  输入 a=2.0, b=3.0 → p=a*b=6；上游 p.grad 手工喂 1.0
PASS 输出 a.grad=3.0, b.grad=2.0  期望 (3.0, 2.0)
```

想手验 `(a*b)**2` 那条（§2.4 算过 36 / 24）就跑：

```
python webapp/checks.py TODO-2 TODO-3
```

（`TODO-3` 需要你先写完 `backward()`；没写完它会说"被前置依赖挡住"，而不是甩你一脸 traceback。）

<details>
<summary>参考答案 · TODO-1 <code>__mul__</code>（判据没过之前别点开；看完关掉自己重写）</summary>

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
```

和 `__add__` 只差两处：正向是乘积；规则里各自乘的是**对方**的 `.data`。
`__rmul__ = __mul__` 那行骨架已经给了，所以 `2 * a` 也自动能跑。

</details>

---

## 5. TODO-2：`__pow__`

**契约**：`self ** k`，`k` 是 int/float（不是 Value，所以只有一个操作数），`_op = f"**{k}"`。

* 正向：`self.data ** k`
* 规则：$\partial x^{k}/\partial x = k\,x^{k-1}$，乘 `out.grad`

**为什么它是除法的底座**：`__truediv__` 里写的是 `self * denom ** -1`，
所以 $k=-1$ 时公式要给出 $-x^{-2}$ —— 顺便确认 `x ** -1` 在 Python 里是浮点幂，
`x=0` 时会抛 ZeroDivisionError，本项目不会遇到，但知道边界在哪儿是好事。

<details>
<summary>参考答案 · TODO-2 <code>__pow__</code></summary>

```python
    def __pow__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError("指数只支持 int/float")
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out
```

三个容易漏的点：`_prev` 只有 `(self,)` 一个元素（指数不是 Value）；
`_op` 要用 **f 串** `f"**{k}"`，写成 `"**{k}"` 打印出来就是花括号原文（不影响正确性，
但 §3 那张图就白画了）；`k=-1` 时公式自动给出 $-x^{-2}$，所以除法白送。

</details>

---

## 6. TODO-3：`backward()` —— 本阶段最需要想清楚的地方

**契约**：`some_value.backward()` 执行后，从它往下的**每个**叶子的 `.grad`
都等于 $\partial(\text{它})/\partial(\text{叶子})$。无返回值。

三件事，顺序不能换：

1. **拓扑排序**：从 `self` 出发递归走 `_prev`，把节点按"孩子在前、父亲在后"排进列表 `topo`。
   * 为什么要**去重**：一个 Value 可以被多个下游用到（比如 `b1` 被整个图用到），
     不去重就会执行多次它的规则，梯度翻倍。用 `id(v)` 做集合判断，别用 `v in seen`
     （`Value` 没定义 `__eq__`/`__hash__`，默认按 id，但显式写 `id()` 更清楚也更保险）。
   * 为什么"后序"就够用：后序遍历保证每个节点出现在它所有父节点**之前**，
     于是倒过来走就是"从 loss 往参数"的正确顺序。（严格说后序在这里是充分必要的，
     因为图是 DAG，且规则只在被调用时读取 `out.grad`。）
2. **起点**：`self.grad = 1.0`。因为 $\partial L/\partial L = 1$，这是整条链的源头。
   这一行漏了，全部梯度是 0，**而且不会报错**。
3. **逆序执行规则**：`for v in reversed(topo): v._backward()`。
   顺序写成正序的话，节点在还没收到上游梯度时就被摊派，梯度偏小 —— 而且 loss 依然会降，
   只是降得慢，只有梯度校验能抓。

下面这张图把"逆拓扑序"播给你看：按 `reversed(topo)` 逐个节点把梯度摊到操作数上，每步显示各节点当前的 `.grad`；切到"错误顺序"（正序 / 少 seed 一遍），梯度当场漏掉、和解析值对不上。

@@fig topobwd@@

**手推一遍再写**（推荐，10 分钟）：用 §2.4 那个 `(a·b)²` 的图，在纸上把 `topo` 列表写出来
（应该是 `a, b, p, loss` 这个顺序），然后倒着走，检查每个节点的 grad。写代码时用它对拍。

<details>
<summary>参考答案 · TODO-3 <code>backward()</code></summary>

```python
    def backward(self) -> None:
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)          # 后序：父节点排在所有孩子后面

        build(self)
        self.grad = 1.0             # ∂L/∂L = 1，整条链的起点
        for v in reversed(topo):    # 逆拓扑序 = 从 loss 往参数方向走
            v._backward()
```

循环体就一行 `v._backward()`。写成 `if v._backward() is not None: v.backward()`
**也能过判据**（规则在条件表达式里已经被执行了，返回值恰好是 None，所以第二句永不跑）——
但那是靠巧合，不是靠逻辑；哪天规则有了返回值就会重复执行。判据抓得住 bug，抓不住丑。

`seen` 用 `id(v)` 而不是 `v`：`Value` 没定义 `__eq__`/`__hash__`，默认虽然也按 id，
但显式写出来更清楚，也不会因为以后你给 `Value` 加了 `__eq__` 而悄悄坏掉。

</details>

---

## 7. TODO-1A：`fit_by_autograd`

前面三个 TODO 是"造引擎"，这一个开始"用引擎做题"。它也是阶段 1 里唯一一个
**要你从零搭一个循环**的地方，所以把契约写在讲义里，不让你去代码注释里找。

### 7.1 谁调用你，传进来什么

[stages/s01_fit_line.py:94](file:///D:/Transformer/stages/s01_fit_line.py#L94) 就一行：

```python
a_g, b_g, curve_g = fit_by_autograd(x, y)
```

只传 `x`、`y`，其余三个参数用默认值。`x`、`y` 是 `make_data()` 造出来的两个 numpy
一维数组，各 100 个数：`x` 在 $[-1, 1]$ 均匀采样，`y` 严格等于 $3x + 1$。
**你不知道**这个式子，程序只能靠"猜 a、b → 看错多少 → 调"来逼近。

| 参数 | 类型 | 默认 | 你该拿它干什么 |
|---|---|---|---|
| `x` | `np.ndarray`，100 个 | — | 自变量。第 ② 步和 `y` 一起遍历 |
| `y` | `np.ndarray`，100 个 | — | 因变量，就是"标准答案" |
| `iters` | int | 3000 | 循环走多少步 |
| `lr` | float | 0.2 | 每步走多大。第 A 部分用的同一个值，两边才能对账 |
| `log_every` | int | 500 | 每多少步打印一次进度条；传 `0` 表示不打印 |

打印那行不用你设计，照第 A 部分抄：

```python
if log_every:
    print_progress(step + 1, iters, every=log_every, prefix="自动微分")
```

### 7.2 你必须返回什么（这条最容易搞反）

```python
return a.data, b.data, curve        # (float, float, list[float])
```

* `a`、`b` 要的是 **`.data`，也就是普通 float**，不是 `Value` 对象。
  判据那两行是 `abs(a_h - a_g)`，`a_h` 是第 A 部分的 float；你返回 `Value`，
  `Value` 没写 `__abs__`，当场 `TypeError`。
* `curve` 是每一轮的 loss，**长度必须正好 `iters`**（后面 `sparkline` 要拿它画图），
  元素是 `loss.data` 这种 float。

### 7.3 循环里五步，顺序不能乱

```flow
① **清零** —— `a.grad = b.grad = 0.0`
② **搭图** —— `loss = Value(0.0)` 起头 → 100 项 `e_i·e_i` 累加 → `loss = loss / len(x)`
③ **反传** —— `loss.backward()`，就是你写的 TODO-3
④ **记账** —— `curve.append(loss.data)`
⑤ **走一步** —— `a.data -= lr * a.grad`（`b` 同理）
```

带空格的骨架（`___` 就是你要填的，结构已经给你了 —— 因为结构不是练习，**填什么才是**）：

```python
    for step in range(iters):
        for v in (a, b):
            v.___ = 0.0                      # ① 哪个属性要清？
        loss = Value(0.0)                     # ② 起点必须是 Value，不能是 0.0
        for xi, yi in zip(x, y):
            e = ___                           #    这一点的误差：a·xi + b − yi
            loss = ___                        #    把 e 的贡献累加到 loss 上
        loss = ___                            #    100 项要取平均（不平均 = lr 放大 100 倍）
        loss.backward()                       # ③ 梯度落到 a.grad / b.grad
        curve.append(loss.___)                # ④ 要 float，画图用
        a.___ -= lr * a.___                   # ⑤ 改数，不重建对象
        b.___ -= lr * b.___
        if log_every:
            print_progress(step + 1, iters, every=log_every, prefix="自动微分")

    return a.___, b.___, curve                # ← float，不是 Value
```

五个填空的答案不在这里，但你可以用两条已知的事反推：
②那三空对照 §2.4 手算用的同一个式子；⑤那两空和第 A 部分 `a -= lr * ga` 是同一件事，
只是 `ga` 现在叫 `a.grad`。填完先跑 §7.6 的 30 秒自测，别急着跑整卷。

三条"为什么"，理解了就不会写错顺序：

* **①为什么必须清**：你的算子写的是 `self.grad += ...`（对的），不清零就等于
  这一轮的梯度叠在上一轮上。真实框架里 `optimizer.zero_grad()` 干的就是这件事。
* **②为什么每轮重建图**：`a`、`b` 这两个 `Value` 对象必须跨轮存在（①⑤都在摸它们），
  但 `loss` 这张图每轮新建、用完就扔。标量引擎慢就慢在这儿 —— 阶段 3 换成 numpy
  数组，为的就是把这 100 次 Python 循环摊平成一次矩阵乘。
* **⑤为什么改 `.data`**：`a = a - lr * a.grad` 会造出一个**新**的 `Value`，
  下一轮的梯度就落到别处去了。改 `.data` 是"同一个对象换个数"。

### 7.4 先在 4 个点上想明白，再上 100 个点

拿 $x=1,\ y=4$（真实 $a=3,b=1$）这一条数据，$a=0,b=0$ 起步：

$$e = a\cdot 1 + b - 4 = -4,\qquad L = e^2 = 16$$

$$\frac{\partial L}{\partial a} = 2e\cdot x = -8,\qquad \frac{\partial L}{\partial b} = 2e = -8$$

所以 $a \leftarrow 0 - 0.2\cdot(-8) = 1.6$。你的实现跑第 1 轮之后 `a.grad` 就该是这个数
（100 个点时是各点的平均，量级会变小，但符号和"往正方向走"这件事不变）。
**先让 4 个点对，再让 100 个点对** —— 一上来就跑 3000 轮，你分不清是引擎错还是循环错。

### 7.5 两个必踩的坑

* `Value(0, 0)` 是错的：`Value` 的第二个位置参数是 `_children`（要一个可迭代对象），
  症状是 `TypeError: 'int' object is not iterable`。参数就写 `Value(0.0)`，
  想起名字写 `Value(0.0, label="a")`。
* `e = a * xi + b - yi` 里 `xi` 是 numpy 的 `float64`，不用你手动包成 `Value`
  （`__mul__` 第一行做了 `isinstance` 转换）。但 **`loss` 起点必须是 `Value(0.0)`**，
  不能写 `loss = 0.0`：那样 `loss / len(x)` 出来的就不是图，`loss.backward()` 直接没这个方法。

### 7.6 写完先这么验（30 秒，不用等整卷）

在项目根目录跑这一行（PowerShell / cmd / Git Bash 通用，没有续行符）：

```
python webapp/checks.py TODO-1A
```

看到 `PASS 输出 a=3.0000 b=1.0000` 就对了；前置的 TODO-1/2/3 没写完时它会说
"被前置依赖挡住"，并告诉你先写哪个。想自己手搓一遍同样的数：

```
python webapp/checks.py
```

列全部 id，`python webapp/checks.py TODO-1 TODO-3 ACT-1` 一次跑几条。
或者直接在网站上点 **TODO-1A** 那个「测」—— 跑的是同一段代码。

<details>
<summary>参考答案 · TODO-1A <code>fit_by_autograd</code></summary>

```python
    a = Value(0.0, label="a")
    b = Value(0.0, label="b")
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    curve = []
    for step in range(iters):
        for v in (a, b):                    # ① 清零
            v.grad = 0.0
        loss = Value(0.0, label="loss")     # ② 搭图（每轮新建）
        for xi, yi in zip(xs, ys):
            e = a * xi + b - yi
            loss = loss + e * e
        loss = loss / len(xs)
        loss.backward()                     # ③ 反传
        curve.append(loss.data)             # ④ 记账
        a.data -= lr * a.grad               # ⑤ 走一步：改 .data，不重建对象
        b.data -= lr * b.grad
        if log_every:
            print_progress(step + 1, iters, every=log_every, prefix="自动微分")
    return a.data, b.data, curve            # ← float，不是 Value
```

对照 §7.3 那五个动作，一一对得上。两处值得注意：
`xs/ys` 先转成 Python float 列表不是多此一举 —— 让 `e = a * xi + b - yi` 里出现的
每个数都走你自己写的算子，别让 numpy 的标量偷偷参与建图；
`return a.data, b.data` 取的是 `.data`，返回 `Value` 会在判据 `abs(a_h - a_g)` 上 `TypeError`。

</details>

---

## 8. 验收判据逐条解读

`python stages/s01_fit_line.py` 输出应有 7 条 `[OK]`：

| 判据 | 它在证明什么 | 没过最可能是 |
|---|---|---|
| 学到斜率 a / 截距 b | 第 A 部分跑对了（基线） | 环境问题，不太可能挂 |
| 最终 MSE(手写) ≤ 1e-4 | 梯度下降本身在工作 | — |
| loss 下降倍数 ≥ 1e3 | 不是"随机游走碰上的" | lr 被你自己改小了 |
| **两种方法 a/b 之差 ≤ 1e-6** | 你的自动微分 = 手推偏导 | `__mul__` 里操作数弄反、或忘了 `out.grad` |
| **梯度校验(有限差分)** | 反向传播数值正确 | 梯度全 0、或顺序错、或用了 `=` 不是 `+=` |
| lr 演示 4 行 info | 观察项，不计入验收 | — |

最后两条是本阶段的全部意义。前面四条只是"程序在跑"，这两条才是"程序是对的"。

顺带看一眼那 4 行 lr 观察：`lr=1.5` 时 loss 变成 `3.9e+186`（发散），
`lr=0.001` 时还在 2.48（没走完）。稳定区间大约 `0.05~0.7`，上限由数据的
`mean(2x²)` 量级决定 —— 以后调 lr 时你脑子里应该有这根"尺度"，而不是一个魔法数字。

---

## 9. 常见失败模式对照表

| 症状 | 病因 | 修法 |
|---|---|---|
| `loss 完全不动`，`grad` 全是 0 | 规则里写了 `self.grad` 而不是 `out.grad`；或忘了 `self.grad = 1.0` | 检查 §3 第 1 点、§6 第 2 点 |
| `NameError: name 'k' is not defined`（在 _backward 里） | 闭包引用了循环外才存在的变量名 | 用 `f"**{k}"` 时的 `k` 是参数名，确认没写错 |
| 梯度是正确值的 **2 倍** | 拓扑列表没去重，同一节点规则跑了两遍 | `seen` 用 `id(v)` |
| 梯度**偏小**且 loss 也在降 | `topo` 顺序用正序遍历了 | 改成 `reversed(topo)` |
| `a*b` 里 a 的梯度用了 a 自己 | 局部导数写反（$\partial(ab)/\partial a=b$，别写成 $a$） | 用 §4 的自检一行验证 |
| `TypeError: unsupported operand` | 忘了把 float 包成 `Value` | 算子第一行统一 `other = other if isinstance(...) else Value(other)` |
| `RecursionError` | `_prev` 里塞进了自己 | 检查构造时传的 children |
| 有限差分对不上，误差 1e-3 级 | 你把 loss 写成 sum 而不是 mean（或反过来） | 判据的期望值是按 mean 算的 |

调试建议：别打印整张图。写三行小实验（§4 的自检），或者
`print(loss, a.grad, b.grad)` 对 §2.4 的手算数字。**先让 4 个节点对，再让 400 个节点对。**

---

## 10. 自查题（不看书答不上来就重读）

1. 为什么 `self.grad += other.data * out.grad` 里是 `+=`？举一个"必须累加"的具体场景。
2. `Value` 的 `_backward` 规则挂在**结果**节点上，而不是操作数上。为什么这样设计才对？
3. 后序遍历得到的 `topo`，为什么倒过来走就是正确的梯度传播顺序？
4. 如果我只把 `__mul__` 写完、`__pow__` 留着不写，`s01` 会在哪一步报错？为什么？
5. `loss.backward()` 调第二次会发生什么？（提示：`grad` 是累加的）训练循环靠哪一行避免这个？
6. 有限差分能发现"顺序写反"的 bug 吗？能发现"忘了清零"吗？各回答能/不能并说明理由。

答案都在上面的讲义正文里；第 5、6 题的答案在后面阶段会反复用到
（第 6 题：**能**发现顺序问题；**不能**发现清零遗漏，因为梯度校验每次都从头重建图 ——
所以清零的正确性要靠第 8 节"两种方法之差"那条判据来兜）。

---

## 11. 下一步

进 [docs/stage-02-非线性与容量.md](file:///D:/Transformer/docs/stage-02-%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%B8%8E%E5%AE%B9%E9%87%8F.md)。
阶段 2 只新增一个算子（tanh）和一个 25 参数的小网络，但会给你两个比技术更重要的概念：
**模型容量**（loss 降不下去有时不是你的错）和**学习率衰减**（为什么固定 lr 到不了最低点）。
