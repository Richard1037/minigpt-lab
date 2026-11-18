# 阶段 3b · 四个层的三件产出

预计：4 个层、约 60 行，1~2 小时。
验收命令：`python tests/test_layers_b.py`
要写的地方：`src/minigpt/nn.py` 的 TODO-9（GELU）、TODO-8（RMSNorm）、TODO-7（Embedding）、TODO-10（MLP）。

前提：[3a](s03a.html) 那 8 条已经绿了。这一卷**不重测 `Linear`**，但每一节都要用到它留下的三样东西：
「`axis=k` 就是被消失的那根轴」（3a §2.3）、"想合成什么形状，就谁转置"那张速查表（3a §2.5）、
以及"缓存用下划线、一次 forward 配一次 backward、梯度只累加"（3a §4）。

阶段 3 的动线（和 3a 顶上那张表是同一张，你现在在中间一行）：

| | 讲什么 | 写几个 TODO | 判据 |
|---|---|---|---|
| [3a](s03a.html) | numpy 轴运算地基 + 玩具层 `Scale` + `Linear` | TODO-5A、5、6 | `tests/test_layers_a.py` 8 条 |
| **3b（这一章）** | `GELU`、`RMSNorm`、`Embedding`、`MLP` | TODO-7~10 | `tests/test_layers_b.py` 6 条 |
| [3c](s03c.html) | `CrossEntropy`，以及"gradcheck 自己可信吗" | TODO-11 | `tests/test_layers_c.py` 5 条 |

---

## 1. 这一步在干什么

四个新层，**协议一个都没变**：`forward` 返回输出并把反向要用的中间量存进 `self._缓存`，
`backward` 交出那三件产出。变的只是每层"算法"那一栏 —— 所以这一卷的难点不是新规矩，
而是三件具体的事：

1. **`RMSNorm` 要你动笔推导**（本章唯一一处）。归一化把一行的 $C$ 个分量绑在一起了，
   少写修正项，梯度校验当场挂。
2. **`Embedding` 考的是 numpy 的一个陷阱**，不是数学：重复索引时 fancy 赋值是**覆盖**。
3. **`MLP` 第一次考"顺序与缓存"**：三个层串起来，反向要反着走，一步都不能乱。

按这个顺序写，每一步都能停下来跑判据：

```flow
**§3 GELU** —— 没有参数，三件产出只剩返回值那一件。10 分钟，确认协议你会了
↓ （做完能停下来：判据第 1 条绿）
**§4 RMSNorm** —— 本章唯一的推导。有 §3 的"只交一件产出"垫底，这里只管一件新事：耦合项
↓ （做完能停下来：判据第 2 条绿）
**§5 Embedding** —— 数学最浅（查表），坑最深（`np.add.at`）。和前两节的推理方式不同，单独放
↓ （做完能停下来：判据第 4~6 条绿）
**§6 MLP** —— 组装层：`Linear`(3a) + GELU(§3) + 缓存纪律(3a §4.1) 三样一起交卷
↓ （做完能停下来：整卷 6 条全绿）
```

---

## 2. 四个层的共同骨架

3a §6.2 那张 `Linear` 的表在这里逐字复用 —— 三件产出就是三个**槽位**，每层往里面填自己的东西：

* **槽位 A｜参数梯度**：累加进 `p.grad`（3a 里是 `dW`）。
* **槽位 B｜"被所有位置共用的那一维参数"的梯度**：形状是 `(C,)` 或 `(out,)`，
  所以永远是"把样本轴压掉"（3a 里是 `db = g2.sum(axis=0)`）。
* **槽位 C｜输入梯度**：`backward` 的返回值，形状和 forward 的输入一致；没有可导输入就返回 `None`。

判据盯的是三件具体的事：缓存了什么、散射还是覆盖、反向的顺序。
少了什么会立刻发生什么，先放这里当路标，每一行在对应小节里展开证明：

| 少了什么 | 立刻发生什么 | 在哪一节 |
|---|---|---|
| `GELU.forward` 里不缓存 $x$ | `backward` 无法算 $\partial y/\partial x$（它需要 $x$ 和 $t$ 两样），只能瞎猜 → 相对误差 $10^{0}$ | §3.4 |
| `GELU` 反向里漏掉链式那一项 | 只剩 $\tfrac12(1+t)$：形状全对、能跑、loss 也降，判据第 1 条红 | §3.3 |
| `RMSNorm` 漏 $-r\hat x\,s$ | 误差从 $10^{-9}$ 跳到 $10^{-1}$ 以上（差的是整整一项，不是小系数） | §4.2 |
| `RMSNorm` 漏 $\epsilon$ | 遇到全 0 行时 $r=1/0$，`inf` 顺残差流污染整张量 | §4.4 |
| `Embedding` 不缓存 `ids` | 反向不知道往哪几行散射 —— 只能把梯度平均分给全词表，那是错的 | §5.2 |
| `Embedding` 用 `gW[ids] = vals` | 重复 id 只保留最后一次：判据"重复 id 梯度累加"期望 12、实测 4 | §5.3 |
| `MLP.backward` 顺序写反 | 好的一种是当场 `ValueError`，坏的一种是形状全对但值错（§6.3） | §6.3 |

---

## 3. TODO-9：`GELU`

### 3.1 这是个激活函数：公式 + 要写的三件事

$$y=\tfrac12 x\Bigl(1+\tanh\bigl(K(x+Ax^{3})\bigr)\Bigr),\qquad
  K=\sqrt{2/\pi}\approx 0.7978846,\qquad A=0.044715$$

$$\frac{\partial y}{\partial x}=\frac12(1+t)\;+\;\frac12 x\,(1-t^{2})\,K(1+3Ax^{2}),
  \qquad t=\tanh\bigl(K(x+Ax^{3})\bigr)$$

| 要写的 | 内容 |
|---|---|
| `forward` | 算 `inner = K*(x + A*x**3)`、`t = np.tanh(inner)`，返回 $y$；缓存 `self._cache = (x, t, inner)` |
| `backward` | 把上面那条 $\partial y/\partial x$ 逐元素乘上游 $g$ 返回 |
| `K`、`A` | 写成类属性 `self.K` / `self.A`，正反向共用，别写两处字面量 |

纯逐元素：$(B,T,C)$ 进、$(B,T,C)$ 出，`backward` 里没有 `sum`、没有 `reshape`。
为什么敢用这个近似式、它和精确式差多少，见 §11。

### 3.2 三个具体数（先在纸上把值算出来，再对代码）

| $x$ | $\text{inner}$ | $t=\tanh(\text{inner})$ | $y$ | $\partial y/\partial x$ |
|---|---|---|---|---|
| $0$ | $0$ | $0$ | $0$ | $0.5$ |
| $+2$ | $1.881188$ | $0.954598$ | $1.954598$ | $1.086099$ |
| $-2$ | $-1.881188$ | $-0.954598$ | $-0.045402$ | $-0.086099$ |

$x=-2$ 那行 $y$ 和斜率**都是负的**：GELU 本来就放一点负值过去，别看到负梯度就以为写错了。

### 3.3 反向那两项怎么来的

$y=\tfrac12 x\,(1+t)$ 是 $x$ 与 $(1+t(x))$ 的**乘积**，所以 $\partial y/\partial x$ 必然有两项
（阶段 1–2 的乘积法则在这里一个字都没变）：

* 前一项 $\tfrac12(1+t)$：把 $(1+t)$ 当常数，对前面那个 $\tfrac12 x$ 求导。
* 后一项：把 $\tfrac12 x$ 当系数，对 $(1+t(x))$ 走链式 —— $\partial\tanh u/\partial u=1-t^{2}$
  （forward 里已经缓存了 $t$，不必再算一次 `tanh`），再乘 $\partial\,\text{inner}/\partial x=K(1+3Ax^{2})$。

### 3.4 缓存：三样，反向只用两样

| 缓存 | 在反向里是哪一项 | 不存会怎样 |
|---|---|---|
| $x$ | $\tfrac12 x(1-t^2)K(1+3Ax^2)$ 里的 $x$ 和 $x^2$ | 只能重算正向（多一次 `tanh`），或者干脆猜 |
| $t$ | $\tfrac12(1+t)$ 和 $1-t^{2}$ | 同上 —— 存它就是"用值表示导数"的那笔交易 |
| `inner` | 不出现 | 无影响（留着便于手算核对） |

### 3.5 写完就跑

```
python webapp/checks.py TODO-9      # 只测这一层（和网站按钮同一段代码）
python tests/test_layers_b.py       # 整卷；现在只有第 1 条该绿，第 2 条会停在 RMSNorm
```

网站里：阶段 3b → 右侧「单项快测」→ `TODO-9` 旁边点「测」。答案版跑出的量级是
$6.84\mathrm{e}{-}08$，阈值是相对误差 $10^{-4}$；量级不对就回头看 §3.3 那两项。

@@fig geludx@@

拖 `x` 看四个读数，拖 `lg ε` 让中心差分当场验一次导数。

<details>
<summary>参考答案 · TODO-9 <code>GELU</code>（判据没过之前别点开）</summary>

```python
    def forward(self, x):
        inner = self.K * (x + self.A * x ** 3)
        t = np.tanh(inner)
        self._cache = (x, t, inner)
        return 0.5 * x * (1.0 + t)

    def backward(self, g):
        x, t, inner = self._cache
        du_dx = self.K * (1.0 + 3.0 * self.A * x * x)
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
        return g * dy_dx
```

三件产出只有最后一件：`return g * dy_dx`，里面**没有任何** `self.xxx.grad +=` —— 它没有参数。
`du_dx` 单独起一行是为了和 §3.3 的公式逐项对上眼：$K(1+3Ax^{2})$ 就是它。

</details>

**这一节做完你能停下来。** 判据第 1 条绿了，说明"层可以只交一件产出"你已经会了。

---

## 4. TODO-8：`RMSNorm`

### 4.1 公式与要写的四件事

正向（最后一维是 $C$，全部**逐行**统计）：

$$r=\frac{1}{\sqrt{\frac1C\sum_{i}x_i^{2}+\epsilon}},\qquad \hat x = r x,\qquad y=w\odot\hat x$$

反向（记 $u = g\odot w$、$s=\frac1C\sum_i u_i\hat x_i$，$s$ 形状 $(B,T,1)$）：

$$\frac{\partial L}{\partial x} = r\,u - r\,\hat x\,s,\qquad
  \frac{\partial L}{\partial w} = \sum_{b,t} g\odot\hat x$$

| 要写的 | 内容 |
|---|---|
| `forward` | `ms = np.mean(x*x, axis=-1, keepdims=True)` → `self._r = 1/np.sqrt(ms+self.eps)` → `self._xh = x*self._r` → `return self._xh * self.w.data` |
| `backward` 返回 | `u = g*self.w.data` → `s = np.mean(u*self._xh, axis=-1, keepdims=True)` → `return self._r*u - self._r*self._xh*s` |
| `backward` 参数 | `self.w.grad += (g*self._xh).reshape(-1, C).sum(axis=0)` |
| 缓存 | `self._r` $(B,T,1)$、`self._xh` $(B,T,C)$；不要缓存 $y$ |

这一层**不是逐元素**的：$x$ 的一个分量变了，整行输出都跟着变，所以反向必须有第二项
$-r\hat x\,s$。它怎么来的、几何上是什么、和 LayerNorm/softmax 什么关系，都在 §11。

### 4.2 手算对账：$2\times2$ 那一组数

$x=[3,4]$、$w=[1,1]$、$g=[1,0]$、$C=2$，手算时 `eps` 传 `1e-12`。先把这五个量算出来：

| 量 | 值 |
|---|---|
| $\operatorname{mean}(x^{2})$ | $12.5$ |
| $r$ | $1/\sqrt{12.5}=0.282843$ |
| $\hat x = r x$ | $[0.848528,\ 1.131371]$ |
| $u=g\odot w$ | $[1,\ 0]$ |
| $s=\operatorname{mean}(u\odot\hat x)$ | $(0.848528+0)/2=0.424264$ |

再算 $\partial L/\partial x$，三条路要对上（$r\hat x=r^{2}x=x/12.5=[0.24,\ 0.32]$）：

| 算法 | $\partial L/\partial x$ |
|---|---|
| 完整式 $r\,u-r\,\hat x\,s$ | $[0.282843,\ 0]-[0.24,\ 0.32]\cdot0.424264=[0.181019,\ -0.135764]$ |
| 有限差分 | $[0.18102,\ -0.13576]$ |
| 只写 $r\,u$（漏耦合项） | $[0.282843,\ 0]$ → 判据第 2 条红 |

一格一格怎么推出来的（$\partial r/\partial x_i$、$2\times2$ 雅可比、少乘/多乘一个 $r$ 的两种错法、
几何解释）在 §11。

@@fig rmscouple@@

拖 $x$ 的四个分量：最下一行是浏览器**当场做的中心差分**（真值），上面两行分别是
"只写 $r\,u$"和完整公式。

### 4.3 参数梯度那一步：就是 3a §2.5 的 `db` 那一行

```
self.w.grad += (g * self._xh).reshape(-1, g.shape[-1]).sum(axis=0)
```

想要 $(C,)$，而 $g\odot\hat x$ 是 $(B,T,C)$：先摊平成 $(N,C)$ 再 `sum(axis=0)`，让 $N=B\cdot T$ 消失。
写成 `sum(axis=0)`（没摊平）得到的是 $(T,C)$，为什么别去"迁就"它见 §11。

### 4.4 这一层里的每一项，删了会怎样

| 式子/代码里的是哪一项 | 它管什么 | 删了/写错的后果 |
|---|---|---|
| $r$（缓存 `_r`） | 把行的 RMS 归到 1 | 没有它这层就退化成一个逐元素乘法 $w\odot x$：激活尺度随输入漂，深网络必炸 |
| $\hat x$（缓存 `_xh`） | 反向两处的"归一化后的 $x$"：$s$ 与 $w$ 的梯度 | 不缓存就得重算，且重算时容易忘记带 $r$ → 差一个 $r$ 因子，判据红 |
| $u=g\odot w$ | 先把"输出的梯度"换算成"归一化那一步的梯度" | 直接用 $g$（漏掉 $w$）：等价于假定 $w\equiv1$。**判据抓不到它** —— `RMSNorm(8)` 的 $w$ 初值就是 1，两种写法给出同一个数。怎么自己把它逼出来见 §4.5 |
| $s$ 与整项 $-r\hat x\,s$ | 归一化的耦合修正（§4.2） | 相对误差从 $10^{-9}$ 跳到 $10^{-1}$ 以上。这是本章最贵的一项 |
| `keepdims=True`（$s$、$ms$ 都要） | 保住 $(B,T,1)$ 这根轴 | 变成 $(B,T)$ 之后广播"从右往左"对齐，会拿 $T$ 去撞 $C$，当场 `operands could not be broadcast together` |
| $\epsilon$（默认 $10^{-5}$） | 分母的地板 | 某行 $x$ 全 0（padding 或早期塌缩）时 $r=1/0=\text{inf}$，`nan` 顺残差流污染整张量 |
| $w$ 的 `wd_exempt=True` | 通道尺度不是"权重矩阵" | AdamW 会把 $w$ 一路衰减向 0（每个通道都被慢慢关掉），模型变笨。阶段 5 起这条才看得见 |

`forward` 三行按 §4.1 的顺序写，不要把 $w$ 乘进缓存（存 `_y` 而不是 `_xh`）——
那样反向里 $s$ 和 $w$ 的梯度都要多除一次 $w$。

### 4.5 写完就跑

```
python webapp/checks.py TODO-8      # 只测这一层
python tests/test_layers_b.py       # 整卷；第 2 条该绿
```

答案版量级 $2.94\mathrm{e}{-}09$。判据用的是默认初值 $w=1$，所以"漏乘 $w$"它测不出；
自己逼它出来：`rm.w.data[:] = 0.5` 之后再跑一次 `gradcheck`，误差应当正好是 $0.50$。
手算 §4.2 时 `eps` 传 `1e-12`，否则第 6 位小数上和数值梯度对不齐。

<details>
<summary>参考答案 · TODO-8 <code>RMSNorm</code>（判据没过之前别点开）</summary>

```python
    def forward(self, x):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        self._r = 1.0 / np.sqrt(ms + self.eps)
        self._xh = x * self._r
        return self._xh * self.w.data

    def backward(self, g):
        u = g * self.w.data
        s = np.mean(u * self._xh, axis=-1, keepdims=True)
        gx = self._r * u - self._r * self._xh * s
        self.w.grad += (g * self._xh).reshape(-1, g.shape[-1]).sum(axis=0)
        return gx
```

对着 §4.1 的符号读这五行：`u`、`s`、`gx = r*u - r*xh*s`，最后 `w.grad` 那一行就是
$\sum_{b,t} g\odot\hat x$。$-r\hat x\,s$ 是全章唯一"少写了没人报错、只有判据知道"的一项。

</details>

**这一节做完你能停下来。** 判据第 2 条绿了，剩下的两层不再需要推导。

---

## 5. TODO-7：`Embedding`

### 5.1 forward 是一行 fancy indexing

$$E(v)=W[v],\qquad W\in\mathbb{R}^{V\times C},\quad \text{ids}\in\mathbb{Z}^{B\times T}
  \;\Longrightarrow\; \text{out}\in\mathbb{R}^{B\times T\times C}$$

`return self.weight.data[ids]` —— 输出形状**跟着 `ids` 走**，最后一维补上 $C$：
`ids` 里出现一次，就把 $W$ 的对应那一行**复制**一份过去。这就是查表的全部数学。

| 要写的 | 内容 |
|---|---|
| `forward` | `self._ids = np.asarray(ids)` → `return self.weight.data[self._ids]` |
| 缓存 | `self._ids` $(B,T)$，**保持整数**（加 `dtype=np.float64` 会直接 `IndexError`，也会让 3a §5 的"整数输入自动跳过"失效） |
| 权重形状 | $W$ 存成 $(V,C)$，第 $v$ 行就是词 $v$ 的向量。为什么不是 $(C,V)$，见 §11 |

### 5.2 backward：三行的散射

```
gW = np.zeros_like(weight.data)                     # (V, C)
np.add.at(gW, ids.reshape(-1), g.reshape(-1, C))    # 同一个 id 出现多次 → 累加
weight.grad += gW
return None                                         # 输入是整数，不可导
```

形状检查还是 3a §2.5 那套：`g` 是 $(B,T,C)$，摊成 $(N,C)$；索引是 $N$ 个整数；
目标是 $(V,C)$。**"谁被重复使用，梯度就往谁身上汇"** —— $N$ 个位置的梯度汇到 $V$ 行上。

### 5.3 必须是 `np.add.at`，不能是 `gW[ids] = vals`

fancy-index **赋值**对重复索引是**覆盖**（numpy 把写入按索引排好，同一行只留最后一次），
`np.add.at(gW, idx, vals)` 是**逐条累加**、顺序无关。判据用 `ids = [3,3,3,0,7]`、
$g$ 全 $1$（$C=4$）钉住这条：

| 写法 | `weight.grad[3].sum()` | 判据"重复 id 梯度累加" |
|---|---|---|
| `np.add.at` | $3\times 4=12$ | 期望 12、实测 12 → 绿 |
| `gW[ids] = g` | $4$（只剩最后一次） | 期望 12、实测 4 → 红，误差 8 |

`+= gW` 而不是 `= gW`（3a 约定 3）：`gW` 加完的是这一批，**多次 backward** 之间还要累加。
gradcheck 为什么可能抓不到覆盖写法、判据为什么必须自己造一条重复 id 的用例，见 §11。

@@fig scatter@@

点上面任意一个 `id` 就写入一次，切换"覆盖 ⇄ 累加"看第 3 行从 4 变 12。

### 5.4 没用到的行梯度恰好是 0 —— 这是正确的

`ids` 里没有 `1`，所以 `weight.grad[1]` 必须**全 0**：判据钉的是"没有越界污染"
（索引弄错了行、或者忘了 `zeros_like` 而从上一批数据带了脏值进来）。

`return None`（3a 约定 4）：token 序号不可导，梯度到这一层就该停住。
一个 batch 里绝大多数行是 0 为什么**不是** bug，见 §11。

### 5.5 写完就跑

```
python webapp/checks.py TODO-7      # 只测这一层
python tests/test_layers_b.py       # 整卷；后三条该绿
```

答案版量级 $5.33\mathrm{e}{-}11$（只查 `weight` 一个参数：输入是整数，有限差分动不了它）。

<details>
<summary>参考答案 · TODO-7 <code>Embedding</code>（判据没过之前别点开）</summary>

```python
    def forward(self, ids):
        self._ids = np.asarray(ids)
        return self.weight.data[self._ids]

    def backward(self, g):
        gW = np.zeros_like(self.weight.data)
        np.add.at(gW, self._ids.reshape(-1), np.asarray(g).reshape(-1, g.shape[-1]))
        self.weight.grad += gW
        return None                     # 输入是整数序号，不可导
```

`np.add.at` 是这一层的全部：**fancy 赋值对重复索引是覆盖，不是累加**。
返回 `None` 也是有意的 —— token 序号不可导，判据在这层只查参数梯度（$5.33\mathrm{e}{-}11$）。

</details>

**这一节做完你能停下来。** 判据后三条（`Embedding(参数梯度)` / `重复 id 梯度累加` /
`未出现的行梯度为 0`）绿了，就剩最后一层组装。

---

## 6. TODO-10：`MLP`

### 6.1 它就是三个层串起来，没有新数学

$$\mathrm{MLP}(x)=W_2\,\mathrm{GELU}(W_1 x+b_1)+b_2,\qquad
  W_1\in\mathbb{R}^{H\times C},\ W_2\in\mathbb{R}^{C\times H}$$

骨架里 `MLP(6, 16)`：$C=6$、$H=16$，输入 $(2,3,6)$、输出 $(2,3,6)$。`__init__` 已给你写好：
`self.fc1 = Linear(C, hidden)`、`self.act = GELU()`、`self.fc2 = Linear(hidden, C)`。

| 要求 | 内容 |
|---|---|
| 进出口同形 | $H$ 只在**内部**出现；阶段 4 的 `Block` 要把 `MLP(x)` 加回 `x`，形状不同就加不了 |
| 梯度形状 | $(B,T,C)\to(B,T,H)\to(B,T,C)$，中间变一次（`fc1` 的 `in_f=C`、`fc2` 的 `in_f=hidden`） |
| 逐位置 | 每个 token 独立过同一个 MLP，token 之间不交换信息（那是注意力的活） |
| 不含残差/norm | 那是阶段 4 `Block` 的事：`norm → attn → 残差 → norm → mlp → 残差` |

### 6.2 forward 三行、backward 三行，后者是前者的镜像

```
forward:  h = self.act(self.fc1(x));  return self.fc2(h)
backward: g → fc2.backward → act.backward → fc1.backward → return
```

三个要点：

1. 用 `self.act(self.fc1(x))` 也就是 `__call__`，不是 `self.fc1.forward(x)`。
   差别在 3a §4.3 说过：以后所有横切的东西（dropout、缓存开关）都挂在 `__call__` 上。
2. `backward` 调的是**子模块的 `backward`** —— 它们会各自把参数梯度累加进自己的 `p.grad`。
   所以 `MLP.backward` 自己**一行 `.grad` 都不写**。
3. 每步的**返回值**是下一步的**入参**；顺序严格反。这就是"反向 = 正向的镜像"，
   整份手册到处成立（阶段 5 的 `Block`、阶段 6 的整个模型都是这一条放大版）。

### 6.3 顺序错了会怎样

| 写法 | 表现 |
|---|---|
| 把 $g$（$(2,3,6)$）直接喂 `fc1.backward` | `ValueError: cannot reshape array of size 36 into shape (16)` |
| 正向写成 `self.fc2(self.fc1(self.act(x)))` | 反向撞上 `operands could not be broadcast together with shapes (2,3,16) (2,3,6)` |
| `backward` 少走一次 `self.act.backward` | 三处形状全合法、代码跑通、loss 也降；判据第 3 条最大相对误差 $0.82$（答案版 $1.72\mathrm{e}{-}07$） |

`gradcheck` 的明细会**点名报错在哪个张量上**（形如 `fc1.W 第 k 个元素`）：
**红报在哪个参数，错就在那一段**。

### 6.4 缓存：`MLP` 自己不缓存任何东西

中间激活由 `fc1`/`act`/`fc2` 各自的缓存记住 —— **缓存归每个子模块自己**。
答案实现里那行 `self._out = h` 反向没用到，删掉判据第 3 条照样绿（留着只为打印形状排查）。

两条不能省的纪律：

* 不要在 `backward` 里重算正向：既覆盖子模块缓存（违反 3a §4.1），又慢一倍。
* 不要用两次 forward 夹一次 backward（比如为了打印 loss 又 forward 一次）：
  形状全对、loss 照降，只有梯度算到了另一条数据上。

### 6.5 写完就跑

```
python webapp/checks.py TODO-10     # 只测这一层
python tests/test_layers_b.py       # 整卷 6 条该全绿
```

@@fig mlpbwd@@

▶ 播放 / ⏭ 单步 / ⏸ 暂停 / ↺ 重置：正向 3 步，反向按 `fc2 → act → fc1` 的真实顺序 7 步，
每步标出输入形状、输出形状、公式和当场算出的小矩阵；最后一步拿逐步累积的结果和一次性直算对拍。

<details>
<summary>参考答案 · TODO-10 <code>MLP</code>（判据没过之前别点开）</summary>

```python
    def forward(self, x):
        h = self.act(self.fc1(x))
        self._out = h
        return self.fc2(h)

    def backward(self, g):
        g_h = self.fc2.backward(g)
        g_h = self.act.backward(g_h)
        return self.fc1.backward(g_h)
```

`g_h` 被赋值两次，正是"每步返回值是下一步入参"的直译：第一次是 $(B,T,H)$，第二次是 $(B,T,C)$。
`MLP.backward` 里没有 `+=`，因为参数梯度是子模块的事 —— 这是**组合层**和**原子层**的分工。

</details>

四节写完，回来把这张表逐格对一遍：每一格是你在哪一节填的。

| 层 | 形状 | 有哪些参数 | A｜参数梯度 | B｜共用一维参数 | C｜输入梯度（返回值） | `forward` 缓存 |
|---|---|---|---|---|---|---|
| `GELU` | $(B,T,C)\to(B,T,C)$ | 无 | **不存在** | 不存在 | $g\odot\dfrac{\partial y}{\partial x}$，和 $x$ 同形 | $(x,\ t,\ \text{inner})$ |
| `RMSNorm` | $(B,T,C)\to(B,T,C)$ | $w$，形状 $(C,)$ | 和槽位 B 合并（§4.3） | $w$：$\sum_{b,t} g\odot\hat x$，$(C,)$ | $r\,u-r\,\hat x\,s$ | `_r` $(B,T,1)$、`_xh` $(B,T,C)$ |
| `Embedding` | $(B,T)\to(B,T,C)$ | `weight`，形状 $(V,C)$ | 按 id **散射累加**进 `gW`，$(V,C)$ | 不存在 | `None`（输入是整数序号） | `_ids` $(B,T)$ |
| `MLP` | $(B,T,C)\to(B,T,C)$ | `fc1`/`fc2` 的 $W$、$b$ 共 4 个 | 由子模块**自己**累加 | 同左（`fc1.b`、`fc2.b`） | 一路 return 到 `fc1.backward` | 自己不缓存（§6.4） |

三个"空格"是判据盯着的三件事（`GELU` 没有参数 / `RMSNorm` 的 A 与 B 是同一件事 /
`Embedding` 的 C 是 `None`），逐条解释见 §11。

---

## 7. 这一卷的验收判据

```checks
judge: tests/test_layers_b.py
```

红了的去向：`GELU` → §3.3 数那两项；`RMSNorm` → §4.2 那三条路对不对得上；
`MLP` → §6.3 那张表；`Embedding` 后两条 → §5.3、§5.4。

---

## 8. 常见失败模式对照表

| 症状 | 大概率原因 | 去哪一节 |
|---|---|---|
| `GELU` 那条误差 $10^{0}$ 量级 | 反向漏了链式那一项，只剩 $\tfrac12(1+t)$ | §3.3 |
| `AttributeError: 'GELU' object has no attribute '_cache'` | `forward` 忘了 `self._cache = ...` | §3.4 |
| 在 `GELU.backward` 里写了 `self.xxx.grad +=` | 这层没有参数，槽位 A、B 是空的 | §2 那张表 |
| `RMSNorm` 误差 $10^{-1}$ 以上（常接近 1） | 少了 $-r\hat x\,s$ 那一项 | §4.2 |
| `RMSNorm` 误差刚好差一个 $r$ 因子 | $s$ 里用了 $x$ 而不是 $\hat x$，或者少乘/多乘一个 $r$ | §4.2 最后一段 |
| `operands could not be broadcast together` | $s$ 或 `ms` 忘了 `keepdims=True` | §4.4 |
| `RMSNorm` 的 `w.grad` 形状是 $(T,C)$ | 没摊平就 `sum(axis=0)` | §4.3 |
| `Embedding` 那条判据报"期望 12 实测 4" | 用了 fancy 赋值（覆盖）而不是 `np.add.at` | §5.3 |
| `Embedding` gradcheck 绿但你还是错了 | 这批 `ids` 恰好没重复 —— 所以判据另有第 5、6 条 | §5.3 |
| `IndexError: arrays used as indices must be integer` | `np.asarray(ids, dtype=np.float64)` 把索引转成了浮点 | §5.1 |
| `cannot reshape array of size 36 into shape (16)` | `MLP.backward` 顺序写反，最外层的 $g$ 喂给了 `fc1` | §6.3 |
| `MLP` 形状全对、loss 照降，误差却 $0.8$ 左右 | `backward` 三行漏了中间那行（没走 `self.act.backward`） | §6.3 |
| `MLP` 误差 $10^{-1}$ 以上但不知错在哪 | 明细点名的张量就是出错那一段（`fc1.W` 和 `fc2.W` 指向不同的毛病） | §6.3 |
| 全过但阶段 4 的 `Block` 报错 | 缓存被复用（forward 调了两次才 backward） | §6.4 |

---

## 9. 自查题

不用查资料。每题先把自己的答案写进输入框再点"判一下"；想不出来按提示 1→2→3，
三级提示从最抽象到最具体；还不行就点"回讲义读这一节"。

```quiz
q: 这四个层（GELU / RMSNorm / Embedding / MLP）各自的"三件产出"里，哪一件是空的？空的原因分别是什么？
keys: GELU无参数/没有参数/缺A、B; 共用/同一个; 整数不可导/不可导; 子模块/自己认领
hint1: 三件产出 = 返回的 dx、累进的 dW、累进的 db。逐层问：它有参数吗？它的输入都可导吗？
hint2: GELU 和 MLP 没有自己的参数；Embedding 的输入是整数序号。
hint3: GELU 缺 dW/db（无参数）；RMSNorm 的 dW/db 是同一个 w；Embedding 缺 dx（整数不可导）；MLP 的三件全由 fc1/fc2 自己认领。
link: sec-2
ans: `GELU` 缺 dW、db（无参数）；`RMSNorm` 的 dW 和 db 是同一个（唯一的 $w$，形状 $(C,)$、被所有位置共用）；`Embedding` 缺 dx（整数不可导）；`MLP` 三件全空 —— 由子模块自己认领（§2）。
```

```quiz
q: `GELU` 的 $\partial y/\partial x$ 为什么必然有两项？$1-t^{2}$ 是从哪来的？
keys: 乘积/两个因子; tanh的导数/自己的值/链式
hint1: 把 y 写成两个因子的乘积，看看几个因子含 x。
hint2: 乘积求导：前导后不导 + 后导前不导。
hint3: $y=\tfrac12 x(1+t)$，两个因子都含 x 所以两项；$1-t^2$ 是 tanh 的导数用**它自己的值** t 表示（阶段 2 的规则）。
link: sec-3
ans: $y=\tfrac12 x(1+t(x))$ 是两个含 $x$ 的因子之积：前因子求导给 $\tfrac12(1+t)$，后因子走链式。$1-t^{2}$ 是 $\tanh$ 的导数用**自己的值**表示（§3）。
```

```quiz
q: RMSNorm 反向里减掉的那一项（$s$ 项）在物理上代表什么？为什么"只写 $r\,u$"是错的？
keys: 长度/钉住/锁住/方向; 常数/ r 也依赖x / r不是常数
hint1: 归一化之后，每行的长度还是自由变量吗？
hint2: 沿 $\hat x$ 方向扰动 x，输出会变吗？
hint3: 归一化把每行长度钉在 $\sqrt C$，沿 $\hat x$ 的扰动被禁止；减掉的就是那份"被禁止方向"的贡献。只写 $ru$ 等于假装 r 是常数。
link: sec-4
ans: 归一化把每行长度钉在 $\sqrt{C}$，沿 $\hat x$ 方向的扰动不会改变输出；减掉的就是那份"被禁止的方向"。只写 $r u$ 等于假装 $r$ 是常数（§4 的几何解释）。
```

```quiz
q: 手算 §4 那组数：$x=[3,4]$、$g=[1,0]$ 时 $\partial L/\partial x$ 是多少？只写 $r u$ 会得到什么？
accept: [0.181019,-0.135764]/[0.181,-0.136]/0.181019,-0.135764
hint1: 先算 r：$\sqrt{(9+16)/2}=5/\sqrt2$。
hint2: 完整式子是 $g/r$ 减去 $s$ 项；只写 $ru$ 时你其实算的是 $g\cdot$ 什么？
hint3: 正确答案 $[0.181019, -0.135764]$；只写 $ru$（即 $g/r$ 漏减项的对照）给 $[0.282843, 0]$ —— 第二维本该收到"被禁止方向"退回的量。
link: sec-4
ans: $[0.181019,\ -0.135764]$；只写 $r u$ 得到 $[0.282843,\ 0]$（§4）。
```

```quiz
q: `np.add.at` 和 `gW[ids] = vals` 的区别是什么？为什么 gradcheck 单独抓不住后者？
keys: 累加/覆盖; 重复/有没有重复/抽到
hint1: 同一个索引出现两次时，两种写法各留下什么？
hint2: gradcheck 的 ids 是随机抽的。
hint3: `=` 对重复索引**覆盖**（只留最后一条），`add.at` **逐条累加**。gradcheck 抓不抓得住取决于抽到的 ids 有没有重复 —— 无重复时全绿，所以判据自己构造重复 id。
link: sec-5
ans: 后者对重复索引**覆盖**，前者**逐条累加**。gradcheck 能不能抓到取决于抽到的 `ids` 有没有重复 —— 实测无重复时全绿，所以判据要自己构造重复 id（§5）。
```

```quiz
q: `Embedding.backward` 为什么返回 `None`？如果返回零数组会掩盖什么？
keys: 不可导/整数; 误以为/断链/接错/还有一路
hint1: 约定 4 对"没有可导输入"的层是怎么规定的？
hint2: 返回零数组不会报错 —— 那问题出在谁身上？
hint3: 整数序号不可导，约定 4 要求返回 `None`。返回零数组会让以后多输入层的反向调度误以为"还有一路可导输入"，把断链接错。
link: sec-5
ans: token 序号不可导，约定 4 要求"没有可导输入就返回 `None`"。返回零数组不会报错，但会让多输入层的反向调度误以为"还有一路可导输入"，把断链接错（§5）。
```

```quiz
q: `MLP` 进出口的 $C$ 为什么必须相同，而 $H$ 可以随便改？
keys: 残差/加回/同形; 支路/内部/容量
hint1: 阶段 4 的 Block 拿 MLP 的输出干什么？
hint2: 两个张量相加的形状要求是什么？
hint3: Block 把 `MLP(x)` 加回 `x`，进出同形才加得动；H 只活在支路内部，改它只影响容量和速度。
link: sec-6
ans: 阶段 4 的 `Block` 把 `MLP(x)` 加回 `x`，进出同形才加得动；$H$ 只活在支路内部，改它只影响容量和速度（§6）。
```

```quiz
q: `MLP.backward` 里为什么一行 `.grad` 都不写？
keys: 子模块/自己/fc1/fc2; 重复累加/加两次
hint1: MLP 的参数挂在谁名下？
hint2: 调 fc1.backward 的时候，谁在写 fc1.W.grad？
hint3: 参数梯度是 fc1/fc2 自己的事 —— 组合层只把梯度按镜像传下去；自己再写一遍会让同一份 W.grad 被加两次。
link: sec-6
ans: 参数梯度是 `fc1`/`fc2` 自己的事 —— 组合层只负责把梯度按镜像传下去。自己再写一遍会重复累加（同一份 `W.grad` 被加两次）（§6）。
```

---

## 10. 下一步

`python tests/test_layers_b.py` 6 条全绿 → 去
[阶段 3c · CrossEntropy 与测谎仪自检](s03c.html)：本阶段最后一个层，也是唯一
**输出标量**的那个层；那一卷还会用两个已知答案的层反过来考 `gradcheck`，
把你以后拿到"最大相对误差 3e-2"时该怀疑谁这件事钉死。

先做一件事再走：把这一卷里你**红过**的那条判据的名字，和当时错误量级，写在纸上。
阶段 4 你会再遇到同一类错，那张纸比讲义快。

---

## 11. 进阶推导（现在可以跳过）

正文只给了公式和要写的代码。这一节是那些公式**怎么来的**、以及每个约定背后的理由。

### 11.1 GELU：为什么敢用 tanh 近似式

精确式是 $y=\tfrac12 x\bigl(1+\operatorname{erf}(x/\sqrt2)\bigr)$。近似式和它在整条实轴上的最大偏差：

| 量 | 最大绝对偏差 | 出现在 |
|---|---|---|
| $y$ | $4.7\mathrm{e}{-}04$ | $x\approx-2.70$ |
| $\partial y/\partial x$ | $8.7\mathrm{e}{-}04$ | $x\approx-2.02$ |

$x=-2$ 那一点的对照：

| 版本 | $y$ | $\partial y/\partial x$ |
|---|---|---|
| tanh 近似（你要写的） | $-0.045402$ | $-0.086099$ |
| 精确 erf | $-0.045500$ | $-0.085232$ |

判据阈值是相对误差 $10^{-4}$，而中心差分自己就带 $10^{-8}$ 量级的噪声；这点偏差远小于
训练里任何一步的舍入。用近似式的收益是 `np.tanh` 一次调用，不必依赖 `scipy.special.erf`。

三个能记住的观察：

* $x=0$ 处斜率正好 $0.5$ —— 它是"软化的 ReLU"，在 0 附近一半通过、一半不通过。
* $x\gtrsim 2$ 之后 $y\approx x$（$t\to1$）；负区 $y\to0$ 但不是 0。
* 负区那点泄漏是真的，精确 erf 版也一样（$-0.0455$），不是你写错了。

负区导数有个极小值：$x\approx-1.4185$ 处 $\partial y/\partial x\approx-0.1290$ ——
GELU 在这一段是**递减**的，`geludx` 那张图拖到那里能看到读数为负。

漏掉链式那一项（只写 $\tfrac12(1+t)$）的症状：形状全对、能跑、loss 照降，
判据第 1 条给 $10^{0}$ 量级的相对误差 —— 差的是整整一项，不是小系数。

$K$、$A$ 写成类属性而不是两处字面量的理由：正向算 `inner`、反向算 $K(1+3Ax^{2})$ 都要用它们。
写两处时最常见的错是反向那份把 $3Ax^{2}$ 打成 $3Ax$：gradcheck 一定红，
但红报只说"误差大"，不会说"你两个字面量不一样"。

### 11.2 RMSNorm：耦合项 $-r\hat x\,s$ 是怎么来的

前面所有层都是逐元素的（$y_c$ 只依赖 $x_c$），这一层不是：$x$ 的某个分量变了，
$\operatorname{mean}(x^{2})$ 也变了，于是**整行**输出都受影响。所以 $\partial L/\partial x$
不能只写 $r\,u$ —— 那等于"把 $r$ 当常数"，而 $r$ 恰恰是这一层唯一由输入自己决定的量。

还是 §4.2 那组数（$x=[3,4]$、$w=[1,1]$、$g=[1,0]$、$C=2$、$\epsilon=0$）。
先看 $r$ 对分量的导数 —— 耦合就藏在这一步：

$$\frac{\partial r}{\partial x_0}=-\frac12\bigl(\operatorname{mean}(x^{2})\bigr)^{-3/2}\cdot\frac{2x_0}{C}
  =-\frac12(12.5)^{-3/2}\cdot 3=-0.033942$$
$$\frac{\partial r}{\partial x_1}=-\frac12(12.5)^{-3/2}\cdot 4=-0.045255$$

$y_0=r x_0$ 里 $r$ 和 $x_0$ **都**依赖 $x_0$，乘积法则给出两项；$y_1=r x_1$ 也一起动了
（这正是"整行联动"）：

$$\frac{\partial y_0}{\partial x_0}=r+x_0\frac{\partial r}{\partial x_0}
  =0.282843+3\cdot(-0.033942)=0.181019$$
$$\frac{\partial y_1}{\partial x_0}=x_1\frac{\partial r}{\partial x_0}
  =4\cdot(-0.033942)=-0.135768$$

$2\times2$ 雅可比剩下两格把下标 $0/1$ 互换。按 $L=\sum_c y_c g_c$ 合并，
$g=[1,0]$ 意味着只取 $y_0$ 那一行：

$$\frac{\partial L}{\partial x_0}=1\cdot0.181019+0\cdot(\cdots)=\mathbf{0.181019},\qquad
  \frac{\partial L}{\partial x_1}=3\cdot(-0.045255)=\mathbf{-0.135764}$$

和 §4.2 表里最终式给的数一模一样 —— 这就是那次手算的全部目的。最容易错在 $r\hat x$ 这一处：
$\hat x$ 里已经含了一个 $r$，所以 $r\hat x\,s$ 整体是 $r^{2}$ 阶的（本例就是 $x/12.5$）。

少乘一个 $r$ 会减掉 $[0.360,\ 0.480]$，得 $[-0.077,\ -0.480]$；多乘一个 $r$ 只减掉
$[0.0288,\ 0.0384]$，得 $[0.2541,\ -0.0384]$ —— 两种都错，两种都会被这条判据按在地上。

几何解释（比公式好记）：归一化把每行的长度钉住了（$\epsilon=0$ 时 $\lVert\hat x\rVert=\sqrt{C}$，
与 $x$ 的尺度无关），所以**沿着 $\hat x$ 方向的扰动是被禁止的**，它不会改变输出。
$r\hat x\,s$ 减掉的正是 $u$ 在 $\hat x$ 方向上的那一份，$s$ 就是那份的（归一化）大小。
`rmscouple` 那张图里"漏项那一行"只在 $s\approx0$ 时追上真值 —— 不是公式简化了，
是那一行恰好没耦合。

同族的兄弟在阶段 4：softmax 反向 $g=p\odot(gp-\sum(gp\odot p))$ 里那个减号，
钉的是"**每行和为 1**"而不是长度。

为什么它比 LayerNorm 流行：少一次减均值（少一个全局统计量）、少一个 bias 参数，
实测效果相当，实现和反向都更短。你刚推的这项耦合修正在 LayerNorm 里**也存在而且更复杂**
（多一个均值方向的约束）—— 这就是少减一次均值的真实收益。

§4.3 为什么必须先 `reshape(-1, C)` 再 `sum(axis=0)`：`(g*self._xh).sum(axis=(0,1))` 也能给出
$(C,)$，但顺手写成 `sum(axis=0)`（没摊平）得到的是 $(T,C)$。形状不对时报错还算客气，
更糟的是有人会把 $w$ 定义成 $(T,C)$ 去"迁就"它 —— 那等于让每个位置有一份自己的通道尺度，
参数多 $T$ 倍，而且换序列长度就废。

### 11.3 Embedding：gradcheck 的盲区，和 $(V,C)$ 的理由

判据为什么要自己造一条重复 id 的用例：`test_layers_b.py` 那组随机 `ids` 恰好有重复
（按判据的抽取顺序是 `[6,6,8,1,7,1,1,8,0,7]`），所以覆盖写法在**这次** gradcheck 里也会红
（最大相对误差 $1.21$）。但换一段没有重复的 `ids`（`[9,7,5,2]`）：覆盖写法给出
$1.1\mathrm{e}{-}11$、**全绿**。这个 bug 抓不抓得到，取决于抽到的数据里有没有重复 id。
这是"测谎仪也有盲区"的第一个实例，[3c §4](s03c.html#sec-4) 会正面谈这件事。

`self.weight.grad = gW`（覆盖而不是累加）和 `test_layers_a.py` 里"backward 两次后梯度翻倍"
那条抓的是同一个毛病，只是那里用 `Scale` 演示。真实触发场景是梯度累积，
以及阶段 5 的 weight tying（同一个张量走两条路，两次 backward）。

$W$ 存成 $(V,C)$ 而不是 $(C,V)$ 有两个理由：第 $v$ 行就是词 $v$ 的向量，取一行是一次连续内存读取；
而且阶段 5 的 weight tying（输出头和词嵌入共用同一个张量）之所以"免费"，
是因为 `Linear(C→V)` 的 $W$ 按 3a §6.1 的约定**也**存成 $(V,C)$ —— 两边要的形状刚好是同一个。

"绝大多数行梯度是 0"为什么不是 bug：一个 batch 里没出现的词占绝大多数
（$V=8192$、$N\approx512$ 时几乎全是 0），稀疏是正常的，这正是查表比 one-hot × 矩阵乘便宜的原因。
阶段 5 的"无死参数"检查要求的是**每个张量**有非零梯度，不要求每个**元素**都有 ——
它管的是接线，不是使用情况。

`return None` 而不是 `np.zeros_like(ids, dtype=float)`：约定 4 要求没有可导输入就返回 `None`，
梯度到这里就该停住（往 token 序号方向没有可微的东西）。返回一个全 0 浮点数组既不报错也没意义，
还会让以后新增的"多输入层"调试逻辑误判。

### 11.4 MLP：三个"空格"，和两次 forward 的陷阱

* `GELU` 的 A、B 是空的 —— 它**没有参数**，所以 `backward` 里出现 `self.xxx.grad` 就是错的。
  判据照样测得到它，因为槽位 C 有东西（$x$ 是浮点数组，有限差分能扰动它）。
* `RMSNorm` 的 A 和 B 是同一件事：唯一的参数 $w$ 形状 $(C,)$，被 $B\cdot T$ 个位置共用，
  走的是 3a §2.3 那条规律（"想得到 $(C,)$，就让 $N=B\cdot T$ 消失"）。
* `Embedding` 的 C 是 `None`：token 序号不可导，这不是偷懒，是 3a 约定 4 的字面要求。

"两次 forward 夹一次 backward"为什么难查：子模块的缓存是**实例属性**，第二次 forward 会把
`_cache`/`_r`/`_xh`/`_ids` 全部覆盖成新数据；`backward` 用的是新缓存，传进来的 $g$ 却是
旧那条数据的。形状完全一致、loss 照降，只有梯度是错的。阶段 5 写训练循环时
（打印 loss、评估、采样）一定会遇到 —— 记住**forward 与 backward 严格一对一**。
