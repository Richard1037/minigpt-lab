# 阶段 3c · CrossEntropy 与测谎仪自检

预计：1 个层、约 15 行，40~80 分钟。
验收命令：`python tests/test_layers_c.py`
要写的地方：`src/minigpt/nn.py` 的 TODO-11（`CrossEntropy.forward` / `CrossEntropy.backward`）。

前提：[3a](s03a.html) 那 8 条、[3b](s03b.html) 那 6 条已经绿了。这一卷不重测前面的层，
但会反复用到它们留下的三样东西：「`reshape` 只把前面的轴乘成一根，最后一轴一步没动」（3a §2.2）、
三件产出的三个槽位（3b §2 那张表），以及"缓存用下划线开头、一次 forward 配一次 backward"（3a §4）。

阶段 3 的动线（和 3a/3b 顶上那张表是同一张，你现在在最后一行）：

| | 讲什么 | 写几个 TODO | 判据 |
|---|---|---|---|
| [3a](s03a.html) | numpy 轴运算地基 + 玩具层 `Scale` + `Linear` | TODO-5A、5、6 | `tests/test_layers_a.py` 8 条 |
| [3b](s03b.html) | `GELU`、`RMSNorm`、`Embedding`、`MLP` | TODO-7~10 | `tests/test_layers_b.py` 6 条 |
| **3c（这一章）** | `CrossEntropy`，以及"gradcheck 自己可信吗" | TODO-11 | `tests/test_layers_c.py` 5 条 |

---

## 1. 这一步在干什么

只有一个层，但它和前面六个（`Scale`、`Linear`、`Embedding`、`RMSNorm`、`GELU`、`MLP`）不一样：

* **它是本阶段唯一输出标量的层。** 前面每一层的 `forward` 都交回一个数组，这一层交回一个
  `float`。所以"三件产出"在这里退化到只剩一件（输入梯度），而 `gradcheck` 要在"输出不是数组"
  的情况下工作 —— §4 会把这件事讲透。
* **它是全项目错法最多的一层。** 三种错法（不减最大值、用 `sum` 不用 `mean`、忘除 $N$）
  **全都不报错**：形状对、能跑、loss 照样降。其中第一种在正常数据上连数值都一样 —— 只有把
  logits 推到 900 那种极端才现形（§2.2 实测给你看）。
* **它给出你以后每天要看的那个数。** $\ln V$：随机初始化时 loss 的理论值。阶段 5 开训第一件事
  就是核对它（§3.5）。

后半卷是反着来的：§4 用两个**已知答案**的层去考 `gradcheck`。放在最后是有原因的 ——
你得先被它判过十几条，才有资格怀疑它。

```flow
§2 先把 softmax 的两件数值事故看清楚：减最大值、log 的位置（工具）
   ↓
§3 写 CrossEntropy：融合写法、五步 forward、一行 backward（手感）
   ↓
§4 反过来考 gradcheck：一个已知对的层必须绿、一个故意漏系数的必须红（测谎仪自检）
   ↓
§5 五条判据逐条对账 → 阶段 3 结束，带着这三卷去阶段 4
```

顺序上没有"先易后难"这一说了：§3 是唯一的实现任务，§2 是它的前置阅读，§4 独立于 §3
（就算 §3 还红着，§4 那两条也能跑 —— 它俩不碰你的代码）。

---

## 2. 先想清楚两件数值事故

### 2.1 logits、概率、loss：三个词的关系

$$p_v=\frac{\exp z_v}{\sum_{v'}\exp z_{v'}},\qquad L=-\log p_{y}$$

模型对词表里每个词打一个分 $z_v$（叫 **logits**，取值任意、可正可负、没有归一化），
softmax 把**一行** $V$ 个分数变成和为 1 的概率，CrossEntropy 只取**真值那一项**的 $-\log p$。

为什么罚函数是 $-\log$ 而不是 $1-p$：看四个数就懂。

| 真值词的概率 $p_y$ | $-\log p_y$ | 直觉 |
|---|---|---|
| $0.99$ | $0.010$ | 基本对了 |
| $0.50$ | $0.693$ | 一半把握 |
| $0.10$ | $2.303$ | 几乎没猜到 |
| $0.01$ | $4.605$ | **自信地错**：罚得极重 |
| $1/V$ | $\ln V$ | 什么都没学（均匀瞎猜）—— §3.5 的基准线 |

$\log$ 在 $p\to 0$ 处竖直上去，所以"非常确定地答错"会被重罚；$1-p$ 最多罚到 1，
一个模型可以靠"全部输出 0.5"把 $1-p$ 稳住在 0.5，却没法躲开 $-\log$。

形状：`logits` 是 $(B,T,V)$，`targets` 是 $(B,T)$ 的整数（每个位置的正确词编号），
输出是**标量**。$B\cdot T$ 个位置每个都是一次"预测下一个词"的小测，最后取平均。

### 2.2 事故一：`exp` 会溢出，而减最大值不改变答案

`float64` 在 $e^{709}\approx 8.2\cdot10^{307}$ 处到顶，$e^{710}$ 就是 `inf`。logits 是
`Linear` 的自由输出，训练早期是几、后期可能几十，采样阶段（阶段 7）你还会乘 $1/\text{温度}$
把它放大 —— 所以这不是"理论风险"。

补救只有一行：每行先减去这一行的最大值 $m$。它合法，因为 softmax **对整体平移不变**：

$$\frac{\exp(z_v+c)}{\sum_{v'}\exp(z_{v'}+c)}=\frac{e^{c}\exp z_v}{e^{c}\sum_{v'}\exp z_{v'}}=\frac{\exp z_v}{\sum_{v'}\exp z_{v'}}$$

减完 $m$ 之后，那一行里最大的一项恰好是 $e^{0}=1$，其余都在 $(0,1]$，所以

$$Z=\sum_v e^{\,z_v-m}\ \in[1,\ V]\qquad\Longrightarrow\qquad \log Z\in[0,\ \ln V]$$

**永远安全**，而且 $\log Z$ 之后不会下溢。

实测两个写法（$z=[2,\,1,\,0.5]$，真值 $y=0$）：

| 写法 | 概率 | loss |
|---|---|---|
| 不减最大值（朴素） | `[0.628532, 0.231224, 0.140244]` | `0.4643687841079447` |
| 减最大值（稳定） | `[0.628532, 0.231224, 0.140244]` | `0.4643687841079449` |

差别 $2\times10^{-16}$ —— 一个 $O(1)$ 的相对数。也就是说**在正常数据上你看不出任何区别**，
这正是它危险的地方：不减最大值的代码能一路绿灯跑到阶段 6。把 $z$ 推到 $[900,900,901]$：

| 写法 | 结果 | 症状 |
|---|---|---|
| `z - log(sum(exp(z)))` | `[-inf, -inf, -inf]` | loss $=\infty$ |
| `log(exp(z)/sum(exp(z)))` | `p = [nan, nan, nan]`（`inf/inf`） | loss $=\mathrm{nan}$ |
| `z - m - log(sum(exp(z-m)))` | `[-1.551445, -1.551445, -0.551445]` | loss $=0.551445$ ✓ |

`inf` 和 `nan` 是两个不同的坏法：第一种是 $900-\infty$，第二种是 $\infty/\infty$。
判据里 `logits=900 不溢出` 那条同时挡掉这两种。

### 2.3 事故二：`log` 要放在除之前，不是概率算完之后再取

拿到概率再 `np.log(p)` 是**另一条**数值事故：$p$ 可能下溢成**恰好 0**，于是 $\log 0=-\infty$。
`float64` 最小的正规数约 $2.2\times10^{-308}$，而 $e^{-800}=0.0$（$e^{-709}\approx1.2\times10^{-308}$
是最后一个还能表示的）。取 $z=[0,\,0,\,800]$、真值 $y=0$（模型极端自信地答错）：

```python
m = z.max(); e = np.exp(z - m); Z = e.sum(); p = e / Z
#   e = [0., 0., 1.]   Z = 1.0   p = [0., 0., 1.]
np.log(p)                       # [-inf, -inf, 0.]   → loss = inf     ← 错
z - m - np.log(Z)               # [-800, -800, 0]    → loss = 800.0   ← 对
```

同一个答案，一个能给、一个给不了。所以稳定写法是**一路用 $z$ 表示 $\log p$**：

$$\log p_v = z_v - m - \log Z$$

代码里 `log_probs` 从头到尾没有出现 `np.log(概率)`；`np.log` 只作用在 $Z$ 上，
而 §2.2 已经证明 $Z\in[1,V]$ 是安全的。

顺带一条：$\log$ 是单调的，所以"概率最大的词"和"$\log p$ 最大的词"是同一个 —— 阶段 7 贪心
采样时你可以直接在 `log_probs` 上取 `argmax`，不需要还原成概率。

两根滑块：拖 `base` 到 710 附近看 `exp` 怎么炸成 `inf`/`nan`（而稳定写法的答案一位没动），
拖 `gap` 到 800 看 `log(p)` 怎么下溢成 `-inf`（而 $z-m-\log Z$ 照样给出 $-800$）。

@@fig softshift@@

### 2.4 这一层的三件产出（回到 3b §2 那张槽位表）

| 槽位 | 有没有 | 内容 |
|---|---|---|
| A｜参数梯度 | **空** | 这层没有参数 —— 权重全在上游的 `Linear` 里 |
| B｜共用一维参数的梯度 | **空** | 同上 |
| C｜输入梯度（返回值） | 有，而且**是全部** | $\partial L/\partial z$，形状 $(B,T,V)$，和 `logits` 同形 |

两个额外的"空"：`targets` 是整数，不可导，所以它的梯度是**不存在**而不是零
（3a 约定 4：没有可导输入才返回 `None`；这里 $z$ 可导，所以返回值必须是数组）。
`backward` 里出现 `self.xxx.grad` 就是错的 —— 和 `GELU` 同一条判断（3b §2）。

---

## 3. TODO-11：`CrossEntropy` —— softmax 和 NLL 融合写

$$L=\frac1N\sum_{n=1}^{N}-\log\frac{\exp z_{n,y_n}}{\sum_v\exp z_{n,v}}
  =\frac1N\sum_{n=1}^{N}\Bigl[\log\!\sum_v e^{\,z_{n,v}-m_n}-\bigl(z_{n,y_n}-m_n\bigr)\Bigr],
  \qquad N=B\cdot T$$

**为什么要融合成一个层，而不是 `Softmax` 层 + `NLL` 层**：

1. 数值稳定只能融合才做得到 —— §2.3 的 `log_probs = z - m - log Z` 是关键那一步，
   分成两层的话中间那个"概率"张量已经把下溢事故请回来了。
2. 梯度干净：融合后 $\partial L/\partial z=p-\mathrm{onehot}$，一行。
   分开写要处理 softmax 的雅可比 $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$，
   那是个 $(V,V)$ 的张量对每个位置都要来一遍 —— 几乎一定写错。阶段 4 判据里另有
   `softmax_backward` 一项（TODO-14），那时你会正面写它一次，就知道现在为什么躲开。

### 3.1 forward 的五步

```
1) 保形状：m = max(logits, axis=-1, keepdims=True)      # 沿最后一轴，每行一个
2) 稳定化：e = exp(logits - m)；Z = sum(e, axis=-1, keepdims=True)
3) log 概率：log_probs = logits - m - log(Z)            # 全程不对概率取 log（§2.3）
4) 缓存：self._probs = e / Z；self._targets = targets   # 反向要用的两样，下划线开头
5) 取真值那一列、求平均 → float
```

第 1、2 步里 `axis=-1, keepdims=True` 是 3a §2.3 那两条规律的直接复用：沿**最后一轴**
（词表轴）归约，`keepdims` 让形状留成 $(B,T,1)$，这样 $(B,T,V)-(B,T,1)$ 广播得动。
写成 `axis=0` 会压掉 batch 轴，得到一个形状 $(T,V)$ 的"最大值"，数值错但**能跑**。

### 3.2 取真值那一列：把 $(B,T,V)$ 摊成 $(B\cdot T,V)$ 再 fancy 索引

第 5 步是这一层最容易绕晕的地方，先想清楚你要什么：从 $(B,T,V)$ 里挑出 $B\cdot T$ 个数 ——
第 $(b,t)$ 个位置的第 $y_{b,t}$ 列。三维数组上没有直接写法，最稳的做法是 3a §2.2 教的那招：
**把前两轴乘成一根**，最后一轴一步没动，于是问题降成二维。

```python
flat = log_probs.reshape(-1, V)            # (B·T, V) —— 合法：只把 B、T 乘起来
col  = targets.reshape(-1)                 # (B·T,)   —— 每个位置的正确词编号
nll  = -flat[np.arange(col.size), col]     # (B·T,)   —— 第 n 行取第 col[n] 列
L    = float(np.mean(nll))
```

`np.arange(col.size)` 和 `col` 两个索引数组配对，读作"第 0 行第 col[0] 列、第 1 行第 col[1] 列……"。
**别写成 `flat[col]`**：那会用真值当**行号**，形状都对，取的是完全另一个数 —— 属于最坏的那类 bug。

`backward` 里构造 one-hot 也是同一条规律，只是不摊平、直接三维配对：

```python
onehot = np.zeros_like(self._probs)                                  # (B,T,V)
onehot[np.arange(B)[:, None], np.arange(T)[None, :], self._targets] = 1.0
```

三个索引数组广播成 $(B,T)$，正好落在每个位置的正确词那一格。注意这里**不用** `np.add.at`
（和 3b §5.3 唯一的区别）：`ids` 可以重复，而 $(b,t,v)$ 三元组永远不重复 ——
一个位置只有一个正确词，所以赋值就是对的。

### 3.3 backward 的一行，和它为什么是这个形状

$$\frac{\partial L}{\partial z_{n,v}}=\frac{p_{n,v}-\mathbf 1[v=y_n]}{N},
  \qquad N=\texttt{targets.size}=B\cdot T$$

形状推理（3a §6.2 那套）：返回值要和 `logits` 同形 $(B,T,V)$；$p$ 同形、one-hot 同形，
$N$ 是标量 —— 唯一拼法就是 `(p - onehot) / N`。没有转置、没有求和，因为**没有参数**。

手算一遍最小的例子（$z=[2,\,1,\,0.5]$、$y=0$，1 个位置、$V=3$、$N=1$）：

$$p=\begin{bmatrix}0.628532&0.231224&0.140244\end{bmatrix},\qquad
dz=p-\mathbf 1[y{=}0]=\begin{bmatrix}-0.371468&0.231224&0.140244\end{bmatrix}$$

两个立刻能验证的性质，以后拿来当"我这层大概没错"的秒查：

* **每一行的梯度和为 0**：$\sum_v(p_v-\mathbf 1[v{=}y])=1-1=0$。
  整层 `(p - onehot)/N` 同样如此。它是 §2.1 那句"softmax 平移不变"的对偶 ——
  沿常数方向扰动 $z$ 不改变 $L$，所以梯度必然垂直于它。实测判据那个 $(2,5,9)$ 的例子，
  `np.abs(g.sum(-1)).max() = 2.26e-17`。
* **正确词的梯度是负的**（$p_y-1<0$，把它的分往上推），其余词都是正的（往下压）。
  如果哪天你看到 `dz` 全正，那一定是 one-hot 放错了地方。

### 3.4 为什么除以 $N$：`mean` 不是 `sum`

forward 用 `np.mean`，所以 backward 必须除 $N$ —— 这是同一件事的两面，不是两个约定。
漏掉它的后果不在这卷里显形（判据第 1 条会抓住，因为有限差分和解析值差 $N$ 倍），
但会在阶段 5 第一次训练时显形：**等效学习率被放大 $B\cdot T$ 倍**。

拿判据的数：$B=2,T=5\Rightarrow N=10$，漏除就是把梯度乘 10。真实配置 $B=16,T=512$
就是 8192 倍 —— 症状是"loss 第一步就飞成 nan"，而大多数人会去调 `lr`，
于是把一个 bug 变成一个永久的错误超参。

$N$ 取 `targets.size` 而不是 `logits.size`：后者是 $B\cdot T\cdot V$，多除了一个 $V$。
以后如果你给序列加了 padding 跳过（`ignore_index`），除数要跟着改成**真正参与平均的位置数**，
否则 loss 会被那些 0 稀释 —— 这是同一条约定，不是新规矩。

`backward(self, grad_out=1.0)` 里那个 `grad_out`：这一层输出标量、又永远在最顶端，
看起来用不上。仍然要乘，因为它让层可以串（判据之外的一件事：实测 `grad_out=2.0` 时
`backward` 的返回值恰好是 `grad_out=1.0` 的 2 倍）。写法上是 `* (float(grad_out) / N)`，
不要先除再乘以外任何形式的"顺手"。

### 3.5 $\ln V$：你会用一辈子的基准线

模型随机初始化时，`Linear` 输出的 logits 都是 $O(0.02)$ 的小数，softmax 出来接近均匀，
于是 $L\approx\ln V$。三条实测：

| 情形 | loss 实测 | $\ln V$ 理论 |
|---|---|---|
| `logits` 全 0，$V=8$ | `2.079442` | $\ln 8=2.079442$（判据第 3 条，容差 $10^{-12}$） |
| `logits` 全 900，$V=8$ | `2.079442` | 每行全相等 ⇒ 仍然均匀，且没 `inf`（判据第 2 条） |
| `logits ~ N(0, 0.02²)`，$V=65$ | `4.1804` | $\ln 65=4.1744$（真实初始化就该在这附近） |

记住三个数：$V=65$（本项目小词表）$\to 4.17$；$V=5000\to 8.52$；$V=8192\to 9.01$。
阶段 5/6 每次开训第一件事：**第一步 loss 是否落在 $\ln V$ 附近**。差得远说明有 bug
（形状、除数、one-hot 位置），而不是"还没收敛"。掉不下去但也不炸，同样是这句话。

判据第 2 条和第 3 条测的是同一个数 $2.079442$，但它们是**两个不同的检查**：
第 2 条查"有没有 `inf`/`nan`"（数值稳定性），第 3 条查"归一化和 mean 有没有写错"（约定）。
不减最大值只坏第 2 条；写成 `sum` 只坏第 3 条。

### 3.6 写完就跑

```
python webapp/checks.py TODO-11       # 只测这一层，和网站按钮同一段代码
python tests/test_layers_c.py         # 整卷 5 条（前提：§3 已写）
```

网站里：阶段 3c → 右侧「单项快测」→ `TODO-11` 旁边点「测」。
它会先打印 `输入 logits=(2, 5, 9)，targets=(2, 5) → 输出 loss（标量）+ 梯度形状 (2, 5, 9)`，
再给三件事：gradcheck 误差、均匀 logits 的 loss、`logits=900` 的结果。
期望 `PASS`，gradcheck 误差在 $10^{-7}$ 量级（答案版实测 $1.73\mathrm{e}{-}07$）。

`(p − onehot) / N` 这一行里，`/N` 是最容易被忘的 —— 忘了 loss 照样降，只是梯度大了 N 倍。
下面给了几行具体的 logits：**点某一行的任意一格**，就把那一行的 target 换成它，算式框逐根写出
「这一根是 (p−1)/N、其余各根是 p/N」；拖 N 会真的增删行，看整块梯度按 1/N 缩，同时两条常驻读数
（每行之和、整块之和）都该是 0。把「logits 尺度」推到 800，朴素写法的 `exp → log` 当场给出
nan / Infinity，而融合写法的反向一位没动。

@@fig cebwd@@

<details>
<summary>参考答案 · TODO-11 <code>CrossEntropy</code>（判据没过之前别点开）</summary>

```python
    def forward(self, logits, targets):
        targets = np.asarray(targets)
        m = np.max(logits, axis=-1, keepdims=True)     # §2.2：减最大值，exp 不再溢出
        e = np.exp(logits - m)
        Z = np.sum(e, axis=-1, keepdims=True)          # Z ∈ [1, V]，log 它永远安全
        log_probs = logits - m - np.log(Z)             # §2.3：全程不对概率取 log
        self._probs = e / Z
        self._targets = targets
        nll = -log_probs.reshape(-1, logits.shape[-1])[
            np.arange(targets.size), targets.reshape(-1)]   # §3.2：摊平成二维再取列
        self._count = targets.size                     # N = B·T，不是 B·T·V
        return float(np.mean(nll))

    def backward(self, grad_out=1.0):
        B, T, V = self._probs.shape
        onehot = np.zeros_like(self._probs)
        onehot[np.arange(B)[:, None], np.arange(T)[None, :], self._targets] = 1.0
        return (self._probs - onehot) * (float(grad_out) / self._count)
```

三处细节：`log_probs` 用 `logits - m - log Z` 而不是 `np.log(self._probs)`（后者在 $p$ 下溢成
0 时给 `-inf`，§2.3）；取真值那一列用的是**摊平后的二维 fancy 索引**，别写成 `flat[targets]`
（§3.2）；`backward` 里没有一行 `.grad` —— 这层没有参数（§2.4）。

`_count` 存进缓存而不是在 `backward` 里重算 `targets.size`，是因为 `backward` 的返回值
必须和**那一次 forward** 的除数配套（3a §4.6：一次 forward 配一次 backward）。

</details>

---

## 4. 测谎仪自检：gradcheck 自己可信吗

到这里你已经让 `gradcheck` 判了 13 条。它说"通过"你就写下一层，说"最大相对误差 $3\mathrm{e}{-}2$"
你就回去改 —— 但你从没检验过**它**。这一节补上，用的方法就是拿已知答案的题考老师。

### 4.1 输出是标量时它怎么查

3a §5 说过：`gradcheck` 先把"输出 → 标量目标"固定下来。数组输出要随机配一个余向量
（cot），$\ f=\sum\mathrm{out}\odot\mathrm{cot}$；**输出本来就是标量时直接用它**，
也就是 $\mathrm{cot}\equiv 1$。CrossEntropy 正好落在这条分支上，所以判据第 1 条照样能测 ——
它查的是 $\partial L/\partial z$ 和有限差分对不对。

抽一个元素验一下（判据那组 $(B,T,V)=(2,5,9)$ 的 logits，取 $z_{1,2,3}$，中心差分步长 $10^{-6}$）：

```
数值：(L(z+eps) - L(z-eps)) / (2·eps) = -8.3163835596e-02
解析：backward 返回的 grad[1,2,3]       = -8.3163835731e-02
```

相对差 $1.6\times10^{-9}$，远小于阈值 $10^{-4}$。这就是判据第 1 条在做的事，只不过它对
**每个张量抽查最多 24 个元素**（3a §5）。

### 4.2 两个已知答案的层

`tests/test_layers_c.py` 里那两个 `_SquareOK` / `_SquareWrong` 不是给你写的，是判据自带的：

$$y=x^{2}\quad(x\ \text{是}\ (2,3,4)\ \text{共 24 个元素}),\qquad \frac{\partial y}{\partial x}=2x$$

| 层 | `backward` | gradcheck 结论 | 最大相对误差 |
|---|---|---|---|
| `_SquareOK` | `g * (2.0 * self._x)` | 说 **OK** | $1.52\mathrm{e}{-}09$ |
| `_SquareWrong` | `g * self._x`（漏了系数 2） | 必须**报错** | $5.00\mathrm{e}{-}01$ |

第一条证明它认得一个正确的导数，第二条证明它抓得住一个错的。两条都绿，
才可以说"以后它给我的红是可信的红"。判据特意写成 `rep.check(..., not ok_wrong, ...)`
—— 它检查的是**这条必须红**，工具如果哪天放行了错误代码，这一卷会立刻告诉你。

### 4.3 为什么"漏掉系数 2"恰好给 $5.00\mathrm{e}{-}01$

$$\text{相对误差}=\frac{|x-2x|}{|2x|}=\frac12$$

这不是巧合，也不是"误差大概一半"这种含糊话：**少一个常数因子，误差就等于那个因子的相对偏差**。
所以拿到 $0.5$ 就该去数系数，而不是去怀疑随机数种子。同一条规律解释你前面见过的数字：

* $10^{0}$ 量级（判据那组输入实测 $1.5$）：漏掉整**项**，不是漏系数 —— $r\,u$ 少写 $-\,r\hat x\,s$
  （3b §4.2），两项量级相当，减掉一项就错掉一半以上。
* $10^{0}\sim10^{1}$ 甚至更大：链式那一步整个走错，或者缓存拿错了对象（3b §3.4）。
* $10^{-9}\sim10^{-7}$：正常。$10^{-8}$ 附近也是正常的，那是有限差分的噪声底。

### 4.4 拿到一句"最大相对误差 $3\mathrm{e}{-}2$"，先跑这两条

正确的怀疑顺序（阶段 4 起真的会用）：

1. **先怀疑自己的代码**，除非这一卷最后两条刚红过。
2. 如果最后两条也红（`SquareOK` 不 OK，或 `SquareWrong` 竟然通过），那才是工具/环境的问题 ——
   比如有人把 `Parameter` 的 `float64` 改成了 `float32`：有限差分噪声底会从 $10^{-9}$
   抬到 $10^{-3}$ 量级，全线判据一起变红（3a §4.2 那张表的第二行就是这条规矩）。
3. 只有个别层红、且明细点名某个张量 → 按张量名定位（MLP 那种组合层的标准打法，3b §6.3）。
4. 报 `nan`/`inf` → 数值问题，不是梯度问题：先看有没有减最大值（§2.2）、有没有 $\epsilon$（3b §4.4）。

还有一条来自 3a §5 的提醒在这一卷特别相关：**gradcheck 在一个元素都没查的时候会判失败**，
不是"通过"。整数输入会被自动跳过，所以 `Embedding` 只查参数梯度（3b §5）；如果你哪天给
`CrossEntropy` 传了浮点的 `targets`，它会被当成可导输入去扰动，得到的红和你写的代码无关。

---

## 5. 这一卷的验收判据

```checks
judge: tests/test_layers_c.py
```

红了的去向：第 1 条 → §3.2、§3.4；第 2、3 条 → §2.2；最后两条是测谎仪自检（§4），
**觉得 `gradcheck` 有 bug 时先跑它们**。

阶段 3 三卷全绿之后，你手里有的不只是这七个层，而是一套能自证的验证链：
8 条 + 6 条 + 5 条 = 19 条硬判据，加上一条反向的"测谎仪自检"。

---

## 6. 常见失败模式对照表

| 症状 | 大概率原因 | 去哪一节 |
|---|---|---|
| `logits=900` 那条给 `inf` | 用了 `z - log(sum(exp z))`，没减最大值（$900-\infty=-\infty$） | §2.2 |
| `logits=900` 那条给 `nan` | 先算概率再取 `log`，`inf/inf` | §2.2 |
| loss 出现 `inf` 但 logits 只有几百 | `np.log(probs)` 在下溢成 0 的那格给出 `-inf` | §2.3 |
| 判据第 1 条误差正好是 $10^{0}$ 量级、明细点名 `logits` | one-hot 的索引数组广播错了（少了 `[:, None]`/`[None, :]`） | §3.2 |
| 判据第 1 条误差恰好 $0.5$ 或 $1/N$ 那种"整倍数" | 漏除 $N$，或 `np.mean` 写成了别的归一化 | §3.4、§4.3 |
| `IndexError: index 8 is out of bounds for axis 1 with size 9` 之类 | `flat[targets]`：把词编号当行号用了 | §3.2 |
| `IndexError` 且数字比 $V$ 大得多 | `log_probs[targets]` 直接在三维数组上按一维整数索引 | §3.2 |
| 均匀 logits 的 loss $=8\times\ln V$ 那种大 $BT$ 倍 | 用了 `sum` 不是 `mean` | §3.4 |
| `TypeError: float() argument must be ...` / 输出是 0 维数组 | 忘了 `float(np.mean(nll))`，`forward` 返回的是 `ndarray` | §3.1 |
| `operands could not be broadcast together` | `axis=-1` 少了 `keepdims=True`，$(B,T)$ 减不动 $(B,T,V)$ | §3.1 |
| 均匀 logits 的 loss 大了整整 $N$ 倍 | 用了 `sum` 不是 `mean`（本卷那条判据是 $(1,4,8)$，$N=4$，会给 $4\ln V$） | §3.4 |
| 训练（阶段 5）第一步 loss 远大于 $\ln V$ | 归一化轴或除数错；先回这一卷，不要去调 `lr` | §3.5 |
| `gradcheck` 全绿但你还是错了 | 上面这些里有一半不改变 $\partial L/\partial z$ 与差分的**一致性**（比如 `sum` 和 `mean` 只差一个常数，而 `logits=900` 那类根本不走 gradcheck）—— 所以判据是 5 条不是 1 条 | §5 |

---

## 7. 自查题

不用查资料。每题先把自己的答案写进输入框再点"判一下"；想不出来按提示 1→2→3，
三级提示从最抽象到最具体；还不行就点"回讲义读这一节"。

```quiz
q: 为什么 `log_probs` 要写成 `logits - m - log(Z)`，而不是先算 `probs` 再 `np.log(probs)`？举一个具体的 $z$ 说明两者结果不同。
keys: 下溢/0/inf/无穷; 例/z=[0,0,800]/800
hint1: probs 里最小的那个数，有没有可能小到 float 存不下？
hint2: log(0) 在 numpy 里等于什么？
hint3: 概率下溢成恰好 0 时 log 给 -inf。例：z=[0,0,800]、真值 0：p=[0,0,1]，log(p[0]) = -inf（loss=inf）；而 z-m-logZ 给 -800（loss=800，正确）。
link: sec-2
ans: 概率可能下溢成恰好 0，$\log 0=-\infty$。例：$z=[0,0,800]$、真值 $0$：`p=[0,0,1]`，`log(p)` 给 `-inf`（loss $=\infty$），而 `z-m-log Z` 给 `-800`（loss $=800$，正确）（§2）。
```

```quiz
q: 减最大值那一步会改变 loss 吗？在 $z=[2,1,0.5]$ 上实测差多少？为什么这个 bug 在正常训练里看不出来？
keys: 不变/平移不变/相消; 2e-16/1e-16/不溢出/等价
hint1: 分子分母同时乘 $e^{c}$ 会怎样？
hint2: 差是 0 吗？还是机器误差？
hint3: 数学上不变（平移不变，$e^c$ 相消）；实测差约 2e-16。正常数据上 exp 不溢出，两种写法数值等价，只有把 logits 推到几百才现形。
link: sec-2
ans: 不变 —— softmax 对整体平移不变（分子分母的 $e^{c}$ 相消）。实测差约 $2\times10^{-16}$。正常数据上 `exp` 不溢出，两种写法数值等价，所以只有把 logits 推到几百才现形（§2）。
```

```quiz
q: $N$ 是什么？写成 `logits.size` 会差多少倍？
keys: 位置数/B·T/targets.size; V/词表
hint1: forward 返回的是"平均"负对数似然 —— 对谁平均？
hint2: logits 比 targets 多几根轴？
hint3: N = B·T = 位置数 = targets.size。logits.size 多一个因子 V，会把梯度缩小 V 倍。
link: sec-3
ans: $N=B\cdot T=$ 位置数 $=$ `targets.size`。`logits.size` 多一个因子 $V$，会把梯度缩小 $V$ 倍；反过来不除 $N$ 等于把 lr 放大 $N$ 倍（§3）。
```

```quiz
q: `CrossEntropy.backward` 为什么一行 `.grad` 都不写？这层和 `GELU` 在这一点上一样吗？
keys: 没有参数/无参数; 一样/相同
hint1: 这层有自己的 W 或 b 吗？
hint2: 没有参数的层，三件产出里哪两件是空的？
hint3: 这层没有参数，权重全在上游 Linear 里；和 GELU 一样，dW/db 两个槽位是空的。
link: sec-2
ans: 这层没有参数，权重全在上游 `Linear` 里；和 `GELU` 一样，dW、db 两个槽位是空的（§2）。
```

```quiz
q: 为什么 `(p - onehot)/N` 每一行的和恰好是 0？这个性质和 §2 的哪句话互为对偶？
keys: 1/和为1/相减为0; 平移不变
hint1: 把一行拆开：p 那一行的和是多少？one-hot 那一行的和是多少？
hint2: 两个 1 相减。
hint3: Σp=1、Σonehot=1，差为 0。对偶的是"softmax 平移不变"：沿常数方向扰动 z 不改变 loss，梯度必然垂直于常数方向。
link: sec-3
ans: $\sum_v p_v=1$、$\sum_v \mathbf 1[v=y]=1$，所以差为 0。对偶的是"softmax 平移不变"：沿常数方向扰动 $z$ 不改变 $L$，梯度必然垂直于常数方向（§3）。
```

```quiz
q: 阶段 5 开训，第一步 loss 应该在哪个数附近？$V=65$ 时它是多少？
accept: 4.17/4.18/ln65/ln(65)/4.174
hint1: 初始化时 logits 接近什么？
hint2: logits 全 0 时 loss 是多少？（s03c 试验台可以直接验）
hint3: ln V 附近；V=65 时 ln 65 ≈ 4.17（实测 N(0, 0.02²) 初始化给 4.18）。
link: sec-3
ans: $\ln V$ 附近；$V=65$ 时 $\ln 65=4.17$（实测 $N(0,0.02^2)$ 初始化给 4.18）（§3）。
```

```quiz
q: 判据第 4、5 条考的不是你的代码，是什么？第 5 条为什么期望 $0.5$ 这个量级？
keys: gradcheck/测谎仪/工具; 系数2/漏掉2/一半/1/2
hint1: 那两条故意往一个已知答案的层里注入错误。
hint2: 漏掉常数因子 2 时，相对误差是多少？
hint3: 考 gradcheck 自己。漏掉系数 2 的相对误差恰为 |x-2x|/|2x| = 1/2 —— 少一个常数因子，误差就等于那个因子的相对偏差。
link: sec-4
ans: 考 `gradcheck` 自己。第 5 条期望 $0.5$ 是因为漏掉系数 2 的相对误差恰为 $|x-2x|/|2x|=1/2$ —— 少一个常数因子，误差就等于那个因子的相对偏差（§4）。
```

```quiz
q: 有人把 `Parameter` 的 dtype 从 `float64` 改成 `float32`，你预计判据怎么变？该怀疑哪一层？
keys: 全线/一起红/全部; 工具/环境/不是代码
hint1: 有限差分的噪声底和 dtype 有关吗？
hint2: 是一条红还是全红？
hint3: 噪声底从 1e-9 抬到 1e-3 量级，全线判据一起变红。先跑 §4 那两条：一起红说明是工具/环境，个别红才是你的代码。
link: sec-4
ans: 有限差分噪声底从 $10^{-9}$ 抬到 $10^{-3}$ 量级，**全线**判据一起变红，而不是某一条。这时候先跑 §4 那两条：它们一起红说明是工具/环境，个别红才是你的代码（§4）。
```

---

## 8. 下一步

`python tests/test_layers_c.py` 5 条全绿 → 阶段 3 结束。进
[阶段 4 · 因果自注意力](s04.html)之前，先确认这三卷都是绿的：

```
python tests/test_layers_a.py     # 8 条
python tests/test_layers_b.py     # 6 条
python tests/test_layers_c.py     # 5 条
```

阶段 4 是本项目第一个"形状不止一种活法"的层：$(B,T,C)$ 要变成 $(B,h,T,d_h)$，
$QK^{\top}$ 收缩的是**特征轴**而不是样本轴（3a §2.5 那张速查表第一次要按轴名而不是位置来读），
外加一个 softmax 层 —— 它的雅可比就是 §3 里我们融合掉的那件事，判据会单独考它一次
（`TODO-14`）。你在这里练的"用形状推、别背公式"，那里是正式上场。

一件值得现在做的事：把阶段 3 三卷里你**红过**的判据名字、当时的误差量级，写在一张纸上。
阶段 4 你会遇到同两类错（漏项、缓存错对象），那张纸比讲义快。
