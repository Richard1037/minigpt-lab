# 阶段 7 · 采样与生成质量：同一台模型的四种性格

预计：2~4 小时。
验收命令：`python stages/s07_generate.py`（30 条硬判据 + 1 条软提示，约 2 秒）
要写的地方：`src/minigpt/sampler.py`（TODO-28、TODO-29、TODO-30、TODO-31）。
网站：选「阶段 7 · 采样与生成质量」，提交 `minigpt/sampler.py`。

---

## 1. 这一步在干什么

训练结束以后，模型对每个位置只交出一样东西：一个长度等于词表的分数向量（logits）。**说出口的是哪一个**，完全由这个文件里的十几行决定。

```flow
起点：最后一个位置的 logits `z (V,)`
↓
**惩罚** —— `apply_penalty(z, seen, penalty)`，`TODO-28`
↓
**贪心岔口** —— `temperature <= 0` 时直接 `argmax`，不经过任何随机
↓
**温度** —— `z / T`
↓
**top-k** —— `keep_top_k(·, k)`，`TODO-29`
↓
**softmax** → 概率
↓
**top-p** —— `keep_top_p(probs, p)`，`TODO-30`
↓
**抽样** —— `rng.choice(V, p=…)` → 一个 token id，`TODO-31`
```

这四个开关各自改的是**不同的东西**：温度改**分布形状**、top-k 改**候选集**、top-p 改**概率质量**、惩罚改**分数本身**。所以它们不能互相替代，而顺序换了结论就换（§2.6）。

因为采样**不参与训练**，它是整个手册里唯一一个"改坏了也不会让 loss 变差"的部分。这带来一个方法论问题：loss 这条万能拐杖失效了，判据必须直接盯着**分布**本身。于是这一卷的设计分两段：

* **前 22 条**在一份固定的合成 logits（12 个分数，`V=12`）上做**大样本统计检验**：不训模型，跑得快，结论干净。每条都重复抽 1200~6000 次，和解析算出来的理论值对账（§2.7 讲那个 `0.14` 是怎么来的）。
* **后 8 条**才上真模型：36 行自造小诗训 300 步（约 2 秒），看它贪心能不能接上下句、升温会不会换说法、加惩罚会不会少复读。

判据里的"候选集"一律指**词表下标的集合**（`{2, 5, 7}` 这种），不是分数。

---

## 2. 必要理论

### 2.1 从 logits 到一个分布：为什么 argmax 不是唯一答案

模型交出的是"每个候选的分数"，不是概率。要抽签就得先变成概率，第三件老工具（阶段 3 的 CrossEntropy、阶段 4 的 `_softmax_rows`，这是第三次出现）：

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$$

贪心（$\arg\max$）是这个分布的一个特例：它只取峰，丢掉全部"第二名有多接近"的信息。后果是**复读**：既然每次都取峰，同一个前缀永远给同一个下一词，模型一旦进入某个循环就出不来 —— 阶段 6 判据最后那条 info 就是活样本（贪心 60 个字符：`the small town sits between two hills an and the the the the anoond the the the...`）。采样把"次优候选"重新变成可能，代价是可能抽到差的。这一章就是学怎么花这笔钱。

### 2.2 温度：分布的陡峭程度

$$p_i(T)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

$T$ 是分数轴的**缩放**，不是加法，所以它改的是形状而不是名次：

* $T\to 0^+$：峰被无限拉高，分布收敛到 one-hot（$\arg\max$）。判据 `温度极小但不为 0 时仍在赌最高分` 用 $T=10^{-3}$ 测的就是这个：`exp` 里其余候选**下溢成 0**，抽出来必然是峰。
* $T\to\infty$：所有分数被压成同一个数，分布趋于均匀，熵趋于上限 $\ln V$。
* 中间：$T$ 越大，第二三名越有机会。**熵单调上升**（判据第 1 条）：$H(T)=-\sum_i p_i\log p_i$，那 12 个分数在 $T=0.2/0.5/1/2/5$ 下的熵是 `0.025 / 0.425 / 1.207 / 1.957 / 2.371`，$\ln 12=2.485$ 是天花板。

两个必须做对的地方：

1. **`temperature <= 0` 要单独走贪心**。$z/0$ 给 `inf`/`nan`，而 $\texttt{softmax}(\text{nan})$ 是 nan，`rng.choice` 收到 nan 概率才炸 —— 报错位置离真正的错误隔了两层。而 $T=0$ 的定义本来就是"取最大分数"，不需要经过任何随机：判据 `temperature=0 就是贪心（与 seed 无关）` 用三个不同种子验它。
2. **温度必须在 softmax 之前**。如果把温度放到概率上（$p_i/T$ 再归一化），分子分母同时除以 $T$，结果**一个字都不变** —— 这是本章最隐蔽的一个 bug：代码看着像"实现了温度"，症状是"T 开得很大，输出仍然很单调"。

温度的实际取值要跟模型自信度匹配。判据后段那台小模型训得很熟（末段 loss 0.145，$\ln V=3.091$），$T=1.0$ 时 24 个种子仍然几乎总是同一个词 —— 所以多样性判据把温度开到 `T=2.0` 并配 `top_k=8`。**这不是采样坏了，这是真实结论：模型越自信，同样的温度越驯服。**

### 2.3 top-k：只保留最高的 k 个候选

$$\mathcal{K}=\text{最大的 }k\text{ 个下标},\qquad z_i\leftarrow \text{MASK}\ (i\notin\mathcal{K})$$

动机：$T$ 放大以后**所有**候选都参与抽签，长尾里的垃圾（本项目词表 22，但真实模型上万）会涌进来。top-k 先把尾巴剪掉，再在剩下的 k 个里按温度抽签。

三个要点：

1. **切在分数上、掩成 MASK，不删元素**（§5 详说）：删了就得到"一个不知道对应哪些 id 的分布"，最后那一步 `rng.choice(V, p=...)` 直接对不上号。
2. **k 与温度可交换**。温度是正的等比缩放，名次不变，所以"k 留下的名单"与 $T$ 无关 —— 判据 `k 这一刀在温度之前生效：升温不引入新候选` 用 `T=4, k=2` 验的正是这个不变性（$T$ 只改这 2 个之间的比例）。这和 §2.2 的第 2 点不矛盾：**可交换的是 k 与温度，不可交换的是温度与 softmax。**
3. **k 是个开关**：`k<=0` 或 `k>=V` 时原样返回。判据 `k=0 或 k>=V 时不剪枝` 查它；把它写成"无论如何都切一刀"，$k=0$ 时会返回一行全 MASK → softmax 给出均匀分布 → 等于把整个词表拉平，非常像"温度开到了无穷大"。

### 2.4 top-p（nucleus）：覆盖多少概率质量

top-k 的老毛病是 $k$ 与"当前这一步有多确定"无关：模型极度确定时 $k=40$ 会硬塞进 39 个垃圾，模型很犹豫时 $k=40$ 又太窄。nucleus 采样改成**把概率从高到低累加，刚攒够 $p$ 就停**：

$$\mathcal{N}=\{\,i_1,\dots,i_n\ \text{(按概率降序)}:\ \underbrace{p_{i_1}+\cdots+p_{i_{n-1}}}_{<\,p\ \text{（轮到 }i_n\text{ 之前还没攒够）}}\,\},\qquad \sum_{i\in\mathcal{N}}p_i\ \ge\ p$$

于是候选数是**自适应**的：分布尖时只有一两个，分布平时几十个。三条判据就是在钉这三件事：核的覆盖 `≥ p`、核是概率降序的**前缀**、归一化后和为 1。

为什么定义里要说"轮到这一格**之前**攒了多少"（§6 的 `before = cum - q[order]`）：判据 `top_p=0.5 的核覆盖原分布 ≥ 0.5 的质量` 里，那 12 个分数的第一名概率是 0.620 > 0.5，所以核就是 `{7}` —— 覆盖 0.620。如果你写成"到这一格**为止**的累计 ≤ p"，跨过 $p$ 的那一格会被踢掉，留下的质量总是**刚不到** $p$（$p=0.8$ 时可能只有 0.62）。所有人都说"nucleus 覆盖 80% 概率"，指的从来是 ≥ 80%。

还有一个角落：**极端 one-hot**。$T=10^{-4}$ 把分布压成第一名 = 1.000000，此时 $p=0.5$ 的核应该只剩它一个，而不是"一格都选不上 → 全 0 → 归一化 0/0 → nan"。判据 `极端 one-hot 时不冒出 nan` 专查这个兜底。

### 2.5 重复惩罚：GPT-2 的两个不对称

$$z_i'=\begin{cases}z_i/\rho & z_i>0\\[3pt] z_i\cdot\rho & z_i\le 0\end{cases}\qquad (i\in\text{seen},\ \rho\ge 1)$$

两条分支只做一件事：**把它往更小的方向推**。如果统一写成"除以 $\rho$"，负分会被**抬向 0** —— 也就是更接近被选中，惩罚变成了奖励。判据 `正分除以惩罚、负分乘以惩罚` 用精确到 $10^{-12}$ 的两个等式验它（`id 7` 的 4.48 → 1.792，`id 4` 的 −4.80 → −12.000）。

它是 **presence** 而不是 **frequency** 惩罚：`seen` 是"出现过"的集合，出现三次和出现一次一样惨。这就是实现里那句 `np.unique` 的意义 —— 不去重的话同一个 id 会被连除两次，强度悄悄变成 $\rho^2$。（想要 frequency 版：把 `np.unique` 换成 `np.bincount(seen, minlength=V)`，按次数开幂。阶段 8 的可选作业。）

三个判据角度：`惩罚不修改调用者的 logits`（§4 的头号坑）、`被罚的那个词经验频率明显下降`（同一颗种子、同一批随机数，`penalty=3` 时 `0.627 → 0.080`）、`penalty=1 等于不惩罚`、`越界 id 被安全忽略`。

### 2.6 顺序为什么是这一个

```flow
**惩罚**（在分数上）→ **贪心岔口**（`T<=0` 就到此为止）
↓
**温度**（还是分数）→ **top-k**（还是分数）
↓
**softmax**
↓
**top-p**（在概率上）→ **`rng.choice`**
```

四个滑块就是上面这根管子：拖 ρ 看词1/词4 的分数怎么被"往小推"（注意正负两条分支），
拖 T 到 0 看贪心岔口怎么让 ρ 和 k 全部失效，按那个"把 top-p 挪到 softmax 之前"的按钮
看同一批参数留下完全不同的词。

@@fig sampling@@

逐条说"换了会怎样"：

* **惩罚必须在贪心岔口之前**。$T=0$ 时它也要生效，否则复读问题在最容易复现的模式下最严重（贪心一定选回同一个词）。判据 `top_k=1 与贪心等价` 和单项 `TODO-31` 里那条 `seen=[best], penalty=8, temperature=0 → 必须换第二个` 两头都在钉它。
* **温度必须在 softmax 之前**（§2.2）：放之后就等于没放。
* **top-k 在 softmax 之前**是定义选择（"最高的 k 个**分数**"），而且掩码写法让后面只跑一次 softmax；**top-p 必须在 softmax 之后**：它的定义在概率质量上，同一刀切在分数上会因为分布陡缓不同而切出完全不同的东西（§2.4）。
* **top-p 之后不要再自己归一化**：`keep_top_p` 交出的已经是一个和为 1 的分布，`rng.choice` 要的也只是"非负 + 和为 1"。多除一次和本身不会炸，但它意味着"归一化"这件事有两个地方各写一半 —— 一旦其中一处漏了（比如你把 `keep_top_p` 写成只置 0、不除），症状就是 `ValueError: Probabilities do not sum to 1`，而你会先去怀疑别的地方。骨架注释里点名的正是这条报错。

### 2.7 怎么证明"真的在按分布抽"

看文本是看不出来的 —— 采样随机，任何一段输出都可能来自任何一套错误的实现。唯一的证据是**大样本频率对得上解析概率**：

$$\sum_{i}\lvert\hat p_i-p_i\rvert\ \le\ 0.14,\qquad \hat p_i=\frac{\text{id }i\text{ 出现次数}}{n}$$

这个 L1 距离（= 2 × 总变差距离）的随机波动量级是 $\sqrt{V/n}$：判据前段 $V=12$、$n=6000$ → 约 0.045，阈值 `0.14` 差不多是 3σ（实测 0.014）。$n=1200$ 那几条同理。

两个设计上的教训，比数字本身值钱：

1. **阈值不能按"实测值 + 一点点"定**。种子虽然固定，但 numpy 的 `default_rng` 序列、`rng.choice` 的算法都属于实现细节，允许你的正确实现和答案实现给出不同的抽样。阈值必须显著大于统计噪声，否则它会变成一次抛硬币。
2. **每条"分布类"判据都配一条"布尔类"判据**。候选集恰好是前 k 名（`support` 是否等于那个集合）这种检验不受抽样噪声影响，它抓的是结构性错误；L1 抓的是加权错误。两个一起才完整。

---

## 3. 已给你的代码：`_softmax`、`MASK`、`entropy`、`temperature_sweep`

配套单项快测 `python webapp/checks.py GIV-5`（现在就该是绿的：它测给你的部分；只有 `temperature_sweep` 那半会提到 TODO-31）。

### 3.1 `_softmax`：减最大值那一步不是风格，是必需

```python
m = np.max(logits); e = np.exp(logits - m); return e / np.sum(e)
```

$\exp(900)$ 在 float64 里是 `inf`（上限约 $1.8\times10^{308}$，即 $\exp(887)$），于是 `inf/inf = nan`，整条管线后面全是 nan。减最大值之后最大的指数项是 $e^0=1$，其余 $\le 1$，永不溢出，而且：

$$\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}=\frac{e^{z_i}e^{-m}}{e^{-m}\sum_j e^{z_j}}=p_i$$

**平移不变**是它的数学内容 —— 单项 `GIV-5` 用 `z` 和 `z-500` 两个输入验同一个结果，还用全负分数（`[-50,-40,0]`）验"减完之后有没有偷偷 clamp"。注意这一版是对**一维向量**做的（`np.max` 不带轴），因为采样只看一个位置；阶段 4 的 `_softmax_rows` 是它的按行版本。

### 3.2 `MASK = -1.0e18`：够大、但没有 `inf`

和 `attention.NEG_INF` 同一个理由。$\exp(-10^{18}-\text{有限值})$ 在浮点里**精确等于 0**，所以被掩掉的候选概率是 0、归一化后仍是 0，不会冒出 `nan`，也不需要任何清理步骤。为什么不用 `-inf`：`-inf - (-inf) = nan`，而 `_softmax` 里那个减最大值恰好会做这件事 —— 一整行都是 `-inf` 时（全被掩掉）立刻炸。用有限大负数时同一情形的表现是"每项都等于 max → `exp(0)=1` → 均匀分布"，不崩，但**结果荒谬**（词表里所有词等概率）—— 这正是 §5 那个并列坑的最终症状。

### 3.3 `entropy`：把"温度改形状"变成一个可比的数

$$H(p)=-\sum_i p_i\log p_i,\qquad H\le\ln V$$

实现里 `q = q[q > 0]` 那一行是必须的：$0\log 0$ 数学上定义为 0，但 `0 * np.log(0)` 给 `nan`（$\log 0=-\infty$）。三个已知值用来自查（判据也是这么查的）：均匀 8 项 → $\ln 8=2.079$，one-hot → $0$，二选一 → $\ln 2=0.693$。

### 3.4 `temperature_sweep`：为什么每档温度重新派一颗子 rng

```python
sub = np.random.default_rng(rng.integers(1 << 31))
```

它调 `sample_next`（TODO-31），所以在写完之前点它会打印"那一半待 TODO-31"，而 `GIV-5` 特意兜住这个 `NotImplementedError` 让其余四件事先能验（它自己那条"温度越高用词越宽"就是这么测的：各档温度下**出现过的不同 id 数**应当随温度不减）。这里的写法值得学：**从父 rng 派生**而不是复用同一颗 —— 各温度档的抽样序列互相独立（否则"温度高的那档用了温度低剩下的随机数"，两档不可比），而整张表在给定父种子时仍可复现（单项 `TODO-31` 里那条"同 seed 同结果"靠的就是同一个模式）。

---

## 4. TODO-28：`apply_penalty`

**契约**：`apply_penalty(logits, seen, penalty=1.0) -> np.ndarray`，返回**新数组**，形状一维、dtype float64。

```
out = np.array(logits, dtype=np.float64, copy=True).reshape(-1)     ← copy，不是 asarray
短路：seen 为空 / penalty == 1.0 → 原样交出
idx = np.unique(seen 转 int64)；筛掉 idx<0 与 idx>=V
pos = out[idx] > 0
out[idx[pos]] /= penalty        # 好的被压低
out[idx[~pos]] *= penalty       # 差的被推得更差
```

三个必须做对的细节，每个都有一条判据或单项检查盯着：

1. **返回新数组**。`np.asarray(x, dtype=np.float64)` 在 `x` 本来就是 float64 时交回的是**视图**，`reshape(-1)` 也还是视图，于是就地 `/=` 会把模型刚算出来的那一行 logits 改掉 —— 而 `generate` 传进来的正是 `logits[0, -1]`，它是整块 `(B,T,V)` 数组的一个切片：改它就等于改那块数组。判据两头都查：`np.array_equal(z, orig)`（入参逐比特没动）**且** `out is not z`（不是同一个对象）。
2. **`np.unique` 去重**（§2.5 的 presence 语义）。
3. **筛掉越界 id，包括负数**。越界是真实情况：`generate` 传进来的 `seen` 是整段序列的 id，换了 tokenizer 就会超出词表；不筛是 `IndexError`。但**负数不会 IndexError** —— numpy 的负索引合法，`-2` 会命中倒数第二个词，于是"惩罚刚生成的词"变成"惩罚词表里倒数第二个词"，一个都不报错的错误。判据 `越界 id 被安全忽略` 特意同时塞进 `V+5` 和 `-2` 两个。

`penalty == 1.0` 的短路除了省时间，更是语义：**默认值必须真的什么都不做**。判据 `penalty=1 等于不惩罚` 在 $T=0$ 那条路上查它（如果写成"照样除一遍"，浮点上仍然等值，倒是不会红 —— 但 `seen` 里有重复时你就得指望 `unique` 兜住）。

顺带一个边界：分界线是 `> 0` 还是 `>= 0` 无所谓，因为 $0/\rho=0$ 且 $0\cdot\rho=0$。单项检查 `TODO-28` 的输入 `[2.0, -1.0, 0.5, 0.0, 3.0]`（`seen=[0,1,3]`，$\rho=2$）期望 `[1.0, -2.0, 0.5, 0.0, 3.0]`，那个 0 号位就是让你确认自己没写反两支。

<details>
<summary>参考答案 · TODO-28 <code>apply_penalty</code>（判据没过之前别点开）</summary>

```python
    out = np.array(logits, dtype=np.float64, copy=True).reshape(-1)
    if not seen or penalty == 1.0:
        return out
    idx = np.unique(np.asarray(seen, dtype=np.int64))
    idx = idx[(idx >= 0) & (idx < out.size)]
    if idx.size == 0:
        return out
    pos = out[idx] > 0
    out[idx[pos]] /= penalty
    out[idx[~pos]] *= penalty
    return out
```

`np.array(..., copy=True)` 一行同时解决 dtype、摊平和"绝不返回视图"三件事 —— 比 `np.asarray(...).reshape(-1).copy()` 更省事，也少一步出错机会。
注意 `if not seen` 依赖 `seen` 是 list 或 None（判据和 `GPT.generate` 都传 list：`ids.reshape(-1).tolist()`）。如果你想让它也吃 ndarray，写 `if seen is None or len(seen) == 0`。
`out[idx[pos]] /= penalty` 是"布尔掩码索引 + 就地写"，这里改的是**自己的副本**，安全。

</details>

---

## 5. TODO-29：`keep_top_k`

**契约**：`keep_top_k(logits, k) -> np.ndarray`，形状和 dtype 与入参一致（float64），落选的格子写成 `MASK`。

```
z = np.asarray(logits, dtype=np.float64)
短路：not k / k <= 0 / k >= z.size → 返回 z.copy()
cutoff = np.partition(z, -k)[-k]          # 第 k 大的**值**
return np.where(z < cutoff, MASK, z)
```

三个坑，按危险程度排：

1. **比较用 `<`，不是 `<=`**。恰好等于 `cutoff` 的那一格要**保留**。写成 `<=` 时，如果最高分有两个并列，它们会**双双被掩掉** → 整行都是 MASK → `_softmax` 里每项都等于 max → `exp(0)=1` → **均匀分布**。它不崩、不报错、NaN 也没有，只是任何词都可能被抽出来（§3.2）。判据为此专门备了两份输入：`keep_top_k([1.0,1.0,0.0], 1)` 期望前两个都活着，以及单项检查里那份 `[0.1,5,2,5,-3,4]`（两个并列的 5）。附带结论：并列时留下的候选**可以多于 k 个** —— 这是正确行为，别为了"凑够 k 个"去加二次筛选。
2. **掩成 MASK 而不是删元素**（§2.3）。删了以后返回的数组长度不再是 $V$，`sample_next` 里那句 `rng.choice(probs.size, p=probs)` 会把**压缩后的位置**当成词表下标交出去 —— 判据 `top_k=k 的候选集恰好前 k 名` 当场就看到 `{0, 1}` 这种不该出现的集合，单项 `TODO-29` 的 `shape_ok` 也会红。更糟的是它不会崩：你会拿到"长度对不上但每个数都像概率"的东西，而这类形状错误在阶段 3/4 已经反复见过。
3. **`np.partition` 而不是 `argsort`**。前者是选择算法，平均 $O(V)$；后者 $O(V\log V)$。$V=22$ 时毫无差别，$V=8192$、每生成一个 token 调一次时差一个量级。更重要的是 `argsort` 之后还要把位置映射回去（谁原来是第几列），多两步就多两个出错机会。`np.partition(z, -k)` 把数组重排成"第 $-k$ 位是第 k 大的值，它右边都比它大"，所以 `[-k]` 就取到边界值本身（`[-k:]` 取到那 k 个，但我们只要边界）。

`k >= V` 时原样返回值得多想一秒：那时"最高的 k 个"就是全部，切一刀和切 0 刀等价；不短路的话 `np.partition(z, -k)` 在 `-k < -V` 时是越界索引。判据用 `keep_top_k(Z, V+3)` 查这条。

<details>
<summary>参考答案 · TODO-29 <code>keep_top_k</code></summary>

```python
    z = np.asarray(logits, dtype=np.float64)
    if not k or k <= 0 or k >= z.size:
        return z.copy()
    cutoff = np.partition(z, -int(k))[-int(k)]
    return np.where(z < cutoff, MASK, z)
```

`not k` 挡掉 `0` 和 `None`，`k <= 0` 挡掉负数（`k=-3` 时 `np.partition(z, -int(k))` 是合法的"第 4 小"分区，于是你留下 $V-3$ 个候选而不是 3 个 —— 静默的灾难，所以要挡）。
`int(k)`：判据可能传 numpy 整数或 `np.int64`，`-int(k)` 保证索引是 Python 整数。
短路那支为什么还要 `.copy()`：`np.asarray` 在 dtype 已对时给视图，返回视图等于把 §4 那个"别把入参交出去"的约定从后门破掉。`np.where` 那支本来就返回新数组，两支行为一致才好推理。

</details>

---

## 6. TODO-30：`keep_top_p`

**契约**：`keep_top_p(probs, p) -> np.ndarray`，输入**概率**、输出**概率**（和为 1，非核内格子为 0）。

```
q = np.array(probs, dtype=np.float64, copy=True).reshape(-1)
短路：p >= 1.0 或 q.size <= 1 → 返回 q / q.sum()
order  = np.argsort(q)[::-1]          # 概率**降序**的下标
cum    = np.cumsum(q[order])          # 到这一格**为止**的累计
before = cum - q[order]               # 轮到这一格**之前**的累计
take   = before < p                   # 还没攒够就继续收
take 全空时只留 order[0]              # p <= 0 的兜底
其余置 0，再除以新的和
```

四条判据对应四个点：

* **`before < p` 而不是 `cum <= p`**（§2.4）。手算一遍最快：`q = [0.5, 0.3, 0.15, 0.04, 0.01]`，`p = 0.8` → `cum = [0.5, 0.8, 0.95, ...]`、`before = [0, 0.5, 0.8, ...]` → `take = [T, T, F, ...]` → 核 = 前两名，覆盖 0.8，归一化 `[0.625, 0.375, 0, 0, 0]`（单项 `TODO-30` 期望的就是这两个数）。换成 `cum <= p` 在 `p=0.6` 时：`take=[T,F,...]` → 只留 0.5，**没覆盖到 0.6**。判据把 $p\in\{0.5,0.8,0.95\}$ 三档的覆盖都算了一遍：`0.620 / 0.823 / 0.963`，全都 `≥ p`。
* **核必须是概率降序的前缀**（判据里那句 `ok_prefix`）。这一条专抓"忘了 `[::-1]`"：`np.argsort` 给的是**升序**，直接 `cumsum` 就是从最小的概率开始收，攒到 0.8 时你留下的是长尾里的一堆小词 —— 覆盖对了、前缀错了，所以判据两条一起查。
* **兜底不能省**。`p <= 0` 时 `take` 全 False → 全 0 → `0/0` → nan，nan 一路传到 `rng.choice` 才 `ValueError`（隔了两层）。`p<=0` 看起来是"不可能有人这么设"，但它是 `sample_next(..., top_p=0)` 的真实路径 —— 单项检查里 `keep_top_p(p, 0.4)`（第一名 0.5 已经 ≥ 0.4）也走同一段代码：`take` 只有第一位 True，输出 one-hot。
* **`p >= 1` 要短路**。此时 nucleus 是整个词表，任何多余的除法都会引入 $10^{-16}$ 级别的数值差 —— 判据 `top_p=1 不改变分布` 用的是 1e-12 的逐位比较。短路支里那次 `/q.sum()` 保留：调用者给的可能没严格归一化。
* 顺带那个 `q.size <= 1` 守卫：空分布（有人把一个空切片传进来）时 `take[0] = True` 会 IndexError，而"只有一个词的词表"本来也没有可切的 —— 角落情形值得一行守卫。

为什么在**概率**上切而不在分数上切，§2.4 说过，这里补一个可见的现象：同一刀 `p=0.8` 切在分数上，$T$ 从 0.3 变到 2.5 时留下的候选集几乎不变（分数量纲整体缩放，但阈值是绝对值）；切在概率上，候选数会随陡缓自适应 —— **后者才是 nucleus 想要的行为**。

<details>
<summary>参考答案 · TODO-30 <code>keep_top_p</code></summary>

```python
    q = np.array(probs, dtype=np.float64, copy=True).reshape(-1)
    if p >= 1.0 or q.size <= 1:
        s = q.sum()
        return q / s if s > 0 else q
    order = np.argsort(q)[::-1]
    cum = np.cumsum(q[order])
    before = cum - q[order]                  # 轮到这一格**之前**攒下的质量
    take = before < max(float(p), 0.0)        # 还没攒够 p 就继续收，攒够了就停
    if not take.any():
        take[0] = True                        # p<=0 也要留一个候选，否则 0/0
    mask = np.zeros_like(q)
    mask[order[take]] = 1.0
    q = q * mask
    s = q.sum()
    return q / s if s > 0 else q
```

`max(float(p), 0.0)` 把负 `p` 折成 0，配合下一行的 `take[0] = True`：无论 `p` 多小，**至少留一个**。这两个是同一件事的两层保险，缺一层都会在某条你没想到的输入上变成 nan。
最后 `if s > 0 else q`：整行都是 0（`probs` 传错了）时不要返回 nan，返回全 0 会让 `rng.choice` 当场报"probabilities do not sum to 1"—— 报错位置离根因更近。

</details>

---

## 7. TODO-31：`sample_next`

**契约**：`sample_next(logits, rng, temperature=1.0, top_k=0, top_p=1.0, seen=None, penalty=1.0) -> int`。它是 §2.6 那条管线本身，五行：

```
z = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
if penalty != 1.0 and seen: z = apply_penalty(z, seen, penalty)
if temperature <= 0: return int(np.argmax(z))            # 贪心：不经过任何随机
probs = _softmax(keep_top_k(z / temperature, top_k))
return int(rng.choice(probs.size, p=keep_top_p(probs, top_p)))
```

四个点，每个都有一条判据：

1. **贪心在惩罚之后、在除温度之前**（§2.6 第一条）。$T=0$ 的定义是"取最大的**分数**"，而惩罚改的正是分数 —— 顺序对了，`TODO-31` 那条 `seen=[best], penalty=8.0, temperature=0` 才会交出**第二个**词。
2. **`rng` 是参数，不是内部 `default_rng()`**。同一段文本能复现，靠的是"调用方决定种子"。内部新建的话，同样参数调两次会给不同结果 —— 单项 `TODO-31` 的"同 seed 同结果"和判据 `temperature=0 时两个不同种子输出完全相同`（$T=0$ 时 `rng` 根本不该被碰）两头夹住它。
3. **温度在 softmax 之前**（§2.2）。`z / temperature` 写在 `_softmax(...)` 的括号里面，而不是 `probs / temperature`。这一条的症状最像"我的实现有 bug 但 loss 正常"：$T$ 开到 2.0 输出仍然单调。
4. **`rng.choice(n, p=probs)` 要求 `p` 非负、长度等于 `n`、和为 1**：前一条与最后一条由 `keep_top_p` 保证，等长靠 `n=probs.size`（别写 `len(logits)`，万一中途摊平过）。不要再自己除一次和 —— §2.6 最后一条。

返回 `int(...)` 是契约的一部分：判据要把它直接当索引喂给 `tok.decode` 和 `np.bincount`，numpy 标量能用但 `generate` 里拼 list 时会把类型搞成一锅粥。

写完之后 `python webapp/checks.py TODO-31` 一次查七件事：贪心（两个种子都 = argmax）、同 seed 可复现、`top_k=1` 恒为 argmax、惩罚在贪心之前生效、4000 次抽样的经验分布与 softmax 的 L1 ≤ 0.14、返回的 id 在 `[0,V)`、不改入参数组。

`rng.choice(n, p=probs)` 把「按概率抽样」整个藏进了 numpy。下面这张图把它摊开：温度滑杆改 p，
转盘扇区面积就是概率、扇区边界就是累积值，点 🎲 抽一次把 $[0,1)$ 的落点画在转盘上；
旁边直方图累积经验频率，白线是理论 p、须是 $\pm2\sigma$，还叠了一条 $1/\sqrt{N}$ 的包络 ——
判据里「4000 次抽样的经验分布与 softmax 的 L1 ≤ 0.14」说的就是这条包络。
图里三个错法开关（累积边界写反 / p 没归一 / 用 argmax 代替抽样）各自会让哪一格露馅，切一下就知道。

@@fig samplewheel@@

<details>
<summary>参考答案 · TODO-31 <code>sample_next</code></summary>

```python
    z = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if penalty != 1.0 and seen:
        z = apply_penalty(z, seen, penalty)
    if temperature <= 0:
        return int(np.argmax(z))              # 贪心：完全确定性，不加任何随机
    z = z / temperature
    z = keep_top_k(z, top_k)
    probs = _softmax(z)
    probs = keep_top_p(probs, top_p)
    return int(rng.choice(probs.size, p=probs))
```

入口那句 `.copy()` 在这里其实不是必需的（`apply_penalty` 返回新数组、`z / temperature` 也是新数组），但它让"这条管线绝不改动调用者的 logits"这件事**不依赖下游怎么写** —— 判据 `sample_next 不改入参数组` 查的就是整条管线的这个性质，用一次 copy 换来"以后随便就地改"的自由，值得。
`probs.size` 而不是 `len(logits)`：入参可能是 `(V,1)` 或 `(1,V)` 的切片（`generate` 传的是 `logits[0, -1]`，一维；但你在 notebook 里很可能传 `logits[:, -1]`）。摊平之后只有一个正确的长度。

</details>

---

## 8. 端到端：`GPT.generate` 怎么把采样用起来

`generate` 是给你的（阶段 5 §3.5 已经说过它刻意没有 KV cache）。它的五行主体决定了这一章最后八条判据的形状（下面说的"第 N 条"按 §9 的编号数）：

```python
ids = np.asarray(prompt_ids).reshape(1, -1)
for _ in range(max_new_tokens):
    if ids.shape[1] >= self.cfg.ctx:
        ids = ids[:, -self.cfg.ctx + 1:]        # 滑窗：丢最老的一个
    logits = self.forward(ids)                  # 整段重算，O(n²)
    nxt = sample_next(logits[0, -1], rng, temperature, top_k, top_p,
                      seen=ids.reshape(-1).tolist(), penalty=repetition_penalty)
    ids = np.concatenate([ids, np.array([[nxt]], dtype=ids.dtype)], axis=1)
return ids.reshape(-1).tolist()
```

读五件事：

* **返回的是"prompt + 新 token"一整串**，所以判据一律用 `g[len(prompt):]` 切出新写的部分。词表下标可能越界的线索就在这里：新 token 直接来自 `sample_next` 的 `int`，而 prompt 是你喂进来的。
* **滑窗**：`>= ctx` 时先丢最老的一个再往前长，所以序列总长永不超过 `ctx`。判据 `prompt 比 ctx 长时走滑窗而不是崩掉` 喂 39 个 token（`ctx=32`）、让它写 5 个，期望**输出长度正好 32**：第 1 步把 39 截成 31 再加回 1 个 = 32，之后每步都是"丢一个、长一个"，稳定在 32。不截断的话 `GPT.forward` 里那句 `assert T <= ctx` 会当场拦住 —— 拦得住，但那就不是生成而是崩了。
* **`seen` 传的是整段 ids**（截断后的），所以惩罚的范围随滑窗移动：太老的 token 会"恢复自由"。这也是 TODO-28 里那句"越界 id 要筛掉"的真实来源 —— `seen` 里可能带着换过 tokenizer 之前的老 id。
* **`seed` → `rng = np.random.default_rng(seed)`**，一颗 rng 贯穿整段生成。这就是判据能用 `{generate(..., seed=s) for s in range(24)}` 来数"多少种可能"的原因：`s` 是**独立实验**的编号，不是同一句里的第几个词。
* **$T=0$ 时 `rng` 一次都不被调**，所以两个不同种子的贪心输出必然逐 token 相同（判据第 24 条）。同理 `top_k=1` 时分布退化成 one-hot，采样和贪心等价（第 25 条）—— 注意前提是**没有并列**：真模型的 logits 是连续浮点，几乎不可能并列，而 §5 说的并列行为在这种子上会给出"两个候选"，那也是正确的。

判据后段那台模型：语料是四句小诗（`the cat sat on the mat .` / `the dog ran in the park .` / `the bird sang at the dawn .` / `the fish swam by the rock .`）各写三遍、整体再重复三遍，963 字符 / 36 行 / 词表 22，`d_model=48`、2 层、`ctx=32`，训 300 步、末段 loss 0.145（$\ln V=3.091$）。喂 `"the cat sat on the "`，让它写 10 个字符：真值取**语料里最后一次出现**的那个位置（`'mat .\nthe '`），所以"逐字符一致率"这个数是有唯一答案的。阈值给 0.70 而不是 1.00，是因为换一行、换一台机器，第 10 个字符可能就不对了 —— 0.70 允许最后一个字符不对，同时排除"完全没学到"。

其余四条多样性判据合起来说一句话：**采样确实在抽签**。$T=0$ 时 24 个种子只有 1 种可能（第 26 条，反面基线）；$T=2.0$ + `top_k=8` 时 24 个种子要有 ≥ 3 种不同首字符（第 27 条）、10 个种子要写出 ≥ 5 种不同续写（第 28 条）。为什么阈值这么低？因为它测的是"有没有随机性"，不是"写得好不好" —— 那是没有判据能替你决定的部分。

最后一条软提示：`重复惩罚换来更多样的字符分布`（`penalty=1.6` 的 20 个字符里不同字符数 ≥ 不加惩罚时）。它被列成软判据而不是硬判据，正是因为 20 个字符的样本量太小、方向感不等于结论。**这条边界你要带走**：样本不够时，正确的做法是降低断言强度，而不是降低阈值。

---

## 9. 验收判据逐条解读

`python stages/s07_generate.py`：30 条硬判据 + 1 条软提示，五组。前四组都在合成的 12 个分数上做（`V=12`，$\arg\max=7$，`n=6000` 或 `1200`）。

**温度组（5 条）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| 温度越高熵越大 | 五档温度的熵严格递增 | §2.2 的单调性。注意这条**不经过** `sample_next`（只走 `_softmax` + `keep_top_k`），所以 TODO-31 写错时它照样绿 —— 下面几条才测组装 |
| 经验熵也跟着温度上升 | $H(T{=}0.3)\le H(T{=}2.5)$ | 从**抽样频率**算的熵，不是公式 —— 采样真的按分布走 |
| `temperature=0` 就是贪心 | 三个种子的结果集合 == `{argmax}` | §7 第 1、2 点：贪心岔口 + `rng` 不参与 |
| $T=10^{-3}$ 仍在赌最高分 | 等于 argmax | §2.2 的下溢极限；顺带抓住"T 很小就乱抽"的实现 |
| 抽样频率 ≈ softmax 概率 | L1 ≤ 0.14（$n=6000$，$3\sigma$） | §2.7：整条管线的分布正确性（阈值不能更紧） |

**top-k 组（6 条）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| `top_k=2/3/6` 候选集恰好前 k 名 | `support(经验频率) == 前 k 名下标`（三条） | §2.3 + §5 坑 2（删元素会让下标体系崩掉） |
| 升温不引入新候选 | $T=4$、$k=2$ 时集合仍是前 2 名 | §2.3 第 2 点：温度与 k 可交换 |
| `k=0` 或 `k>=V` 时不剪枝 | 逐位等于入参 | §5 的短路（写成"照切"会退化成均匀分布） |
| 并列的最高分都保留 | `[1,1,0]`、`k=1` → 前两个都活着 | §5 坑 1：`<` 不是 `<=` |

**top-p 组（6 条）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| `top_p=0.5/0.8/0.95` 的核覆盖 ≥ p 的质量 | 和为 1、原分布里覆盖 `≥ p`、是概率降序前缀（三条） | §6：`before < p`、`[::-1]`、归一化 |
| `top_p=1` 不改变分布 | 逐位 `atol 1e-12` | §6 的短路支（多余的除法会引入 $10^{-16}$） |
| 极端 one-hot 时不冒出 nan | 全有限、和为 1、核只有一个下标 | §6 的兜底（`0/0` → nan → `rng.choice` 才炸） |
| 按 top-p 抽样后仍与归一化核一致 | L1 ≤ 0.14（$n=1200$） | §7 第 4 点：`rng.choice` 收到的是那个核 |

**惩罚组（5 条）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| 惩罚不修改调用者的 logits | 入参逐比特相同 **且** `out is not z` | §4 坑 1（视图就地改） |
| 正分除 / 负分乘 | 两个等式精确到 `1e-12` | §2.5 的不对称；统一除会红 |
| 被罚词经验频率明显下降 | `f_pen3 <= 0.75 × f_pen1` | 端到端效果（0.627 → 0.080），同一批随机数 |
| `penalty=1` 等于不惩罚 | $T=0$ 时仍是 argmax | §4 的短路语义 |
| 越界 id 被安全忽略 | 逐位等于入参 | §4 坑 3：`V+5` **和** `-2`（负索引不会报错） |

**端到端组（8 条）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| 贪心续写与原文逐字符一致率 | ≥ 0.70 | 模型 + `generate` + 贪心岔口一起（阶段 5 的机器真的能写东西） |
| $T=0$ 时两个不同种子输出完全相同 | 逐 token 相同 | §8 的 `rng` 语义 |
| `top_k=1` 与贪心等价 | 逐 token 相同 | §5 + §7：退化情形 |
| 贪心：24 个种子只有 1 种可能 | 集合大小 == 1 | 多样性的**反面基线**，没有它下一条无意义 |
| $T=2.0$ 时 24 个种子的不同首字符数 | ≥ 3 | §2.2：温度把分差抹平后老二老三开口 |
| 十个种子的不同续写数（$T=2.0$） | ≥ 5 | 整段有随机性，不只是第一个词 |
| 生成的 token 都能被 tokenizer 解回文本 | 每个 id 在 `[0,V)` | §7 返回 `int` + §5 坑 2 的下标体系 |
| prompt 比 `ctx` 长时走滑窗而不是崩掉 | `len == ctx` | §8 的滑窗 |
| *软：惩罚换来更多样的字符分布* | 不同字符数不降 | 样本只有 20 个字符，只给方向感（§8 末） |

---

## 10. 常见失败模式对照表

| 症状 | 病因 | 去处 |
|---|---|---|
| `ValueError: probabilities do not sum to 1` | `keep_top_p` 归一化后你又除了一次；或 nan 传进来了 | §2.6 末条、§6 兜底 |
| `ValueError: a and p must have same size` | `keep_top_k` 删元素而不是掩 MASK，`rng.choice` 的长度对不上 | §5 坑 2 |
| $T$ 开到 2.0 输出仍然只有一个词 | 温度写在 softmax 之后（除以 $T$ 再归一化等于没做） | §2.2 第 2 点 |
| 输出经常是词表里最冷僻的那个词 | 全行被掩成 MASK → softmax 给均匀分布 | §5 坑 1 + §3.2 |
| `top_p=0.8` 时核只覆盖 0.62 | 用了 `cum <= p`（跨过阈值那一格被踢掉） | §6 第一条，手算那个例子 |
| 核里全是长尾小词 | 忘了 `[::-1]`，`argsort` 升序 | §6 第二条（判据 `前缀` 专抓它） |
| 同一批 forward 连跑两次，第二次输出变了 | `apply_penalty` 就地改了模型那一行 logits | §4 坑 1 |
| 换了 tokenizer 以后偶尔惩罚打在奇怪的词上 | `seen` 里负 id 没筛（负索引合法、不报错） | §4 坑 3 |
| 同一段 prompt、同一个 seed 两次跑出不同结果 | `sample_next` 内部新建 rng | §7 第 2 点 |
| `temperature=0` 时崩，报 nan 或 inf | 没有贪心岔口，直接除 `T=0` | §7 第 1、2 点 |
| 反复出现同一个词，加 `repetition_penalty` 没用 | 惩罚写在贪心岔口**之后** | §2.6 第一条 |
| `top_k=1` 与贪心不等价 | 并列时 `<=` 把两个都掩掉 → 均匀；或短路条件写成 `k > 1` | §5 坑 1 |
| `IndexError` 在 `np.partition(z, -k)` | `k >= V` 没短路 | §5 末段 |
| 单项 `GIV-5` 变红 | 你顺手改了 `_softmax` / `entropy`（那是给你的代码） | §3；拿 `solutions/minigpt/sampler.py` 对拍那一半 |
| 加了惩罚却没效果 | `seen` 传了标量：`not 0` 直接短路（id 0 正好是换行符），惩罚静默失效 | §4：`generate` 传的是 **list**，你自己试的时候包一层 `[…]` |

---

## 11. 自查题

1. 为什么 `np.asarray(logits, dtype=np.float64).reshape(-1)` 可能给回一个**视图**？给出一个具体条件（dtype 是什么时），以及 `apply_penalty` 就地写之后谁会遭殃。
2. 惩罚为什么正分除、负分乘？如果 `z_i = -4.8`、$\rho=2.5$，两种写法各自给什么数，softmax 之后它的排名怎么变？
3. `keep_top_k` 与温度的顺序为什么**可以**交换？`_softmax` 与温度为什么**不可**以交换？请用一条公式各回答一次。
4. 手算一遍：`q = [0.5, 0.3, 0.15, 0.04, 0.01]`、`p = 0.6`，`before < p` 和 `cum <= p` 各留下哪些下标、覆盖多少质量？哪一个满足"nucleus 覆盖 ≥ 60%"这句话？
5. 判据里为什么每条分布类检验（L1 ≤ 0.14）都配一条布尔类检验（候选集 == 前 k 名）？各自能抓到对方抓不到的哪一类错误？
6. `temperature=0` 时 `rng` 一次都不该被调用。说出两条会因此变红的判据，并解释为什么"贪心也带一点随机"是很糟的设计。
7. 判据 `贪心：24 个种子只有 1 种可能` 是多样性判据的**反面基线**。如果去掉它，剩下哪一条会失去意义，为什么？
8. 把 `seen` 从"整段序列"改成"最后 32 个 id"（现在就是这样），会让重复惩罚产生什么随长度变化的行为？如果要修掉，你会改成什么（提示：frequency penalty，§2.5）？

---

## 12. 下一步

阶段 7 交付以后，你就有了一台**能训、能存、能说话**的完整小 GPT。剩下的两件事都在阶段 8（还没发布）：

* **速度**：`GPT.generate` 现在每生成一个 token 都把整段重跑一遍，$O(n^2)$。KV cache 把它降到 $O(n)$ —— 那时你会主动想回来读 §8 这段滑窗代码。
* **规模**：字符级 22 词表 → BPE 4096/8192。`tokenizer.py` 的协议已经留好（阶段 5 §9），而 `data/corpus.txt` 那种自撰语料要换成真数据，`checkpoint.py` 的格式一个字都不用改。

想继续深挖采样本身，`docs/advanced-cpp-roadmap.md §6` 的 M6 里程碑给的验收是："生成 200 token 不重复、无乱码、速度 ≥ 150 tok/s"。到那一步你大概会需要 frequency penalty、no-repeat-ngram、以及 beam search 之类的东西 —— 它们的骨架就是你这章写的那四个开关。
