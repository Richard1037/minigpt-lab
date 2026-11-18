# 阶段 5 · 完整 GPT + 训练循环：把零件开成一台机器

预计：约 120 行，4~8 小时（本手册代码量最大的一章）。
验收命令：`python stages/s05_char_lm.py`（36 条判据，CPU 上约 7 秒）
要写的地方：`src/minigpt/gpt.py`（TODO-15~19）、`tokenizer.py`（TODO-20）、`dataset.py`（TODO-21）、`optim.py`（TODO-22~24）。
网站：选「阶段 5 · 完整 GPT + 训练循环」，提交这四个文件。

---

## 1. 这一步在干什么

前四个阶段你写的是零件：`Value`、numpy 版网络层、因果自注意力。这一阶段把它们接成一台**能训练的机器**，然后当场开起来：

```flow
**文本** → `TODO-20` → **id 序列**
**id 序列** → `TODO-21` → `(B, ctx)` 批次
**批次** → `TODO-18` → `logits (B, T, V)`
**logits** → `TODO-19` → loss 标量 + 梯度
**loss** → `TODO-15/16/17` → 每一层把梯度交回去
**梯度** → `TODO-22/23/24` → 参数往前走一小步
**重复 400 次** → loss 从 ln V 掉到 0.1
```

一句话：**语言模型的全部秘密就是"预测下一个 token"这七个字**，其余都是工程。

这一章有三条判据是"其它阶段看不见的"，因为它们测的是**装配**而不是零件：

1. `残差出口置零后 Block 是恒等映射` —— 查你两条残差加没加、梯度有没有原样透传；
2. `tied 的两路梯度是累加不是覆盖` —— 查输出头和词嵌入共用那块内存时你写的是 `+=` 还是 `=`；
3. `随机数据学不动（反作弊判据）` —— 查移位方向。**这条最重要**，§8 会解释为什么。

阶段 4 的考卷 `tests/test_model.py` 有 9 条要等你写完 `gpt.py` 才真的跑到（Block、整机梯度、因果性那几组）。在那之前它们靠的是答案版 `gpt.py` —— 也就是说：**阶段 4 那天你看到的 17/17 里，有 9 条其实不是你自己写的东西过了**。现在轮到你了。

---

## 2. 必要理论

### 2.1 残差流：为什么主干是"高速公路"

一个 block 干的事是

$$x_{1}=x+\mathrm{attn}\bigl(n_1(x)\bigr),\qquad
  x_{2}=x_{1}+\mathrm{mlp}\bigl(n_2(x_{1})\bigr)$$

主干 $x$ 从头到尾没有被**覆写**过，只被**加过**。把 $L$ 层展开：

$$x_{\text{出}}=x_{\text{入}}+\sum_{\ell=1}^{L}\Bigl[\mathrm{attn}_\ell(n_1^{(\ell)}(\cdot))+\mathrm{mlp}_\ell(n_2^{(\ell)}(\cdot))\Bigr]$$

三个推论，全都是这一章判据的依据：

* **梯度有直达通道。** $\partial x_{\text{出}}/\partial x_{\text{入}} = I + \sum_\ell \partial(\cdot)$，那个 $I$ 就是"原样透传"那一路。深层网络不怕梯度消失，怕的是相反的方向（见 §3.4 的 `resid_scale`）。
* **子层输出为零时，这一层必须等于恒等映射。** 这就是判据 `残差出口置零后 Block 是恒等映射` 的数学依据 —— 它是一条**不依赖有限差分**的强检验，forward 和 backward 各查一半。
* **信息不会丢。** 第 3 层想引用第 0 层的 token 内容，不需要经过任何非线性变换 —— 加法和恒等通路一直在。

### 2.2 pre-norm 而不是 post-norm

你写的顺序是 `x + attn(norm(x))`（**pre**-norm），不是 `norm(x + attn(x))`（**post**-norm，2017 年原论文那种）。差别在层数上去以后才露出来：post-norm 每个 block 出口都有一次归一化，梯度反传时要连乘 $L$ 个"把长度除回去"的因子，24 层以上就训不动了。pre-norm 的主干是纯加法，所以今天的 LLM 全部用 pre-norm。

代价：最后一层出来以后主干的长度已经不是初始量级了，所以**末端还要再 norm 一次**（`norm_f`），否则 tied 输出头的 logit 尺度会随深度漂 —— 这是 TODO-18 里那次 `self.norm_f(x)` 的存在理由。

### 2.3 权重共享（weight tying）：一行减法省下的东西

输出层不做成独立的 `Linear(C, V)`，而是直接用词嵌入矩阵的转置：

$$\mathrm{logits}=x\,E^{\mathsf T},\qquad E\in\mathbb R^{V\times C}\ \text{就是输入侧那张嵌入表}$$

省下 $V\cdot C$ 个参数。本项目 $V=25,C=48$ 省 1200 个；真实模型 $V=8192,C=512$ 省 400 万个 —— 在小模型上这是 10%~20% 的参数。

第二个理由是概念的绑定：输入侧 $E$ 的第 $i$ 行回答"字符 $i$ 长什么样"，输出侧同一行回答"要输出字符 $i$，我该往哪个方向走"。这两个问题高度相关，绑在一起等于**共享一份先验**，小数据上明显更稳。

代价是数学上的：`tok.weight` 现在有**两条**梯度来源 ——

$$\frac{\partial L}{\partial E}=\underbrace{\frac{\partial L}{\partial E}\Big|_{\text{查表}}}_{\text{Embedding.backward 散射回行}}+\underbrace{g_{\text{logits}}^{\mathsf T}x_{\text{flat}}}_{\text{TiedHead.backward}}$$

两条必须**相加**。写成赋值 `=`，谁后跑谁赢 —— 这就是判据 `tied 的两路梯度是累加不是覆盖` 存在的全部理由。

### 2.4 下一 token 预测与 $\ln V$ 基准

训练目标是"看到前缀，猜下一个字符"：

$$L=-\frac{1}{BT}\sum_{b,t}\log p_\theta\bigl(x_{b,t+1}\mid x_{b,\le t}\bigr)$$

两个必须刻在脑子里的数字：

* **随机初始化的模型 loss 应该等于 $\ln V$。** 权重是随机的小数，softmax 出来接近均匀，$-\frac1V\sum_i\ln\frac1V=\ln V$。本项目 $V=25$，所以 $\ln V=3.219$。这就是判据 `初始 loss ≈ ln V`。差得远 → 移位、归一化或者 CrossEntropy 有问题。
* **不可预测的数据，loss 只能降到 $\ln V$，降不下去是**对的**。** 随机字符序列没有任何规律，模型再强也猜不到下一个字符。所以"随机数据上学不动"是**正确答案的标志**。反过来，如果你的模型在随机数据上把 loss 打到了 0.001，它一定在作弊 —— 而作弊唯一的可能就是"预测自己"：答案就在输入里，抄一下就行。

这条反直觉的判据（学不动 = 通过）是整份手册里唯一"结果和你期望相反"的检验，所以 §8 还会再讲一次。

下面这张图把 $V$ 从 2 拖到 8192，多批采样真算随机初始化时的交叉熵，和 $\ln V$ 那条线叠在一起；再给一个"目标不错位（等于看答案）"的开关 —— 一打开，loss 立刻塌到 $10^{-4}$ 量级，假得一目了然。

@@fig lnv@@

### 2.5 Adam 的偏差修正为什么不是可有可无

一二阶矩的滑动平均都从 0 开始：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

在 $m_0=v_0=0$ 的前提下，$E[m_t]=(1-\beta_1)\sum_{k}\beta_1^{k}g_{t-k}\ne E[g]$ —— 前几十步系统性偏小。除以 $1-\beta^t$ 把它拉回无偏：

$$\hat m_t=\frac{m_t}{1-\beta_1^{t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{t}},\qquad
  \theta_t=\theta_{t-1}-\mathrm{lr}\cdot\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

**忘了会怎样（可以手算）**：$t=1$、$g$ 恒定时 $\hat m/\sqrt{\hat v}=1$，一步正好走 lr；不修正的话 $m/( \sqrt v)=\frac{0.1g}{\sqrt{0.05g^2}}=0.447$，第一步只走了 $0.447\,\mathrm{lr}$，第二步 0.608、第三步 0.725……曲线前 200 步整体偏慢，症状是"warmup 之后还在爬"。判据 `AdamW 单步幅度 = lr` 用**恒梯度**构造了一个可解析验证的场景：正确实现两步都恰好走 lr，误差 3e-11 量级。

---

## 3. 已给你的代码：装配图纸与三个决定

这一节讲**不是我让你写**的那些行。它们每个都有一个"删了会怎样"，你以后排错要靠这些。

### 3.1 `Block.__init__`：四个子模块，顺序就是数据流

```python
self.norm1 = RMSNorm(cfg.d_model, cfg.eps)
self.attn  = CausalSelfAttention(cfg, rng=rng)
self.norm2 = RMSNorm(cfg.d_model, cfg.eps)
self.mlp   = MLP(cfg.d_model, hidden, rng=rng)
```

`hidden` 那行有个分支：`cfg.use_swiglu` 为真时用 `cfg.swiglu_hidden()`（≈ $\frac{8}{3}d$ 取 64 的倍数），否则用经典的 $4d$。基础阶段 `use_swiglu=False`，所以你现在的 FFN 是 `Linear(C→4C) → GELU → Linear(4C→C)`。**为什么留这个钩子**：SwiGLU 把激活函数和门控合在一起，同参数量下更强，但要三层 Linear —— 结构改动会影响检查点格式，所以钩子留在这里，改动留到阶段 8。

`Module._collect` 按属性名递归收集参数，**下划线开头的属性被跳过**（`self._cache` 之类不算子模块）。这就是"缓存必须以下划线开头"这条约定的由来：写成 `self.probs = ...` 的话，参数收集器会去里面找参数，浪费一次递归，而且把 `(B,H,T,T)` 的大数组当成结构的一部分。

### 3.2 `GPT.__init__`：五件东西，两件有讲究

```python
self.tok = Embedding(cfg.vocab, cfg.d_model, cfg.init_std, rng=rng)
self.pos = Embedding(cfg.ctx,   cfg.d_model, cfg.init_std, rng=rng)   # 行数 = ctx，不是 vocab
self.blocks = [Block(cfg, rng=rng) for _ in range(cfg.n_layer)]
self.norm_f = RMSNorm(cfg.d_model, cfg.eps)
self.head = TiedHead(self.tok.weight)      # ← 传的是对象，不是数据
```

* `pos` 的**行数**是 `ctx` 而不是 `vocab`：它按"第几个位置"取行，所以只需要 `ctx` 行。删掉 `pos` 的话模型变成一个"字符袋"—— 每个位置的输出只取决于**当前字符**和它前面的**多重集合**，学不会"动词跟着主语"这类位置敏感的结构。判据 `打乱顺序输出会变` 就是查它在不在。
* `TiedHead(self.tok.weight)` 传的是**Parameter 对象**。如果写成 `TiedHead(self.tok.weight.data)`，你复制的就是一份脱离关系的数值：梯度永远回不到嵌入表，而且 `参数量与手算一致` 那条会多算一份（因为不再共享）。

### 3.3 `init_resid_scale`：那个 `1/√(2L)`

`cfg.resid_scale` 为真时，构造的最后一步调用 `init_resid_scale(self, cfg.n_layer)`，它把每层**残差出口**的两个矩阵（`attn.proj.W`、`mlp.fc2.W`）乘上 $1/\sqrt{2L}$。就这两处，别的张量一位不动。

判据 `GIV-3` 数的是出口矩阵的范数比值：$L=3$ 时应当正好是 $\sqrt 6=2.4495$。

删掉它：形状、loss、判据的梯度检查全部照常，只是层数一深就开始不稳 —— 这类"现在没症状"的决定最难查，所以判据把它钉死了。为什么正好是 $1/\sqrt{2L}$，见 §19 进阶推导 A（pre/post-norm 的完整对比推在 [阶段 3a §11](s03a.html#sec-11)）。

两根滑块 + 一个 pre/post-norm 开关：红柱是不缩放、绿柱是乘了 $1/\sqrt{2L}$。
注意纵轴是 $\log_{10}$ 的方差比 —— pre-norm 下不缩放是**线性**起飞（每层加的是常数份），
不是指数爆炸；那个指数版本只在 post-norm 成立，切过去看一眼你就知道两种假设差多远。

@@fig resid@@

### 3.4 `GPT.backward`：为什么这段给你写

```python
g = self.head.backward(grad_logits)
g = self.norm_f.backward(g)
for blk in reversed(self.blocks):
    g = blk.backward(g)
self.pos.backward(g)
self.tok.backward(g)
```

它没有任何新数学，只是"正向的镜像"。之所以给你：这一层出错时症状是"某个参数梯度恒为 0"，判据直接抓得住。**两个细节值得注意**：

* `reversed(self.blocks)`：反传顺序必须严格倒过来，跳着走的话形状一般还能对上（都是 `(B,T,C)`），数值是错的。
* `pos` 和 `tok` **都要**走。只走 `tok` 是最常见的写法（"位置嵌入又不是内容"），结果 `pos.weight` 永远是死参数 —— `无死参数` 那条判据就是为它准备的。
* 传进来的 `grad_logits` 是**完整**的 `(B,T,V)`：最后一位是 0。§8 会讲为什么不能省。

### 3.5 `GPT.generate`：一个刻意很笨的实现

```python
if ids.shape[1] >= self.cfg.ctx:
    ids = ids[:, -self.cfg.ctx + 1:]        # 滑窗：丢最老的一个
logits = self.forward(ids)                  # 整段重算！
nxt = sample_next(logits[0, -1], rng, ...)  # ← 阶段 7 的 TODO 挂在这一行
```

**完全没有 KV cache**：每生成一个 token 都把整段重跑一遍，复杂度 $O(n^2)$。这是故意的 —— cache 是阶段 8 的性能主题，而"每次重算"的正确性一眼可查（它和训练用的是同一个 `forward`）。`sample_next` 属于阶段 7，所以现在这个函数在你写完整机的梯度之后还会**卡住一次**（`TODO-28~31`），这是正常的，不是坏掉。

---

## 4. TODO-15：`Block.forward`

**契约**：`(B,T,C) → (B,T,C)`。不需要缓存任何新东西（每个子层已经自己存了）。

五行，顺序就是 §2.2 那张图：

```
h1 = norm1(x)      # 归一化只在进子层之前做，主干 x 一个字都不改
a  = attn(h1)
x1 = x + a         # 第一次加回主干
h2 = norm2(x1)     # ← 注意吃的是 x1，不是 x
return x1 + mlp(h2)
```

**三个必须想清楚的地方**：

1. `norm2` 的输入是 $x_1$ 还是 $x$？写成 `norm2(x)` 时形状全对、loss 照样降、`Block 反向正确` 会挂（梯度链路和正向不一致），但 `残差出口置零` 那条**也会挂** —— 后者更便宜，先跑它。
2. 不要在这里加"再 norm 一次"的额外步骤。pre-norm 的末端 norm 只在 `GPT.forward` 最后做一次（§3.4）。
3. `Block` 是纯函数式的组合：`forward` 里不许出现随机数、不许读 `self.cfg`。整个项目"层内不藏随机性"的约定（`nn.py` 头部第 5 条）是靠得上的。

写完先跑单项：`python webapp/checks.py TODO-15`（会打印输入形状、输出形状，还有那条"残差出口置零 = 恒等映射"）。

<details>
<summary>参考答案 · TODO-15 <code>Block.forward</code>（判据没过之前别点开）</summary>

```python
    def forward(self, x):
        h1 = self.norm1(x)
        a = self.attn(h1)
        x1 = x + a
        h2 = self.norm2(x1)
        m = self.mlp(h2)
        return x1 + m
```

五行里没有一处可以"顺手改改"：`h2` 必须是 `norm2(x1)`（第二个残差流从 $x_1$ 出发），`return` 必须是 `x1 + m` 而不是 `x + m`。
`self.attn(h1)` / `self.mlp(h2)` 走的是 `Module.__call__`，它只是 `forward` 的别名 —— 但**一定要用调用而不是直接写 `self.attn.forward(h1)`**，两者等价，写 `__call__` 是为了以后给子层加钩子时不用改这里。

</details>

---

## 5. TODO-16：`Block.backward`

**契约**：收到 `dL/d(out)`（形状 `(B,T,C)`），返回 `dL/dx`（同形状），并把梯度**累加**进 13 个参数张量（每个 block：norm1.w、attn 的 8 个、norm2.w、mlp 的 4 个）。

核心是 §2.1 那个 $I$。收到的 `g` 同时是**两个变量的输出**（`x1` 和 `mlp` 的输出），所以它要抄两份：

$$g_{x_1}=g+\frac{\partial n_2}{\partial x_1}\bigl(g^{\text{mlp}}\bigr),\qquad
  g_x=g_{x_1}+\frac{\partial n_1}{\partial x}\bigl(g^{\text{attn}}\bigr)$$

其中 $g^{\text{mlp}}=$ `mlp.backward(g)`，$g^{\text{attn}}=$ `attn.backward(g_x1)`。

```
g_h2 = mlp.backward(g)              # 支路
g_x1 = g + norm2.backward(g_h2)     # 直通 + 绕道
g_h1 = attn.backward(g_x1)
return g_x1 + norm1.backward(g_h1)  # 再抄一次
```

**三个坑**：

* **忘了 `+ g`**：只交支路那半。症状：loss 降得慢、`Block 反向正确` 挂在 norm 的 `w` 和更靠前的层上，而 `残差出口置零时梯度原样透传` 这条会直接挂（支路是 0，你交出的是 0 而不是 `g`）。
* **顺序倒过来**（先算 `norm2.backward` 再 `mlp.backward`）：`mlp.backward` 要用它 forward 时缓存的中间量，先消费后产生 → 拿到的是上一次的缓存，形状常常还兼容，于是**悄悄错**。判据抓得住，但你得花一小时定位，所以按"倒着走 §4"的节奏写。
* **`return` 之前又加了一次 `g`**：把残差抄成两遍（`g + g + ...`）。这个 bug 会让所有靠前的梯度翻倍，而 `gradcheck` 一定挂 —— 有限差分是最诚实的裁判。

<details>
<summary>参考答案 · TODO-16 <code>Block.backward</code></summary>

```python
    def backward(self, g):
        g_h2 = self.mlp.backward(g)              # 支路：穿过 mlp
        g_x1 = g + self.norm2.backward(g_h2)     # 直通 + 绕道 norm2
        g_h1 = self.attn.backward(g_x1)          # 第二层残差同理
        return g_x1 + self.norm1.backward(g_h1)
```

两处 `+ g`（第二个是 `+ g_x1`）就是 §2.1 展开式里那个恒等项 $I$。
`mlp.backward` / `attn.backward` 内部已经把参数梯度累加完了，这里**只**处理输入梯度，不要再去碰任何 `p.grad`。

</details>

---

## 6. TODO-17：`TiedHead.forward / backward`

**契约**：`forward(x)`：`(B,T,C) → (B,T,V)`，`V` 从借来的 `self.weight.data.shape[0]` 读（**不许**再从 cfg 拿一遍，那样两处口径会漂）。
`backward(g)`：返回 `dL/dx`，并把 `dL/dW` **累加**进 `self.weight.grad`。

正向就是 `Linear` 的那一行，只是没有 bias、而且 `W` 是别人的：

$$\text{logits} = x_{\text{flat}}\,W^{\mathsf T},\qquad x_{\text{flat}}\in\mathbb R^{BT\times C},\ W\in\mathbb R^{V\times C}$$

反向三条（形状自己推：想让 $(V,C)$ 出现，只能是 $g^{\mathsf T}x$ 这个次序）：

$$\frac{\partial L}{\partial W}=g_{\text{flat}}^{\mathsf T}x_{\text{flat}}\ \ (V,C),\qquad
  \frac{\partial L}{\partial x_{\text{flat}}}=g_{\text{flat}}W\ \ (BT,C),\qquad
  W.\text{grad}\ \textbf{+}= \frac{\partial L}{\partial W}$$

**为什么这个 `+=` 是本阶段最重要的一行**：`tok.weight` 有两个用户（查表的 `Embedding.backward` 散射回行 + 这里）。写成 `=` 时"谁后跑谁赢"：`GPT.backward` 的顺序是 head → blocks → pos → tok，**tok 在最后**，所以覆盖的方向恰好是"查表那半把输出头那半抹掉了"。整机 loss 仍然会降（因为查表那半才是主体），但 `tok.weight` 的输出头梯度永远缺席 —— 单测 `TiedHead` 时看不出问题，只有"跑两次 backward 看梯度翻不翻倍"能抓住它，所以判据里有这一条。

**缓存**：`_x_shape`（还原返回值形状用）和 `_x_flat`（算 `dL/dW` 用）。`_probs` 之类的东西不需要 —— 这里没有归一化。

<details>
<summary>参考答案 · TODO-17 <code>TiedHead</code></summary>

```python
    def forward(self, x):
        self._x_shape = x.shape
        self._x_flat = x.reshape(-1, x.shape[-1])
        V = self.weight.data.shape[0]
        return (self._x_flat @ self.weight.data.T).reshape(x.shape[:-1] + (V,))

    def backward(self, g):
        # 关键：这个梯度要**累加**进词嵌入的 grad，和查表层回传的梯度叠在一起
        V = self.weight.data.shape[0]
        g2 = g.reshape(-1, V)
        self.weight.grad += g2.T @ self._x_flat
        return (g2 @ self.weight.data).reshape(self._x_shape)
```

`x.shape[:-1] + (V,)` 这个写法让 `B`、`T` 随便变化：阶段 7 生成时 `B=1`、阶段 5 训练时 `B=4`，同一个头都用得上。
`self.weight.grad +=` 而不是 `self.weight.grad =` —— 这一行就是 §6 整段话的全部内容。

</details>

为什么必须是 `+=`：tied 头这一块内存同时被"查表"和"输出投影"两条反向路径写入。下面这张时间线把两路梯度先后落进去，对照 `+=` 与 `=` 两种写法的最终结果 —— 用 `=` 会把先到的那路直接抹掉。

@@fig tiedgrad@@

---

## 7. TODO-18：`GPT.forward`

**契约**：`ids` 是 `(B,T)` 整数数组，返回 `(B,T,vocab)`。

```
positions = np.arange(T)[None, :].repeat(B, 0)      # (B,T)，每一行都是 0..T-1
x = tok(ids) + pos(positions)                       # 内容 + 位置
for blk in self.blocks: x = blk(x)
return head(norm_f(x))
```

**四个点**：

1. `positions` 为什么不是 `ids`：位置嵌入按**下标**取行，和这个位置是什么字符无关。这是"学习式位置嵌入"的全部实现（`use_rope` 那条路把这两项相加换成对 $q,k$ 做旋转，见 `docs/advanced-cpp-roadmap.md §6.2`）。
2. `assert T <= ctx`：`pos` 只有 `ctx` 行。越界时 numpy 会给 IndexError，看不出是自己搞错了长度；assert 的消息直接说"去改 `Config.ctx`"。
3. `np.asarray(ids)`：判据和训练循环传的可能是 list（分词器返回 list）。**在这一层入口统一一次**，后面所有层就不必各自防御 —— 这是"边界处规范化类型"的通用做法。
4. `head(norm_f(x))`：末端 norm 的必要性见 §2.2。删掉它 `GPT 整机反向正确` 不会挂（数学上仍然是个合法网络），但 logit 尺度随深度漂，`初始 loss ≈ ln V` 那条就会偏出容差 —— 一个"结构决定"被一条数值判据钉住，这个模式你以后会常见。

单项快测 `python webapp/checks.py TODO-18` 查四件事：形状、位置嵌入清零后输出会变、打乱顺序输出会变、`T > ctx` 被拦住。

<details>
<summary>参考答案 · TODO-18 <code>GPT.forward</code></summary>

```python
    def forward(self, ids):
        ids = np.asarray(ids)
        B, T = ids.shape
        assert T <= self.cfg.ctx, f"序列长度 {T} 超过配置 ctx={self.cfg.ctx}"
        positions = np.arange(T)[None, :].repeat(B, 0)
        x = self.tok(ids) + self.pos(positions)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm_f(x))
```

`np.arange(T)[None, :]` 是 `(1,T)`，`.repeat(B, 0)` 变 `(B,T)` —— 每个 batch 的位置都一样，所以不用真的复制数据也能算（`np.broadcast_to` 也行，但那会是个只读视图，`Embedding.forward` 要拿它当索引，复制更省事）。

</details>

---

## 8. TODO-19：`GPT.loss_backward`（本阶段最贵的一条）

**契约**：给 `(B,T)` 的 id，返回**标量** loss，并把梯度累加进所有参数。它是训练循环唯一需要调的接口。

```
logits = forward(ids)                                    # (B,T,V)
loss   = CrossEntropy(logits[:, :-1, :], ids[:, 1:])     # ← 移一位
grad   = ce.backward(1.0)                                # (B,T-1,V)
full   = zeros(logits.shape); full[:, :-1, :] = grad     # 补回最后一位
backward(full)
```

### 8.1 为什么是 `logits[:, :-1]` 配 `ids[:, 1:]`

位置 $t$ 的输出**只能看到** $x_{\le t}$（因果掩码保证），所以它唯一合法的用途是预测 $x_{t+1}$。把 $t$ 的输出配 $t$ 的字符当目标，就是"预测自己"。

下面这张图把"错位量"做成滑杆（$-1/0/+1$），真算每种配法的交叉熵：配对了 loss 老实地停在 $\ln V$ 附近，配错（等于看答案）立刻掉到接近 0 —— 假得一眼能认。

@@fig shiftdir@@

### 8.2 预测自己的伪装有多好

代码里有个对照实验（`tests/test_model.py` 的 `check_label_shift`）：在一份**完全随机**的 token 序列上训 150 步。

| 移位 | 训练 loss | 新数据上的 loss | 判据怎么读 |
|---|---|---|---|
| 正确（配 `ids[:, 1:]`） | 会降一点 | $\approx \ln V$（学不动） | 学不动 = 通过 |
| 错误（配 `ids[:, :-1]`） | 掉到 0.001 | 同样掉到 0.001 | 秒收敛 = 有 bug |

"预测自己"之所以在任何数据上都收敛，是因为任务本身是平凡的：答案就在输入里。`stages/s05_char_lm.py` 用两个方向各堵一次：

* `训练 loss 掉到 ln V 的零头` + `贪心下一字符准确率 ≥ 0.85`（模式数据上**必须**学得动）；
* `随机数据学不动（反作弊判据）`：末段 loss 必须 $\ge 0.9\ln V$（**不许**学得动）。

只写第一条的话，"预测自己"的实现在模式语料上照样满分 —— 它是靠输入抄答案，不需要学到任何规律。

### 8.3 那个补零的 `full`

`CrossEntropy.backward` 给的是 `(B, T-1, V)`：$T-1$ 个参与 loss 的位置。而 `GPT.backward` 的入口约定是**完整形状**（§3.4）：

* 形状对不上就当场 `ValueError`（最好的情况）；
* 更糟的是"顺手 `reshape` 一下"或者把末位当成第一个位置 —— 错位一格，`head` 的 $g^{\mathsf T}x$ 会把**下一个位置的输入**乘到**这个位置的梯度**上，`参数量`、`形状`全对，只有 `GPT 整机反向正确` 抓得住。

末位没有被任何 loss 项用到，所以它的梯度**必须是 0**，而不是"不存在"。补零就是把这件事写清楚。

### 8.4 `backward(1.0)` 里那个 1.0

链式法则的最上游：$\partial L/\partial L = 1$。传别的数（比如 `batch`）就等于偷偷把整个梯度放大 —— 症状和 lr 调错一模一样，但根因完全不同。`CrossEntropy.backward` 内部会除以样本数 $BT$（阶段 3 讲过），所以这里真的只需要传 1.0。

<details>
<summary>参考答案 · TODO-19 <code>loss_backward</code></summary>

```python
    def loss_backward(self, ids) -> float:
        ids = np.asarray(ids)
        logits = self.forward(ids)
        loss = self._loss_fn(logits[:, :-1, :], ids[:, 1:])
        grad = self._loss_fn.backward(1.0)              # (B, T-1, V)
        full = np.zeros(logits.shape, dtype=np.float64)  # 补回最后一位：它的梯度是 0
        full[:, :-1, :] = grad
        self.backward(full)
        return loss
```

`self._loss_fn` 是构造时建的**一个** CrossEntropy 实例（`_` 开头 → 不被当成子模块）。复用同一个实例的意义：`forward` 缓存的概率正好被 `backward` 消费，中间没有任何重新计算。如果你在这里每次新建 `CrossEntropy()`，也能跑，但缓存会跟着实例一起新，等于强迫你 `forward`/`backward` 紧挨着写 —— 现在这样更宽容。

</details>

---

## 9. TODO-20：`CharTokenizer.encode / decode`

**契约**：`encode(text, drop_unknown=True) -> list[int]`；`decode(ids) -> str`；构造时词表已经是 `sorted(set(语料))`，你的任务只是查表/反查表。

这一条只有十几行，但它有**三条独立判据**，因为分词错了的代价是"三天后在 loss 曲线里找原因"：

1. `往返一致 decode(encode(x)) == x` —— 唯一的、不依赖模型的正确性检验；
2. `词表口径 vocab == 去重字符数`，且所有 id 落在 `[0, V)`；
3. `语料外的字符按契约处理`：默认丢掉、`drop_unknown=False` 时抛 `KeyError`。

**最容易中的那个 bug 是"用假值判断有没有找到"**：`i = self._index.get(ch)` 之后写 `if not i:` —— id **0 是合法值**（本项目里 0 是换行符 `\n`），于是第一个字符永远编不进去。症状是 loss 莫名高 0.05、生成里从不出现换行。所以必须写 `if i is None:`。判据里 `id 0 是 '\n'（不是假值陷阱）` 就是在数这件事。

为什么 `encode` 返回 `list` 而不是 numpy 数组：将来换 BPE 时"一段文本对应几个 token"是不定的，`list` 才是对的抽象；`dataset.py` 需要数组时自己 `encode_array`。

<details>
<summary>参考答案 · TODO-20 <code>encode / decode</code></summary>

```python
    def encode(self, text: str, drop_unknown: bool = True) -> list[int]:
        out: list[int] = []
        for pos, ch in enumerate(text):
            i = self._index.get(ch)
            if i is None:
                if drop_unknown:
                    continue
                raise KeyError(f"字符 {ch!r}（第 {pos} 个）不在词表里；要么 drop_unknown=True，要么换 tokenizer")
            out.append(i)
        return out

    def decode(self, ids) -> str:
        return "".join(self._chars[int(i)] for i in ids)
```

`decode` 里越界**不要**取模、不要 clamp：让 `IndexError` 炸出来。生成时模型吐出一个越界 id 是"训练崩了"的现场证据，悄悄折回词表就把线索抹掉了。

</details>

---

## 10. TODO-21：`Batcher.__call__`

**契约**：给 `(B, ctx)` 的输入和 `(B, ctx)` 的目标，满足 `y[:, :-1] == x[:, 1:]`。

```
起点：batch 个随机整数 ∈ [0, max_start]      # 有放回，所以 batch 可以大于窗口总数
窗口：ids[s : s+ctx+1]                        # 长度 ctx+1
返回：win[:, :-1], win[:, 1:]                 # 各 (B, ctx)
```

**三件事要说清楚**：

* **为什么窗口是 `ctx+1` 而不是 `ctx`**：`x` 要 `ctx` 个输入位置，而答案 `y` 需要第 `ctx+1` 个字符（最后一个位置的目标）。`__init__` 里那句 `max_start = ids.size - ctx - 1` 就是这个 `-1`：写成 `ids.size - ctx` 时最后一个窗口只有 `ctx` 个 token，切片出来形状对、内容缺一格（`IndexError` 或者静默错值）。
* **`rng` 从外面传**：不传就每次新建同一颗种子 → 每一步吃同一批数据。判据 `换 rng 换窗口、同 rng 可复现` 两头都查。日志和网页结果之所以能对上，靠的就是"第二头"。
* **`y` 训练时用不到**（`loss_backward` 自己会移一位，TODO-19），为什么还要返回它：① `evaluate()` 拿它算验证 loss，必须和训练同一种移位，否则 train/val 两个数不可比；② 它把"谁是答案"写成数据，能被 `y[:, :-1] == x[:, 1:]` 直接检查。

**为什么用随机窗口而不是顺序 epoch**：语言模型的 batch 是"从语料里随手抓一段"。所以下游没有 epoch 概念，只有 step 数 —— 阶段 6 的学习率调度按 step 算就是因为这个。

错位一格这件事，盯着代码是看不出来的。下面这张图用 `data/corpus.txt` 开头的真字符当字节流：
拖 T 和 batch 看窗口怎么切，x 与 y 两排格子并排，箭头把「x 的第 k 格 → y 的第 k 格」连起来，
y 最后那一格单独描边 —— 它不在 x 里，来自流的下一个字符，这就是窗口要取 `ctx+1` 的全部理由。
▶ 播放让窗口沿语料一格一格滑（每一帧的状态都由纯函数从头重算，所以单步和播放看到的是同一帧）。
两个错法开关：`y = x` 不错位，会让「复读机命中率」变成 100%；先摊平再错位，则会画出串到隔壁
样本的格子数 —— 后者形状全对、判据还不红，是真正阴险的那一种。

@@fig batcher@@

<details>
<summary>参考答案 · TODO-21 <code>Batcher.__call__</code></summary>

```python
        rng = rng if rng is not None else np.random.default_rng(0)
        starts = rng.integers(0, self.max_start + 1, size=int(batch))
        win = np.stack([self.ids[s:s + self.ctx + 1] for s in starts])   # (B, ctx+1)
        return win[:, :-1], win[:, 1:]
```

用列表推导 + `np.stack` 而不是高级索引：一次性取 `ids[starts[:, None] + np.arange(ctx+1)]` 也能算对，但形状是 `(B, ctx+1)` 需要自己想清楚广播，出错时症状是"每行都是同一个窗口"或"行列转置了"，比逐行切片难查十倍。

</details>

---

## 11. TODO-22：`AdamW.step`

**契约**：`step(lr=None) -> float`，返回这一步**真正用到的** lr（判据和日志都要它）。`lr=None` 时用构造时的 `self.lr`；训练循环传 `lr_at(...)` 的结果，调度才生效。

$$\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t & v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2\\
c_1&=1-\beta_1^{t} & c_2&=1-\beta_2^{t}\\
\theta_t&=\theta_{t-1}-\mathrm{lr}\cdot\frac{m_t/c_1}{\sqrt{v_t/c_2}+\epsilon},
&\theta_{t-1}&\leftarrow\theta_{t-1}\,(1-\mathrm{lr}\cdot\lambda)\ \text{（仅未豁免参数）}
\end{aligned}$$

**四个必须写对的地方**：

1. `self.step_count += 1` 要在算 `c1/c2` **之前**：$t$ 从 1 开始。放在后面的话第一步用 $t=0$ → $c_1=0$ → 除零 → `inf` 传染。
2. **`mi[...] =` 而不是 `mi =`**。`Optimizer.__init__` 把数组引用存进 `self.m` 列表；重新绑定名字等于丢掉缓存（下一步又从零开始）。症状：训不动，loss 抖成一个平台。这条和 `p.data[...] =` 一样是"原地写"约定，`GIV-1` 已经查过 `zero_grad` 的原地性。
3. `eps` 在根号**外面**：$\sqrt{\hat v}+\epsilon$。放到根号里面（$\sqrt{\hat v+\epsilon}$）数值不一样：对小梯度更宽容，会把稀疏方向的更新压小。项目按外面的约定，判据的容差也照它设 —— 换写法不会崩，只是单步幅度对不上 $\mathrm{lr}$ 那个精确等式。
4. 梯度为 0 时必须**不动**（$m=v=0\Rightarrow 0/(0+\epsilon)=0$）。如果你为了"防除零"在分子也加了个 eps，参数会自己漂移 —— 判据 `梯度为 0 时参数一动不动` 就是数这个。

**为什么 `beta2=0.95` 而不是默认 0.999**（`__init__` 里已经给你写了）：本项目只有几百步，0.999 的二阶矩记忆太长，前期估计不准；0.9~0.95 收敛更快。真实训练（几万步）才反过来用 0.95/0.98 之类。

<details>
<summary>参考答案 · TODO-22 <code>AdamW.step</code></summary>

```python
    def step(self, lr: float | None = None) -> float:
        lr = self.lr if lr is None else float(lr)
        self.step_count += 1
        t = self.step_count
        # 偏差修正：初始 m、v 都是 0，前几步的均值被严重低估，要除以 (1-β^t)
        c1 = 1.0 - self.b1 ** t
        c2 = 1.0 - self.b2 ** t
        for p, mi, vi in zip(self.params, self.m, self.v):
            self._decay(p, lr)
            g = p.grad
            mi[...] = self.b1 * mi + (1.0 - self.b1) * g
            vi[...] = self.b2 * vi + (1.0 - self.b2) * g * g
            p.data -= lr * (mi / c1) / (np.sqrt(vi / c2) + self.eps)
        return lr
```

`self._decay(p, lr)` 是基类给的解耦衰减：`if self.weight_decay and not p.wd_exempt: p.data *= (1 - lr*self.weight_decay)`。**先衰减再走 Adam 那一步**是 AdamW 原论文的次序（两者相差一个 $O(\mathrm{lr}^2\lambda)$ 项，无关紧要，但次序要固定）。判据 `weight decay 只作用于非豁免参数` 用 $w=0.5$ 手算了这个和：$0.01+0.5\times0.01\times0.1=0.0105$，正好是实测值。

</details>

下面这张图把 SGD / Momentum / AdamW 三条轨迹放在同一个二次目标上真算迭代，滑杆拖 `lr`、`β₁`、`β₂`、`wd`；三个叠画开关分别看"漏掉偏差修正""把 wd 加进梯度里""bias/norm 不豁免"各自会怎样。

@@fig adamw@@

再把视角拉远一点：优化器不是在抽象地"更新参数"，它是在一张真的 loss 曲面上下山。
下面这张 3D 曲面当场算出来（41×41 网格），小球按你选的优化器和 `lr` 一步步滚；
把 `lr` 拖大看它在谷里来回跳，拖小看它走得慢 —— 等高线插图是同一张曲面的俯视图。

@@fig losssurf@@

---

## 12. TODO-23：`clip_grad_norm`

**契约**：`clip_grad_norm(target, max_norm) -> float`，返回**裁剪前**的全局 L2 范数。

$$g=\sqrt{\sum_{p}\sum_{i} \bigl(p.\mathrm{grad}_i\bigr)^2},\qquad
  \text{若 } g>\text{max\_norm}:\ \forall p,\; p.\mathrm{grad}\ \mathrel{*}=\ \frac{\text{max\_norm}}{g}$$

**三个点**：

1. **全局**范数、**同一个** scale。逐张量各裁各的话，参数量大的层（嵌入表、FFN）相对权重被改变，等于悄悄改了各层的步长配比。实现上就是：先扫一遍算出 `total`，再扫一遍乘同一个 `scale`。
2. 返回裁剪**前**的范数。训练日志要它 —— `grad_norm` 突然抬头是发散的先行指标（阶段 6 的判据里 `grad_norm 中位数/最大值` 这条 info 就来自这里）。返回裁剪后的话你永远看不到尖刺。
3. `g == 0` 时不要除。全零梯度是合法的（比如整个 batch 恰好都被忽略），`total > max_norm and total > 0.0` 那个额外条件就是防 `0/0`。

**顺序**：`forward → backward → clip → step`。写在 `step()` 之后等于裁了一份已经被用掉的梯度 —— 不报错，只是永远没生效。

这一节的核心只有一句：**全局**算一个范数、所有张量乘**同一个** scale，所以方向一点都不动。
下面这张图给了 6 个参数张量的梯度：拖 6 根长度滑杆和 `max_norm`，总范数条与阈值线当场算，
裁剪前的虚线箭头和裁剪后的实线箭头叠画在一起，六个夹角读数恒为 0.000° ——
那就是「只缩长度、不改方向」的现场证据。三个错法开关正好对应上面那三个点：逐张量各裁各的
（总范数仍然超阈值）、先裁再算范数（等于没裁）、无条件乘 scale（把本来不大的梯度放大）。

@@fig clipnorm@@

<details>
<summary>参考答案 · TODO-23 <code>clip_grad_norm</code></summary>

```python
    params = _as_params(target)
    total = math.sqrt(float(sum(np.sum(p.grad * p.grad) for p in params)))
    if total > max_norm and total > 0.0:
        scale = max_norm / total
        for p in params:
            p.grad *= scale
    return total
```

`p.grad * p.grad` 而不是 `p.grad ** 2` 再 `np.abs`：梯度是实数，平方和就是 L2；写成 `np.linalg.norm(p.grad)` 逐张量算再相加是**错的**（那是 $\sum_p\|g_p\|$，不是 $\sqrt{\sum_p\|g_p\|^2}$）。

</details>

---

## 13. TODO-24：`lr_at`

**契约**：`lr_at(step, total_steps, peak, warmup=400, final_frac=0.1) -> float`。

$$\mathrm{lr}(s)=
\begin{cases}
\mathrm{peak}\cdot\dfrac{s+1}{\mathrm{warmup}}, & s<\mathrm{warmup}\\[8pt]
\mathrm{peak}\cdot\Bigl[f+(1-f)\cdot\dfrac{1+\cos\bigl(\pi\frac{s-\mathrm{warmup}}{\mathrm{total}-\mathrm{warmup}}\bigr)}{2}\Bigr], & \text{否则}
\end{cases}$$

* **`+1`**：写 `s/warmup` 时第 0 步 lr 正好是 0 —— 那一步白走（`opt.step(0)` 仍然会执行衰减，于是第一步是**纯衰减**，权重被无谓地缩小了一点）。判据 `第 0 步的 lr 不是 0`。
* **夹在 $[0,1]$**：`step` 会走到 `total_steps`（日志/判据常按 `range(steps+1)` 取样），不夹就出现 $\cos(1.001\pi)$ 这种"略微回升"。
* **`total_steps <= warmup` 单独返回 peak**：否则分母 `total-warmup` 是 0 或负数。判据里 `short_ok` 就在测这个角落。
* **尾端留 10% 而不是归零**：后期还需要一点步长继续微调；归零会让模型卡在离最优点还有一小段的平台上。

**warmup 存在的理由**（不是惯例，是 Adam 的性质）：§2.5 说前几十步 $\hat v$ 估不准。$\hat v$ 偏小时更新量被放大，这时用大 lr 容易把第一层权重打飞。所以"warmup 期"和"偏差修正"是同一个问题的两个解药 —— 都写在你已经实现的代码里。

<details>
<summary>参考答案 · TODO-24 <code>lr_at</code></summary>

```python
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if total_steps <= warmup:
        return peak
    frac = (step - warmup) / max(1, total_steps - warmup)
    frac = min(1.0, max(0.0, frac))
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return peak * (final_frac + (1.0 - final_frac) * cos)
```

`max(1, ...)` 是给"分母至少是 1"的第二重保险（前面的 `total_steps <= warmup` 已经拦住了退化情形）。这种"两处都防"的写法看起来啰嗦，但 lr 函数被训练循环每步调用一次，一旦 NaN 会毁掉整轮训练，值得多一行。

</details>

warmup 到底在防什么，下面这张图叠了两条线：`lr_at(i)` 本身，和前 30 步每步的权重更新幅度 $\|\Delta w\|$ —— 一上来就给满 lr 时更新幅度 spikes 到离谱；再给一组"有 warmup vs 满 lr"的 loss 对照。

@@fig lrsched@@

---

## 14. 训练循环：把这些串起来（已给你，写在判据脚本里）

```python
for i in range(STEPS):
    opt.zero_grad()                                  # ① 清零（参数梯度是累加的）
    x, _y = batcher(B, rng)                          # ② 随机窗口
    curve.append(model.loss_backward(x))             # ③ 正向 + 反向
    clip_grad_norm(model, 1.0)                       # ④ 裁剪
    opt.step(lr_at(i, STEPS, LR, warmup=WARMUP))     # ⑤ 更新，lr 由调度决定
```

**为什么 `zero_grad` 在循环里而不是在 `loss_backward` 里**：梯度累加是**约定**（`nn.py` 头部第 4 条），因为多条路径会汇到同一参数（tied 就是活例子），层自己清就会把别的支路的贡献抹掉。谁调 forward 谁负责清，所以是训练循环的活。

**为什么这个循环写在判据脚本里而不是给你当 TODO**：它没有新数学，而且顺序错了的表现（"loss 在降但很慢"）最难归因。你在阶段 1 已经手写过一个训练循环（TODO-1A/2B），这里的六步和那里一一对应：`zero_grad ↔ 手工把 grad 归零`、`step(lr_at(...)) ↔ a.data -= lr * a.grad`。看懂它、能背出来就够了；真要有意识地练一次，是阶段 6 的判据脚本里那个加长版（多了 grad_norm 记录）。

**判据脚本里还有一个"单批过拟合"**：同一批数据（`fixed_batch`）训 250 步，loss 必须 $\le 0.05$。这是最狠的一条，因为它把所有"能训"的可能性排除掉：数据固定 → 不存在泛化问题；模型 5.9 万参数 vs 4×32 个 token → 背下来毫无压力。背不下来只能是**机器坏了**（反向错、移位错、优化器不动）。`docs/advanced-cpp-roadmap.md §7.1` 把它列为第一条诊断，理由就是"能抓住 90% 的实现 bug"。

---

## 15. 验收判据逐条解读

`python stages/s05_char_lm.py`，36 条硬判据 + 2 条软提示，分五组：

**分词组（4 条）**
| 判据 | 查的是 |
|---|---|
| 往返一致 `decode(encode(x)) == x` | TODO-20，两个方向都要对 |
| 词表口径 / 编码结果全在词表内 | `vocab` 的定义、id 范围 |
| 语料外的字符按契约处理 | 默认丢弃 + `drop_unknown=False` 抛错 |

**数据组（5 条）**
| 判据 | 查的是 |
|---|---|
| 批次形状 (B, ctx) | 窗口切片 |
| `y[:, :-1] == x[:, 1:]` | 移位关系写成数据 |
| 随机窗口真的随机且可复现 | `rng` 的传法、`max_start+1` |
| train/val 按行切、不重叠 | `train_val_texts`（给你的那半，用一份每行唯一的语料测） |

**整机梯度组（8 条）**
| 判据 | 查的是 |
|---|---|
| Block 反向正确 | TODO-15/16 + 阶段 3/4 的一切 |
| 残差出口置零后 Block 是恒等映射 | forward 漏加 / backward 漏透传 |
| tied 输出头反向正确 | TODO-17 的数学 |
| tied 的两路梯度是累加不是覆盖 | `+=` vs `=`（本组唯一能抓它的一条） |
| GPT 整机反向正确(含权重共享) | 整链，包括 `GPT.backward` 那 6 行 |
| 无死参数 / 参数量与手算一致 / 输出头复用同一对象 | 装配：`pos` 参与了吗、多算了一份头吗 |

这一组用 `eps=1e-5, tol=1e-4, atol=1e-8`：整机的 `q.b` 梯度常常只有 `1e-6` 量级（大数相减的抵消结果），差分的舍入噪声 $\approx \mathrm{ulp}(目标)/\epsilon\approx10^{-11}$，相对误差看着像 `6e-3` 其实是噪声底。**1e-8 已经是量级上的让步**：真写错 backward 的误差是 `1e-2~1e-1`，不会被这条放过。

**训练组（7 条）**
| 判据 | 查的是 |
|---|---|
| 初始 loss ≈ ln V | 随机初始化的基准（§2.4） |
| 单批过拟合最终 loss ≤ 0.05 | 反向 + 优化器 + 移位三件事同时成立 |
| 训练 loss 掉到 ln V 的零头 / 下降倍数 ≥ 5× | 真训得动 |
| 验证 loss ≤ 0.35 ln V | 按行切出来的 val 上也能降 |
| 贪心下一字符准确率 ≥ 0.85 | 模型确实知道下一个字符 |
| **随机数据学不动（反作弊）≥ 0.9 ln V** | **移位方向**（§8.2） |

**优化器组（10 条）**
| 判据 | 查的是 |
|---|---|
| AdamW 单步幅度 = lr | §2.5 偏差修正 |
| weight decay 只作用于非豁免参数 / 梯度为 0 时只衰减不推进 | 解耦衰减 + `wd_exempt` |
| clip 返回裁剪前范数 / 裁后 = 上限 / 同一系数 / 没超限不动 | 全局 L2 的四条性质 |
| lr 单调性 / 不超 peak / warmup 末到 peak / 末端 10% / step0 不为 0 | §13 的分段定义 |

软提示两条不计入通过率：`train/val 没有明显裂缝`（1.4K 字符的语料背得下来，别把背下来当学会）和 `loss=0 测试` 的对照说明。

---

## 16. 常见失败模式对照表

| 症状 | 病因 | 修法 |
|---|---|---|
| `残差出口置零后 Block 是恒等映射` 挂 | `norm2` 吃了 `x`，或 return 写成 `x + m` | §4 那五行照抄一遍 |
| `Block 反向正确` 挂在 norm 的 `w` 上、mlp 全对 | backward 漏了 `+ g`（只交支路） | §5 两处 `+` 都要 |
| `GPT 整机反向正确` 挂在 `tok.weight` | TiedHead 写了 `=` 而不是 `+=` | §6；单项快测会直接指出 |
| `无死参数` 挂：`pos.weight` | `GPT.backward` 里漏了 `self.pos.backward(g)` | 那是给你的代码，检查你有没有在 TODO-19 里改坏 |
| 单批过拟合 loss 卡住不动 | lr 太小 / `step()` 没被调 / `zero_grad` 位置错 | 先在循环里打印第一次更新前后的参数变化 |
| 初始 loss 明显小于 ln V | 移位写成"预测自己"，或 targets 用了同一批 logits | §8，判据 `随机数据学不动` 会一起红 |
| 随机数据上 loss 掉到接近 0 | 同上（这是它的**唯一**症状） | 只改一处：`ids[:, 1:]` |
| 前 200 步 loss 几乎不动，之后正常 | AdamW 忘了偏差修正 | §11 那个 $c_1,c_2$ |
| loss 一开始就 `nan` | `step_count += 1` 位置错 → 除零；或 eps 放进了根号 | §11 |
| lr 一直是 0 / warmup 没到 peak | `step/warmup` 少 `+1`，或 `frac` 没夹区间 | §13 |
| `参数量与手算一致` 差 `V·C` | TiedHead 自己新建了一份权重 | §3.2：传对象，别传 `.data` |
| `Batcher` 窗口总数少 1 / IndexError | `max_start` 的 `-1`，或 `rng.integers` 上界写成 `max_start` | §10 |
| `词表口径` 挂且第一个字符丢了 | 用 `if not i:` 判断"没找到" | §9：`if i is None:` |
| 每一步吃的 batch 一模一样 | `rng` 在 `__call__` 里新建 | §10 |

---

## 17. 自查题

1. 把一个 block 的 `attn.proj.W` 和 `mlp.fc2.W` 全置零，forward 和 backward 各自应当交出什么？为什么这两条合起来比一次有限差分更能定位残差 bug？
2. 为什么"预测自己"的实现在模式语料上**也**能拿高分？写出判据里是哪一条把它抓出来的，并说明这条判据的"通过"为什么看起来像"失败"。
3. `tok.weight` 的两条梯度来源分别来自哪个函数的哪一行？如果 `Embedding.backward` 用 `=`、`TiedHead.backward` 用 `+=`，整机梯度会是什么结果？
4. 手算一遍：$g$ 恒定、$\beta_1=0.9$、$\beta_2=0.95$ 时第一步和第二步的更新幅度各是多少？忘了偏差修正时呢？
5. `clip_grad_norm` 为什么必须跨张量算全局范数？给出一个"逐张量裁剪"会让哪一层的相对步长被改变的算术例子。
6. warmup 和 Adam 的偏差修正在解决同一个问题，具体是什么问题？如果只留一个，你会留哪个，为什么？
7. `Batcher` 返回的 `y` 在训练里根本没用到，为什么要它？（两个理由，并说出哪一个更接近判据的本质。）
8. 阶段 3 里 CrossEntropy 的梯度是 `(probs - one_hot)/N`。这里 $N$ 是什么？把它写错会伪装成哪个别的 bug？

---

## 18. 下一步

阶段 6：`checkpoint.py`（TODO-25~27）—— 把训好的权重写成 MGPT 二进制格式，让另一个进程（或 C++）读走；再把语料换成真的 `data/corpus.txt`，并且做一次"断了能续"的对账。

顺便一提：写完这一章，`tests/test_model.py`（阶段 4 的考卷）里 Block、整机梯度、因果性那三组才第一次真的跑在你自己写的代码上。回去跑一次，看它们从"靠答案版绿"变成"靠你绿"。

---

## 19. 进阶推导（现在可以跳过）

### A. 残差出口为什么乘 $1/\sqrt{2L}$

由 §2.1 的展开式，主干方差大致是

$$\operatorname{Var}(x_{\text{出}})\approx\operatorname{Var}(x_{\text{入}})+\sum_\ell\Bigl[\operatorname{Var}(\mathrm{attn}_\ell)+\operatorname{Var}(\mathrm{mlp}_\ell)\Bigr]$$

子层出口如果不缩小，每加一层就加一份方差，$L$ 层后方差长成 $1+L$ 到 $1+2L$ 的量级，末端 norm 之前的数值尺度会随深度漂，训练前期不稳。让**每个出口贡献 $\frac{1}{2L}$ 份方差**，$2L$ 个出口加起来正好是 1 份 → 主干长度与深度无关。这就是 `GIV-3` 数的那个比值应当正好是 $\sqrt{2L}$（$L=3$ 时 $\sqrt 6=2.4495$）的来历。

pre-norm 下"线性涨"、post-norm 下"指数涨"的完整递推，在 [阶段 3a §11 进阶推导 B](s03a.html#sec-11)；§3.3 的动画能切换这两种假设当场对比。
