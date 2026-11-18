# 阶段 4 · 因果自注意力：整个 transformer 里唯一的新数学

预计：约 35 行，2~4 小时。
验收命令：`python tests/test_attention.py`（只查这一卷；Block/GPT 整机在阶段 5 的 `test_model.py`）
要写的地方：`src/minigpt/attention.py` 的 TODO-12 / TODO-13 / TODO-14。
网站：选「阶段 4 · 因果自注意力」，提交 `minigpt/attention.py`。

---

## 1. 这一步在干什么

前面的层都是**逐位置**的：第 $t$ 个 token 的输出只看它自己。注意力让 token 之间互相看一眼：
**每个 token 提出一个查询，和所有（允许的）token 的键比相似度，按相似度加权收集它们的值。**

$$s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_h}}
  \;\longrightarrow\; a_{ij}=\mathrm{softmax}_j(s_{ij})
  \;\longrightarrow\; \mathrm{out}_i=\sum_j a_{ij}\,v_j$$

"因果"= 位置 $i$ 只允许看 $j\le i$。这一条是自回归生成的根本原因：训练时若允许看后面，
模型直接抄答案，loss 低得离谱、生成时一塌糊涂。

按这个顺序写，每一步都能停下来跑判据：

```flow
**① 扫一眼 §2 的五条决定**
   ↓ 写的时候回来查表
**② TODO-12 `forward`**
   ↓ 停点：`python webapp/checks.py TODO-12` 绿
**③ TODO-14 `softmax_backward`**
   ↓ 停点：`python webapp/checks.py TODO-14` 绿
**④ TODO-13 `backward`**
   ↓ 停点：`python tests/test_attention.py` 11 条全绿
```

---

## 2. 五条决定（写之前扫一眼，写完回来对）

| 决定 | 值 | 写错会怎样 |
|---|---|---|
| 几个投影 | q/k/v/proj 四个 `Linear(C, C)`；q/v/proj 有 bias，k **没有** | 参数量判据红（按 $4d^{2}+3d$ 算）；k 的 bias 是死参数 |
| 缩放 | $\text{scale}=1/\sqrt{d_h}$，$d_h=C/H$ | 写成 $1/\sqrt{C}$：$H=1$ 时全对，多头时分数系统性偏小 |
| 掩码位置 | 加在 **softmax 之前** | 加在之后：分母被污染，因果性判据红 |
| 掩码值 | `NEG_INF = -1e18` | 用 `-inf`：反向 $0\cdot\infty=\mathrm{nan}$ |
| 合头之后 | 必须再过 `proj` | 多头之间不交换信息，"多头"只多了参数 |

每条为什么是这个数、这个顺序，见 §11。

---

## 3. 已给你的代码

| 给你的 | 它唯一的用途 | 读的时候看什么 |
|---|---|---|
| `_split_heads` | `(B,T,C) → (B,H,T,hd)` | 先 `reshape(B,T,H,hd)` 再 `transpose(0,2,1,3)` |
| `_causal_mask` | 返回 `(1,1,T,T)`，上三角 `-1e18` | `np.triu(..., k=1)` 保留对角线 |
| `_softmax_rows` | 每行减最大值再归一化 | 和阶段 3 `CrossEntropy` 那三行同构 |
| `self.k` 的 `use_bias=False` | k 没有 bias | 参数量按 $4d^{2}+3d$ 算，不是 $4d^{2}+4d$ |

拆头/合头为什么必须先 `reshape` 再 `transpose`、`triu` 的 `k=1` 是什么、
$b_q$ 为什么活着而 $b_k$ 死了、`proj` 到底在混什么 —— 见 §11.4。

---


## 4. TODO-12：`forward`

**契约**：`(B,T,C) → (B,T,C)`。中间量缓存：`self._q/_k/_v/_probs/_merged_shape`。

六步（`attention.py` 的 TODO 注释里逐步写了形状，照着对）：

```
q,k,v: _split_heads(Linear(x))                    (B,H,T,hd)
att  : (q @ k.transpose(0,1,3,2)) * scale         (B,H,T,T)
att  : att + _causal_mask(T, x.dtype)             ← 加在 softmax 之前
probs: _softmax_rows(att)                         每行和 = 1
y    : probs @ v                                  (B,H,T,hd)
out  : proj( y.transpose(0,2,1,3).reshape(B,T,C) )
```

写的时候三个自查点：

1. **`assert` 保留**：`C == self.C` 那句能帮你及早发现配置串错。
2. `transpose` 的轴序：`(B,T,H,hd) ↔ (B,H,T,hd)` 只交换第 1、2 轴（即 `0,2,1,3`）；
   算分数时转置的是最后两轴（`0,1,3,2`）。这两个是**不同的**转置，别混。
3. 缓存必须在 forward 里做。`backward` 不许重新算一遍 q/k/v（重算会用当时的参数，
   而参数在梯度累积里可能已经变了；而且浪费一倍时间）。

**写完先自查形状**（比跑整卷快得多，在项目根目录跑）：

```
python webapp/checks.py TODO-12
```

它会打印 `输入 x=shape(2,5,8) …`、`输出 y=shape(2,5,8)`、`_probs=shape(2,2,5,5)`，
最后一行结论期望是 `PASS 形状 (2,5,8)/(2,2,5,5)；行和偏差 …；上三角概率全为 0：True`。
如果 `_probs[0,0]` 的第 i 行前缀和不是 1，说明掩码没生效（第 i 行只允许 i+1 个格子）。

点某一行 query 看它把 1.0 分给了哪些 key；拖 $d_h$ 看 $\div\sqrt{d_h}$ 把分数方差压回 1 附近。

@@fig attnflow@@

<details>
<summary>参考答案 · TODO-12 <code>forward</code>（判据没过之前别点开）</summary>

```python
    def forward(self, x):
        B, T, C = x.shape
        assert C == self.C, f"输入通道 {C} 与配置 d_model={self.C} 不符"
        q = self._split_heads(self.q(x), B, T)
        k = self._split_heads(self.k(x), B, T)
        v = self._split_heads(self.v(x), B, T)
        self._q, self._k, self._v = q, k, v          # 反向要用，必须缓存拆好头的版本

        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale        # (B,H,T,T)
        att = att + self._causal_mask(T, x.dtype)               # 掩码加在 softmax 之前
        self._probs = _softmax_rows(att)
        y = self._probs @ v                                     # (B,H,T,hd)
        merged = y.transpose(0, 2, 1, 3).reshape(B, T, C)        # 合头：拆头的逆操作
        self._merged_shape = (B, T, C)
        return self.proj(merged)
```

`self._q/_k/_v` 存的是**拆头之后**的 $(B,H,T,hd)$，不是原始 $(B,T,C)$ ——
反向里 $g_q = g_{att}k\cdot\text{scale}$ 用的就是这个形状，存错了就得再拆一次（还容易拆错轴）。
合头那行是"先转置再摊平"，和 §11.4 里拆头的顺序正好相反。

</details>

---

## 5. TODO-14：`softmax_backward`

已知每行 `p`（和为 1）和对它的梯度 `gp`，求对分数 `z` 的梯度：

$$\boxed{\;g \;=\; p \odot \bigl(\mathrm{gp} - \sum_{\text{last}}(\mathrm{gp}\odot p)\bigr)\;}$$

求和沿最后一维，`keepdims=True` 才能广播回去。写完用两条性质自查，不用等 gradcheck：

- 给整行 `gp` 同一个常数 → 结果必须精确为 0。softmax 对整行加常数不变，梯度就必须是 0，
  这和 k-bias 是死参数是同一个性质。
- 正向是均匀分布时 $g=\frac1n(\mathrm{gp}-\overline{\mathrm{gp}})$，就是"每行减均值"。

那个减项的推导、以及它和阶段 3 RMSNorm 里 `s` 的同源关系，见 §11.5。

拖「锐度」换一组 p、拖四根 `gp` 滑杆，`s = Σ p·gp` 与 `ds = p ⊙ (gp − s)` 当场算；
正确实现下常驻的 `Σ ds` 读数是 1e-17 量级，打开错法开关就变非零。

@@fig softmaxbwd@@

<details>
<summary>参考答案 · TODO-14 <code>softmax_backward</code></summary>

```python
def softmax_backward(probs: np.ndarray, g_probs: np.ndarray) -> np.ndarray:
    s = np.sum(g_probs * probs, axis=-1, keepdims=True)
    return probs * (g_probs - s)
```

少了 `s` 那一项 = 直接 `return g_probs * probs`：loss 照样降，gradcheck 挂在 q/k 上、
相对误差 $10^{-1}$ 量级。

</details>

---

## 6. TODO-13：`backward`

结构上是 §4 的镜像。四组乘法，注意每一组的轴序：

```
g_merged = proj.backward(g)                                   (B,T,C)
gy       = g_merged.reshape(B,T,H,hd).transpose(0,2,1,3)      (B,H,T,hd)

g_probs  = gy @ v.transpose(0,1,3,2)                          (B,H,T,T)   ← y=probs@v 对 probs
g_v      = probs.transpose(0,1,3,2) @ gy                      (B,H,T,hd)  ← 对 v
g_att    = softmax_backward(probs, g_probs)                   (B,H,T,T)
g_q      = (g_att @ k) * scale                                (B,H,T,hd)
g_k      = g_att.transpose(0,1,3,2) @ q * scale               (B,H,T,hd)

返回 q.backward(g_q合头) + k.backward(g_k合头) + v.backward(g_v合头)      (B,T,C)
```

三个必须想清楚的点：

1. **为什么 `g_att` 不用清掉禁止位置**：那里 `probs` 精确为 0，所以
   `probs * (g_probs − s)` 就是 0。加 `np.where(mask==0, 0, g_att)` 不但多余，
   而且**条件极容易写反**（`mask==0` 是**允许**的位置！）。写反的症状：loss 照降、
   但降得很慢、gradcheck 挂在 q/k 上 —— 本项目真踩过这一条。
2. **`scale` 出现两次**：正向 `att = (q@kᵀ)*scale`，反向时 q 支路和 k 支路各要乘一次。
3. **返回值相加**：`q/k/v` 三个 Linear 都以同一个 `x` 为输入，三条支路的 `dx` 要相加。
   参数梯度已经在各自 `backward` 里累加完成，这里只处理输入梯度。

<details>
<summary>参考答案 · TODO-13 <code>backward</code></summary>

```python
    def backward(self, g):
        g_merged = self.proj.backward(g)                         # (B,T,C)
        B, T, C = self._merged_shape
        gy = g_merged.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)   # (B,H,T,hd)

        probs, v = self._probs, self._v
        g_probs = gy @ v.transpose(0, 1, 3, 2)                   # y = probs @ v
        g_v = probs.transpose(0, 1, 3, 2) @ gy

        # 被掩掉的格子正向已经是 exp(-1e18)=0，梯度天然为 0，这里不要再 where。
        g_att = softmax_backward(probs, g_probs)

        g_q = (g_att @ self._k) * self.scale
        g_k = g_att.transpose(0, 1, 3, 2) @ self._q * self.scale

        g_q_flat = g_q.transpose(0, 2, 1, 3).reshape(B, T, C)
        g_k_flat = g_k.transpose(0, 2, 1, 3).reshape(B, T, C)
        g_v_flat = g_v.transpose(0, 2, 1, 3).reshape(B, T, C)
        return (self.q.backward(g_q_flat)
                + self.k.backward(g_k_flat)
                + self.v.backward(g_v_flat))
```

三条最容易漏的：`scale` 在 q 支路和 k 支路**各乘一次**（正向只有一个 `*scale`，
但它同时作用于 $qk^{\top}$，两条支路各自求导都要带上）；三个子模块的返回值要**相加**
（它们吃的是同一个 `x`，参数梯度已经在各自 `backward` 里累加完了）；
`g_k` 用的是 `g_att` **转置**后 @ `q` —— 形状对不上时先别改括号，回去对 §4 那六行的轴序。

</details>

---

## 7. 这一卷的判据

```checks
judge: tests/test_attention.py
```

红了的去向：softmax 两条 → §5 的公式；单头/多头 gradcheck → §4、§6 的轴序；
因果三条 → §2 「掩码位置」那一行；参数量 → §2 「几个投影」那一行。
Block/GPT 整机在阶段 5 的 `tests/test_model.py`，不归这一卷。

改掉一个未来 token，逐位比较过去位置的输出：掩码对时一个比特都不变，写成"允许看未来"立刻渗回过去。

@@fig causalfix@@

---

## 8. 常见失败模式对照表

| 症状 | 病因 | 修法 |
|---|---|---|
| gradcheck 挂在 q/k，v/proj 全对 | 少了 `softmax_backward` 的减项 | §5 公式重抄，推导在 §11.5 |
| 所有梯度偏小、loss 降得慢 | `np.where` 条件写反，清掉了允许位置的梯度 | 删掉 where（本来就不需要） |
| `nan` 一路传染 | 掩码用了 `-inf` | 用 `NEG_INF=-1e18` |
| 因果性判据挂（前缀变了） | 掩码加在 softmax 之后，或根本忘了加 | 加在 softmax **之前** |
| `cannot reshape array of size ...` | 合头时 transpose 轴序错 | `(B,H,T,hd) → (B,T,H,hd)` 用 `0,2,1,3` |
| gradcheck 挂在 proj 之后的某个参数 | forward 缓存没存（backward 里重算 q/k/v） | 缓存 `self._q/_k/_v/_probs` |
| 无死参数挂：`attn.k.b` | 给 k 加了 bias | 用 `use_bias=False`（判据的期望参数量按 10d 算） |
| 单头对、多头错 | 拆头/合头顺序反了（`reshape` 与 `transpose` 换了次序） | 用 §4 的自查命令看 `_probs` 每行和 |
| `参数量与手算一致` 差 C | 上面那个 bias，或某个 Linear 忘了关 bias | 数一遍：q/v/proj 有 bias，k 没有 |

---

## 9. 自查题

先把答案写进输入框再点"判一下"；卡住按提示 1→2→3，还不行点"回讲义读这一节"。

```quiz
q: 为什么 `att` 必须是 $(B,H,T,T)$ 而不是 $(B,T,T,H)$？两种写法都能算，差别在哪儿？
keys: 批量矩阵乘/一次算/每个头一张; 最后一维/沿key/加权; 合头/先转置再摊平
hint1: 哪一种排法能让每个头自己是一张 $T\times T$，批量矩阵乘直接可用？
hint2: `probs @ v` 那一步沿最后一维加权，key 维必须在最后。
hint3: $(B,H,T,T)$ 的最后一维是 key，`probs @ v` 一行搞定；头在中间则每次都要 transpose。合头 $(B,H,T,hd)\to(B,T,C)$ 恰好也是"先转置再摊平"。
link: sec-4
ans: 最后一维要是 key，`probs @ v` 才能直接沿它加权并批量广播；头放最后则每步都要转置，合头时还要把两个维度换回来。$(B,H,T,T)$ 让拆头/算/合头三处轴序统一。
```

```quiz
q: 一句话说清 $b_k$ 为什么是死参数，它和 `softmax_backward` 里那个减项是同一个性质的两种表现 —— 这个性质是什么？
keys: 整行加常数/平移不变/加同一个数; 每行和恒为1/守恒; 梯度为0/不影响输出
hint1: 给一个位置的分数整体加常数，softmax 输出变吗？
hint2: k 的所有维同时加 $b$，等价于每个分数都加同一个数。
hint3: 性质是"softmax 对整行平移不变"，等价说法是"每行和恒为 1"；所以 $b_k$ 的梯度恒为 0，减项做的就是减掉那份不改变输出的方向。
link: sec-11-5
ans: 性质是 softmax 对整行加常数不变（每行和恒为 1）。$b_k$ 的所有维加上去等于给每个分数加同一个数，输出不变 → 梯度恒为 0；softmax 反向里减掉 $p\odot\sum(\mathrm{gp}\odot p)$ 正是减掉这份"不改变输出"的方向。
```

```quiz
q: 因果掩码加在 softmax 之后会怎样？
keys: 分母/归一化被污染; 未来位置概率非零/不由0; 因果性失效/偷看
hint1: softmax 的分母是哪些格子的指数和？
hint2: 先归一化再置零，剩下的格子之和不再是 1，得重新分配。
hint3: 分数不变，但分母含被掩位置；置零后要重新归一化，未来位置的权重被摊到过去格子上 —— 因果性失效。
link: sec-11-3
ans: 分数不变、分母被污染：把上三角置零后每行和不再为 1，得重新归一化，未来 token 的权重被摊回过去格子，因果性失效。
```

```quiz
q: 多头数 $H$ 变大、`d_model` 不变，每个头的 $hd$ 会怎样？影响哪一步的耗时和梯度尺度？
keys: 变小/C/H; 缩放1/sqrt(hd)变大; 分数方差/更尖; 参数量不变
hint1: $d_h=C/H$。
hint2: 缩放因子 $\sqrt{d_h}$ 跟着变小，分数的方差会怎样？
hint3: $d_h$ 变小 → 缩放 $1/\sqrt{d_h}$ 变大、分数更容易饱和；参数量只由 $C$ 决定，与 $H$ 无关。
link: sec-11-2
ans: $d_h=C/H$ 变小：$\div\sqrt{d_h}$ 变小（缩放变强），每头分数方差被压得更平，softmax 更接近均匀 —— 该头的表达能力下降、梯度尺度变小。参数量只随 $C$ 变，与 $H$ 无关。
```

```quiz
q: 把 $\sqrt{d_h}$ 换成 $d_h$ 会发生什么？从方差角度说。
keys: 方差被压到1/d_h / 过度缩小; softmax变尖/饱和/梯度消失
hint1: 先算 $\mathrm{Var}(q\cdot k)$。
hint2: 除 $\sqrt{d_h}$ 后方差是 1；除 $d_h$ 呢？
hint3: $q\cdot k$ 的方差是 $d_h$，除 $\sqrt{d_h}$ 归一为 1；除 $d_h$ 后降到 $1/d_h$，softmax 过度平坦，梯度信号被摊薄。
link: sec-11-2
ans: $q\cdot k$ 的方差约 $d_h$。除 $\sqrt{d_h}$ 把方差压到 1；除 $d_h$ 后方差只剩 $1/d_h$，softmax 过度平坦，权重几乎均分、梯度被摊薄，模型学不动。
```

```quiz
q: 自己推一遍 $\mathrm{Var}(q\cdot k)=d_h$（假设每维独立、零均值、方差 1）。
keys: 求和/内积; 方差可加/独立; d_h项/每项1
hint1: 内积是 $d_h$ 项的求和。
hint2: 独立随机变量之和的方差等于方差之和。
hint3: 每项 $q_i k_i$ 方差为 1，共 $d_h$ 项且两两独立 → 总和方差 $d_h$。
link: sec-11-2
ans: $q\cdot k=\sum_{i=1}^{d_h}q_ik_i$；每项因独立零均值方差为 $\mathrm{Var}(q_i)\mathrm{Var}(k_i)=1$，各维独立故方差可加 → 总和方差 $d_h$，标准差 $\sqrt{d_h}$。
```

```quiz
q: 判据"改当前 token 确实改变当前输出"如果失败，说明模型出了什么问题？此时"改未来 token 不影响过去"那条的通过还有意义吗？
keys: 常数/不依赖输入/输出不变; 没有意义/毫无意义/无效
hint1: 输出一个 bit 都不变，说明 forward 和输入无关。
hint2: 一个常数函数"当然"不会把未来泄漏到过去。
hint3: 模型退化成常数映射；一个常数函数天然满足因果性，所以那条判据白绿。
link: sec-7
ans: 说明 forward 退化成常数（不看输入），此时"不影响过去"必然成立但毫无意义 —— 这正是要配一条正面对照判据的原因。
```

```quiz
q: `forward` 里不缓存 `probs` 而是 `backward` 时重算，为什么错？两个理由。
keys: 参数可能已变/不一致; 浪费/两倍时间/多算一遍
hint1: 梯度累积时参数会动吗？
hint2: 重算时用的是哪一份参数？
hint3: ① 用当时的参数重算，和正向不是同一份数，梯度对不上；② 白算一遍，速度减半。
link: sec-4
ans: ① 重算用的是当前参数，参数在梯度累积里可能已变，`probs` 和正向不一致，梯度就错了；② 白白多算一遍前向，时间翻倍。
```

---

## 10. 下一步

阶段 5：`gpt.py`（Block + 权重共享输出头 + 标签移位）、`optim.py`、`tokenizer.py`、`dataset.py`。
那时你写的 Block 会把注意力接进残差结构，`loss_backward` 里的移位会成为新的判据主角
（`tests/test_model.py` 的"正确移位：新数据上学不动"一条，等你写完 `gpt.py` 才真正测到你）。

---

## 11. 进阶推导（现在可以跳过）

草稿先把这一卷写绿，再回来读"为什么是这个数、这个顺序"。

### 11.1 为什么是 q/k/v/proj 四个矩阵

| 投影 | 角色 | 拿掉会怎样 |
|---|---|---|
| $W_q$ | 提出查询 | 没有"问什么"，注意力退化成固定的键匹配 |
| $W_k$ | 生成索引 | 与 $W_q$ 必须分开，否则相似度只剩一个投影在决定 |
| $W_v$ | 携带内容 | 只能按相似度搬运原向量，"关注谁"和"取走什么"绑死 |
| $W_{proj}$ | 混合各头输出 | 各头并列而不交流，"多头"只多参数、不多能力 |

索引和内容必须是两种东西：若 `k` 与 `v` 共用一个投影，模型只能表达"和我相似的词"，
表达不了"我该关注的那个词携带什么"。

四个都是 `Linear(C, C)`，切头不改变计算，只是把 $H$ 个独立的低维注意力并排放进一个张量。
`k` 不要 bias，所以参数量是 $4C^{2}+3C$ —— 理由见 §11.4 ④。

### 11.2 除 $\sqrt{d_h}$ 是方差算术，不是玄学系数

$q\cdot k=\sum_{i=1}^{d_h}q_ik_i$，各维独立、零均值、方差 1：

$$\operatorname{Var}(q\cdot k)=\sum_{i=1}^{d_h}\operatorname{Var}(q_ik_i)=d_h
  \quad\Longrightarrow\quad \operatorname{std}(q\cdot k)=\sqrt{d_h}$$

| 写法 | 分数方差 | 后果 |
|---|---|---|
| 除 $\sqrt{d_h}$ | $1$ | 分数落在 $\pm 3$ 量级，softmax 有梯度 |
| 不除 | $d_h$ | $d_h=64$ 时分数动不动 $\pm 16$，softmax 近似 one-hot，反向梯度 $\approx 0$ |
| 除 $\sqrt C$ | $d_h/C=1/H$ | $H=1$ 时全对，$H>1$ 时分数被系统性压平 |

最后一行是最难自查的错：单头判据全绿、多头才露馅。项数是每个头自己的 $d_h$（$C=H\,d_h$），不是 $C$。

### 11.3 掩码：为什么必须在 softmax 之前，为什么是 $-10^{18}$

因果 softmax 的分母只该含 $j\le i$：

$$a_{ij}=\frac{\exp(s_{ij})}{\sum_{j'\le i}\exp(s_{ij'})}$$

掩码就是把 $j>i$ 从分母里踢出去，所以只能在 softmax **之前**动手。

- 加在之后（先归一化再乘 0）：每行和不再是 1，得重新归一化，等价于把未来格子的权重摊回
  过去格子上 —— 因果性失效。
- 用 `-inf`：至少有一处 $0\cdot\infty$ 就出 $\mathrm{nan}$，反向一路传染。
- 用 `-1e18`：$\exp(-10^{18}-\max)$ 在 float64 里精确为 0 → 概率精确 0 → 梯度精确 0，
  不需要任何清理。
- 清理反而是坑：`np.where(mask==0, 0, g)` 里 `mask==0` 是**允许**的位置，条件写反就把
  该留的梯度清掉了。

`(1,1,T,T)` 的两个 1 是 batch 和 head 维，让同一张掩码广播到所有样本、所有头。

### 11.4 已给代码里的六件事

**① `_split_heads`：先 `reshape` 再 `transpose`。**

```python
y.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)     # (B,T,C) → (B,H,T,hd)
```

`reshape` 不搬数据，只是把 `C` 重新解释成 `H` 段；`transpose` 只改轴的记账。反过来先
`transpose` 再 `reshape`：转置后的数组不连续，`reshape` 会**复制并按新顺序重排** —— 形状照样对、
数值是乱的。前向看着"也是个合理的向量"，只有 gradcheck 抓得住。合头是严格逆操作，backward
里对 $g_q/g_k/g_v$ 走同样两步。

**② `_causal_mask`：`np.triu(..., k=1)` 与 `(1,1,T,T)`。**

```python
m = np.triu(np.ones((T, T), dtype=dtype), k=1) * NEG_INF
return m.reshape(1, 1, T, T)
```

`k=1` 取严格上三角，对角线保留 —— 位置 $i$ 必须能看自己；写成 `k=0` 就变成"不许看自己"，
模型学成"预测前一个 token 的前一个 token"。乘 `NEG_INF` 后允许处仍是 0，加到分数上无影响。
`dtype` 跟着分数走：$-10^{18}$ 在 float32 上限（$3.4\times10^{38}$）之内，换精度不用改这一行。

**③ `_softmax_rows` 和阶段 3 的 `CrossEntropy` 是同一招。**

减最大值、取指数、除行和，三行完全同构；区别只在返回什么 —— 这里返回概率（要拿它加权求和），
CrossEntropy 返回 loss 并缓存概率。两处都靠 $\exp(-10^{18}-m)=0$ 把禁止位置干净地抹掉。

**④ $b_q$ 活着而 $b_k$ 死了：行与列的不对称。**

$$\frac{\partial L}{\partial b_q}=\sum_{i}\Bigl(\sum_j g_{ij}\,k_j\Bigr),\qquad
  \frac{\partial L}{\partial b_k}=\sum_{j}\Bigl(\sum_i g_{ij}\Bigr)q_i$$

softmax 的守恒性质给出每行 $\sum_j g_{ij}=0$，于是 $b_k$ 那条整个消失；$b_q$ 那条是不同 $k_j$
的加权和，抵消不掉。一句话：$b_k$ 给整行加同一个常数（softmax 不变），$b_q$ 改变行内各列的
相对分数（真变）。$b_v$ 也活着 —— 它作用在加权和的结果上。

**⑤ `proj` 是 $H$ 个头唯一的交流点。**

`reshape` 只把各头并排放好，没有任何信息交换。没有 `proj`，多头就等价于 $H$ 个互不相干的
低维网络；它同时把结果混回主干需要的 $C$ 维表示，所以也是残差出口缩放的对象。

**⑥ 合头必须过 `proj`，`scale` 用 $d_h$ 不用 $C$。**

两处都在 §2 的五条决定里，数字的来历见 §11.2；判据盯的是参数量与多头 gradcheck。

### 11.5 `softmax_backward`：那个减项从哪来

从定义出发：

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
  \;\Longrightarrow\; \frac{\partial p_i}{\partial z_k}=p_i\bigl(\delta_{ik}-p_k\bigr)$$

链式法则（记 $\mathrm{gp}_i=\partial L/\partial p_i$）：

$$\frac{\partial L}{\partial z_k}=\sum_i \mathrm{gp}_i\,p_i(\delta_{ik}-p_k)
  =p_k\Bigl(\mathrm{gp}_k-\sum_i \mathrm{gp}_i p_i\Bigr)$$

两个自查事实都不用等 gradcheck：

- $\mathrm{gp}$ 整行等于同一个常数 $c$：$g=p(c-c\sum p)=0$。给一行所有分数加常数不改变
  softmax，梯度就必须是 0 —— 和 $b_k$ 死参数同一个性质。
- 正向均匀 $p_i=1/n$：$g=\frac1n(\mathrm{gp}-\overline{\mathrm{gp}})$，就是"每行减均值"。

为什么漏减项难发现：`return g_probs * probs` 仍是一个"看起来对"的局部梯度，loss 照样降；
只有和有限差分逐格比，才在 q/k 上暴露 $10^{-1}$ 量级的相对误差。

它和 RMSNorm 里那个 $s$ 同源：两者都在减掉"被约束钉住"的方向。softmax 钉住每行和 $=1$，
RMSNorm 钉住每行长度 $=\sqrt C$，所以反向里都出现一个"减去本行加权均值"的项
（[阶段 3b §11.2](s03b.html#sec-11-2)）。
