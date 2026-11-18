# 阶段 6 · 真数据训练 + 检查点：让模型活得比这次运行更久

预计：1.5~3 小时（新内容只有存档；训练是把阶段 5 的循环拉长）。
验收命令：`python stages/s06_train.py`（19 条硬判据 + 2 条软提示，CPU 上约 10 秒）
要写的地方：`src/minigpt/checkpoint.py`（TODO-25、TODO-26、TODO-27）。
网站：选「阶段 6 · 真数据训练 + 检查点」，提交 `minigpt/checkpoint.py`。

---

## 1. 这一步在干什么

阶段 5 结束时你有一台会训练的机器，但它只活在那一次 `python` 进程里 —— 关掉终端，十万个参数就没了。这一阶段给它接上两个口：一份真数据、一块磁盘。

```flow
`data/corpus.txt`（64 行 · 4,061 字符 · 27 个不同字符）
│  阶段 5 的 tokenizer / Batcher / GPT（一字不改）
▼
真语料训练 300 步 ← **判据组①**：学得动，而且 val 没偷看
├─ `TODO-25 save_checkpoint` → `town.ckpt` ← **判据组②**：按公开格式排布
├─ `TODO-26 load_checkpoint` → 灌回新机 ← **判据组②的读端** + 四种拒绝
└─ `TODO-27 save/load_train_state`
   ├─ 训 100 步存档 → 读回一台新建的模型续 50 步
   └─ 和一口气训 150 步比：逐点最大差 ≤ 1e-4 ← **判据组③**：断了能续
```

一句话：**这一章的主角不是训练，是"协议"**。存档是你和"另一个进程"（未来的 C++ 版、崩溃后重开的自己）之间唯一的约定，而协议的正确性不能靠"我自己存、我自己读得回来"来证明。三件只有这一章会考的事：

1. **参考解析器**（§2.1）：判据故意不用你写的 `load_checkpoint` 来验你写的 `save_checkpoint`。
2. **字节账**（§2.2）：那个文件是 417,005 字节而不是 831,488 字节，说明每个数真的只占 4 字节 —— 判据把这笔账逐段加一遍。
3. **四元组** $\theta, m, v, t$（§2.4）：续训要恢复的是这四样，少任何一样，恢复点之后那条曲线就会跳。

第一次跑 `python stages/s06_train.py`（阶段 5 已写完的话）会**先把 300 步真语料训练跑完**，语料组 + 训练组那 8 条亮绿，然后停在 `TODO-25` 把已跑完的清单一并打出来。那不是坏掉，那是它认识你卡在哪。如果阶段 5 还欠着，它会先停在阶段 5 的那条上 —— 顺序执行，别跳。

---

## 2. 必要理论

### 2.1 跨语言只有一样东西靠得住：自己定的字节布局

`np.save` 存的是 pickle 包起来的 numpy 对象；`torch.save` 存的是框架的对象图（类路径、版本号、设备信息）。两者都要求**读的那一端有同一个框架**。本手册的最终目标是把同一套结构用 C++ 重写一遍（`docs/advanced-cpp-roadmap.md §6.4` 就是这份格式的 C++ 侧说明书），所以这里手写一个几十行的容器：字段、顺序、字节数全写在注释里，两边各读各的，不需要任何转换脚本。

由此得到这一章最重要的方法论：**协议的两端永远是对称写、不对称证的。** 写的那端"看起来对"、读的那端"看起来对"，合起来仍然可能是错的 —— 因为两边的错误可以互相包庇。

活例子：你把权重写成 float64，而你的 `load` 也按 float64 读，于是"存了能读回来"一路绿灯；C++ 侧照公开的表读 fp32，第一个张量之后全部错位，读出一堆看起来合理的垃圾。判据用一个**不认你代码的**解析器（`stages/s06_train.py` 里的 `parse_ckpt`，二十行，照着文件头那张字节表推进）就是为了打破这种包庇。

### 2.2 这张字节表为什么长成这样

```
[0, 4)     magic  b"MGPT"        优化器状态用 b"MGTR"，其余完全同构
[4, 8)     u32    version = 1
[8, 12)    u32    n              头 JSON 的字节数
[12,12+n)  bytes  config JSON    —— 就是 Config.to_dict()
然后按张量重复，直到 EOF：
    u32 name_len | name(UTF-8) | u32 ndim | i64[ndim] shape | fp32 data（行主序、连续）
```

每个设计都有一个"删了会怎样"：

* **长度前缀 + 读到 EOF**：文件是自描述的，不需要旁边再放一个 manifest。读侧每次只看"接下来 4 个字节是什么"，`f.read(4)` 给空就是记录结束。删掉长度前缀，你就得靠"读到解析失败为止"——那会把真正的截断错误伪装成正常结束。
* **先写 $n$ 再写 JSON**：读侧要先知道读几个字节才能解 JSON。这也让"头"和"张量区"有一条明确的边界。
* **小端**（`<I` / `<i8`）：x86 与 ARM 的原生字节序，C++ 侧 `memcpy` 就能读。写 native（不带 `<`）在个别平台上会给出另一种顺序，而且**在本机永远测不出来**。（张量名为什么也进文件：§2.6。）

于是文件总字节数可以手算（$\lvert n_t\rvert$ 是名字字节数、$d_t$ 是维数、$N_t$ 是元素个数）：

$$\text{size}=12+n_{\text{JSON}}+\sum_{t}\Bigl[\,4+\lvert n_t\rvert+4+8d_t+4N_t\,\Bigr]$$

阶段 6 那台模型（2 层、`d_model=64`、`ctx=40`、`vocab=27`，29 个张量、103,936 个参数）代进去：$12+187+1062+415744=417005$ —— 和判据打印的那个数字分毫不差。单项快测 `TODO-25` 里那个 `acct_ok` 就是在重演这笔账，所以多写一个字的 padding 都会红。

把这笔账画出来：四段按字节数成比例，点一段看它是怎么来的；按右边那个按钮换成 float64，
数据段立刻翻倍 —— 而"自己存、自己读"照样全绿，这就是 §2.1 那个陷阱的形状。

@@fig ckpt@@

### 2.3 权重降 float32，优化器状态留 float64

同一个容器、两种 dtype，理由完全不同，而判据两头都钉了（一条查 fp32、一条按 8 字节读完）。

* **权重 → fp32。** float32 的相对精度约 $2^{-23}\approx1.2\times10^{-7}$，也就是七位有效数字；而 `init_std=0.02` 量级的权重被量化后误差在 $10^{-9}$ 量级，穿过两层 block 放大到 logits 上实测 $1.6\times10^{-7}$ —— 判据 `读回来能算出一样的 logits` 的阈值给到 `2e-4`，留了两个数量级的余地。第 8 位有效数字对生成没有任何影响，但文件正好小一半，而且 C++ 侧全程用 float32 推理。
* **矩 → float64。** 优化器状态**只给自己读**，而且它服务的目标是"断点续训和一口气训完对得上"（§2.4）。$m,v$ 是直接进分母的：$\sqrt{\hat v}$ 若被量化到 $10^{-7}$ 的相对误差，恢复段每一步的更新量都带一点偏差，50 步攒下来就是 $10^{-3}$ 量级的 loss 差 —— 那条 `逐步吻合 ≤ 1e-4` 立刻红。规模感：同一台 `d_model=48` 的模型，权重文件 237,805 字节、状态文件 949,152 字节 —— 四倍，来自"两份矩 × 每数 8 字节"对"一份权重 × 每数 4 字节"。真实工程里这正是"checkpoint 留三份、优化器状态只留一份"的原因。

### 2.4 断点续训要恢复的到底是什么

Adam 的更新是个马尔可夫过程（阶段 5 §2.5 那三行）：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

$$\theta_t=\theta_{t-1}-\mathrm{lr}_t\cdot\frac{m_t/(1-\beta_1^{t})}{\sqrt{v_t/(1-\beta_2^{t})}+\epsilon}$$

右边的式子里出现了 $m_{t-1}$、$v_{t-1}$ 和 **$t$**。所以"训到哪儿了"这个状态是四元组 $(\theta, m, v, t)$，一样都不能少：

* **只存 $\theta$**：$m,v$ 从 0 重新开始，前几十步等于偷偷重来一次 warmup，曲线在恢复点"抖一下再往下"。
* **忘了 $t$**（=`opt.step_count`）：$1-\beta_1^{1}=0.1$，而 $m$ 里装着 100 步攒下的量 —— 恢复的第一步把 $\hat m$ 放大十倍、$\hat v$ 放大二十倍，$\hat m/\sqrt{\hat v}$ 直接跳出去。症状是"续训的第一个 loss 比存档时高一大截"。
* **$\mathrm{lr}_t$ 本身不在四元组里**：每步的实际步长由训练脚本的 `lr_at(i, total, peak, warmup)` 现算。文件头里的 `lr` 记的是"存档那一刻优化器的构造 lr"，用途是让人和脚本认得出这份状态属于哪个训练计划 —— 判据 `训练状态的头里带 step / lr / curve` 只查它在不在、值对不对，不拿它驱动训练（`curve` 同理，它是为了让续训后的曲线能接上画，不影响数值对账）。

最后是这条边界，也是最容易踩的：**文件里存"跑到哪儿了"，脚本里存"打算怎么跑"**。`weight_decay`、`beta1/beta2`、`eps`、`peak`、`warmup` 都不在状态文件里。判据特意跑第三条曲线 C：同样的权重和矩、只把续训脚本里的 `weight_decay` 从 0.0 换成 0.5，恢复段立刻分叉（实测逐点差 $3.3\times10^{-3}$，而正确续训是 $2.7\times10^{-9}$，相差六个数量级）—— 那条软判据说的就是这件事：**拿旧状态配新脚本，是不会报错的**。

### 2.5 为什么语料是 4KB 的自撰英文，而不是 TinyShakespeare

`data/corpus.txt` 是本手册自己写的一篇小镇速写：64 行、4,061 个字符、27 个不同字符（23 个字母，没有 j/x/z，外加空格、逗号、句号、换行）。它不是"缩水版数据集"，它是**判据的固定输入**，理由三条：

1. **判据必须自带数据**：不下载、不装包、离线可跑；换台机器结论一样。种子全固定（`SEED=0`、`GPT(cfg, rng=1)`），所以你看到的每个数字都能复现。
2. **跑得动才看得趋势**：0.10M 参数的模型、CPU 上 300 步、约 7 秒。真数据集要几万步才动，而"每写完一个 TODO 就跑一次整卷"的节奏需要它十秒内出结论。
3. **但它仍然是真英文**：有词频长尾、有标点、有换行、句子之间只有部分重复。所以 loss 不会像阶段 5 的模式语料那样掉到 0.1 —— 实测末段 1.974，验证 1.985，贪心下一字符准确率 0.369。判据的阈值（$\le 0.75\ln V=2.472$、准确率 $\ge 0.30$）是按这个难度定的，而基准仍然是 $\ln V$：$\ln 27=3.296$ 是"瞎猜一个字符"的成本，初始 loss 实测 3.345（容差 0.2）。

两个泛化细节：`train_val_texts(text, 0.1)` 按**行**切，train 59 行 / val 5 行，判据 `train/val 零重叠（整行级）` 逐行求交集验证；按字符砍会把一句话劈成两半，val 就成了送分题。val 那 347 个 token 必须大于 `ctx+1=41` 才切得出窗口（判据用它当 info 打出来）—— 语料再短一半，"验证 loss"这条判据就会因为 `Batcher.__init__` 的长度检查而根本跑不起来。

### 2.6 为什么文件里 tied 权重只有一份

`Module.named_parameters()` 递归收集时**按 `id(val)` 去重**（`nn.py`），而 `TiedHead` 持有的是 `self.tok.weight` 这同一个对象，所以它天然只登记一次：29 个张量、103,936 个参数，`tok.weight` 只出现一回，名字是 `tok.weight`。

这件事有两个后果：

* **写侧**：你只要遍历 `named_parameters()`，就不可能把输出头存成第二份。反过来说，如果你在 `save_checkpoint` 里自己 `vars(model)` 翻一遍属性，就会翻出 `head.weight`（同一个对象的第二个名字），于是 30 个张量、多存 27×64 个数 —— 判据 `张量名与数量 == named_parameters` 和 `磁盘上是 float32`（字节数）都会红。
* **读侧**：`_read_tensors` 里那句 `if name in out: raise` 是在防另一半错误（比如有人手改了写侧顺序）。名字唯一才能当键 —— 这就是 §2.2 里"名字进文件而不是裸下标"的另一面。

---

## 3. 已给你的代码：字节层和它的四种拒绝

这一节讲**不是我让你写**的那些行（`_u32` / `_write_header` / `_write_tensors` / `_read_header` / `_read_tensors` / `_moment_lists`）。它们占了这个文件的一大半，因为字节序、截断、重名这些最容易出错的部分留给你写，你只会花一个下午和 `struct` 格式串搏斗，而这一章真正要练的是"协议两端怎么对账"。配套的单项快测是 `python webapp/checks.py GIV-4`（它现在就该是绿的 —— 它测的是给你的部分）。

### 3.1 `_u32` 与 `_write_header`：`sort_keys=True` 不是强迫症

* `<I` 是那"四个字节的小端无符号整数"。删掉 `<` 用 native 序：本机一切正常，跨平台对拍时版本号读成 `16777216`。
* `sort_keys=True`：JSON 的 dict 顺序按插入走，`Config.to_dict()` 用的是 `dataclasses.asdict`（顺序=字段声明顺序），今天稳定；但头里一旦加进别的来源的 dict（例如 `**opt.__dict__`），两次存档的**字节**就会不一样 —— 哈希对拍、`cmp` 两个 `.ckpt`、C++↔Python 逐字节 diff 全部误报。删了不挂判据，但会挂"未来那条对拍"。
* `VERSION` 写在 `_write_header` 而不是各调用点：版本号与格式的绑定必须在**一处**。

### 3.2 `_write_tensors`：三行防御，防的是三件事

```python
a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
f.write(_u32(len(raw))); f.write(raw)
f.write(_u32(a.ndim)); f.write(np.asarray(a.shape, dtype="<i8").tobytes())
f.write(a.tobytes(order="C"))
```

1. `np.asarray(arr, dtype=dtype)` —— **这就是"降到 fp32 / 留 fp64"发生的地方**，也是 `"<f4"` / `"<f8"` 这两个字符串唯一被解释的地方。删掉 `dtype` 参数：内存里的 float64 被原样写出，文件翻倍，判据 `磁盘上是 float32` 红；而你的 `load`（若照抄 fp64）仍自洽 —— 正是 §2.1 说的包庇。
2. `np.ascontiguousarray(...)` —— 把"行主序、连续"写成代码而不是注释。本项目的 `Parameter.data` 本来就是连续的，删了今天不会错；但哪天你存一个转置视图（RoPE 的 cos 表、KV cache 的某个轴），`shape` 记录的是逻辑形状而 `tobytes(order="C")` 也按逻辑 C 序复制 —— 两者一致才靠得上，显式化比"我记得它是连续的"可靠。
3. `np.asarray(a.shape, dtype="<i8").tobytes()` —— 一次写 $d$ 个 i64。为什么 $d$ 也要写：`shape` 有几个数得先知道才能读；空元组（0 维标量）走 `cnt = 1` 那条兜底。返回值是张量条数，只用于写侧自我核对 —— 判据 `格式合规` 里那个 `consumed == len(raw)` 用的是**读侧**推进的偏移。

### 3.3 `_read_header`：拒绝一与拒绝二（魔数、版本）

* 拒绝一（魔数）：报错位置在最前面，代价是四个字节。删掉它，把 `.npy`、`.state` 或一张 JPEG 喂给 `load` 时会读到 `n = ` 某个几千万的数，然后 `f.read(n)` 直接把内存吃穿 —— 报错离真正的错误已经隔了两层。
* 拒绝二（版本）：以后格式升到 v2，v1 的读法会**静默**读出垃圾。宁可现在拒绝。
* 第三种 `raise` 是"截断"的第一半：`len(body) != n`。`f.read(n)` 在文件不够时不报错，只给短数组；不检查就 `json.loads`，你看到的是 `JSONDecodeError`，而不是"这个文件在头这里就断了"。

### 3.4 `_read_tensors`：拒绝三与拒绝四（截断、重名），和最隐蔽的那个 bug

```python
if not nb: break                     # EOF = 正常结束（没有更多张量）
if len(nb) < 4: raise ValueError("文件被截断（还剩几个字节？）")
item = np.frombuffer(f.read(cnt * np.dtype(dtype).itemsize), dtype=dtype)
if item.size != cnt: raise ValueError(f"张量 {name} 期望 {cnt} 个数，只读到 {item.size}")
if name in out: raise ValueError(f"张量名重复：{name}（tied 权重被登记了两次？）")
out[name] = item.reshape(shape).copy()      # copy：frombuffer 给的是只读视图
```

* `if item.size != cnt` 是本文件最重要的一行。`np.frombuffer` 对"数据不够"的态度是**少给几个而不是报错** —— 这是二进制格式里最隐蔽的静默失败：张量形状照样是 `(64, 64)`，数值是前 4096 个（可能全是下一个张量的开头），模型照样能跑。检查点写到一半崩机（磁盘满、Ctrl-C）就是这个形状。
* 重名检测对应 §2.6：名字是键，键重复就是格式被写坏了。
* `.copy()`：`np.frombuffer` 返回的是那段 bytes 的**只读视图**。不 copy，读回来的数组不能改（你排错时想 `t[0] = 0` 试一下会撞上 `ValueError: assignment destination is read-only`），而且它的生命期绑在那个 bytes 上。判据 `TODO-26` 还要求 `info["tensors"]["a"].dtype == np.float32` —— 别顺手在这里升成 float64。
* `break`（EOF）和"还剩 1~3 字节"是两条不同的路：前者正常，后者截断。写格式时不留 padding，这个区分才成立 —— 这也是单项检查里 `consumed == len(raw)` 的意义。

### 3.5 `_moment_lists`：为什么不直接写 `opt.m` / `opt.v`

它把优化器的动量缓冲归一成 `[(属性名, 与参数同序的列表), …]`。三个理由都不是装饰：

1. **`SGD` 只有 `buf`**（`optim.py`）。直接写 `opt.m` 的话，换 SGD 就 `AttributeError`；而 SGD 的动量同样需要断点续训。
2. 返回 `(属性名, 列表)`，于是张量名的前缀 `f"{attr}.{name}"` 是**从数据里来的**，不是硬编码的两串字符串 —— 写侧和读侧共用同一个来源，就不会出现"存的时候叫 `m.`、读的时候找 `moment1.`"。
3. 它顺手给出顺序保证：每个列表都和 `opt.params` 同序，于是 TODO-27 可以 `zip(named, buf)`。这也是为什么那里要比 `len(buf) == len(named)`：不等说明**这台优化器配的不是那个模型**（`Optimizer.__init__` 在构造时就把 `_as_params(target)` 拍死在 `self.params` 里，换个模型不会自动跟上）。

### 3.6 两个魔数与 `checkpoint_summary`

* `MAGIC = b"MGPT"` / `STATE_MAGIC = b"MGTR"`：为什么要两个。动量数组**和参数同形同数**，如果用同一个魔数，把 `.state` 当权重文件读会"成功" —— 逐位照读、形状全对、`applied = 29`，只是模型里灌进去的是 $m,v$。这类错误必须靠一个四字节的键拦住。判据 `状态存档用的是 MGTR 魔数` 就查头四个字节。
* `checkpoint_summary(path)`：不建模型、纯读文件的一行摘要（张量数、fp32 字节数、`vocab/d_model/n_layer`）。它是"检查点出问题了"的第一条命令，也正是 TODO-26 里 `model is None` 那条早返回路径的现成用法。
* `VERSION = 1` 和格式注释同处一文件：改格式时必须同时改这里（连 `MAGIC`/`STATE_MAGIC` 一起），否则老读法会硬啃新文件。

---

## 4. TODO-25：`save_checkpoint`

**契约**：`save_checkpoint(path, model, cfg=None) -> int`，返回**文件字节数**。

你要定的只有两件事（字节层给你了）：

```
1) 存哪些张量、按什么顺序   → items = [(name, p.data) for name, p in model.named_parameters()]
2) 头 JSON 放什么           → {"config": cfg.to_dict()}
                              cfg 缺省 → getattr(model, "cfg", None) → 都没有就 ValueError
dtype 固定 "<f4"；返回 os.path.getsize(path)
```

**三个决定，逐条说清它是哪一项、删了会怎样：**

1. **为什么顺序是 `named_parameters()` 而不是自己数张量**：它按 `id` 去重（§2.6，tied 天然一份）、顺序稳定（递归 `vars()` 的插入序）、而且名字自带前缀（`blocks.0.attn.q.W`）。注意名字要用它返回的 `name`，**不要用 `p.name`**：`Parameter.name` 只在 `_collect` 里"空名字时补 `prefix+key`"，你单独拿 `p.name` 会在某些构造路径下拿到 `""` 或者一个不带父前缀的短名 —— 于是两个 block 的 `norm1.w` 同名，读回来只剩一份。
2. **为什么没有 config 就抛错**：没有它，形状能读回来，但没人知道 `d_model/n_layer/n_head`，C++ 侧建不出模型；而 Python 侧会走 `Config()` 的默认值（`d_model=128, n_layer=4`），**然后形状检查会先炸**，报一个离根因很远的话。半截存档比没有存档更危险。
3. **为什么返回字节数而不是张量数**：判据 `int(size) == len(raw)` 和那条字节账（§2.2）都拿它核对"是不是真的 4 字节一个数"。返回 `len(items)` 会让这两条同时红，而文件本身是对的 —— 这是"接口签名也是协议"的示范。另一个不起眼但会当场炸的点：`open(path, "wb")` 必须是**二进制模式** —— 文本模式下写 bytes 直接 `TypeError`，而在某些环境里 `\n → \r\n` 的转换会把那笔字节账撕破。

写完先跑单项：`python webapp/checks.py TODO-25`。它用一个小 `Tiny(Module)`：`a` 是 (2,3)、`b` 是 (3,) 且 `wd_exempt`、`c = self.a`（**tied 别名**）。期望是：解析出的张量名**恰好** `{a, b}`（没有 `c`，`a` 也没存两遍）、数值 `atol 1e-7` 内一致、dtype 是 float32、头 JSON 等于 `cfg.to_dict()`、返回值 == 文件长度、且每段字节加起来分毫不差。

<details>
<summary>参考答案 · TODO-25 <code>save_checkpoint</code>（判据没过之前别点开）</summary>

```python
    cfg = cfg if cfg is not None else getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("没拿到配置：显式传 cfg=，或者用带 .cfg 属性的模型（GPT 就有）")
    items = [(name, p.data) for name, p in model.named_parameters()]
    with open(path, "wb") as f:
        _write_header(f, MAGIC, {"config": cfg.to_dict()})
        _write_tensors(f, items, "<f4")
    return os.path.getsize(path)
```

七行里没有一处可以"顺手改改"。`items` 用列表推导而不是生成器：`_write_tensors` 要返回条数，生成器会被消费掉一次就没了。
`os.path.getsize(path)` 放在 `with` **之后**：文件还没关就取大小，在 Windows 上会拿到还没刷盘的旧值（这里凑巧也常常对，但不是每次）。

</details>

---

## 5. TODO-26：`load_checkpoint`

**契约**：`load_checkpoint(path, model=None, strict=True) -> dict`，返回至少含 `config`（一个**`Config` 对象**，不是 dict）、`tensors`（dict），给了 model 还要含 `applied`（灌回去的张量数）。

这一步真正要练的是**校验与装配**。四道拒绝的顺序是有道理的（先看整体对不对，再看个体能不能对上）：

```
① _read_header(f, MAGIC) + _read_tensors(f, "<f4")     ← 给你的字节层已经在拒绝魔数/版本/截断/重名
② head 里没有 "config" → ValueError；有 → Config.from_dict(...)
   model is None → 现在就返回（排错第一条命令）
③ cfg != model.cfg → ValueError，并把**具体哪几个字段**写进消息
④ 逐个参数按名字取：缺名 → ValueError；形状不符 → ValueError（写期望/实际）
   存档里有模型没有的名字 → strict 时 ValueError，否则放进返回值的 "extra"
⑤ p.data[...] = t      ← 原地写
```

**为什么③要这么狠（这不是官僚主义）**：`Config` 是 dataclass，`!=` 会自动逐字段比。反例最有说服力 —— 存档是 `n_head=4` 训的，你拿 `n_head=8` 的模型去读：**所有张量形状完全一致**（`q.W` 永远是 `(d, d)`，头数只体现在 forward 里的 reshape），加载不报错、loss 照样是个数、`无死参数` 也照样绿，只是那是一台被搬错零件的机器。`eps`、`init_std`、`use_rope`、`use_swiglu` 同理。形状检查抓不住"结构参数"，所以必须有一道配置比对；而判据 `配置不一致时拒绝加载` 还额外要求**消息里点名差异字段**（它把 `vocab` 改 3 之后检查 `"vocab" in str(exc)`）—— 因为报错的价值在于能直接定位。想列出差异就遍历 `cfg.to_dict()`、逐键比 `getattr`（参考答案里那三行列表推导就是全部）。

**为什么⑤必须原地写**：`p.data[...] = t` 而不是 `p.data = t`。`p.data = t` 换掉的是**那个对象**，而此刻有三方还握着旧对象：`TiedHead`（借的正是 `tok.weight`，`gᵀ@x` 要往它上面累加）、`Optimizer.params`（构造时拍平引用）、以及 `p.grad` 的同形缓冲。症状是"加载没报错，生成出来的还是随机权重" —— 判据 `加载是原地写` 用 `fresh.tok.weight.data is keep_data` 直接查**对象身份**。dtype 不用担心：`t` 是 float32、`p.data` 是 float64，切片赋值会自动升上去，这正是我们想要的（磁盘 fp32、计算 fp64）。

**④里两个方向的缺件要分开处理**：模型要的东西存档里没有 = 缺件，必抛（这台机器装不齐）；存档里的东西模型不要 = `extra`，`strict=True` 时抛、`strict=False` 时只报告。后者是"以后加了新层但要读旧存档"的兼容钩子（阶段 8 会真用到），所以签名里有 `strict`。

单项快测 `python webapp/checks.py TODO-26`（它需要 TODO-25 先写完）查五件事：`load_checkpoint(path)` **不建模型**也拿到 `Config` 对象和 `tensors`；灌进新模型后数值一致且 `applied == 2`；`id(dst.a.data)` 没变（原地）；配置不一致抛 `ValueError`；头四个字节被改成 `XXXX` 时抛 `ValueError`。

<details>
<summary>参考答案 · TODO-26 <code>load_checkpoint</code></summary>

```python
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
```

`tuple(t.shape) != p.shape` 两边都要 `tuple()`：`p.shape` 是 `Parameter` 的 `@property`，返回的是 `self.data.shape`（一个 tuple），而 `t.shape` 是 `numpy` 的 `ShapeTuple` —— 直接比 `!=` 在元组元素相同的情况下也相等，但混着 list 写就会永远"不等"。判据里那个故意把形状改成 `(3, 3)` 的 `warped` 就走这条分支。
注意 `ref is not None` 那个前提：没有 `.cfg` 的裸 Module（比如单项检查里的 `Tiny`）不该被配置校验拦住 —— 它根本没有配置可比。

</details>

---

## 6. TODO-27：`save_train_state` / `load_train_state`

**契约**：`save_train_state(path, model, opt, step, curve=(), cfg=None) -> int`；`load_train_state(path, model=None, opt=None) -> dict`（返回头里的元信息 + `config` 对象 + `tensors`，给了 model 和 opt 就顺手恢复）。

和 TODO-25/26 只有**两处**不同，请想清楚再写（这正是这条 TODO 的全部内容）：

1. **头 JSON 从一字段变四字段**：`{"config":…, "step":…, "lr":…, "curve":[…]}`。`curve` 要逐个 `float()` —— 训练循环里攒的是 `model.loss_backward(x)` 的返回，`CrossEntropy` 交出来的是 `numpy.float64`，而 `json.dumps` 不认它（`TypeError: Object of type float64 is not JSON serializable`）。这是本章唯一一处"当场就报"的坑，好事。
2. **张量是矩，不是权重**：`dtype="<f8"`（§2.3）、魔数 `STATE_MAGIC`（§3.6）、命名规则固定 `f"{attr}.{参数名}"`，例如 `m.blocks.0.norm1.w`。写侧就是 `_moment_lists(opt)` 与 `named_parameters()` 的 `zip`（比 `len(buf) == len(named)`，不等就是配错了）。用名字而不是裸下标，是为了让 load 那侧能校验"这台优化器配的是那个模型吗"（§3.5 第 3 点）。

读侧把顺序反过来 —— **顺序是它的全部风险所在**：

```
key = f"{attr}.{name}"                # 按名字取，绝不按位置
t = tensors.get(key) → None 就 ValueError 点名缺哪个名字
tuple(t.shape) != p.shape → ValueError
buf[i][...] = t                       # 又是原地写
opt.step_count = int(head["step"])    # §2.4 里那个 t
```

`buf[i][...] = t` 的 `[...]` 和 TODO-26 的 `p.data[...] = t` 是同一个约定（`Optimizer` 构造时把数组**引用**存进 `self.m` / `self.v`，重新绑定名字等于把刚读回来的东西扔掉 —— 阶段 5 §11 的坑 #2 在这里复活，代价从"训不动"变成"续训不吻合"）。而拿错优化器状态最坏的下场**不是报错，是用别人的动量继续走**：`m` 的形状和参数完全一致（本来就是照参数建的），位置错位不会撞形状，只会撞名字 —— 这就是坚持按 `key` 取、取不到就抛的全部理由。

判据这边：`训练状态的头里带 step / lr / curve`（`step == 100`、`lr == 6e-3`、`len(curve) == 100`）、`状态存档用的是 MGTR 魔数`、以及 §7 那条对账。单项快测 `python webapp/checks.py TODO-27` 还会专门按 `"<f8"` 读一遍并要求尾部**一个字节都不剩**（写成 fp32 就会读到一半就断）。

<details>
<summary>参考答案 · TODO-27 <code>save_train_state</code> / <code>load_train_state</code></summary>

```python
    cfg = cfg if cfg is not None else getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("没拿到配置：显式传 cfg=，或者用带 .cfg 属性的模型")
    named = model.named_parameters()
    items = []
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
```

```python
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
```

`out = dict(head)` 再覆盖 `config`：头里的 `config` 是 dict，返回值里要换成 `Config` 对象，而 `step / lr / curve` 保持原样给调用方（判据读的就是这三个键）。
`opt.step_count` 那一行是整条对账判据的开关 —— 想验证自己理解对了，就把它注释掉跑一次 §7，看那个 `2.7e-9` 变成多少（预期 $10^{-2}$ 量级）。

</details>

---

## 7. 训练循环与"断了能续"这条对账怎么读

阶段 6 不新写训练循环，判据脚本里那个是阶段 5 §14 的加长版（`stages/s06_train.py` 的 `train_loop`）：

```python
for i in range(steps):
    opt.zero_grad()
    x, _y = batcher(B, rng)
    curve.append(model.loss_backward(x))              # 多了：曲线交出来，续训要写进头里
    gnorms.append(clip_grad_norm(model, 1.0))          # 多了：每步的裁剪前范数
    opt.step(lr_at(i, total, lr_peak, warmup=warmup))  # 多了：total 来自 horizon
```

新增的两样各有一条判据依赖它：`gnorms` 支撑 info 那条"grad_norm 中位数/最大值"（实测 1.147 / 6.603，说明相当一部分步被 `MAX_GRAD=1.0` 裁过 —— 这不是问题，这正是裁剪存在的理由）；`curve` 进 TODO-27 的头 JSON。

**`horizon` 是这一节唯一的新概念，也是续训最容易做错的一处。** `train_loop(m, o, rb, R_FIRST, rng, 6e-3, 20, 0, "B 前段", horizon=R_STEPS)` —— 学习率调度看到的**总步数**必须是原计划的总步数（150），不是剩余步数。写成"还剩多少步"的话，cosine 会在恢复段自己又走完一个完整的余弦 —— 不报错、loss 照样降，只是 B 曲线和 A 曲线不再重合。`lr_at` 是纯函数，它不知道你"是第二次跑"。

**判据是怎么三条一组设计的**（同一份语料、同一个 `rng` 种子 12、同一串随机窗口）：

| 曲线 | 走法 | 期望 |
|---|---|---|
| A | 一口气训 `R_STEPS=150` 步 | 基准 |
| B | 训 `R_FIRST=100` 步 → 存权重 + 状态 → **另建一台**（`rng=999`）+ 另建优化器（`lr=1e-3`）→ 读回来 → 再训 50 步 | 与 A 在恢复段逐点差 ≤ `1e-4`（实测 `2.7e-9`） |
| C | 同 B，但续训时把 `weight_decay` 从 0.0 换成 0.5 | 与 A 分叉（实测 `3.3e-3`，比 B 大六个数量级） |

B 的"另建一台用别的种子、别的 lr"是故意的：**这两样都必须被存档覆盖掉，否则"续训"是假的**。`lr` 之所以能被覆盖，是因为续训循环每步显式传 `lr_at(...)`，优化器构造时的 `1e-3` 从来没生效过。

`stepwise` 是恢复段（第 100~150 步）**逐点**最大差，不是末段均值差 —— 逐点看才能抓住"只错了一步"的 bug（比如 `step_count` 少恢复一步，只有恢复段的前几步偏，末段均值会被稀释掉）。**读结论**：能读回来 ≠ 能续上。`存了能读`（TODO-25/26 那四条）证明的是格式，`断了能续`证明的是**状态完整** —— 这一章真正的新知识就在后一条上。

下面这张图把上表真算一遍：一条"一口气 150 步"、一条"100 步存档 → 新建读回 → 续 50 步"，逐点对齐；四个勾选框分别丢掉 θ / m / v / t，勾上哪个，续训那段曲线立刻跳 —— 四元组缺一不可就这么看。

@@fig resume@@

---

## 8. 验收判据逐条解读

`python stages/s06_train.py`：19 条硬判据 + 2 条软提示 + 5 条 info，三组。

**语料组（3 条）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| 语料读到了 | ≥40 行、≥2000 字符 | `data/corpus.txt` 在不在、判据自带数据是否完整（§2.5） |
| 分词在这份语料上往返一致 | `decode(encode(x)) == x` 且 `vocab == 去重字符数` | 阶段 5 TODO-20 在这份真数据上仍成立（词表口径 27） |
| train/val 零重叠（整行级） | 交集为空、val ≥ 3 行 | `train_val_texts` 的"按行下刀"（§2.5），防泄漏 |

**训练组（5 条 + 1 软）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| 初始 loss ≈ ln V | $\lvert \text{初始 loss}-\ln 27\rvert \le 0.2$ | 装配没跑偏（阶段 5 §2.4；真语料的 $\ln V=3.296$） |
| 训练 loss 下降倍数 | ≥ 1.5× | 300 步确实学动了（真实英文只要求 1.5×，模式语料那卷要求 5×） |
| 训练 loss 掉到 ln V 的四分之三以下 | ≤ $0.75\ln V=2.472$ | 绝对水平，不只是"降过" |
| 验证 loss（5 行没吃过的） | ≤ $0.75\ln V$ | 泛化，不是背下来；`evaluate()` 与训练同一种移位 |
| 贪心下一字符准确率 | ≥ 0.30（随机猜 0.037、最高频字符 0.205） | 模型真的在预测下一个字符，而不是复读高频字 |
| *软：没有明显过拟合* | `val - train < 0.5` | 4K 字符 + 0.1M 参数 = 背得下来；两个数一起看才有意义 |

info 那条 `grad_norm 中位数/最大值` 不计入通过率，但它是你以后调 lr 的第一诊断指标（尖刺抬头 = 发散先行）。

**检查点组（11 条 + 1 软）**

| 判据 | 阈值 | 查的是 |
|---|---|---|
| 格式合规（参考解析器读得通） | 魔数/版本/`head["config"] == cfg.to_dict()`/`consumed == len(raw)` | TODO-25：字节表 + 头 + 无 padding 尾巴（§2.1、§2.2） |
| 磁盘上是 float32，不是 float64 | 全部 dtype fp32 且 `len(raw) < 8×参数量` | TODO-25 的 `"<f4"`（§2.3）。417,005 vs 831,488 |
| 张量名与数量 == named_parameters | 排序后相等、无重复 | TODO-25 用收集器而不是自己翻（§2.6，29 个） |
| 读回来能算出一样的 logits | `applied == 29` 且最大差 < `2e-4` | TODO-26 的装配完整性；实测 1.6e-7 = fp32 量化 |
| 加载是原地写 | `fresh.tok.weight.data is keep_data` | TODO-26 的 `p.data[...] =`（§5） |
| 配置不一致时拒绝加载 | 抛 ValueError **且消息含 "vocab"** | §5 的③：形状兼容也要拦 |
| 魔数不对时拒绝加载 | 抛 ValueError | `_read_header` 拒绝 1（§3.3） |
| 形状对不上时拒绝加载 | 抛 ValueError | §5 的④（`warped.tok.weight.data = zeros((3,3))`） |
| 续训与直训在恢复段逐步吻合 | 逐点最大差 ≤ `1e-4` | TODO-27 的**完整性**：$\theta$ + $m$ + $v$ + $t$（§2.4、§7） |
| 训练状态的头里带 step / lr / curve | `step==100`、`lr==6e-3`、`len(curve)==100` | TODO-27 的头四字段（§6） |
| 状态存档用的是 MGTR 魔数 | 头四字节 == `b"MGTR"` | §3.6：两个魔数分隔两类文件 |
| *软：状态不恢复 weight_decay* | `stepwise_C > 100 × stepwise_B` | "文件存跑到哪儿，脚本存打算怎么跑"这条边界（§2.4） |

注意这组里**没有一条**是"用你的 `load` 读你的 `save`"：格式那条用的是脚本里另写的 `parse_ckpt`（§2.1）。这是本章判据设计上最值得你记住的一招。

---

## 9. 常见失败模式对照表

| 症状 | 病因 | 去处 |
|---|---|---|
| `格式合规` 挂，且 `consumed != len(raw)` | 写侧多塞了 padding / 少写了一个字段的字节 | §2.2 那笔账手算一遍；单项 `TODO-25` 的 `acct_ok` 更直接 |
| `磁盘上是 float32` 挂，文件约 831K | `dtype` 传了 `"<f8"` 或干脆没传 | §2.3；你的 `load` 也要同步改成 `"<f4"` |
| 单项 `TODO-25` 说张量名多了个 `c`；或读的时候炸 `张量名重复` | 用 `vars(model)` 自己翻属性，把 tied 别名登记两次 | §2.6：只用 `named_parameters()`（`Module._collect` 的 `seen` 负责去重） |
| 读回来的名字不带 `blocks.0.` 前缀 | 用了 `p.name` 而不是收集器返回的 `name` | §4 决定 1 |
| `读回来能算出一样的 logits` 的 `applied` 对不上 | 忘了往返回值里塞 `applied`，或只灌了一半参数 | §5 契约（返回值也是协议） |
| 加载不报错，但生成仍是随机的 | `p.data = t` 换了对象，tied 头/优化器还看着旧的 | §5 那段原地写；判据 `加载是原地写` 就是它 |
| `配置不一致时拒绝加载` 挂（抛了错但消息不含字段名） | 只写 `raise ValueError("配置不一致")` | §5 那段 `diff` 列表推导 |
| 把 `.state` 当 `.ckpt` 读却没报错 | 两类文件用了同一个魔数 | §3.6（判据 `MGTR` 那条会红） |
| `TypeError: Object of type float64 is not JSON serializable` | `curve` 里是 numpy 标量 | §6：`[float(v) for v in curve]` |
| 续训对账差 $10^{-2}$ 量级 | 漏了 `opt.step_count`，或只存了 `m` 没存 `v` | §2.4 那个四元组；§3.5 的 `_moment_lists` 返回两个列表 |
| 续训对账差 $10^{-3}$，但存档读档都"对" | `buf[i] = t` 重新绑定名字，优化器仍用旧数组 | §6：`buf[i][...] = t` |
| B 曲线在恢复段整体比 A 高一块，且不收回 | 续训时 `lr_at` 的总步数传成剩余步数 | §7 的 `horizon` |
| C 曲线也和 A 重合了（软判据提示消失） | 你顺手把 `weight_decay` 也存进状态并恢复了 | §2.4 的边界：超参属于脚本。软判据专门在防这个 |
| 训练组里"验证 loss"挂而 train 正常 | val 太短切不出窗口 / `train_val_texts` 的刀落在行中间 | §2.5；info 那条 `val 的 token 数` 要先 > `ctx+1` |

---

## 10. 自查题

1. 判据为什么**不**用你写的 `load_checkpoint` 来验证 `save_checkpoint`？给出一个只有"参考解析器"能抓到、而自存自读抓不到的 bug。
2. 把 §2.2 那笔字节账代到 $12+n_{\text{JSON}}+\sum(\cdot)$ 里，解释为什么"每个数 4 字节"这句话必须由判据来数，而不是由你的代码来声明。
3. 权重存 fp32、矩存 fp64，各自的理由是什么？如果把矩也降成 fp32，哪一条判据会红、大概红到哪个量级？
4. 断点续训要恢复的四元组是哪四样？只漏掉 `step_count` 时，恢复后的第一步在数学上发生了什么（写出 $c_1=1-\beta_1^{t}$ 的两组值）？
5. `n_head=4` 的存档灌进 `n_head=8` 的模型，形状检查抓得住吗？`eps` 不同呢？这说明配置校验和形状校验各自负责什么？
6. 为什么 `MAGIC` 和 `STATE_MAGIC` 必须不同？（要求：给出一个具体的静默错误，它的 loss 看起来还挺正常。）判据用"恢复段**逐点**最大差"而不是"末段均值差"，又是为什么？
7. `weight_decay` 不在状态文件里，所以换脚本会分叉。你认为应该把它存进去吗？说出两边各自的代价（提示：`lr` 存了但不用于驱动）。

---

## 11. 下一步

阶段 7：`src/minigpt/sampler.py`（TODO-28~31）。模型已经能存能续了，接下来决定的是"**同一份权重、同一个 prompt，它开口说什么**"：温度、top-k、top-p、重复惩罚。顺带一提，阶段 5 §3.5 留了一句预告：`GPT.generate` 里那一行 `sample_next(...)` 会在你写完整机之后**卡住一次**（`NotImplementedError: TODO-28`）。这不是坏掉 —— 那是阶段 7 的家。

如果你打算走 C++ 那条路（`docs/advanced-cpp-roadmap.md §6.4`）：现在你手上已经有一份可以直接对拍的存档了。用 `checkpoint_summary` 看内容，用 `parse_ckpt` 那种"照表推进"的写法在 C++ 侧读一遍，同一份权重两边逐 token 一致 —— 那是 M4 里程碑的验收方式。
