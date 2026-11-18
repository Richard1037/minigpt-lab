# 分 Part 学习逻辑审查（2026-11-14）

> 2026-11-15 局部执行：Embedding（layer_embedding）已增加完整教学与构造/前后向实现题，原8项保留并新增独立构造测评，进入 review。其他层与跨 Part 闭环仍按清单待办，不代表 Part 04 全部完成。

> 当前任务执行更新：初始化已改为 `initialize(shape,rng=None)` 代码题，六项真实用例，并接入个人 Linear。对应 `logic-part-04-initialize-code` 与 `logic-part-04-initialization-order` 已进入 review；下文的“现状”保留为首次审查证据。Part 07 接收 GPT 专用初始化及其他条目仍待执行。最新结构见事实快照，状态以网页清单为准。

本轮只审查并记录：覆盖当前 56 个任务、全部骨架接口及相关最终导入链。没有改动讲义、题目、代码骨架、评分标准或个人文件。所有新修订项均为 todo，实际状态唯一来源是 webapp/visual-roadmap.json。

## 为什么前次没有发现

前次重点核对接口、公式、显式依赖顺序、数值测评和展示入口；把“参考实现存在、当前测评能通过”误当成“学习者已被安排独立完成”。还把高级知识名称的出现误当作前置已学完。因此初始化两道选择题与公开 helper 的实际缺口、以及基础初始化混入 GPT 残差规则未被识别。先前测试通过和 review 状态都不等于教学闭环成立。

## 本次审查方法与边界

1. 沿真实目录读取每一任务的正文、推导、接口、骨架、参考实现入口和用例类型；检查实现前是否已解释所需对象、操作与返回值。
2. 对骨架做 AST 盘点，区分 TODO、给定构造、合理无参数层、概念预览和故意有错的拓展题；不把所有概念题强改成代码题。
3. 追踪 learner_core、网页 RUNNER、独立模型依赖、实验室与本地 CLI，区分有个人文件、实际调用个人代码、由平台提供的控制工具。
4. 每个问题登记归属 part、证据、修改动作和验收条件。迁移知识点同时登记来源和去向；未来用户指定 part 后才执行修改。
5. 这是静态教学逻辑审查；未进行 56 页浏览器逐步学习验收，也未在本轮运行用户实现。暂未确认的问题不冒充数学或数值错误。

## 全局事实

- 56 个任务全部映射到本轮 31 项审查条目；九个编号 part（含暂缓 Part 03）和拓展分别归档。
- 显式 dependencies 未发现前后倒置；这不排除正文引用未学知识。
- 初始化：concept、2 题、无代码骨架；initialize 来源为 foundation_core，不是学习者初始化实现。
- 完整 GPT 主线采用 RMSNorm，不使用当前 Dropout/LayerNorm；这些题需明确对照用途和主线地位。
- 独立 Linear 有构造题，但后续独立 Block/GPT 显式链仍导入 layer_linear。
- 实验室与 CLI 另有控制循环；train 学生函数的通过不能证明实验室运行的是这一函数。

## 逐 Part 修改清单

### 第一部分 · 整体架构（part-01）

#### Part 01 · 概念预览与实际交付的边界

条目：`logic-part-01-overview` · 优先级 P2；状态见网页清单。

**已确认的现状**：map 的两题是编号到向量、优化器更新；architecture 直接列出 Pre-Norm/RMSNorm/GELU、QKV 和残差反向，这些尚未学习。概念预览可以保留，不能把列过名称等同于已掌握。

**将来修改**：概览主线讲文本→编号→向量→分数→训练/生成；高级变体只作带去向的预览，原始 Transformer 对照放拓展。

**验收条件**：

- 两道概念任务保留合理的识别用途，不为概览强加随机编码题。
- 每个测评答案都能由本页具体例子推出，不要求预先掌握后续层的算法。
- 明确 B/T/C/V，图中同一 token 的编号、向量和预测位置连续可追踪。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

### 第二部分 · NumPy 与张量（part-02）

#### Part 02 · 数组与索引先讲语法再出题

条目：`logic-part-02-array-basics` · 优先级 P2；状态见网页清单。

**已确认的现状**：axes、index 正文为空，依赖任务表、公式和过程展示；shape 的交付是形状数组，标题中的数组创建和类型转换并非独立可观察产物。

**将来修改**：先教数组/shape/axis/索引语法，再给用途明确的小题，保留当前清楚的 ids、k 含义。

**验收条件**：

- 说明 shape 题是形状识别的语法应用，测评只要求可观察的返回值。
- axis 归约与 ids 重复取行、截前 k 列均先有完整例子。
- 题面明确 NumPy 返回类型、索引起点、k 范围及保持输入原值。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 02 · 广播到去均值的前置链

条目：`logic-part-02-broadcast-reduce` · 优先级 P2；状态见网页清单。

**已确认的现状**：广播已排在去均值之前，但正文主要是规则和公式；(N,1) 的广播、keepdims 与 Python/NumPy 实际操作还需从完整例子衔接。

**将来修改**：把按列加偏置和每行减均值的来源对应清楚，再给实现题。

**验收条件**：

- 沿 X→μ(N,1)→Y(N,C) 演算，N 与 C 取不同值。
- 学员在题前会使用 mean、axis、keepdims 和数组减法。
- 说明该技能用于 Linear 偏置和归一化，保留输入与输出形状检查。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 02 · 数组反向先连接已会的链式法则

条目：`logic-part-02-gradient-bridges` · 优先级 P2；状态见网页清单。

**已确认的现状**：两题已有真实手写实现和链式法则推导；mean_backward 突然从二维前向进入多 batch/token 的 G，部分正文为空，仍需把前向对象、G 来源及广播操作连续连接。

**将来修改**：保留必要推导，以同一前向例子引出 G，再解释反向目标和完整前置维度。

**验收条件**：

- 区分给定上游 G 与需要返回 dX，不要求先实现损失层。
- 先展示均值梯度均分，再展示去均值两路合流。
- 标明是归一化必需的知识练习，完整输入/输出与公开测评相符。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 02 · 标题、交付与高维形状一致

条目：`logic-part-02-shape-path` · 优先级 P2；状态见网页清单。

**已确认的现状**：reshape 标题称合并与恢复，但只交付展平，restore 才恢复；transpose 标题含批量矩阵乘法，题目只交换轴；heads 标题含还原，实际只拆头。

**将来修改**：收束每题标题与实现入口，同时串起后续 Linear/注意力使用路径。

**验收条件**：

- 矩阵乘法、reshape、恢复、transpose、拆头分别标清本题产物。
- 先教 @、-1、轴排列和 C/H 的含义，再给题。
- 保留 B>1 和完整 head/token，说明为何直接 reshape 不能替代换轴。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 02 · 副本与重复索引的练习用途

条目：`logic-part-02-copy-scatter` · 优先级 P2；状态见网页清单。

**已确认的现状**：copy 用固定 -99 观测独立副本，属于必要语法练习；scatter 是 Embedding 反向准备，但高级索引临时数组与重复回写的差异只用短句说明。

**将来修改**：明确语法练习用途，连续演示原数组/副本及重复编号梯度的变化。

**验收条件**：

- -99 说明为观测副本的例值，不虚称模型计算。
- 讲清引用、视图和 copy，在题前知道哪些值会改变。
- 先比较一次普通索引与重复索引，再教 add.at 或等价循环。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 02 · RNG 练习连接真实初始化

条目：`logic-part-02-random` · 优先级 P2；状态见网页清单。

**已确认的现状**：已有四项/题的真实实现检查，但 seed、Generator 和连续状态的知识与 Part 04 的初始化交付没有连成闭环。

**将来修改**：保留标准正态和连续状态练习，在两题结束处明确将用同一个 RNG 初始化多份权重。

**验收条件**：

- 区分创建 RNG 与继续使用 RNG，使用连续两次采样例子。
- 补 shape 元组和 standard_normal 的必要语法说明。
- 后续初始化题接受传入 RNG，不在函数内反复重置种子。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

### 第三部分 · 标量网络（part-03）

#### Part 03 · 标量网络继续暂缓

条目：`logic-part-03-paused` · 优先级 P2；状态见网页清单。

**已确认的现状**：第三部分已 paused，用户已掌握基础 FP/BP；旧清单仍有标量自动微分待办。

**将来修改**：维持暂缓，按需链接已有标量知识，不能在执行其他 part 时顺带扩写本章。

**验收条件**：

- 仅用户明确要求才执行本部分。
- 其余部分解释张量操作和局部导数即可，不强制重学自动微分。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

### 第四部分 · 完整基础层（part-04）

#### Part 04 · 对象遍历和给定辅助代码讲完整

条目：`logic-part-04-module-python` · 优先级 P2；状态见网页清单。

**已确认的现状**：学习者已实际误解 vars(value).items() 中 name、child 为 Module 的两个成员；正文仅简短描述属性遍历，__call__ 与 *args 此前也未讲清。

**将来修改**：加入属性字典→items 键值对→解包→递归访问的完整例子，保留已认可的五部分解释。

**验收条件**：

- 三个属性示例展示每轮 name/child，不假定对象固定有两个属性。
- self、类实例、继承及 __call__ 的必要用法在实现前可理解。
- 只为尚未讲解的框架部分注释，保留单编辑框与折叠参考实现。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 04 · 初始化必须由学习者实现并接入模型

条目：`logic-part-04-initialize-code` · 优先级 P1；状态见网页清单。

**已确认的现状**：initialization.kind=concept，仅两项选择题，没有 starter/file；参考实现含 initialize，而 learner_core 仍从 foundation_core 导入 initialize。

**将来修改**：新增或迁移为 initialize(shape,rng) 编程交付，提供 OI 题面、骨架、真实 Python 测评，并接到个人 Linear。

**验收条件**：

- 检查输出形状、float64、缩放值、不同 C/D 和非默认 RNG。
- 检查连续使用同一 RNG，拒绝硬编码样例或内部重置种子。
- 网页 RUNNER、本地导入、依赖指纹和最终模型实际使用个人初始化实现。
- 保留旧选择题成绩；新增实现标准单独版本化，不因文案排版使旧记录失效。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`exercises/learner_core.py`、`exercises/foundation_core.py`、`webapp/pilot_server.py`、`scripts/learner_layout.py`。

#### Part 04 · 基础初始化不混入未学的 GPT 配置

条目：`logic-part-04-initialization-order` · 优先级 P1；状态见网页清单。

**已确认的现状**：正文 1–3 讲基础 Linear 初始化，第 4 节立即要求理解 GPT 的 0.02、Block、残差分支和出口缩放；这些对象在 Part 06 才实现。

**将来修改**：本节只闭环基础初始化；把 GPT 专用初始化的教学及实现验收移至 Part 07，Part 04 只保留用途去向。

**验收条件**：

- 学完本节能独立完成基础初始化，不需要先学 Block 或残差出口。
- 迁移 GPT 的规则、样例、题目和参考代码一起处理，避免只删正文却仍考高级规则。
- Part 07 对应条目负责接收并落实迁入内容。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`scripts/learner_layout.py`。

#### Part 04 · 从填前后向到完整构造的闭环

条目：`logic-part-04-constructors` · 优先级 P1；状态见网页清单。

**已确认的现状**：Scale/Linear/Embedding/RMSNorm/MLP 的 __init__ 都预先给出；只存在独立 Linear 构造题，Embedding/RMSNorm/MLP 没有后续独立构造交付。

**将来修改**：按前后向台阶逐步放开构造器，在本 part 或最终整合明确由学习者补写核心参数/子模块创建。

**验收条件**：

- 给定构造器有教学用途说明，不把前后向通过等同于整层独立完成。
- Embedding/RMSNorm/MLP 的参数形状、初值、子层创建由学习者实际实现并测试。
- 衔接个人 initialize；构造代码只在必要时作为提示或折叠参考。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 04 · 完整基础层先讲职责再讲公式

条目：`logic-part-04-layer-teaching` · 优先级 P2；状态见网页清单。

**已确认的现状**：部分正文只有缓存句子或公式，首次完整层已同时要求缓存、输入梯度、参数梯度和清零；Embedding 的整数输入及 Norm/MLP 位置语义需要可操作的解释。

**将来修改**：每层按职责→输入→前向→缓存→反向产物→状态变化讲清，再用真实模块题验收。

**验收条件**：

- Linear 保留完整前置轴，分别解释 dX、dW、db 的来源和累加轴。
- Embedding 连接前面的取行/散射；GELU 解释逐元素非线性，RMSNorm 先解释均方与缩放。
- MLP 按子层 forward 顺序和 backward 逆序走同一样例，未来变体放拓展。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 04 · 独立 Linear 的后续复用不能只写在链接里

条目：`logic-part-04-independent-linear` · 优先级 P1；状态见网页清单。

**已确认的现状**：rebuild_linear.next_use 指向 rebuild_block，但 rebuild_block 的依赖仍是 layer_linear；独立 Linear 未进入最终独立 GPT 的显式依赖。

**将来修改**：确定后续独立 Block/MHA 复用哪一个 Linear，统一代码导入、网页执行和验收；用个人类身份确认。

**验收条件**：

- 说明练习复用还是算法能力迁移，文案与实际调用一致。
- 若称最终复用，验证 Block/GPT 的 Linear 实例确实来自独立实现。
- 跨 Part 06/09 的改动分步登记，未接通前不宣称完成。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`scripts/build_independence.py`。

### 第五部分 · 概率与损失（part-05）

#### Part 05 · 先解释分数到分布，再写稳定 Softmax

条目：`logic-part-05-softmax` · 优先级 P2；状态见网页清单。

**已确认的现状**：正文开头是减最大值、指数化、归一化，标题同时使用向量雅可比积；容易从术语和公式直接跳进实现。

**将来修改**：先展示一行分数如何得到每行和为 1 的概率，再说明缓存和上游梯度的行内耦合。

**验收条件**：

- 解释指数、逐行求和、广播除法和减最大值的操作；稳定性证明放拓展。
- 先教概率间的相互影响，再给已有完整 backward 实现题。
- 保留多 batch/token 和不同类数，公式与 NumPy 轴操作逐项对应。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 05 · 交叉熵的标签选择与梯度接口

条目：`logic-part-05-loss` · 优先级 P2；状态见网页清单。

**已确认的现状**：正文直接使用 logp、LogSumExp、take_along_axis；前面只做过二维取行，未完整教三维按标签取类别以及零维损失。

**将来修改**：沿 logits、整数 targets、正确类 log 概率、位置平均和标量上游 g 走一遍。

**验收条件**：

- 先教 take_along_axis 所需索引轴与形状，或给等价清楚索引方式。
- 题前明确 forward 的零维 loss、backward(g) 的同形 logits 梯度及 targets 不求导。
- 说明训练只在损失处平均一次，检查非单位 g 和不同 B/T/V。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

### 第六部分 · 注意力与 Transformer（part-06）

#### Part 06 · 注意力四个前向台阶连续讲解

条目：`logic-part-06-attention-front` · 优先级 P2；状态见网页清单。

**已确认的现状**：att_scores 已解释 QKV，后面多个台阶正文为空；掩码一项主要写先屏蔽再 Softmax，仍需同一样例串联坐标与信息来源。

**将来修改**：用固定 Q/K/V 连续产生 S、掩码、A、Y，再合头；把每题输入已由谁提供讲明。

**验收条件**：

- 明确 query/key/value、两条 token 轴和保留的 batch/head。
- 在题前教负无穷、布尔掩码、换最后两轴以及加权读取。
- 台阶均为完整 AttentionCore/MHA 的必要操作，不设无关数值题。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 06 · 反向从已有输出逐路返回

条目：`logic-part-06-attention-back` · 优先级 P2；状态见网页清单。

**已确认的现状**：两道反向台阶已有代码测评；完整 AttentionCore 又首次出现 dA、dQ/dK 组合，正文主要直接给公式。

**将来修改**：保留当前公式，增加 Y→V/A→S→Q/K 的来源说明，完整任务再组装这些计算。

**验收条件**：

- 区分台阶只返回 dV/dS 和完整 backward 返回 dQ,dK,dV 的顺序。
- 说明缩放与掩码怎样作用于反向，覆盖因果/非因果 Tq/Tk。
- 动画若保留，必须追踪实际来源与合流，不仅切换公式标题。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 06 · 头、投影、子模块与残差的构造

条目：`logic-part-06-mha-block` · 优先级 P1；状态见网页清单。

**已确认的现状**：MHA 构造器已创建 qkv/proj/core，Block 已创建两个 Norm、Attention 和 MLP；独立 Block 补了构造，但 MHA 没有独立构造交付。

**将来修改**：在认识子模块后解释投影、split/concatenate 与两条残差；补齐完整 MHA 的构造责任，接通独立 Linear 的实际复用。

**验收条件**：

- 学习者亲自创建并验证 qkv、proj、core，清楚 C、3C、H、d。
- Block 每条残差明确前向分叉和反向合流，不仅呈现雅可比表达。
- 独立 Block 实际使用约定的个人子模块，验证参数收集、缓存实例和维度。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`scripts/build_independence.py`。

跨 Part 先决项：`logic-part-04-independent-linear`。

#### Part 06 · RMSNorm 主线与 Dropout/LayerNorm 对照分清

条目：`logic-part-06-architecture-boundary` · 优先级 P1；状态见网页清单。

**已确认的现状**：t_gpt 的实际依赖不含 Dropout/LayerNorm，二者却作为主线必做；LayerNorm 正文说上一节 causal=False 的 core，实际上一任务为 Dropout。

**将来修改**：按既定 RMSNorm miniGPT 架构明确两者为对照/选修用途，或在另一个明确配置中接入；修复失效的上一节引用。

**验收条件**：

- 不为使用某层而无说明地切换主线架构。
- 需要实现的对照层有独立清楚的题面及用处；主线完成条件与可选内容一致。
- Dropout 先讲训练/评估与掩码，LayerNorm 先连接已学均值/去均值，再给反向。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

### 第七部分 · 组装 GPT（part-07）

#### Part 07 · 接收并实现 GPT 专用初始化

条目：`logic-part-07-gpt-init` · 优先级 P1；状态见网页清单。

**已确认的现状**：GPT 的 __init__ 已给出两张表、Block 堆叠、二维权重重采样和两条出口缩放；讲义引用规则，普通题只实现前后向。

**将来修改**：在 Block 已学完后正式讲解 GPT 的 0.02 与残差出口，逐个指出实际被处理的对象，并安排构造/初始化代码交付。

**验收条件**：

- 讲清原地改写数组与只初始化一次；前向不重复缩小权重。
- 显示单一 RNG 依序采样、Norm/偏置保留默认值和两个具体出口。
- 验证种子、参数顺序、缩放对象及重复 forward 权重不变。
- 与 Part 04 的移出条目对应，迁移后既无重复考查也无知识缺口。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`scripts/learner_layout.py`。

跨 Part 先决项：`logic-part-04-initialization-order`。

#### Part 07 · GPT 组装与共享表两路梯度

条目：`logic-part-07-gpt-flow` · 优先级 P2；状态见网页清单。

**已确认的现状**：讲义从 ids/位置相加直接到共享头与参数梯度；完整模型构造责任、位置索引广播与两路共享梯度仍需连成一次可操作流程。

**将来修改**：沿一组多 batch/token 经过两张表、Blocks、Norm 和输出头，再按逆序返回。

**验收条件**：

- 先教位置 arange 和 batch 广播，再要求写对应代码。
- 指出共享是同一 Parameter，输出头转置与未查表行也有梯度的原因可追踪。
- 列出需要保存的缓存、backward 返回 None 与各参数 grad 的变化。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 07 · 文本、词表与移位批次补语法衔接

条目：`logic-part-07-data` · 优先级 P2；状态见网页清单。

**已确认的现状**：tokenize/make_batch 已有编码实现和四项测评；正文主要从现成 ids 开始，sorted(set(text))、字典映射和反向解码在后续程序使用，说明仍不完整。

**将来修改**：从含重复字符/空格/换行的小文本建立词表，再编码、取窗口和右移标签；明确后续训练如何使用。

**验收条件**：

- 讲清字符串/排序去重/字典查表及整数数组的必要操作。
- 用两个起点和最后合法起点演示 X/targets 对齐，词表仅在约定语料上建立。
- GPT 接受编号，训练接收批次，生成最后解码；数据流在题前可复述。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

### 第八部分 · 优化与训练（part-08）

#### Part 08 · 优化器状态需要自己建立

条目：`logic-part-08-optimizer` · 优先级 P1；状态见网页清单。

**已确认的现状**：AdamW.__init__ 已创建 params/m/v/t，本题只留 clip_grad 和 step；未发现后续独立 AdamW 构造题。

**将来修改**：先展示参数、梯度、状态、步数的一次与连续两次更新，再补齐状态创建的学习者实现。

**验收条件**：

- 学生创建同形 m/v 与单一步数，理解它们跨步保留。
- 先解释统一范数裁剪，再解释逐元素更新；衰减分组是本项目约定。
- 检查多个不同形状参数、两次更新、空/零梯度及共享对象更新一次。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 08 · 固定批次训练先闭环再扩展

条目：`logic-part-08-train-focus` · 优先级 P2；状态见网页清单。

**已确认的现状**：固定批次 train 是真实实现题，但正文同时引入 micro-batch 加权、独立评估和 warmup/cosine，后两者没有本题实现交付。

**将来修改**：主线先完成一次/多次固定批次训练；高级累计和调度放明确拓展，评估说明输入、输出与缓存时机。

**验收条件**：

- 一张流程对应清零、前向、损失、反向、裁剪、更新及更新前 loss。
- 优化器在循环外创建，传入 model/批次与返回 loss 数组明确。
- 主文不把未来拓展写成当前必做，必要公式与已有测评一致。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 08 · 练习训练函数与实验室真正衔接

条目：`logic-part-08-training-reuse` · 优先级 P1；状态见网页清单。

**已确认的现状**：实验室 experiment 和 run_gpt.main 内部自行实现训练循环，没有直接调用个人 train；train 本身每次调用会重新创建优化器。

**将来修改**：标明当前控制循环由平台提供；规划复用个人训练步骤/连续优化器状态的接口，不能通过每批调用 train 来重置优化器。

**验收条件**：

- 网页告知个人模块、个人训练步骤和平台记录/绘图各自的职责。
- 最终若宣称使用个人训练实现，实验室与本地必须实际接通并验证。
- 训练与评估缓存时序、随机批次和状态跨步保留在实现前讲明。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`exercises/_transformer_lab.py`、`exercises/run_gpt.py`。

### 第九部分 · 生成与保存（part-09）

#### Part 09 · 采样与窗口生成先演算再实现

条目：`logic-part-09-generation` · 优先级 P2；状态见网页清单。

**已确认的现状**：已有真实实现与随机/滑窗检查，但采样并列规则直接给 stable argsort；生成正文主要是位置/缓存限制，核心取末位置和追加的语法依赖短表。

**将来修改**：沿分数→温度→top-k→概率→编号，再完整走两步窗口截取、末位置预测和追加。

**验收条件**：

- 解释 argsort、切片、choice 和 concatenate 的必要用法以及 RNG 由谁持有。
- 说明训练输入和生成前缀的区别、context 和返回原提示的约定。
- KV cache 留拓展，保留多 batch 且实际模型参数/梯度不被生成更改。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 09 · 检查点要先教状态如何存取

条目：`logic-part-09-checkpoint` · 优先级 P2；状态见网页清单。

**已确认的现状**：已有两个函数 TODO 和五项恢复测评，但讲义直接要求数组/JSON/NPZ/allow_pickle=False；初学者尚未得到这些序列化操作的完整说明。

**将来修改**：先列一个模型和优化器的状态清单，展示文件中的数组与元数据，再做保存→继续→恢复→重做的验收。

**验收条件**：

- 说明 path、参数顺序、形状、m/v/t、超参和 RNG 状态的读写方式。
- 用必要语法例子教 save/load 操作，给出的支持代码只注释未讲解部分。
- 区分本函数保存内容和 CLI 另存的词表、模型配置、数据进度；不把局部恢复称完整训练档案。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

#### Part 09 · 综合一步与本地完整交付的边界

条目：`logic-part-09-project-delivery` · 优先级 P2；状态见网页清单。

**已确认的现状**：project_step 是一次更新的组合题，完整训练/验证/文件读写由 run_gpt.py 的程序提供；该程序拥有自己的控制循环。

**将来修改**：明确综合题复用哪些个人模块、平台 CLI 提供哪些支持，并设计一条可重复的从用户语料训练到重载生成的验收步骤。

**验收条件**：

- 题面要求与 project_step(model,optimizer,ids,targets) 的真实输入输出一致。
- 完整交付指明 standard/independent 模型来源，语料、词表、配置和状态相互对应。
- 不把平台提供的控制程序误说成学员已完整编写，参考实现不替代实际编码。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`exercises/run_gpt.py`。

#### Part 09 · 最终独立实现的全部路径复核

条目：`logic-part-09-independent-closure` · 优先级 P1；状态见网页清单。

**已确认的现状**：独立 GPT 的 __init__/forward/backward 均需写且有 16 项检查；但依赖仍包含 layer_linear，缺少 rebuild_linear，多种子层与 AdamW 构造仍来自预填代码，initialize 仍为支持函数。

**将来修改**：建立最终核心接口清单，逐项标出个人实现、已给定代码与支持工具；接通完整个人链路后再验收 from-scratch 目标。

**验收条件**：

- 核心参数管理、初始化、各层构造、前后向、优化器和训练步骤都能找到学习者实现与对应检查。
- 通过类型/来源及新尺寸运行验证真正使用独立 Linear/Block/GPT，平台测评/绘图辅助可保持提供。
- 改变接口或标准时保留历史个人文件与成绩，说明依赖迁移和必须重测的范围。
- 在完成整条链之前，不用单页 16 项通过宣称所有核心代码已独立完成。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`、`exercises/learner_core.py`、`scripts/build_independence.py`。

跨 Part 先决项：`logic-part-04-initialize-code`、`logic-part-04-constructors`、`logic-part-04-independent-linear`、`logic-part-06-mha-block`、`logic-part-08-optimizer`、`logic-part-08-training-reuse`。

### 拓展 · 典型错误修复（optional）

#### 拓展 · 修错任务继续独立于主线

条目：`logic-optional-repairs` · 优先级 P2；状态见网页清单。

**已确认的现状**：五题按设计提供可运行但错误的代码，这属于排错题；不能用“骨架已有代码”把它们计为缺少学习者实现，也不能用修错题替代主线构造任务。

**将来修改**：维持独立拓展，题前指出错误行为、输入输出与相关主线，随所属知识完成后按需做。

**验收条件**：

- 偏置、合头、残差、共享表、清零各自关联对应主线，主线不依赖修错完成。
- 排错提示和参考答案默认折叠，完整函数中继承的未来拓展不重复堆入题面。

证据：`webapp/foundations.json`、`docs/课程逻辑审查事实.json`。

## 全部任务覆盖表

此表记录本次读取范围与对应条目，不代表修订已完成。

| Part | 任务 | 当前入口 | 对应逻辑项 |
|---|---|---|---|
| part-01 | `map` | 概念判断 2 题 | `logic-part-01-overview` |
| part-01 | `architecture` | 概念判断 2 题 | `logic-part-01-overview` |
| part-02 | `shape` | 代码实现/修错，4 项检查 | `logic-part-02-array-basics` |
| part-02 | `axes` | 代码实现/修错，4 项检查 | `logic-part-02-array-basics` |
| part-02 | `index` | 代码实现/修错，5 项检查 | `logic-part-02-array-basics` |
| part-02 | `broadcast` | 代码实现/修错，4 项检查 | `logic-part-02-broadcast-reduce` |
| part-02 | `reduce` | 代码实现/修错，4 项检查 | `logic-part-02-broadcast-reduce` |
| part-02 | `mean_backward` | 代码实现/修错，4 项检查 | `logic-part-02-gradient-bridges` |
| part-02 | `center_backward` | 代码实现/修错，4 项检查 | `logic-part-02-gradient-bridges` |
| part-02 | `matmul` | 代码实现/修错，4 项检查 | `logic-part-02-shape-path` |
| part-02 | `reshape` | 代码实现/修错，4 项检查 | `logic-part-02-shape-path` |
| part-02 | `restore` | 代码实现/修错，4 项检查 | `logic-part-02-shape-path` |
| part-02 | `transpose` | 代码实现/修错，4 项检查 | `logic-part-02-shape-path` |
| part-02 | `heads` | 代码实现/修错，4 项检查 | `logic-part-02-shape-path` |
| part-02 | `copy` | 代码实现/修错，4 项检查 | `logic-part-02-copy-scatter` |
| part-02 | `scatter` | 代码实现/修错，4 项检查 | `logic-part-02-copy-scatter` |
| part-02 | `random` | 代码实现/修错，4 项检查 | `logic-part-02-random` |
| part-02 | `random_sequence` | 代码实现/修错，4 项检查 | `logic-part-02-random` |
| part-04 | `infrastructure` | 代码实现/修错，4 项检查 | `logic-part-04-module-python` |
| part-04 | `initialization` | 概念判断 2 题 | `logic-part-04-initialize-code`、`logic-part-04-initialization-order` |
| part-04 | `layer_scale` | 代码实现/修错，8 项检查 | `logic-part-04-constructors`、`logic-part-04-layer-teaching` |
| part-04 | `linear` | 代码实现/修错，6 项检查 | `logic-part-04-layer-teaching` |
| part-04 | `layer_linear` | 代码实现/修错，8 项检查 | `logic-part-04-initialize-code`、`logic-part-04-constructors`、`logic-part-04-layer-teaching` |
| part-04 | `layer_embedding` | 代码实现/修错，8 项检查 | `logic-part-04-constructors`、`logic-part-04-layer-teaching` |
| part-04 | `layer_gelu` | 代码实现/修错，9 项检查 | `logic-part-04-layer-teaching` |
| part-04 | `layer_rmsnorm` | 代码实现/修错，8 项检查 | `logic-part-04-constructors`、`logic-part-04-layer-teaching` |
| part-04 | `layer_mlp` | 代码实现/修错，9 项检查 | `logic-part-04-constructors`、`logic-part-04-layer-teaching` |
| part-04 | `rebuild_linear` | 代码实现/修错，8 项检查 | `logic-part-04-initialize-code`、`logic-part-04-constructors`、`logic-part-04-independent-linear` |
| part-05 | `t_softmax` | 代码实现/修错，6 项检查 | `logic-part-05-softmax` |
| part-05 | `t_loss` | 代码实现/修错，5 项检查 | `logic-part-05-loss` |
| part-06 | `att_scores` | 代码实现/修错，3 项检查 | `logic-part-06-attention-front` |
| part-06 | `att_probs` | 代码实现/修错，3 项检查 | `logic-part-06-attention-front` |
| part-06 | `att_weighted` | 代码实现/修错，3 项检查 | `logic-part-06-attention-front` |
| part-06 | `att_merge` | 代码实现/修错，3 项检查 | `logic-part-06-attention-front` |
| part-06 | `att_value_backward` | 代码实现/修错，4 项检查 | `logic-part-06-attention-back` |
| part-06 | `att_score_backward` | 代码实现/修错，4 项检查 | `logic-part-06-attention-back` |
| part-06 | `t_attention` | 代码实现/修错，8 项检查 | `logic-part-06-attention-back` |
| part-06 | `t_mha` | 代码实现/修错，6 项检查 | `logic-part-06-mha-block` |
| part-06 | `t_block` | 代码实现/修错，6 项检查 | `logic-part-06-mha-block` |
| part-06 | `t_dropout` | 代码实现/修错，6 项检查 | `logic-part-06-architecture-boundary` |
| part-06 | `t_layernorm` | 代码实现/修错，6 项检查 | `logic-part-06-architecture-boundary` |
| part-06 | `rebuild_block` | 代码实现/修错，6 项检查 | `logic-part-06-mha-block` |
| part-07 | `t_gpt` | 代码实现/修错，6 项检查 | `logic-part-07-gpt-init`、`logic-part-07-gpt-flow` |
| part-07 | `t_data` | 代码实现/修错，4 项检查 | `logic-part-07-data` |
| part-08 | `t_optim` | 代码实现/修错，4 项检查 | `logic-part-08-optimizer` |
| part-08 | `t_train` | 代码实现/修错，4 项检查 | `logic-part-08-train-focus`、`logic-part-08-training-reuse` |
| part-09 | `t_sample` | 代码实现/修错，4 项检查 | `logic-part-09-generation` |
| part-09 | `t_generate` | 代码实现/修错，3 项检查 | `logic-part-09-generation` |
| part-09 | `t_checkpoint` | 代码实现/修错，5 项检查 | `logic-part-09-checkpoint` |
| part-09 | `t_project` | 代码实现/修错，8 项检查 | `logic-part-09-project-delivery` |
| part-09 | `rebuild_gpt` | 代码实现/修错，16 项检查 | `logic-part-09-independent-closure` |
| optional | `fix_bias` | 代码实现/修错，3 项检查 | `logic-optional-repairs` |
| optional | `fix_heads` | 代码实现/修错，3 项检查 | `logic-optional-repairs` |
| optional | `fix_residual` | 代码实现/修错，3 项检查 | `logic-optional-repairs` |
| optional | `fix_shared` | 代码实现/修错，6 项检查 | `logic-optional-repairs` |
| optional | `fix_clear` | 代码实现/修错，4 项检查 | `logic-optional-repairs` |

## 按 Part 执行约定

用户说“执行 Part 04 清单中的所有修改”时，先汇总 roadmap 中 part=part-04 的未完成项及既有全局标准在本 part 的适用项；完成该 part 内的讲解、题面、骨架、真实实现与测评衔接，并同步记录。跨 part 接收项保留到用户指定的对应部分，不因当前任务偷偷重构后续课程。

同一 part 的缺少实现与知识提前使用必须同时处理。新增编码题或改接口/评分需采用可追踪的版本/迁移方式，保留个人文件、草稿和历史成绩。已修改的内容最多标 review，用户验收后才能 accepted。

暂缓 Part 03 不能被“执行其他 part 所有修改”带动；网站全局清单入口不进入章节正文。

## 后续防漏核对

执行每个 part 前后都问：① 前置对象和语法讲过吗；② 题前能说清输入→过程→输出吗；③ 核心函数实际留给学习者实现了吗；④ 有真实代码测评吗；⑤ 后续真实路径调用这个实现了吗；⑥ 拓展是否混入当前必做；⑦ 新标准迁移是否保护已有数据。

只读事实盘点可用 `python scripts/audit_learning_logic.py --output docs/课程逻辑审查事实.json` 更新；该脚本不自动判断“讲明白”，不修改课程和个人代码。

## 本轮记录检查结果

- 56/56 个任务都有对应逻辑审查项，31 个新增项目均为 todo，10 个 part 分组（含暂缓 Part 03 与拓展）全部能够由现有网页清单组件渲染。
- 原清单条目与状态保持原值；本轮前后课程 JSON、网页 JS/CSS/HTML、用例 JSON 与全部练习 Python 文件的哈希一致。
- 事实快照同时盘点骨架和参考代码接口；显式依赖无倒置。没有运行课程重建、个人实现或浏览器逐页学习测试。


## 推导位置复核（2026-11-14，侧聊授权）

- 对56任务进行正文/推导结构扫描，重点阅读 Scale、Linear、GELU、RMSNorm、MLP、损失、注意力、多头及 Block 的相关内容；结构存在独立推导区不直接判错。
- Scale 已将局部导数和链式法则贴近各自梯度讲解，取消末尾重复块，review。
- 后续待改：Linear 推导贴近三类梯度；RMSNorm 对照移出前向与反向之间；MLP 的具名前向中间量先于缓存；损失先定义 logp 再索引；注意力将 Softmax/AV 纳入完整前向链。分别记录在 derivation-placement-part04、derivation-placement-part05、derivation-placement-part06，todo。
- GELU 的标准公式与链式法则相邻，MHA 的前向形状链接反向还原、Block 的前向接逆序残差推导顺序合理，本轮未要求机械重排。后续仍按学习者指定 part 完整修订。


## Part04独立Linear执行记录（2026-11-15）

- 独立构造、初始化函数真实接入、完整高维前后向/缓存/三类梯度、OI题面和公式速查已实现（review）。新增构造与两个当前类实际串联测评，当前10项通过代表本题交付可验证，不代表后续最终GPT已接入。
- 本Part构造台阶、推导就地与对应来源视图组项已实现待验收。logic-part-04-independent-linear继续todo；Part06/09登记网页依赖/本地导入/内层Linear身份接收检查，本轮不修改后续课程。

## Part04收尾与Part05执行更新

- 新增基础层综合任务替换主线重复Linear；旧独立题保留可选历史，构造交付合并到layer_linear。新增组合及八组组件能力验收，章末增量与覆盖表落地，review。
- Part05两节已按对象/输入/过程/输出、必要就地推导、OI题面、任务清单和公式速查重写；来源展示分别解释行内耦合与标签选择/位置平均，原检查保留并增加缓存/标量上游/高维稳定性检查，review。
- 当前57个任务，新增一项并保留原56项历史；最新事实见JSON，状态仍以网页清单为准。未修改其他Part课程，Part06/09接收与去重复重构保留待办，Part03继续暂缓。
