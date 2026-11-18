"""本地测评网站：在浏览器里写模块代码 → 提交 → 跑判据 → 看结果。

    python webapp/server.py            # 打开 http://127.0.0.1:8000

为什么值得有这个东西：这份手册的每个阶段都靠"跑一条命令看判据"来推进，
命令行容易跑错目录、忘切实现、看到一屏 traceback 就懵。网站把这件事压成一次点击：
左边是提示（该交哪些文件、判据在测什么、常见的坑），中间是代码，右边是判据输出。

三条设计原则，读代码时可以对照：
1) **绝不覆盖你的源码**。提交只在 workspaces/<stage>/ 里的一份副本上跑；
   只有你显式点"保存到 src/"才会写入，且写之前把旧内容备份到 backups/。
2) **判据只有一份**。网站跑的就是 `stages/*.py` 和 `tests/*.py` 本身，
   通过 `MGPT_PKG=<工作区>` 让 pathsetup 用你的提交版本（见 pathsetup.py）。
   网站不重新实现任何检查，也就无法"网站说过了但命令行没过"。
3) 只用标准库。没有 flask、没有构建步骤，装不上包不影响你用这份手册。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
WORKSPACES = WEB / "workspaces"
BACKUPS = WEB / "backups"
SRC_PKG = ROOT / "src"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manual_assets import by_url        # noqa: E402  哪些 URL 该存在，只有这一处定义

# /manual/xxx 只有出现在这张表里才读得出文件；表外的路径连 resolve 都不做
MANUAL_URLS = by_url()
MANUAL_TYPES = {".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8"}
sys.path.insert(0, str(HERE))            # 让 checks.py 在 "python webapp/server.py" 下可导入
from checks import CHECKS, PRELUDE, checks_for    # noqa: E402  单项快测的登记表 + 共用前置


# ------------------------------------------------------------------ 阶段登记表
# 这一份同时驱动左边的"提交提示"，所以它既是路由表也是教材的一部分。
# available=False 的阶段会在页面上灰掉并告诉你它还没发布 —— 别猜，等讲义。
STAGES = [
    dict(
        id="s01",
        title="阶段 1 · 标量自动微分",
        available=True,
        manual="docs/manual/s01.html",
        judge="stages/s01_fit_line.py",
        files=["minigpt/scalar.py", "minigpt/exercises.py"],
        todos=["scalar.py TODO-1 __mul__", "scalar.py TODO-2 __pow__",
               "scalar.py TODO-3 backward()", "exercises.py TODO-1A fit_by_autograd"],
        mins="20~60 分钟",
        criteria=[
            "学到斜率 a / 截距 b（第 A 部分，已给）",
            "最终 MSE ≤ 1e-4；loss 下降 ≥ 1e3 倍",
            "两种方法 a/b 之差 ≤ 1e-6 ← 手推 == 自动微分",
            "梯度校验(有限差分) ← 本阶段的全部意义",
        ],
        tips=[
            "算子规则里读的是 out.grad，不是 self.grad：写错的话梯度全 0，且不报错。",
            "是 += 不是 =：一个 Value 会被多处用到，梯度要沿每条路径累加。",
            "backward() 三件事按顺序：拓扑排序（要去重）→ self.grad = 1.0 → 逆序执行规则。",
            "别先去看 solutions/。卡住 30 分钟以上再看，看完关掉自己重写。",
        ],
    ),
    dict(
        id="s02",
        title="阶段 2 · 非线性与容量",
        available=True,
        manual="docs/manual/s02.html",
        judge="stages/s02_fit_curve.py",
        files=["minigpt/scalar.py", "minigpt/exercises.py"],
        todos=["scalar.py TODO-4 tanh()", "exercises.py TODO-2A ScalarMLP.forward",
               "exercises.py TODO-2B train_mlp"],
        mins="30~90 分钟",
        criteria=[
            "线性模型 MSE ≥ 0.05（对照组必须学不动）",
            "MLP 最终 MSE ≤ 6e-3，且比线性好 ≥ 20 倍",
            "MLP loss 下降倍数 ≥ 80",
            "梯度校验(25 参数 MLP)",
        ],
        tips=[
            "先读讲义 §2.3–§2.9：激活函数的定义、导数、饱和、零中心，以及为什么 FFN 最后用 GELU。这一节讲的是**已给你的代码**，不是你要写的。",
            "z 的起点用 self.b1[j]，不是 Value(0)：漏 bias 的症状是形状对、整体偏移。",
            "out = out + W2[j]*hidden[j] 是建图累加；写成 out = W2[j]*hidden[j] 就只剩最后一个神经元。",
            "每轮顺序：zero_grad → loss → backward → 更新。写成「更新在前」会悄悄用错梯度。",
            "tanh 的导数是 1−t²（用 t 自己，不用 x）。",
        ],
    ),
    dict(
        id="s03a",
        title="阶段 3a · 形状、轴与第一个层",
        available=True,
        manual="docs/manual/s03a.html",
        judge="tests/test_layers_a.py",
        files=["minigpt/nn.py"],
        todos=["nn.py TODO-5A Scale", "nn.py TODO-5/6 Linear"],
        mins="1~2 小时（地基 + 一次完整跑绿）",
        criteria=[
            "Scale 的 gradcheck + 梯度累加（两次 backward 要翻倍）",
            "Linear / Linear 无 bias 两个 gradcheck",
            "Linear 的 dW / db / dx 与逐元素手算对账（B=1,T=2,in=2,out=3）",
        ],
        tips=[
            "先写 Scale，别先写 Linear：它没有形状问题，只练流程（讲义 §3）。",
            "`axis=k` 就是被消失的那根轴；想合成什么形状就谁转置，见 §2.5 那张速查表。",
            "不会矩阵微分也能推反向：§2.6 给了三步机械流程（逐元素写出来 → 数下标 → 收回形状），"
            "外加唯一的新东西 —— 广播的反向是求和。写 `Linear.backward` 卡住就先回那一节。",
            "dW 和 db 都压掉样本轴 N，因为 W、b 被所有位置共用；dx 不压轴，因为 x 和 y 一一对应（§6.3）。",
            "判据里第 6~8 条是拿定义算的，和有限差分互相独立 —— 两个都过才说明不是碰巧形状对得上。",
            "一次 forward 配一次 backward：中间插第二次 forward，缓存被覆盖，形状全对但梯度算在另一条数据上（§4.1）。",
        ],
    ),
    dict(
        id="s03b",
        title="阶段 3b · 四个层的三件产出",
        available=True,
        manual="docs/manual/s03b.html",
        judge="tests/test_layers_b.py",
        files=["minigpt/nn.py"],
        todos=["TODO-9 GELU", "TODO-8 RMSNorm", "TODO-7 Embedding", "TODO-10 MLP"],
        mins="1~2 小时（每层都只是换算法，协议你已经熟了）",
        criteria=[
            "GELU / RMSNorm / MLP 三个 gradcheck",
            "Embedding 参数梯度 + 重复 id 累加=3·C + 未用行梯度为 0",
        ],
        tips=[
            "一次只写一层，写完立刻跑判据（`python webapp/checks.py TODO-9`）。",
            r"RMSNorm 少了 $-r\hat x\,s$ 这一项，梯度校验一定挂 —— 归一化把整行 $C$ 个分量耦合在一起了。",
            "Embedding 必须 np.add.at：fancy 赋值对重复索引是覆盖，不是累加。",
            "GELU 没有参数，所以它的 backward 只有返回值那一件产出；MLP 反过来，三件产出要串两次。",
        ],
    ),
    dict(
        id="s03c",
        title="阶段 3c · CrossEntropy 与测谎仪自检",
        available=True,
        manual="docs/manual/s03c.html",
        judge="tests/test_layers_c.py",
        files=["minigpt/nn.py"],
        todos=["TODO-11 CrossEntropy"],
        mins="40~80 分钟（一个层，但它是本阶段最容易写错的）",
        criteria=[
            "CrossEntropy gradcheck（输出是标量，余因子为 1）",
            "logits=900 不溢出；均匀 logits 的 loss $=\\ln V$",
            "gradcheck 自检：已知对的层说 OK，故意漏系数 2 的层必须报错",
        ],
        tips=[
            r"CrossEntropy 用 mean 不用 sum，否则等于把 lr 放大 $BT$ 倍；随机初始化时 loss 应在 $\ln V$ 附近。",
            "减最大值那一步不是可选的数值技巧：不减的话 e^900 直接 inf，而 loss 的数值结果一模一样 —— 只有 gradcheck 抓得住。",
            "梯度是 `(p - onehot) / N`，不是 `(p - onehot)`：忘了除以 B·T 的话后面每个阶段的 lr 都要重新调。",
            "最后两条判据考的是 gradcheck 自己。以后拿到一句『最大相对误差 3e-2』，你先跑这两条决定怀疑谁（§4）。",
        ],
    ),
    dict(
        id="s04",
        title="阶段 4 · 因果自注意力",
        available=True,
        manual="docs/manual/s04.html",
        judge="tests/test_attention.py",
        files=["minigpt/attention.py"],
        todos=["attention.py TODO-12 forward", "TODO-13 backward", "TODO-14 softmax 反向"],
        mins="2~4 小时",
        criteria=[
            "单头/多头注意力 gradcheck（步长 1e-5 + 绝对容差 1e-9）",
            "softmax 反向的两个守恒性质：每行梯度之和为 0、常数余向量下为 0",
            "改未来 token 不影响过去（逐位相同）← 掩码正确性；改当前 token 必须改变当前输出",
            "参数量 4d²+3d 与无死参数（k 没有 bias）",
        ],
        tips=[
            "掩码必须加在 softmax **之前**；加在后面等于没掩（模型能偷看未来）。",
            r"用 `-1e18` 而不是 `-inf`：$0\cdot\infty=\mathrm{nan}$ 会在反向里毁掉整条链。",
            "被掩掉的位置不用额外处理：exp(-1e18-max) 精确等于 0，梯度天然是 0。"
            "顺手写 np.where(mask==0, 0, g) 会把条件弄反，清掉允许位置的梯度。",
            r"softmax 反向是 $g=p\odot(gp-\sum(gp\odot p))$，那个减号就是「每行和为 1」的约束。",
        ],
    ),
    dict(
        id="s05",
        title="阶段 5 · 完整 GPT + 训练循环",
        available=True,
        manual="docs/manual/s05.html",   # 由 scripts/md_to_html.py 从 docs/stage-05-*.md 生成
        judge="stages/s05_char_lm.py",
        files=["minigpt/gpt.py", "minigpt/optim.py", "minigpt/tokenizer.py", "minigpt/dataset.py"],
        todos=["gpt.py TODO-15 Block.forward", "gpt.py TODO-16 Block.backward",
               "gpt.py TODO-17 TiedHead 的两路梯度", "gpt.py TODO-18 GPT.forward",
               "gpt.py TODO-19 loss_backward 的移位", "tokenizer.py TODO-20 encode/decode",
               "dataset.py TODO-21 Batcher 的随机窗口", "optim.py TODO-22 AdamW.step",
               "optim.py TODO-23 clip_grad_norm", "optim.py TODO-24 lr_at"],
        mins="4~8 小时（本手册代码量最大的一章）",
        criteria=[
            "分词往返 / 词表口径 / 陌生字符按契约处理（3 条）",
            "批次形状、$y[:, :-1] == x[:, 1:]$、换 rng 换窗口、train/val 按行不重叠",
            "Block / TiedHead / GPT 整机有限差分 + tied 两路累加 + 参数量 $Vd+ctx\\,d+L(12d^2+10d)+d$",
            "残差出口置零 ⇒ Block 是恒等映射（forward 与 backward 各一半）",
            "初始 loss $\\approx \\ln V$；单批过拟合 $\\to \\le 0.05$；模式数据学得动、随机数据学不动",
            "AdamW 单步幅度 $=$ lr、decay 只作用于非豁免参数、clip 后全局范数 $=$ 上限、lr 调度形状",
        ],
        tips=[
            "TODO-19 的移位：logits 去掉**最后一位**、targets 去掉**第一位**。写反了不报错，"
            "loss 掉得飞快、生成全是复读 —— 判据'随机数据学不动'专抓它。",
            "TODO-17 的 `weight.grad +=`：tied 权重同时被查表层和输出层写，写成 = 时单独测输出头看不出问题。",
            "TODO-16 最容易漏'原样透传'那一路：残差的梯度是 g（直通）+ 支路回传，两条都要。",
            "TODO-22 的偏差修正 1-β^t 不能省：m、v 初值是 0，前几步的均值被严重低估，省了等于第一步只走 0.1·lr。",
            "TODO-21 里 rng.integers 的上界是开区间：写 max_start 会漏掉最后一个合法窗口，写 ids.size 会越界。",
            "clip_grad_norm 要在 optimizer.step() **之前**，反过来等于没裁；范数是所有张量拼起来的全局 L2，不是逐张量。",
            "顺序建议：TODO-15/16 → 17 → 18 → 19（每写完跑一次单项），再 20/21（数据侧），最后 22/23/24（优化器）。",
            "写完 gpt.py 之后 `python tests/test_model.py`（阶段 4 的考卷）才会真的走到 Block/整机那 9 条 —— 之前它们是靠答案版跑绿的。",
        ],
    ),
    dict(
        id="s06",
        title="阶段 6 · 真数据训练 + 检查点",
        available=True,
        manual="docs/manual/s06.html",   # 由 scripts/md_to_html.py 从 docs/stage-06-*.md 生成
        judge="stages/s06_train.py",
        files=["minigpt/checkpoint.py"],
        todos=["checkpoint.py TODO-25 save_checkpoint", "checkpoint.py TODO-26 load_checkpoint",
               "checkpoint.py TODO-27 save/load_train_state"],
        mins="1.5~3 小时（新内容只有存档，训练是把阶段 5 的循环拉长）",
        criteria=[
            "语料读得通、分词往返、train/val 整行级零重叠",
            "真语料上 loss 下降 ≥ 1.5× 且 $\\le \\tfrac34 \\ln V$；贪心下一字符 ≥ 0.30",
            "参考解析器能读走 MGPT 文件（魔数 / 版本 / config JSON / fp32 / 张量名与 tied 唯一性）",
            "加载后 logits 一致、配置/魔数/形状三种不匹配都拒绝、赋值是原地写",
            "断点续训与一口气训完在恢复段逐步吻合（最大差 ≤ 1e-4）",
        ],
        tips=[
            "TODO-25 用 `model.named_parameters()` 而不是自己数张量：它按 id 去重，tied 的输出头天然只有一份。",
            "权重存 `\"<f4\"`、优化器状态存 `\"<f8\"`：前者要给 C++ 读且文件减半，后者只给自己读而且要能逐位续训。",
            "TODO-26 的赋值必须 `p.data[...] = t` 而不是 `p.data = t`：tied 头和优化器握着的是同一块内存，换了对象等于什么都没加载。",
            "配置校验不是官僚主义：vocab 改了但形状碰巧兼容时，模型照样能跑、loss 照样是个数 —— 只是搬错了零件。",
            "续训对账要三样一起恢复：权重、Adam 的一二阶矩、`opt.step_count`。漏 step_count 时偏差修正的 t 从 1 重来，第一步就跳。",
            "lr 调度看的是**总步数**而不是剩余步数（判据里 B 组第一段特意传 horizon）；weight_decay 不在状态存档里，换脚本要自己配。",
            "判据用了一个不认你代码的参考解析器：'我自己存、自己读得回来'证明不了格式合规。",
        ],
    ),
    dict(
        id="s07",
        title="阶段 7 · 采样与生成质量",
        available=True,
        manual="docs/manual/s07.html",   # 由 scripts/md_to_html.py 从 docs/stage-07-*.md 生成
        judge="stages/s07_generate.py",
        files=["minigpt/sampler.py"],
        todos=["sampler.py TODO-28 apply_penalty", "sampler.py TODO-29 keep_top_k",
               "sampler.py TODO-30 keep_top_p", "sampler.py TODO-31 sample_next"],
        mins="2~4 小时",
        criteria=[
            "温度：熵单调上升、$T=0$ 等于 argmax、抽样频率与 softmax 的 L1 ≤ 0.14",
            "top-k：候选集恰好是前 k 名、升温不引入新候选、并列都保留、$k=0$ 不剪",
            "top-p：核覆盖原分布 $\\ge p$ 的质量、是概率降序的前缀、one-hot 不冒出 nan",
            "重复惩罚：正分除 / 负分乘、不改调用者的数组、被罚词经验频率下降",
            "端到端：贪心续写与原文一致率 ≥ 0.70、top_k=1 等价于贪心、超长 prompt 走滑窗",
        ],
        tips=[
            "顺序是这一章的全部内容：惩罚 → 温度 → top-k → softmax → top-p → 抽签。"
            "但只有三处顺序真的换了会出事：惩罚必须在贪心岔口之前、温度必须在 softmax 之前、"
            "top-p 必须在 softmax 之后。top-k 与温度可交换（除以 T 是单调缩放，切出来同一批）。",
            "`np.asarray(x, dtype=float64).reshape(-1)` 在 dtype 已经对的时候返回**视图**，就地 /= 会毁掉模型刚算出的那一行 logits。",
            "top-k 的比较要用 `<` 不是 `<=`：并列最高分时两个都该留下，写成 `<=` 会把它们一起掩掉，softmax 出来是全行同值的怪分布。",
            "top-p 要在概率上切、且保证覆盖 $\\ge p$：`cum <= p` 那种写法留下的质量总是刚不到 p；p<=0 时记得兜底留一个，否则 0/0 = nan。",
            "rng 是参数不是内部新建：同一段文本能复现，靠的是调用方决定种子。",
            "这台模型很自信（loss 0.15），T=1.0 时 24 个种子还是同一个词 —— 判据把多样性设在 T=2.0，别以为是采样坏了。",
        ],
    ),
]


# ------------------------------------------------------------------ 测评执行
def read_stage_file(stage: dict, rel: str) -> str:
    """编辑器内容：**永远优先 src/**（你真正的那份）。

    早先版本优先给工作区副本，结果我用答案跑过一次测评之后，编辑器里显示的就是答案 ——
    等于自己给自己剧透。工作区只是"跑判据用的临时拷贝"，不该成为阅读来源。
    src/ 里确实没有这个文件时（阶段还没发布），才退回 solutions/ 让你看清接口长什么样。
    """
    src = SRC_PKG / rel
    if src.exists():
        return src.read_text(encoding="utf-8")
    sol = ROOT / "solutions" / rel
    if sol.exists():
        return sol.read_text(encoding="utf-8")
    return "# 找不到这个文件\n"


def submitted_is_answer(stage: dict, files: dict) -> bool:
    """这次提交的每一份是否和 `solutions/` 里那份一字不差 —— 那是答案，不是用户的代码。

    判据记录必须说清"是谁跑的"。我每发布一个阶段都要拿答案版跑一遍验证，
    那些 pass 一旦和用户自己的结果混在同一个字段里，左栏就会出现
    「已过」配「TODO 0/7」这种自相矛盾的画面 —— 徽章只有一格，两个数说的是两件事。
    """
    if not files:
        return False
    for rel, text in files.items():
        sol = ROOT / "solutions" / rel
        if not sol.exists():
            return False
        want = sol.read_text(encoding="utf-8").replace("\r\n", "\n")
        if want != (text or "").replace("\r\n", "\n"):
            return False
    return True


def safe_rel(rel: str) -> str:
    """提交路径只允许 minigpt/ 下的 .py，防止越界写文件。"""
    rel = rel.replace("\\", "/").strip().lstrip("./")
    if not rel.startswith("minigpt/") or not rel.endswith(".py"):
        raise ValueError(f"只能提交 minigpt/ 下的 .py 文件，收到 {rel!r}")
    if ".." in Path(rel).parts:
        raise ValueError("路径里不许有 ..")
    return rel


def build_workspace(stage_id: str, files: dict[str, str]) -> Path:
    """src/minigpt 整份拷进工作区，再用提交内容覆盖 —— 你的 src/ 一个字节都不会被动。"""
    ws = WORKSPACES / stage_id
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "minigpt").mkdir(parents=True)
    for p in (SRC_PKG / "minigpt").glob("*"):
        if p.is_file() and p.suffix == ".py":
            shutil.copy2(p, ws / "minigpt" / p.name)
    for rel, content in files.items():
        target = ws / safe_rel(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return ws


PASS_RE = re.compile(r"验收通过\D*(\d+)/(\d+)")
FAIL_RE = re.compile(r"验收未通过\D*(\d+)/(\d+)")
TODO_RE = re.compile(r"还没写完 → (.+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")     # toolkit 给汇总行上色，转义码会插进关键词中间
# 必须和 src/minigpt/toolkit.py 的 JSON_MARK 一模一样。服务端故意不 import minigpt
# （原则二：网站不重新实现任何检查，也不碰你的实现），所以这里只能抄一份常量。
# 抄错的后果是可控的：split_json_channel 认不出标记就把那行原样留在文本里，
# 网站退回纯文本渲染 —— 判据一条都不会少。
JSON_MARK = "@@MGPT-JSON@@"


def classify(out: str, code: int) -> dict:
    m = PASS_RE.search(out)
    if m:
        return dict(status="pass", got=int(m.group(1)), total=int(m.group(2)))
    m = FAIL_RE.search(out)
    if m:
        return dict(status="fail", got=int(m.group(1)), total=int(m.group(2)))
    m = TODO_RE.search(out)
    if m:
        return dict(status="todo", got=0, total=0, todo=m.group(1).strip())
    return dict(status="error", got=0, total=0, todo=f"退出码 {code}")


def split_json_channel(text: str) -> tuple[str, dict | None]:
    """把 toolkit 的机器可读那一行摘出来，剩下的仍是给人看的文本。

    判据本身没变：同一份脚本、同一条命令，只是多吐一行 JSON。
    解析失败就退回纯文本渲染 —— 展示层永远不能把判据弄没。
    """
    lines, payload = [], None
    for ln in text.split("\n"):
        if JSON_MARK in ln:
            try:
                payload = json.loads(ln.split(JSON_MARK, 1)[1])
            except (json.JSONDecodeError, ValueError):
                lines.append(ln)                 # 看不懂就原样留着，别偷偷丢数据
            continue
        lines.append(ln)
    return "\n".join(lines), payload


def run_judge(ws: Path, judge: str, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env["MGPT_PKG"] = str(ws)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["MGPT_JSON"] = "1"                      # 网站要卡片和曲线，终端用户看不到这行
    env.pop("MGPT_ANSWER", None)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, judge], cwd=str(ROOT), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        raw = ANSI_RE.sub("", proc.stdout.decode("utf-8", errors="replace"))
        code = proc.returncode
    except subprocess.TimeoutExpired:
        raw, code = f"超时（>{timeout}s）。阶段脚本不该跑这么久，检查一下是不是写了死循环。", -1
    text, data = split_json_channel(raw)
    dt = time.perf_counter() - t0
    out = dict(output=text, exit=code, secs=round(dt, 1), **classify(raw, code))
    if data:
        out["criteria"] = data["rows"]
        out["curves"] = data["curves"]
        out["fits"] = data.get("fits", [])        # 拟合图像：阶段脚本自己算的，网站只画
        out["got"], out["total"] = data["got"], data["total"]
        out["elapsed"] = data["elapsed"]
    return out


# ------------------------------------------------------------------ 调参试验台
def run_lab(stage: dict, files: dict, params: dict) -> dict:
    """跑 webapp/lab.py：用你提交的实现 + 你调的参数，只出观察值，不写进度。

    和 run_judge 的关键区别有三个：
      · 参数由前端传来（判据那套是写死在脚本里的）；
      · 结果**不进 state.json** —— 试验台跑一百次也不该改变"你完成到哪"；
      · 超时给得更短：试验是要立刻看到反馈的，等 15 秒还没结果就是参数不对。
    """
    from lab import KNOBS
    if stage["id"] not in KNOBS:
        return dict(status="pending", output="这个阶段还没有试验台")
    ws = build_workspace(stage["id"], files) if files else WORKSPACES / stage["id"]
    env = dict(os.environ)
    env["MGPT_PKG"] = str(ws)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "webapp/lab.py", stage["id"],
                               json.dumps(params, ensure_ascii=False)],
                              cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=120)
        raw = ANSI_RE.sub("", proc.stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        return dict(status="error", output="试验台超时（>120s）。参数太极端了？")
    text, data = split_json_channel(raw)
    if not data:
        return dict(status="error", output=text or "lab.py 没有输出结果")
    return dict(status="ok", output=text, secs=round(time.perf_counter() - t0, 1),
                criteria=data["rows"], curves=data["curves"], mats=data.get("mats", []),
                flow=data.get("flow", []), bars=data.get("bars", []), text=data.get("text", []),
                title=data["title"], subtitle=data["subtitle"])


# ------------------------------------------------------------------ 单项快测
# PRELUDE（brief / card）住在 checks.py 里：网站这里和 `python webapp/checks.py <id>`
# 命令行跑的是同一段代码、同一个前置，所以两边不可能给出不同结论。


def run_check(ws: Path, check: dict, timeout: int = 180) -> dict:
    """跑一条最小检查：只测某个 TODO，不要求整卷能跑通。

    检查代码外面套一层 try：
      · NotImplementedError → BLOCKED（"被前置依赖挡住"，说明还得先写哪个 TODO）
      · 其它异常           → FAIL，并把异常本身当结论显示
    不套的话，形状写错会变成一屏 traceback + "出错"，对新手来说和失败没区别但更难读；
    现在能看到 `FAIL ValueError: matmul: ...`，一眼就知道是矩阵乘法那一步的形状不对。
    """
    body = textwrap.indent(check["code"].strip(), "    ")
    script = ws / "_check.py"
    script.write_text(
        PRELUDE + "import traceback\ntry:\n" + body +
        "\nexcept NotImplementedError as _e:\n"
        "    print('BLOCKED', _e)\n"
        "except Exception as _e:\n"
        "    print('FAIL', type(_e).__name__ + ': ' + str(_e))\n"
        "    traceback.print_exc(limit=6)\n",
        encoding="utf-8")
    env = dict(os.environ)
    env["MGPT_PKG"] = str(ws)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("MGPT_ANSWER", None)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        text = ANSI_RE.sub("", proc.stdout.decode("utf-8", "replace")).rstrip()
        code = proc.returncode
    except subprocess.TimeoutExpired:
        text, code = f"这条检查超过 {timeout}s，多半是循环没退出条件。", -1
    first = next((ln for ln in text.split("\n") if ln.startswith(("PASS", "FAIL", "BLOCKED"))), "")
    head, _, rest = first.partition(" ")
    if head == "BLOCKED":
        status, line = "blocked", f"被前置依赖挡住：{rest}"
    elif head == "PASS":
        status, line = "pass", rest
    elif head == "FAIL":
        status, line = "fail", rest
    else:
        status, line = "error", f"退出码 {code}，没打印 PASS/FAIL"
    return dict(status=status, line=line, output=text, secs=round(time.perf_counter() - t0, 2))


# ------------------------------------------------------------------ 保存到 src
def save_to_src(stage: dict, files: dict[str, str]) -> list[str]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    saved = []
    for rel, content in files.items():
        rel = safe_rel(rel)
        target = SRC_PKG / rel
        if target.exists():                        # 覆盖前必备份，出事能找回
            bak = BACKUPS / stamp / rel
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, bak)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        saved.append(rel)
    return saved


# ------------------------------------------------------------------ 进度
# 三件事合起来才是"进度"，单看任何一个都会骗人：
#   1) 实现度：这个 TODO 的占位 raise 还在不在（还在=没写；被 return 挡在后面=死代码，算写完）。
#      看的是 src/ 和"最近一次提交的内容"里较新的那份，所以在网页编辑器里写完也算数。
#   2) 单项检查：ACT-*/GIV-*/TODO-* 各条最近一次结论
#   3) 整卷判据：那个阶段的 suite 结果
# 存在 webapp/state.json，换浏览器/重开服务都不丢。
STATE_PATH = WEB / "state.json"
PANE_KEYS = ("w1", "w3", "ch")      # 左栏宽 / 右栏宽 / 控制台高 —— 和前端 PANES 表一致
TODO_SPLIT = re.compile(r"TODO-([0-9]+(?:/[0-9]+)*[A-Za-z]?)")
# 抓整个 "TODO-x"，连前缀一起：登记表里的 id 也带前缀，少抓一个字符就会全都对不上，
# 结果是"占位符统统不见了"，进度条直接 100%。
RAISE_RE = re.compile(r'NotImplementedError\("(TODO-[0-9A-Za-z/]+)')


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001  没有/坏了都当空档
        return {}


def src_stamp(stage: dict) -> float:
    """这个阶段要交的几个文件里最新的修改时间。用来判断整卷结果是不是过期了。"""
    out = 0.0
    for rel in stage["files"]:
        p = SRC_PKG / rel
        if p.exists():
            out = max(out, p.stat().st_mtime)
    return out


def record(stage_id: str, *, check: tuple | None = None, suite: dict | None = None) -> None:
    st = load_state()
    d = st.setdefault(stage_id, {"checks": {}, "suite": {}})
    if check:
        d["checks"][check[0]] = {"status": check[1], "when": time.strftime("%m-%d %H:%M"),
                                 "answer": bool(check[2]) if len(check) > 2 else False}
    if suite:
        # mt 记下"这次整卷跑的是哪一版代码"：之后 src/ 又变了，结果就该标成过期，
        # 否则你会看到"4/4 写完 + 单项全过"却挂着一个上次留下的旧结论。
        d["suite"] = dict(suite, when=time.strftime("%m-%d %H:%M"), mt=src_stamp(find_stage(stage_id)))
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ 结果窗口 / 分栏
LASTRUN = WEB / "last_run.json"
_OUT_MAX = 60_000          # 一次判据输出上限 60KB：ASCII 图 + traceback 也够了，别把磁盘喂胖


def remember_run(stage_id: str, res: dict, kind: str = "suite") -> None:
    """把最近一次结果单独存一份，给「↗ 结果窗口」轮询。

    为什么不塞进 state.json：那个文件每次刷新都要读（/api/stages），
    而判据输出动辄几十 KB。分开放，state.json 继续只当"进度档案"。
    """
    runs = {}
    if LASTRUN.exists():
        try:
            runs = json.loads(LASTRUN.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001  坏了就当没有
            runs = {}
    text = (res.get("output") or "")
    runs[stage_id] = dict(
        kind=kind, when=time.strftime("%m-%d %H:%M:%S"),
        status=res.get("status"), got=res.get("got"), total=res.get("total"),
        secs=res.get("secs"), elapsed=res.get("elapsed"),
        title=res.get("title", ""), criteria=res.get("criteria") or [],
        answer=bool(res.get("answer")), id=res.get("id", ""), label=res.get("label", ""),
        curves=res.get("curves") or [], fits=res.get("fits") or [],
        line=res.get("line", ""), output=text[:_OUT_MAX], truncated=len(text) > _OUT_MAX)
    LASTRUN.parent.mkdir(parents=True, exist_ok=True)
    LASTRUN.write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")


def load_run(stage_id: str) -> dict:
    try:
        return json.loads(LASTRUN.read_text(encoding="utf-8")).get(stage_id) or {}
    except Exception:                                       # noqa: BLE001
        return {}


def save_panes(p: dict) -> None:
    """三栏宽度存进 state.json 的 _panes（不是某个浏览器的事，是这台机器/这个仓库的布局）。"""
    st = load_state()
    clean = {k: int(v) for k, v in (p or {}).items() if k in PANE_KEYS and str(v).isdigit()
             or k in PANE_KEYS and isinstance(v, (int, float))}
    st["_panes"] = clean
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def get_panes() -> dict:
    return load_state().get("_panes") or {}


def expand_todo(text: str) -> list[str]:
    """登记表里的 "nn.py TODO-5/6 Linear" → ["TODO-5", "TODO-6"]。"""
    out: list[str] = []
    for m in TODO_SPLIT.findall(text):
        out.extend(f"TODO-{p}" for p in m.split("/"))
    return out


def _lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n") if path.exists() else []


def _pending_in(lines: list[str]) -> list[str]:
    """这一份代码里还"没写完"的 TODO。

    占位 raise 不见了 = 写完；被自己写的 return 挤到后面变成死代码 = 也算写完（只是没删干净）；
    被注释掉了 = 也算写完（你亲手关掉的，说明那一条你已经处理过了）。
    """
    pending: list[str] = []
    for i, ln in enumerate(lines):
        m = RAISE_RE.search(ln)
        if not m:
            continue
        if "#" in ln[:m.start()]:
            continue                         # 整行被注释：占位已经作废，别再算进"未写"
        indent = len(ln) - len(ln.lstrip())
        dead = False
        for prev in reversed(lines[:i]):
            if not prev.strip():
                continue
            ind = len(prev) - len(prev.lstrip())
            if ind < indent or re.match(r"(def |class )", prev.strip()):
                break                      # 走出这个函数了
            if ind == indent and re.match(r"(return|raise)\b", prev.strip()) and "NotImplementedError" not in prev:
                dead = True
                break
        if not dead and m.group(1) not in pending:
            pending.append(m.group(1))
    return pending


def pending_for_file(stage_id: str, rel: str, cache: dict) -> set[str]:
    """只看 src/ 里那份 —— 也就是你保存过、命令行真的会跑的代码。

    曾经这里还去看"最近一次在网站上提交的内容"，好让没保存的改动也算进度。
    结果适得其反：编辑器里只要那一行占位不见了（删掉、改错、甚至贴错标签页），
    整条 TODO 立刻变成"已完成"。实现度只该由 src/ 说话；
    "我刚写的到底对不对"由旁边的「单项 X/Y」和整卷结果说话，它们是真跑出来的。
    """
    key = rel
    if key not in cache:
        cache[key] = set(_pending_in(_lines_of(SRC_PKG / rel)))
    return cache[key]


def progress_for(stage: dict) -> dict:
    cache: dict = {}
    ids: list[str] = []
    for t in stage["todos"]:
        for tid in expand_todo(t):
            if tid not in ids:
                ids.append(tid)
    pending: set[str] = set()
    for rel in stage["files"]:
        pending |= pending_for_file(stage["id"], rel, cache)
    # 只认登记表里列出的 TODO：讲义里顺手写的 TODO-1A/2A 也在那张表里，
    # 而 src/ 里偶尔出现的裸 raise NotImplementedError 不该被算进进度。
    undone = [t for t in ids if t in pending]
    done = [t for t in ids if t not in pending]
    st = load_state().get(stage["id"], {})
    checks = st.get("checks", {})
    home = [c["id"] for c in checks_for(stage["id"]) if c["stage"] == stage["id"]]
    suite = st.get("suite") or {}
    # answer 记录（我拿 solutions/ 验证明跑出来的 pass）不参与"单项 X/Y"，
    # 也不在 last 里冒充 pass —— 前端看到 'answer' 就该知道那不是你的代码过的。
    def real(cid: str) -> str | None:
        c = checks.get(cid) or {}
        if not c.get("status"):
            return None
        return "answer" if c.get("answer") else c["status"]
    return dict(
        todo_total=len(ids), todo_done=len(done), todo_pending=undone,
        checks_total=len(home),
        checks_pass=sum(1 for cid in home if checks.get(cid, {}).get("status") == "pass"
                        and not checks.get(cid, {}).get("answer")),
        checks_done=sum(1 for cid in home if checks.get(cid, {}).get("status")),
        suite=suite,
        suite_stale=bool(suite.get("status")) and src_stamp(stage) > suite.get("mt", 0.0),
        last={cid: real(cid) for cid in home},
    )


# ------------------------------------------------------------------ HTTP
INDEX = HERE / "index.html"
RESULT = HERE / "result.html"           # 「↗ 结果窗口」那一页，和测评台同源


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                 # 安静一点，不然输出全是 GET 行
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _serve_manual(self, rel: str) -> None:
        """讲义页面和它的 assets —— 测评台的第二个视图，不是另一个网站。

        放行哪些 URL 由 webapp/manual_assets.py 那张登记表决定：讲义引用它、总览页展示它、
        `python webapp/manual_assets.py` 自检它。这里只补扩展名、查表、读文件，
        不拼目录 —— 所以 /manual/../../webapp/state.json 这类路径根本没有机会被解析，
        /api/manual（读 ROOT 下的讲义 md）也不会长成任意文件读取。
        """
        rel = unquote(rel).replace("\\", "/").strip("/")
        if ".." in Path(rel).parts:
            self._send(404, b"not found", "text/plain")
            return
        if not Path(rel).suffix:
            rel = (rel or "index") + ".html"   # /manual/ 和 /manual/s02 都不带扩展名
        p = MANUAL_URLS.get("/manual/" + rel)
        if p is None or not p.is_file():
            self._send(404, "not found（登记表里没有这个 URL；跑 python webapp/manual_assets.py 看清单）"
                       .encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send(200, p.read_bytes(), MANUAL_TYPES[p.suffix.lower()])

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/manual/"):
            self._serve_manual(path[len("/manual/"):])
        elif path == "/api/stages":
            from lab import KNOBS                    # 旋钮定义只在 lab.py 一处，前端照着长表单
            payload = []
            for s in STAGES:
                d = dict(s)
                d["progress"] = progress_for(s) if s["available"] else {}
                d["lab"] = KNOBS.get(s["id"], [])
                last = d["progress"].get("last", {})
                d["checks"] = [dict(id=c["id"], label=c["label"], needs=c["needs"],
                                    home=c["stage"], here=(c["stage"] == s["id"]),
                                    last=last.get(c["id"]))
                               for c in checks_for(s["id"])]
                payload.append(d)
            prog = [p["progress"] for p in payload if p["progress"]]
            done = sum(g["todo_done"] for g in prog)
            total = sum(g["todo_total"] for g in prog)
            self._json(dict(root=str(ROOT), stages=payload, python=sys.version.split()[0],
                            overall=dict(done=done, total=total),
                            panes=get_panes(),
                            todo_labels={c["id"]: c["label"] for c in CHECKS}))
        elif path == "/result":
            self._send(200, RESULT.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/result":
            q = parse_qs(urlparse(self.path).query)
            try:
                stage = find_stage(q["stage"][0])
                self._json(dict(load_run(stage["id"]), stage=stage["id"], title=stage["title"],
                                judge=stage["judge"]))
            except Exception as exc:                       # noqa: BLE001
                self._json(dict(error=str(exc)), 400)
        elif path == "/api/file":
            q = parse_qs(urlparse(self.path).query)
            try:
                stage = find_stage(q["stage"][0])
                rel = safe_rel(q["path"][0])
                self._json(dict(path=rel, content=read_stage_file(stage, rel)))
            except Exception as exc:                       # noqa: BLE001
                self._json(dict(error=str(exc)), 400)
        elif path == "/api/manual":
            q = parse_qs(urlparse(self.path).query)
            p = ROOT / q["path"][0]
            if ".." in p.parts or not str(p).startswith(str(ROOT)) or not p.exists():
                self._json(dict(error="找不到讲义"), 404)
            else:
                self._json(dict(text=p.read_text(encoding="utf-8")))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._json(dict(error=f"请求体不是合法 JSON：{exc}"), 400)
            return
        try:
            if path == "/api/run":
                stage = find_stage(body["stage"])
                if not stage["available"]:
                    self._json(dict(status="pending", output=f"阶段 {stage['title']} 的骨架还没发布"), 200)
                    return
                files = body.get("files") or {}
                if not files:
                    self._json(dict(status="error", output="没有收到任何文件"), 200)
                    return
                ws = build_workspace(stage["id"], files)
                res = run_judge(ws, stage["judge"])
                res["workspace"] = str(ws)
                res["answer"] = submitted_is_answer(stage, files)   # 那是答案，不是你的代码
                suite = {k: res[k] for k in ("status", "got", "total")}
                if res["answer"]:
                    suite["answer"] = True
                record(stage["id"], suite=suite)
                remember_run(stage["id"], res, kind="suite")     # 结果窗口轮询的就是这份
                self._json(res)
            elif path == "/api/check":
                stage = find_stage(body["stage"])
                if not stage["available"]:
                    self._json(dict(status="pending", line="这个阶段还没发布"))
                    return
                from checks import BY_ID
                check = BY_ID.get(body["id"])
                if check is None:
                    self._json(dict(status="error", line=f"没有编号 {body['id']} 的检查"))
                    return
                ws = build_workspace(stage["id"], body.get("files") or {})
                res = run_check(ws, check)
                res["workspace"] = str(ws)
                res["id"] = check["id"]
                res["label"] = check["label"]
                res["answer"] = submitted_is_answer(stage, body.get("files") or {})
                record(stage["id"], check=(check["id"], res["status"], res["answer"]))
                remember_run(stage["id"], res, kind="check")
                self._json(res)
            elif path == "/api/lab":
                stage = find_stage(body["stage"])
                if not stage["available"]:
                    self._json(dict(status="pending", output="这个阶段还没发布试验台"))
                    return
                res = run_lab(stage, body.get("files") or {}, body.get("params") or {})
                self._json(res)
            elif path == "/api/panes":
                save_panes(body or {})
                self._json(dict(panes=get_panes()))
            elif path == "/api/save":
                stage = find_stage(body["stage"])
                saved = save_to_src(stage, body.get("files") or {})
                self._json(dict(saved=saved, backups=str(BACKUPS)))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:                       # noqa: BLE001
            self._json(dict(status="error", output=f"{type(exc).__name__}: {exc}"), 200)

    def do_OPTIONS(self):                              # 同源页面用不到，但留着方便本地试 API
        self._send(204, b"", "text/plain")


def find_stage(sid: str) -> dict:
    for s in STAGES:
        if s["id"] == sid:
            return s
    raise KeyError(f"未知阶段 {sid!r}")


class GraderServer(ThreadingHTTPServer):
    """ThreadingHTTPServer + 两个刻意的设置。

    allow_reuse_address = False：Windows 默认允许两个进程绑同一个端口，
    结果是"改了代码重启，请求却打到旧实例"——我调试时就踩到过四个旧进程同时在线。
    宁可启动失败说清楚，也不要静默地答错。
    """

    daemon_threads = True
    allow_reuse_address = False


def main() -> int:
    ap = argparse.ArgumentParser(description="本地测评网站（只监听 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    try:
        srv = GraderServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"端口 {args.host}:{args.port} 起不来：{exc}")
        print("多半是已经有一个测评台在跑（同一个端口在 Windows 上会被静默复用，所以这里禁止）。")
        print("要么直接用已经开着的那个，要么换个端口：python webapp/server.py --port 8123")
        return 1
    print(f"MiniGPT 测评台 → http://{args.host}:{args.port}   （Ctrl+C 停）")
    print(f"仓库根目录：{ROOT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
