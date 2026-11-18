"""阶段 6 · 真数据训练 + 检查点：把模型接到一份真语料上，并让它活得比这次运行更久。

阶段 5 的机器已经在一段自造的小模式上跑通了。这一阶段换两件东西：

* **数据换成真的**：`data/corpus.txt` —— 64 行、4061 个字符的英文短文（本手册自己写的，
  刻意只用 27 个不同字符、句式反复，这样一台 0.1M 参数的小模型在 CPU 上几秒钟就能看出趋势）。
  它按**行**切成 train/val，val 那 5 行模型从没吃过，所以"验证 loss"是真的验证。
* **权重要能存下来**：MGPT 二进制格式（`minigpt/checkpoint.py`）。这是本阶段的全部新内容 ——
  训练本身只是把阶段 5 的循环拉长。

三件事必须一起成立，判据才好看：
① 真语料上学得动（loss 掉到 ln V 的一半以下）；
② 存下来的文件**按格式**是对的（C++ 侧能用参考解析器读走，不需要你的 load）；
③ **断了能续**：训 K 步存档、读回来接着训 N-K 步，和一口气训 N 步对得上。
第 ③ 条是这一章唯一说得清"存档真的有用"的检验 —— 只测"存了能读"是测不出丢了优化器状态的。

跑法：
    python stages/s06_train.py
    MGPT_ANSWER=1 python stages/s06_train.py
想训久一点再看曲线（不会改变判据）：`MGPT_JSON=1` 之外再加 `MGPT_TRAIN_STEPS=3000`。
"""

from __future__ import annotations

import io
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pathsetup  # noqa: E402  选 src/ 还是 solutions/，见 pathsetup.py

import numpy as np

from minigpt.checkpoint import (MAGIC, STATE_MAGIC, VERSION, load_checkpoint, load_train_state,   # # noqa: E402
                                save_checkpoint, save_train_state, checkpoint_summary)
from minigpt.config import Config
from minigpt.dataset import Batcher, evaluate, token_stream, train_val_texts
from minigpt.gpt import GPT
from minigpt.optim import AdamW, clip_grad_norm, lr_at
from minigpt.tokenizer import CharTokenizer
from minigpt.toolkit import Report, print_progress, sparkline

CORPUS = str(pathsetup.ROOT / "data" / "corpus.txt")
CTX = int(os.environ.get("MGPT_CTX", 40))
B = int(os.environ.get("MGPT_BATCH", 4))
D_MODEL, N_LAYER, N_HEAD = 64, 2, 4
STEPS = int(os.environ.get("MGPT_TRAIN_STEPS", 300))
LR, WARMUP = 4e-3, 40
MAX_GRAD = 1.0
SEED = 0

# 续训对账用的小配置：这一条要跑两遍同样的步数，别让整卷慢到没人愿意跑
R_CTX, R_STEPS, R_FIRST = 32, 150, 100
R_CFG = Config(vocab=27, d_model=48, n_layer=2, n_head=4, ctx=R_CTX)


def read_corpus() -> str:
    with open(CORPUS, encoding="utf-8") as f:
        return f.read()


def train_loop(model, opt, batcher, steps, rng, lr_peak, warmup, log_every, prefix,
               curve=None, horizon=None):
    """阶段 5 那个循环的加长版，多了"每步记录 grad_norm"。

    `horizon` 是学习率调度看到的**总步数**。续训时它必须还是原来那个总数 ——
    写成"还剩多少步"会把 cosine 重新拉长一遍，等于偷偷改了训练计划。
    下面 B 组第一段就用 horizon=R_STEPS，尽管它只跑 R_FIRST 步。
    """
    curve = curve if curve is not None else []
    total = steps if horizon is None else horizon
    gnorms = []
    for i in range(steps):
        opt.zero_grad()
        x, _y = batcher(B, rng)
        curve.append(model.loss_backward(x))
        gnorms.append(clip_grad_norm(model, MAX_GRAD))
        opt.step(lr_at(i, total, lr_peak, warmup=warmup))
        if log_every:
            print_progress(i + 1, total, every=log_every, prefix=prefix)
    return curve, gnorms


# ------------------------------------------------------------------ 参考解析器
def parse_ckpt(buf: bytes):
    """照着骨架文件头那张字节表读一遍 —— 这就是"C++ 侧的读法"。

    判据故意不用你写的 `load_checkpoint`，否则"自己存、自己读"会把格式错误一起包庇掉。
    """
    o = 0
    magic = buf[o:o + 4]
    o += 4
    ver, n = struct.unpack("<I", buf[o:o + 4])[0], struct.unpack("<I", buf[o + 4:o + 8])[0]
    o += 8
    head = __import__("json").loads(buf[o:o + n].decode("utf-8"))
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


# ------------------------------------------------------------------ 1) 语料
def check_corpus(rep: Report, tok: CharTokenizer, text: str):
    rep.group("真语料 · data/corpus.txt 与 train/val 切分")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    rep.check("语料读到了", len(lines) >= 40 and len(text) >= 2000,
              f"{len(text):,} 个字符 · {len(lines)} 行（`data/corpus.txt`，本手册自撰，"
              f"只有 {tok.vocab_size} 个不同字符）")
    ids = tok.encode_array(text)
    rt = tok.decode(tok.encode(text)) == text
    rep.check("分词在这份语料上往返一致", rt and tok.vocab_size == len(set(text)),
              f"vocab={tok.vocab_size} == 去重字符 {len(set(text))} · 最长 id {int(ids.max())}")

    tr, va = train_val_texts(text, 0.1)
    tr_lines = {ln for ln in tr.split("\n") if ln.strip()}
    va_lines = [ln for ln in va.split("\n") if ln.strip()]
    leak = [ln for ln in va_lines if ln in tr_lines]
    rep.check("train/val 零重叠（整行级）", len(leak) == 0 and len(va_lines) >= 3,
              f"train {len(tr_lines)} 行 / val {len(va_lines)} 行 · 重叠 {len(leak)} 行 · "
              f"val 第一行：{va_lines[0][:46]!r}")
    rep.info("val 的 token 数", f"{token_stream(va, tok).size}（> ctx+1={CTX + 1} 才切得出窗口）")
    return tr, va


# ------------------------------------------------------------------ 2) 训练
def check_training(rep: Report, cfg: Config, tok, tr_bat, va_bat, text: str):
    rep.group("训练 · 真语料上的 loss 与泛化")
    rng = np.random.default_rng(SEED)
    ln_v = math.log(cfg.vocab)

    probe = GPT(cfg, rng=np.random.default_rng(42))
    x0, _ = tr_bat(B, np.random.default_rng(1))
    init = float(probe.loss_backward(x0))
    probe.zero_grad()
    rep.close("初始 loss ≈ ln V", init, ln_v, 0.2,
              note=f"ln({cfg.vocab})={ln_v:.3f}；真语料的 ln V 就是「瞎猜一个字符」的成本")

    print(f"\n第 1 步：{STEPS} 步真语料训练（d={cfg.d_model} L={cfg.n_layer} ctx={cfg.ctx} B={B}）")
    model = GPT(cfg, rng=np.random.default_rng(1))
    opt = AdamW(model, lr=LR, weight_decay=0.1)
    curve, gnorms = train_loop(model, opt, tr_bat, STEPS, rng, LR, WARMUP, 50, "真语料训练")
    print()
    print(sparkline(curve, label=f"loss(真语料 {STEPS} 步)"))
    rep.curve("真语料训练 loss", curve, note=f"{STEPS} 步 · lr {LR} · wd 0.1 · clip {MAX_GRAD}")
    final = float(np.mean(curve[-20:]))
    val = evaluate(model, va_bat, B, 10, np.random.default_rng(9))
    rep.above("训练 loss 下降倍数", curve[0] / max(final, 1e-9), 1.5,
              note=f"{curve[0]:.3f} → {final:.3f}（真实英文比阶段 5 的模式语料难学得多）")
    rep.below("训练 loss 掉到 ln V 的四分之三以下", final, 0.75 * ln_v,
              note=f"末段均值 {final:.3f} vs ln V {ln_v:.3f}")
    rep.below("验证 loss（5 行没吃过的）", val, 0.75 * ln_v,
              note=f"val {val:.3f}；和 train 一起看才有意义，单看一个数会骗人")
    rep.soft("没有明显过拟合", val - final < 0.5,
             f"train {final:.3f} · val {val:.3f}（差 {val - final:+.3f}）· 4K 字符的语料，"
             "0.1M 的模型背得下来 —— 这正是「验证集」存在的理由")

    ids_all = tok.encode_array(text)
    hits = tot = 0
    for s in range(0, ids_all.size - cfg.ctx - 1, 5):
        lg = model.forward(ids_all[s:s + cfg.ctx][None])
        hits += int(int(np.argmax(lg[0, -1])) == int(ids_all[s + cfg.ctx]))
        tot += 1
    acc = hits / max(tot, 1)
    rep.above("贪心下一字符准确率", acc, 0.30,
              note=f"{hits}/{tot} 个位置 · 随机猜约 {1.0 / cfg.vocab:.3f}，"
                   f"字符频率最高的约 {np.max(np.bincount(ids_all) / ids_all.size):.3f}")
    rep.info("grad_norm 中位数/最大值", f"{float(np.median(gnorms)):.3f} / {max(gnorms):.3f}"
                                        f"（超过 {MAX_GRAD} 的那些步被裁过）")
    return model


# ------------------------------------------------------------------ 3) 检查点
def check_checkpoint(rep: Report, cfg: Config, model, tmp: str, tr_ids):
    rep.group("检查点 · MGPT 格式、加载校验、断点续训")
    path = os.path.join(tmp, "town.ckpt")
    size = save_checkpoint(path, model, cfg)
    raw = open(path, "rb").read()
    magic, ver, head, tensors, consumed = parse_ckpt(raw)
    named = model.named_parameters()
    names = [n for n, _ in named]

    fmt_ok = magic == MAGIC and ver == VERSION and head.get("config") == cfg.to_dict()
    rep.check("格式合规（参考解析器读得通）", bool(fmt_ok) and consumed == len(raw),
              f"魔数 {magic!r} · 版本 {ver} · 头 JSON 的 config 与 Config.to_dict 相同 · "
              f"读到 EOF 正好用完 {len(raw):,} 字节")
    fp32_ok = all(v.dtype == np.float32 for v in tensors.values()) and \
        len(raw) < 8 * sum(p.data.size for _, p in named)
    rep.check("磁盘上是 float32，不是 float64", bool(fp32_ok),
              f"{len(raw):,} 字节 vs float64 的 {8 * sum(p.data.size for _, p in named):,} 字节 · "
              f"参数量 {sum(p.data.size for _, p in named):,}")
    rep.check("张量名与数量 == named_parameters（tied 只存一份）",
              sorted(tensors) == sorted(names) and len(tensors) == len(set(names)),
              f"{len(tensors)} 个张量 · 名字形如 {sorted(names)[:2]} · 输出头没有第二份")
    rep.info("checkpoint_summary", checkpoint_summary(path))

    fresh = GPT(cfg, rng=np.random.default_rng(777))
    keep_data = fresh.tok.weight.data        # 记住对象身份，等下查"原地写"
    info = load_checkpoint(path, model=fresh)
    logits_ref = model.forward(np.arange(cfg.ctx)[None] % cfg.vocab)
    logits_new = fresh.forward(np.arange(cfg.ctx)[None] % cfg.vocab)
    diff = float(np.max(np.abs(logits_ref - logits_new)))
    rep.check("读回来能算出一样的 logits", bool(info.get("applied") == len(names)) and diff < 2e-4,
              f"{len(names)} 个参数灌回新机 · 同一份输入的 logits 最大差异 {diff:.2e}"
              "（float32 量化就在这个量级）")
    inplace_ok = fresh.tok.weight.data is keep_data
    rep.check("加载是原地写（没换掉 Parameter 的数据对象）", bool(inplace_ok),
              "tied 的输出头和优化器拿着的还是原来那块内存；换了对象就等于什么都没加载")

    bad_cfg = Config(**{**cfg.to_dict(), "vocab": cfg.vocab + 3})
    try:
        load_checkpoint(path, model=GPT(bad_cfg, rng=np.random.default_rng(3)))
        cfg_guard = False
        why = "没报错"
    except ValueError as exc:
        cfg_guard = "vocab" in str(exc)
        why = str(exc)[:60]
    rep.check("配置不一致时拒绝加载", bool(cfg_guard), f"抛 ValueError 并点名差异字段：{why}")

    with open(path, "rb") as f:
        broken = b"XXXX" + f.read()[4:]
    bp = os.path.join(tmp, "broken.ckpt")
    with open(bp, "wb") as f:
        f.write(broken)
    try:
        load_checkpoint(bp)
        magic_guard = False
    except ValueError:
        magic_guard = True
    rep.check("魔数不对时拒绝加载", bool(magic_guard), "读到 4 个字节就知道这不是 MGPT 存档")

    warped = GPT(cfg, rng=np.random.default_rng(8))
    warped.tok.weight.data = np.zeros((3, 3))          # 故意把形状改掉（只有测试代码可以这么干）
    try:
        load_checkpoint(path, model=warped)
        shape_guard = False
    except ValueError:
        shape_guard = True
    rep.check("形状对不上时拒绝加载", bool(shape_guard), "名字相同但 (3,3) ≠ (vocab, d_model)")

    # ---- 断点续训对账：这是这一章最好的判据
    print("\n第 2 步：断点续训对账。同一份语料、同一个种子、同一串随机窗口：")
    print("  A 一口气训到底；B 训到中途存档（权重 + 优化器矩 + step_count）再读回来接着训；")
    print("  C 只把 weight_decay 换掉接着训（状态存档里没有它）—— 三条曲线，两条该重合。")
    rb = Batcher(tr_ids, R_CTX)

    def run_a():
        m = GPT(R_CFG, rng=np.random.default_rng(11))
        o = AdamW(m, lr=6e-3, weight_decay=0.0)
        rng = np.random.default_rng(12)
        c, _g = train_loop(m, o, rb, R_STEPS, rng, 6e-3, 20, 0, "A 直训")
        return c

    def run_b(wd=0.0, tag="B", suffix=""):
        """前 R_FIRST 步照 A 的方式训，然后存档、读回一台**新建的**模型接着训。

        新建时用另一个 rng 种子、另一个 lr —— 这两样都必须被存档覆盖掉，
        否则"续训"就是假的。`wd` 是故意留的旋钮：状态存档里**没有** weight_decay，
        传错就会分叉（下面那条软判据演示的就是这件事）。
        """
        m = GPT(R_CFG, rng=np.random.default_rng(11))
        o = AdamW(m, lr=6e-3, weight_decay=0.0)
        rng = np.random.default_rng(12)
        c, _g = train_loop(m, o, rb, R_FIRST, rng, 6e-3, 20, 0, tag + " 前段",
                           horizon=R_STEPS)      # 调度看的是总步数，不是剩余步数
        wp = os.path.join(tmp, "resume" + suffix + ".ckpt")
        sp = os.path.join(tmp, "resume" + suffix + ".state")
        save_checkpoint(wp, m, R_CFG)
        save_train_state(sp, m, o, R_FIRST, c, R_CFG)
        m2 = GPT(R_CFG, rng=np.random.default_rng(999))        # 故意用另一个种子建
        o2 = AdamW(m2, lr=1e-3, weight_decay=wd)              # lr 每步由 lr_at 给，这里故意写错
        load_checkpoint(wp, model=m2)                          # 权重读回来
        st = load_train_state(sp, model=m2, opt=o2)            # 矩 + step_count 读回来
        c2 = list(c)
        for i in range(R_FIRST, R_STEPS):
            o2.zero_grad()
            x, _y = rb(B, rng)
            c2.append(m2.loss_backward(x))
            clip_grad_norm(m2, MAX_GRAD)
            o2.step(lr_at(i, R_STEPS, 6e-3, warmup=20))
        return c2, st

    ca = run_a()
    cb, st = run_b()
    cc, _st_c = run_b(wd=0.5, tag="C", suffix="-c")
    a_end, b_end = float(np.mean(ca[-10:])), float(np.mean(cb[-10:]))
    c_end = float(np.mean(cc[-10:]))
    gap = abs(a_end - b_end)
    seg = slice(R_FIRST, R_STEPS)
    stepwise = float(np.max(np.abs(np.asarray(ca)[seg] - np.asarray(cb)[seg])))
    stepwise_c = float(np.max(np.abs(np.asarray(ca)[seg] - np.asarray(cc)[seg])))
    rep.curve("A：一口气训完", ca, note=f"{R_STEPS} 步")
    rep.curve(f"B：训 {R_FIRST} 步存档后继续", cb, note="两条曲线该几乎重合")
    rep.curve(f"C：续训时换错超参（wd=0.5）", cc, note="这条该分叉")
    rep.below("续训与直训在恢复段逐步吻合", stepwise, 1e-4,
              f"恢复段（第 {R_FIRST}~{R_STEPS} 步）逐点最大差 {stepwise:.2e}，末段均值差 {gap:.2e}"
              f"（A {a_end:.5f} · B {b_end:.5f}）。少存一样——权重、一二阶矩、step_count——"
              "这个数就会跳到 1e-2 量级")
    rep.check("训练状态的头里带 step / lr / curve",
              int(st.get("step", -1)) == R_FIRST and abs(float(st.get("lr", 0)) - 6e-3) < 1e-9
              and len(st.get("curve") or []) == R_FIRST,
              f"step={st.get('step')} · lr={st.get('lr')} · curve 长度 {len(st.get('curve') or [])}")
    sp = os.path.join(tmp, "resume.state")
    st_bytes = open(sp, "rb").read()[:4] == STATE_MAGIC
    rep.check("状态存档用的是 MGTR 魔数（和权重区分开）", bool(st_bytes),
              f"{os.path.getsize(sp):,} 字节的 .state 文件，头 4 个字节 = {open(sp, 'rb').read()[:4]!r}")
    rep.soft("状态存档不恢复 weight_decay：换错超参会分叉",
             stepwise_c > 100 * max(stepwise, 1e-15),
             f"同样的权重和矩、只把 wd 从 0.0 换成 0.5 接着训 → 恢复段逐点最大差 {stepwise_c:.2e}"
             f"（正确续训是 {stepwise:.2e}）。step / 矩 / 曲线在文件里，超参在你的启动脚本里 —— "
             "知道这条边界才不会拿旧状态配新脚本。")
    return path, size


def main() -> int:
    rep = Report("阶段 6 · 真数据训练 + 检查点",
                 f"data/corpus.txt · B={B} ctx={CTX} d={D_MODEL} L={N_LAYER} · {STEPS} 步")
    text = read_corpus()
    tok = CharTokenizer(text)
    cfg = Config(vocab=tok.vocab_size, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, ctx=CTX)
    tr, va = train_val_texts(text, 0.1)
    tr_bat = Batcher(token_stream(tr, tok), CTX)
    va_bat = Batcher(token_stream(va, tok), CTX)
    rep.info("配置", cfg.describe())

    check_corpus(rep, tok, text)
    model = check_training(rep, cfg, tok, tr_bat, va_bat, text)
    tmp = tempfile.mkdtemp(prefix="mgpt-s06-")
    check_checkpoint(rep, cfg, model, tmp, token_stream(tr, tok))

    prompt = tok.encode_array("the small town sits between two hills")
    win = np.zeros((1, CTX), dtype=np.int64)
    win[0, -prompt.size:] = prompt
    gen = [int(np.argmax(model.forward(win)[0, -1]))]
    for _ in range(59):
        win = np.concatenate([win[:, 1:], np.array([[gen[-1]]], dtype=np.int64)], axis=1)
        gen.append(int(np.argmax(model.forward(win)[0, -1])))
    rep.info("贪心续写 60 个字符", repr("the small town sits between two hills" + tok.decode(gen)))

    passed = rep.done()
    print("\n这一步你确认了三件事：")
    print("  1) 同一台机器在真语料上仍然学得动，而且 val（没吃过的 5 行）和 train 一起看才知道有没有背下来。")
    print("  2) 存下来的文件是**按公开格式**排布的：一个不认识你的代码的解析器能把它读干净。")
    print("  3) 断点续训和一口气训完对得上 —— 这说明权重、优化器矩、step_count 三样一样都没漏。")
    print(f"\n临时存档写在 {tmp}（判据用完就可以删，它不属于仓库）。")
    print("\n下一步（阶段 7）：同一个模型，用温度/top-k/top-p 决定它开口说什么。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(pathsetup.run(main))
