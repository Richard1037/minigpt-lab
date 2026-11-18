# scripts/estimate.py —— 训练预算估算器（Python 3.14，仅标准库）
# 用法：把 bench_gemm 实测的有效 GFLOPS 填到 GFLOPS，直接读出"要挂几天"。
def n_params(L, d, V):
    return 12 * L * d * d + V * d          # 4d² attention + 8d² SwiGLU FFN，输出 embedding tied

def days(tokens, N, gflops):
    return tokens * 1e6 * 6 * N / (gflops * 1e9) / 86400

# 主线配置（见 §5）：6 层 / d=576 / V=8192 ≈ 28.6M 参数
MODEL = dict(L=6, d=576, heads=9, ctx=512, V=8192)
N = n_params(MODEL["L"], MODEL["d"], MODEL["V"])
GFLOPS = 150    # ← 换成你 bench_gemm 在真实训练 batch 形状下测到的数字

print(f"params = {N/1e6:.1f}M   chinchilla = {20*N/1e6:.0f}M tokens")
print(f"weights+grads+Adam = {N*16/1e6:.0f} MB   (param+grad+2 moment × 4B)")
for tok_m in (50, 200, 400, 600):
    print(f"{tok_m:>4}M tokens -> {days(tok_m, N, GFLOPS):5.1f} days @ {GFLOPS} GFLOPS")
    # 反向核对：想要 3 天出结果，能跑多少 token？
print(f"3 天能跑 {3*86400*GFLOPS*1e9/(6*N)/1e6:.0f}M tokens")
