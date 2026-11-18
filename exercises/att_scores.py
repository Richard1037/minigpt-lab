import numpy as np

def solve(Q, K):
    raise NotImplementedError("TODO：实现本节计算")

# 具名导出供后续模块复用；仍用 solve 接收本题测评。
attention_scores = solve


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _checks import run_local
    run_local("att_scores", solve)
