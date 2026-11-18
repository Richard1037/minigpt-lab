import numpy as np

def solve(S):
    raise NotImplementedError("TODO：实现本节计算")

# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _checks import run_local
    run_local("att_probs", solve)
