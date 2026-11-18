import numpy as np

def solve(A, dA):
    # TODO：按本节来源与形状关系实现。
    raise NotImplementedError("请实现 注意力反向二：同一行概率的耦合")

# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _checks import run_local
    run_local("att_score_backward", solve)
