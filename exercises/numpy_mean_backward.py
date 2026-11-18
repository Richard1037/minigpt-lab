import numpy as np

def solve(G, C):
    # TODO：按本节来源与形状关系实现。
    raise NotImplementedError("请实现 均值反向：把上游分回每个特征")

if __name__ == "__main__":
    from _checks import run_local
    run_local("mean_backward", solve)
