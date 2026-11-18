import numpy as np

def solve(seed, dims):
    # TODO：按本节来源与形状关系实现。
    raise NotImplementedError("请实现 随机状态：同一生成器连续采两次")

if __name__ == "__main__":
    from _checks import run_local
    run_local("random_sequence", solve)
