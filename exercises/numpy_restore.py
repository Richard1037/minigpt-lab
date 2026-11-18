import numpy as np

def solve(Y2, out_shape):
    # TODO：按本节来源与形状关系实现。
    raise NotImplementedError("请实现 形状恢复：让每一行回到原 batch/token")

if __name__ == "__main__":
    from _checks import run_local
    run_local("restore", solve)
