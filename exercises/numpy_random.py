import numpy as np

def solve(seed, dims):
    # TODO：实现本节计算，返回 NumPy 数组。
    rng=np.random.default_rng(int(seed))
    return rng.standard_normal(tuple(map(int,dims)))
    raise NotImplementedError("请完成本节 TODO")

from _checks import run_local

if __name__ == "__main__":
    run_local("random", solve)
