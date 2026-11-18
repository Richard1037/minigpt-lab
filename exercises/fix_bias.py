import numpy as np
from foundation_core import Parameter, Module, initialize

def solve(G):
    # 此实现能运行，但梯度轴不正确。
    return G.sum(axis=-1)

if __name__ == "__main__":
    from _checks import run_local
    run_local("fix_bias", solve)
