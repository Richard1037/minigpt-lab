import numpy as np
from foundation_core import Parameter, Module, initialize

def solve(X, W, b):
    # 最后一轴做线性映射，再加偏置
    input_shape=X.shape
    C,D=X.shape[-1],W.shape[-1] # 输入，输出特征维度
    Y=X.reshape(-1,C)@W+b
    return Y.reshape(*input_shape[:-1],D)
    raise NotImplementedError("请实现 Linear 前向")

from _checks import run_local
if __name__ == "__main__":
    run_local("linear", solve)
