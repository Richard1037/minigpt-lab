# 使用本人的公共接口与初始化函数。
import numpy as np
from learner_core import Parameter, Module, initialize

class Linear(Module):
    def __init__(self, C, D, use_bias=True):
        # TODO：调用 initialize 创建权重并包装为 Parameter；创建可选偏置。
        raise NotImplementedError("请实现 Linear 构造")

    def forward(self, x):
        # TODO：保存原形状和二维输入副本，计算输出并恢复前置维度。
        raise NotImplementedError("请实现 Linear 前向")

    def backward(self, g):
        # TODO：累加参数梯度，返回恢复为原输入形状的输入梯度。
        raise NotImplementedError("请实现 Linear 反向")

# 直接运行执行本节测评；被其他模块导入时跳过。

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("rebuild_linear", globals())
