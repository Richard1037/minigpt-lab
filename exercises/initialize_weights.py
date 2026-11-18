import numpy as np


def initialize(shape, rng=None):
    # rng=None 表示调用方没有传生成器，需在这里创建 default_rng(0)。
    # TODO：生成 shape 对应的标准正态数组，按输入宽度缩放并返回。
    rng= rng if rng is not None else np.random.default_rng(0)
    return rng.standard_normal(shape)*np.sqrt(2/shape[0])
    raise NotImplementedError("请实现基础 Linear 权重初始化")


# 直接运行文件时检查当前实现；被 Linear 等模块导入时跳过测评。
if __name__ == "__main__":
    from _initialization_checks import run_local
    run_local(initialize)
