"""X 每行一条样本；b 每个值对应一列。返回相加后的新数组。
在项目目录执行：python exercises/02_broadcast.py
只替换 solve 中的 TODO；下面的输入创建、调用和验收已写好。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _checks import CASES, run_local


def solve(X, b):
    # TODO：在此实现本节计算，并返回 NumPy 数组。
    raise NotImplementedError("请填写 solve 的 TODO")


if __name__ == "__main__":
    # 读取本节示例，将普通列表转换为 NumPy 数组。
    example = CASES["broadcast"][0]
    inputs = [np.array(value, dtype=np.float64) for value in example["args"]]
    print("示例输入：", inputs)
    print("solve 接收这些数组，返回计算结果。")
    # 测评入口会逐项调用 solve(*inputs)，打印预期与实际结果。
    run_local("broadcast", solve)
