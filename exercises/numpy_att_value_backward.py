import numpy as np

def solve(A, G):
    # TODO：按本节来源与形状关系实现。
    raise NotImplementedError("请实现 注意力反向一：所有 query 汇入 dV")

# 具名导出供后续模块复用；仍用 solve 接收本题测评。
value_gradient = solve


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _checks import run_local
    run_local("att_value_backward", solve)
