"""Embedding 构造检查；网页与本地测评共用，不提供实现答案。"""
import numpy as np


def check_initialization(scope, case):
    from foundation_core import Parameter, Module
    P, M = scope.get('Parameter', Parameter), scope.get('Module', Module)
    config = case['input']
    def require(ok, message):
        if not ok:
            raise AssertionError(message)
    def same(actual, expected, message):
        require(isinstance(actual, np.ndarray) and actual.shape == expected.shape
                and actual.dtype == np.float64
                and np.allclose(actual, expected, rtol=0, atol=1e-12), message)

    global_before = np.random.get_state()
    shapes = []
    try:
        for V, C in config['configs']:
            obj = scope['Embedding'](V, C)
            require(isinstance(obj, M), 'Embedding 应继承当前 Module')
            weight = obj.weight
            require(isinstance(weight, P), 'weight 应使用当前 Parameter 包装')
            expected = np.random.default_rng(config['seed']).normal(0, config['std'], (V,C))
            same(weight.data, expected, f'构造 ({V},{C}) 的词表形状、float64 或初值不符合约定')
            same(weight.grad, np.zeros((V,C)), '构造后 grad 应为同形 float64 零数组')
            shapes.append(list(weight.data.shape))
            require(not np.shares_memory(weight.data, weight.grad), 'data 和 grad 应各自持有数值')
            parameters = obj.parameters()
            require(isinstance(parameters, list) and len(parameters) == 1 and parameters[0] is weight,
                    '词表必须被 parameters() 收集且只出现一次')
            other = scope['Embedding'](V,C)
            require(other.weight is not weight
                    and not np.shares_memory(other.weight.data, weight.data)
                    and not np.shares_memory(other.weight.grad, weight.grad), '两个实例不能共享词表或梯度存储')
            # 模拟优化器改过参数、当前轮已有梯度，再检查前向是否真正复用。
            weight.data[...] += .125
            weight.grad[...] = .7
            data = weight.data.copy()
            ids = (np.arange(6).reshape(2,3) % V).astype(np.int64)
            for _ in range(2):
                same(obj.forward(ids), data[ids], 'forward 应查当前词表，保留全部 batch/token')
                require(obj.weight is weight, 'forward 不应重新创建 Parameter')
                same(weight.data, data, 'forward 应保留当前参数数值')
                same(weight.grad, np.full((V,C), .7), 'forward 应保留已经累计的梯度')
        after = np.random.get_state()
        require(global_before[0] == after[0] and np.array_equal(global_before[1], after[1])
                and global_before[2:] == after[2:], '使用局部 default_rng，保留全局随机状态')
    finally:
        np.random.set_state(global_before)
    return {'参数形状':shapes, '初值符合给定种子':True, '初始梯度为零':True,
            'Parameter已注册':True, '实例相互独立':True,
            '前向复用当前参数':True, '全局随机状态保持':True}
