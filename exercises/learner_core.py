"""后续个人模块使用本人的 Parameter/Module 与初始化函数。"""
from foundation_core import scale_residual_outputs
from initialize_weights import initialize
from layer_infrastructure import Parameter, Module

__all__ = ['Parameter', 'Module', 'initialize', 'scale_residual_outputs']
