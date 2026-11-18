"""网页与本地共用的RMSNorm独立构造验收。"""
import numpy as np
from foundation_core import Parameter, Module

def check_initialization(scope, case):
    P=scope.get('Parameter',Parameter);M=scope.get('Module',Module)
    for C,eps in case['input']['configs']:
        obj=scope['RMSNorm'](C,eps);other=scope['RMSNorm'](C,eps)
        assert isinstance(obj,M) and isinstance(obj.w,P),'w必须是当前Parameter'
        assert obj.w.data.shape==(C,) and obj.w.data.dtype==np.float64,'w形状或dtype错误'
        np.testing.assert_array_equal(obj.w.data,np.ones(C))
        np.testing.assert_array_equal(obj.w.grad,np.zeros(C))
        assert obj.eps==eps,'保存传入eps'
        assert obj.parameters()==[obj.w],'参数应被注册一次'
        assert not np.shares_memory(obj.w.data,other.w.data) and not np.shares_memory(obj.w.grad,other.w.grad),'实例存储应独立'
        obj.w.data[:]=np.linspace(.5,2,C);obj.w.grad[:]=.7
        x=np.arange(1,1+6*C,dtype=float).reshape(2,3,C)/100
        expected=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*obj.w.data
        for _ in range(2):
            np.testing.assert_allclose(obj.forward(x),expected,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(obj.w.grad,.7)
    return {'构造与复用':True}
