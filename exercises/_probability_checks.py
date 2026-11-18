"""概率与损失的缓存、标量上游、极端分数及接口检查。"""
import numpy as np


def behavior(scope,case):
    p=case['input'];cls=p['class'];rng=np.random.default_rng(p['seed'])
    shape=tuple(p['shape']);x=rng.normal(size=shape);xb=x.copy()
    z=x-x.max(-1,keepdims=True);logp=z-np.log(np.exp(z).sum(-1,keepdims=True));prob=np.exp(logp)
    obj=scope[cls]()
    if cls=='Softmax':
        y=obj(x);np.testing.assert_allclose(y,prob);np.testing.assert_array_equal(x,xb)
        y[:]=17;x[:]=-55
        g=rng.normal(size=shape);gb=g.copy();expected=prob*(g-(g*prob).sum(-1,keepdims=True))
        for _ in range(2):
            actual=obj.backward(g);assert isinstance(actual,np.ndarray)
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-9)
        np.testing.assert_array_equal(g,gb)
        # 单行改变分数，其余batch/token的概率和梯度保持原位。
        x2=xb.copy();x2.reshape(-1,shape[-1])[0,0]+=2
        out=obj(x2);np.testing.assert_allclose(out.reshape(-1,shape[-1])[1:],prob.reshape(-1,shape[-1])[1:])
    else:
        targets=rng.integers(shape[-1],size=shape[:-1]);tb=targets.copy();out=obj(x,targets)
        assert isinstance(out,np.ndarray) and out.shape==(),'返回零维NumPy数组'
        expected_loss=-np.take_along_axis(logp,tb[...,None],-1).mean()
        np.testing.assert_allclose(out,expected_loss)
        np.testing.assert_array_equal(x,xb);np.testing.assert_array_equal(targets,tb)
        x[:]=66;targets[:]=(targets+1)%shape[-1]
        expected=prob.copy().reshape(-1,shape[-1]);expected[np.arange(expected.shape[0]),tb.reshape(-1)]-=1;expected/=expected.shape[0];expected=expected.reshape(shape)
        for g in [0.,-2.5,3.]:
            actual=obj.backward(g);np.testing.assert_allclose(actual,g*expected,rtol=1e-6,atol=1e-9)
            actual[:]=19
        np.testing.assert_allclose(obj.backward(),expected)
        # 极端错误类别不能经过log(已经下溢为0的概率)。
        extreme=np.zeros(shape);extreme[...,0]=-2000;target=np.zeros(shape[:-1],dtype=int)
        loss=obj(extreme,target);assert np.isfinite(loss)
        assert float(loss)>1900 if shape[-1]>1 else float(loss)==0
        assert np.all(np.isfinite(obj.backward()))
    assert obj.parameters()==[],'本层无可训练参数'
    obj.zero_grad()
    return {'概率损失接口与缓存':True}
