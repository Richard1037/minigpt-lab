import numpy as np
from learner_core import Parameter, Module, initialize

def tokenize(text):
    vocab=sorted(set(text))
    stoi={c:i for i,c in enumerate(vocab)}
    return vocab,np.array([stoi[c] for c in text],dtype=np.int64)
    raise NotImplementedError("请建立词表并编码")

def make_batch(ids, starts, T):
    return np.stack([ids[s:s+T] for s in starts]),np.stack([ids[s+1:s+1+T] for s in starts])
    raise NotImplementedError("请取窗口与右移目标")


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_data", globals())
