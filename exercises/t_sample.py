import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from t_softmax import Softmax

def sample(logits, temperature, top_k, rng):
    if temperature <= 0: raise ValueError('temperature 必须大于 0')
    z = logits.copy()/temperature
    if top_k:
        indices = np.argsort(z,axis=-1,kind='stable')[...,-top_k:]
        kept = np.full_like(z,-np.inf)
        np.put_along_axis(kept,indices,np.take_along_axis(z,indices,-1),-1)
        z = kept
    p = Softmax()(z)
    return np.array([rng.choice(p.shape[-1],p=row) for row in p],dtype=np.int64)

if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_sample", globals())
