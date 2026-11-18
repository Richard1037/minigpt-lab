import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_embedding import Embedding
    from layer_rmsnorm import RMSNorm
    from layer_linear import Linear
    from layer_gelu import GELU
    from layer_mlp import MLP
    from t_softmax import Softmax
    from t_attention import AttentionCore
    from t_mha import MultiHeadAttention
    from t_block import Block
    from t_gpt import GPT
    from t_loss import CrossEntropy
    from t_optim import clip_grad, AdamW

def save_checkpoint(path, model, optimizer, rng):
    import json
    arrays = {f'p{i}':p.data for i,p in enumerate(model.parameters())}
    arrays.update({f'm{i}':m for i,m in enumerate(optimizer.m)})
    arrays.update({f'v{i}':v for i,v in enumerate(optimizer.v)})
    arrays['meta'] = np.array(json.dumps(dict(t=optimizer.t,rng=rng.bit_generator.state,
        lr=optimizer.lr,beta1=optimizer.beta1,beta2=optimizer.beta2,eps=optimizer.eps,decay=optimizer.decay)))
    np.savez(path,**arrays)

def load_checkpoint(path, model, optimizer, rng):
    import json
    with np.load(path,allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        for i,p in enumerate(model.parameters()):
            if p.data.shape != data[f'p{i}'].shape: raise ValueError('模型形状不一致')
            p.data[...] = data[f'p{i}']
        for i in range(len(optimizer.m)):
            optimizer.m[i][...] = data[f'm{i}']
            optimizer.v[i][...] = data[f'v{i}']
        optimizer.t = meta['t']
        for key in ['lr','beta1','beta2','eps','decay']: setattr(optimizer,key,meta[key])
        rng.bit_generator.state = meta['rng']
    model.zero_grad()

if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_checkpoint", globals())
