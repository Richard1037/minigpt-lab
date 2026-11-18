"""第五部分起的完整 NumPy 对照实现；不导入个人练习。"""
import numpy as np
from foundation_layers import Linear, Embedding, MLP, RMSNorm
from foundation_core import Module, Parameter

class Softmax(Module):
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self._p = e / e.sum(axis=-1, keepdims=True)
        return self._p.copy()

    def backward(self, g):
        return self._p * (g - np.sum(g*self._p, axis=-1, keepdims=True))

class CrossEntropy(Module):
    def forward(self, logits, targets):
        z = logits - logits.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        self._p = np.exp(logp)
        self._targets = targets.copy()
        return np.array(-np.take_along_axis(logp, targets[...,None], axis=-1).mean())

    def backward(self, g=1.):
        dx = self._p.copy().reshape(-1, self._p.shape[-1])
        dx[np.arange(dx.shape[0]), self._targets.reshape(-1)] -= 1
        return (dx / dx.shape[0] * g).reshape(self._p.shape)

class AttentionCore(Module):
    def __init__(self, causal=True):
        self.causal = causal
        self.softmax = Softmax()

    def forward(self, q, k, v):
        self._q, self._k, self._v = q.copy(), k.copy(), v.copy()
        self._scale = np.sqrt(q.shape[-1])
        scores = q @ k.swapaxes(-1,-2) / self._scale
        if self.causal:
            if q.shape[-2] != k.shape[-2]:
                raise ValueError('因果自注意力要求 Q/K 序列长度相同')
            self._mask = np.triu(np.ones(scores.shape[-2:], dtype=bool), 1)
            scores = np.where(self._mask, -np.inf, scores)
        self._a = self.softmax(scores)
        return self._a @ v

    def backward(self, g):
        dv = self._a.swapaxes(-1,-2) @ g
        da = g @ self._v.swapaxes(-1,-2)
        ds = self.softmax.backward(da)
        if self.causal:
            ds = np.where(self._mask, 0., ds)
        return ds @ self._k / self._scale, ds.swapaxes(-1,-2) @ self._q / self._scale, dv

class MultiHeadAttention(Module):
    def __init__(self, C, H):
        if C % H: raise ValueError('C 必须能被 H 整除')
        self.C, self.H = C, H
        self.qkv = Linear(C,3*C)
        self.core = AttentionCore(True)
        self.proj = Linear(C,C)

    def forward(self, x):
        B,T,C = x.shape
        self._shape = x.shape
        q,k,v = np.split(self.qkv(x),3,axis=-1)
        split = lambda a: a.reshape(B,T,self.H,C//self.H).transpose(0,2,1,3)
        y = self.core(split(q),split(k),split(v))
        return self.proj(y.transpose(0,2,1,3).reshape(B,T,C))

    def backward(self, g):
        B,T,C = self._shape
        dy = self.proj.backward(g).reshape(B,T,self.H,C//self.H).transpose(0,2,1,3)
        grads = self.core.backward(dy)
        joined = [a.transpose(0,2,1,3).reshape(B,T,C) for a in grads]
        return self.qkv.backward(np.concatenate(joined,axis=-1))

class Block(Module):
    def __init__(self, C, H, hidden):
        self.norm1, self.norm2 = RMSNorm(C), RMSNorm(C)
        self.attn, self.mlp = MultiHeadAttention(C,H), MLP(C,hidden)

    def forward(self, x):
        x1 = x + self.attn(self.norm1(x))
        return x1 + self.mlp(self.norm2(x1))

    def backward(self, g):
        g1 = g + self.norm2.backward(self.mlp.backward(g))
        return g1 + self.norm1.backward(self.attn.backward(g1))

class GPT(Module):
    def __init__(self, V, C, H, hidden, layers, context, seed=0):
        self.context = context
        self.token, self.position = Embedding(V,C), Embedding(context,C)
        self.blocks = [Block(C,H,hidden) for _ in range(layers)]
        self.norm = RMSNorm(C)
        rng = np.random.default_rng(seed)
        # 初始化每个独立参数一次，不能让同形状矩阵都使用同一个随机序列。
        for p in self.parameters():
            if p.data.ndim == 2: p.data[...] = rng.normal(0,.02,p.data.shape)
        for block in self.blocks:
            block.attn.proj.W.data /= np.sqrt(2*layers)
            block.mlp.fc2.W.data /= np.sqrt(2*layers)

    def forward(self, ids):
        if ids.shape[1] > self.context: raise ValueError('序列超过 context')
        x = self.token(ids) + self.position(np.arange(ids.shape[1]))[None,:,:]
        for block in self.blocks: x = block(x)
        self._h = self.norm(x)
        return self._h @ self.token.weight.data.T

    def backward(self, g):
        C = self._h.shape[-1]
        self.token.weight.grad += g.reshape(-1,g.shape[-1]).T @ self._h.reshape(-1,C)
        dx = self.norm.backward(g @ self.token.weight.data)
        for block in reversed(self.blocks): dx = block.backward(dx)
        self.position.backward(dx.sum(axis=0))
        self.token.backward(dx)
        return None

class Dropout(Module):
    def __init__(self, p=.2, seed=0):
        if not 0 <= p < 1: raise ValueError('p 应在 [0,1)')
        self.p, self.rng = p, np.random.default_rng(seed)

    def forward(self, x, training=True):
        self._mask = (self.rng.random(x.shape) >= self.p)/(1-self.p) if training else np.ones_like(x)
        return x*self._mask

    def backward(self, g):
        return g*self._mask

class LayerNorm(Module):
    def __init__(self, C, eps=1e-5):
        self.w, self.b = Parameter(np.ones(C)), Parameter(np.zeros(C))
        self.eps = eps

    def forward(self, x):
        centered = x-x.mean(axis=-1,keepdims=True)
        self._r = 1/np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)+self.eps)
        self._h = centered*self._r
        return self._h*self.w.data+self.b.data

    def backward(self, g):
        self.w.grad += (g*self._h).reshape(-1,g.shape[-1]).sum(axis=0)
        self.b.grad += g.reshape(-1,g.shape[-1]).sum(axis=0)
        u = g*self.w.data
        return self._r*(u-u.mean(axis=-1,keepdims=True)-self._h*np.mean(u*self._h,axis=-1,keepdims=True))

class AdamW:
    def __init__(self, parameters, lr=.001, beta1=.9, beta2=.999, eps=1e-8, decay=.01):
        self.params = list(parameters)
        self.lr,self.beta1,self.beta2,self.eps,self.decay = lr,beta1,beta2,eps,decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for p,m,v in zip(self.params,self.m,self.v):
            m *= self.beta1
            m += (1-self.beta1)*p.grad
            v *= self.beta2
            v += (1-self.beta2)*p.grad**2
            mh, vh = m/(1-self.beta1**self.t), v/(1-self.beta2**self.t)
            # 本课程约定仅二维权重衰减，bias 和 norm scale 不衰减。
            if p.data.ndim >= 2: p.data *= 1-self.lr*self.decay
            p.data -= self.lr*mh/(np.sqrt(vh)+self.eps)

def clip_grad(parameters, limit):
    norm = float(np.sqrt(sum(np.sum(p.grad**2) for p in parameters)))
    scale = min(1.,limit/(norm+1e-12))
    for p in parameters: p.grad *= scale
    return np.array(norm)

def tokenize(text):
    vocab = sorted(set(text))
    stoi = {c:i for i,c in enumerate(vocab)}
    return vocab, np.array([stoi[c] for c in text],dtype=np.int64)

def make_batch(ids, starts, T):
    x = np.stack([ids[s:s+T] for s in starts])
    y = np.stack([ids[s+1:s+T+1] for s in starts])
    return x,y

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

def generate(model, ids, steps, temperature=1., top_k=0, seed=0):
    result = ids.copy()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        logits = model(result[:,-model.context:])[:,-1,:]
        token = sample(logits,temperature,top_k,rng)
        result = np.concatenate([result,token[:,None]],axis=1)
    return result

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

def train(model, x, y, steps, lr):
    """固定小批次过拟合实验；每步清零、前向、损失、反向、裁剪、更新。"""
    loss_fn = CrossEntropy()
    opt = AdamW(model.parameters(),lr=lr,decay=.01)
    losses = []
    for _ in range(steps):
        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits,y)
        model.backward(loss_fn.backward())
        clip_grad(model.parameters(),1.)
        opt.step()
        losses.append(float(loss))
    return np.array(losses)

def project_step(model, optimizer, ids, targets):
    """最终项目的一次训练更新：返回更新前的平均损失（零维数组）。"""
    model.zero_grad()
    loss_fn = CrossEntropy()
    loss = loss_fn(model(ids), targets)
    model.backward(loss_fn.backward())
    clip_grad(model.parameters(), 1.)
    optimizer.step()
    return loss
