"""个人 miniGPT 实验：固定划分、诊断曲线及源码指纹，可复现记录。"""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()

def _generate_with_trace(generate, model, prefix, tokens, temperature, top_k, seed, vocab):
    """Observe the probabilities passed to the learner's sampler without drawing again."""
    namespace = getattr(generate, '__globals__', {})
    sample = namespace.get('sample')
    distributions = []
    vocabulary_size = len(vocab)

    def observe_sample(logits, temperature, top_k, rng):
        rows = []

        class RecordingRng:
            def __getattr__(self, name):
                return getattr(rng, name)

            def choice(self, a, *args, **kwargs):
                picked = rng.choice(a, *args, **kwargs)
                probabilities = kwargs.get('p', args[2] if len(args) > 2 else None)
                if not isinstance(picked, (int, np.integer)) or probabilities is None:
                    return picked
                if isinstance(a, (int, np.integer)) and a == vocabulary_size:
                    rows.append((int(picked), np.asarray(probabilities).copy()))
                elif not isinstance(a, (int, np.integer)):
                    ids = np.asarray(a)
                    if ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and np.all((ids >= 0) & (ids < vocabulary_size)):
                        expanded = np.zeros(vocabulary_size)
                        np.add.at(expanded, ids, probabilities)
                        rows.append((int(picked), expanded))
                return picked

        picked = sample(logits, temperature, top_k, RecordingRng())
        distributions.append(rows)
        return picked

    if callable(sample):
        namespace['sample'] = observe_sample
    try:
        generated = generate(model, prefix, tokens, temperature, top_k, seed)
    finally:
        if callable(sample):
            namespace['sample'] = sample

    result = [''.join(vocab[index] for index in row) for row in generated]
    trace = [dict(prefix=''.join(vocab[index] for index in row), steps=[]) for row in prefix]
    appended = generated[:, prefix.shape[1]:]
    valid = len(distributions) == appended.shape[1] and all(
        len(rows) == len(trace) and all(picked == appended[b, step] for b, (picked, _) in enumerate(rows))
        for step, rows in enumerate(distributions)
    )
    if valid:
        for rows in distributions:
            for run, (picked, probabilities) in zip(trace, rows):
                ranked = np.argsort(-probabilities, kind='stable')
                leading = [int(i) for i in ranked[:8] if probabilities[i] > 0]
                if picked not in leading:
                    leading.append(picked)
                candidates = [dict(char=vocab[i], probability=float(probabilities[i])) for i in leading]
                remaining = float(probabilities.sum() - probabilities[leading].sum())
                run['steps'].append(dict(chosen=vocab[picked], candidates=candidates, other=max(0.0, remaining)))
    return result, trace

def experiment(scope, config, sources=None):
    def number(key,lo,hi,default,integer=True):
        v=float(config.get(key,default))
        if not np.isfinite(v) or not lo<=v<=hi or integer and not v.is_integer():raise ValueError(f'{key} 必须在 {lo}…{hi} 之间'+('且为整数' if integer else ''))
        return int(v) if integer else v
    C=number('C',4,32,8);H=number('H',1,8,2)
    if C%H:raise ValueError('C 必须能被 H 整除')
    T=number('T',2,24,8);B=number('B',2,8,2);layers=number('layers',1,3,1);hidden=number('hidden',4,128,32)
    steps=number('steps',1,300,60);seed=number('seed',0,99999,7);lr=number('lr',.00001,.1,.01,False)
    temperature=number('temperature',.05,3,1.,False);tokens=number('tokens',1,60,24)
    text=str(config.get('text',''));validation=str(config.get('validation_text',''))
    if not 40<=len(text)<=10000:raise ValueError('训练语料需为 40…10000 个字符')
    if validation and not 20<=len(validation)<=4000:raise ValueError('独立验证文本需为 20…4000 个字符')
    vocab,ids=scope['tokenize'](text);V=len(vocab);k=number('top_k',0,V,0)
    stoi={c:i for i,c in enumerate(vocab)}
    if validation:
        unknown=sorted(set(validation)-set(stoi))
        if unknown:raise ValueError('验证文本含训练词表外字符：'+''.join(unknown[:20])+'。请修改语料；实验不会用验证文本扩充训练词表。')
        tr,va=ids,np.array([stoi[c] for c in validation],dtype=np.int64);split=None;split_mode='单独提供验证文本';vocab_policy='词表仅由训练语料构建。'
    else:
        split=int(len(ids)*.8);tr,va=ids[:split],ids[split:];split_mode='按字符位置前80%/后20%';vocab_policy='词表由完整语料构建，包含后20%验证区出现的字符。'
    if min(len(tr),len(va))<=T:raise ValueError('训练或验证集过短，请增大语料或减小 T')
    def encode_prompt(value, name):
        unknown=sorted(set(value)-set(stoi))
        if unknown:
            raise ValueError(name+'含词表外字符：'+repr(''.join(unknown[:20]))+'。请使用语料中出现过的字符。')
        return np.array([[stoi[c] for c in value]],dtype=np.int64)
    resolved=dict(C=C,H=H,T=T,B=B,layers=layers,hidden=hidden,steps=steps,seed=seed,lr=lr,temperature=temperature,tokens=tokens,top_k=k,text=text,validation_text=validation)
    uses_prompt_list = 'prompts' in config
    if uses_prompt_list:
        prompts = config['prompts']
        if not isinstance(prompts, list) or not all(isinstance(value, str) for value in prompts):
            raise ValueError('续写开头需为文本列表。')
        if len(prompts) > 3:
            raise ValueError('一次实验最多添加 3 条续写开头。')
        entered = [(index, value) for index, value in enumerate(prompts, 1) if value]
        prefixes = [encode_prompt(value, f'第 {index} 条续写开头') for index, value in entered]
        resolved['prompts'] = [value for _, value in entered]
    else:
        # Keep the format used by saved experiments and callers without a prompt list.
        prefixes = []
        for key, name in [('prompt', '续写提示词'), ('comparison_prompt', '对照提示词')]:
            if value := str(config.get(key, '')):
                prefixes.append(encode_prompt(value, name))
                resolved[key] = value
    model=scope['GPT'](V,C,H,hidden,layers,T,seed);params=model.parameters();opt=scope['AdamW'](params,lr=lr)
    ce=scope['CrossEntropy']();rng=np.random.default_rng(seed)
    def eval_batch(data):
        starts=np.linspace(0,len(data)-T-1,num=min(16,len(data)-T),dtype=int)
        return scope['make_batch'](data,starts,T)
    tx,ty=eval_batch(tr);vx,vy=eval_batch(va)
    def evaluate():return float(ce(model(tx),ty)),float(ce(model(vx),vy))
    train0,val0=evaluate();points=[dict(step=0,train=train0,validation=val0)]
    activations={};hooks=[]
    def monitor(module,name):
        original=module.forward
        def wrapped(*args,**kwargs):
            value=original(*args,**kwargs);arr=np.asarray(value)
            activations[name]=float(np.sqrt(np.mean(arr*arr)))
            return value
        module.forward=wrapped;hooks.append((module,original))
    for i,block in enumerate(model.blocks):monitor(block,f'Block {i}')
    monitor(model.norm,'最终 RMSNorm')
    losses=[];norms=[];updates=[];activation_series=[];start=time.perf_counter()
    try:
        for step in range(steps):
            starts=rng.integers(0,len(tr)-T,size=B);x,y=scope['make_batch'](tr,starts,T)
            if step==0:
                batch_preview=[dict(input=''.join(vocab[i] for i in row),target=''.join(vocab[i] for i in target))
                               for row,target in zip(x[:2],y[:2])]
            model.zero_grad();loss=float(ce(model(x),y));stage_rms=dict(activations)
            model.backward(ce.backward());norm=float(scope['clip_grad'](params,1.))
            if not np.isfinite(loss+norm):raise ValueError('损失或梯度非有限；检查实现或降低学习率')
            before=[p.data.copy() for p in params];base_norm=np.sqrt(sum(np.sum(w*w) for w in before))
            opt.step();delta=np.sqrt(sum(np.sum((p.data-w)**2) for p,w in zip(params,before)))
            losses.append(loss);norms.append(norm);updates.append(float(delta/(base_norm+1e-12)));activation_series.append(stage_rms)
            if (step+1)%max(1,steps//20)==0 or step==steps-1:
                train_loss,val_loss=evaluate();points.append(dict(step=step+1,train=train_loss,validation=val_loss))
    finally:
        for module,original in hooks:module.forward=original
    elapsed=time.perf_counter()-start
    if uses_prompt_list and not prefixes:
        prefixes.append(vx[:1, :min(3, T)])
    if not uses_prompt_list and 'prompt' not in resolved:
        prefixes.insert(0, vx[:B, :min(3, T)])
    generated, trace = [], []
    for prefix in prefixes:
        result, steps_trace = _generate_with_trace(scope['generate'], model, prefix, tokens, temperature, k, seed, vocab)
        generated.extend(result)
        trace.extend(steps_trace)
    comparison = generated.pop() if 'comparison_prompt' in resolved else None
    provenance=dict(source_sha256=fingerprint(sources or {}),data_sha256=fingerprint(dict(text=text,validation=validation)),
        configuration_sha256=fingerprint(resolved),lab_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=platform.python_version(),numpy=np.__version__,sources_supplied=bool(sources))
    return dict(losses=losses,norms=norms,relative_updates=updates,activation_rms=activation_series,evaluation=points,
        validation_before=val0,validation_after=points[-1]['validation'],parameters=sum(p.data.size for p in params),vocab=V,
        train_chars=len(tr),validation_chars=len(va),split=dict(mode=split_mode,boundary=split,evaluation_train_windows=len(tx),evaluation_validation_windows=len(vx),policy='训练/验证各自取至多16个均匀分布窗口；训练只从训练区内随机采样，不跨划分边界。',vocab_policy=vocab_policy),
        seconds=elapsed,tokens_per_second=steps*B*T/max(elapsed,1e-9),generated=generated,comparison=comparison,
        generation_trace=trace,batch_preview=batch_preview,pipeline_source=Path(__file__).read_text(encoding='utf-8'),config=resolved,provenance=provenance)
