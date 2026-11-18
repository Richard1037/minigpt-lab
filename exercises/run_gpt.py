"""完成个人练习后，用本地语料训练/生成；只导入个人模块。"""
import argparse
import json
from pathlib import Path
import numpy as np
from t_gpt import GPT
from t_loss import CrossEntropy
from t_optim import AdamW, clip_grad
from t_data import tokenize, make_batch
from t_generate import generate
from t_checkpoint import save_checkpoint, load_checkpoint

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text',help='UTF-8 语料文件；恢复时词表必须一致')
    parser.add_argument('--out',default='runs/my-gpt')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--C',type=int,default=16);parser.add_argument('--H',type=int,default=2)
    parser.add_argument('--T',type=int,default=16);parser.add_argument('--B',type=int,default=2)
    parser.add_argument('--hidden',type=int,default=64);parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--lr',type=float,default=.003);parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--independent',action='store_true',help='使用 rebuild_gpt.py 中的个人独立实现');parser.add_argument('--resume',action='store_true');parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--tokens',type=int,default=80);parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--top-k',type=int,default=0)
    args=parser.parse_args()
    text=Path(args.text).read_text(encoding='utf-8');vocab,ids=tokenize(text)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=dict(V=len(vocab),C=args.C,H=args.H,hidden=args.hidden,layers=args.layers,context=args.T,seed=args.seed)
    rng=np.random.default_rng(args.seed)
    if args.resume or args.generate_only:
        meta=json.loads((out/'meta.json').read_text(encoding='utf-8'))
        if meta['vocab']!=vocab or meta['text']!=text:raise ValueError('续训语料或词表与检查点不一致')
        config=meta['config']
    B=meta['B'] if args.resume or args.generate_only else args.B;T=config['context'];split=int(len(ids)*.8);tr,va=ids[:split],ids[split:]
    if min(len(tr),len(va))<=T:raise ValueError('语料过短，需让训练和验证集均长于 T')
    model_source=meta.get('model_source','standard') if args.resume or args.generate_only else ('independent' if args.independent else 'standard')
    if model_source=='independent':
        from rebuild_gpt import GPT as ModelClass
    else:ModelClass=GPT
    model=ModelClass(**config);opt=AdamW(model.parameters(),lr=args.lr);ce=CrossEntropy()
    if args.resume or args.generate_only:load_checkpoint(str(out/'checkpoint.npz'),model,opt,rng)
    losses=[]
    if not args.generate_only:
        for step in range(args.steps):
            starts=rng.integers(0,len(tr)-T,size=B);x,y=make_batch(tr,starts,T)
            model.zero_grad();loss=float(ce(model(x),y));model.backward(ce.backward())
            norm=float(clip_grad(model.parameters(),1.));opt.step();losses.append(loss)
            if step%10==0 or step==args.steps-1:print(f'step={opt.t} loss={loss:.5f} grad_norm={norm:.4f}')
        save_checkpoint(str(out/'checkpoint.npz'),model,opt,rng)
        (out/'meta.json').write_text(json.dumps(dict(config=config,vocab=vocab,text=text,B=B,model_source=model_source),ensure_ascii=False,indent=2),encoding='utf-8')
    vx,vy=make_batch(va,np.arange(B)%(len(va)-T),T)
    val=float(ce(model(vx),vy))
    result=generate(model,vx[:,:min(3,T)],args.tokens,args.temperature,args.top_k,args.seed)
    report=dict(losses=losses,validation_loss=val,generated=[''.join(vocab[i] for i in row) for row in result])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('validation_loss=',val)
    print('\n'.join(report['generated']))

if __name__=='__main__':main()
