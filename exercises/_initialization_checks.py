"""初始化任务的网页/本地共用测评；检查数值、类型和 RNG 状态。"""
import json
import traceback
from pathlib import Path
import numpy as np


def evaluate_initialize(initialize, cases):
    rows=[]
    for case in cases:
        try:
            config=case['input']; rng=np.random.default_rng(config['seed'])
            oracle=np.random.default_rng(config['seed']); actual={}; expected={}; diagnostics=[]
            shapes=config.get('shapes',[config.get('shape')])
            for i,shape in enumerate(shapes):
                name='W' if i==0 else f'W{i+1}'
                shape=tuple(shape)
                if config.get('default_rng'):
                    out=initialize(shape) if i==0 else initialize(shape,None)
                    ref=np.random.default_rng(0).normal(0,np.sqrt(2/shape[0]),shape)
                else:
                    out=initialize(shape,rng)
                    ref=oracle.normal(0,np.sqrt(2/shape[0]),shape)
                expected[name]=ref.tolist()
                if not isinstance(out,np.ndarray):
                    diagnostics.append(f'{name} 必须返回 NumPy 数组')
                arr=np.asarray(out);actual[name]=arr.tolist()
                if arr.dtype!=np.float64:diagnostics.append(f'{name} dtype 应为 float64，实际为 {arr.dtype}')
                if arr.shape!=shape:diagnostics.append(f'{name} 形状应为 {shape}，实际为 {arr.shape}')
                elif not np.allclose(arr,ref,rtol=0,atol=1e-12):diagnostics.append(f'{name} 数值不符：核对输入宽度、标准差和所使用的 RNG')
            if not config.get('default_rng') and rng.bit_generator.state!=oracle.bit_generator.state:
                diagnostics.append('传入 RNG 的状态不符：应从该生成器按顺序采样，不能重置、复制生成器或额外抽样')
            rows.append(dict(label=case['label'],passed=not diagnostics,actual=actual,diagnostics=diagnostics))
        except Exception:
            rows.append(dict(label=case['label'],passed=False,errorText=traceback.format_exc(limit=4)))
    return rows


def run_local(initialize):
    book=json.loads((Path(__file__).resolve().parents[1]/'webapp/foundations.json').read_text(encoding='utf-8'))
    cases=book['lessons']['initialization']['cases'];rows=evaluate_initialize(initialize,cases)
    for case,row in zip(cases,rows):
        print(('PASS' if row['passed'] else 'FAIL')+' | '+row['label'])
        print('输入:',json.dumps(case['input'],ensure_ascii=False))
        print('预期:',json.dumps(case['expected'],ensure_ascii=False))
        print('实际:',row.get('actual',row.get('errorText')))
        if row.get('diagnostics'):print('原因:','；'.join(row['diagnostics']))
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} 项通过")
    raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
