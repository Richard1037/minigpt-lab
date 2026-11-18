"""第四部分的网页与本地共用测评。输入与预期值均来自公开 cases。"""
import json
import traceback
from pathlib import Path
import numpy as np
from foundation_core import Parameter, Module, initialize, scale_residual_outputs


def pack(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return np.asarray(value).tolist()


def compare(actual, expected):
    if expected is None:
        return actual is None
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(compare(actual[k], v) for k, v in expected.items())
    if not isinstance(actual, np.ndarray):
        return False
    target = np.asarray(expected)
    return actual.shape == target.shape and bool(np.allclose(actual, target, rtol=1e-6, atol=2e-6))


def instance(scope, case):
    obj = scope[case['class']](*case.get('init', []))
    for name, value in case.get('parameters', {}).items():
        part = obj
        path = name.split('.')
        for token in path[:-1]:
            part = getattr(part, token)
        target = getattr(part, path[-1])
        target.data[...] = np.array(value)
    return obj


def evaluate_layer(scope, cases):
    rows = []
    for case in cases:
        if case.get('assessment'):
            from _assessment import evaluate_special
            rows.append(evaluate_special(scope,case));continue
        try:
            obj = instance(scope, case)
            x = np.array(case['X'], dtype=np.int64 if case['class'] == 'Embedding' else np.float64)
            g = np.array(case['G'], dtype=np.float64)
            before = x.copy()
            weights = [p.data.copy() for p in obj.parameters()]
            y = obj.forward(x)
            dx = obj.backward(g)
            if case.get('repeat'):
                obj.forward(x)
                dx = obj.backward(g)
            actual = {'Y': y, 'dX': dx}
            for name in case['expected']:
                if name.endswith('.grad'):
                    target = obj
                    for token in name.split('.'):
                        target = getattr(target, token)
                    actual[name] = target
            unchanged = np.array_equal(x, before) and all(np.array_equal(p.data, w) for p, w in zip(obj.parameters(), weights))
            passed = compare(actual, case['expected']) and unchanged
            from _assessment import diagnose
            diagnostics=diagnose(actual,case['expected'])
            if not unchanged:diagnostics.append('输入或权重被原地修改')
            snapshot = pack(actual)
            if case.get('check_zero'):
                obj.zero_grad()
                zero = all(np.all(p.grad == 0) for p in obj.parameters())
                passed = passed and zero
                snapshot['清零后参数梯度全零'] = zero
                if not zero:diagnostics.append('zero_grad 没有清空所有参数梯度')
            rows.append(dict(label=case['label'], passed=bool(passed), actual=snapshot, unchanged=bool(unchanged),diagnostics=diagnostics))
        except Exception:
            rows.append(dict(label=case['label'], passed=False, errorText=traceback.format_exc(limit=4)))
    return rows


def evaluate_infrastructure(scope, cases):
    rows=[]
    for case in cases:
        if case.get('assessment'):
            from _assessment import evaluate_special
            rows.append(evaluate_special(scope,case));continue
        try:
            P, M = scope['Parameter'], scope['Module']
            p = P([[1., 2.]])
            assert isinstance(p.data, np.ndarray) and p.grad.shape == p.data.shape
            assert p.data.dtype == np.float64 and np.all(p.grad == 0)
            p.grad[...] = 7
            p.zero_grad()
            assert np.all(p.grad == 0) and np.array_equal(p.data, [[1.,2.]])
            child=M();child.weight=p
            root=M();root.child=child;root.shared=p;root.blocks=[child]
            assert len(root.parameters()) == 1 and root.parameters()[0] is p
            p.grad[...] = 5;root.zero_grad();assert np.all(p.grad==0)
            rows.append(dict(label=case['label'],passed=True,actual={'参数形状':[1,2], '共享参数计数':1, '清零后梯度':[[0.,0.]]}))
        except Exception:
            rows.append(dict(label=case['label'],passed=False,errorText=traceback.format_exc(limit=4)))
    return rows


def run_local(lesson, scope):
    catalog = json.loads((Path(__file__).parent.parent / 'webapp' / 'foundations.json').read_text(encoding='utf-8'))
    meta = catalog['lessons'][lesson]
    rows = (evaluate_infrastructure if meta['kind']=='infrastructure' else evaluate_layer)(scope, meta['cases'])
    for case, row in zip(meta['cases'], rows):
        print(('PASS' if row['passed'] else 'FAIL') + ' | ' + row['label'])
        print('输入:', json.dumps(case.get('X',case.get('input')),ensure_ascii=False))
        print('预期:', json.dumps(case['expected'],ensure_ascii=False))
        print('实际:', row.get('actual',row.get('errorText')))
    total=sum(r['passed'] for r in rows)
    print(f'{total}/{len(rows)} 项通过')
    raise SystemExit(0 if total==len(rows) else 1)
