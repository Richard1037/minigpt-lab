"""网站与本地练习共用的输入、预期结果和判断规则。"""
import traceback
import numpy as np

from pathlib import Path
import json
CASES = json.loads((Path(__file__).with_name('cases.json')).read_text(encoding='utf-8'))

def evaluate(solve, cases):
    rows = []
    for case in cases:
        try:
            args = [np.array(a, dtype=case.get('dtypes', ['float64'] * len(case['args']))[i]) for i, a in enumerate(case['args'])]
            before = [a.copy() for a in args]
            actual = solve(*args)
            valid = isinstance(actual, np.ndarray)
            actual = np.asarray(actual)
            expected = np.array(case['expected'], dtype=np.float64)
            shape = actual.shape == expected.shape
            dtype = not case.get('expected_dtype') or actual.dtype == np.dtype(case['expected_dtype'])
            same = all(np.array_equal(a, b) for a, b in zip(args, before))
            error = float(np.max(np.abs(actual - expected))) if shape else None
            passed = valid and shape and dtype and same and np.allclose(actual, expected, rtol=0, atol=1e-10)
            from _assessment import diagnose
            diagnostics=diagnose(actual,case['expected'],atol=1e-10,rtol=0) if valid else ['返回值必须是 NumPy 数组']
            if not same:diagnostics.append('输入数组被原地修改')
            if not dtype:diagnostics.append(f"返回 dtype 应为 {case['expected_dtype']}，实际 {actual.dtype}")
            rows.append(dict(diagnostics=diagnostics,label=case['label'], passed=bool(passed), actual=actual.tolist(), shape=list(actual.shape), expectedShape=list(expected.shape), error=error, unchanged=same, array=valid))
        except Exception:
            rows.append(dict(label=case['label'],passed=False,errorText=traceback.format_exc(limit=3)))
    return rows

def run_local(lesson, solve):
    rows = evaluate(solve, CASES[lesson])
    for case, row in zip(CASES[lesson], rows):
        print(('PASS' if row['passed'] else 'FAIL') + ' | ' + row['label'])
        print('输入:', case['args'])
        print('预期:', case['expected'])
        print('实际:', row.get('actual', row.get('errorText')))
    passed = sum(r['passed'] for r in rows)
    print(f'本节结果：{passed}/{len(rows)} 项通过')
    raise SystemExit(0 if passed == len(rows) else 1)
