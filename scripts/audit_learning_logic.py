"""读取课程和骨架，记录可核对事实；不构建课程、不执行或改写个人代码。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    source = ROOT / 'webapp/foundations.json'
    book = json.loads(source.read_text(encoding='utf-8'))
    parts, tasks = [], []
    for number, stage in enumerate(book['stages'], 1):
        part = 'optional' if stage.get('optional') else f'part-{number:02d}'
        ids = [i for module in stage['modules'] for i in module['tasks']]
        parts.append(dict(id=part, title=stage['title'], paused=bool(stage.get('paused')), tasks=ids))
        for module in stage['modules']:
            for name in module['tasks']:
                meta = book['lessons'][name]
                def record(nodes, output, owner=''):
                    for node in nodes:
                        if isinstance(node, ast.ClassDef):
                            record(node.body, output, node.name)
                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            todo = any(isinstance(x, ast.Name) and x.id == 'NotImplementedError' for x in ast.walk(node))
                            output.append(dict(name=(owner+'.' if owner else '')+node.name, state='todo' if todo else 'provided'))
                methods,reference_methods=[],[]
                record(ast.parse(meta.get('starter', '')).body,methods)
                record(ast.parse(meta.get('reference', '')).body,reference_methods)
                tasks.append(dict(id=name, part=part, module=module['name'], title=meta['title'], kind=meta['kind'],
                    dependencies=meta.get('dependencies', []), next_use=meta.get('task_brief', {}).get('next_use'),
                    exports=meta.get('exports', []), file=meta.get('file'),
                    methods=methods, reference_methods=reference_methods, cases=len(meta.get('cases', [])), questions=len(meta.get('questions', [])),
                    note='provided 表示骨架已给出代码；拓展修错题的 provided 可能故意有错，不能按缺少实现误判。'))
    ordered = [t['id'] for t in tasks]
    assert len(ordered) == len(set(ordered)) and set(ordered) == set(book['lessons'])
    positions = {name:i for i,name in enumerate(ordered)}
    inversions = [(t['id'], d) for t in tasks for d in t['dependencies'] if positions[d] >= positions[t['id']]]
    return dict(source='webapp/foundations.json', sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        parts=parts, tasks=tasks, explicit_dependency_inversions=inversions,
        scope='静态事实快照；显式依赖有序不证明教学前置已讲清，provided 不自动等于缺陷。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = snapshot()
    text = json.dumps(data, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
        print(f"Recorded {len(data['tasks'])} tasks; explicit dependency inversions: {len(data['explicit_dependency_inversions'])}")
    else:
        print(text)
