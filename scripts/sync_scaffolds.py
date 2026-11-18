"""同步新版空骨架；已有实现/注释不匹配旧模板时一律保留。"""
import ast
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def unfinished_only(source):
    try:tree=ast.parse(source)
    except SyntaxError:return False
    functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name not in ('__init__','__call__')]
    if not functions:return False
    for fn in functions:
        body=[n for n in fn.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        if len(body)!=1 or not isinstance(body[0],ast.Raise):return False
        if not any(isinstance(n,ast.Name) and n.id=='NotImplementedError' for n in ast.walk(body[0])):return False
    return True

def sync_scaffolds(book, exercises):
    changed=[];preserved=[]
    for key,meta in book['lessons'].items():
        if not meta.get('sync_empty_scaffold'):continue
        notes=meta.get('scaffold_notes',{});canonical=notes.get('starter')
        if not canonical or '__main__' not in canonical or not meta.get('file'):continue
        path=exercises/meta['file']
        if not path.exists():continue
        current=path.read_text(encoding='utf-8')
        if current==canonical:continue
        legacy=notes.get('source')
        if not legacy or current!=legacy or not unfinished_only(current):
            preserved.append(key);continue
        backups=exercises/'backups';backups.mkdir(exist_ok=True)
        backup=backups/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-skeleton-'+path.name)
        shutil.copy2(path,backup)
        path.write_text(canonical,encoding='utf-8')
        meta['local_code']=canonical
        notes['source']=canonical
        changed.append(key)
    return dict(updated=changed,preserved=preserved)

if __name__=='__main__':
    path=ROOT/'webapp/foundations.json';book=json.loads(path.read_text(encoding='utf-8'))
    report=sync_scaffolds(book,ROOT/'exercises')
    if report['updated']:path.write_text(json.dumps(book,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
