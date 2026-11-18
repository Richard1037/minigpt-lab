"""第一、二、四部分本地学习网站：python webapp/pilot_server.py。"""
import json
import subprocess
import sys
import tempfile
import shutil
import math
import mimetypes
import threading
from urllib.parse import unquote, urlsplit
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
EXERCISES = HERE.parent / "exercises"
FILE_LOCK = threading.RLock()


def json_safe(value):
    """错误实现返回 NaN/Inf 时也必须给出可被浏览器读取的失败报告。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value

def catalog():
    data = json.loads((HERE / "foundations.json").read_text(encoding="utf-8"))
    data['auto_save_after_pass'] = True
    data['visual_roadmap'] = json.loads((HERE / "visual-roadmap.json").read_text(encoding="utf-8"))
    return data

RUNNER = r'''import json,sys,traceback,ast
p=json.load(sys.stdin)
sys.path.insert(0,p['exercises'])
try:
 import numpy as np
 from foundation_core import Parameter,Module,initialize,scale_residual_outputs
 from _foundation_checks import evaluate_layer,evaluate_infrastructure
 from _checks import evaluate
 from _initialization_checks import evaluate_initialize
 from _transformer_checks import evaluate_project
 scope=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize,scale_residual_outputs=scale_residual_outputs,__name__='student')
 def execute(source,filename):
  tree=ast.parse(source,filename=filename)
  class KeepLearnerTypes(ast.NodeTransformer):
   def visit_ImportFrom(self,node):
    if node.module in ('foundation_core','learner_core'):
     node.names=[a for a in node.names if a.name not in ('Parameter','Module','initialize')]
     node.module='foundation_core'
     if not node.names:return None
    return node
  tree=ast.fix_missing_locations(KeepLearnerTypes().visit(tree))
  exec(compile(tree,filename,'exec'),scope)
 for name,source in p['dependencies']:
  execute(source,'<前置练习 '+name+'>')
 if p['kind']=='initialization':scope.pop('initialize',None)
 execute(p['code'],'<你的提交>')
 kind=p['kind']
 if p.get('experiment'):
  from _transformer_lab import experiment
  print(json.dumps(experiment(scope,p['experiment'],dict(p['dependencies'],__entry__=p['code'])),ensure_ascii=False));sys.exit(0)
 if kind=='array': rows=evaluate(scope['solve'],p['cases'])
 elif kind=='infrastructure': rows=evaluate_infrastructure(scope,p['cases'])
 elif kind=='project': rows=evaluate_project(scope,p['cases'])
 elif kind=='initialization': rows=evaluate_initialize(scope['initialize'],p['cases'])
 else: rows=evaluate_layer(scope,p['cases'])
 print(json.dumps({'rows':rows},ensure_ascii=False))
except Exception:
 print(json.dumps({'error':traceback.format_exc(limit=5)},ensure_ascii=False))
'''


LOCAL_RUNNER = '''import json,sys,runpy,traceback
p=json.load(sys.stdin)
sys.path.insert(0,p['exercises'])
try:
 from _foundation_checks import evaluate_layer,evaluate_infrastructure
 from _checks import evaluate
 from _initialization_checks import evaluate_initialize
 from _transformer_checks import evaluate_project
 scope=runpy.run_path(p['file'],run_name='local_verification')
''' + RUNNER[RUNNER.index(" kind=p['kind']"):]


def verify_saved(meta):
    payload=dict(exercises=str(EXERCISES),file=str(EXERCISES/meta['file']),kind=meta['kind'],cases=meta['cases'])
    # 使用独立缓存目录，避免同秒内同长度改动命中旧pyc。
    with tempfile.TemporaryDirectory(prefix='saved-verify-') as cache:
        proc=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+cache,'-c',LOCAL_RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=20)
    try:result=json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError):raise ValueError('本地验证未返回结果：'+proc.stderr[-1200:])
    if result.get('error') or len(result.get('rows',[]))!=len(meta['cases']) or not all(r.get('passed') is True for r in result['rows']):
        raise ValueError('本地运行未通过，已恢复原文件：'+str(result.get('error') or [r.get('label') for r in result.get('rows',[]) if not r.get('passed')]))


def save_after_pass(book,meta,body,deps,result,verify=None):
    """仅保存刚通过的提交快照；备份全部变更，并验证真实本地导入。"""
    if not body.get('auto_save') or meta['kind']=='concept' or result.get('error'):return
    rows=result.get('rows',[])
    if len(rows)!=len(meta['cases']) or not rows or not all(r.get('passed') is True for r in rows):return
    sources=dict(deps)
    if not meta.get('integration_only'):sources[meta['id']]=body['code']
    prepared=[];backups=[]
    try:
        with FILE_LOCK:
            for key,source in sources.items():
                item=book['lessons'][key];target=(EXERCISES/item['file']).resolve()
                if not target.is_relative_to(EXERCISES.resolve()):raise ValueError('保存路径不在练习目录')
                text=standalone(item,source);compile(text,str(target),'exec')
                if target.read_text(encoding='utf-8')!=text:prepared.append((key,target,text))
            folder=EXERCISES/'backups';folder.mkdir(exist_ok=True)
            stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            for key,target,text in prepared:
                backup=folder/(stamp+'-passed-'+target.name);shutil.copy2(target,backup);backups.append((target,backup))
            try:
                for key,target,text in prepared:target.write_text(text,encoding='utf-8')
                (verify or verify_saved)(meta)
            except Exception:
                for target,backup in backups:shutil.copy2(backup,target)
                raise
            result['auto_save']=dict(ok=True,files=[book['lessons'][key]['file'] for key in sources],updated=[dict(lesson=key,code=text) for key,_,text in prepared],local_verified=True)
    except Exception as exc:result['auto_save']=dict(ok=False,error=str(exc))


def standalone(meta, code):
    if meta['kind']=='initialization':
        if '__main__' in code:return code
        return 'import numpy as np\n\n'+code+'\n\nif __name__ == "__main__":\n    from _initialization_checks import run_local\n    run_local(initialize)\n'
    if 'infrastructure' in meta.get('dependencies',[]):
        code=code.replace('from foundation_core import','from learner_core import')
    if '__main__' in code:
        return code
    support='learner_core' if 'infrastructure' in meta.get('dependencies',[]) else 'foundation_core'
    imports=f'import numpy as np\nfrom {support} import Parameter, Module, initialize\n'
    if meta.get('dependencies'):
        imports+='if __name__ != "student":\n'
        book=catalog()
        for dep in meta['dependencies']:
            item=book['lessons'][dep]
            exports=item.get('exports',[item.get('class','')])
            imports+=f'    from {Path(item["file"]).stem} import {", ".join(exports)}\n'
    if meta['kind']=='array':
        tail=f'from _checks import run_local\nif __name__ == "__main__":\n    run_local("{meta["id"]}", solve)\n'
    else:
        checker='_transformer_checks' if meta['kind']=='project' else '_foundation_checks'
        tail=f'if __name__ == "__main__":\n    from {checker} import run_local\n    run_local("{meta["id"]}", globals())\n'
    return imports+'\n'+code+'\n\n'+tail


class Handler(BaseHTTPRequestHandler):
    def send(self,data,status=200,kind='application/json; charset=utf-8'):
        raw=data if isinstance(data,bytes) else data.encode('utf-8');self.send_response(status)
        self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/minigpt-lab.svg':
            return self.send((HERE / 'minigpt-lab.svg').read_bytes(), kind='image/svg+xml')
        if self.path in ('/source-map.js', '/source-map.css', '/module-views.js'):
            filename=self.path.lstrip('/')
            kind='text/javascript' if filename.endswith('.js') else 'text/css'
            return self.send((HERE/filename).read_text(encoding='utf-8'),kind=kind+'; charset=utf-8')
        if self.path in ('/python-editor.js','/vendor/highlight/python-highlight.js'):
            return self.send((HERE/self.path.lstrip('/')).read_text(encoding='utf-8'),kind='text/javascript; charset=utf-8')
        if self.path=='/process-animations.js':
            return self.send((HERE/'process-animations.js').read_text(encoding='utf-8'),kind='text/javascript; charset=utf-8')
        if self.path == '/course-katex.css':
            css=(HERE/'vendor/katex/katex.min.css').read_text(encoding='utf-8')
            return self.send(css.replace('url(fonts/','url(/vendor/katex/fonts/'),kind='text/css; charset=utf-8')
        if self.path.startswith('/vendor/katex/'):
            target=(HERE/unquote(urlsplit(self.path).path).lstrip('/')).resolve()
            base=(HERE/'vendor/katex').resolve()
            if not target.is_relative_to(base) or not target.is_file() or target.suffix not in {'.js','.css','.woff2','.woff','.ttf'}:
                return self.send('{}',404)
            return self.send(target.read_bytes(),kind=mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        if self.path=='/api/catalog':
            data=catalog()
            for meta in data['lessons'].values():
                if 'file' in meta:
                    meta['local_code']=(EXERCISES/meta['file']).read_text(encoding='utf-8')
            return self.send(json.dumps(data,ensure_ascii=False))
        files={'/course-navigation.js':('course-navigation.js','text/javascript'),'/math-renderer.js':('math-renderer.js','text/javascript'),'/visual-roadmap.js':('visual-roadmap.js','text/javascript'),'/index-animation.js':('index-animation.js','text/javascript'),'/laboratory.js':('laboratory.js','text/javascript'),'/learning-tools.js':('learning-tools.js','text/javascript'),'/understanding.js':('understanding.js','text/javascript'),'/':('pilot.html','text/html'),'/index.html':('pilot.html','text/html'),'/app.js':('app.js','text/javascript'),'/workbench.css':('workbench.css','text/css')}
        if self.path in files:
            filename,kind=files[self.path]
            return self.send((HERE/filename).read_text(encoding='utf-8'),kind=kind+'; charset=utf-8')
        self.send('{}',404)

    def do_POST(self):
        if self.path not in ('/api/check','/api/save','/api/experiment'):
            return self.send('{}',404)
        try:
            if self.headers.get('Origin') not in (None,'http://127.0.0.1:8124','http://localhost:8124'):
                return self.send('{}',403)
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=250000:return self.send('{}',413)
            body=json.loads(self.rfile.read(size))
            book=catalog();meta=book['lessons'][body['lesson']]
            if self.path=='/api/save':
                if meta['kind']=='concept':raise ValueError('本节没有代码文件')
                target=EXERCISES/meta['file']
                backup=EXERCISES/'backups';backup.mkdir(exist_ok=True)
                stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                with FILE_LOCK:
                    shutil.copy2(target,backup/(stamp+'-'+target.name))
                    text=standalone(meta,body['code'])
                    target.write_text(text,encoding='utf-8')
                return self.send(json.dumps(dict(path='exercises/'+meta['file'],code=text),ensure_ascii=False))
            if meta['kind']=='concept':
                answers=body.get('answers',[])
                rows=[dict(label=q['prompt'],passed=i<len(answers) and answers[i]==q['answer'],actual=q['options'][answers[i]] if i<len(answers) and isinstance(answers[i],int) and 0<=answers[i]<len(q['options']) else '未选择',expected=q['options'][q['answer']],explain=q['explain']) for i,q in enumerate(meta['questions'])]
                return self.send(json.dumps(dict(rows=rows),ensure_ascii=False))
            deps=[]
            is_experiment=self.path=='/api/experiment'
            if is_experiment and not meta.get('lab'):raise ValueError('当前任务没有实验室')
            for id in meta.get('experiment_dependencies' if is_experiment else 'dependencies',[]):
                source=body.get('dependencies',{}).get(id)
                if not isinstance(source,str):
                    source=(EXERCISES/book['lessons'][id]['file']).read_text(encoding='utf-8')
                deps.append((id,source))
            payload=dict(exercises=str(EXERCISES),code=body['code'],cases=meta['cases'],kind=meta['kind'],dependencies=deps)
            if is_experiment: payload['experiment']=body.get('config',{})
            temp_root=HERE.parent/'.check-tmp';temp_root.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='transformer-check-',dir=temp_root) as tmp:
                proc=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=40 if is_experiment else 12,cwd=tmp)
            try: result=json.loads(proc.stdout.strip().splitlines()[-1])
            except (ValueError,IndexError):result=dict(error=proc.stderr[-4000:] or '程序没有返回测评结果。')
            if not is_experiment:save_after_pass(book,meta,body,deps,result)
            self.send(json.dumps(json_safe(result),ensure_ascii=False,allow_nan=False))
        except subprocess.TimeoutExpired:self.send(json.dumps(dict(error='运行超时，已停止。普通测评限 12 秒，训练实验限 40 秒；请减少步数或模型尺寸。'),ensure_ascii=False))
        except Exception as exc:self.send(json.dumps(dict(error=str(exc)),ensure_ascii=False),400)

if __name__=='__main__':
    print('学习网站：http://127.0.0.1:8124',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8124),Handler).serve_forever()
