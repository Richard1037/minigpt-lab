"""验证学习契约对应真实测评与最终复用，而不是仅修改网页说明。"""
import copy
import html
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'webapp'),str(ROOT/'scripts')]
from _checks import evaluate
from _assessment import evaluate_special
from foundation_core import Parameter,Module,initialize
from pilot_server import RUNNER,standalone


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))

    def test_learning_order_and_explicit_dependencies(self):
        tasks=[task for stage in self.book['stages'] for module in stage['modules'] for task in module['tasks']]
        self.assertEqual(len(tasks),len(set(tasks)))
        self.assertEqual(set(tasks),set(self.book['lessons']))
        order={task:i for i,task in enumerate(tasks)}
        for task in tasks:
            for dependency in self.book['lessons'][task].get('dependencies') or []:
                self.assertLess(order[dependency],order[task],(dependency,task))
        for first,second in [('broadcast','reduce'),('reduce','mean_backward'),
                             ('mean_backward','center_backward'),('att_weighted','att_value_backward'),
                             ('att_value_backward','att_score_backward'),
                             ('t_attention','t_mha')]:
            self.assertLess(order[first],order[second],(first,second))

    def test_centering_explains_and_preserves_both_gradient_paths(self):
        reduce=self.book['lessons']['reduce']
        self.assertIn('N=C',reduce['body'])
        derivation=self.book['lessons']['center_backward']['derivation']
        self.assertIn('直接路径',derivation)
        self.assertIn('均值路径',derivation)
        self.assertIn(r'\frac{\partial L}{\partial\mu}',derivation)
        x=np.array([[[1.,2.,4.],[3.,-2.,1.]],[[2.,0.,-1.],[5.,4.,3.]]])
        g=np.array([[[2.,-1.,3.],[1.,4.,-2.]],[[3.,2.,-4.],[1.,-3.,5.]]])
        scope={'np':np};exec(self.book['lessons']['center_backward']['reference'],scope)
        actual=scope['solve'](g)
        def loss(a): return np.sum((a-a.mean(axis=-1,keepdims=True))*g)
        for index in np.ndindex(x.shape):
            plus=x.copy();minus=x.copy();plus[index]+=1e-5;minus[index]-=1e-5
            self.assertAlmostEqual(actual[index],(loss(plus)-loss(minus))/2e-5,places=7)
        square=np.array([[1.,2.],[10.,20.]])
        correct=square-square.mean(axis=-1,keepdims=True)
        wrong=square-square.mean(axis=-1)
        self.assertFalse(np.allclose(correct,wrong))

    def test_attention_prerequisites_and_causal_contract_are_explicit(self):
        first=self.book['lessons']['att_scores']['body']
        for symbol in ('Q','K','V','Linear'):
            self.assertIn(symbol,first)
        conditions=self.book['lessons']['t_attention']['body']
        self.assertIn('causal=True',conditions)
        self.assertIn('要求两个序列长度相等',conditions)
        self.assertIn('非因果模式',conditions)
        scope={'np':np,'Module':Module}
        for lesson in [*self.book['lessons']['t_attention']['dependencies'],'t_attention']:
            exec(self.book['lessons'][lesson]['reference'],scope)
        q=np.zeros((2,2,2,3));k=np.zeros((2,2,3,3));v=np.zeros((2,2,3,4))
        with self.assertRaises(ValueError):scope['AttentionCore'](True).forward(q,k,v)
        self.assertEqual(scope['AttentionCore'](False).forward(q,k,v).shape,(2,2,2,4))
        init=re.sub(r'</?code>','',self.book['lessons']['initialization']['body'])
        self.assertIn('标准差',init)
        self.assertIn(r'\sqrt{\frac{2}{C}}',html.unescape(init))
        self.assertNotIn('残差',init)
        self.assertEqual(self.book['lessons']['initialization']['kind'],'initialization')
        softmax=self.book['lessons']['t_softmax']
        self.assertIn('公共分母',softmax['body'])
        self.assertIn(r'\frac{\partial p_i}{\partial x_j}',softmax['body']+softmax['derivation'])
        self.assertIn('不是给每个层单独求一个系数',self.book['lessons']['t_optim']['body'])

    def test_public_interface_explains_traversal_and_initialization(self):
        body=self.book['lessons']['infrastructure']['body']
        for phrase in ('深度优先遍历','vars(value).items()','同一个对象','循环引用','zero_grad'):
            self.assertIn(phrase,body)
        init=self.book['lessons']['initialization']['body']
        for phrase in ('C=4','rng.standard_normal(shape)','rng=None'):
            self.assertIn(phrase,init)
        for phrase in ('单个输出的方差','输入分量独立'):
            self.assertIn(phrase,self.book['lessons']['initialization']['extension'])
        snippet=next(html.unescape(s) for s in re.findall(r'<pre>(.*?)</pre>',body,re.S) if 'def parameters(self):' in s)
        scope={'Parameter':Parameter,'Module':Module}
        exec(snippet,scope)
        root=Module();child=Module();a=Parameter([1]);b=Parameter([2]);c=Parameter([3])
        root.weight=a;root.blocks=[child,{'shared':a}];child.W=b;child.b=c;root._cache=Parameter([4])
        self.assertEqual(scope['parameters'](root),[a,b,c])

    def test_learner_layout_teaches_before_model_problem(self):
        infra=self.book['lessons']['infrastructure']
        case=infra['cases'][0]
        scope={'np':np};exec(infra['reference'],scope)
        parameter=scope['Parameter']([[1,2]])
        child=scope['Module']();child.weight=parameter
        root=scope['Module']();root.child=child;root.shared=parameter;root.blocks=[child]
        parameter.grad[...] = 5
        count=len(root.parameters());root.zero_grad()
        self.assertEqual({'参数形状':list(parameter.data.shape),'共享参数计数':count,'清零后梯度':parameter.grad.tolist()},case['expected'])
        script=r'''const fs=require('fs'),vm=require('vm');
const src=fs.readFileSync('webapp/app.js','utf8');
const context={esc:s=>String(s),taskBrief:()=>'<div>LEGACY-CONTRACT</div>'};
vm.createContext(context);vm.runInContext(src.slice(src.indexOf('const lessonFoldStates='),src.indexOf('function render(){')),context);
const book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
for(const id of ['infrastructure','initialization']){
 const meta=book.lessons[id],out=context.learningContent(meta,meta.kind==='concept');
 if(!(out.indexOf(meta.body)<out.indexOf(meta.learning_page.statement)))throw Error('Problem precedes teaching');
 if(out.includes('LEGACY-CONTRACT'))throw Error('Duplicated old contract');
 if(!out.includes('输入格式')||!out.includes('输出格式'))throw Error('Missing input/output');
 if((out.match(/id="sample"/g)||[]).length!==(meta.kind==='concept'?0:1))throw Error('Wrong sample host');
 if(meta.kind!=='concept'&&(!out.includes('data-lesson-fold="sample">')||!out.includes('data-lesson-fold="animation" open')))throw Error('Missing independent sample/demo folds');
}
if(!context.learningContent({body:'old',derivation:''},false).includes('LEGACY-CONTRACT'))throw Error('Old lesson changed');
'''
        proc=subprocess.run(['node','-e',script],cwd=ROOT,text=True,capture_output=True,encoding='utf-8',timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr)

    def test_late_course_contracts_match_examples_and_checkpoint_probe(self):
        data=self.book['lessons']['t_data'];case=data['cases'][0]
        scope={'np':np};exec(data['reference'],scope)
        vocab,ids=scope['tokenize'](case['input']['text'])
        x,y=scope['make_batch'](ids,case['input']['starts'],case['input']['T'])
        self.assertEqual(''.join(vocab),case['expected']['vocab'])
        self.assertEqual(x.tolist(),case['expected']['X'])
        self.assertEqual(y.tolist(),case['expected']['targets'])
        self.assertIn('0≤s', ' '.join(data['task_brief']['conditions']))
        self.assertIn('起点 3',data['body'])
        self.assertIn('T0≥1',next(v['shape'] for v in self.book['lessons']['t_generate']['task_brief']['variables'] if v['name']=='ids'))
        self.assertIn('另行记录',self.book['lessons']['t_checkpoint']['derivation'])
        js=r'''const fs=require('fs'),vm=require('vm'),src=fs.readFileSync('webapp/app.js','utf8');
const part=src.slice(src.indexOf('function caseInput('),src.indexOf('function taskBrief('));
const context={m:()=>({kind:'project'}),current:'t_checkpoint',esc:String,arrayView:(name,value)=>`[${name}=${JSON.stringify(value)}]`,constructorView:value=>`[init=${JSON.stringify(value)}]`};
vm.createContext(context);vm.runInContext(part,context);
const rendered=context.caseInput({mode:'checkpoint',input:{init:[1,2],ids:[[0]],targets:[[1]],steps:12,lr:.02}});
if(rendered.includes('12')||rendered.includes('0.02')||!rendered.includes('0.01')||!rendered.includes('42'))process.exit(1);'''
        proc=subprocess.run(['node','-e',js],cwd=ROOT,text=True,capture_output=True,encoding='utf-8',timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr)

    def test_shape_float_return_rejected_and_heads_not_hardcoded(self):
        meta=self.book['lessons']['shape']
        rows=evaluate(lambda values:np.array(values.shape,dtype=np.float64),meta['cases'])
        self.assertFalse(any(r['passed'] for r in rows))
        self.assertTrue(all('dtype' in ' '.join(r['diagnostics']) for r in rows))
        heads=self.book['lessons']['heads'];scope={'np':np};exec(heads['reference'],scope)
        self.assertTrue(all(r['passed'] for r in evaluate(scope['solve'],heads['cases'])))
        wrong=lambda x,h:x.reshape(x.shape[0],x.shape[1],2,x.shape[-1]//2).transpose(0,2,1,3)
        self.assertFalse(all(r['passed'] for r in evaluate(wrong,heads['cases'])))
        self.assertEqual({int(c['args'][1]) for c in heads['cases']},{2,3,4})

    def test_personal_parameter_type_reaches_gpt_and_gradient_probe(self):
        scope=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize)
        meta=self.book['lessons']['t_gpt']
        for dep in meta['dependencies']:exec(self.book['lessons'][dep]['reference'],scope)
        exec(meta['reference'],scope)
        self.assertIsNot(scope['Parameter'],Parameter)
        model=scope['GPT'](5,4,2,8,2,6,7)
        self.assertIs(type(model.token.weight),scope['Parameter'])
        case=next(c for c in meta['cases'] if c.get('assessment')=='gradient')
        result=evaluate_special(scope,case)
        self.assertTrue(result['passed'],result)
        self.assertGreater(result['actual']['参数张量数'],10)

    def test_full_source_import_cannot_replace_personal_infrastructure(self):
        meta=self.book['lessons']['t_gpt'];deps=[]
        for id in meta['dependencies']:
            source=self.book['lessons'][id]['reference']
            if id=='infrastructure':source=source.replace('self.data = np.array(data, dtype=np.float64, copy=True)','raise RuntimeError("PERSONAL_PARAMETER_USED")')
            deps.append([id,standalone(self.book['lessons'][id],source)])
        payload=dict(exercises=str(ROOT/'exercises'),code=standalone(meta,meta['reference']),kind=meta['kind'],cases=[meta['cases'][0]],dependencies=deps)
        proc=subprocess.run([sys.executable,'-I','-c',RUNNER],input=json.dumps(payload),text=True,capture_output=True,encoding='utf-8',timeout=15)
        result=json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertFalse(any(r['passed'] for r in result.get('rows',[])))
        self.assertIn('PERSONAL_PARAMETER_USED',json.dumps(result))

    def test_source_rebuild_and_strict_math(self):
        from course_math import apply_math
        from refine_course import apply_refinements
        book=copy.deepcopy(self.book)
        apply_refinements(book);apply_math(book)
        self.assertEqual(book,self.book)
        self.assertIn('有限差分',book['common_extension'])
        for task in ('reduce','layer_gelu'):
            self.assertTrue(book['lessons'][task]['extension'])
        self.assertFalse(book['lessons']['broadcast']['extension'])
        self.assertTrue(all('数值误差分析' not in meta['extension'] for meta in book['lessons'].values()))
        formulas=[html.unescape(s) for s in re.findall('data-tex="([^"]+)"',book['common_extension'])]
        for meta in book['lessons'].values():
            self.assertTrue(meta['task_brief']['variables'])
            for field in ('body','derivation','extension'):
                formulas.extend(html.unescape(s) for s in re.findall('data-tex="([^"]+)"',meta.get(field,'')))
            formulas.extend(meta.get('formula_fragments',{}).values())
            formulas.extend(html.unescape(s) for s in re.findall('data-tex="([^"]+)"',meta.get('learning_page',{}).get('statement','')))
        script="const k=require('./webapp/vendor/katex/katex.min.js');let s='';process.stdin.on('data',d=>s+=d);process.stdin.on('end',()=>{const a=JSON.parse(s);for(const tex of a)k.renderToString(tex,{displayMode:true,throwOnError:true,strict:'error',trust:false});console.log(a.length);});"
        proc=subprocess.run(['node','-e',script],input=json.dumps(formulas),cwd=ROOT,text=True,capture_output=True,encoding='utf-8',timeout=12)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertGreater(int(proc.stdout),70)

    def test_bridge_and_cache_bugs_are_rejected(self):
        wrong={
            'mean_backward':lambda g,c:np.broadcast_to(g,(*g.shape[:-1],int(c))).copy(),
            'center_backward':lambda g:g.copy(),
            'restore':lambda y,s:y.copy(),
            'random_sequence':lambda seed,dims:np.stack([np.random.default_rng(int(seed)).standard_normal(tuple(dims))]*2),
            'att_value_backward':lambda a,g:a@g,
            'att_score_backward':lambda a,da:a*da,
        }
        for id,func in wrong.items():
            with self.subTest(id=id):self.assertFalse(all(r['passed'] for r in evaluate(func,self.book['lessons'][id]['cases'])))
        scope=dict(np=np,Parameter=Parameter,Module=Module,initialize=initialize)
        exec(self.book['lessons']['layer_linear']['reference'].replace('x.reshape(-1, self.W.data.shape[0]).copy()','x.reshape(-1, self.W.data.shape[0])'),scope)
        case=next(c for c in self.book['lessons']['layer_linear']['cases'] if c.get('assessment')=='cache')
        self.assertFalse(evaluate_special(scope,case)['passed'])

    def test_process_coordinate_maps_keep_every_batch_and_head(self):
        script=r"""
const fs=require('fs'),m=require('./webapp/process-animations.js'),katex=require('./webapp/vendor/katex/katex.min.js');
const book=JSON.parse(fs.readFileSync('webapp/foundations.json','utf8'));
let count=0;
for(const id of ['shape','axes','reduce','broadcast','reshape','restore','transpose','heads','copy','att_merge','scatter','layer_embedding','mean_backward','center_backward','t_loss','t_data','t_sample','t_generate','t_layernorm','t_dropout','att_scores','att_probs','att_weighted','att_value_backward','att_score_backward']){
 const lesson=book.lessons[id],cases=id==='layer_embedding'?[lesson.cases[lesson.sample_index||0]]:lesson.cases.filter(c=>!c.assessment);
 for(const c of cases){
  const showScores=id==='t_generate'&&c===lesson.cases[lesson.sample_index||0];
  const scores=showScores?lesson.generation_scores:undefined;
  if(showScores&&scores?.length!==c.input.steps)throw Error('Missing generation scores');
  const p=m.makePlan(id,c,scores),state=new Map(),key=(id,c)=>id+':'+c.join(',');
  if(scores){
   const expectedPanels=['prompt','window','scores','probabilities','sample','result'];
   if(p.panels.map(panel=>panel.id).join(',')!==expectedPanels.join(','))throw Error('Missing or duplicate generation panels');
   if(p.events.some(event=>event.visible.join(',')!==expectedPanels.join(',')))throw Error('Generation panels move between steps');
   if(p.panels.some(panel=>panel.id!=='prompt'&&panel.initial==='full'))throw Error('Generation panels reveal future values');
   for(let round=0;round<c.input.steps;round++){
    const [window,ranking,draw]=p.events.slice(1+3*round,4+3*round);
    const writes=draw.moves.filter(op=>op.to==='sample');
    if(window.clear.join(',')!=='window,scores,probabilities')throw Error('Generation window retains stale scores');
    if([...window.moves,...ranking.moves].some(op=>op.to==='sample')||writes.length!==c.input.ids.length)throw Error('Generation reveal timing');
    scores[round].forEach((row,b)=>{
     const kept=row.map((_,j)=>j).sort((a,d)=>row[a]-row[d]||a-d).slice(-(c.input.top_k||row.length));
     const probabilities=ranking.moves.filter(op=>op.to==='probabilities'&&op.target[0]===b);
     if(ranking.extra.filter(op=>op.coord[0]===b).map(op=>op.coord[1]).join(',')!==kept.join(','))throw Error('Incorrect top-k display');
     if(probabilities.length!==row.length||probabilities.some(op=>!kept.includes(op.target[1])&&op.value!=='0%'))throw Error('Incorrect probability masking');
     if(Math.abs(probabilities.reduce((sum,op)=>sum+parseFloat(op.value),0)-100)>0.25)throw Error('Generation probabilities do not sum to 100%');
     if(writes[b].target.join(',')!==`${b},${round}`||!kept.includes(writes[b].value))throw Error('Incorrect sampled candidate or round');
    });
   }
  }
  for(const panel of p.panels)if(panel.initial==='full')for(const co of m.coords(panel.shape))state.set(key(panel.id,co),co.reduce((v,i)=>v[i],panel.values));
  for(const event of p.events){
   if(event.tex)katex.renderToString(event.tex,{displayMode:true,throwOnError:true,strict:'error'});
   for(const panel of p.panels.filter(panel=>event.clear.includes(panel.id)))for(const co of m.coords(panel.shape))state.delete(key(panel.id,co));
   for(const op of event.moves){const value=op.value??state.get(key(op.from,op.source));if(value===undefined)throw Error(id+' has missing source '+key(op.from,op.source));const to=key(op.to,op.target);state.set(to,op.mode==='add'?(state.get(to)||0)+value:value);}
  }
  for(const [panel,expected] of Object.entries(p.expected))for(const co of m.coords(m.shape(expected))){const a=state.get(key(panel,co)),e=co.reduce((v,i)=>v[i],expected);if(a===undefined||Math.abs(a-e)>1e-9)throw Error(id+' '+key(panel,co)+' '+a+' != '+e);}
  count++;
 }
}
console.log(count);
"""
        proc=subprocess.run(['node','-e',script],cwd=ROOT,text=True,capture_output=True,encoding='utf-8',timeout=15)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertGreater(int(proc.stdout),25)


if __name__=='__main__':unittest.main(verbosity=2)
