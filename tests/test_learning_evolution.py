"""独立实现、拓展隔离及实验可复现性。"""
import ast,json,sys,subprocess,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exercises'),str(ROOT/'solutions')]
import transformer_layers as ref
from _transformer_lab import _generate_with_trace, experiment

class LearningEvolutionTests(unittest.TestCase):
    def test_generation_trace_keeps_the_draw_and_records_its_distribution(self):
        vocab = list(' abcdefghi')
        model = ref.GPT(len(vocab), 8, 2, 16, 1, 6, 7)
        prefix = np.array([[1, 2], [2, 1]])
        expected = ref.generate(model, prefix, 4, 0.8, 0, 17)
        result, trace = _generate_with_trace(ref.generate, model, prefix, 4, 0.8, 0, 17, vocab)
        self.assertEqual(result, [''.join(vocab[i] for i in row) for row in expected])
        for i, row in enumerate(trace):
            self.assertEqual(len(row['steps']), 4)
            for j, step in enumerate(row['steps']):
                probs = {candidate['char']: candidate['probability'] for candidate in step['candidates']}
                self.assertEqual(step['chosen'], vocab[expected[i, j + prefix.shape[1]]])
                self.assertIn(step['chosen'], probs)
                self.assertGreater(step['other'], 0)
                self.assertAlmostEqual(sum(probs.values()) + step['other'], 1.0)

    def test_optional_repairs_do_not_gate_main_path(self):
        book=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        main=[id for stage in book['stages'] if not stage.get('optional') for m in stage['modules'] for id in m['tasks']]
        extras=[id for stage in book['stages'] if stage.get('optional') for m in stage['modules'] for id in m['tasks']]
        self.assertEqual(set(main)|set(extras),set(book['lessons']))
        self.assertFalse(set(main)&set(extras))
        self.assertIn('rebuild_gpt',extras)
        self.assertIn('t_project',extras)
        self.assertEqual(main[-3:],['t_sample','t_generate','t_checkpoint'])
        for i in main:self.assertFalse(set(book['lessons'][i].get('dependencies',[]))&set(extras))
        self.assertIn('保持两个输入原值',book['lessons']['fix_residual']['body'])
        self.assertIn('目录中的训练实验室',book['lessons']['fix_clear']['body'])

    def test_constructor_and_methods_are_empty_in_independent_tasks(self):
        b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'))
        for id in ['rebuild_linear','rebuild_block','rebuild_gpt']:
            nodes=ast.parse(b['lessons'][id]['starter'])
            cls=next(n for n in nodes.body if isinstance(n,ast.ClassDef))
            self.assertIn('__init__',[n.name for n in cls.body if isinstance(n,ast.FunctionDef)])
            for fn in cls.body:
                if isinstance(fn,ast.FunctionDef):self.assertEqual(len(fn.body),1);self.assertIsInstance(fn.body[0],ast.Raise)
        self.assertTrue(b['lessons']['rebuild_gpt']['lab'])

    def test_different_text_validation_and_replay_are_deterministic(self):
        train=('清晨，小猫来到河边。小猫看见了鱼，鱼游向桥下。午后，小狗沿着河边跑，小猫坐在桥边等它。夜晚，它们一起回家。')*5
        val=('午后，小猫沿着河边跑。小狗坐在桥下等它。小猫看见了鱼，夜晚它们一起回家。')*3
        cfg=dict(text=train,validation_text=val,steps=4,tokens=3)
        a=experiment(vars(ref),cfg,{'entry':'source v1'});c=experiment(vars(ref),cfg,{'entry':'source v1'})
        for key in ['losses','norms','relative_updates','activation_rms','evaluation','generated','provenance']:self.assertEqual(a[key],c[key],key)
        self.assertEqual(a['split']['mode'],'单独提供验证文本')
        self.assertEqual(a['train_chars'],len(train));self.assertEqual(a['validation_chars'],len(val))
        self.assertEqual(len(a['evaluation']),5)
        self.assertTrue(all(np.isfinite(a['relative_updates'])))
        other=experiment(vars(ref),cfg,{'entry':'source v2'})
        self.assertNotEqual(a['provenance']['source_sha256'],other['provenance']['source_sha256'])
        with self.assertRaisesRegex(ValueError,'训练词表外字符'):experiment(vars(ref),{**cfg,'validation_text':'不存在的验证词汇'*5})

    def test_map_dimensions_and_shared_parameter_count(self):
        cfg=dict(B=2,T=3,C=8,H=2,V=12,L=2,F=32,context=12)
        source="global.esc=String;const m=require('./webapp/learning-tools.js');process.stdout.write(JSON.stringify(m.modelShapes("+json.dumps(cfg)+")));"
        out=subprocess.run(['node','-e',source],capture_output=True,text=True,cwd=ROOT,timeout=5)
        self.assertEqual(out.returncode,0,out.stderr);shapes=json.loads(out.stdout)
        self.assertEqual(shapes['heads'],[2,2,3,4]);self.assertEqual(shapes['scores'],[2,2,3,3]);self.assertEqual(shapes['logits'],[2,3,12])
        self.assertEqual(shapes['parameters'],sum(p.data.size for p in ref.GPT(12,8,2,32,2,12).parameters()))

    def test_independent_lab_uses_current_rebuilt_class(self):
        b=json.loads((ROOT/'webapp/foundations.json').read_text(encoding='utf-8'));meta=b['lessons']['rebuild_gpt']
        from foundation_core import Module,Parameter,initialize
        scope=dict(np=np,Module=Module,Parameter=Parameter,initialize=initialize,__name__='student')
        for id in meta['experiment_dependencies']:exec(b['lessons'][id]['reference'],scope)
        exec(meta['reference'],scope)
        result=experiment(scope,dict(text='abcdabcdbcdacdab'*10,steps=3,tokens=3))
        self.assertEqual(len(result['losses']),3)

    def test_lab_preserves_validation_text_and_discloses_vocabulary_source(self):
        train=('ab c\n'*20)
        validation=' \n'+('abc '*6)+'\n '
        cfg=dict(text=train,validation_text=validation,steps=1,tokens=1,T=3)
        result=experiment(vars(ref),cfg)
        self.assertEqual(result['config']['validation_text'],validation)
        self.assertEqual(result['validation_chars'],len(validation))
        self.assertIn('仅由训练语料',result['split']['vocab_policy'])
        split=experiment(vars(ref),{**cfg,'validation_text':''})
        self.assertIn('包含后20%',split['split']['vocab_policy'])
        with self.assertRaisesRegex(ValueError,'训练词表外字符'):
            experiment(vars(ref),{**cfg,'validation_text':'\t'+('abc '*6)})

    def test_lab_trains_on_corpus_and_continues_user_prompt(self):
        cfg=dict(text='ab c\n'*20,steps=2,tokens=3,T=3)
        prompt=' b\nab '
        baseline=experiment(vars(ref),cfg)
        result=experiment(vars(ref),{**cfg,'prompt':prompt})
        self.assertEqual(result['config']['prompt'],prompt)
        self.assertEqual(result['losses'],baseline['losses'])
        self.assertEqual(result['evaluation'],baseline['evaluation'])
        self.assertEqual(len(result['generated']),1)
        self.assertTrue(result['generated'][0].startswith(prompt))
        self.assertEqual(len(result['generated'][0]),len(prompt)+cfg['tokens'])
        self.assertEqual(experiment(vars(ref),{**cfg,'prompt':''})['generated'],baseline['generated'])
        with self.assertRaisesRegex(ValueError,'续写提示词含词表外字符'):
            experiment(vars(ref),{**cfg,'prompt':'新'})

    def test_lab_runs_up_to_three_equal_prompts_and_replays_in_order(self):
        cfg = dict(text='ab c\n' * 20, steps=2, tokens=3, T=3, seed=18)
        prompts = ['a', ' b\na ', 'c ab']
        result = experiment(vars(ref), {**cfg, 'prompts': prompts})
        self.assertEqual(result['config']['prompts'], prompts)
        self.assertEqual([item['prefix'] for item in result['generation_trace']], prompts)
        self.assertEqual([out[:len(p)] for out, p in zip(result['generated'], prompts)], prompts)
        self.assertEqual([len(item['steps']) for item in result['generation_trace']], [3, 3, 3])
        replay = experiment(vars(ref), result['config'])
        self.assertEqual(replay['generated'], result['generated'])
        self.assertEqual(replay['generation_trace'], result['generation_trace'])
        legacy = experiment(vars(ref), {**cfg, 'prompt': prompts[0], 'comparison_prompt': prompts[1]})
        self.assertEqual(legacy['losses'], result['losses'])
        self.assertEqual(legacy['generated'] + [legacy['comparison']], result['generated'][:2])
        empty_row = experiment(vars(ref), {**cfg, 'prompts': ['', prompts[1]]})
        self.assertEqual(empty_row['generated'], result['generated'][1:2])
        self.assertEqual(len(experiment(vars(ref), {**cfg, 'prompts': []})['generated']), 1)
        with self.assertRaisesRegex(ValueError, '最多添加 3 条'):
            experiment(vars(ref), {**cfg, 'prompts': prompts + ['a']})

if __name__=='__main__':unittest.main(verbosity=2)
