import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sync_scaffolds import sync_scaffolds

OLD='''class Model:
    def __init__(self):
        self.value = 1
    def forward(self, x):
        raise NotImplementedError("TODO")
'''
NEW=OLD+'\nif __name__ == "__main__":\n    print("test")\n'
class ScaffoldTests(unittest.TestCase):
    def book(self,source=OLD):return {'lessons':{'task':dict(file='task.py',sync_empty_scaffold=True,local_code=source,scaffold_notes=dict(source=source,starter=NEW))}}
    def test_exact_old_empty_template_is_backed_up_and_synced(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as directory:
            p=Path(directory);(p/'task.py').write_text(OLD,encoding='utf-8');b=self.book()
            self.assertEqual(sync_scaffolds(b,p)['updated'],['task'])
            self.assertEqual((p/'task.py').read_text(encoding='utf-8'),NEW)
            self.assertEqual(next((p/'backups').glob('*.py')).read_text(encoding='utf-8'),OLD)
            self.assertEqual(b['lessons']['task']['local_code'],NEW)
            self.assertFalse(sync_scaffolds(b,p)['updated'])
    def test_personal_implementation_or_notes_are_never_replaced(self):
        for source in [OLD.replace('raise NotImplementedError("TODO")','return x'),OLD+'\n# 我的笔记\n',OLD.replace('raise NotImplementedError("TODO")','self.x = x\n        raise NotImplementedError("TODO")')]:
            with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as directory:
                p=Path(directory);(p/'task.py').write_text(source,encoding='utf-8')
                self.assertFalse(sync_scaffolds(self.book(),p)['updated'])
                self.assertEqual((p/'task.py').read_text(encoding='utf-8'),source)
                self.assertFalse((p/'backups').exists())
    def test_even_matching_metadata_cannot_replace_implemented_methods(self):
        source=OLD.replace('raise NotImplementedError("TODO")','return x')
        with tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp') as directory:
            p=Path(directory);(p/'task.py').write_text(source,encoding='utf-8')
            self.assertFalse(sync_scaffolds(self.book(source),p)['updated'])
if __name__=='__main__':unittest.main()
