import copy,json,sys,tempfile,unittest,shutil
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'webapp'),str(ROOT/'exercises')]
import pilot_server as server

class AutoSaveTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.check-tmp');self.folder=Path(self.temp.name)
  self.meta=dict(id='main',file='main.py',kind='array',cases=[dict(label='same',args=[[1,2]],expected=[1,2])])
  self.book={'lessons':{'main':self.meta,'dep':dict(id='dep',file='dep.py',kind='array')}}
  for name in ['main.py','dep.py']:(self.folder/name).write_text('# original '+name,encoding='utf-8')
  self.files=patch.object(server,'EXERCISES',self.folder);self.files.start()
  self.wrap=patch.object(server,'standalone',lambda m,s:s);self.wrap.start()
 def tearDown(self):self.wrap.stop();self.files.stop();self.temp.cleanup()
 def result(self):return {'rows':[{'passed':True}]}
 def test_passed_snapshot_and_dependencies_saved_with_backups(self):
  result=self.result();source='value=7\n'
  server.save_after_pass(self.book,self.meta,dict(auto_save=True,code=source),[('dep','value=9\n')],result,verify=lambda m:None)
  self.assertTrue(result['auto_save']['ok']);self.assertEqual((self.folder/'main.py').read_text(),source)
  self.assertEqual((self.folder/'dep.py').read_text(),'value=9\n');self.assertEqual(len(list((self.folder/'backups').iterdir())),2)
 def test_failure_partial_results_and_optout_never_write(self):
  for result,flag in [({'rows':[{'passed':False}]},True),({'rows':[]},True),(self.result(),False),({'error':'failed'},True)]:
   server.save_after_pass(self.book,self.meta,dict(auto_save=flag,code='x=3'),[],result,verify=lambda m:None)
   self.assertEqual((self.folder/'main.py').read_text(),'# original main.py');self.assertNotIn('auto_save',result)
 def test_local_failure_restores_every_file(self):
  def fail(m):raise ValueError('local import failed')
  result=self.result();server.save_after_pass(self.book,self.meta,dict(auto_save=True,code='x=3'),[('dep','x=4')],result,verify=fail)
  self.assertFalse(result['auto_save']['ok'])
  for name in ['main.py','dep.py']:self.assertEqual((self.folder/name).read_text(),'# original '+name)
 def test_integration_saves_all_modules_without_replacing_driver(self):
  m=copy.deepcopy(self.meta);m['integration_only']=True;result=self.result()
  server.save_after_pass(self.book,m,dict(auto_save=True,code='driver_should_not_be_saved=1'),[('dep','x=9')],result,verify=lambda m:None)
  self.assertTrue(result['auto_save']['ok']);self.assertEqual((self.folder/'main.py').read_text(),'# original main.py');self.assertEqual((self.folder/'dep.py').read_text(),'x=9')
 def test_real_local_import_validation_passes_and_rejects_bad_file(self):
  # 在临时副本验证真实本地导入，完全不改个人文件。
  for src in (ROOT/'exercises').glob('*.py'):shutil.copy2(src,self.folder/src.name)
  shutil.copy2(ROOT/'exercises/cases.json',self.folder/'cases.json')
  (self.folder/'main.py').write_text('import numpy as np\ndef solve(x): return np.array(x)\n',encoding='utf-8')
  server.verify_saved(self.meta)
  (self.folder/'main.py').write_text('import numpy as np\ndef solve(x): return np.array([99])\n',encoding='utf-8')
  with self.assertRaises(ValueError):server.verify_saved(self.meta)

if __name__=='__main__':unittest.main()
