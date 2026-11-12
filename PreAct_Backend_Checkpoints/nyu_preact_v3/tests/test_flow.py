"""Fast CPU orchestration test with a tiny stand-in model, NOT an accuracy experiment.
Real PreAct forward/backward and formula parity are tested in test_pipeline.py.
"""
import contextlib
import copy
import csv
import io
import json
import signal
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace as Args
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core
import run
from PIL import Image
from core import torch

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,4,1);self.bn=torch.nn.BatchNorm2d(4)
        self.head=torch.nn.Linear(4,10);self.joint_head=torch.nn.Linear(4,40);self.scale=1.
    def features(self,x):return self.bn(self.conv(x)).mean((2,3))

class Flow(unittest.TestCase):
    def test_development_resume_select_full_export_replay(self):
        torch.set_num_threads(2);core.seed_all(42)
        cfg=run.read_config(core.ROOT/'configs/development.json')
        cfg.update(epochs=3,batch_size=4,warmup_epochs=1,workers=0,calibration_images=4)
        x=torch.randn(4,3,32,32);y=torch.tensor([0,1,2,3]);tr=list(range(8));va=list(range(8,5008))
        def inputs(images,labels,c):return [],{},tr if c['validation_size'] else tr+va,va if c['validation_size'] else []
        def batches(*args,**kwargs):return [(x,y)]
        # Disabled scaler/autocast let this exercise checkpoint orchestration on CPU.
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        oldsignals={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack:
                d=Path(folder);buf=io.BytesIO();Image.new('RGB',(32,32),(3,30,60)).save(buf,format='PNG')
                with zipfile.ZipFile(d/'test.zip','w') as z:
                    for i in range(2000):z.writestr(f'{i}.png',buf.getvalue())
                for name,head,rows in [('sample.csv',['id','label'],[(i,0) for i in range(2000)]),('test.csv',['id','filename'],[(i,f'{i}.png') for i in range(2000)])]:
                    with (d/name).open('w',newline='') as f:w=csv.writer(f);w.writerow(head);w.writerows(rows)
                test=core.UnlabeledCompetitionDataset(d/'sample.csv',d/'test.zip',core.transform(),d/'test.csv')
                sig=dict(images=core.test_fingerprint(test),metadata=core.digest(d/'test.csv'),sample=core.digest(d/'sample.csv'))
                gate=dict(test_signature=sig);dev=d/'development';dev.mkdir()
                stack.enter_context(patch.object(run,'make_model',Tiny))
                stack.enter_context(patch.object(run,'load_training',inputs))
                stack.enter_context(patch.object(run,'loader',batches))
                stack.enter_context(patch.object(run,'dataset_fingerprint',return_value='synthetic-only'))
                stack.enter_context(patch.object(torch.amp,'GradScaler',return_value=scaler))
                stack.enter_context(patch.object(torch,'autocast',return_value=contextlib.nullcontext()))
                stack.enter_context(patch.object(run,'budget',return_value=3600))
                # Resume from the saved complete run is a no-op and must not append epochs.
                a=Args(device='cpu',seconds=3600,policy=None)
                run.train_locked(a,cfg,dev,[],{},tr,va,'synthetic-only',None,gate)
                ck=torch.load(dev/'last.pt',weights_only=False)
                self.assertEqual(ck['completed_epochs'],3)
                before=copy.deepcopy(ck['raw'])
                run.train_locked(a,cfg,dev,[],{},tr,va,'synthetic-only',ck,gate)
                resumed=torch.load(dev/'last.pt',weights_only=False)
                self.assertEqual(len(resumed['history']),3)
                for k,v in before.items():torch.testing.assert_close(v,resumed['raw'][k],atol=0,rtol=0)
                selection=d/'selection.json'
                run.select(Args(run=dev,images='unused',labels='unused',device='cpu',batch_size=64,output=selection))
                policy=json.loads(selection.read_text())
                self.assertEqual(set(policy['variants']),{'raw','ema','flip'})
                full_config=d/'full.json';run.promote(Args(policy=selection,output=full_config))
                full_cfg=json.loads(full_config.read_text());self.assertEqual(full_cfg['validation_size'],0)
                full=d/'full';full.mkdir()
                a.policy=selection
                run.train_locked(a,full_cfg,full,[],{},tr+va,[],'synthetic-only',None,gate)
                pred=Args(policy=selection,variant='selected',checkpoint=full/'last.pt',images='unused',labels='unused',test_images=d/'test.zip',test_metadata=d/'test.csv',sample=d/'sample.csv',output=d/'export',device='cpu',batch_size=64)
                run.predict(pred)
                self.assertEqual(len(core.read_csv(d/'export/submission.csv')),2000)
                pred.checkpoint=d/'export/model.pt';pred.output=d/'replay'
                run.replay(pred)
                self.assertEqual((d/'export/submission.csv').read_bytes(),(d/'replay/submission.csv').read_bytes())
                with self.assertRaises(FileExistsError):run.replay(pred)
        finally:
            for sig,handler in oldsignals.items():signal.signal(sig,handler)

if __name__=='__main__':unittest.main(verbosity=2)
