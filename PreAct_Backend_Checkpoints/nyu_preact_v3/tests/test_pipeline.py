"""CPU checks of loss, resume, inference, input alignment, and file preservation."""
import csv
import copy
import io
import tempfile
import unittest
import zipfile
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core
import run
import dataset
from core import torch, F, np
from PIL import Image

class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        core.seed_all(42)
        self.cfg=run.read_config(core.ROOT/'configs/development.json')

    def test_parameter_count_and_original_loss_gradients(self):
        m=core.make_model().eval()
        self.assertEqual(core.count_params(m),4706674)
        x=torch.randn(2,3,32,32); y=torch.tensor([2,7])
        # Independently written original notebook loss, including stop-gradient teacher.
        z=m.features(torch.stack([torch.rot90(x,k,(2,3)) for k in range(4)],1).flatten(0,1))
        j=m.joint_head(z); p=m.head(z[::4]); target=(y[:,None]*4+torch.arange(4)).flatten()
        joint=F.cross_entropy(j,target,reduction='none').view(2,4).mean(1)
        aggregate=torch.stack([j.view(2,4,40)[:,k,k::4] for k in range(4)]).mean(0).detach()
        sd=F.kl_div(p.log_softmax(1),aggregate.softmax(1),reduction='none').sum(1)
        expected=(joint+sd+F.cross_entropy(p,y,label_smoothing=.1,reduction='none')).mean()
        actual,_=core.sla_loss(m,x,y,.1)
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,(m.stem.weight,m.head.weight,m.joint_head.weight))
        ge=torch.autograd.grad(expected,(m.stem.weight,m.head.weight,m.joint_head.weight))
        for a,b in zip(ga,ge):torch.testing.assert_close(a,b)

    def test_original_inference_and_batch_independence(self):
        m=core.make_model().eval();x=torch.randn(2,3,32,32)
        with torch.no_grad():
            def agg(x):
                j=m.joint_head(m.features(core.rotate4(x))).view(len(x),4,40)
                return torch.stack([j[:,k,k::4] for k in range(4)]).mean(0)
            def original(fn):
                mirror=lambda v:(fn(v)+fn(v.flip(-1)))*.5
                p=F.pad(x,(1,)*4,mode='reflect')
                return (mirror(x)*.5+(mirror(p[:,:,:32,:32])+mirror(p[:,:,2:,2:]))*.25).softmax(1)
            expected=(original(m)+original(agg))*.5
            actual=core.predict_policies(m,x,('none','flip','original6'))
            torch.testing.assert_close(actual['original6','combined'],expected)
            single=torch.cat([core.predict_policies(m,v[None],('none',))['none','plain'] for v in x])
            torch.testing.assert_close(actual['none','plain'],single,atol=2e-6,rtol=2e-5)

    def test_checkpoint_resume_next_random_update(self):
        m=core.make_model();e=copy.deepcopy(m);opt=core.optimizer(m,self.cfg)
        def step(net,avg,optimizer):
            x=torch.randn(2,3,32,32);y=torch.randint(0,10,(2,))
            optimizer.zero_grad();v,_=core.sla_loss(net,x,y,.1);v.backward();optimizer.step();core.ema_update(avg,net,.999)
        step(m,e,opt)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt'
            core.atomic_save(p,dict(raw=core.cpu_state(m),ema=core.cpu_state(e),optimizer=opt.state_dict(),rng=core.rng_get()))
            ck=torch.load(p,weights_only=False)
            step(m,e,opt)
            resumed=core.make_model();re=copy.deepcopy(resumed);ro=core.optimizer(resumed,self.cfg)
            resumed.load_state_dict(ck['raw']);re.load_state_dict(ck['ema']);ro.load_state_dict(ck['optimizer']);core.rng_set(ck['rng'])
            step(resumed,re,ro)
            for a,b in zip(m.state_dict().values(),resumed.state_dict().values()):torch.testing.assert_close(a,b,atol=0,rtol=0)
            for a,b in zip(e.state_dict().values(),re.state_dict().values()):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_ema_and_bn_calibration_preserve_weights(self):
        m=core.make_model();e=copy.deepcopy(m)
        with torch.no_grad():m.stem.weight.add_(2);m.bn.running_mean.fill_(4);m.bn.num_batches_tracked.fill_(3)
        old=e.stem.weight.detach().clone();core.ema_update(e,m,.9)
        torch.testing.assert_close(e.stem.weight,old+.2)
        torch.testing.assert_close(e.bn.running_mean,m.bn.running_mean)
        self.assertEqual(int(e.bn.num_batches_tracked),3)
        params={k:p.detach().clone() for k,p in e.named_parameters()}
        core.calibrate(e,[(torch.randn(2,3,32,32),torch.tensor([1,2]))],'cpu')
        self.assertEqual(int(e.bn.num_batches_tracked),1)
        self.assertFalse(e.training);self.assertEqual(e.bn.momentum,.1)
        for k,p in e.named_parameters():torch.testing.assert_close(p,params[k],rtol=0,atol=0)

    def test_schedule_and_budget(self):
        self.assertAlmostEqual(core.learning_rate(499,32000,500,.1,.00005),.1)
        self.assertAlmostEqual(core.learning_rate(31999,32000,500,.1,.00005),.00005)
        self.assertEqual(run.duration('1-02:03:04'),93784)
        self.assertEqual(run.duration('04:00'),240)
        with patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch.object(run.shutil,'which',return_value='/bin/squeue'),patch.object(run.subprocess,'run') as s:
            s.return_value.stdout='00:04:00'
            with self.assertRaises(RuntimeError):run.budget(14400)
            s.return_value.stdout='00:30:00'
            self.assertEqual(run.budget(14400),1800)
            self.assertEqual(run.budget(0),1800)
            s.return_value.stdout='10:00:00'
            self.assertEqual(run.budget(0),36000)
            self.assertEqual(run.budget(21600),21600)
            s.return_value.stdout=''
            with self.assertRaisesRegex(RuntimeError,'active allocation'):run.budget(0)
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(run.budget(0),float('inf'))
            with self.assertRaises(ValueError):run.budget(-1)

    def test_split_reproducibility_and_normalization(self):
        samples=[dataset.LabeledImage(str(i),Path(f'{i}.png'),str(i%10)) for i in range(50000)]
        tr,va=dataset.stratified_split(samples,5000,42)
        self.assertEqual((len(tr),len(va)),(45000,5000));self.assertFalse(set(tr)&set(va))
        self.assertEqual((tr,va),dataset.stratified_split(samples,5000,42))
        self.assertEqual([sum(samples[i].label==str(k) for i in va) for k in range(10)],[500]*10)
        arr=np.random.default_rng(4).integers(0,256,(32,32,3),dtype=np.uint8);im=Image.fromarray(arr)
        manual=(torch.from_numpy(arr.copy()).permute(2,0,1).float()/255-torch.tensor(core.MEAN)[:,None,None])/torch.tensor(core.STD)[:,None,None]
        torch.testing.assert_close(core.transform()(im),manual,rtol=0,atol=0)
        train_ops=core.transform(True,self.cfg).transforms
        self.assertEqual(train_ops[-2].mean,core.MEAN);self.assertEqual(train_ops[-2].std,core.STD)

    def test_zip_alignment_missing_duplicate_reordered(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);buf=io.BytesIO();Image.new('RGB',(32,32),(5,30,80)).save(buf,format='PNG')
            zpath=d/'test.zip'
            with zipfile.ZipFile(zpath,'w') as z:
                for i in range(2000):z.writestr(f'test_images/x{i}.png',buf.getvalue())
            ids=[str(i) for i in range(2000)]
            def write(path,head,rows):
                with path.open('w',newline='') as f:w=csv.writer(f);w.writerow(head);w.writerows(rows)
            write(d/'sample.csv',['id','label'],[(i,0) for i in ids])
            rows=[(i,f'x{i}.png') for i in ids]
            write(d/'test.csv',['id','filename'],rows)
            ds=dataset.UnlabeledCompetitionDataset(d/'sample.csv',zpath,core.transform(),d/'test.csv')
            self.assertEqual(ds[123][1],'123');self.assertEqual(len(ds),2000)
            for bad in (list(reversed(rows)),rows[:-1],rows[:-1]+[rows[0]]):
                write(d/'bad.csv',['id','filename'],bad)
                with self.assertRaises(ValueError):dataset.UnlabeledCompetitionDataset(d/'sample.csv',zpath,None,d/'bad.csv')
            write(d/'bad.csv',['id','filename'],rows[:-1]+[('1999','missing.png')])
            with self.assertRaises(FileNotFoundError):dataset.UnlabeledCompetitionDataset(d/'sample.csv',zpath,None,d/'bad.csv')
            with zipfile.ZipFile(d/'duplicate.zip','w') as z:z.writestr('a/x.png',b'1');z.writestr('b/x.png',b'2')
            with self.assertRaises(ValueError):dataset._archive_index(str(d/'duplicate.zip'))

    def test_csv_exact_schema_and_no_overwrite(self):
        ids=[str(i) for i in range(2000)];labels=[i%10 for i in range(2000)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'submission.csv';core.write_csv(p,ids,labels,ids);original=p.read_bytes()
            with self.assertRaises(FileExistsError):core.write_csv(p,ids,labels,ids)
            self.assertEqual(p.read_bytes(),original)
            for bad in (ids[:-1],list(reversed(ids)),ids[:-1]+[ids[0]]):
                with self.assertRaises(ValueError):core.write_csv(p,bad,labels,ids)
            with self.assertRaises(ValueError):core.write_csv(p,ids,[10]*2000,ids)
            self.assertEqual(len(dataset.read_csv(p)),2000)

    def test_audit_gate_rejects_changed_source_before_training(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audit.json';core.json_new(p,dict(status='passed',sources={},recipe=run.recipe(self.cfg)))
            with self.assertRaisesRegex(ValueError,'audit'):run.train(SimpleNamespace(config=core.ROOT/'configs/development.json',audit=p))

if __name__=='__main__':unittest.main(verbosity=2)
