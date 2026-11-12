import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap,probe

class TestLocal(unittest.TestCase):
    def test_recipe(self):
        c={k:1 for k in ('name','epochs','batch_size','lr','min_lr','warmup_epochs','momentum','nesterov','weight_decay','label_smoothing','ema_decay','seed','validation_size','augmentation','cutout','precision','workers','calibration_images')}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';p.write_text(json.dumps(c));x=bootstrap.proposed_config(p)
            self.assertEqual(x['epochs'],200);self.assertEqual(x['batch_size'],128)
            self.assertEqual(x['validation_size'],5000)
            self.assertEqual(x['seed'],0)
    def test_bootstrap_copy_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v3=root/'nyu_preact_v3';(v3/'configs').mkdir(parents=True)
            for name in bootstrap.REQUIRED:
                p=v3/name;p.parent.mkdir(parents=True,exist_ok=True)
                if name.endswith('development.json'):
                    x={k:1 for k in ('name','epochs','batch_size','lr','min_lr','warmup_epochs','momentum','nesterov','weight_decay','label_smoothing','ema_decay','seed','validation_size','augmentation','cutout','precision','workers','calibration_images')}
                    p.write_text(json.dumps(x))
                else:p.write_text('# placeholder\n')
            target=root/'installed';bootstrap.install(root, destination=target)
            bootstrap.install(root,destination=target)
            self.assertEqual((target/'models.py').read_text(),'# placeholder\n')
            (target/'models.py').write_text('# modified')
            with self.assertRaises(RuntimeError):bootstrap.install(root,destination=target)
    def test_compare_csv(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def write(p, rows):
                with p.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(('id','label'));w.writerows(rows)
            ids=[f'{i:04d}' for i in range(2000)]
            a=root/'a.csv';b=root/'b.csv';sample=root/'sample.csv'
            write(a,zip(ids,[0]*2000));write(b,zip(ids,[1]*2000));write(sample,zip(ids,[0]*2000))
            probe.compare(a,b,sample)
            write(b,zip(reversed(ids),[1]*2000))
            with self.assertRaises(ValueError):probe.compare(a,b,sample)

if __name__=='__main__':unittest.main(verbosity=2)
