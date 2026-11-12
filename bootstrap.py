#!/usr/bin/env python3
"""Create an isolated, exact-source copy of the user's locally installed V3 PreAct package.

This package intentionally does not pretend to include the user's proprietary
original 0.906 model checkpoint. It reuses the previously audited V3 model source.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
REQUIRED = ('models.py', 'core.py', 'dataset.py', 'run.py', 'nyu.sh', 'overnight.sh', 'configs/development.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposed_config(source):
    c = json.loads(Path(source).read_text())
    required = ('name', 'epochs', 'batch_size', 'lr', 'min_lr', 'warmup_epochs',
                'momentum', 'nesterov', 'weight_decay', 'label_smoothing',
                'ema_decay', 'seed', 'validation_size', 'augmentation',
                'cutout', 'precision', 'workers', 'calibration_images')
    absent = [key for key in required if key not in c]
    if absent:
        raise RuntimeError('Unexpected V3 config schema; missing: ' + ', '.join(absent))
    c.update(name='preact_baseline_recovery_v4', epochs=200,
             batch_size=128, lr=0.1, min_lr=0.0,
             warmup_epochs=5, momentum=0.9, nesterov=True,
             weight_decay=0.0005, label_smoothing=0.1,
             ema_decay=0.999, seed=0, validation_size=5000,
             augmentation='trivial', cutout=12, precision='fp16',
             workers=2, calibration_images=10000)
    return c


def install(project, v3=None, destination=None):
    project = Path(project).expanduser().resolve()
    source = Path(v3).expanduser().resolve() if v3 else project / 'nyu_preact_v3'
    target = Path(destination).expanduser().resolve() if destination else BASE / 'backend'
    for name in REQUIRED:
        if not (source / name).is_file():
            raise FileNotFoundError(f'V3 source required: {source/name}. Install your original NYU_PreAct_SLA_V3.zip in the project first.')
    if target.exists():
        expected = target/'v4_bootstrap_manifest.json'
        if not expected.is_file():
            raise RuntimeError(f'Existing backend not managed by this package: {target}')
        manifest=json.loads(expected.read_text())
        current={str(p.relative_to(target)):digest(p) for p in target.glob('*.py')}
        if current != manifest['python_sources']:
            raise RuntimeError('Previously bootstrapped Python source has changed. Preserve it; do not overwrite or train.')
        if json.loads((target/'configs/development.json').read_text()) != manifest['config']:
            raise RuntimeError('Previously bootstrapped recipe changed. Preserve checkpoints.')
        print('Already bootstrapped and verified:', target)
        return target
    # Keep the original source unmodified. Only copy execution files and tests.
    (target/'configs').mkdir(parents=True)
    (target/'tests').mkdir(parents=True)
    for name in ('models.py','core.py','dataset.py','run.py','nyu.sh','overnight.sh'):
        shutil.copy2(source/name,target/name)
    for test in (source/'tests').glob('test_*.py') if (source/'tests').exists() else ():
        shutil.copy2(test,target/'tests'/test.name)
    c=proposed_config(source/'configs/development.json')
    (target/'configs/development.json').write_text(json.dumps(c,indent=2)+'\n')
    hashes={str(p.relative_to(target)):digest(p) for p in sorted(target.glob('*.py'))}
    manifest={'source_v3':str(source),'v3_source_sha256':{name:digest(source/name) for name in ('models.py','core.py','dataset.py','run.py')},
              'python_sources':hashes, 'config':c,
              'notes':'V4 reuses V3 architecture/loss implementation verbatim, but changes baseline recipe. This is NOT the original 0.906 checkpoint.'}
    (target/'v4_bootstrap_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('NEW isolated backend:',target)
    print('Architecture and SLA implementation copied unchanged from V3.')
    print('Baseline-inspired recipe: 200 epochs, batch 128, lr 0.1, seed 0.')
    return target


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True)
    p.add_argument('--v3')
    p.add_argument('--destination')
    a=p.parse_args()
    install(a.project,a.v3,a.destination)

if __name__=='__main__':main()
