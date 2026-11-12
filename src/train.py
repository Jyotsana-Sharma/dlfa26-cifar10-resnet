"""PreAct SLA+SD training entry point. Never exports test-set predictions.

Architecture and loss are unchanged from nyu_preact_v3. Config changes: 200 epochs,
seed 0, min_lr 0.0, batch 128/lr 0.1 (memory-safe scale-down from historic 512/0.4).
Historical executed runs remain the exact record; this entry point has not been used
for a new 200-epoch reproduction on NYU GPU.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .augmentation import transform, MEAN, STD
from .data import (
    CompetitionDataset, UnlabeledCompetitionDataset,
    class_mapping, dataset_fingerprint, load_labeled_images,
    read_csv, stratified_split, validate_training_distribution,
)
from .model import PreActResNet, count_params

EXPECTED_PARAMETERS = 4_706_674


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_get():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def rng_set(r):
    random.setstate(r['python'])
    np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch'])
    if r['cuda'] is not None:
        torch.cuda.set_rng_state_all(r['cuda'])


def worker_seed(_):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: digest(p) for p in sorted(root.glob('*.py'))}


def atomic_save(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model():
    net = PreActResNet((64, 128, 256), (4, 4, 3), ssl='sla')
    if count_params(net) != EXPECTED_PARAMETERS:
        raise ValueError(f'Unexpected parameter count: {count_params(net)}')
    return net


# ---------------------------------------------------------------------------
# SLA + SD loss (four-rotation self-supervised + self-distillation)
# ---------------------------------------------------------------------------

def rotate4(x):
    return torch.stack([torch.rot90(x, k, (2, 3)) for k in range(4)], 1).flatten(0, 1)


def heads(net, x):
    z = net.features(rotate4(x))
    j = (net.joint_head(z) * net.scale).float()
    p = (net.head(z[::4]) * net.scale).float()
    a = torch.stack([j.view(len(x), 4, 40)[:, k, k::4] for k in range(4)]).mean(0)
    return p, a, j


def sla_loss(net, x, y, smoothing):
    p, a, j = heads(net, x)
    target = (y[:, None] * 4 + torch.arange(4, device=y.device)).flatten()
    joint = F.cross_entropy(j, target)
    plain = F.cross_entropy(p, y, label_smoothing=smoothing)
    sd = F.kl_div(F.log_softmax(p, 1), F.softmax(a.detach(), 1), reduction='batchmean')
    return joint + plain + sd, p


# ---------------------------------------------------------------------------
# EMA and learning rate schedule
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(ema, net, decay):
    for e, p in zip(ema.parameters(), net.parameters()):
        e.lerp_(p, 1 - decay)
    for e, p in zip(ema.buffers(), net.buffers()):
        e.copy_(p)


def learning_rate(step, total, warm, peak, floor):
    if step < warm:
        return peak * (step + 1) / warm
    return floor + (peak - floor) * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm - 1)))


def make_optimizer(net, cfg):
    return torch.optim.SGD(
        [
            dict(params=[p for p in net.parameters() if p.ndim > 1], weight_decay=cfg['weight_decay']),
            dict(params=[p for p in net.parameters() if p.ndim <= 1], weight_decay=0),
        ],
        lr=cfg['lr'], momentum=cfg['momentum'], nesterov=cfg['nesterov'],
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_training(images, labels, cfg):
    samples = load_labeled_images(labels, images)
    mapping, _ = class_mapping(samples)
    validate_training_distribution(samples)
    tr, va = stratified_split(samples, 5000, cfg['seed'])
    if cfg['validation_size'] == 0:
        tr = list(range(len(samples)))
        va = []
    elif cfg['validation_size'] != 5000:
        raise ValueError('Only fixed 45K/5K or full50K permitted')
    return samples, mapping, tr, va


def loader(samples, mapping, indices, cfg, epoch=0, training=False, batch_size=None):
    g = torch.Generator().manual_seed(cfg['seed'] + epoch * 1009 + (0 if training else 17))
    return DataLoader(
        CompetitionDataset(samples, mapping, transform(training, cfg), indices),
        batch_size=batch_size or cfg['batch_size'],
        shuffle=training,
        num_workers=cfg['workers'],
        generator=g,
        worker_init_fn=worker_seed,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_policies(net, x, modes):
    if not modes or any(m not in ('none', 'flip', 'original6') for m in modes):
        raise ValueError('Unknown inference policy')
    net.eval()
    cache = {}
    specs = [(0, 0, False)]
    if any(m != 'none' for m in modes):
        specs.append((0, 0, True))
    if 'original6' in modes:
        specs.extend([(s, s, f) for s in (-1, 1) for f in (False, True)])
    for y, z, flip in specs:
        view = x if y == 0 and z == 0 else F.pad(x, (1,) * 4, mode='reflect')[:, :, 1 + y:33 + y, 1 + z:33 + z]
        p, a, _ = heads(net, view.flip(-1) if flip else view)
        cache[y, z, flip] = (p, a)
    result = {}
    for mode in modes:
        spec = [((0, 0, False), 1.)] if mode == 'none' else [((0, 0, f), .5) for f in (False, True)]
        if mode == 'original6':
            spec = [((0, 0, f), .25) for f in (False, True)] + [((s, s, f), .125) for s in (-1, 1) for f in (False, True)]
        p = sum(cache[k][0] * w for k, w in spec).softmax(1)
        a = sum(cache[k][1] * w for k, w in spec).softmax(1)
        for key, value in [('plain', p), ('aggregate', a), ('combined', (p + a) / 2)]:
            result[mode, key] = value
    return result


@torch.inference_mode()
def evaluate(net, dl, device, modes=('none',), progress=False):
    hits = {}
    n = 0
    for batch, (x, y) in enumerate(dl):
        preds = predict_policies(net, x.to(device), modes)
        n += len(y)
        for (mode, head), p in preds.items():
            key = f'{mode}/{head}'
            hits[key] = hits.get(key, 0) + int((p.argmax(1).cpu() == y).sum())
        if progress and (batch + 1) % 20 == 0:
            print(f'Validation: {n} images evaluated', flush=True)
    if not n:
        raise ValueError('Cannot evaluate full-data run on its training data')
    return {k: v / n for k, v in hits.items()}


@torch.no_grad()
def calibrate(net, dl, device):
    saved = {}
    net.eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            saved[m] = m.momentum
            m.reset_running_stats()
            m.momentum = None
            m.train()
    for batch, (x, _) in enumerate(dl):
        net.features(rotate4(x.to(device)))
        if (batch + 1) % 20 == 0:
            print(f'EMA normalization: {batch + 1}/{len(dl)} training batches', flush=True)
    for m, momentum in saved.items():
        m.momentum = momentum
    net.eval()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def read_config(path):
    c = json.loads(Path(path).read_text())
    if c['precision'] != 'fp16' or c['augmentation'] != 'trivial':
        raise ValueError('This recipe uses FP16 and TrivialAugment')
    if c['batch_size'] < 1 or c['epochs'] <= c['warmup_epochs'] or not 0 <= c['min_lr'] <= c['lr']:
        raise ValueError('Invalid schedule')
    if (not 0 <= c['label_smoothing'] < 1 or not 0 < c['ema_decay'] < 1
            or not 0 <= c['cutout'] <= 32 or c['workers'] < 0
            or not 1 <= c['calibration_images'] <= 45000):
        raise ValueError('Invalid augmentation, EMA, or loader settings')
    return c


def recipe(c):
    return {k: v for k, v in c.items() if k not in ('name', 'validation_size')}


def environment():
    import torchvision
    import platform
    import PIL
    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        numpy=np.__version__,
        pillow=PIL.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _calibration_loader(samples, mapping, tr, cfg, device):
    indices = np.random.default_rng(cfg['seed'] + 77).permutation(tr)[:cfg['calibration_images']].tolist()
    lc = cfg if device == 'cuda' else dict(cfg, workers=0)
    return loader(samples, mapping, indices, lc)


def _train_loop(args, cfg, run, samples, mapping, tr, va, fp, ck, test_signature):
    seed_all(cfg['seed'])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    net = make_model().to(args.device)
    ema = copy.deepcopy(net).eval()
    ema.requires_grad_(False)
    opt = make_optimizer(net, cfg)
    scaler = torch.amp.GradScaler('cuda')
    start = 0
    step = 0
    history = []
    best = {}
    refs = {}
    if ck:
        net.load_state_dict(ck['raw'])
        ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        rng_set(ck['rng'])
        start = ck['completed_epochs']
        step = ck['step']
        history = ck['history']
        best = ck['best']
        refs = ck['best_files']
        if ck['environment'] != environment():
            print('NOTE: runtime differs from saved environment; bitwise reproducibility is not guaranteed.', flush=True)
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    steps = math.ceil(len(tr) / cfg['batch_size'])

    def base():
        return dict(
            config=cfg, sources=source_hashes(), fingerprint=fp,
            train_indices=tr, validation_indices=va,
            environment=environment(), trained_from_scratch=True,
            test_signature=test_signature,
        )

    def state(epoch):
        return dict(
            **base(), raw=cpu_state(net), ema=cpu_state(ema),
            optimizer=opt.state_dict(), scaler=scaler.state_dict(),
            rng=rng_get(), step=step, completed_epochs=epoch,
            history=history, best=best, best_files=refs,
        )

    if ck is None:
        json_new(run / 'config.json', cfg)
        atomic_save(run / 'last.pt', state(0))

    print(json.dumps(dict(
        status='training', completed_epochs=start, total_epochs=cfg['epochs'],
        parameters=count_params(net),
    )), flush=True)

    for epoch in range(start, cfg['epochs']):
        if stop[0]:
            print('Stopping: interrupted.', flush=True)
            break
        t = time.monotonic()
        net.train()
        total_loss = 0.
        correct = 0
        n = 0
        skips = 0
        dl = loader(samples, mapping, tr, cfg, epoch, True)
        for batch, (x, y) in enumerate(dl):
            if stop[0]:
                print('Incomplete epoch discarded; resume from last committed epoch.', flush=True)
                return
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            lr = learning_rate(step, steps * cfg['epochs'], steps * cfg['warmup_epochs'], cfg['lr'], cfg['min_lr'])
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, p = sla_loss(net, x, y, cfg['label_smoothing'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; previous completed checkpoint retained')
            old = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old:
                ema_update(ema, net, cfg['ema_decay'])
            else:
                skips += 1
            step += 1
            n += len(y)
            correct += int((p.argmax(1) == y).sum())
            total_loss += float(loss.detach()) * len(y)
            if (batch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1}/{cfg["epochs"]} batch {batch + 1}/{steps}', flush=True)

        metrics = {}
        if va:
            for kind, m in [('raw', net), ('ema', ema)]:
                print(f'Epoch {epoch + 1}: validating {kind}', flush=True)
                metrics[kind] = evaluate(m, loader(samples, mapping, va, cfg), args.device)

        record = dict(
            epoch=epoch + 1, train_loss=total_loss / n, train_accuracy=correct / n,
            validation=metrics, lr=lr, skipped_updates=skips, seconds=time.monotonic() - t,
        )
        record['estimated_training_hours_remaining'] = (cfg['epochs'] - epoch - 1) * record['seconds'] / 3600
        history.append(record)

        for kind, m in [('raw', net), ('ema', ema)]:
            if va:
                for head in ('plain', 'combined'):
                    key = f'{kind}_{head}'
                    acc = metrics[kind]['none/' + head]
                    if acc > best.get(key, -1):
                        name = f'{key}_e{epoch + 1:03d}_{uuid.uuid4().hex[:8]}.pt'
                        atomic_save(run / name, dict(**base(), model=cpu_state(m), model_kind=kind, completed_epochs=epoch + 1))
                        best[key] = acc
                        refs[key] = name

        atomic_save(run / 'last.pt', state(epoch + 1))
        atomic_json(run / 'progress.json', dict(completed_epochs=epoch + 1, history=history, best=best, best_files=refs))
        print(json.dumps(record), flush=True)

    if len(history) == cfg['epochs']:
        print(f'TRAINING COMPLETE: {len(history)}/{cfg["epochs"]} epochs saved.', flush=True)
    else:
        print(f'Committed {len(history)}/{cfg["epochs"]} epochs. Resume continues this run in the next allocation.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/development.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no GPU is allocated; full training requires a GPU.')
    cfg = read_config(args.config)
    run = args.output

    images = args.course_root / 'train_images.zip'
    if not images.exists():
        images = args.course_root / 'train_images'
    labels = args.course_root / 'train_labels.csv'

    samples, mapping, tr, va = load_training(images, labels, cfg)
    fp = dataset_fingerprint(samples)

    ck = None
    if args.resume:
        resume_path = run / 'last.pt'
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        ck = torch.load(resume_path, map_location='cpu', weights_only=False)
        if ck['config'] != cfg or ck['sources'] != source_hashes() or ck['fingerprint'] != fp or ck['train_indices'] != tr or ck['validation_indices'] != va:
            raise ValueError('Resume config/data/code/split mismatch')
    else:
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f'Use a new run directory or --resume: {run}')
        run.mkdir(parents=True, exist_ok=True)

    lock = run / 'training.lock'
    with lock.open('x') as f:
        f.write(f'{os.uname().nodename} pid={os.getpid()}\n')
    try:
        _train_loop(args, cfg, run, samples, mapping, tr, va, fp, ck, test_signature=None)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
