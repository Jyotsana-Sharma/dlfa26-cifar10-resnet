"""Verify the extracted src/ implementation without requiring GPU or trained checkpoints.

Checks:
  1. Model parameter count matches the expected 4,706,674.
  2. A CPU forward pass with random data produces finite logits.
  3. SLA+SD loss is finite and backward is clean.
  4. Augmentation pipeline runs without error on a synthetic image.
  5. Config keys in configs/development.json are complete.

Does not require course data, trained weights, or a CUDA device.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def main():
    torch.set_num_threads(2)
    checks = []

    # 1. Config completeness
    cfg_path = ROOT / 'configs/development.json'
    cfg = json.loads(cfg_path.read_text())
    required_keys = (
        'name', 'epochs', 'batch_size', 'lr', 'min_lr', 'warmup_epochs',
        'momentum', 'nesterov', 'weight_decay', 'label_smoothing', 'ema_decay',
        'seed', 'validation_size', 'augmentation', 'cutout', 'precision',
        'workers', 'calibration_images',
    )
    missing = [k for k in required_keys if k not in cfg]
    assert not missing, f'Config missing keys: {missing}'
    checks.append(dict(check='config_complete', keys=len(cfg), status='passed'))

    # 2. Model parameter count
    from src.model import PreActResNet, count_params
    seed(0)
    net = PreActResNet((64, 128, 256), (4, 4, 3), ssl='sla')
    params = count_params(net)
    assert params == 4_706_674, f'Parameter count mismatch: {params}'
    checks.append(dict(check='parameter_count', parameters=params, status='passed'))

    # 3. CPU forward pass
    from src.train import sla_loss, make_model
    seed(42)
    net = make_model()
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        logits = net(x)
    assert torch.isfinite(logits).all(), 'Non-finite logits in forward pass'
    checks.append(dict(check='forward_pass_finite', batch=4, status='passed'))

    # 4. SLA+SD loss and backward
    seed(42)
    net = make_model()
    net.zero_grad(set_to_none=True)
    loss, p = sla_loss(net, x, y, cfg['label_smoothing'])
    assert torch.isfinite(loss), f'Non-finite loss: {loss.item()}'
    loss.backward()
    grad_ok = all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in net.parameters()
    )
    assert grad_ok, 'Non-finite gradients after backward'
    checks.append(dict(check='sla_loss_and_backward', loss=round(float(loss.detach()), 6), status='passed'))

    # 5. Augmentation pipeline on a synthetic 32x32 PIL image
    from src.augmentation import transform
    seed(0)
    aug = transform(training=True, cfg=cfg)
    img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    t = aug(img)
    assert t.shape == (3, 32, 32) and torch.isfinite(t).all(), 'Augmentation output malformed'
    checks.append(dict(check='augmentation_pipeline', output_shape=list(t.shape), status='passed'))

    result = dict(
        status='passed',
        torch=torch.__version__,
        device='cpu',
        checks=checks,
        scope='CPU-only: parameter count, forward pass, loss+backward, augmentation. No CUDA training or checkpoint parity check.',
    )
    out = ROOT / 'docs/reports/verify_local.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
