"""Standalone prediction: load a checkpoint and write a submission CSV.

This command writes predictions; it never uploads to Kaggle.
Checkpoint identity is verified by SHA-256 before inference.
"""
import argparse
import json
from pathlib import Path

import torch

from .augmentation import MEAN, STD
from .data import UnlabeledCompetitionDataset, read_csv
from .train import (
    calibrate, cpu_state, digest, environment, make_model,
    predict_policies, source_hashes, _calibration_loader,
    load_training,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--variant', choices=['selected', 'raw', 'ema', 'flip'], default='selected')
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f'Choose a fresh output path; existing submissions are preserved: {args.output}')
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Checkpoint SHA-256 mismatch — wrong file or corrupted download')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no GPU is available; use --device cpu')

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck.get('config')
    if cfg is None:
        raise ValueError('Checkpoint does not contain a config entry')

    # Determine weights and model kind from checkpoint.
    if args.variant == 'selected':
        policy_path = Path(args.checkpoint).parent / 'selection.json'
        if policy_path.exists():
            sel = json.loads(policy_path.read_text())['selected']
            weights = ck.get(sel['model_kind']) or ck.get('model')
            model_kind = sel['model_kind']
            tta = sel['tta']
            head = sel['head']
        else:
            weights = ck.get('model') or ck.get('raw')
            model_kind = 'raw'
            tta = 'flip'
            head = 'combined'
    else:
        weights = ck.get(args.variant) or ck.get('model')
        model_kind = args.variant if args.variant in ('raw', 'ema') else 'raw'
        tta = 'flip' if args.variant == 'flip' else 'none'
        head = 'combined'

    if weights is None:
        raise ValueError(f'No weights found for variant "{args.variant}" in checkpoint')

    net = make_model().to(args.device)
    net.load_state_dict(weights, strict=True)

    if model_kind == 'ema':
        images = args.course_root / 'train_images.zip'
        if not images.exists():
            images = args.course_root / 'train_images'
        labels = args.course_root / 'train_labels.csv'
        samples, mapping, tr, _ = load_training(images, labels, cfg)
        print('Calibrating EMA BN using training images only', flush=True)
        calibrate(net, _calibration_loader(samples, mapping, tr, cfg, args.device), args.device)

    if not all(torch.isfinite(v).all() for v in net.state_dict().values()):
        raise ValueError('Non-finite values in checkpoint weights')

    sample_csv = args.course_root / 'sample_submission.csv'
    test_csv = args.course_root / 'test.csv'
    test_images = args.course_root / 'test_images'

    sample_rows = read_csv(sample_csv)
    expected_ids = [r['id'] for r in sample_rows]

    from .augmentation import transform
    from torch.utils.data import DataLoader
    test_ds = UnlabeledCompetitionDataset(sample_csv, test_images, transform(), test_csv)

    ids = []
    labels = []
    net.eval()
    for x, bids in DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0):
        preds = predict_policies(net, x.to(args.device), (tta,))
        labels.extend(int(i) for i in preds[tta, head].argmax(1).cpu().tolist())
        ids.extend(bids)
        print(f'Predicted {len(ids)}/2000', flush=True)

    if ids != expected_ids:
        raise ValueError('Prediction ID order differs from sample_submission.csv')
    if any(not 0 <= lb <= 9 for lb in labels):
        raise ValueError('Predicted label out of range 0-9')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as f:
        import csv as csv_mod
        w = csv_mod.writer(f)
        w.writerow(['id', 'label'])
        w.writerows(zip(ids, labels))

    manifest = dict(
        rows=len(ids),
        parameters=sum(p.numel() for p in net.parameters()),
        checkpoint_sha256=args.checkpoint_sha256,
        csv_sha256=digest(args.output),
        device=args.device,
        variant=args.variant,
        tta=tta,
        head=head,
        precision='FP16-trained / FP32-inference',
        submitted=False,
        source_sha256=source_hashes(),
        environment=environment(),
    )
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('DOWNLOAD THIS FILE:', args.output.resolve(), flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
