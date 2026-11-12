# Agent instructions

## Purpose and entry points

This is a NYU Deep Learning competition recovery package targeting a PreAct ResNet
baseline. It reuses the nyu_preact_v3 architecture and SLA+SD loss verbatim with a
revised recipe (200 epochs, seed 0, batch 128/lr 0.1).

Read `README.md`, the relevant `src/` modules and `configs/` before making changes.
`src/train.py` is the main training entry point. `src/predict.py` generates submission
CSVs. `src/verify.py` runs local CPU verification without data or GPU.

`bootstrap.py` and `run.sh` are NYU HPC orchestration wrappers and are separate from
the self-contained `src/` package.

## Preservation and scientific accuracy

- Preserve course images, labels, existing checkpoints, submission CSVs and run artifacts.
  Never quietly edit labels or remove difficult examples.
- Keep architecture and loss implementation (`src/model.py`, `src/train.py`) unchanged
  from V3 unless explicitly requested. Document any departures from the V3 recipe.
- Keep development, public and private scores distinct. Development accuracy and
  Kaggle public scores are not independent. Do not invent missing measurements.
- The deployed model must be one residual CNN with at most 5,000,000 trainable
  parameters including its head. No external datasets or pretrained weights.
- Hidden images remain prediction-only; no label recovery or hidden-data tuning.

## Validation and operation

Run from the repository root in a PyTorch-capable environment:

- `python -m src.verify` — CPU-only smoke test, no data or GPU required
- `python -m src.train --help`
- `python -m src.predict --help`

For NYU HPC runs use `run.sh` as the entry point. Training and prediction are separate
commands. Do not launch training, paid cloud resources, or Kaggle submissions without
an explicit request.

## Public and local files

Keep secrets, course data, weights and temporary artifacts out of Git. `backend/`,
`experiments/`, `data/`, `models/` and `*.pth`/`*.pt` files are ignored local storage.
The public package must be understandable without those local directories.
