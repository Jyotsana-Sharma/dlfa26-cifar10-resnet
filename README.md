# NYU PreAct Baseline Recovery V4

A Pre-Activation ResNet trained with SLA+SD loss on CIFAR-10, under a 5M parameter constraint.
**90.7% public leaderboard accuracy (rank 23rd) · 87.4% private accuracy (rank 29th)**

## Model

| Item | Value |
|---|---|
| Architecture | PreActResNet, widths 64/128/256, depths 4/4/3 |
| Parameters | 4,706,674 |
| Loss | SLA+SD + cross-entropy, label smoothing 0.1 |
| Augmentation | TrivialAugmentWide, RandomCrop, HorizontalFlip, Cutout 12 |
| Training | 200 epochs, batch 128, lr 0.1, cosine decay, FP16, EMA 0.999 |
| Inference | 6-view TTA |

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.8+, CUDA GPU for training.

## Run (local — standalone src/ package)

**Verify installation (no GPU or data needed):**
```bash
python -m src.verify
```

**Train on 45k/5k development split:**
```bash
python -m src.train \
  --course-root /path/to/data \
  --config configs/development.json \
  --output runs/dev_run
```

**Resume a stopped run:**
```bash
python -m src.train --course-root /path/to/data --output runs/dev_run --resume
```

**Generate submission CSV from a checkpoint:**
```bash
python -m src.predict \
  --course-root /path/to/data \
  --checkpoint runs/dev_run/best_raw.pt \
  --checkpoint-sha256 <sha256> \
  --output submissions/submission.csv
```

Expected data layout:
```
data/
  train_labels.csv
  train_images/   (or train_images.zip)
  test.csv
  test_images/
  sample_submission.csv
```

## Run (NYU HPC — run.sh orchestrator)

```bash
export PYTHON_BIN="$HOME/resnet-env/bin/python"
bash preact_recovery_v4/run.sh setup       # bootstrap V3 backend
bash preact_recovery_v4/run.sh audit       # verify data before training
bash preact_recovery_v4/run.sh overnight   # train 45k dev split, auto-resume
bash preact_recovery_v4/run.sh overnight-full  # train all 50k, export submission.csv
bash preact_recovery_v4/run.sh status      # check progress
```

**Compare two submission CSVs:**
```bash
CSV_A=/path/a.csv CSV_B=/path/b.csv bash preact_recovery_v4/run.sh compare-csv
```

## Repository Layout

```
src/              Self-contained ML package
  model.py        PreActResNet architecture
  train.py        SLA+SD training loop
  augmentation.py TrivialAugment + Cutout pipeline
  data.py         Data loading and splits
  predict.py      Submission CSV generation
  verify.py       Local CPU smoke test
configs/          Training recipes and source provenance
docs/reports/     Project report and figures
tests/            Unit tests
bootstrap.py      NYU HPC backend setup
run.sh            NYU HPC orchestration entry point
```
