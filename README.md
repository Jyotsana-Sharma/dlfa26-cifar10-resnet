# NYU PreAct Baseline Recovery V4

A Pre-Activation ResNet trained with SLA+SD loss on CIFAR-10, under a 5 million parameter constraint.

**90.7% public leaderboard accuracy (rank 23rd) · 87.4% private accuracy (rank 29th)**

---

## Model

| Item | Value |
|---|---|
| Architecture | Pre-Activation ResNet, widths 64/128/256, depths 4/4/3 |
| Parameters | 4,706,674 (≤ 5M limit) |
| Loss | SLA+SD + cross-entropy with label smoothing 0.1 |
| Augmentation | TrivialAugmentWide, RandomCrop(32, pad=4), HorizontalFlip, Cutout 12×12 |
| Optimizer | Nesterov SGD, lr 0.1, cosine decay, 5-epoch warmup |
| Training | 200 epochs, batch 128, FP16 AMP, EMA 0.999 |
| Inference | 6-view TTA (original + flip + ±1px shifts) |

The model uses **SLA (Self-Supervised Label Augmentation)** — each image is rotated 0°/90°/180°/270° and trained with a joint 40-way (class × rotation) head — combined with **SD (Self-Distillation)** from the rotation-aggregate head back to the main classifier. This was the single most effective technique, adding +0.74 holdout accuracy over the plain PreActResNet baseline.

---

## Setup

```bash
git clone https://github.com/Jyotsana-Sharma/dlfa26-cifar10-resnet.git
cd dlfa26-cifar10-resnet
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.8+, and a CUDA GPU for training.

---

## Running Locally

Arrange course data as follows before running any command:

```
data/
  train_labels.csv
  train_images/        (directory or train_images.zip)
  test.csv
  test_images/
  sample_submission.csv
```

**1. Verify installation — no GPU or data required:**
```bash
python -m src.verify
```
Checks parameter count (4,706,674), forward pass, SLA+SD loss, and augmentation pipeline on CPU.

**2. Train on the 45k/5k development split:**
```bash
python -m src.train \
  --course-root data/ \
  --config configs/development.json \
  --output runs/dev_run
```

**3. Resume a stopped or interrupted run:**
```bash
python -m src.train \
  --course-root data/ \
  --output runs/dev_run \
  --resume
```

**4. Generate a submission CSV from a trained checkpoint:**
```bash
python -m src.predict \
  --course-root data/ \
  --checkpoint runs/dev_run/best_raw.pt \
  --checkpoint-sha256 <sha256-of-checkpoint> \
  --output submissions/submission.csv
```
This writes `submission.csv` and a provenance manifest alongside it. Run `sha256sum runs/dev_run/best_raw.pt` to get the checkpoint hash.

---

## Running on NYU HPC

Set your Python environment and use `run.sh` as the entry point:

```bash
export PYTHON_BIN="$HOME/resnet-env/bin/python"

bash preact_recovery_v4/run.sh setup          # copy V3 source into isolated backend
bash preact_recovery_v4/run.sh audit          # verify training/test data before training
bash preact_recovery_v4/run.sh overnight      # train 45k split, auto-resumes across sessions
bash preact_recovery_v4/run.sh overnight-full # train all 50k, exports final submission.csv
bash preact_recovery_v4/run.sh status         # show training progress
```

`overnight` and `overnight-full` are safe to re-run — they automatically resume from the last saved checkpoint if training was interrupted by an allocation timeout.

---

## Repository Layout

```
src/
  model.py          PreActResNet + SLA heads (architecture)
  train.py          SLA+SD training loop with EMA, resume, checkpointing
  augmentation.py   TrivialAugment + Cutout pipeline
  data.py           Image loading, stratified splits, ZIP support
  predict.py        Standalone inference → submission CSV
  verify.py         CPU smoke test (no data or GPU needed)
configs/
  development.json          V4 training recipe
  source_provenance.json    V3 source SHA-256 hashes
docs/reports/
  figures/          Training curve, experiment comparison, accuracy progression charts
tests/
  test_local.py     Unit tests for bootstrap and CSV validation
bootstrap.py        NYU HPC: copies V3 source into isolated backend/
run.sh              NYU HPC orchestration entry point
requirements.txt    Python dependencies
```

---

## Results

| Model | Holdout (45k) | Kaggle Public | Kaggle Private |
|---|---|---|---|
| AirResNet baseline | 95.58% | 85.0% | — |
| + TrivialAugment, 120 ep | 96.20% | 86.0% | — |
| PreActResNet, 200 ep | 96.82% | 88.4% | — |
| **PreActResNet + SLA+SD, all 50k** | **97.56%** | **90.7% (rank 23)** | **87.4% (rank 29)** |

---

## References

- He et al. (2016). Identity Mappings in Deep Residual Networks. ECCV. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
- Lee, Hwang & Shin (2020). Self-supervised Label Augmentation via Input Transformations. ICML. [arXiv:1910.05872](https://arxiv.org/abs/1910.05872)
- Müller & Hutter (2021). TrivialAugment. ICCV. [arXiv:2103.10158](https://arxiv.org/abs/2103.10158)
- DeVries & Taylor (2017). Improved Regularization with Cutout. [arXiv:1708.04552](https://arxiv.org/abs/1708.04552)
