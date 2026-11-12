# PreAct SLA V3 — NYU training and submission

This is a new, isolated experiment based on the supplied PreAct model associated with the reported 0.906 score. It is a candidate for improvement, **not a measured 92% model**. No new GPU training or Kaggle submission has been performed while preparing this package.

## 1. Install this package in your existing HPC project

Upload `NYU_PreAct_SLA_V3.zip` to your HPC home folder using Jupyter's upload button. In a Jupyter terminal:

```bash
cd "/scratch/$USER/cifar10-resnet"
unzip -n "$HOME/NYU_PreAct_SLA_V3.zip" -d "$PWD"
bash nyu_preact_v3/nyu.sh help
```

The package uses your existing `$HOME/resnet-env/bin/python`. It does not reinstall PyTorch or touch V1/V2. The existing NYU environment shown in your logs uses PyTorch 2.8 with CUDA 12.8. Choose an A100 session with one GPU, at least 4 CPU cores and 16 GB host RAM. More session time reduces the number of resumes needed.

By default, data is read from your HPC home folder:

- `train_images.zip` (or `train_images/`) and `train_labels.csv`
- `test_images.zip` (or `test_images/`) and `test.csv`
- `sample_submission.csv`; if absent, the existing project file `v2/data/sample_submission.csv` is used

For an extracted Kaggle download elsewhere, set the enclosing folder before every command/session:

```bash
export DATA_DIR="/scratch/$USER/cifar10-resnet/data"
```

Use that setting only if the named data files/directories actually exist there. `TRAIN_IMAGES`, `TRAIN_LABELS`, `TEST_IMAGES`, `TEST_METADATA`, and `SAMPLE_SUBMISSION` also accept explicit path overrides. `PYTHON_BIN` overrides the Python executable. ZIPs are read directly, without extracting or downloading training data.

## 2. Recommended: run overnight on 45K with a fixed 5K holdout

Request your 10-hour GPU session, connect to Jupyter, then run this once:

```bash
cd "/scratch/$USER/cifar10-resnet"
bash nyu_preact_v3/nyu.sh overnight
```

This starts a background launcher and prints the log path and a `tail -f` command. It audits first, starts (or resumes) the 320-epoch development run, and compares validation policies after training finishes. You can close the terminal or let your laptop sleep; computation stays on the allocated NYU node. **Keep the NYU GPU session running.** Do not press its Cancel button. Starting another notebook or terminal does not extend the allocation.

There is **no artificial four-hour cutoff and no accuracy-based early stopping**. The default `SESSION_SECONDS=0` means no additional process time cap. When Slurm reports an allocation deadline, the trainer saves progress and stops just before that real deadline. The cluster can terminate processes when the allocation expires; no script can guarantee 320 epochs in a fixed allocation. If necessary, open a new GPU allocation and run the same `overnight` command to continue. Do not launch it a second time while the first copy is running.

The launcher prints `OVERNIGHT COMPLETE` only after all 320 epochs and development validation finish. An error or incomplete allocation stop is reported explicitly. Resume restores weights, EMA, optimizer momentum, scaler, random states and schedule position. A partially completed epoch is repeated from its preceding saved checkpoint.

For manual, foreground operation instead:

```bash
cd "/scratch/$USER/cifar10-resnet"
bash nyu_preact_v3/nyu.sh audit
bash nyu_preact_v3/nyu.sh train
```

`audit` runs the tests, decodes all images, verifies the split and CSV mapping, rejects duplicate training pixels and train/test exact overlap, and runs a real PreAct CPU forward/backward update. These temporary smoke weights are discarded. Training checks the audit's source, recipe and training-data hashes before starting. **Run the audit on HPC even though a local audit is supplied.**

Training writes only to:

```text
/scratch/<your user>/cifar10-resnet/experiments/preact_sla_ta320_v3/development/
```

The script prints batch progress every 50 batches, validation status, complete-epoch metrics and an estimated remaining training duration. Estimate runtime from those measured epochs; the old FP32 Wide98 time is not a benchmark for this model.

Only if you deliberately want an extra process limit, set one explicitly; otherwise leave it at zero. For example, a six-hour extra limit is:

```bash
SESSION_SECONDS=21600 bash nyu_preact_v3/nyu.sh train
```

For an existing run, including in a fresh GPU allocation, use:

```bash
cd "/scratch/$USER/cifar10-resnet"
bash nyu_preact_v3/nyu.sh resume
```

No time setting extends a cluster allocation. The script never changes the 320-epoch schedule to fit one session.

For a foreground run, Ctrl+C **once** requests a graceful stop. In the overnight log viewer, Ctrl+C stops only the viewer. If the job is killed abruptly, the previous committed `last.pt` remains usable; the last in-progress epoch may be lost. A leftover `training.lock` or `overnight_development.lock` / `overnight_full50k.lock` then requires checking that the recorded host/PID is no longer running before removing that lock. Do not delete checkpoints.

In a second terminal, view recent completed progress without interrupting training:

```bash
bash nyu_preact_v3/nyu.sh status
```

`train` refuses an existing run directory. `resume` requires its checkpoint and matching code/config/data. Keep this package unchanged for the life of the experiment. For a later different experiment, use a different package/study folder. No checkpoints, submissions or logs from earlier projects are overwritten. Within this new run, `last.pt` and `progress.json` are atomic rolling snapshots; best candidate checkpoints and per-launch logs have unique names.

## 3. Compare raw, EMA and inference policies on the A100

The overnight launcher performs this step automatically after 320/320. For manual operation:

```bash
bash nyu_preact_v3/nyu.sh select
```

This writes `experiments/preact_sla_ta320_v3/selection.json`. Review the per-candidate validation results. It compares raw/EMA separately, without averaging their predictions, and checks:

- No external TTA, horizontal flip TTA, and the original baseline's six-view TTA.
- Plain head, SLA aggregated head, and their 50/50 probability combination.

“None” refers to external TTA; the SLA aggregate itself uses its four trained rotations. Six-view TTA averages logits over the original/mirrored image and two reflected one-pixel diagonal translations, before softmax. The combined head averages the two resulting probability vectors. This matches the original baseline's inference formula; it is not V2's configurable probability-averaging sweep.

EMA parameters start as an exact model copy, update only after successful optimizer steps, and never average integer BatchNorm counters. Online EMA copies raw BatchNorm buffers. For selection and final EMA export, BatchNorm statistics are recalculated using a fixed 10K subset of **training** images only, with the same fixed batch size. No holdout or test normalization adaptation occurs.

Candidate epochs are retained by nonaugmented plain and combined validation accuracy. Selection checks those candidates and freezes a policy using only the 5K holdout. Ties prefer fewer external views, then combined head, then raw weights. The same holdout chooses both epochs and policies, so its reported accuracy is optimistic, not an independent test score. The historical 0.906 model trained on all 50K; it cannot serve as a leakage-free holdout comparator.

For a development-model export, if needed:

```bash
bash nyu_preact_v3/nyu.sh predict
```

Avoid consuming leaderboard submissions to choose among policies; inspect validation first. Nothing submits to Kaggle automatically.

## 4. Train the final model on all permitted 50K images

If development performance is satisfactory, start the full-data stage in a sufficiently long GPU allocation:

```bash
bash nyu_preact_v3/nyu.sh overnight-full
```

This prepares the full-data config, starts or resumes all 320 epochs, and generates the selected final `submission.csv` automatically on completion. It uses the same no-extra-cap default and background behavior. This is a **second training run**, so budget another allocation; the development run does not automatically start it before you review validation. To continue after allocation expiry, use `overnight-full` again in the next allocation.

For manual, foreground operation instead:

```bash
bash nyu_preact_v3/nyu.sh prepare-full
bash nyu_preact_v3/nyu.sh train-full
```

Continue it in another allocation with:

```bash
bash nyu_preact_v3/nyu.sh resume-full
```

The full-data config is derived from the validated development config: only the name and validation size change. It trains **from random initialization**, not by fine-tuning a checkpoint. It keeps the fixed 320-epoch duration and frozen inference choices. It does not evaluate the model on its own training set and call that validation. Results go to `experiments/preact_sla_ta320_v3/full50k/`.

## 5. Create the final CSV

After full training reaches 320/320, in a GPU session:

```bash
bash nyu_preact_v3/nyu.sh predict-full
```

The last line says **DOWNLOAD THIS FILE:** and gives the complete path. That new directory contains:

- **`submission.csv`** — exactly 2,000 predictions, columns `id,label`, IDs in the supplied sample order, integer labels 0–9.
- `model.pt` — the single selected model, including its final normalization buffers and inference policy.
- `inference.json` — checkpoint, policy, input and CSV hashes, parameters and runtime versions.

Download the CSV from the exact folder printed by this command. In NYU's **Files** page navigate to that directory and download `submission.csv`. Keep the required filename when uploading to Kaggle. Do not reuse a previous copy in your home folder; that caused an earlier stale-file submission.

Optional independent raw, EMA and flip exports use previously validated choices. Each creates its own folder and checks its CSV independently:

```bash
VARIANT=raw bash nyu_preact_v3/nyu.sh predict-full
VARIANT=ema bash nyu_preact_v3/nyu.sh predict-full
VARIANT=flip bash nyu_preact_v3/nyu.sh predict-full
```

`raw` and `ema` use their plain head without external TTA. `flip` uses the raw/EMA and head choice that scored best under flip TTA on development validation. `selected` is the default overall validation-selected policy. These are separate single-model outputs, not a multi-model ensemble. Generate alternatives for analysis/reproducibility, not automatic leaderboard probing. The supplied rules allow only **five submissions per day for the whole team**.

To reproduce an exported model's predictions later, set its actual path (no training data or EMA recalibration needed):

```bash
export EXPORTED_MODEL="/absolute/path/printed/export-folder/model.pt"
bash nyu_preact_v3/nyu.sh replay
```

For memory-conservative CPU replay:

```bash
DEVICE=cpu BATCH_SIZE=1 bash nyu_preact_v3/nyu.sh replay
```

CPU replay is slower. It retains the checkpoint and inference policy; numerical differences across hardware can affect near-tied predictions. Prefer an allocated GPU for initial selection/export, especially EMA normalization recalculation. This package's training intentionally requires a GPU. Use `replay` for CPU inference after exporting once, rather than rerunning expensive policy selection.

## Configuration and changes

`configs/development.json` is the complete recipe.

| Setting | Value |
|---|---|
| Model | Original PreAct ResNet + SLA/SD, widths 64/128/256, blocks 4/4/3 |
| Trainable parameters | **4,706,674**, including both heads |
| Training data | Fixed stratified 45,000 / 5,000; 500 validation examples per class |
| Final training | Separate 50,000-image scratch run after validation |
| Epochs | 320; no training-accuracy early stopping |
| Batch size | 128 original images; SLA makes four rotated copies |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Weight decay | 0.0005 on convolution/linear weights; none on norm parameters/biases |
| Learning rate | Peak 0.1; five-epoch linear warmup; cosine decay to 0.00005 |
| Augmentation | TrivialAugmentWide → horizontal flip → reflected crop(32,padding4) → normalization → Cutout12 |
| Cutout | One 12×12 patch per training image, normalized mean fill |
| Label smoothing | 0.1 on plain-head CE only; original unsmoothed joint CE and detached SLA self-distillation |
| EMA | 0.999; selected separately from raw weights |
| Precision | FP16 autocast + GradScaler for training; FP32 evaluation/inference |
| Seed | 42; deterministic split, saved RNG states and per-epoch loader seeds |
| Normalization | Mean (0.4914,0.4822,0.4465); std (0.247,0.2435,0.2616) |

This new candidate supersedes neither the saved V2 run nor the earlier `preact_sla_sd_320_clean_v1` files. Unlike that earlier requested weak-augmentation experiment, V3 retains TrivialAugment, Cutout12 and smoothing0.1 from the stronger supplied baseline. The baseline used batch512 and actual peakLR0.4; V3 uses batch128 and actual peakLR0.1, keeping that LR/batch ratio. It adds holdout-based selection, longer training, EMA, robust resume, provenance and strict export checks. Augmentation is implemented with torchvision PIL transforms and independent flips, rather than the original GPU loader's alternating flips; it is the same augmentation family, not an identical training trajectory.

The original model definitions are copied byte-for-byte from the notebook extraction. Unused model classes remain in that source, but the factory constructs only the stated PreAct architecture. No downloaded weights, teacher checkpoints, external training data, Mixup, CutMix, or test-time learning are used. The parameter check includes the 10-class and 40-class heads; exported inference uses one raw **or** EMA model.

## Verification and limits

See `reports/RELEASE_CHECKS.md` and the machine-readable reports. The current independent data/inference pipeline has passed local checks. That does **not** establish the cause of the earlier Wide98 validation/Kaggle discrepancy. High training accuracy is not a target to tune toward or away from. The previous screenshots changed checkpoints and inference policies as well as training duration.

No claim is made that this recipe is globally best, that 320 epochs is better than 200 for this data, or that it will score 0.92. It is a reproducible candidate anchored to the stronger supplied architecture and augmentation. Public leaderboard accuracy measures only 1,000 of the 2,000 evaluation images under the provided 50/50 split; the private result can differ. Do not use hidden-image inspection or external labels to choose or modify predictions.

Keep this package, audit, configuration, selection report, logs, full checkpoint and exported model alongside the submission for the course's reproducibility requirements. Code and data hashes are checked on resume; environment versions are recorded and a change is reported. Exact bitwise identity across hardware/library versions is not guaranteed. GPU AMP behavior must be verified in the user's actual NYU environment; local verification used CPU only.

Dependencies: Python 3.10+, PyTorch 2.8+ with a compatible torchvision, numpy, Pillow. See `requirements.txt` for ranges. Use the already-working GPU environment. Preserve its exact package versions for reproduction rather than upgrading during a run.

## References and attribution

- He et al., [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), 2016: pre-activation residual blocks.
- Lee, Hwang and Shin, [Self-supervised Label Augmentation via Input Transformations](https://proceedings.mlr.press/v119/lee20c.html), ICML 2020; [authors' implementation](https://github.com/hankook/SLA): joint class/rotation objective, aggregation and self-distillation.
- Müller and Hutter, [TrivialAugment](https://arxiv.org/abs/2103.10158), ICCV 2021; implemented with torchvision TrivialAugmentWide.
- DeVries and Taylor, [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552), 2017.
- PyTorch [automatic mixed precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): autocast and GradScaler usage.
- The unchanged notebook model file also contains unused AirResNet, BlurPool and Shake-Shake code. Their notebook credits remain in the file; only PreAct is instantiated in V3.
- Competition rules are included as supplied in `rules/supplied_rules.txt`; the course staff remains the authority on eligibility.
