# V3 release checks — 2026-11-04

This package is ready for an audited NYU GPU experiment. It is **not a trained or scored 92% model**. No 320-epoch run, Kaggle upload, commit, push or deployment was performed during package preparation. Existing project checkpoints, submissions, scripts and logs were preserved.

## Measured local results

The actual `nyu.sh audit` launcher completed successfully using the supplied local images and CSVs. Its machine-readable result is `final_audit/audit.json`; `final_launcher_log.txt` records the invocation, tests and audit. The report contains source hashes and runtime versions. Local verification used Python 3.12.13, PyTorch 2.15.0, torchvision 0.29.2 and CPU; NYU GPU/CUDA/FP16 execution still needs the first real run. Run the audit on NYU, using the launcher, before training there.

| Check | Result |
|---|---|
| Full PreAct model, including both heads | 4,706,674 trainable parameters; below 5,000,000 |
| Training data decoded | 50,000 RGB images, 32×32; 5,000 per class |
| Stratified development split | 45,000 / 5,000; 500 holdout examples per class |
| Train/validation index overlap | 0 |
| Exact duplicate training pixels | 0 |
| Exact training/evaluation pixel overlap | 0 |
| Evaluation data | 2,000 images, unique IDs, metadata/sample ordering identical |
| Train/evaluation normalization | Shared constants; direct numerical-equivalence test passed |
| Real-model CPU forward/backward/update | Finite loss/gradients/parameters and nonzero weight update |
| Unit/integration checks | 11 passed |
| Original model source | Byte-identical to supplied notebook extraction |
| Shell parsing | Both launch scripts passed `bash -n` |

Model source SHA256: `e03b2687dbdae6fff70362ae7f8c22616f43ff237a6faac28c819d67bc84442d`.

The tests verify original SLA loss and selected gradients, original six-view inference arithmetic, batch independence, EMA parameter/buffer updates, BatchNorm recalibration without changing parameters, exact CPU next-update resume equivalence, learning-rate endpoints, cluster time parsing, split reproducibility, ZIP loading, and rejection of missing/duplicate/reordered IDs. The CSV checks require exactly 2,000 predictions with `id,label` and valid integer classes. Existing output paths are rejected.

A small CPU stand-in model exercises the complete training → selection → full-data configuration → full-data training → CSV export → standalone replay flow. Its synthetic accuracies are test fixtures, not model results. Real PreAct forward/backward, parameter count and inference arithmetic are tested separately. The overnight shell test substitutes a fake child launcher with completed/incomplete checkpoint metadata: it verifies fresh training, resume, automatic post-training selection/export, exclusive launch locks, and that 313/320 is reported as incomplete. It does not run GPU training.

## Findings relevant to the historical score gap

The supplied baseline archive's cached train/test pixels and training labels matched the supplied local files in the prior archive audit. Its 2,000 saved submission IDs matched the sample ordering. The recorded original PreAct configuration trained on all 50,000 images for 200 epochs with no held-out validation. Its batch size was 512 and the PreAct optimizer's actual peak LR was 0.4; generic `lr=9` and `wd=0.012` fields were not the active PreAct optimizer settings. V3 uses batch128 and LR0.1, preserving that LR/batch ratio, with longer training and a holdout stage.

Inspection of the local V2 inference implementation shows that `plain_weight` blends the plain classification head with the SLA aggregate head **within one loaded checkpoint**. It does not blend raw and EMA checkpoints. TTA B is original plus horizontal flip. V2's probability-averaging mode averages view probabilities; the supplied original inference averages weighted view logits before softmax. These are real inference differences, but their causal effect on Kaggle accuracy has not been measured here.

The screenshots also showed an older CSV being copied after newer training, followed by a corrected 320-epoch submission scoring 0.891. That explains the stale-file incident, not the remaining generalization gap. V3 prints the exact new CSV location and records all input/checkpoint/policy/output hashes.

**The underlying reason for approximately 97% validation versus approximately 89% Kaggle is unresolved.** The new data checks found no exact overlap, ID-order or shared-normalization defect in this new pipeline. They do not prove the historical live HPC pipeline was leak-free, rule out near-duplicates or distribution shift, or establish that more epochs caused lower test accuracy. Historical runs changed architecture, training subset, augmentation and inference policy. No hidden labels were used, and the original submission was not replayed, following the user's later instruction.

## Source and configuration changes

All files below are confined to the new `nyu_preact_v3` folder.

| File | Purpose |
|---|---|
| `models.py` | Unchanged notebook architecture definitions; factory uses only the proven PreAct SLA network |
| `dataset.py` | Strict supplied-data loading, numeric class mapping, deterministic stratified split, lazy ZIP reads and identity fingerprints |
| `core.py` | Shared transforms, original SLA/SD objective, SGD groups, FP16-compatible losses, EMA, schedule, inference policies, BN recalibration, atomic saves and CSV checks |
| `run.py` | Audited train/resume, validation selection, full50K promotion, separate raw/EMA/flip exports, provenance checks and standalone replay |
| `configs/development.json` | Complete new batch128, 320-epoch, seed42 FP16/SGD/TrivialAugment recipe |
| `nyu.sh` | Actual NYU command wrapper, explicit input overrides, unique logs/exports, status and background launch commands |
| `overnight.sh` | Fresh-or-resume coordination, completion verification and automatic selection or final export |
| `tests/` | Model, loader, inference, resume, export and overnight orchestration checks |
| `README.md`, `requirements.txt`, `rules/` | Exact instructions, dependencies, citations, limitations and supplied rules |

This is a separate candidate from the previously requested weak-augmentation `preact_sla_sd_320_clean_v1`. It retains the original stronger baseline's TrivialAugment, Cutout12 and smoothing0.1, and preserves its SGD weight-decay grouping. Both old experiments remain untouched.

## Overnight execution and remaining limits

FP16 autocast with GradScaler is enabled for training. Evaluation/export use FP32. No speedup or 10-hour completion claim is based on a measured A100 benchmark; use the progress log's estimated hours remaining after several complete epochs.

The default extra time budget is **zero, meaning no artificial cap**. The trainer runs toward all 320 epochs, without training-accuracy early stopping. It respects the actual Slurm allocation deadline if available and retains atomic per-epoch resume state. The background launcher survives terminal closure; it cannot survive the cluster cancelling or expiring the GPU allocation. A new allocation can resume the same run. An unexpected failure is reported, not silently retried forever.

Development validation chooses the checkpoint/head/TTA policy; EMA and raw predictions remain separate. The same holdout selects epochs and inference, so validation accuracy is optimistic. Full50K training is a separate scratch run started only after the user reviews validation. Test images are used only for shape/identity checks and final predictions, never training, BN adaptation, or policy selection. No pretrained/external weights or data are included. A single raw or EMA model, including both heads, is exported.

This package cannot guarantee a Kaggle score. Do not interpret training accuracy, elapsed training time, or completion of 320 epochs as a promise of better generalization.
