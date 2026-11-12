#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$PACKAGE_DIR")"
STUDY_DIR="${STUDY_DIR:-$PROJECT_DIR/experiments/preact_sla_ta320_v3}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/resnet-env/bin/python}"
DATA_DIR="${DATA_DIR:-$HOME}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SESSION_SECONDS="${SESSION_SECONDS:-0}"
VARIANT="${VARIANT:-selected}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN. Set PYTHON_BIN to your PyTorch Python." >&2
  exit 1
fi
ACTION="${1:-help}"
if [[ "$#" -gt 1 ]]; then echo 'Use one action per command; see help.' >&2; exit 1; fi
if [[ "$ACTION" == help ]]; then
  echo 'Actions: overnight, overnight-full, check, audit, train, resume, select, prepare-full, train-full, resume-full, predict, predict-full, status, replay'
  echo 'Environment: DATA_DIR, PYTHON_BIN, STUDY_DIR, SESSION_SECONDS, DEVICE, BATCH_SIZE, VARIANT'
  echo 'File overrides: TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_METADATA, SAMPLE_SUBMISSION'
  echo 'Replay only: EXPORTED_MODEL=/absolute/path/to/export/model.pt'
  exit 0
fi
if [[ "$ACTION" == overnight || "$ACTION" == overnight-full ]]; then
  mkdir -p "$STUDY_DIR/logs"
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  OVERNIGHT_LOG="$STUDY_DIR/logs/${ACTION}_${STAMP}.log"
  MODE=development
  if [[ "$ACTION" == overnight-full ]]; then MODE=full50k; fi
  nohup env STUDY_DIR="$STUDY_DIR" PYTHON_BIN="$PYTHON_BIN" DATA_DIR="$DATA_DIR" SESSION_SECONDS="$SESSION_SECONDS" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" bash "$PACKAGE_DIR/overnight.sh" "$MODE" > "$OVERNIGHT_LOG" 2>&1 < /dev/null &
  NIGHT_PID=$!
  printf '%s\n' "$NIGHT_PID" > "$STUDY_DIR/logs/${ACTION}_${STAMP}.pid"
  echo "Background launcher PID: $NIGHT_PID"
  echo "Follow its progress: tail -f \"$OVERNIGHT_LOG\""
  echo 'It will run to 320 epochs unless interrupted, an error occurs, or the GPU allocation expires.'
  echo 'Closing this terminal will not stop the launcher. Do not cancel the NYU GPU session.'
  exit 0
fi
if [[ "$ACTION" == check ]]; then
  exec "$PYTHON_BIN" -m unittest discover -s "$PACKAGE_DIR/tests" -v
fi
if [[ "$ACTION" == status ]]; then
  "$PYTHON_BIN" - "$STUDY_DIR" <<'PY'
import json,sys
from pathlib import Path
for name in ('development','full50k'):
    path=Path(sys.argv[1])/name/'progress.json'
    if not path.exists():
        print(name+': no completed-epoch report yet');continue
    p=json.loads(path.read_text());last=p['history'][-1]
    print(name+': '+json.dumps(last,indent=2))
PY
  exit 0
fi
resolve_images() {
  if [[ -d "$DATA_DIR/$1" ]]; then printf '%s' "$DATA_DIR/$1"
  elif [[ -f "$DATA_DIR/$1.zip" ]]; then printf '%s' "$DATA_DIR/$1.zip"
  else echo "Missing $DATA_DIR/$1 (directory or ZIP). Set DATA_DIR or the file override." >&2; return 1
  fi
}
if [[ "$ACTION" != prepare-full && "$ACTION" != replay ]]; then
  TRAIN_IMAGES="${TRAIN_IMAGES:-$(resolve_images train_images)}"
  TRAIN_LABELS="${TRAIN_LABELS:-$DATA_DIR/train_labels.csv}"
  [[ -f "$TRAIN_LABELS" ]] || { echo "Missing $TRAIN_LABELS" >&2; exit 1; }
fi
if [[ "$ACTION" == audit || "$ACTION" == predict || "$ACTION" == predict-full || "$ACTION" == replay ]]; then
  TEST_IMAGES="${TEST_IMAGES:-$(resolve_images test_images)}"
  TEST_METADATA="${TEST_METADATA:-$DATA_DIR/test.csv}"
  SAMPLE_SUBMISSION="${SAMPLE_SUBMISSION:-$DATA_DIR/sample_submission.csv}"
  # Existing NYU installation stores its supplied sample here.
  if [[ ! -f "$SAMPLE_SUBMISSION" && -f "$PROJECT_DIR/v2/data/sample_submission.csv" ]]; then
    SAMPLE_SUBMISSION="$PROJECT_DIR/v2/data/sample_submission.csv"
  fi
  [[ -f "$TEST_METADATA" && -f "$SAMPLE_SUBMISSION" ]] || { echo 'Missing test.csv or sample_submission.csv; set explicit file overrides.' >&2; exit 1; }
  TEST_ARGS=(--test-images "$TEST_IMAGES" --test-metadata "$TEST_METADATA" --sample "$SAMPLE_SUBMISSION")
fi
mkdir -p "$STUDY_DIR/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
DEV_CONFIG="$PACKAGE_DIR/configs/development.json"
FULL_CONFIG="$STUDY_DIR/full50k.json"
AUDIT="$STUDY_DIR/audit.json"
POLICY="$STUDY_DIR/selection.json"
CMD=("$PYTHON_BIN" -u "$PACKAGE_DIR/run.py")
case "$ACTION" in
  audit)
    [[ ! -e "$AUDIT" ]] || { echo "Audit already exists: $AUDIT. Preserve it; use a fresh STUDY_DIR for changed code/config." >&2; exit 1; }
    "$PYTHON_BIN" -m unittest discover -s "$PACKAGE_DIR/tests" -v 2>&1 | tee "$STUDY_DIR/logs/check_${STAMP}.log"
    CMD+=(audit --config "$DEV_CONFIG" --images "$TRAIN_IMAGES" --labels "$TRAIN_LABELS" "${TEST_ARGS[@]}" --output "$AUDIT")
    ;;
  train|resume|train-full|resume-full)
    RUN="$STUDY_DIR/development"; CONFIG="$DEV_CONFIG"
    if [[ "$ACTION" == *-full ]]; then RUN="$STUDY_DIR/full50k"; CONFIG="$FULL_CONFIG"; fi
    CMD+=(train --config "$CONFIG" --images "$TRAIN_IMAGES" --labels "$TRAIN_LABELS" --audit "$AUDIT" --run "$RUN" --device cuda --seconds "$SESSION_SECONDS")
    if [[ "$ACTION" == resume* ]]; then CMD+=(--resume); fi
    if [[ "$ACTION" == *-full ]]; then CMD+=(--policy "$POLICY"); fi
    ;;
  select)
    CMD+=(select --images "$TRAIN_IMAGES" --labels "$TRAIN_LABELS" --run "$STUDY_DIR/development" --output "$POLICY" --device "$DEVICE" --batch-size "$BATCH_SIZE")
    ;;
  prepare-full)
    CMD+=(promote --policy "$POLICY" --output "$FULL_CONFIG")
    ;;
  predict|predict-full)
    CMD+=(predict --images "$TRAIN_IMAGES" --labels "$TRAIN_LABELS" "${TEST_ARGS[@]}" --policy "$POLICY" --device "$DEVICE" --batch-size "$BATCH_SIZE" --variant "$VARIANT" --output "$STUDY_DIR/exports/${ACTION}_${VARIANT}_${STAMP}")
    if [[ "$ACTION" == predict-full ]]; then CMD+=(--checkpoint "$STUDY_DIR/full50k/last.pt"); fi
    ;;
  replay)
    : "${EXPORTED_MODEL:?Set EXPORTED_MODEL to the exported model.pt}"
    CMD+=(replay --checkpoint "$EXPORTED_MODEL" "${TEST_ARGS[@]}" --device "$DEVICE" --batch-size "$BATCH_SIZE" --output "$STUDY_DIR/exports/replay_${STAMP}")
    ;;
  *) echo "Unknown action $ACTION. Run: bash nyu_preact_v3/nyu.sh help" >&2; exit 1 ;;
esac
printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "$STUDY_DIR/logs/${ACTION}_${STAMP}.log"
