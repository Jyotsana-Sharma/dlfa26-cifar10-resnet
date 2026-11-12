#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$BASE")"
STUDY_DIR="${STUDY_DIR:-$PROJECT/experiments/preact_baseline_recovery_v4}"
export STUDY_DIR
export PYTHON_BIN="${PYTHON_BIN:-$HOME/resnet-env/bin/python}"
export SESSION_SECONDS="${SESSION_SECONDS:-0}"
export DEVICE="${DEVICE:-cuda}"
export DATA_DIR="${DATA_DIR:-$HOME}"
export SAMPLE_SUBMISSION="${SAMPLE_SUBMISSION:-$PROJECT/v2/data/sample_submission.csv}"
ACTION="${1:-help}"
if (( $# > 1 ));then echo 'Usage: bash preact_recovery_v4/run.sh ACTION' >&2; exit 2;fi
case "$ACTION" in
  help)
    cat <<'USAGE'
Actions:
  setup         Isolate unmodified V3 architecture/source into V4 backend
  check         Run existing V3 unit tests against isolated source
  audit         Audit permitted training/test data before training
  overnight     Auto audit (if needed), train/resume 45K dev, select 5K policy
  status        Show development and full50K progress
  select        Select raw/EMA/TTA from the held-out validation set
  predict       Optional 45K CSV; diagnostic only
  overnight-full Auto prepare, train/resume all 50K, export final CSV
  predict-full  Re-export final CSV from completed 50K checkpoint
  replay        Replay exported model with EXPORTED_MODEL=...
  probe-v3      Independently check V3 selected model validation predictions
  compare-csv   Compare two saved submissions: CSV_A=... CSV_B=...
Note: Requires preexisting nyu_preact_v3 source in this same project.
USAGE
    exit 0;;
esac
if [[ ! -x "$PYTHON_BIN" ]];then echo "Missing Python environment: $PYTHON_BIN" >&2;exit 1;fi
if [[ "$ACTION" == compare-csv ]];then
  : "${CSV_A:?Set CSV_A=/path/to/first.csv}" "${CSV_B:?Set CSV_B=/path/to/second.csv}"
  exec "$PYTHON_BIN" "$BASE/probe.py" compare --a "$CSV_A" --b "$CSV_B" --sample "$SAMPLE_SUBMISSION"
fi
# Resolve V3 source: prefer local PreAct_Backend_Checkpoints copy, then project-level.
V3_ARG=""
if [[ -d "$BASE/PreAct_Backend_Checkpoints/nyu_preact_v3" ]];then
  V3_ARG="--v3 $BASE/PreAct_Backend_Checkpoints/nyu_preact_v3"
elif [[ -d "$PROJECT/nyu_preact_v3" ]];then
  V3_ARG=""  # bootstrap.py default resolves to $PROJECT/nyu_preact_v3
else
  echo "ERROR: nyu_preact_v3 not found. Expected at $BASE/PreAct_Backend_Checkpoints/nyu_preact_v3 or $PROJECT/nyu_preact_v3" >&2
  exit 1
fi
"$PYTHON_BIN" "$BASE/bootstrap.py" --project "$PROJECT" $V3_ARG
if [[ "$ACTION" == setup ]];then exit 0;fi
if [[ "$ACTION" == probe-v3 ]];then
  exec "$PYTHON_BIN" "$BASE/probe.py" v3 --project "$PROJECT" --device "$DEVICE" --batch-size "${BATCH_SIZE:-32}" --max-images "${PROBE_IMAGES:-512}"
fi
case "$ACTION" in
  check|audit|overnight|overnight-full|select|status|predict|predict-full|replay)
    if [[ "$ACTION" == overnight || "$ACTION" == overnight-full ]];then
       if [[ "$ACTION" == overnight ]];then STAGE=development; else STAGE=full50k;fi
       exec bash "$BASE/backend/overnight.sh" "$STAGE"
    else
       exec bash "$BASE/backend/nyu.sh" "$ACTION"
    fi
    ;;
  *) echo "Unknown action: $ACTION" >&2;exit 2;;
esac
