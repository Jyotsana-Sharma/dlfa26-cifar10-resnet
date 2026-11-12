#!/usr/bin/env bash
# Called by nyu.sh overnight/overnight-full; no fixed four-hour stop.
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${STUDY_DIR:?}" "${PYTHON_BIN:?}"
MODE="${1:?development or full50k}"
if [[ "$MODE" != development && "$MODE" != full50k ]]; then echo 'Invalid stage' >&2; exit 1; fi
LAUNCH_LOCK="$STUDY_DIR/overnight_${MODE}.lock"
if ! mkdir "$LAUNCH_LOCK"; then
  echo "Another overnight launcher or leftover lock exists: $LAUNCH_LOCK. Check its process before removing it." >&2
  exit 1
fi
printf '%s pid=%s\n' "$(hostname)" "$$" > "$LAUNCH_LOCK/owner.txt"
cleanup() {
  code=$?
  rm -f -- "$LAUNCH_LOCK/owner.txt"
  rmdir -- "$LAUNCH_LOCK"
  if [[ "$code" -ne 0 ]]; then echo "OVERNIGHT STOPPED (exit $code). Read the error above; committed checkpoints are retained."; fi
}
trap cleanup EXIT
printf 'Overnight stage: %s. Extra process cap: %s seconds (0 = none).\n' "$MODE" "${SESSION_SECONDS:-0}"
if [[ ! -f "$STUDY_DIR/audit.json" ]]; then bash "$PACKAGE_DIR/nyu.sh" audit; fi
TRAIN_ACTION=train
RESUME_ACTION=resume
if [[ "$MODE" == full50k ]]; then
  if [[ ! -f "$STUDY_DIR/selection.json" ]]; then echo 'Complete and review development validation before starting full50K.' >&2; exit 1; fi
  if [[ ! -f "$STUDY_DIR/full50k.json" ]]; then bash "$PACKAGE_DIR/nyu.sh" prepare-full; fi
  TRAIN_ACTION=train-full
  RESUME_ACTION=resume-full
fi
if [[ -f "$STUDY_DIR/$MODE/last.pt" ]]; then
  bash "$PACKAGE_DIR/nyu.sh" "$RESUME_ACTION"
else
  bash "$PACKAGE_DIR/nyu.sh" "$TRAIN_ACTION"
fi
# Never treat a safe allocation stop as completed training.
"$PYTHON_BIN" - "$STUDY_DIR/$MODE/last.pt" <<'PY'
import sys,torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
n,total=c['completed_epochs'],c['config']['epochs']
if n!=total:
    print(f'Allocation ended before completion: {n}/{total}. Start the same overnight command in a new GPU allocation.',flush=True)
    sys.exit(75)
print(f'All {total} epochs are complete.',flush=True)
PY
if [[ "$MODE" == development ]]; then
  if [[ ! -f "$STUDY_DIR/selection.json" ]]; then bash "$PACKAGE_DIR/nyu.sh" select; fi
  echo "OVERNIGHT COMPLETE. Review validation: $STUDY_DIR/selection.json"
else
  bash "$PACKAGE_DIR/nyu.sh" predict-full
  echo 'OVERNIGHT COMPLETE. The new submission.csv path is printed above.'
fi
