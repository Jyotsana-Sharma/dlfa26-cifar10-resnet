"""Exercise the real overnight shell coordinator with a fake training launcher.

No training runs here: tiny metadata checkpoints stand in for complete/incomplete
GPU runs. The real model and training behavior are covered by separate tests.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
FAKE_LAUNCHER='''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$1" >> "$STUDY_DIR/actions.txt"
case "$1" in
  audit) echo '{}' > "$STUDY_DIR/audit.json" ;;
  prepare-full) echo '{}' > "$STUDY_DIR/full50k.json" ;;
  train|resume) mkdir -p "$STUDY_DIR/development"; cp "$FAKE_CHECKPOINT" "$STUDY_DIR/development/last.pt" ;;
  train-full|resume-full) mkdir -p "$STUDY_DIR/full50k"; cp "$FAKE_CHECKPOINT" "$STUDY_DIR/full50k/last.pt" ;;
  select) echo '{}' > "$STUDY_DIR/selection.json" ;;
  predict-full) touch "$STUDY_DIR/prediction-was-requested" ;;
  *) exit 19 ;;
esac
'''

class OvernightChecks(unittest.TestCase):
    def test_completion_resume_and_incomplete_stop(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);pkg=base/'package';pkg.mkdir()
            shutil.copyfile(ROOT/'overnight.sh',pkg/'overnight.sh')
            (pkg/'nyu.sh').write_text(FAKE_LAUNCHER)
            fake=base/'metadata.pt'
            torch.save(dict(completed_epochs=320,config=dict(epochs=320)),fake)
            study=base/'study';study.mkdir()
            env=dict(os.environ,STUDY_DIR=str(study),PYTHON_BIN=sys.executable,FAKE_CHECKPOINT=str(fake),SESSION_SECONDS='0')
            def launch(mode):
                return subprocess.run(['bash',str(pkg/'overnight.sh'),mode],env=env,capture_output=True,text=True,timeout=60)
            def actions():return (study/'actions.txt').read_text().splitlines()
            result=launch('development')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(actions(),['audit','train','select'])
            self.assertIn('OVERNIGHT COMPLETE',result.stdout)
            self.assertFalse((study/'overnight_development.lock').exists())
            result=launch('development')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(actions()[-1],'resume')
            result=launch('full50k')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(actions()[-2:],['train-full','predict-full'])
            self.assertTrue((study/'prediction-was-requested').exists())
            torch.save(dict(completed_epochs=313,config=dict(epochs=320)),fake)
            before=len(actions());result=launch('full50k')
            self.assertEqual(result.returncode,75,result.stdout+result.stderr)
            self.assertEqual(actions()[before:],['resume-full'])
            self.assertNotIn('OVERNIGHT COMPLETE',result.stdout)
            self.assertIn('313/320',result.stdout)
            self.assertFalse((study/'overnight_full50k.lock').exists())
            lock=study/'overnight_development.lock';lock.mkdir()
            before=len(actions());result=launch('development')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(len(actions()),before)
            self.assertTrue(lock.exists())

if __name__=='__main__':unittest.main()
