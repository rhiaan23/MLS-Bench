#!/usr/bin/env bash
# Strange Loop <-> MLS-Bench official-verifier entrypoint.
#
#   Usage:  bash tools/sl_climb_entrypoint.sh <task_id>
#   Run from the repo root of this (cloned) fork, AFTER the coding agent has
#   edited the task's editable file inside its locked line range.
#
# It reconstructs the Harbor layout the official verifier expects, runs the
# UNMODIFIED official verifier (guard -> run-evals -> score), and writes
# `result.json` with `objective` = the official combined_score (reward.txt).
# The Strange Loop run harness reads result.json and records objective as the
# climb's score, so the number is the benchmark's own, never the agent's.
#
# Design notes:
#  - The editable file (e.g. pytorch-vision/custom_activation.py) IS the training
#    entrypoint; the eval scripts `cd /workspace && python <pkg>/<file> ...`.
#  - We repopulate /workspace from the (edited) _scaffold every run.
#  - We re-checkout tests/ from the pinned ref so the agent cannot have tampered
#    with the scorer/hidden evals (best-effort honest-run integrity for v1).
#  - guard rc==10 => edit-range violation => objective 0 (matches Harbor).
set -uo pipefail

TASK="${1:?usage: sl_climb_entrypoint.sh <task_id>}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKROOT="$REPO_DIR/harbor/tasks/mls-bench__${TASK}"
META="$TASKROOT/tests/meta"
EVALROOT="$TASKROOT/tests/eval"
SCORER="$TASKROOT/tests/score_task.py"
RESULT="${CLIMB_OBJECTIVE_PATH:-$REPO_DIR/result.json}"
case "$RESULT" in /*) : ;; *) RESULT="$REPO_DIR/$RESULT" ;; esac
LOGDIR=/logs/verifier

[ -f "$SCORER" ] || { echo "FATAL: scorer not found at $SCORER (task '$TASK' present?)" >&2; exit 1; }
WORKDIR="$(cat "$META/workdir" 2>/dev/null || echo /workspace)"

mksudo() { mkdir -p "$1" 2>/dev/null || sudo mkdir -p "$1"; }
mksudo "$LOGDIR"; mksudo /data

# --- mlsbench must be importable (pure-python; heavy deps come from the image) -
python -c "import mlsbench" 2>/dev/null || pip install -e "$REPO_DIR" >/dev/null 2>&1 || true

# --- CLIMB_SMOKE: fast image-prebuild validation (bounded to 900s by the harness).
# Prove the entrypoint RUNS in this image WITHOUT the full ~1hr verifier: import the
# heavy stack, confirm the editable file compiles, run the (data-free) guard, and
# write result.json. No dataset download, no training. ----------------------------- #
if [ "${CLIMB_SMOKE:-}" = "1" ]; then
  echo "[sl_climb_entrypoint] CLIMB_SMOKE=1 — fast structural validation (no training)"
  rm -rf "$WORKDIR" 2>/dev/null || sudo rm -rf "$WORKDIR"; mksudo "$WORKDIR"
  cp -a "$TASKROOT/environment/_scaffold/." "$WORKDIR/"
  EDIT_REL="$(python -I - "$META" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]+"/config.json"))["files"][0]["filename"])
PY
)"
  ( cd "$WORKDIR" && python - "$EDIT_REL" <<'PY'
import sys, py_compile
import torch        # noqa: F401 — prove the image carries the heavy stack
import mlsbench      # noqa: F401
py_compile.compile(sys.argv[1], doraise=True)   # editable file is valid python
print("[smoke] torch+mlsbench import OK; editable file compiles:", sys.argv[1])
PY
  ) || { echo "[sl_climb_entrypoint] SMOKE FAILED (image missing deps or file broken)" >&2; exit 1; }
  python -I "$SCORER" guard --task-meta "$META" --pristine "$META/pristine" \
      --workspace "$WORKDIR" --violation-out "$LOGDIR/violation.txt" || true  # cheap, data-free
  python - "$RESULT" <<'PY'
import json,sys
json.dump({"objective":0.0,"metrics":{"smoke":1.0}}, open(sys.argv[1],"w"))
PY
  echo "[sl_climb_entrypoint] CLIMB_SMOKE ok — image validated, result.json written"
  exit 0
fi

# --- reconstruct /workspace from the (edited) scaffold ---------------------- #
rm -rf "$WORKDIR" 2>/dev/null || sudo rm -rf "$WORKDIR"
mksudo "$WORKDIR"
cp -a "$TASKROOT/environment/_scaffold/." "$WORKDIR/"

# --- clean verifier from the pinned ref (discard any edits under tests/) ----- #
git -C "$REPO_DIR" checkout -- "harbor/tasks/mls-bench__${TASK}/tests" 2>/dev/null || true

emit_fail() { # objective, note
  python - "$1" "$2" "$RESULT" <<'PY'
import json,sys
json.dump({"objective":float(sys.argv[1]),"metrics":{},"note":sys.argv[2]}, open(sys.argv[3],"w"))
PY
}

# --- official verifier: guard ----------------------------------------------- #
python -I "$SCORER" guard --task-meta "$META" --pristine "$META/pristine" \
    --workspace "$WORKDIR" --violation-out "$LOGDIR/violation.txt"
g=$?
if [ "$g" -eq 10 ]; then
  echo "EDIT-RANGE VIOLATION:"; cat "$LOGDIR/violation.txt" 2>/dev/null
  emit_fail 0 "edit-range violation"; exit 0
fi
[ "$g" -ne 0 ] && { echo "FATAL: guard rc=$g" >&2; exit 1; }

# --- official verifier: run all evals (visible + hidden) -------------------- #
python -I "$SCORER" run-evals --task-meta "$META" --workspace "$WORKDIR" \
    --eval-root "$EVALROOT" --out-dir "$LOGDIR" || { echo "FATAL: run-evals failed" >&2; exit 1; }

# --- official verifier: score -> reward.txt --------------------------------- #
python -I "$SCORER" score --task-meta "$META" --out-dir "$LOGDIR" \
    --reward-out "$LOGDIR/reward.txt" || { echo "FATAL: score failed" >&2; exit 1; }

# --- translate reward.txt + metrics.json -> result.json --------------------- #
python - "$LOGDIR" "$RESULT" <<'PY'
import json, os, sys
logdir, out = sys.argv[1], sys.argv[2]
reward = float(open(os.path.join(logdir, "reward.txt")).read().strip())
metrics = {}
mp = os.path.join(logdir, "metrics.json")
if os.path.isfile(mp):
    d = json.load(open(mp))
    metrics["combined_score"] = float(d.get("combined_score", reward))
    for k, v in (d.get("mean_metrics") or {}).items():
        try: metrics[k] = float(v)
        except (TypeError, ValueError): pass
json.dump({"objective": reward, "metrics": metrics}, open(out, "w"))
print(f"[sl_climb_entrypoint] objective={reward} -> {out}")
PY
echo "[sl_climb_entrypoint] done."
