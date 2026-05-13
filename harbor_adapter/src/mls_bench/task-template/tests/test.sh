#!/bin/bash
# Harbor verifier for an MLS-Bench task.
#
# Harbor mounts this directory at /tests/ only at verification time, so
# anything here is hidden from the agent during its work session. Layout
# expected inside /tests/:
#   /tests/test.sh                 (this script)
#   /tests/score_task.py           (the guard/run-evals/score helper)
#   /tests/meta/config.json
#   /tests/meta/parser.py
#   /tests/meta/score_spec.py
#   /tests/meta/leaderboard.csv
#   /tests/meta/[budget_check.py]
#   /tests/meta/pristine/<rel>     (declared-file pristines for byte-segment diff)
#   /tests/meta/pristine_manifest.json   (sha256 of every file under a guarded prefix)
#   /tests/eval/scripts/*.sh       (every eval script, visible + hidden)

# Reset PATH so an agent-left python/pip shim under /workspace can't shadow
# the system interpreter the verifier uses. Strip every env var Python
# inspects during startup so an agent-planted PYTHONSTARTUP /
# PYTHONUSERBASE / sitecustomize.py won't be imported.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE \
      PYTHONNOUSERSITE PYTHONIOENCODING PYTHONHASHSEED \
      LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT
export PYTHONNOUSERSITE=1
# Prefer the python that lives inside the package base image. /usr/bin/python3
# is what shipped with the base OS (Debian); /opt/conda/bin/python3 etc. live
# on the pytorch base; either is a system path the agent shouldn't write to
# unless they were already root, in which case all bets are off. We still
# pass -I (isolated mode) on every invocation: no user site, ignore PYTHON*
# envs, ignore current directory on sys.path.
PYTHON_BIN=/usr/bin/python3
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN=/usr/local/bin/python3
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN=$(command -v python3)
fi
export MLSBENCH_VERIFIER_PYTHON="${PYTHON_BIN}"

# Snapshot the verifier python's integrity so a tampered binary is at least
# logged for postmortem. We cannot prevent a root agent from replacing it,
# but a mismatch against the base image's recorded sha lets us flag it.
"${PYTHON_BIN}" -c "import hashlib,sys; print('verifier_python_sha256='+hashlib.sha256(open(sys.executable,'rb').read()).hexdigest())" 2>&1 \
    | tee -a /logs/verifier/python_audit.txt >/dev/null || true

set -uo pipefail

mkdir -p /logs/verifier

TASK_ID="$(cat /tests/meta/task_id 2>/dev/null || echo unknown)"
PKG_NAME="$(cat /tests/meta/package 2>/dev/null || echo unknown)"
WORKDIR="$(cat /tests/meta/workdir 2>/dev/null || echo /workspace)"

PRIVATE_ROOT="$(mktemp -d /tmp/mlsbench-verifier.XXXXXX)"
PRIVATE_META="${PRIVATE_ROOT}/meta"
cp -a /tests/meta "${PRIVATE_META}"
cp /tests/score_task.py "${PRIVATE_ROOT}/score_task.py"
chmod -R a-w /tests/meta
chmod -R a-w "${PRIVATE_META}" "${PRIVATE_ROOT}/score_task.py"
chmod go-rwx "${PRIVATE_ROOT}" || true
trap 'rm -rf "${PRIVATE_ROOT}"' EXIT

# Step 1: edit-range diff guard. The pristine baseline is the
# per-task-rendered tree under tests/meta/pristine/ (mounted only at verify
# time), so the agent had no opportunity to tamper with it.
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" guard \
    --task-meta "${PRIVATE_META}" \
    --pristine "${PRIVATE_META}/pristine" \
    --workspace "${WORKDIR}" \
    --violation-out /logs/verifier/violation.txt
guard_rc=$?

if [ "${guard_rc}" -eq 10 ]; then
    echo "0" > /logs/verifier/reward.txt
    echo "edit-range violation — see /logs/verifier/violation.txt" >&2
    exit 0
fi
if [ "${guard_rc}" -ne 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    echo "guard script failed unexpectedly (rc=${guard_rc})" >&2
    exit 0
fi

# Step 2: run every eval script (visible + hidden), with cwd = the package root
# (config.json::files[].filename is workdir-relative; PKG_NAME is the first
# path component, e.g. "causal-learn").
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" run-evals \
    --task-meta "${PRIVATE_META}" \
    --workspace "${WORKDIR}" \
    --eval-root /tests/eval \
    --out-dir /logs/verifier

# Step 3: aggregate metrics → combined_score → reward.txt.
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" score \
    --task-meta "${PRIVATE_META}" \
    --out-dir /logs/verifier \
    --reward-out /logs/verifier/reward.txt

exit 0
