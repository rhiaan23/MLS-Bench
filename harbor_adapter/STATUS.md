# MLS-Bench → Harbor adapter — status

Branch: `harbor-adapter` (this branch). Last updated: 2026-05-13.

## What this branch does

Publishes MLS-Bench's 140 ML-research tasks as a [Harbor](https://github.com/harbor-framework/harbor)
dataset. Once the per-package base images are on Docker Hub and the dataset
is rendered, any Harbor-supported agent (`claude-code`, `codex`,
`terminus-2`, `openhands`, `aider`, …) can run the suite via:

```
harbor run -p <dataset>/<task-dir> -a <agent> -m <model>
```

End-to-end already validated against Harbor v0.6.6 with the `oracle` agent
on `causal-observational-linear-gaussian`: oracle applies the `pc`
baseline, reward = 0.026, metrics byte-identical to MLS-Bench's
`baseline:pc` leaderboard row across all 5 eval cmds including the hidden
ER20-Noisy one. Hostile out-of-range solve.sh → reward 0 + violation file
populated.

## Architecture

Two-tier images + verifier-side pristine baseline:

1. **Per-package harbor base**
   `bohanlyu2022/mlsbench-harbor-<pkg>:latest`. Built once by
   `harbor_adapter/scripts/build_base_image.py`. Layers the existing
   `bohanlyu2022/mlsbench-<pkg>:latest` (which carries the package source
   + data deps from MLS-Bench's native image set) with package-level
   `pre_edit.py` applied. Pushed to Docker Hub.

2. **Per-task Dockerfile** — emitted by `harbor_adapter/src/mls_bench/adapter.py`
   into each rendered task dir. 1-line `FROM <harbor base>` + a single
   `COPY _scaffold/` line that overlays the task's `mid_edit` scaffolding
   onto `/workspace/<pkg>/`. Built locally by `harbor run`.

3. **Pristine baseline** for the edit-range guard is **not** in the image.
   The adapter ships per-task pristines under `tests/meta/pristine/<rel>`
   (full bytes for declared files; needed for content-based fixed-segment
   matching) plus `tests/meta/pristine_manifest.json` (sha256 of every
   file under any guarded prefix). Harbor uploads `tests/` to `/tests/`
   only at verify time, so a root agent cannot tamper with the baseline.

4. **Edit-range guard** (`tests/score_task.py::cmd_guard`) is
   content-based, not line-number-based: pristine is split into
   alternating fixed/editable segments per `config.json::files[].edit`,
   then every fixed segment is anchored verbatim in the agent's
   submission (rightmost feasible occurrence between surrounding
   anchors, robust to duplicate separators). Files under guarded
   prefixes but not declared are hash-checked against the manifest;
   missing manifest → fail closed.

5. **Hidden eval scripts** — all eval scripts (visible + hidden) live in
   `tests/eval/scripts/*.sh`. Agent shell never sees `tests/` during the
   work session (Harbor uploads it only at verify time). `instruction.md`
   lists labels for all `test_cmds` since native MLS-Bench's default
   (`hide_hidden=False`) returns hidden metrics to the agent.

## Where files come from

| Adapter file | Source |
|---|---|
| `task.toml.j2` → `task.toml` | per-task budgets from `tasks/<t>/config.json::test_cmds[].time`, package + data resource lookup |
| `instruction.md.j2` → `instruction.md` | `tasks/<t>/task_description.md` + every baseline's editable region as read-only reference (mirrors native `base.py:425-498`) |
| `environment/Dockerfile.j2` | 1-line `FROM` + scaffold COPY |
| `environment/_scaffold/` | `tasks/<t>/edits/mid_edit.py` (create + replace + insert + delete ops) applied to materialized package source |
| `solution/solve.sh.j2` | Strongest baseline from `tasks/<t>/leaderboard.csv`, replays its `edit_ops` from `tasks/<t>/edits/<name>.edit.py` |
| `tests/meta/*` | per-task verifier metadata (parser.py, score_spec.py, leaderboard.csv, budget_check.py if present) |
| `tests/eval/scripts/*` | every `tasks/<t>/scripts/*.sh` |
| `tests/mlsbench_src/` | `src/mlsbench/` Python tree (verifier-only, never in image) |
| `tests/meta/pristine/` | declared-file bytes from materialized package source |
| `tests/meta/pristine_manifest.json` | sha256 of every file under any guarded prefix |

## Security posture

| Threat | Mitigation |
|---|---|
| Agent reads eval script content | Scripts live in `tests/`, uploaded at verify time only |
| Agent edits non-editable lines | Content-based guard in `score_task.py::cmd_guard` |
| Agent overwrites pristine baseline | Pristine lives in `tests/`, not in image |
| Agent shadows `python3` via PATH | `test.sh` resets PATH + unsets `PYTHON*`/`LD_*` envs + invokes `${PYTHON_BIN}` with `-I` (isolated) |
| Agent plants `sitecustomize.py` | `PYTHONNOUSERSITE=1` + `-I` flag |
| Agent replaces `/usr/bin/python3` | Logged via sha256 audit to `/logs/verifier/python_audit.txt`; cannot prevent without Harbor `[verifier_environment]` feature |
| Hostile config.json points eval at `../../workspace/payload.sh` | `_safe_join` rejects path traversal in `cmd_run_evals` |

Same caveat as PostTrainBench's adapter (PR #8): a root agent that
`pip install`s a tampered package the verifier imports is currently not
defeatable. Tracked upstream as Harbor's `[verifier_environment]`
feature.

## Build progress (as of last update)

Image build:`harbor_adapter/scripts/build_base_image.py --all --push`
is running in the background, building all 65 per-package harbor base
images. Current state:

- ✅ **Pushed to Hub**: 10 confirmed in log
  (`alphaflow-main`, `badge`, `basicts`, `causal-bnlearn`,
  `causal-learn`, `chatdev-macnet`, `chebnetii`, `cleandiffuser`,
  `cleanrl`, `climax`)
- ⚠️ **Push reported error but manifest exists on Hub** (probably retry
  succeeded): `cfgpp-main`, `climsim`, `continual-learning`, `corl`
- ⏳ **Currently building**: `dbim-codebase` (463 GB data layer — bottleneck;
  likely needs `data_deps: []` config fix like we did for Time-Series-Library)
- ⌛ **Queued**: ~50 packages

Dataset render: validated end-to-end against the live build at the time
of validation. After fresh build completes, re-render with:

```
cd harbor_adapter && PYTHONPATH=src python3 -m mls_bench.main \
  --output-dir datasets/mls-bench \
  --mls-bench-root <mls-bench-checkout> --overwrite
```

Expect `Generated 140/140 tasks; 0 failed`.

## Open questions for review

1. **Should the agent be able to run eval mid-session?**
   PostTrainBench bakes `evaluate.py` into the agent's workspace and
   lets it iterate. Frontier-CS drops iteration entirely (single-shot
   `.cpp` submission). MLS-Bench's native `WorkspaceTools.test()` is
   somewhere between. Current adapter behavior: **agent cannot run any
   eval during its session** (eval scripts only exist in `/tests/`
   which is uploaded at verify time). This is stricter than native MLS.

2. **`dbim-codebase` 463 GB data layer**. Same shape as the
   Time-Series-Library issue we already fixed (data already baked into
   the upstream `mlsbench-<pkg>:latest` image, no separate
   `data_deps[]` needed for the harbor layer). Suggested fix: empty
   `data_deps` in `vendor/pkg_configs/dbim-codebase/config.json`.

3. **Multi-package tasks** (`tasks/llm-pretrain-attention/` etc.) —
   each task currently picks ONE primary package for the image
   (`test_cmds[0].package`). Secondary packages are staged into the
   scaffold but not their full source. For tasks that genuinely need
   multiple package envs simultaneously, this may need richer handling.

4. **Hidden test_cmds visibility in `instruction.md`** — currently lists
   labels for ALL test_cmds since native default (`hide_hidden=False`)
   exposes them. If we want to operate under `hide_hidden=True`
   semantics for Harbor, adapter needs a flag.

## Files to read first

- `harbor_adapter/README.md` — short overview + critical invariants
- `harbor_adapter/src/mls_bench/adapter.py::render_task` — entry point
  for per-task rendering (line ~702)
- `harbor_adapter/src/mls_bench/task-template/tests/score_task.py::cmd_guard`
  — the edit-range guard
- `harbor_adapter/scripts/build_base_image.py::build_one` — per-package
  harbor base build

## Validation done so far

- AST OK across every `.py` under `harbor_adapter/`
- Full render: 140 / 140 tasks, 0 failures
- E2E with real Harbor binary (v0.6.6):
  - Oracle on `causal-observational-linear-gaussian` → reward 0.026,
    metrics byte-match `baseline:pc` leaderboard row
  - Hostile out-of-range solve.sh → reward 0, violation file populated
- Codex review against the diff: 9 substantive findings (1 critical,
  3 high, 5 medium), all addressed in this branch.
