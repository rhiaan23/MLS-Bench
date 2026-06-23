# Strange Loop ↔ MLS-Bench run contract

This fork is driven by the Strange Loop campaign engine. Each **climb = one
MLS-Bench task**; each **run** proposes one algorithmic change and is scored by
the **official MLS-Bench Harbor verifier** — never by the agent itself.

## What a run may edit

For task `<task_id>`, the ONLY editable surface is the file + line range declared
in `harbor/tasks/mls-bench__<task_id>/tests/meta/config.json` → `files[]`
(`filename` + `edit: [{start,end}]`, 1-indexed inclusive). Editing anything
outside that range — or any file under `tests/` — makes the submission invalid
(the verifier's `guard` step zeroes the reward).

Example (`dl-activation-function`): edit only
`pytorch-vision/custom_activation.py` **lines 32–49** (the `CustomActivation`
class). That file is also the training entrypoint the eval scripts run.

## How a run is scored

Run, from the repo root, after editing:

```bash
bash tools/sl_climb_entrypoint.sh <task_id>
```

This reconstructs the Harbor layout (`/workspace`, datasets under `/data`,
`tests/`), runs the unmodified official verifier
(`tests/score_task.py guard → run-evals → score`), and writes `result.json`:

```json
{"objective": <combined_score in [0,1]>, "metrics": {"combined_score": ..., "<per-setting>": ...}}
```

`objective` is the official normalized `combined_score` (e.g. the gmean over the
task's settings, including HIDDEN eval settings the agent never sees). Maximize it.

## Do NOT

- Do not edit anything under `tests/` (scorer + hidden evals); it is re-checked
  out from the pinned ref before scoring.
- Do not hand-write `result.json`; only `sl_climb_entrypoint.sh` writes it.
- Do not change the eval scripts, optimizer, data pipeline, or training loop.
