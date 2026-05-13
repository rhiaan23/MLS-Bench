# MLS-Bench → Harbor adapter

Converts the 140 [MLS-Bench](https://github.com/Bohan22/MLS-Bench) tasks into
[Harbor](https://github.com/harbor-framework/harbor) tasks so any Harbor agent
(`claude-code`, `openhands`, `codex`, `terminus-2`, …) can be evaluated against
the MLS-Bench problem set:

```bash
harbor run --dataset mls-bench@1.0 \
    --agent claude-code \
    --model anthropic/claude-opus-4-1 \
    --n-concurrent 4
```

## How it works

For each MLS-Bench task `tasks/<t>/` the adapter emits a Harbor task directory:

```
mls-bench__<t>/
├── task.toml                 # name, timeouts, cpus/memory/gpus
├── instruction.md            # task_description.md + editable-range list (visible labels only)
├── environment/
│   ├── Dockerfile            # FROM bohanlyu2022/mlsbench-<pkg>:latest + data + eval scripts staged
│   └── docker-compose.yaml   # only when use_cuda=true
├── solution/solve.sh         # oracle: applies the strongest baseline edit_ops
└── tests/
    ├── test.sh               # edit-range diff guard → run all eval scripts → score → reward
    └── score_task.py         # mlsbench.scoring.evaluate_task wrapper
```

### Critical invariants preserved

- **Editable line ranges** (`config.json::files[].edit`): enforced by `tests/test.sh`
  diffing the agent's workspace against the pristine baseline shipped at
  `tests/meta/pristine/<rel>` (full bytes for declared files) plus
  `tests/meta/pristine_manifest.json` (sha256 of every file under any
  guarded prefix). Harbor uploads `tests/` only at verify time so the agent
  cannot tamper with the baseline. Any modified line outside an allowed
  range → reward 0 + `/logs/verifier/violation.txt`.
- **`allow_create: false`**: new files in workspace → reward 0.
- **`budget_check.py`** (e.g. `llm-pretrain-normalization`): runs as part of
  the eval scripts; no extra wrapping.
- **Hidden test_cmds**: eval scripts (visible + hidden alike) live at
  `tests/eval/scripts/*.sh`, **outside** `/workspace` so a generic
  shell agent cannot read them — Harbor mounts `tests/` at `/tests/` only
  at verify time. `instruction.md` lists labels for all test_cmds; only the
  script *contents* are hidden, matching native MLS-Bench's default.

## CLI

```bash
uv run python -m mls_bench.main \
    --output-dir ./output \
    [--limit N] [--overwrite] [--task-ids t1,t2,…] \
    [--mls-bench-root /path/to/MLS-Bench]   # default: auto-detect from cwd
```

## Status

Initial draft — see `parity_experiment.json` (populated after parity runs).
