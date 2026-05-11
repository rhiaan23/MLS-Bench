"""Pre-edit operations for the tdmpc2 package.

Applied to the codebase before any task's mid_edit:
1. Inject EVAL_METRIC print after evaluation logging in online_trainer.py
2. Inject TRAIN_METRICS print after training logging in online_trainer.py

Operations are ordered bottom-to-top to avoid line-number shifts.
"""

# ── TRAIN_METRICS: printed after each training episode ───────────────
_TRAIN_METRICS = (
    '\t\t\t\t\tprint(f"TRAIN_METRICS step={train_metrics[\'step\']}'
    ' episode_reward={float(train_metrics[\'episode_reward\']):.2f}"'
    ", flush=True)\n"
)

# ── EVAL_METRIC: printed after each evaluation ──────────────────────
_EVAL_METRIC = (
    '\t\t\t\t\tprint(f"EVAL_METRIC step={eval_metrics[\'step\']}'
    ' episode_reward={eval_metrics[\'episode_reward\']:.2f}"'
    ", flush=True)\n"
)

OPS = [
    # Insert TRAIN_METRICS after line 100 (self.logger.log(train_metrics, 'train'))
    {
        "op": "insert",
        "file": "tdmpc2/tdmpc2/trainer/online_trainer.py",
        "after_line": 100,
        "content": _TRAIN_METRICS,
    },
    # Insert EVAL_METRIC after line 87 (self.logger.log(eval_metrics, 'eval'))
    {
        "op": "insert",
        "file": "tdmpc2/tdmpc2/trainer/online_trainer.py",
        "after_line": 87,
        "content": _EVAL_METRIC,
    },
]
