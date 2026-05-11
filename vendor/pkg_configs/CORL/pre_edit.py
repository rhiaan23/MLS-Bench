"""Pre-edit operations for the CORL package.

Injects TRAIN_METRICS print statements after wandb.log() in ALL baseline
algorithm files so that training metrics appear on stdout when wandb is
disabled.  Covers algorithms used across all CORL-based tasks
(continuous-control, adroit, offline-to-online).
"""

# ── TRAIN_METRICS snippets ───────────────────────────────────────────

# Standard snippet (log_dict variable)
_TRAIN_METRICS_SNIPPET = (
    '        if (t + 1) % 1000 == 0:\n'
    '            metrics_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in log_dict.items())\n'
    '            print(f"TRAIN_METRICS step={t+1} {metrics_str}", flush=True)'
)

# Variant for awac.py which uses update_result instead of log_dict
_TRAIN_METRICS_SNIPPET_AWAC = (
    '        if (t + 1) % 1000 == 0:\n'
    '            metrics_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in update_result.items())\n'
    '            print(f"TRAIN_METRICS step={t+1} {metrics_str}", flush=True)'
)

# ReBRAC (JAX-based) uses epoch-based logging with mean_metrics dict
_TRAIN_METRICS_SNIPPET_REBRAC = (
    '        metrics_str = " ".join(f"{k}={float(v):.4f}" for k, v in mean_metrics.items())\n'
    '        print(f"TRAIN_METRICS epoch={epoch} {metrics_str}", flush=True)'
)

# ── Pre-edit operations ──────────────────────────────────────────────

OPS = [
    # --- algorithms/offline/ (used by continuous-control and adroit tasks) ---
    # cql.py (line 946)
    {"op": "insert", "file": "CORL/algorithms/offline/cql.py",
     "after_line": 946, "content": _TRAIN_METRICS_SNIPPET},
    # iql.py (line 635)
    {"op": "insert", "file": "CORL/algorithms/offline/iql.py",
     "after_line": 635, "content": _TRAIN_METRICS_SNIPPET},
    # td3_bc.py (line 499)
    {"op": "insert", "file": "CORL/algorithms/offline/td3_bc.py",
     "after_line": 499, "content": _TRAIN_METRICS_SNIPPET},
    # any_percent_bc.py (line 386)
    {"op": "insert", "file": "CORL/algorithms/offline/any_percent_bc.py",
     "after_line": 386, "content": _TRAIN_METRICS_SNIPPET},
    # awac.py (line 481)
    {"op": "insert", "file": "CORL/algorithms/offline/awac.py",
     "after_line": 481, "content": _TRAIN_METRICS_SNIPPET_AWAC},
    # rebrac.py (line 737)
    {"op": "insert", "file": "CORL/algorithms/offline/rebrac.py",
     "after_line": 737, "content": _TRAIN_METRICS_SNIPPET_REBRAC},

    # --- algorithms/finetune/ (used by offline-to-online task) ---
    # finetune/awac.py (line 594)
    {"op": "insert", "file": "CORL/algorithms/finetune/awac.py",
     "after_line": 594, "content": _TRAIN_METRICS_SNIPPET_AWAC},
    # finetune/spot.py (line 881)
    {"op": "insert", "file": "CORL/algorithms/finetune/spot.py",
     "after_line": 881, "content": _TRAIN_METRICS_SNIPPET},
    # finetune/cal_ql.py (line 1197)
    {"op": "insert", "file": "CORL/algorithms/finetune/cal_ql.py",
     "after_line": 1197, "content": _TRAIN_METRICS_SNIPPET},
    # finetune/iql.py (line 734)
    {"op": "insert", "file": "CORL/algorithms/finetune/iql.py",
     "after_line": 734, "content": _TRAIN_METRICS_SNIPPET},
]
