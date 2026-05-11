"""Pre-edit operations for robomimic package.

1. Skip overwrite prompt in train_utils.py (concurrent seeds race).
2. Inject TRAIN_METRICS output into train.py for parser consumption.
3. Inject TEST_METRICS output at end of training with best success rate.

Ops are ordered bottom-to-top within each file so line numbers stay stable.
"""

# ── train.py: TEST_METRICS at end of training ──────────────────────────
# Insert after line 462 (data_logger.close()) to print final success rate.
_TEST_METRICS = """\

    # MLS-Bench: print final best success rate
    if best_success_rate is not None:
        _best_sr = max(best_success_rate.values()) if best_success_rate else 0.0
        print("TEST_METRICS success_rate={:.6f}".format(_best_sr), flush=True)
"""

# ── train.py: TRAIN_METRICS per epoch ──────────────────────────────────
# Insert after line 325 (print(json.dumps(step_log, ...))) to print loss.
_TRAIN_METRICS = """\
        if "Loss" in step_log:
            print("TRAIN_METRICS epoch={} train_loss={:.6f}".format(epoch, step_log["Loss"]), flush=True)
"""

# ── train_utils.py: skip overwrite prompt entirely ───────────────────
# Replace lines 66-73 (the entire overwrite check) with pass.
# Concurrent seeds in the same SLURM job race on rmtree; just skip.
_SKIP_OVERWRITE = """\
    elif os.path.exists(base_output_dir):
        pass  # MLS-Bench: skip overwrite prompt (concurrent seeds)
"""

OPS = [
    # 0. Skip overwrite block in train_utils.py (replace lines 66-73)
    {
        "op": "replace",
        "file": "robomimic/robomimic/utils/train_utils.py",
        "start_line": 66,
        "end_line": 73,
        "content": _SKIP_OVERWRITE,
    },
    # 1. TEST_METRICS in train.py (after line 462) — inserted FIRST (bottom-to-top)
    {
        "op": "insert",
        "file": "robomimic/robomimic/scripts/train.py",
        "after_line": 462,
        "content": _TEST_METRICS,
    },
    # 2. TRAIN_METRICS in train.py (after line 325) — inserted SECOND
    {
        "op": "insert",
        "file": "robomimic/robomimic/scripts/train.py",
        "after_line": 325,
        "content": _TRAIN_METRICS,
    },
]
