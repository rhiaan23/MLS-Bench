"""Pre-edit operations for InverseBench package.
Injects TRAIN_METRICS and TEST_METRICS printing into main.py.
Patches dataset.py to handle optional sigpy import.
Fixes eval.Image metric_state accumulation bug (list += float TypeError).

Line references (original file):
- training/dataset.py line 5: import sigpy as sp
- eval.py lines 175, 179: self.metric_state[metric_name] += ...
- main.py line 115: logger.info(f"Metric results: {metric_dict}...")
- main.py line 120: logger.info(f"Final metric results: {metric_state}...")

Operations ordered: different files first, then bottom-to-top within same file.
"""

OPS = [
    # 1. Patch dataset.py line 5: make sigpy import optional (replace 1 line with 4)
    {
        "op": "replace",
        "file": "InverseBench/training/dataset.py",
        "start_line": 5,
        "end_line": 5,
        "content": "try:\n    import sigpy as sp\nexcept ImportError:\n    sp = None\n",
    },
    # 2. Fix eval.py line 179: metric_state is a list (init in base Evaluator),
    #    so `list += float` raises TypeError. Use .append() instead.
    #    Patch bottom-to-top within eval.py to avoid line shifting.
    {
        "op": "replace",
        "file": "InverseBench/eval.py",
        "start_line": 179,
        "end_line": 179,
        "content": "                self.metric_state[metric_name].append(val)\n",
    },
    # 3. Fix eval.py line 175: metric_dict[metric_name] is a tensor scalar here
    #    (accumulated from metric_func(...).sum()), so cast via .item() before append.
    {
        "op": "replace",
        "file": "InverseBench/eval.py",
        "start_line": 175,
        "end_line": 175,
        "content": "                _v = metric_dict[metric_name]\n                self.metric_state[metric_name].append(_v.item() if hasattr(_v, 'item') else float(_v))\n",
    },
    # 4. Inject TEST_METRICS after main.py line 120 (before TRAIN_METRICS to avoid shifting)
    {
        "op": "insert",
        "file": "InverseBench/main.py",
        "after_line": 120,
        "content": '    for _k, _v in metric_state.items():\n        print(f"TEST_METRICS {_k}={_v}", flush=True)\n',
    },
    # 5. Inject TRAIN_METRICS after main.py line 115
    {
        "op": "insert",
        "file": "InverseBench/main.py",
        "after_line": 115,
        "content": '        _metric_parts = " ".join(f"{k}={v}" for k, v in metric_dict.items())\n        print(f"TRAIN_METRICS sample={data_id} {_metric_parts}", flush=True)\n',
    },
]
