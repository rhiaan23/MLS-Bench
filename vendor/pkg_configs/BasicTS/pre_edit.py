"""Pre-edit operations for the BasicTS package.
Injects TRAIN_METRICS print statements into the runner so that
training metrics appear on stdout for parser consumption.
Also injects parseable test metrics output.
"""

# After line 676 (self.print_meters("train")) — inject TRAIN_METRICS per epoch
_TRAIN_METRICS = (
    '        try:\n'
    '            _loss = self.meter_pool.get_value("train/loss")\n'
    '            print(f"TRAIN_METRICS epoch={epoch} train_loss={_loss:.7f}", flush=True)\n'
    '        except Exception:\n'
    '            pass'
)

# After line 611 (self.print_meters("test")) — inject parseable test metrics
_TEST_METRICS = (
    '        try:\n'
    '            _mae = self.meter_pool.get_value("test/MAE")\n'
    '            _rmse = self.meter_pool.get_value("test/RMSE")\n'
    '            _mape = self.meter_pool.get_value("test/MAPE")\n'
    '            print(f"mae:{_mae:.6f},rmse:{_rmse:.6f},mape:{_mape:.6f}", flush=True)\n'
    '        except Exception:\n'
    '            pass'
)

OPS = [
    # Fix torch.Tensor(scalar) compatibility with PyTorch 2.5+ (line 63 of z_score_scaler.py)
    {
        "op": "replace",
        "file": "BasicTS/src/basicts/scaler/z_score_scaler.py",
        "start_line": 63,
        "end_line": 63,
        "content": "            self.stats['mean'], self.stats['std'] = torch.tensor(mean).float(), torch.tensor(std).float()",
    },
    # Test metrics (insert after line 611 — self.print_meters("test"))
    {
        "op": "insert",
        "file": "BasicTS/src/basicts/runners/basicts_runner.py",
        "after_line": 611,
        "content": _TEST_METRICS,
    },
    # Train metrics (insert after line 676, shifted by 7 lines from above insert → 683)
    {
        "op": "insert",
        "file": "BasicTS/src/basicts/runners/basicts_runner.py",
        "after_line": 683,
        "content": _TRAIN_METRICS,
    },
]
