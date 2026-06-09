"""Pre-edit operations for robomimic package.

1. Skip overwrite prompt in train_utils.py (concurrent seeds race).
2. Inject TRAIN_METRICS output into train.py for parser consumption.
3. Inject TEST_METRICS output at end of training with best success rate.
4. Route AutoTokenizer in lang_utils.py through the same cache_dir the
   CLIPTextModelWithProjection load uses ($HF_HOME/clip). The upstream
   code passes cache_dir only for the model and falls through to the
   default ($HF_HOME/hub) for the tokenizer, which means a baked
   snapshot at HF_HOME/clip satisfies the model load but the tokenizer
   load goes online — fatal under HF_HUB_OFFLINE.

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

# ── lang_utils.py: thread cache_dir into AutoTokenizer ───────────────
# Replace line 10 so the tokenizer load also looks under HF_HOME/clip.
_LANG_TOKENIZER_CACHE = """\
tz = AutoTokenizer.from_pretrained(
    tokenizer,
    TOKENIZERS_PARALLELISM=True,
    cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip")),
)
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
    # 3. AutoTokenizer cache_dir alignment in lang_utils.py
    {
        "op": "replace",
        "file": "robomimic/robomimic/utils/lang_utils.py",
        "start_line": 10,
        "end_line": 10,
        "content": _LANG_TOKENIZER_CACHE,
    },
]
