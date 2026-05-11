"""Package-level patches for CleanDiffuser.

The container pins typing_extensions<4.6.0 (required for mujoco_py's Cython
compile), which breaks wandb's top-level import. The diffusion pipelines
import wandb at module load time via pipelines/utils.py, so we replace the
two top-level imports with lazy proxies — wandb is only actually loaded
when an attribute is accessed (i.e. when a user enables --use_wandb).
"""

OPS = [
    {
        "op": "replace",
        "file": "CleanDiffuser/pipelines/utils.py",
        "start_line": 6,
        "end_line": 7,
        "content": (
            "# wandb imports deferred — container pins typing_extensions<4.6 "
            "for mujoco_py; wandb 0.18+ wants newer.\n"
            "class _LazyWandb:\n"
            "    _w = None\n"
            "    def __getattr__(self, name):\n"
            "        if _LazyWandb._w is None:\n"
            "            import wandb as _w\n"
            "            _LazyWandb._w = _w\n"
            "        return getattr(_LazyWandb._w, name)\n"
            "class _LazyWv:\n"
            "    def __getattr__(self, name):\n"
            "        import wandb.sdk.data_types.video as _wv\n"
            "        return getattr(_wv, name)\n"
            "wandb = _LazyWandb()\n"
            "wv = _LazyWv()\n"
        ),
    },
]
