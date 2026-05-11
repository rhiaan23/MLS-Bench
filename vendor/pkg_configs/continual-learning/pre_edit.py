"""Pre-edit operations for continual-learning package.

Applied before any task-specific edits:
1. Fix data directory to use /data/continual-learning for pre-downloaded datasets
   (compute nodes have no network, data is bind-mounted from host)

Line numbers reference the original repo at commit e6d795a.
"""

OPS = [
    # Replace store path (line 6) FIRST — bottom-to-top order
    # Uses DATA_ROOT env var for local runtime, /data for container
    {
        "op": "replace",
        "file": "continual-learning/params/options.py",
        "start_line": 6,
        "end_line": 6,
        "content": 'store = os.path.join(os.environ.get("DATA_ROOT", "/data"), "continual-learning")\n',
    },
    # Insert 'import os' after line 1 (after 'import argparse') SECOND
    {
        "op": "insert",
        "file": "continual-learning/params/options.py",
        "after_line": 1,
        "content": "import os\n",
    },
]
