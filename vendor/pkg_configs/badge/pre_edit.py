"""Pre-edit operations for badge package.

1. Replaces strategy.py with CPU-compatible version (removes .cuda() calls)
2. Registers the custom strategy in __init__.py
"""

from pathlib import Path

# Read original strategy.py and patch for CPU compatibility
_ORIG = Path(__file__).resolve().parent.parent.parent.parent / "vendor" / "external_packages" / "badge" / "query_strategies" / "strategy.py"
if _ORIG.exists():
    _CONTENT = _ORIG.read_text()
    _CONTENT = _CONTENT.replace(".cuda()", "")
    _CONTENT = _CONTENT.replace("Variable(x)", "x")
    _CONTENT = _CONTENT.replace("Variable(y)", "y")
    _LINE_COUNT = len(_CONTENT.splitlines())
else:
    _CONTENT = ""
    _LINE_COUNT = 1

OPS = [
    # Register CustomSampling in query_strategies/__init__.py
    {
        "op": "insert",
        "file": "badge/query_strategies/__init__.py",
        "after_line": 17,
        "content": "from .custom_sampling import CustomSampling\n",
    },
    # Replace entire strategy.py with CPU-compatible version
    {
        "op": "replace",
        "file": "badge/query_strategies/strategy.py",
        "start_line": 1,
        "end_line": _LINE_COUNT,
        "content": _CONTENT,
    },
    # Apptainer requires bind-mount destinations to pre-exist inside the target
    # filesystem. The OpenML cache is bind-mounted at /workspace/badge/oml.
    # Since the workspace `badge/` dir is itself bind-mounted into the
    # container, creating the empty `oml/` subdir in the workspace makes the
    # apptainer mount succeed. Docker silently creates missing destinations,
    # so this is a harmless no-op there. The "create" op auto-mkdirs parents.
    {
        "op": "create",
        "file": "badge/oml/.placeholder",
        "content": "# Placeholder so apptainer can bind-mount /workspace/badge/oml.\n",
    },
]
