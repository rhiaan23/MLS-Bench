#!/usr/bin/env python3
"""Test: does the cmd_guard patch (a) stop false-zeroing on agent scratch files,
and (b) still catch real tampering? Runs the REAL GitHub guard (orig) vs the
patched guard via subprocess on identical fixtures, exactly as test.sh invokes it:

    score_task.py guard --task-meta META --pristine META/pristine \
        --workspace WS --violation-out VIOL

rc 10 => violation => reward 0 (eval skipped). rc 0 => pass (eval runs).
"""
import json, os, hashlib, subprocess, shutil, sys, tempfile
from pathlib import Path

ORIG = "/tmp/guardfix/score_task_orig.py"
PATCHED = "/tmp/guardfix/score_task_patched.py"

# ---- pristine package source (the "fixed" baseline) ----
PRISTINE = {
    "pkg/model.py": "# FIXED model definition\nclass Model:\n    def forward(self, x):\n        return x\n",
    "pkg/custom.py": (
        "import torch\n"            # line 1 (fixed)
        "# EDITABLE REGION START\n"  # line 2 (fixed)
        "def custom(x):\n"           # line 3 (editable)
        "    return x  # baseline\n" # line 4 (editable)
        "# EDITABLE REGION END\n"    # line 5 (fixed)  -> editable range = [3,4]
    ),
}
CONFIG = {
    "allow_create": False,
    "files": [{"filename": "pkg/custom.py", "edit": [{"start": 3, "end": 4}]}],
}

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def build_meta(root: Path):
    meta = root / "meta"
    (meta / "pristine").mkdir(parents=True)
    for rel, txt in PRISTINE.items():
        p = meta / "pristine" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    (meta / "config.json").write_text(json.dumps(CONFIG))
    manifest = {rel: sha(txt.encode()) for rel, txt in PRISTINE.items()}
    (meta / "pristine_manifest.json").write_text(json.dumps(manifest))
    return meta

def build_ws(root: Path, files: dict):
    ws = root / "ws"
    for rel, txt in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    return ws

def run_guard(script: str, meta: Path, ws: Path):
    viol = ws.parent / "violation.txt"
    if viol.exists():
        viol.unlink()
    r = subprocess.run(
        [sys.executable, "-I", script, "guard",
         "--task-meta", str(meta), "--pristine", str(meta / "pristine"),
         "--workspace", str(ws), "--violation-out", str(viol)],
        capture_output=True, text=True,
    )
    msg = viol.read_text().strip() if viol.exists() else ""
    return r.returncode, msg

# valid in-range edit (only lines 3-4 changed)
VALID_EDIT = (
    "import torch\n# EDITABLE REGION START\n"
    "def custom(x):\n    return torch.relu(x) * 1.5  # my solution\n"
    "# EDITABLE REGION END\n"
)
# out-of-range edit (line 1 import changed = fixed line)
OOR_EDIT = (
    "import torch as T  # CHANGED FIXED LINE\n# EDITABLE REGION START\n"
    "def custom(x):\n    return x\n# EDITABLE REGION END\n"
)

SCENARIOS = {
    "S1 scratch-file (THE BUG)": {
        "pkg/model.py": PRISTINE["pkg/model.py"],
        "pkg/custom.py": VALID_EDIT,
        "pkg/test_scratch.py": "# agent experiment\nprint('try idea 1')\n",
        "pkg/custom.py.bak": "backup\n",
    },
    "S2 modify-fixed-file (tamper)": {
        "pkg/model.py": "# HACKED model\nclass Model:\n    def forward(self,x):\n        return 999\n",
        "pkg/custom.py": VALID_EDIT,
    },
    "S3 out-of-range edit (tamper)": {
        "pkg/model.py": PRISTINE["pkg/model.py"],
        "pkg/custom.py": OOR_EDIT,
    },
    "S4 clean in-range edit": {
        "pkg/model.py": PRISTINE["pkg/model.py"],
        "pkg/custom.py": VALID_EDIT,
    },
    "S5 delete fixed file (tamper)": {
        # model.py intentionally absent
        "pkg/custom.py": VALID_EDIT,
    },
}

print(f"{'scenario':32s} | {'ORIG rc':>8} | {'PATCHED rc':>11} | verdict")
print("-" * 86)
EXPECT = {  # (orig_rc, patched_rc) expected
    "S1 scratch-file (THE BUG)": (10, 0),
    "S2 modify-fixed-file (tamper)": (10, 10),
    "S3 out-of-range edit (tamper)": (10, 10),
    "S4 clean in-range edit": (0, 0),
    "S5 delete fixed file (tamper)": (10, 10),
}
allok = True
for name, files in SCENARIOS.items():
    o_rc = p_rc = None
    for script, tag in ((ORIG, "o"), (PATCHED, "p")):
        d = Path(tempfile.mkdtemp())
        meta = build_meta(d)
        ws = build_ws(d, files)
        rc, msg = run_guard(script, meta, ws)
        if tag == "o": o_rc, o_msg = rc, msg
        else:
            p_rc, p_msg = rc, msg
            # for S1, also confirm scratch was removed by the patch
            removed = not (ws / "pkg/test_scratch.py").exists()
        shutil.rmtree(d, ignore_errors=True)
    exp = EXPECT[name]
    ok = (o_rc, p_rc) == exp
    allok &= ok
    extra = ""
    if name.startswith("S1"):
        extra = f"  (patch removed scratch: {removed})"
    print(f"{name:32s} | {o_rc:>8} | {p_rc:>11} | {'OK' if ok else 'MISMATCH exp='+str(exp)}{extra}")

print("-" * 86)
print("ALL EXPECTED:", allok)
print()
print("Interpretation: rc 10 -> reward.txt=0 (eval skipped); rc 0 -> eval runs.")
