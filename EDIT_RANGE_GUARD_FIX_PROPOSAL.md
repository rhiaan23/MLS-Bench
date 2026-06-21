# Proposal: stop the edit-range guard from hard-zeroing correct solutions that leave scratch files

**Component:** `tests/score_task.py :: cmd_guard` (the edit-range diff guard), rendered into every task's `tests/` dir by the Harbor adapter.
**Severity:** false negatives — a *correct* solution scores **0** (eval never runs) because the agent left a harmless file behind.
**Status:** fix implemented + tested below; ready to land via the adapter template.

---

## 1. Symptom

When an agent does some experimentation and leaves a by-product inside the workspace package dir — an experiment script, a `*.bak`, a `test_*.py`, a variant of its own solution — the run gets a hard **0**, regardless of how good the actual edit is. This was reported independently by a user and reproduced while evaluating Kimi-K2.7.

It hits **agents that scratch inside the package dir** (Kimi-code family is the prime example) far more than agents that scratch in `/tmp` (Opus / Codex in our data did not trigger it).

## 2. Root cause

`tests/test.sh` runs the guard **before** the eval and treats a guard violation as a terminal zero:

```sh
"${PYTHON_BIN}" -I score_task.py guard --task-meta ... --pristine ... --workspace ... --violation-out ...
guard_rc=$?
if [ "${guard_rc}" -eq 10 ]; then
    echo "0" > /logs/verifier/reward.txt    #  <-- HARD ZERO, eval is skipped
    exit 0
fi
```

`cmd_guard` returns `10` for **any** workspace file that is not in the render-time manifest, whenever `allow_create=false`:

```python
# score_task.py :: cmd_guard  (current)
if not allow_create:
    for rel in sorted(workspace_files):
        rel_str = rel.as_posix()
        if rel_str in manifest:
            continue
        violations.append(f"created new file (allow_create=false): {rel_str}")
...
if violations:
    violation_out.write_text("\n".join(violations) + "\n")
    return 10            # -> test.sh writes reward 0, never runs the eval
```

So **creating any new file** = instant 0. But creating a new file is **not the threat model** the guard exists for. The thing worth protecting is *modification* or *deletion* of the fixed baseline (the scorer, hidden tests, data, frozen model source) — and those are handled separately (sha-compare against the manifest, deletion check, and the per-file edit-range check). A brand-new `test_idea.py` cannot tamper with the score; at worst it could shadow-import a package module, which is trivially neutralized by removing it.

## 3. Evidence

In our Mangrove harness the guard is *skipped* (the per-task pristine/manifest isn't staged), so we can see what these same runs score **when the eval is actually allowed to run**. They are fine solutions — the guard would have zeroed every one of them purely for leftover files:

| Kimi run (task) | score with guard **skipped** | files the guard would flag → **0** |
|---|---|---|
| ml-clustering-algorithm | **0.580** | 18× `scikit-learn/test_*.py`, `compare.py` |
| ts-exogenous-forecast | **0.520** | 4× `Time-Series-Library/test_*.py` |
| llm-rl-importance-sampling | **0.472** | `verl/trainer/ppo/custom_policy_loss.py` |
| optimization-variance-reduction | **0.463** | `opt-vr-bench/custom_vr.py.my`, `…/.sarah` |
| optimization-multi-objective | **0.446** | `deap/custom_moea_orig.py` |
| ai4sci-pla-binding-affinity | **0.346** | `EHIGN_PLA/sanity_train.py` |
| llm-pretrain-optimizer | 0.286 / 0.039 | `nanoGPT/custom_pretrain_baseline.py`, `…_lookahead.py`, `…py.bak` |

Every affected task has `allow_create=false`. None of the leftover files touch the scorer/tests/data — they are the agent's own scratch.

## 4. The fix

A newly-created file is not tampering: **remove it before the eval** (so it can't shadow-import anything) and continue, instead of failing the whole run. Modification / deletion of baseline files and out-of-range edits are still hard failures — unchanged.

```diff
--- a/tests/score_task.py
+++ b/tests/score_task.py
@@ def cmd_guard(args):
-    # Disallowed creation: anything in workspace that is NOT in the manifest
-    # (= agent created it post-start).
+    # Newly-created files (present in the workspace, absent from the render-time
+    # manifest) are NOT tampering. The anti-cheat surface this guard protects is
+    # *modification* or *deletion* of the fixed baseline (scorer/tests/data/model
+    # source) — handled below. A brand-new file is, at worst, agent scratch
+    # (an experiment script, a `*.bak`, a `test_*.py`); hard-failing the whole
+    # run for it (reward 0, eval never runs) zeroes otherwise-correct solutions.
+    # Instead, REMOVE such files before the eval so they cannot influence it
+    # (e.g. shadow-import a package module), then continue. A task that
+    # legitimately needs the agent to author new files sets allow_create=true.
+    cleaned_created: list[str] = []
     if not allow_create:
         for rel in sorted(workspace_files):
             rel_str = rel.as_posix()
             if rel_str in manifest:
                 continue
-            violations.append(f"created new file (allow_create=false): {rel_str}")
+            try:
+                _safe_join(workspace_root, rel_str).unlink()
+                cleaned_created.append(rel_str)
+            except OSError:
+                pass
+        if cleaned_created:
+            workspace_files = {p for p in workspace_files
+                               if p.as_posix() not in set(cleaned_created)}
+            workspace_rel_strs = {p.as_posix() for p in workspace_files}
+            # Debug breadcrumb only — NOT a violation.
+            (violation_out.parent / "cleaned_created.txt").write_text(
+                "\n".join(cleaned_created) + "\n"
+            )
```

Notes:
- `allow_create=true` tasks are untouched (the whole block is under `if not allow_create`).
- `_walk_workspace` already excludes `__pycache__`, `*.pyc`, the mounted `_task` tree, and `guard_exclude` prefixes, so those are never deleted.
- The later sha-compare loop already `continue`s on files whose `manifest.get(rel_str) is None`, so removing them first changes nothing for modification/deletion detection.

## 5. Why it's safe — test matrix

Harness builds a minimal task (`allow_create=false`, one editable file with range `[3,4]`, a fixed `model.py` in the manifest) and runs the **real** guard (`orig`) vs the **patched** guard (`patched`) via subprocess, exactly as `test.sh` invokes it. `rc 10` ⇒ reward 0 / eval skipped; `rc 0` ⇒ eval runs.

| scenario | orig rc | patched rc | result |
|---|---|---|---|
| **S1** valid in-range edit **+ leftover `test_scratch.py` / `custom.py.bak`** | `10` | `0` (scratch removed) | **bug fixed** |
| **S2** modify a fixed manifest file (`model.py`) | `10` | `10` | still caught ✓ |
| **S3** edit a fixed line outside the editable range | `10` | `10` | still caught ✓ |
| **S4** clean in-range edit, nothing else | `0` | `0` | unchanged ✓ |
| **S5** delete a fixed manifest file | `10` | `10` | still caught ✓ |

Only S1 changes. Every real-tampering path is preserved. (Test script: `test_guard_fix.py`, included alongside this doc.)

## 6. How to land it (touches many files, but mechanically)

`tests/score_task.py` is **byte-identical across all 140 tasks** (verified: single md5 across the rendered tree) — it is copied verbatim by the adapter, with **no per-task templating**. So:

1. Apply the patch to the **adapter template**: `harbor_adapter/src/mls_bench/task-template/tests/score_task.py`.
2. Re-render **only `score_task.py`** into each task's `tests/` dir (it carries no per-task data). A full task re-render is **not** required and should be avoided here, because re-rendering `task_description.md` (the instruction) produces spurious diffs — **do not re-render instructions**. Because the file is verbatim, regeneration of this one file is equivalent to copying the patched template over every `tests/score_task.py`.

A one-liner equivalent (no adapter run needed) if preferred:
```sh
find harbor/tasks -path '*/tests/score_task.py' \
  -exec cp harbor_adapter/.../task-template/tests/score_task.py {} \;
```

## 7. Lighter alternatives (if removing files is unwanted)

- **Ignore instead of delete**: don't append a violation for created files and don't delete them. Simpler, but a created file could still shadow-import a package module during the eval; deletion closes that.
- **Scratch allowlist**: extend `guard_exclude` with common scratch patterns (`test_*`, `*.bak`, `*_baseline*`, `*.my`, …). Narrower, but agents invent new names; the create-is-not-tampering framing is more robust.

The recommended fix (delete-then-continue) is the smallest change that both removes the false-zero and keeps the eval running against a clean tree.
