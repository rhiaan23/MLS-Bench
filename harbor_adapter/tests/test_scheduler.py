from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_score_task():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "score_task.py"
    )
    spec = importlib.util.spec_from_file_location("score_task_scheduler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePopen:
    records: list[dict] = []
    next_pid = 1000

    def __init__(
        self,
        cmd,
        *,
        cwd=None,
        env=None,
        stdout=None,
        stderr=None,
        start_new_session=False,
    ):
        assert start_new_session is True
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or {}
        self.stdout = stdout
        self.stderr = stderr
        self.pid = FakePopen.next_pid
        FakePopen.next_pid += 1
        self.returncode = None
        FakePopen.records.append({
            "event": "start",
            "cmd": cmd,
            "label": self.env.get("ENV"),
            "seed": self.env.get("SEED"),
            "cuda": self.env.get("CUDA_VISIBLE_DEVICES"),
        })

    def poll(self):
        FakePopen.records.append({
            "event": "poll",
            "label": self.env.get("ENV"),
            "seed": self.env.get("SEED"),
        })
        self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _write_task(tmp_path: Path, config: dict, gpu_count: int):
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    task_meta.mkdir()
    eval_root.mkdir()
    workspace.mkdir()
    (task_meta / "config.json").write_text(json.dumps(config))
    default_pkg = config["test_cmds"][0].get("package", "pkg")
    (task_meta / "package").write_text(default_pkg + "\n")
    (task_meta / "task_id").write_text("scheduler-task\n")
    (task_meta / "gpu_count").write_text(f"{gpu_count}\n")

    packages = {default_pkg}
    for tc in config["test_cmds"]:
        script = eval_root / tc["cmd"]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("printf 'ok\\n'\n")
        packages.add(tc.get("package", default_pkg))
    for package in packages:
        (workspace / package).mkdir(parents=True, exist_ok=True)

    return argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(tmp_path / "out"),
    )


def _install_fake_popen(monkeypatch, score_task):
    FakePopen.records = []
    FakePopen.next_pid = 1000
    monkeypatch.setattr(score_task.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")


def _start_records():
    return [record for record in FakePopen.records if record["event"] == "start"]


def _start_batches():
    batches = []
    current = []
    for record in FakePopen.records:
        if record["event"] == "start":
            current.append(record)
        elif record["event"] == "poll" and current:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def test_tdmpc2_group_runs_three_parallel_launches_per_seed(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/walker.sh", "label": "walker", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "tdmpc2"},
                {"cmd": "scripts/cheetah.sh", "label": "cheetah", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "tdmpc2"},
                {"cmd": "scripts/cartpole.sh", "label": "cartpole", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "tdmpc2", "hidden": True},
            ],
            "seeds": [42, 123, 456],
        },
        gpu_count=1,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert len(starts) == 9
    assert {record["cuda"] for record in starts} == {"0"}
    batches = _start_batches()
    assert [len(batch) for batch in batches] == [3, 3, 3]
    assert [[record["seed"] for record in batch] for batch in batches] == [
        ["42", "42", "42"],
        ["123", "123", "123"],
        ["456", "456", "456"],
    ]


def test_llm_pretrain_attention_runs_groups_sequentially(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/gpt.sh", "label": "gpt-345m", "group": 1, "compute": 4.0, "time": "0:01:00", "package": "nanoGPT"},
                {"cmd": "scripts/lm_eval.sh", "label": "lm-eval-345m", "group": 2, "compute": 1.0, "time": "0:01:00", "package": "lm-evaluation-harness", "hidden": True},
            ],
            "seeds": [42],
        },
        gpu_count=4,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert [(record["label"], record["cuda"]) for record in starts] == [
        ("gpt-345m", "0,1,2,3"),
        ("lm-eval-345m", "0"),
    ]
    first_lm_start = next(i for i, record in enumerate(FakePopen.records) if record.get("label") == "lm-eval-345m" and record["event"] == "start")
    assert any(record["event"] == "poll" and record.get("label") == "gpt-345m" for record in FakePopen.records[:first_lm_start])


def test_ai4sci_fractional_group_packs_onto_two_gpus(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/bbbp.sh", "label": "BBBP", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "Uni-Mol"},
                {"cmd": "scripts/bace.sh", "label": "BACE", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "Uni-Mol"},
                {"cmd": "scripts/tox21.sh", "label": "Tox21", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "Uni-Mol", "hidden": True},
            ],
            "seeds": [42],
        },
        gpu_count=2,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert len(starts) == 3
    assert [len(batch) for batch in _start_batches()] == [3]
    cuda_values = [record["cuda"] for record in starts]
    assert set(cuda_values) == {"0", "1"}


def test_cv_3dgs_separate_groups_run_sequentially(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/garden.sh", "label": "garden", "group": 1, "compute": 1.0, "time": "0:01:00", "package": "gsplat"},
                {"cmd": "scripts/bicycle.sh", "label": "bicycle", "group": 2, "compute": 1.0, "time": "0:01:00", "package": "gsplat"},
                {"cmd": "scripts/bonsai.sh", "label": "bonsai", "group": 3, "compute": 1.0, "time": "0:01:00", "package": "gsplat"},
                {"cmd": "scripts/stump.sh", "label": "stump", "group": 4, "compute": 1.0, "time": "0:01:00", "package": "gsplat", "hidden": True},
            ],
            "seeds": [42],
        },
        gpu_count=1,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert [(record["label"], record["cuda"]) for record in starts] == [
        ("garden", "0"),
        ("bicycle", "0"),
        ("bonsai", "0"),
        ("stump", "0"),
    ]
    assert [len(batch) for batch in _start_batches()] == [1, 1, 1, 1]
