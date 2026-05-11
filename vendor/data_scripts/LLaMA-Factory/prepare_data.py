#!/usr/bin/env python3
"""Prepare LLaMA-Factory training data + math models.

Downloads and converts preference / SFT datasets into LLaMA-Factory sharegpt
format, and downloads the Qwen2.5-Math-1.5B-Instruct base model used by the
llm-offline-rl task. Outputs are bind-mounted into the container at
``/data/llama-factory-data`` and ``/models/Qwen2.5-Math-1.5B-Instruct``.

Reference paper alignment:
  Step-DPO (Lai et al. 2024, arXiv:2406.18629). We reuse their preference
  dataset (xinlai/Math-Step-DPO-10K, 10K math step-level pairs) and evaluate
  DPO variants on GSM8K, MATH-500, AIME-2024 — matching the paper's eval suite.

Produces:
  {data_root}/llama-factory-data/math_step_dpo.json    (Math-Step-DPO-10K, ~10K)
  {data_root}/llama-factory-data/metamathqa.json       (MetaMathQA SFT, 50K)
  {data_root}/models/Qwen2.5-Math-1.5B-Instruct        (~3 GB safetensors)
"""

import argparse
import json
from pathlib import Path


def prepare_math_step_dpo(out_dir: Path) -> None:
    """xinlai/Math-Step-DPO-10K → LLaMA-Factory sharegpt DPO format.

    Each sample's `prompt` is the math problem, `full_chosen` and `full_rejected`
    are full step-by-step solutions; we use them directly as response-level
    chosen / rejected pairs (compatible with all DPO/SimPO/IPO/ORPO variants).
    """
    import datasets

    out_path = out_dir / "math_step_dpo.json"
    if out_path.exists():
        print(f"  math_step_dpo: {out_path} exists, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("xinlai/Math-Step-DPO-10K", split="train")
    print(f"  math_step_dpo: downloaded {len(ds)} rows")

    data = []
    for row in ds:
        data.append({
            "conversations": [{"from": "human", "value": row["prompt"]}],
            "chosen": {"from": "gpt", "value": row["full_chosen"]},
            "rejected": {"from": "gpt", "value": row["full_rejected"]},
        })
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"  math_step_dpo: {len(data)} samples -> {out_path}")


def prepare_metamathqa(out_dir: Path) -> None:
    """MetaMathQA → LLaMA-Factory sharegpt SFT format (kept for llm-sft-loss)."""
    out_path = out_dir / "metamathqa.json"
    if out_path.exists():
        print(f"  metamathqa: {out_path} exists, skipping")
        return

    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("meta-math/MetaMathQA", split="train[:50000]")
    data = [
        {"conversations": [
            {"from": "human", "value": row["query"]},
            {"from": "gpt", "value": row["response"]},
        ]}
        for row in ds
    ]
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"  metamathqa: {len(data)} samples -> {out_path}")


def prepare_qwen_math_instruct(models_dir: Path) -> None:
    """Download Qwen2.5-Math-1.5B-Instruct (math-specialized SFT, ~3 GB)."""
    target = models_dir / "Qwen2.5-Math-1.5B-Instruct"
    sentinel = target / "model.safetensors"
    if sentinel.exists() or any(target.glob("model-*.safetensors")):
        print(f"  Qwen2.5-Math-1.5B-Instruct: {target} exists, skipping")
        return

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"  Qwen2.5-Math-1.5B-Instruct: downloading to {target}")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    print(f"  Qwen2.5-Math-1.5B-Instruct: ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    data_dir = root / "llama-factory-data"
    models_dir = root / "models"

    prepare_math_step_dpo(data_dir)
    prepare_metamathqa(data_dir)
    prepare_qwen_math_instruct(models_dir)

    print("\nAll LLaMA-Factory datasets + models ready.")


if __name__ == "__main__":
    main()
