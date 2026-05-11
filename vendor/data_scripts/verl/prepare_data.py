#!/usr/bin/env python3
"""Prepare verl training and evaluation data.

Downloads and converts datasets into verl-compatible parquet format.
Run via: python -m mlsbench data verl --config <config>
Or directly: python vendor/data_scripts/verl/prepare_data.py --data-root vendor/data

Produces:
  {data_root}/verl-data/gsm8k/{train,test}.parquet
  {data_root}/verl-data/math500/test.parquet
  {data_root}/verl-data/aime2024/test.parquet
  {data_root}/verl-data/deepmath/train.parquet          (full 103K)
  {data_root}/verl-data/deepmath/train_lv3-5.parquet    (filtered ~30K)
  {data_root}/verl-data/deepmath/train_5k.parquet       (small train split)
  {data_root}/verl-data/metamath/train.parquet
"""

import argparse
import os
import subprocess
from pathlib import Path


BOXED_SUFFIX = "Please put your final answer within \\boxed{}."
REASON_BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def write_verl_parquet(rows, dst_parquet):
    """Write rows with the exact schema verl/datasets expects across files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([
            ("content", pa.string()),
            ("role", pa.string()),
        ]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([
            ("ground_truth", pa.string()),
            ("style", pa.string()),
        ])),
        ("extra_info", pa.struct([
            ("answer", pa.string()),
            ("index", pa.int64()),
            ("question", pa.string()),
            ("split", pa.string()),
        ])),
    ])
    os.makedirs(os.path.dirname(dst_parquet), exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, dst_parquet)


def mathruler_to_verl_parquet(src_jsonl, dst_parquet, data_source, split):
    """Convert MathRuler JSONL to verl parquet format."""
    import json

    rows = []
    with open(src_jsonl) as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            question = item.get("question") or item.get("problem", "")
            answer = item["answer"]
            prompt_text = question + "\n" + (REASON_BOXED_SUFFIX if data_source != "openai/gsm8k" else BOXED_SUFFIX)
            rows.append({
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "math",
                "reward_model": {"ground_truth": answer, "style": "rule"},
                "extra_info": {"answer": answer, "index": idx, "question": question, "split": split},
            })

    write_verl_parquet(rows, dst_parquet)
    print(f"  {data_source} {split}: {len(rows)} samples -> {dst_parquet}")


def prepare_deepmath(deepmath_dir: Path):
    """Download DeepMath-103K and create full + filtered + 5k-sample parquets.

    train_5k.parquet is a fixed seed=42 random subset of train_lv3-5.parquet
    used by llm-rl-advantage / llm-rl-importance-sampling for short ablations.
    """
    import datasets

    full_path = deepmath_dir / "train.parquet"
    filtered_path = deepmath_dir / "train_lv3-5.parquet"
    sample_5k_path = deepmath_dir / "train_5k.parquet"
    deepmath_dir.mkdir(parents=True, exist_ok=True)

    if full_path.exists() and filtered_path.exists() and sample_5k_path.exists():
        print(f"  deepmath: train.parquet, train_lv3-5.parquet, train_5k.parquet all exist, skipping")
        return

    # Fast path: lv3-5 already there, only need to sample train_5k from it.
    if full_path.exists() and filtered_path.exists() and not sample_5k_path.exists():
        import pandas as pd
        df_lv35 = pd.read_parquet(str(filtered_path))
        df_lv35.sample(n=5000, random_state=42).to_parquet(str(sample_5k_path), index=False)
        print(f"  deepmath 5k: 5000 samples -> {sample_5k_path}")
        return

    ds = datasets.load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"  deepmath: downloaded {len(ds)} rows")

    target_features = datasets.Features({
        "data_source": datasets.Value("string"),
        "prompt": [{"content": datasets.Value("string"), "role": datasets.Value("string")}],
        "ability": datasets.Value("string"),
        "reward_model": {"ground_truth": datasets.Value("string"), "style": datasets.Value("string")},
        "extra_info": {
            "answer": datasets.Value("string"),
            "index": datasets.Value("int64"),
            "question": datasets.Value("string"),
            "split": datasets.Value("string"),
        },
    })

    def to_verl_format(example, idx):
        prompt_text = example["question"] + "\n" + BOXED_SUFFIX
        return {
            "data_source": "deepmath",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"ground_truth": example["final_answer"], "style": "rule"},
            "extra_info": {
                "answer": example["final_answer"],
                "index": idx,
                "question": example["question"],
                "split": "train",
            },
        }

    # Full dataset
    ds_verl = ds.map(to_verl_format, with_indices=True, remove_columns=ds.column_names)
    ds_verl = ds_verl.cast(target_features)
    ds_verl.to_parquet(str(full_path))
    print(f"  deepmath full: {len(ds_verl)} samples -> {full_path}")

    # Filtered lv3-5
    ds_filtered = ds.filter(lambda x: 3.0 <= x["difficulty"] <= 5.0)
    ds_filt_verl = ds_filtered.map(to_verl_format, with_indices=True, remove_columns=ds_filtered.column_names)
    ds_filt_verl = ds_filt_verl.cast(target_features)
    ds_filt_verl.to_parquet(str(filtered_path))
    print(f"  deepmath lv3-5: {len(ds_filt_verl)} samples -> {filtered_path}")

    # 5k subset of train_lv3-5 with fixed seed=42 (referenced by
    # llm-rl-advantage/scripts/train.sh and llm-rl-importance-sampling/scripts/{train,train_1gpu}.sh).
    # NOTE: matches the existing artifact at vendor/data/verl-data/deepmath/train_5k.parquet —
    # using a different sampling strategy (e.g. table.slice(0, 5000)) would silently make
    # historical leaderboard rows non-reproducible.
    if not sample_5k_path.exists():
        import pandas as pd
        df_lv35 = pd.read_parquet(str(filtered_path))
        df_5k = df_lv35.sample(n=5000, random_state=42)
        df_5k.to_parquet(str(sample_5k_path), index=False)
        print(f"  deepmath 5k: {len(df_5k)} samples -> {sample_5k_path}")


def prepare_metamath(metamath_dir: Path):
    """Download MetaMathQA and save as parquet."""
    import datasets

    out_path = metamath_dir / "train.parquet"
    if out_path.exists():
        print(f"  metamath: already exists, skipping")
        return

    metamath_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("meta-math/MetaMathQA", split="train")
    ds.to_parquet(str(out_path))
    print(f"  MetaMathQA: {len(ds)} samples saved")


def prepare_amc(amc_dir: Path):
    """Download AI-MO/aimo-validation-amc (83 AMC 2022-2023 problems) as verl parquet.

    Schema from HF: {id:int64, problem:str, answer:float64, url:str}
    Converted to verl format with data_source='amc23' so our parser routes it.
    The math_dapo scorer handles numeric-answer problems; we route amc23 via
    pre_edit's reward_score/__init__.py patch.
    """
    import datasets

    out_path = amc_dir / "test.parquet"
    if out_path.exists():
        print(f"  amc: already exists, skipping")
        return

    amc_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")

    rows = []
    for idx, item in enumerate(ds):
        question = item["problem"]
        # answer is stored as float; AMC answers are integers
        answer = str(int(item["answer"])) if float(item["answer"]).is_integer() else str(item["answer"])
        prompt_text = question + "\n" + REASON_BOXED_SUFFIX
        rows.append({
            "data_source": "amc23",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {"answer": answer, "index": idx, "question": question, "split": "test"},
        })
    write_verl_parquet(rows, str(out_path))
    print(f"  amc: {len(rows)} samples -> {out_path}")


def prepare_simplerl_math35(out_dir: Path):
    """Download simpleRL-Zoo MATH level-3-5 Qwen train split and convert to verl parquet.

    Source: hkust-nlp/SimpleRL-Zoo-Data, subdir simplelr_qwen_level3to5/train.parquet
    ~8K MATH problems, level 3-5, Qwen-style prompt.

    Converts to verl format with data_source="deepmath" so the math_dapo reward
    scorer (added to the deepmath route in pre_edit) handles grading.  Rewrites
    the prompt to use the BOXED_SUFFIX used by eval datasets for consistency.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out_path = out_dir / "train.parquet"
    if out_path.exists():
        print(f"  simplerl_math35: already exists, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    src = hf_hub_download(
        repo_id="hkust-nlp/SimpleRL-Zoo-Data",
        filename="simplelr_qwen_level3to5/train.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(src)

    rows = []
    for idx, row in df.iterrows():
        question = row["extra_info"]["question"]
        answer = row["reward_model"]["ground_truth"]
        prompt_text = question + "\n" + BOXED_SUFFIX
        rows.append({
            "data_source": "deepmath",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {"answer": answer, "index": int(idx), "question": question, "split": "train"},
        })
    write_verl_parquet(rows, str(out_path))
    print(f"  simplerl_math35: {len(rows)} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    verl_data = Path(args.data_root) / "verl-data"

    # GSM8K, MATH-500, AIME-2024 from MathRuler repo
    mr_repo = verl_data / "_mathruler_repo"
    if not mr_repo.exists():
        print("Cloning MathRuler...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/hiyouga/MathRuler.git", str(mr_repo)],
            check=True,
        )
    mr_data = mr_repo / "data"

    gsm8k_dir = verl_data / "gsm8k"
    math500_dir = verl_data / "math500"
    aime2024_dir = verl_data / "aime2024"

    # GSM8K
    if not (gsm8k_dir / "test.parquet").exists():
        mathruler_to_verl_parquet(
            mr_data / "gsm8k_splits" / "test.jsonl",
            gsm8k_dir / "test.parquet", "openai/gsm8k", "test")
    if not (gsm8k_dir / "train.parquet").exists():
        mathruler_to_verl_parquet(
            mr_data / "gsm8k_splits" / "train.jsonl",
            gsm8k_dir / "train.parquet", "openai/gsm8k", "train")

    # MATH-500
    if not (math500_dir / "test.parquet").exists():
        mathruler_to_verl_parquet(
            mr_data / "math_splits" / "test.jsonl",
            math500_dir / "test.parquet", "HuggingFaceH4/MATH-500", "test")

    # AIME-2024
    if not (aime2024_dir / "test.parquet").exists():
        mathruler_to_verl_parquet(
            mr_data / "aime_splits" / "aime_2024.jsonl",
            aime2024_dir / "test.parquet", "aime2024", "test")

    # AMC (AI-MO/aimo-validation-amc, 83 problems from AMC 12 2022-2023)
    prepare_amc(verl_data / "amc23")

    # DeepMath (full + filtered)
    prepare_deepmath(verl_data / "deepmath")

    # MetaMathQA
    prepare_metamath(verl_data / "metamath")

    # SimpleRL MATH level-3-5 (Qwen-style, ~8K problems)
    prepare_simplerl_math35(verl_data / "simplerl_math35")

    print("\nAll verl datasets ready.")


if __name__ == "__main__":
    main()
