# Source Data Provenance

The canonical sources are public benchmark datasets:

- `THUDM/LongBench`: LongBench-E `hotpotqa_e`,
  `passage_retrieval_en_e`, and `repobench-p_e`
- `THUDM/LongBench-v2` / `zai-org/LongBench-v2`: train split multiple-choice
  long-context benchmark
- `openai/gsm8k`: `main` test split

The runtime reads these datasets through the Hugging Face dataset cache mounted
by `transformers-kv-lab`:

- `THUDM/LongBench` `data.zip`
- `THUDM/LongBench-v2` / `zai-org/LongBench-v2` `train` split
- `openai/gsm8k` `main` test split

Checked-in sample rows are for provenance and lightweight parser inspection;
canonical runs use the public datasets above.

Sample files:

- `data/longbench/*.sample.jsonl`
  - representative LongBench-E rows for smoke/debugging
- `data/math/gsm8k.sample.jsonl`
  - expected row fields are:
    - `question`
    - `answer`
    - optional `unique_id`

Offline runtime contract:

- The default model is `Qwen/Qwen2.5-3B-Instruct`, matching the
  `transformers-kv-lab` package setup.
- Full runs require the model and datasets to be available through the
  build-time or mounted Hugging Face cache used by `transformers-kv-lab`.
- Runtime dataset/model loading uses local-cache-only resolution; compute-node
  evaluation must not rely on network access.
- `SELECTION_KV_MAX_EXAMPLES=0` means use the full available workload.
- Positive `SELECTION_KV_MAX_EXAMPLES` values are for smoke validation only.
