"""Pre-edit for dbim-codebase.

Applied to a fresh copy of the package before any task/baseline edits.
Keeps the upstream repo unmodified (vendor/external_packages/ is read-only).
"""

OPS = [
    # 1. Make Inception model path absolute relative to the repo's assets dir.
    {
        "op": "replace",
        "file": "dbim-codebase/evaluations/feature_extractor.py",
        "start_line": 19,
        "end_line": 19,
        "content": '        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")\n',
    },
    # 2. Respect the CLI --num_samples instead of overwriting it with the full
    #    dataset length. Without this, --num_samples is ignored and we always
    #    sample the entire train set (138k for e2h).
    {
        "op": "replace",
        "file": "dbim-codebase/sample.py",
        "start_line": 80,
        "end_line": 80,
        "content": "    args.num_samples = min(args.num_samples, len(dataloader.dataset))\n",
    },
    # 3. Add an early-exit to the sampling loop so runs don't generate images
    #    past args.num_samples. Replaces the trailing rank-0 log line with a
    #    log + break-on-enough block.
    {
        "op": "replace",
        "file": "dbim-codebase/sample.py",
        "start_line": 154,
        "end_line": 155,
        "content": (
            "        if dist.get_rank() == 0:\n"
            "            logger.log(f\"sampled {num} images\")\n"
            "        if num >= args.num_samples:\n"
            "            break\n"
        ),
    },
    # 4. Make the evaluator find whatever sample file was actually written,
    #    instead of looking for a hardcoded size (138567/16502/10000) and a
    #    specific nfe number. Heun/high-order solvers return a different `nfe`
    #    than the --steps arg, and num_samples is now configurable, so both
    #    the size and nfe can differ from the hardcoded SAMPLE_NAME.
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/evaluate.sh",
        "start_line": 44,
        "end_line": 45,
        "content": (
            "SAMPLE_DIR=workdir/${PREFIX}/split=${SPLIT}/${SAMPLER}/steps=${N}\n"
            "SAMPLE_PATH=$(ls ${SAMPLE_DIR}/samples_*.npz 2>/dev/null | head -1)\n"
            'if [ -z "$SAMPLE_PATH" ]; then\n'
            '    echo "ERROR: no samples_*.npz found under $SAMPLE_DIR" >&2\n'
            "    exit 1\n"
            "fi\n"
            "# Imagenet evaluator also needs a labels_*.npz — glob it too (size/nfe vary).\n"
            "LABEL_NAME=$(ls ${SAMPLE_DIR}/labels_*.npz 2>/dev/null | head -1 | xargs -n1 basename)\n"
        ),
    },
    # 5. Forward a --num_samples CLI arg to sample.py (read from the `num_samples`
    #    env var that the per-task run scripts export). No-op if the var isn't set.
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/sample.sh",
        "start_line": 53,
        "end_line": 59,
        "content": (
            "torchrun $run_args sample.py --steps $N --sampler $GEN_SAMPLER --batch_size $BS \\\n"
            " --model_path $MODEL_PATH --class_cond $CLASS_COND --noise_schedule $PRED \\\n"
            ' ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \\\n'
            " --condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \\\n"
            " --dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \\\n"
            " --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT \\\n"
            ' ${num_samples:+ --num_samples="${num_samples}"} ${SEED:+ --seed="${SEED}"} \\\n'
            ' ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} \\\n'
        ),
    },
    # 6b. lmdb max_readers=1 crashes when DataLoader forks into 2x num_workers per
    #     rank (~16 readers). Bump to 256 so distributed sampling works.
    {
        "op": "replace",
        "file": "dbim-codebase/datasets/imagenet_inpaint.py",
        "start_line": 60,
        "end_line": 60,
        "content": "    data_set.lmdb_data = lmdb.open(lmdb_path, readonly=True, max_readers=256, lock=False, readahead=False, meminit=False)\n",
    },
    # 6d. Enforce NFE budget on the denoiser inside karras_sample.
    # Agents were cheating (double-denoise / Heun corrector -> 6-9 actual calls
    # while reporting nfe=len(ts)-1). Wrap the closure so it:
    #   * counts every actual diffusion.denoise invocation
    #   * raises RuntimeError on call N+1 where N = steps+1 (true NFE budget)
    #   * prints "ACTUAL_NFE: <n> / EXPECTED_NFE: <n>" for the parser
    # All agent edits go through this closure — there's no other way to call
    # the model. A raised error stops sample.py mid-run, no samples file is
    # written, evaluator finds nothing, agent gets feedback "budget exceeded".
    {
        "op": "replace",
        "file": "dbim-codebase/ddbm/karras_diffusion.py",
        "start_line": 275,
        "end_line": 279,
        "content": (
            "    _nfe_budget = steps + 1\n"
            "    _nfe_counter = [0]\n"
            "    def denoiser(x_t, sigma):\n"
            "        if _nfe_counter[0] >= _nfe_budget:\n"
            "            raise RuntimeError(\n"
            "                f'NFE_BUDGET_EXCEEDED: used {_nfe_counter[0]+1} denoiser calls '\n"
            "                f'but budget is {_nfe_budget}. Do not double-denoise / Heun-correct '\n"
            "                f'beyond the allowed NFE.'\n"
            "            )\n"
            "        _nfe_counter[0] += 1\n"
            "        _, denoised, _ = diffusion.denoise(model, x_t, sigma, **model_kwargs)\n"
            "        if clip_denoised:\n"
            "            denoised = denoised.clamp(-1, 1)\n"
            "        return denoised\n"
        ),
    },
    # (Removed OP 6e: it used original line numbers that were stale after
    # OP 6d inserted 9 lines, overwriting `return denoised` and breaking
    # the denoiser closure. RuntimeError from OP 6d already contains the
    # string "NFE_BUDGET_EXCEEDED" which the parser detects from stderr.)
    # 6c. evaluation/resnet.py imports `from ipdb import set_trace as debug` only
    #     for debugging. ipdb transitively needs the `decorator` package which
    #     is not in our install_cmds. Strip the ipdb import; `debug` isn't used
    #     anywhere in the runtime path.
    {
        "op": "replace",
        "file": "dbim-codebase/evaluation/resnet.py",
        "start_line": 14,
        "end_line": 14,
        "content": "\n",
    },
    # 6. For imagenet_inpaint sampling: reuse the val10k testset as trainset/valset
    #    placeholders instead of opening the val lmdb three times. The train lmdb is
    #    a symlink to val (we only downloaded val), so two opens of the same file
    #    raises `lmdb.Error: environment already open in this process`. sample.py
    #    only iterates `all_dataloaders[2]` (the testset) when split=test, so the
    #    other two loaders never yield — they just need to be constructible.
    {
        "op": "replace",
        "file": "dbim-codebase/datasets/__init__.py",
        "start_line": 217,
        "end_line": 226,
        "content": (
            '    elif "imagenet_inpaint" in dataset:\n'
            '        corrupt_type = dataset.split("_")[-1]\n'
            '        assert corrupt_type in ["center", "freeform2030"]\n'
            "        from .imagenet_inpaint import ImageNetInpaintingDataset, InpaintingVal10kSubset\n"
            "\n"
            "        if include_test:\n"
            "            # Sampling path: only the testset (val10k subset) is read.\n"
            "            # Reuse it for train/val slots to avoid re-opening the lmdb.\n"
            "            testset = InpaintingVal10kSubset(root, image_size, corrupt_type)\n"
            "            trainset = testset\n"
            "            valset = testset\n"
            "        else:\n"
            "            trainset = ImageNetInpaintingDataset(root, image_size, corrupt_type, train=True)\n"
            "            valset = ImageNetInpaintingDataset(root, image_size, corrupt_type, train=False)\n"
        ),
    },
    # 7. Match torchrun's nproc_per_node to the actual number of GPUs
    #    visible inside the container. Upstream sample.sh hardcodes
    #    `CUDA_VISIBLE_DEVICES=0,1,...,7` and `--nproc_per_node 8`, so on a
    #    smaller allocation torchrun spins up 8 ranks but only N<8 GPUs
    #    are visible — ranks past N-1 abort with `CUDA error: invalid
    #    device ordinal`. Detect via nvidia-smi (always present when
    #    docker --gpus / apptainer --nv is used). Falls back to 8 if the
    #    probe fails so single-host runs without nvidia-smi still work.
    {
        "op": "replace",
        "file": "dbim-codebase/scripts/sample.sh",
        "start_line": 11,
        "end_line": 13,
        "content": (
            "if command -v nvidia-smi >/dev/null 2>&1; then\n"
            "    NGPU=$(nvidia-smi --list-gpus 2>/dev/null | grep -c '^GPU ')\n"
            "fi\n"
            "if [ -z \"${NGPU:-}\" ] || [ \"$NGPU\" -lt 1 ]; then\n"
            "    NGPU=8\n"
            "fi\n"
            "run_args=\"--nproc_per_node $NGPU \\\n"
            "          --master_port 29511\"\n"
        ),
    },
]
