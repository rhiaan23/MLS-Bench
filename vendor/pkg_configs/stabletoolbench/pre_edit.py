"""Pre-edit: Patch StableToolBench for MLS-Bench compatibility.

Applied to the stabletoolbench workspace before the agent starts.
1. Remove the model name check so any OpenAI-compatible model works (lines 39-42).
2. Remove pdb.set_trace() in the error handler (line 50).
3. Remove ToolRetriever import to avoid sentence-transformers dependency (line 14).
4. Defensive check in _step against non-dict json.loads results (line 333):
   models occasionally emit double-encoded action_input strings, which would
   crash the run loop with AttributeError on .keys() and abort all remaining
   tasks. Treat such results as unparseable and fall back to error code 2.
5. Defensive None check in rank2_subfix (rank_candidate.py line 48): the LLM
   pairwise ranker occasionally returns a message with no `content` field
   (only function_call), which crashes the entire dfs_ranked run on the very
   first task with `'NoneType' object has no attribute 'strip'`. Treat empty
   content as "B wins" so the run survives — same effect as if the LLM
   answered "B" — instead of aborting all remaining tasks.
"""

# Ordered bottom-to-top within each file so line numbers remain stable across ops.
OPS = [
    # ── chatgpt_function_model.py ──
    # 1. Remove pdb.set_trace() on error (line 50)
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/LLM/chatgpt_function_model.py",
        "start_line": 50,
        "end_line": 50,
        "content": "",
    },
    # 2. Remove the model name guard — allow any model, not just "gpt*"
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/LLM/chatgpt_function_model.py",
        "start_line": 39,
        "end_line": 42,
        "content": "        client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)\n",
    },
    # ── rapidapi_multithread.py ──
    # 3a. Defensive isinstance check on json_data before .keys() (line 332).
    #     The original line is `if "return_type" not in json_data.keys():` which
    #     crashes the entire run loop with AttributeError when the model emits
    #     a double-encoded action_input that json.loads parses to a string/list
    #     instead of a dict. Replacing the same line keeps the indented body
    #     (the return on line 333) intact.
    #     Must be applied BEFORE the line 10-14 op (bottom-to-top order).
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/Downstream_tasks/rapidapi_multithread.py",
        "start_line": 332,
        "end_line": 332,
        "content": "            if not isinstance(json_data, dict) or \"return_type\" not in json_data:\n",
    },
    # 3b. Remove all local-model imports (lines 10-14): Davinci, ToolLLaMA*, ToolRetriever
    #    These cascade-import torch, transformers, peft, deepspeed etc.
    #    We only use ChatGPTFunction (API-based).
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/Downstream_tasks/rapidapi_multithread.py",
        "start_line": 10,
        "end_line": 14,
        "content": "# from toolbench.inference.LLM.davinci_model import Davinci\n# from toolbench.inference.LLM.tool_llama_lora_model import ToolLLaMALoRA\n# from toolbench.inference.LLM.tool_llama_model import ToolLLaMA\n# from toolbench.inference.LLM.tool_llama_vllm_model import ToolLLaMA_vllm\n# from toolbench.inference.LLM.retriever import ToolRetriever\n",
    },
    # ── rank_candidate.py ──
    # 5. Defensive None-content check in rank2_subfix (line 48). The LLM
    #    pairwise ranker can return a message whose `content` field is None
    #    (only function_call set). The original line indexes [-1] on
    #    `output["content"].strip().lower()`, which crashes with
    #    `'NoneType' object has no attribute 'strip'` and aborts the entire
    #    dfs_ranked run on the very first task. Coerce missing/None content
    #    to empty string and use endswith("a") instead — empty string falls
    #    through to "B wins", same as the existing else branch.
    #    Must be applied BEFORE the line 47 op (bottom-to-top order).
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/LLM_rank/rank_candidate.py",
        "start_line": 48,
        "end_line": 48,
        "content": "    if ((output or {}).get(\"content\") or \"\").strip().lower().endswith(\"a\"):\n",
    },
    # 4. Fix parse() call: 'functions=' keyword doesn't match 'tools' positional param
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/LLM_rank/rank_candidate.py",
        "start_line": 47,
        "end_line": 47,
        "content": "    output,error_code, total_tokens = llm_interface.parse(tools=LLM_rank_args[\"functions\"],function_call=\"none\",process_id=LLM_rank_args[\"process_id\"])\n",
    },
]
