"""Pre-edit operations for HypSeek package.

Injects TEST_METRICS printing into test_task.py for DUD-E, LIT-PCBA, DEKOIS.
Patches get_uniprot_seq to use pre-cached FASTA files (no network at runtime).

Line references (original file):
- test_task.py line 118: def get_uniprot_seq(uniprot):
- test_task.py line 918: print("auc mean", ...) in test_dude
- test_task.py line 738: print("auc mean", ...) in test_pcba
- test_task.py line 1096: print("auc mean", ...) in test_dekois
"""

# TEST_METRICS snippet: pre-compute values then print (avoids quote issues in f-strings)
_TEST_METRICS = """\
        _auc_m = np.mean(auc_list)
        _bed_m = np.mean(bedroc_list)
        _ef005 = np.mean(ef_list.get('0.005', [0]))
        _ef01 = np.mean(ef_list.get('0.01', [0]))
        _ef05 = np.mean(ef_list.get('0.05', [0]))
        print(f"TEST_METRICS auc_mean={_auc_m:.6f} bedroc_mean={_bed_m:.6f} ef005_mean={_ef005:.4f} ef01_mean={_ef01:.4f} ef05_mean={_ef05:.4f}", flush=True)
"""

_MODELS_INIT_AUTODISCOVERY = """\
from pathlib import Path
import importlib

# auto-import all model files (including custom_vs_model)
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("unimol.models." + file.name[:-3])
"""

OPS = [
    # 0. Fix WORLD_SIZE KeyError in pair_dataset.py (single-GPU runs don't set it)
    {
        "op": "replace",
        "file": "HypSeek/unimol/data/pair_dataset.py",
        "start_line": 78, "end_line": 78,
        "content": "            world_size = int(os.environ.get(\"WORLD_SIZE\", \"1\"))\n",
    },
    # 0a. Patch train_task.py to read test JSONs from /data/test_datasets/
    {
        "op": "replace",
        "file": "HypSeek/unimol/tasks/train_task.py",
        "start_line": 516, "end_line": 516,
        "content": "            test_datasets_root = \"/data/test_datasets\"\n",
    },
    # 0b. Patch models/__init__.py to auto-discover all model files
    {
        "op": "replace",
        "file": "HypSeek/unimol/models/__init__.py",
        "start_line": 1,
        "end_line": 3,
        "content": _MODELS_INIT_AUTODISCOVERY,
    },
    # 0c. Fix DataLoader num_workers in test_task.py (prevent MemoryError from pickle)
    #     Single-line replaces don't shift line numbers, so safe before multi-line ops.
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 577, "end_line": 577,
     "content": "            mol_dataset, batch_size=bsz, num_workers=0,\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 657, "end_line": 657,
     "content": "                                               num_workers=0)\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 762, "end_line": 762,
     "content": "            mol_dataset, batch_size=bsz, num_workers=0,\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 840, "end_line": 840,
     "content": "                                               num_workers=0)\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 937, "end_line": 937,
     "content": "            mol_dataset, batch_size=bsz, num_workers=0,\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 1015, "end_line": 1015,
     "content": "                                               num_workers=0)\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 1353, "end_line": 1353,
     "content": "        mol_data = torch.utils.data.DataLoader(pdbbind_dataset, num_workers=0, batch_size=bsz,\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 1392, "end_line": 1392,
     "content": "        mol_data = torch.utils.data.DataLoader(bdb_dataset, num_workers=0, batch_size=bsz,\n"},
    {"op": "replace", "file": "HypSeek/unimol/tasks/test_task.py", "start_line": 1415, "end_line": 1415,
     "content": "        pocket_data = torch.utils.data.DataLoader(pocket_dataset, num_workers=0, batch_size=bsz,\n"},
    # 1. Patch get_uniprot_seq to use pre-cached FASTA (no network)
    #    Replace lines 118-132 (15 lines → 11 lines, delta=-4)
    {
        "op": "replace",
        "file": "HypSeek/unimol/tasks/test_task.py",
        "start_line": 118,
        "end_line": 132,
        "content": (
            "def get_uniprot_seq(uniprot):\n"
            "    import os\n"
            "    # Use pre-cached FASTA files (no network at runtime)\n"
            "    fasta_dirs = ['./uniport_fasta', '/data/uniport_fasta']\n"
            "    for fasta_dir in fasta_dirs:\n"
            "        fasta_path = os.path.join(fasta_dir, f'{uniprot}.fasta')\n"
            "        if os.path.exists(fasta_path):\n"
            "            with open(fasta_path, 'r') as f:\n"
            "                lines = [l.strip() for l in f if not l.startswith('>')]\n"
            "            return ''.join(lines)\n"
            "    raise FileNotFoundError(f'FASTA not found for {uniprot} in {fasta_dirs}')\n"
        ),
    },
    # 1b. Fix OOM in test_pcba_target: remove np.repeat that creates [348K,348K] matrix
    #     Original lines 611-621 (prot_reps repeat + matmul), shifted -4 → 607-617
    {
        "op": "replace",
        "file": "HypSeek/unimol/tasks/test_task.py",
        "start_line": 607,
        "end_line": 617,
        "content": (
            "        prot_np = prot_emb.cpu().numpy()\n"
            "\n"
            "        # pocket-ligand\n"
            "        sim_poc    = pocket_reps @ mol_reps.T    # [N_poc, N_lig]\n"
            "        poc_scores = sim_poc.max(axis=0)        # [N_lig]\n"
            "        # protein-ligand (memory-efficient: no repeat)\n"
            "        sim_prot    = prot_np @ mol_reps.T     # [B_pr or 1, N_lig]\n"
            "        prot_scores = sim_prot.max(axis=0)      # [N_lig]\n"
            "        prot_reps = prot_np  # for np.save below\n"
        ),
    },
    # 2. Inject TEST_METRICS after test_dekois summary
    #    Original line 1103, shifted -4 (op1) -2 (op1b) → 1097
    {
        "op": "insert",
        "file": "HypSeek/unimol/tasks/test_task.py",
        "after_line": 1097,
        "content": _TEST_METRICS,
    },
    # 3. Inject TEST_METRICS after test_dude summary
    #    Original line 925, shifted -4 (op1) -2 (op1b) → 919
    {
        "op": "insert",
        "file": "HypSeek/unimol/tasks/test_task.py",
        "after_line": 919,
        "content": _TEST_METRICS,
    },
    # 4a. Inject gc.collect + cuda.empty_cache at end of PCBA target loop body
    #     Original line 732, shifted -4 (op1) -2 (op1b) → 726
    {
        "op": "insert",
        "file": "HypSeek/unimol/tasks/test_task.py",
        "after_line": 726,
        "content": "            import gc; gc.collect()\n            torch.cuda.empty_cache()\n",
    },
    # 4b. Inject TEST_METRICS after test_pcba summary
    #     Original line 752, shifted -4 (op1) -2 (op1b) → 746, then +2 (gc insert) → 748
    {
        "op": "insert",
        "file": "HypSeek/unimol/tasks/test_task.py",
        "after_line": 748,
        "content": _TEST_METRICS,
    },
]
