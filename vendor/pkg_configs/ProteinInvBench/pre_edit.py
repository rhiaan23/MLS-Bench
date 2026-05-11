"""Pre-edit operations for ProteinInvBench package.
Patches:
  1. Fix turtle import in pifold_module.py (requires tkinter, unavailable in container).
  2. Fix hardcoded ESM2 cache paths to use /opt/hf_cache (baked into image).
"""

_CACHE_DIR = "/opt/hf_cache"

OPS = [
    # 1. Remove turtle import (needs tkinter)
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/modules/pifold_module.py",
        "start_line": 2,
        "end_line": 2,
        "content": "# from turtle import forward  # removed: turtle requires tkinter\n",
    },
    # 2. featurizer.py:18 — module-level tokenizer
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/datasets/featurizer.py",
        "start_line": 18,
        "end_line": 18,
        "content": f'tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}") # mask token: 32\n',
    },
    # 3. cath_dataset.py:26 — CATHDataset.__init__
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/datasets/cath_dataset.py",
        "start_line": 26,
        "end_line": 26,
        "content": f'        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n',
    },
    # 4. pifold_model.py:30 — PiFold_Model.__init__
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/models/pifold_model.py",
        "start_line": 30,
        "end_line": 30,
        "content": f'        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n',
    },
    # 5. PretrainESM_model.py:16-17 — PretrainESM_Model.__init__ (non-colab branch)
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/models/PretrainESM_model.py",
        "start_line": 16,
        "end_line": 17,
        "content": (
            f'            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n'
            f'            self.model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n'
        ),
    },
    # 6. PretrainESM_model.py:37 — standalone test
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/models/PretrainESM_model.py",
        "start_line": 37,
        "end_line": 37,
        "content": f'    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n',
    },
    # 7. esmif_model.py:25 — GVPTransformerModel.__init__
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/models/esmif_model.py",
        "start_line": 25,
        "end_line": 25,
        "content": f'        alphabet = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}") \n',
    },
    # 8. Tuning.py:51 — Tuning model
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/models/Tuning.py",
        "start_line": 51,
        "end_line": 51,
        "content": f'        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n',
    },
    # 9. pretrain_interface.py:16-17 — direct snapshot path
    {
        "op": "replace",
        "file": "ProteinInvBench/PInvBench/src/interface/pretrain_interface.py",
        "start_line": 16,
        "end_line": 17,
        "content": (
            f'            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n'
            f'            self.pretrain_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="{_CACHE_DIR}")\n'
        ),
    },
]
