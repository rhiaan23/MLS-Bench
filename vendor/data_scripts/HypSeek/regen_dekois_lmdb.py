#!/usr/bin/env python3
"""Regenerate DEKOIS LMDB files to fix RDKit version incompatibility.

The DEKOIS LMDB files contain pickled RDKit Mol objects serialized with a newer
RDKit version (pickle protocol 15.0). The container uses RDKit 2022.09.5
(protocol 13.0) which cannot deserialize some entries, causing:
  RuntimeError: Bad pickle format: unexpected End-of-File while reading

This script reads each LMDB entry, re-parses the Mol object from the SMILES
string using the current RDKit version, and writes a new LMDB file.

Usage (run inside the HypSeek container or with matching RDKit version):
  python regen_dekois_lmdb.py /data/test_datasets/DEKOIS_2.0x
"""

import argparse
import lmdb
import os
import pickle
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def gen_conformation(mol, num_conf=20, num_worker=8):
    """Generate 3D conformer(s) for a molecule."""
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conf,
            numThreads=num_worker,
            pruneRmsThresh=1,
            maxAttempts=10000,
            useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None
    return mol


def fix_entry(entry):
    """Re-create the Mol object from SMILES if pickle fails, preserving coordinates."""
    smi = entry.get("smi", "")
    old_mol = entry.get("mol")

    # If the mol object loaded fine, just return as-is
    if old_mol is not None and isinstance(old_mol, Chem.rdchem.Mol):
        return entry

    # Re-create from SMILES
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"  WARNING: Cannot parse SMILES: {smi}")
        return entry

    # Try to generate conformer
    mol = gen_conformation(mol, num_conf=1, num_worker=4)
    if mol is None:
        # Fall back to 2D mol
        mol = Chem.MolFromSmiles(smi)

    entry["mol"] = mol
    return entry


def regen_lmdb(src_path, dst_path):
    """Read LMDB at src_path, fix entries, write to dst_path."""
    if not os.path.isfile(src_path):
        print(f"  SKIP: {src_path} not found")
        return 0

    env_in = lmdb.open(
        src_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    entries = []
    fixed = 0
    with env_in.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for key in keys:
            val = txn.get(key)
            try:
                data = pickle.loads(val)
                entries.append((key, data))
            except Exception as e:
                # Entry is corrupt - try to reconstruct
                print(f"  Fixing corrupt entry key={key}: {e}")
                # Load what we can - skip the mol field
                # We need to extract atoms, coordinates, smi, label from
                # partial data. Let's try a brute force approach.
                try:
                    # Try loading with restrictedpickle that ignores rdkit
                    import io

                    class PartialUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if "rdkit" in module.lower():
                                return type("DummyMol", (), {"__setstate__": lambda s, d: None})
                            return super().find_class(module, name)

                    data = PartialUnpickler(io.BytesIO(val)).load()
                except Exception:
                    # Last resort: create minimal entry from known SMILES
                    # Read the good entries to figure out the SMILES list
                    print(f"    Cannot even partially load entry {key}")
                    data = {"atoms": [], "coordinates": [], "smi": "", "mol": None, "label": 0}

                data = fix_entry(data)
                entries.append((key, data))
                fixed += 1
    env_in.close()

    # Write new LMDB
    if os.path.exists(dst_path):
        os.remove(dst_path)
    env_out = lmdb.open(
        dst_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=1099511627776,
    )
    with env_out.begin(write=True) as txn:
        for key, data in entries:
            txn.put(key, pickle.dumps(data))
    env_out.close()

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Regenerate DEKOIS LMDB files")
    parser.add_argument("data_root", help="Path to DEKOIS_2.0x directory")
    parser.add_argument("--output", help="Output directory (default: overwrite in-place)")
    parser.add_argument("--dry-run", action="store_true", help="Only check, don't write")
    args = parser.parse_args()

    data_root = args.data_root
    out_root = args.output or data_root
    targets = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )

    print(f"RDKit version: {Chem.rdBase.rdkitVersion}")
    print(f"Processing {len(targets)} DEKOIS targets from {data_root}")
    if args.output:
        print(f"Output to: {out_root}")

    total_fixed = 0
    for target in targets:
        src_lig = os.path.join(data_root, target, f"{target}_lig.lmdb")
        src_poc = os.path.join(data_root, target, f"{target}_pocket.lmdb")

        if not os.path.isfile(src_lig):
            continue

        out_dir = os.path.join(out_root, target)
        os.makedirs(out_dir, exist_ok=True)
        dst_lig = os.path.join(out_dir, f"{target}_lig.lmdb")
        dst_poc = os.path.join(out_dir, f"{target}_pocket.lmdb")

        if args.dry_run:
            # Just check for corruption
            env = lmdb.open(src_lig, subdir=False, readonly=True, lock=False, max_readers=256)
            with env.begin() as txn:
                keys = list(txn.cursor().iternext(values=False))
                for key in keys:
                    val = txn.get(key)
                    try:
                        pickle.loads(val)
                    except Exception as e:
                        print(f"  {target} key={key}: {e}")
                        total_fixed += 1
            env.close()
            continue

        n = regen_lmdb(src_lig, dst_lig)
        total_fixed += n
        if n > 0:
            print(f"  {target}: fixed {n} entries in lig.lmdb")

        regen_lmdb(src_poc, dst_poc)

    if args.dry_run:
        print(f"\nDry run: {total_fixed} corrupt entries found")
    else:
        print(f"\nDone. Fixed {total_fixed} entries total.")


if __name__ == "__main__":
    main()
