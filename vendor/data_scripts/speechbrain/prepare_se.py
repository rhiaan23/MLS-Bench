#!/usr/bin/env python3
"""Prepare speech enhancement datasets for speech-enhancement task.

Downloads:
  - VoiceBank-DEMAND (~5GB, paired noisy/clean speech, 28 speakers)
  - Noisy-VCTK-56spk (~8GB, paired noisy/clean speech, 56 speakers)
  - LibriMix (Libri2Mix min, train-100, generated from LibriSpeech + WHAM!)

Output: {data_root}/speech/se/{voicebank-demand,noisy-vctk-56spk,librimix}/

Usage:
    python vendor/data_scripts/speechbrain/prepare_se.py --data-root vendor/data
"""

import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf

# Container path prefix — manifests must use this, not host paths
CONTAINER_PREFIX = "/data/speech/se"


def _container_path(host_path: Path, dataset_dir: Path, dataset_name: str) -> str:
    """Convert a host path to a container path for manifest entries."""
    rel = host_path.relative_to(dataset_dir)
    return f"{CONTAINER_PREFIX}/{dataset_name}/{rel}"


def download_voicebank_demand(out_dir: Path):
    """Download VoiceBank-DEMAND dataset (Valentini et al., 2016)."""
    vb_dir = out_dir / "voicebank-demand"
    vb_dir.mkdir(parents=True, exist_ok=True)

    marker = vb_dir / ".done"
    if marker.exists():
        print("  VoiceBank-DEMAND: already downloaded")
    else:
        # Download noisy + clean pairs
        base_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
        files = [
            "noisy_trainset_28spk_wav.zip",
            "clean_trainset_28spk_wav.zip",
            "noisy_testset_wav.zip",
            "clean_testset_wav.zip",
        ]

        for fname in files:
            url = f"{base_url}/{fname}"
            archive = vb_dir / fname
            if (vb_dir / fname.replace(".zip", "")).exists():
                print(f"  {fname}: already extracted")
                continue
            print(f"  Downloading {fname}...", flush=True)
            subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
            subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(vb_dir)], check=True)
            archive.unlink()

        marker.touch()
        print("  VoiceBank-DEMAND: done")

    generate_voicebank_manifests(vb_dir)


def generate_voicebank_manifests(vb_dir: Path):
    """Generate JSONL manifests for VoiceBank-DEMAND."""
    split_configs = {
        "train": ("noisy_trainset_28spk_wav", "clean_trainset_28spk_wav"),
        "test": ("noisy_testset_wav", "clean_testset_wav"),
    }
    for name, (noisy_subdir, clean_subdir) in split_configs.items():
        manifest_path = vb_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  VoiceBank manifest {name}: already exists")
            continue
        noisy_dir = vb_dir / noisy_subdir
        clean_dir = vb_dir / clean_subdir
        if not noisy_dir.exists() or not clean_dir.exists():
            print(f"  VoiceBank {name} dirs not found, skipping manifest")
            continue
        entries = []
        for noisy_wav in sorted(noisy_dir.glob("*.wav")):
            clean_wav = clean_dir / noisy_wav.name
            if not clean_wav.exists():
                continue
            info = sf.info(str(noisy_wav))
            container_noisy = _container_path(noisy_wav, vb_dir, "voicebank-demand")
            container_clean = _container_path(clean_wav, vb_dir, "voicebank-demand")
            entries.append({
                "id": noisy_wav.stem,
                "noisy": container_noisy,
                "clean": container_clean,
                "duration": info.duration,
            })
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  VoiceBank manifest {name}: {len(entries)} pairs")


def download_noisy_vctk_56spk(out_dir: Path):
    """Download Noisy-VCTK-56spk dataset from Edinburgh DataShare.

    This provides paired noisy/clean speech for 56 speakers (training)
    and a shared test set.
    """
    nv_dir = out_dir / "noisy-vctk-56spk"
    nv_dir.mkdir(parents=True, exist_ok=True)

    marker = nv_dir / ".done"
    if marker.exists():
        print("  Noisy-VCTK-56spk: already downloaded")
        generate_noisy_vctk_manifests(nv_dir)
        return

    base_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
    files = [
        "noisy_trainset_56spk_wav.zip",
        "clean_trainset_56spk_wav.zip",
        "noisy_testset_wav.zip",
        "clean_testset_wav.zip",
    ]

    for fname in files:
        # Check if already extracted (strip .zip for directory name)
        extracted_name = fname.replace(".zip", "")
        if (nv_dir / extracted_name).exists():
            print(f"  {fname}: already extracted")
            continue
        url = f"{base_url}/{fname}"
        archive = nv_dir / fname
        print(f"  Downloading {fname}...", flush=True)
        subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
        print(f"  Extracting {fname}...", flush=True)
        subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(nv_dir)], check=True)
        archive.unlink()

    marker.touch()
    print("  Noisy-VCTK-56spk: done")
    generate_noisy_vctk_manifests(nv_dir)


def generate_noisy_vctk_manifests(nv_dir: Path):
    """Generate JSONL manifests for Noisy-VCTK-56spk."""
    split_configs = {
        "train": ("noisy_trainset_56spk_wav", "clean_trainset_56spk_wav"),
        "test": ("noisy_testset_wav", "clean_testset_wav"),
    }
    for name, (noisy_subdir, clean_subdir) in split_configs.items():
        manifest_path = nv_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  Noisy-VCTK-56spk manifest {name}: already exists")
            continue
        noisy_dir = nv_dir / noisy_subdir
        clean_dir = nv_dir / clean_subdir
        if not noisy_dir.exists() or not clean_dir.exists():
            print(f"  Noisy-VCTK-56spk {name} dirs not found, skipping manifest")
            continue
        entries = []
        for noisy_wav in sorted(noisy_dir.glob("*.wav")):
            clean_wav = clean_dir / noisy_wav.name
            if not clean_wav.exists():
                continue
            info = sf.info(str(noisy_wav))
            container_noisy = _container_path(noisy_wav, nv_dir, "noisy-vctk-56spk")
            container_clean = _container_path(clean_wav, nv_dir, "noisy-vctk-56spk")
            entries.append({
                "id": noisy_wav.stem,
                "noisy": container_noisy,
                "clean": container_clean,
                "duration": info.duration,
            })
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  Noisy-VCTK-56spk manifest {name}: {len(entries)} pairs")


def download_librimix(out_dir: Path, data_root: str):
    """Generate LibriMix from LibriSpeech (Libri2Mix, min mode, train-100).

    Requires LibriSpeech data to already be at {data_root}/speech/asr/librispeech-100/.
    Downloads WHAM! noise and runs the LibriMix generation script.
    """
    lm_dir = out_dir / "librimix"
    lm_dir.mkdir(parents=True, exist_ok=True)

    marker = lm_dir / ".done"
    if marker.exists():
        print("  LibriMix: already generated")
        generate_librimix_manifests(lm_dir)
        return

    print("  Setting up LibriMix generation...", flush=True)

    # Clone LibriMix generation scripts
    repo_dir = lm_dir / "LibriMix"
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/JorisCos/LibriMix.git",
            str(repo_dir),
        ], check=True)

    # Download WHAM! noise
    wham_dir = lm_dir / "wham_noise"
    wham_archive = lm_dir / "wham_noise.zip"
    if not wham_dir.exists():
        wham_url = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
        print("  Downloading WHAM! noise (~17GB)...", flush=True)
        subprocess.run(["wget", "-c", "-q", wham_url, "-O", str(wham_archive)], check=True)
        print("  Extracting WHAM! noise...", flush=True)
        subprocess.run(["unzip", "-q", "-o", str(wham_archive), "-d", str(lm_dir)], check=True)
        wham_archive.unlink()
    else:
        print("  WHAM! noise: already downloaded")

    # Check if LibriSpeech data exists
    ls_dir = Path(data_root) / "speech" / "asr" / "librispeech-100" / "LibriSpeech"
    if not ls_dir.exists():
        print(f"  WARNING: LibriSpeech not found at {ls_dir}")
        print("  Run prepare_asr.py first. Skipping LibriMix generation.")
        marker.touch()
        generate_librimix_manifests(lm_dir)
        return

    # Run LibriMix generation: Libri2Mix, min mode, from train-clean-100
    # The generation script expects specific arguments
    gen_script = repo_dir / "generate_librimix.sh"
    if gen_script.exists():
        print("  Generating Libri2Mix (min mode, train-clean-100)...", flush=True)
        try:
            subprocess.run(
                ["bash", str(gen_script), str(ls_dir.parent), str(lm_dir)],
                cwd=str(repo_dir),
                check=True,
                timeout=7200,  # 2 hour timeout
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  WARNING: LibriMix generation failed or timed out: {e}")
            print("  The WHAM! noise is downloaded. You may need to run generation manually:")
            print(f"    cd {repo_dir}")
            print(f"    bash generate_librimix.sh {ls_dir.parent} {lm_dir}")
    else:
        print(f"  WARNING: generate_librimix.sh not found at {gen_script}")
        print("  WHAM! noise is downloaded. Run LibriMix generation manually.")

    marker.touch()
    print("  LibriMix: done")
    generate_librimix_manifests(lm_dir)


def generate_librimix_manifests(lm_dir: Path):
    """Generate JSONL manifests for LibriMix.

    LibriMix generates: Libri2Mix/wav16k/min/train-100/{mix_both,mix_clean,s1,s2,noise}/

    For speech enhancement (denoising), we pair:
      - noisy = mix_both  (two speakers + noise)
      - clean = mix_clean (same two speakers without noise)

    This formulates a proper denoising task (remove noise from the mixture).
    Note: using s1 as clean would be a separation task, which a single-output
    enhancement model cannot learn effectively.
    """
    manifest_path = lm_dir / "train.json"
    if manifest_path.exists():
        print("  LibriMix manifest: already exists")
        return

    # Check if Libri2Mix was generated
    mix_dir = lm_dir / "Libri2Mix" / "wav16k" / "min" / "train-100"
    if mix_dir.exists():
        mix_clean_dir = mix_dir / "mix_clean"
        mix_both_dir = mix_dir / "mix_both"
        entries = []
        if mix_clean_dir.exists() and mix_both_dir.exists():
            for wav_file in sorted(mix_both_dir.glob("*.wav")):
                clean_file = mix_clean_dir / wav_file.name
                if not clean_file.exists():
                    continue
                info = sf.info(str(wav_file))
                container_noisy = _container_path(wav_file, lm_dir, "librimix")
                container_clean = _container_path(clean_file, lm_dir, "librimix")
                entries.append({
                    "id": wav_file.stem,
                    "noisy": container_noisy,
                    "clean": container_clean,
                    "duration": info.duration,
                })
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  LibriMix manifest: {len(entries)} mixtures")
    else:
        manifest_path.write_text("")
        print("  LibriMix manifest: created (empty -- run generate_librimix.sh first)")

    # Also generate test manifest if test data exists
    test_manifest_path = lm_dir / "test.json"
    if test_manifest_path.exists():
        print("  LibriMix test manifest: already exists")
        return

    test_mix_dir = lm_dir / "Libri2Mix" / "wav16k" / "min" / "test"
    if test_mix_dir.exists():
        mix_clean_dir = test_mix_dir / "mix_clean"
        mix_both_dir = test_mix_dir / "mix_both"
        entries = []
        if mix_clean_dir.exists() and mix_both_dir.exists():
            for wav_file in sorted(mix_both_dir.glob("*.wav")):
                clean_file = mix_clean_dir / wav_file.name
                if not clean_file.exists():
                    continue
                info = sf.info(str(wav_file))
                container_noisy = _container_path(wav_file, lm_dir, "librimix")
                container_clean = _container_path(clean_file, lm_dir, "librimix")
                entries.append({
                    "id": wav_file.stem,
                    "noisy": container_noisy,
                    "clean": container_clean,
                    "duration": info.duration,
                })
        with open(test_manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  LibriMix test manifest: {len(entries)} mixtures")


def main():
    parser = argparse.ArgumentParser(description="Prepare speech enhancement datasets")
    parser.add_argument("--data-root", type=str, required=True, help="Root data directory")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "speech" / "se"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing speech enhancement datasets ===", flush=True)
    download_voicebank_demand(out_dir)
    download_noisy_vctk_56spk(out_dir)
    download_librimix(out_dir, data_root=args.data_root)
    print("=== SE data preparation complete ===")


if __name__ == "__main__":
    main()
