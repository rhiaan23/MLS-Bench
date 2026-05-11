#!/usr/bin/env python3
"""Prepare TTS/Vocoder datasets for speech-vocoder task.

Downloads:
  - LJSpeech-1.1 (~2.6GB, single speaker English)
  - VCTK subset (5 speakers, ~2GB, multi-speaker English)
  - AISHELL-3 subset (5 speakers, ~3GB, multi-speaker Chinese)

Output: {data_root}/speech/tts/{ljspeech,vctk-5spk,aishell3-5spk}/

Usage:
    python vendor/data_scripts/speechbrain/prepare_tts.py --data-root vendor/data
"""

import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf

# Container path prefix — manifests must use this, not host paths
CONTAINER_PREFIX = "/data/speech/tts"


def _container_path(host_path: Path, dataset_dir: Path, dataset_name: str) -> str:
    """Convert a host path to a container path for manifest entries."""
    rel = host_path.relative_to(dataset_dir)
    return f"{CONTAINER_PREFIX}/{dataset_name}/{rel}"


def download_ljspeech(out_dir: Path):
    """Download LJSpeech-1.1."""
    lj_dir = out_dir / "ljspeech"
    lj_dir.mkdir(parents=True, exist_ok=True)

    if (lj_dir / "LJSpeech-1.1" / "wavs").exists():
        print("  LJSpeech: already downloaded")
    else:
        url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        archive = lj_dir / "LJSpeech-1.1.tar.bz2"
        print("  Downloading LJSpeech-1.1...", flush=True)
        subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
        print("  Extracting...", flush=True)
        subprocess.run(["tar", "xjf", str(archive), "-C", str(lj_dir)], check=True)
        archive.unlink()

    print("  LJSpeech: done")
    generate_ljspeech_manifests(lj_dir)


def generate_ljspeech_manifests(lj_dir: Path):
    """Generate JSONL manifests for LJSpeech (train/test split)."""
    wavs_dir = lj_dir / "LJSpeech-1.1" / "wavs"
    meta_file = lj_dir / "LJSpeech-1.1" / "metadata.csv"
    if not wavs_dir.exists() or not meta_file.exists():
        print("  LJSpeech data not found, skipping manifest")
        return

    # Check both manifests first
    train_manifest = lj_dir / "train.json"
    test_manifest = lj_dir / "test.json"
    if train_manifest.exists() and test_manifest.exists():
        print("  LJSpeech manifests: already exist")
        return

    entries = []
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            wav_path = wavs_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                continue
            info = sf.info(str(wav_path))
            container_wav = _container_path(wav_path, lj_dir, "ljspeech")
            entries.append({
                "id": utt_id,
                "wav": container_wav,
                "duration": info.duration,
            })

    # 90/10 train/test split
    n_test = max(1, len(entries) // 10)
    train_entries = entries[:-n_test]
    test_entries = entries[-n_test:]

    for name, ents in [("train", train_entries), ("test", test_entries)]:
        manifest_path = lj_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  LJSpeech manifest {name}: already exists")
            continue
        with open(manifest_path, "w") as f:
            for e in ents:
                f.write(json.dumps(e) + "\n")
        print(f"  LJSpeech manifest {name}: {len(ents)} utterances")


def download_vctk_subset(out_dir: Path, n_speakers: int = 5):
    """Download VCTK and select a subset of speakers."""
    vctk_dir = out_dir / "vctk-5spk"
    vctk_dir.mkdir(parents=True, exist_ok=True)

    marker = vctk_dir / ".done"
    wav_dir = vctk_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed"
    if not wav_dir.exists():
        wav_dir = vctk_dir / "wav48_silence_trimmed"

    # Check marker AND verify data integrity — if marker exists but wav dir
    # is missing or has too many speakers, fix before returning.
    if marker.exists():
        if wav_dir.exists() and any(wav_dir.iterdir()):
            # Verify speaker count — pruning may have been skipped on a prior run
            current_speakers = sorted([d.name for d in wav_dir.iterdir() if d.is_dir()])
            if len(current_speakers) <= n_speakers:
                print("  VCTK subset: already downloaded")
                generate_vctk_manifests(vctk_dir)
                return
            else:
                print(f"  VCTK subset: marker exists but found {len(current_speakers)} speakers "
                      f"(expected <= {n_speakers}), re-pruning...")
                import shutil
                keep = set(current_speakers[:n_speakers])
                for spk_dir in wav_dir.iterdir():
                    if spk_dir.is_dir() and spk_dir.name not in keep:
                        shutil.rmtree(spk_dir)
                print(f"  Pruned to {n_speakers} speakers: {sorted(keep)}")
                # Delete stale manifests so they get regenerated
                for mf in (vctk_dir / "train.json", vctk_dir / "test.json"):
                    mf.unlink(missing_ok=True)
                generate_vctk_manifests(vctk_dir)
                return
        else:
            print("  VCTK subset: marker exists but data is incomplete, re-downloading...")
            marker.unlink()

    # Download full VCTK
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    archive = vctk_dir / "VCTK-Corpus-0.92.zip"
    print("  Downloading VCTK...", flush=True)
    subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
    print("  Extracting...", flush=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(vctk_dir)], check=True)
    archive.unlink()

    # Select first n_speakers (sorted by ID)
    if wav_dir.exists():
        speakers = sorted([d.name for d in wav_dir.iterdir() if d.is_dir()])[:n_speakers]
        print(f"  Selected speakers: {speakers}")

        # Remove non-selected speakers to save space
        import shutil
        for spk_dir in wav_dir.iterdir():
            if spk_dir.is_dir() and spk_dir.name not in speakers:
                shutil.rmtree(spk_dir)

    marker.touch()
    print(f"  VCTK subset ({n_speakers} speakers): done")
    generate_vctk_manifests(vctk_dir)


def download_aishell3_subset(out_dir: Path, n_speakers: int = 5):
    """Download AISHELL-3 and select a subset of speakers."""
    ai3_dir = out_dir / "aishell3-5spk"
    ai3_dir.mkdir(parents=True, exist_ok=True)

    marker = ai3_dir / ".done"
    train_wav_dir = ai3_dir / "train" / "wav"
    if marker.exists():
        if train_wav_dir.exists() and any(train_wav_dir.iterdir()):
            current_speakers = sorted([d.name for d in train_wav_dir.iterdir() if d.is_dir()])
            if len(current_speakers) <= n_speakers:
                print("  AISHELL-3 subset: already downloaded")
                generate_aishell3_manifests(ai3_dir)
                return
            else:
                print(f"  AISHELL-3 subset: marker exists but found {len(current_speakers)} speakers "
                      f"(expected <= {n_speakers}), re-pruning...")
                import shutil
                keep = set(current_speakers[:n_speakers])
                for spk_dir in train_wav_dir.iterdir():
                    if spk_dir.is_dir() and spk_dir.name not in keep:
                        shutil.rmtree(spk_dir)
                print(f"  Pruned to {n_speakers} speakers: {sorted(keep)}")
                for mf in (ai3_dir / "train.json", ai3_dir / "test.json"):
                    mf.unlink(missing_ok=True)
                generate_aishell3_manifests(ai3_dir)
                return
        else:
            print("  AISHELL-3 subset: marker exists but data is incomplete, re-downloading...")
            marker.unlink()

    url = "https://www.openslr.org/resources/93/data_aishell3.tgz"
    archive = ai3_dir / "data_aishell3.tgz"
    print("  Downloading AISHELL-3...", flush=True)
    subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
    print("  Extracting...", flush=True)
    subprocess.run(["tar", "xzf", str(archive), "-C", str(ai3_dir)], check=True)
    archive.unlink()

    # Select first n_speakers from train set
    train_wav_dir = ai3_dir / "train" / "wav"
    if train_wav_dir.exists():
        speakers = sorted([d.name for d in train_wav_dir.iterdir() if d.is_dir()])[:n_speakers]
        print(f"  Selected speakers: {speakers}")
        import shutil
        for spk_dir in train_wav_dir.iterdir():
            if spk_dir.is_dir() and spk_dir.name not in speakers:
                shutil.rmtree(spk_dir)

    marker.touch()
    print(f"  AISHELL-3 subset ({n_speakers} speakers): done")
    generate_aishell3_manifests(ai3_dir)


def generate_vctk_manifests(vctk_dir: Path):
    """Generate JSONL manifests for VCTK subset."""
    # VCTK uses 48kHz FLAC files in wav48_silence_trimmed
    # Try both possible layouts (with or without VCTK-Corpus-0.92 subdirectory)
    wav_dir = vctk_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed"
    if not wav_dir.exists():
        wav_dir = vctk_dir / "wav48_silence_trimmed"
    if not wav_dir.exists():
        print("  VCTK wav dir not found, skipping manifest")
        return

    # Check both manifests first
    train_manifest = vctk_dir / "train.json"
    test_manifest = vctk_dir / "test.json"
    if train_manifest.exists() and test_manifest.exists():
        print("  VCTK manifests: already exist")
        return

    entries = []
    for spk_dir in sorted(wav_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        for wav_file in sorted(spk_dir.glob("*.flac")):
            info = sf.info(str(wav_file))
            container_wav = _container_path(wav_file, vctk_dir, "vctk-5spk")
            entries.append({
                "id": wav_file.stem,
                "wav": container_wav,
                "duration": info.duration,
            })
    n_test = max(1, len(entries) // 10)
    train_entries = entries[:-n_test]
    test_entries = entries[-n_test:]
    for name, ents in [("train", train_entries), ("test", test_entries)]:
        manifest_path = vctk_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  VCTK manifest {name}: already exists")
            continue
        with open(manifest_path, "w") as f:
            for e in ents:
                f.write(json.dumps(e) + "\n")
        print(f"  VCTK manifest {name}: {len(ents)} utterances")


def generate_aishell3_manifests(ai3_dir: Path):
    """Generate JSONL manifests for AISHELL-3 subset."""
    train_wav_dir = ai3_dir / "train" / "wav"
    if not train_wav_dir.exists():
        print("  AISHELL-3 wav dir not found, skipping manifest")
        return

    # Check both manifests first
    train_manifest = ai3_dir / "train.json"
    test_manifest = ai3_dir / "test.json"
    if train_manifest.exists() and test_manifest.exists():
        print("  AISHELL-3 manifests: already exist")
        return

    entries = []
    for spk_dir in sorted(train_wav_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        for wav_file in sorted(spk_dir.glob("*.wav")):
            info = sf.info(str(wav_file))
            container_wav = _container_path(wav_file, ai3_dir, "aishell3-5spk")
            entries.append({
                "id": wav_file.stem,
                "wav": container_wav,
                "duration": info.duration,
            })
    n_test = max(1, len(entries) // 10)
    train_entries = entries[:-n_test]
    test_entries = entries[-n_test:]
    for name, ents in [("train", train_entries), ("test", test_entries)]:
        manifest_path = ai3_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  AISHELL-3 manifest {name}: already exists")
            continue
        with open(manifest_path, "w") as f:
            for e in ents:
                f.write(json.dumps(e) + "\n")
        print(f"  AISHELL-3 manifest {name}: {len(ents)} utterances")


def main():
    parser = argparse.ArgumentParser(description="Prepare TTS/Vocoder datasets")
    parser.add_argument("--data-root", type=str, required=True, help="Root data directory")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "speech" / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing TTS/Vocoder datasets ===", flush=True)
    download_ljspeech(out_dir)
    download_vctk_subset(out_dir, n_speakers=5)
    download_aishell3_subset(out_dir, n_speakers=5)
    print("=== TTS data preparation complete ===")


if __name__ == "__main__":
    main()
