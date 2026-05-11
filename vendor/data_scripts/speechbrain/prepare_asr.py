#!/usr/bin/env python3
"""Prepare ASR datasets for speech tasks.

Downloads and pre-processes:
  - LibriSpeech train-clean-100 (~30GB audio)
  - AISHELL-1 (~15GB audio)
  - MLS Spanish opus (~14GB audio)

Output: {data_root}/speech/asr/{librispeech-100,aishell-1,mls-spanish}/

Usage:
    python vendor/data_scripts/speechbrain/prepare_asr.py --data-root vendor/data
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf

# Container path prefix — manifests must use this, not host paths
CONTAINER_PREFIX = "/data/speech/asr"


def _container_path(host_path: Path, dataset_dir: Path, dataset_name: str) -> str:
    """Convert a host path to a container path for manifest entries."""
    rel = host_path.relative_to(dataset_dir)
    return f"{CONTAINER_PREFIX}/{dataset_name}/{rel}"


def download_librispeech(out_dir: Path):
    """Download LibriSpeech train-clean-100 from OpenSLR."""
    ls_dir = out_dir / "librispeech-100"
    ls_dir.mkdir(parents=True, exist_ok=True)

    archive = ls_dir / "train-clean-100.tar.gz"
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"

    if (ls_dir / "LibriSpeech" / "train-clean-100").exists():
        print("  LibriSpeech train-clean-100: already downloaded")
    else:
        print("  Downloading LibriSpeech train-clean-100...", flush=True)
        subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
        print("  Extracting...", flush=True)
        subprocess.run(["tar", "xzf", str(archive), "-C", str(ls_dir)], check=True)
        archive.unlink()

    # Also download test sets
    for split in ["test-clean", "test-other"]:
        split_url = f"https://www.openslr.org/resources/12/{split}.tar.gz"
        split_archive = ls_dir / f"{split}.tar.gz"
        if (ls_dir / "LibriSpeech" / split).exists():
            print(f"  LibriSpeech {split}: already downloaded")
            continue
        print(f"  Downloading LibriSpeech {split}...", flush=True)
        subprocess.run(["wget", "-c", "-q", split_url, "-O", str(split_archive)], check=True)
        subprocess.run(["tar", "xzf", str(split_archive), "-C", str(ls_dir)], check=True)
        split_archive.unlink()

    print("  LibriSpeech: done")

    # Generate manifests
    generate_librispeech_manifests(ls_dir)


def generate_librispeech_manifests(ls_dir: Path):
    """Generate JSONL manifests for LibriSpeech splits."""
    ls_root = ls_dir / "LibriSpeech"
    split_map = {
        "train": "train-clean-100",
        "test": "test-clean",
    }
    for manifest_name, split_name in split_map.items():
        manifest_path = ls_dir / f"{manifest_name}.json"
        if manifest_path.exists():
            print(f"  LibriSpeech manifest {manifest_name}: already exists")
            continue
        split_dir = ls_root / split_name
        if not split_dir.exists():
            print(f"  LibriSpeech split {split_name} not found, skipping manifest")
            continue
        entries = []
        for trans_file in sorted(split_dir.rglob("*.trans.txt")):
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) < 2:
                        continue
                    utt_id, text = parts
                    # Find corresponding FLAC file
                    flac_path = trans_file.parent / f"{utt_id}.flac"
                    if not flac_path.exists():
                        continue
                    info = sf.info(str(flac_path))
                    container_wav = _container_path(flac_path, ls_dir, "librispeech-100")
                    entries.append({
                        "id": utt_id,
                        "wav": container_wav,
                        "text": text,
                        "duration": info.duration,
                    })
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  LibriSpeech manifest {manifest_name}: {len(entries)} utterances")


def download_aishell1(out_dir: Path):
    """Download AISHELL-1 from OpenSLR."""
    ai_dir = out_dir / "aishell-1"
    ai_dir.mkdir(parents=True, exist_ok=True)

    if (ai_dir / "data_aishell").exists():
        print("  AISHELL-1: already downloaded")
    else:
        url = "https://www.openslr.org/resources/33/data_aishell.tgz"
        archive = ai_dir / "data_aishell.tgz"
        print("  Downloading AISHELL-1...", flush=True)
        subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
        print("  Extracting...", flush=True)
        subprocess.run(["tar", "xzf", str(archive), "-C", str(ai_dir)], check=True)
        archive.unlink()

        # Extract inner wav archives
        wav_dir = ai_dir / "data_aishell" / "wav"
        for tgz in sorted(wav_dir.glob("*.tar.gz")):
            print(f"  Extracting {tgz.name}...", flush=True)
            subprocess.run(["tar", "xzf", str(tgz), "-C", str(wav_dir)], check=True)
            tgz.unlink()

    print("  AISHELL-1: done")

    # Generate manifests
    generate_aishell1_manifests(ai_dir)


def generate_aishell1_manifests(ai_dir: Path):
    """Generate JSONL manifests for AISHELL-1."""
    transcript_file = ai_dir / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    if not transcript_file.exists():
        print("  AISHELL-1 transcript file not found, skipping manifest")
        return

    # Read transcripts
    transcripts = {}
    with open(transcript_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1].replace(" ", "")  # remove spaces between chars

    wav_dir = ai_dir / "data_aishell" / "wav"
    # AISHELL-1 splits: train, dev, test
    split_map = {"train": "train", "test": "test"}
    for manifest_name, split_name in split_map.items():
        manifest_path = ai_dir / f"{manifest_name}.json"
        if manifest_path.exists():
            print(f"  AISHELL-1 manifest {manifest_name}: already exists")
            continue
        split_dir = wav_dir / split_name
        if not split_dir.exists():
            print(f"  AISHELL-1 split {split_name} not found, skipping manifest")
            continue
        entries = []
        for wav_file in sorted(split_dir.rglob("*.wav")):
            utt_id = wav_file.stem
            text = transcripts.get(utt_id, "")
            if not text:
                continue
            info = sf.info(str(wav_file))
            container_wav = _container_path(wav_file, ai_dir, "aishell-1")
            entries.append({
                "id": utt_id,
                "wav": container_wav,
                "text": text,
                "duration": info.duration,
            })
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  AISHELL-1 manifest {manifest_name}: {len(entries)} utterances")


def download_mls_spanish(out_dir: Path):
    """Download MLS Spanish opus dataset (~14GB)."""
    mls_dir = out_dir / "mls-spanish"
    mls_dir.mkdir(parents=True, exist_ok=True)

    marker = mls_dir / ".done"
    if marker.exists():
        print("  MLS Spanish: already downloaded")
        generate_mls_spanish_manifests(mls_dir)
        return

    url = "https://dl.fbaipublicfiles.com/mls/mls_spanish_opus.tar.gz"
    archive = mls_dir / "mls_spanish_opus.tar.gz"

    print("  Downloading MLS Spanish opus (~14GB)...", flush=True)
    subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
    print("  Extracting...", flush=True)
    subprocess.run(["tar", "xzf", str(archive), "-C", str(mls_dir)], check=True)
    archive.unlink()

    marker.touch()
    print("  MLS Spanish: done")
    generate_mls_spanish_manifests(mls_dir)


def generate_mls_spanish_manifests(mls_dir: Path):
    """Generate JSONL manifests for MLS Spanish.

    MLS structure: mls_spanish_opus/{train,dev,test}/audio/<speaker>/<book>/*.opus
    Transcripts: mls_spanish_opus/{train,dev,test}/transcripts.txt
    Transcript format: <uttid>\\t<text>
    """
    mls_root = mls_dir / "mls_spanish_opus"
    if not mls_root.exists():
        # Try without subdirectory (in case it extracted flat)
        mls_root = mls_dir
        if not (mls_root / "train").exists():
            print("  MLS Spanish data directory not found, skipping manifest")
            return

    split_map = {"train": "train", "test": "test"}
    for manifest_name, split_name in split_map.items():
        manifest_path = mls_dir / f"{manifest_name}.json"
        if manifest_path.exists():
            print(f"  MLS Spanish manifest {manifest_name}: already exists")
            continue

        split_dir = mls_root / split_name
        if not split_dir.exists():
            print(f"  MLS Spanish split {split_name} not found, skipping manifest")
            continue

        # Read transcripts
        transcripts = {}
        trans_file = split_dir / "transcripts.txt"
        if trans_file.exists():
            with open(trans_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]
        else:
            print(f"  MLS Spanish transcripts.txt not found for {split_name}")

        # Find all audio files
        audio_dir = split_dir / "audio"
        entries = []
        if audio_dir.exists():
            for opus_file in sorted(audio_dir.rglob("*.opus")):
                utt_id = opus_file.stem
                text = transcripts.get(utt_id, "")
                if not text:
                    # Try with full path-based ID: speaker_book_uttid
                    continue
                try:
                    info = sf.info(str(opus_file))
                    duration = info.duration
                except Exception:
                    # If soundfile can't read opus, estimate from file size
                    # or skip
                    continue
                container_wav = _container_path(opus_file, mls_dir, "mls-spanish")
                entries.append({
                    "id": utt_id,
                    "wav": container_wav,
                    "text": text,
                    "duration": duration,
                })
        else:
            print(f"  MLS Spanish audio dir not found for {split_name}")

        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  MLS Spanish manifest {manifest_name}: {len(entries)} utterances")


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR datasets")
    parser.add_argument("--data-root", type=str, required=True, help="Root data directory")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "speech" / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing ASR datasets ===", flush=True)
    download_librispeech(out_dir)
    download_aishell1(out_dir)
    download_mls_spanish(out_dir)
    print("=== ASR data preparation complete ===")


if __name__ == "__main__":
    main()
