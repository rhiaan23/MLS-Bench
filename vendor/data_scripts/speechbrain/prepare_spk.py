#!/usr/bin/env python3
"""Prepare speaker verification datasets for speech-speaker-embedding task.

Downloads:
  - VoxCeleb1 dev + test (~31GB total, from HuggingFace mirror ProgramComputer/voxceleb)
  - CN-Celeb eval subset (~5GB, Chinese speaker verification)

Output: {data_root}/speech/spk/{voxceleb1,cnceleb}/

Usage:
    python vendor/data_scripts/speechbrain/prepare_spk.py --data-root vendor/data
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import soundfile as sf

# Container path prefix — manifests must use this, not host paths
CONTAINER_PREFIX = "/data/speech/spk"


def _container_path(host_path: Path, dataset_dir: Path, dataset_name: str) -> str:
    """Convert a host path to a container path for manifest entries."""
    rel = host_path.relative_to(dataset_dir)
    return f"{CONTAINER_PREFIX}/{dataset_name}/{rel}"


def download_voxceleb1(out_dir: Path):
    """Download VoxCeleb1 dataset from HuggingFace mirror (ProgramComputer/voxceleb).

    Downloads:
      - vox1_test_wav.zip (~1GB, test trials)
      - vox1_dev_wav_partaa through vox1_dev_wav_partad (split zip, ~30GB total)
    Extracts to: {out_dir}/voxceleb1/wav/<speaker_id>/<video_id>/<utterance>.wav
    """
    vox_dir = out_dir / "voxceleb1"
    vox_dir.mkdir(parents=True, exist_ok=True)

    marker = vox_dir / ".done"
    if marker.exists() and (vox_dir / "wav").exists():
        print("  VoxCeleb1: already prepared")
        generate_voxceleb1_manifests(vox_dir)
        return

    # Download trial lists (freely available)
    trials_url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"
    trials_file = vox_dir / "veri_test2.txt"
    if not trials_file.exists():
        print("  Downloading VoxCeleb1 trial lists...", flush=True)
        subprocess.run(["wget", "-c", "-q", trials_url, "-O", str(trials_file)], check=False)

    trials_h_url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt"
    trials_h_file = vox_dir / "list_test_hard2.txt"
    if not trials_h_file.exists():
        subprocess.run(["wget", "-c", "-q", trials_h_url, "-O", str(trials_h_file)], check=False)

    # Download from HuggingFace mirror using huggingface_hub
    print("  Downloading VoxCeleb1 from HuggingFace mirror...", flush=True)

    try:
        from huggingface_hub import hf_hub_download

        hf_repo = "ProgramComputer/voxceleb"

        # Download test set (~1GB)
        test_zip = vox_dir / "vox1_test_wav.zip"
        if not test_zip.exists() and not (vox_dir / "wav").exists():
            print("  Downloading vox1_test_wav.zip (~1GB)...", flush=True)
            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename="vox1/vox1_test_wav.zip",
                repo_type="dataset",
                local_dir=str(vox_dir),
            )
            # hf_hub_download may place file in a subdirectory; move if needed
            downloaded_path = Path(downloaded)
            if downloaded_path != test_zip:
                downloaded_path.rename(test_zip)

        # Download dev set (split zip, ~30GB total)
        dev_parts = ["vox1_dev_wav_partaa", "vox1_dev_wav_partab",
                      "vox1_dev_wav_partac", "vox1_dev_wav_partad"]
        dev_zip = vox_dir / "vox1_dev_wav.zip"

        if not dev_zip.exists() and not (vox_dir / "wav").exists():
            for part_name in dev_parts:
                part_path = vox_dir / part_name
                if part_path.exists():
                    print(f"  {part_name}: already downloaded")
                    continue
                print(f"  Downloading {part_name}...", flush=True)
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=f"vox1/{part_name}",
                    repo_type="dataset",
                    local_dir=str(vox_dir),
                )
                downloaded_path = Path(downloaded)
                if downloaded_path != part_path:
                    downloaded_path.rename(part_path)

            # Concatenate split zip parts
            print("  Concatenating dev zip parts...", flush=True)
            part_files = [str(vox_dir / p) for p in dev_parts]
            # Check all parts exist
            if all((vox_dir / p).exists() for p in dev_parts):
                with open(str(dev_zip), "wb") as out_f:
                    for pf in part_files:
                        with open(pf, "rb") as in_f:
                            while True:
                                chunk = in_f.read(64 * 1024 * 1024)  # 64MB chunks
                                if not chunk:
                                    break
                                out_f.write(chunk)
                # Clean up parts
                for p in dev_parts:
                    (vox_dir / p).unlink(missing_ok=True)
            else:
                missing = [p for p in dev_parts if not (vox_dir / p).exists()]
                print(f"  WARNING: Missing dev parts: {missing}")

        # Extract test zip
        if test_zip.exists():
            print("  Extracting vox1_test_wav.zip...", flush=True)
            subprocess.run(["unzip", "-q", "-o", str(test_zip), "-d", str(vox_dir)], check=True)
            test_zip.unlink()

        # Extract dev zip
        if dev_zip.exists():
            print("  Extracting vox1_dev_wav.zip...", flush=True)
            subprocess.run(["unzip", "-q", "-o", str(dev_zip), "-d", str(vox_dir)], check=True)
            dev_zip.unlink()

        if (vox_dir / "wav").exists():
            marker.touch()
            print("  VoxCeleb1: done")
        else:
            print("  WARNING: VoxCeleb1 wav directory not found after extraction")

    except ImportError:
        print("  WARNING: 'huggingface_hub' package not available.")
        print("  Install with: pip install huggingface_hub")
        print("  Alternatively, manually download VoxCeleb1 and extract to:")
        print(f"    {vox_dir}/wav/<speaker_id>/<video_id>/<utterance>.wav")
        return
    except Exception as e:
        print(f"  WARNING: VoxCeleb1 download failed: {e}")
        print("  You may need to authenticate with HuggingFace:")
        print("    huggingface-cli login")
        print("  Then accept terms at: https://huggingface.co/datasets/ProgramComputer/voxceleb")
        return

    generate_voxceleb1_manifests(vox_dir)


def download_cnceleb(out_dir: Path):
    """Download CN-Celeb eval subset from OpenSLR."""
    cn_dir = out_dir / "cnceleb"
    cn_dir.mkdir(parents=True, exist_ok=True)

    marker = cn_dir / ".done"
    if marker.exists():
        print("  CN-Celeb: already downloaded")
        generate_cnceleb_manifests(cn_dir)
        return

    # CN-Celeb eval set from OpenSLR
    url = "https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz"
    archive = cn_dir / "cn-celeb_v2.tar.gz"

    if not (cn_dir / "data").exists():
        print("  Downloading CN-Celeb...", flush=True)
        subprocess.run(["wget", "-c", "-q", url, "-O", str(archive)], check=True)
        print("  Extracting...", flush=True)
        subprocess.run(["tar", "xzf", str(archive), "-C", str(cn_dir)], check=True)
        archive.unlink()

    marker.touch()
    print("  CN-Celeb: done")
    generate_cnceleb_manifests(cn_dir)


def generate_voxceleb1_manifests(vox_dir: Path):
    """Generate JSONL manifests for VoxCeleb1.

    Uses official speaker-based dev/test split: test speakers are those
    appearing in veri_test2.txt (40 speakers), dev speakers (1211) are
    used for training. This prevents data leakage.
    """
    wav_dir = vox_dir / "wav"
    if not wav_dir.exists():
        print("  VoxCeleb1 wav dir not found, skipping manifest")
        return

    train_manifest = vox_dir / "train.json"
    test_manifest = vox_dir / "test.json"
    if train_manifest.exists() and test_manifest.exists():
        print("  VoxCeleb1 manifests: already exist")
        return

    # Extract test speakers from standard trial list (40 speakers)
    test_speakers = set()
    trials_file = vox_dir / "veri_test2.txt"
    if trials_file.exists():
        with open(trials_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    test_speakers.add(parts[1].split("/")[0])
                    test_speakers.add(parts[2].split("/")[0])
    print(f"  VoxCeleb1 test speakers (from veri_test2.txt): {len(test_speakers)}")

    entries = []
    for spk_dir in sorted(wav_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker = spk_dir.name
        for wav_file in sorted(spk_dir.rglob("*.wav")):
            try:
                info = sf.info(str(wav_file))
            except Exception:
                continue
            container_wav = _container_path(wav_file, vox_dir, "voxceleb1")
            entries.append({
                "id": wav_file.stem,
                "wav": container_wav,
                "speaker": speaker,
                "duration": info.duration,
            })

    # Speaker-based split: train = dev speakers, test = test speakers
    train_entries = [e for e in entries if e["speaker"] not in test_speakers]
    test_entries = [e for e in entries if e["speaker"] in test_speakers]

    for name, ents in [("train", train_entries), ("test", test_entries)]:
        manifest_path = vox_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  VoxCeleb1 manifest {name}: already exists")
            continue
        with open(manifest_path, "w") as f:
            for e in ents:
                f.write(json.dumps(e) + "\n")
        print(f"  VoxCeleb1 manifest {name}: {len(ents)} utterances")


def generate_cnceleb_manifests(cn_dir: Path):
    """Generate JSONL manifests for CN-Celeb."""
    # CN-Celeb layout: CN-Celeb_flac/data/<speaker_id>/*.flac or data/<speaker_id>/*.wav
    data_dir = cn_dir / "data"
    if not data_dir.exists():
        for candidate in [cn_dir / "CN-Celeb_flac" / "data", cn_dir / "CN-Celeb_flac", cn_dir / "cn-celeb_v2", cn_dir]:
            if candidate.exists() and (any(candidate.rglob("*.wav")) or any(candidate.rglob("*.flac"))):
                data_dir = candidate
                break
        else:
            print("  CN-Celeb data not found, skipping manifest")
            return

    train_manifest = cn_dir / "train.json"
    test_manifest = cn_dir / "test.json"
    if train_manifest.exists() and test_manifest.exists():
        print("  CN-Celeb manifests: already exist")
        return

    entries = []
    for wav_file in sorted(list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.flac"))):
        speaker = wav_file.parent.name
        try:
            info = sf.info(str(wav_file))
        except Exception:
            continue
        container_wav = _container_path(wav_file, cn_dir, "cnceleb")
        entries.append({
            "id": wav_file.stem,
            "wav": container_wav,
            "speaker": speaker,
            "duration": info.duration,
        })

    n_test = max(1, len(entries) // 10)
    train_entries = entries[:-n_test]
    test_entries = entries[-n_test:]

    for name, ents in [("train", train_entries), ("test", test_entries)]:
        manifest_path = cn_dir / f"{name}.json"
        if manifest_path.exists():
            print(f"  CN-Celeb manifest {name}: already exists")
            continue
        with open(manifest_path, "w") as f:
            for e in ents:
                f.write(json.dumps(e) + "\n")
        print(f"  CN-Celeb manifest {name}: {len(ents)} utterances")


def main():
    parser = argparse.ArgumentParser(description="Prepare speaker verification datasets")
    parser.add_argument("--data-root", type=str, required=True, help="Root data directory")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "speech" / "spk"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing speaker verification datasets ===", flush=True)
    download_voxceleb1(out_dir)
    download_cnceleb(out_dir)
    print("=== Speaker data preparation complete ===")


if __name__ == "__main__":
    main()
