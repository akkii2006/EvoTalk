import os
import json
import shutil
import random
import argparse

import torch

import prepare

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None, unit=None):
        total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        last = [-1]
        for i, x in enumerate(iterable):
            if total:
                pct = int(100 * i / total)
                if pct != last[0] and pct % 5 == 0:
                    print(f"  {desc or ''} {pct:3d}%  ({i}/{total})", flush=True)
                    last[0] = pct
            yield x
        print(f"  {desc or ''} 100%  ({total}/{total})", flush=True)


SRC_DIR = "data_long"
OUT_DIR = "data_xlong"


def load_manifest(data_dir, split):
    with open(os.path.join(data_dir, split, "manifest.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def silence_block(n_frames, n_mels, sil_id, pitch_value):
    import math
    mel = torch.full((n_frames, n_mels), math.log(prepare.LOG_CLIP), dtype=torch.float32)
    return {
        "phonemes": torch.tensor([sil_id], dtype=torch.long),
        "durations": torch.tensor([float(n_frames)], dtype=torch.float32),
        "mel": mel,
        "energy": mel.norm(dim=1),
        "pitch": torch.full((n_frames,), float(pitch_value), dtype=torch.float32),
    }


def concat_items(items, sil_frames, n_mels, sil_id):
    phonemes, durations, mels, pitches, energies = [], [], [], [], []
    for i, item in enumerate(items):
        if i > 0:
            sil = silence_block(sil_frames, n_mels, sil_id, pitches[-1][-1])
            phonemes.append(sil["phonemes"])
            durations.append(sil["durations"])
            mels.append(sil["mel"])
            pitches.append(sil["pitch"])
            energies.append(sil["energy"])
        phonemes.append(item["phonemes"])
        durations.append(item["durations"].float())
        mels.append(item["mel"])
        pitches.append(item["pitch"])
        energies.append(item["energy"])
    return {
        "phonemes": torch.cat(phonemes),
        "speaker_id": items[0]["speaker_id"],
        "durations": torch.cat(durations),
        "mel": torch.cat(mels),
        "pitch": torch.cat(pitches),
        "energy": torch.cat(energies),
    }


def validate(item):
    n_frames = item["mel"].size(0)
    assert int(item["durations"].sum().item()) == n_frames
    assert item["pitch"].size(0) == n_frames
    assert item["energy"].size(0) == n_frames
    assert item["durations"].size(0) == item["phonemes"].size(0)
    return n_frames


def pack(paths, out_split_dir, meta, max_frames, max_phonemes, sil_frames,
         keep_ratio, rng, desc):
    n_mels = meta["n_mels"]
    sil_id = meta["token2id"]["<sil>"]

    written, buffer, bf, bp = [], [], 0, 0
    stats = {"packed": 0, "single": 0, "frames": []}

    def flush():
        nonlocal buffer, bf, bp
        if not buffer:
            return
        item = concat_items(buffer, sil_frames, n_mels, sil_id) if len(buffer) > 1 else buffer[0]
        n_frames = validate(item)
        out_path = os.path.join(out_split_dir, f"xlong_{len(written):06d}.pt")
        torch.save(item, out_path)
        written.append(out_path)
        stats["frames"].append(n_frames)
        stats["packed" if len(buffer) > 1 else "single"] += 1
        buffer, bf, bp = [], 0, 0

    for path in tqdm(paths, total=len(paths), desc=desc, unit="ex"):
        item = torch.load(path, weights_only=True)
        n_frames = item["mel"].size(0)
        n_phon = item["phonemes"].size(0)

        if n_frames > max_frames or n_phon > max_phonemes:
            continue

        if rng.random() < keep_ratio:
            flush()
            buffer = [item]
            bf, bp = n_frames, n_phon
            flush()
            continue

        ef = sil_frames if buffer else 0
        ep = 1 if buffer else 0
        if bf + ef + n_frames > max_frames or bp + ep + n_phon > max_phonemes:
            flush()
            ef, ep = 0, 0

        buffer.append(item)
        bf += ef + n_frames
        bp += ep + n_phon

    flush()

    with open(os.path.join(out_split_dir, "manifest.txt"), "w") as f:
        for p in written:
            f.write(p + "\n")

    return written, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default=SRC_DIR)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--max_frames", type=int, default=4000)
    parser.add_argument("--max_phonemes", type=int, default=2000)
    parser.add_argument("--sil_frames", type=int, default=30)
    parser.add_argument("--keep_ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    with open(os.path.join(args.src_dir, "metadata.json")) as f:
        meta = json.load(f)
    if "<sil>" not in meta["token2id"]:
        raise SystemExit("metadata.json has no <sil> token")

    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(os.path.join(args.src_dir, "metadata.json"),
                os.path.join(args.out_dir, "metadata.json"))

    max_frames_seen = 0
    max_phon_seen = 0

    for split in ("train", "val"):
        paths = load_manifest(args.src_dir, split)
        out_split_dir = os.path.join(args.out_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)

        rng = random.Random(args.seed)
        kr = args.keep_ratio if split == "train" else 0.0

        written, stats = pack(paths, out_split_dir, meta, args.max_frames,
                              args.max_phonemes, args.sil_frames, kr, rng, split)

        frames = stats["frames"]
        if not frames:
            raise SystemExit(f"no examples written for split '{split}'")
        max_frames_seen = max(max_frames_seen, max(frames))
        secs = [n * prepare.HOP_LENGTH / prepare.SAMPLE_RATE for n in frames]
        print(f"[{split}] {len(paths)} -> {len(written)} examples "
              f"({stats['packed']} packed, {stats['single']} single)")
        print(f"[{split}] frames min/mean/max: {min(frames)} / "
              f"{sum(frames)//len(frames)} / {max(frames)}")
        print(f"[{split}] seconds min/mean/max: {min(secs):.1f} / "
              f"{sum(secs)/len(secs):.1f} / {max(secs):.1f}")

    print(f"\nwrote {args.out_dir}")
    print(f"longest example is {max_frames_seen} frames")
    print(f"run long_finetune.py with --max_mel_len >= {max_frames_seen}")


if __name__ == "__main__":
    main()