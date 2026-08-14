import os
import json
import math
import random
import shutil
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


SRC_DIR = "data"
OUT_DIR = "data_long"
SPLIT_SEED = 1234


def load_manifest(data_dir, split):
    path = os.path.join(data_dir, split, "manifest.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def silence_segment(n_frames, n_mels, sil_id):
    mel = torch.full((n_frames, n_mels), math.log(prepare.LOG_CLIP), dtype=torch.float32)
    energy = mel.norm(dim=1)
    return {
        "phonemes": torch.tensor([sil_id], dtype=torch.long),
        "durations": torch.tensor([n_frames], dtype=torch.float32),
        "mel": mel,
        "energy": energy,
    }


def concat_items(items, sil_frames, n_mels, sil_id):
    phonemes, durations, mels, pitches, energies = [], [], [], [], []

    for i, item in enumerate(items):
        if i > 0:
            sil = silence_segment(sil_frames, n_mels, sil_id)
            edge = torch.full((sil_frames,), float(pitches[-1][-1]), dtype=torch.float32)
            phonemes.append(sil["phonemes"])
            durations.append(sil["durations"])
            mels.append(sil["mel"])
            pitches.append(edge)
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
    dur_sum = int(item["durations"].sum().item())
    assert dur_sum == n_frames, f"durations {dur_sum} != frames {n_frames}"
    assert item["pitch"].size(0) == n_frames, "pitch length mismatch"
    assert item["energy"].size(0) == n_frames, "energy length mismatch"
    assert item["durations"].size(0) == item["phonemes"].size(0), "phoneme/duration mismatch"
    return n_frames


def pack(paths, out_split_dir, meta, max_frames, max_phonemes, sil_frames,
         keep_short_ratio, rng, desc="packing"):
    n_mels = meta["n_mels"]
    sil_id = meta["token2id"]["<sil>"]

    written, buffer, buf_frames, buf_phonemes = [], [], 0, 0
    stats = {"packs": 0, "singles": 0, "frames": []}

    def flush():
        nonlocal buffer, buf_frames, buf_phonemes
        if not buffer:
            return
        item = concat_items(buffer, sil_frames, n_mels, sil_id) if len(buffer) > 1 else buffer[0]
        n_frames = validate(item)
        idx = len(written)
        out_path = os.path.join(out_split_dir, f"pack_{idx:06d}.pt")
        torch.save(item, out_path)
        written.append(out_path)
        stats["frames"].append(n_frames)
        if len(buffer) > 1:
            stats["packs"] += 1
        else:
            stats["singles"] += 1
        buffer, buf_frames, buf_phonemes = [], 0, 0

    for path in tqdm(paths, total=len(paths), desc=desc, unit="utt"):
        item = torch.load(path, weights_only=True)
        n_frames = item["mel"].size(0)
        n_phon = item["phonemes"].size(0)

        if n_frames > max_frames or n_phon > max_phonemes:
            continue

        if rng.random() < keep_short_ratio:
            flush()
            buffer = [item]
            buf_frames, buf_phonemes = n_frames, n_phon
            flush()
            continue

        extra_frames = sil_frames if buffer else 0
        extra_phon = 1 if buffer else 0

        if (buf_frames + extra_frames + n_frames > max_frames or
                buf_phonemes + extra_phon + n_phon > max_phonemes):
            flush()
            extra_frames, extra_phon = 0, 0

        buffer.append(item)
        buf_frames += extra_frames + n_frames
        buf_phonemes += extra_phon + n_phon

    flush()

    with open(os.path.join(out_split_dir, "manifest.txt"), "w") as f:
        for p in written:
            f.write(p + "\n")

    return written, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default=SRC_DIR)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--max_frames", type=int, default=2000)
    parser.add_argument("--max_phonemes", type=int, default=1000)
    parser.add_argument("--sil_frames", type=int, default=25)
    parser.add_argument("--keep_short_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    with open(os.path.join(args.src_dir, "metadata.json")) as f:
        meta = json.load(f)

    if "<sil>" not in meta["token2id"]:
        raise SystemExit("metadata.json has no <sil> token; cannot insert pauses.")

    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(os.path.join(args.src_dir, "metadata.json"),
                os.path.join(args.out_dir, "metadata.json"))

    for split in ("train", "val"):
        paths = load_manifest(args.src_dir, split)
        out_split_dir = os.path.join(args.out_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)

        rng = random.Random(args.seed)
        keep_ratio = args.keep_short_ratio if split == "train" else 0.0

        written, stats = pack(paths, out_split_dir, meta, args.max_frames,
                              args.max_phonemes, args.sil_frames, keep_ratio, rng,
                              desc=split)

        frames = stats["frames"]
        if not frames:
            raise SystemExit(f"no examples written for split '{split}'")
        secs = [n * prepare.HOP_LENGTH / prepare.SAMPLE_RATE for n in frames]
        print(f"[{split}] {len(paths)} utterances -> {len(written)} examples "
              f"({stats['packs']} packed, {stats['singles']} single)")
        print(f"[{split}] frames min/mean/max: {min(frames)} / "
              f"{sum(frames)//len(frames)} / {max(frames)}")
        print(f"[{split}] seconds min/mean/max: {min(secs):.1f} / "
              f"{sum(secs)/len(secs):.1f} / {max(secs):.1f}")

    print(f"\nwrote {args.out_dir} (metadata.json copied unchanged)")


if __name__ == "__main__":
    main()