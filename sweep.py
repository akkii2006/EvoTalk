"""Synthesize one fixed script with EVERY checkpoint and zip the results.

Loads each checkpoints/evotalk_XXXXXX.pt in a step range, runs inference on a
fixed multi-sentence passage, vocodes natively with Vocos, and writes
audio_sweep/evotalk_XXXXXX.wav plus a single evotalk_sweep.zip.

Files are named by step so they sort in order in any file browser.

Usage:
    python sweep_checkpoints.py
    python sweep_checkpoints.py --min_step 5000 --max_step 110000
    python sweep_checkpoints.py --devices cuda:0,cuda:1     # split across GPUs
    python sweep_checkpoints.py --text "custom sentence."
"""

import os
import re
import glob
import json
import zipfile
import argparse
import threading

import torch
import torchaudio

from model import EvoTalk, EvoTalkConfig
from prepare import text_to_arpabet, phonemes_to_ids


CHECKPOINT_DIR = "checkpoints"
METADATA = "data/metadata.json"
OUT_DIR = "audio_sweep"
ZIP_PATH = "evotalk_sweep.zip"

# Fixed evaluation script. Deliberately long and varied: statements, a question,
# numbers-as-words, short and long sentences, and a range of phonemes, so that
# differences between checkpoints (clarity, prosody, stability over long spans)
# are easy to hear.
DEFAULT_TEXT = (
    "This is a test of EvoTalk, a text to speech model built entirely from scratch "
    "by Mistyoz AI. Every part of this system was written by hand, from the phoneme "
    "front end and the forced alignment, all the way through the encoder, the "
    "variance adaptor, and the decoder. The voice you are hearing now was trained on "
    "a single male speaker for many thousands of steps. If this recording sounds "
    "clear and natural, then the acoustic model has learned to turn written words "
    "into speech without any leaked information from the target audio. Can you tell "
    "which checkpoint sounds the best? Listen carefully to the rhythm, the pitch, and "
    "the way each sentence begins and ends. Thank you very much for listening to this "
    "demonstration."
)


def find_checkpoints(ckpt_dir, min_step, max_step):
    found = []
    for path in glob.glob(os.path.join(ckpt_dir, "evotalk_*.pt")):
        m = re.search(r"evotalk_(\d+)\.pt$", os.path.basename(path))
        if not m:
            continue  # skips evotalk_final.pt, which duplicates the last step
        step = int(m.group(1))
        if min_step <= step <= max_step:
            found.append((step, path))
    return sorted(found)


def load_meta(path):
    with open(path) as f:
        meta = json.load(f)
    if int(meta["sample_rate"]) != 24000 or int(meta.get("n_mels", 0)) != 100:
        raise SystemExit(
            "Expected a Vocos-matched front-end (24 kHz, 100 mels), got "
            f"sample_rate={meta['sample_rate']}, n_mels={meta.get('n_mels')}."
        )
    return meta


def load_vocos(device):
    try:
        from vocos import Vocos
    except ImportError:
        raise SystemExit("Vocos is not installed. Run:  pip install vocos")
    return Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()


def split_sentences(text):
    """Split into sentences. The model was trained on single utterances and its
    decoder only spans max_mel_len frames (2048 = ~21s), so a long passage must
    be synthesized sentence by sentence and joined, not fed in one shot."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def synth(model, vocos, phoneme_ids, speaker_ids, duration_scale):
    with torch.no_grad():
        mel = model.inference(
            phoneme_ids, speaker_ids, src_mask=None, duration_scale=duration_scale
        )
        wav = vocos.decode(mel.float().transpose(1, 2).contiguous())
    return wav.squeeze(0).float().cpu()


def synth_passage(model, vocos, sentences, token2id, spk, duration_scale,
                  device, sr, gap_s=0.35):
    """Synthesize each sentence separately and concatenate with a short pause."""
    gap = torch.zeros(int(gap_s * sr))
    chunks = []
    for sent in sentences:
        phonemes = text_to_arpabet(sent)
        if not phonemes:
            continue
        ids = torch.tensor([phonemes_to_ids(phonemes, token2id)],
                           dtype=torch.long, device=device)
        chunks.append(synth(model, vocos, ids, spk, duration_scale))
        chunks.append(gap)
    if not chunks:
        raise RuntimeError("no sentence could be phonemized")
    return torch.cat(chunks[:-1])  # drop the trailing gap


def save_wav(wav, sr, path):
    w = wav.unsqueeze(0) if wav.dim() == 1 else wav
    peak = w.abs().max()
    if peak > 0:
        w = w / peak * 0.98
    torchaudio.save(path, w, sr)


def run_on_device(device, jobs, meta, text, speaker, duration_scale,
                  out_dir, results, lock):
    """Process a list of (step, ckpt_path) on one device. Vocos is loaded once
    per device; the acoustic model is rebuilt per checkpoint."""
    vocos = load_vocos(device)

    sentences = split_sentences(text)
    if not sentences:
        raise SystemExit("Empty text.")
    # Validate phonemization once, up front, so failures are obvious.
    if not text_to_arpabet(sentences[0]):
        raise SystemExit("Could not phonemize the text (is espeak-ng installed?).")
    spk = torch.tensor([min(int(speaker), meta.get("n_speakers", 1) - 1)],
                       dtype=torch.long, device=device)

    for step, path in jobs:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            config = ckpt.get("config", None) or EvoTalkConfig()
            model = EvoTalk(config).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            wav = synth_passage(model, vocos, sentences, meta["token2id"], spk,
                                duration_scale, device, meta["sample_rate"])
            out = os.path.join(out_dir, f"evotalk_{step:06d}.wav")
            save_wav(wav, meta["sample_rate"], out)

            secs = wav.numel() / meta["sample_rate"]
            with lock:
                results.append(out)
                print(f"[{device}] step {step:>6} -> {out}  ({secs:.1f}s)", flush=True)

            del model, ckpt
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            with lock:
                print(f"[{device}] step {step:>6} FAILED: {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    p.add_argument("--metadata", type=str, default=METADATA)
    p.add_argument("--min_step", type=int, default=5000)
    p.add_argument("--max_step", type=int, default=110000)
    p.add_argument("--text", type=str, default=DEFAULT_TEXT)
    p.add_argument("--speaker", type=int, default=0)
    p.add_argument("--duration_scale", type=float, default=1.0)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--zip_path", type=str, default=ZIP_PATH)
    p.add_argument("--devices", type=str, default=None,
                   help="comma-separated devices, e.g. 'cuda:0,cuda:1'. "
                        "One worker thread per device.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = load_meta(args.metadata)
    ckpts = find_checkpoints(args.checkpoint_dir, args.min_step, args.max_step)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir} "
                         f"between {args.min_step} and {args.max_step}.")

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        n_gpu = torch.cuda.device_count()
        for d in devices:
            if d.startswith("cuda"):
                idx = int(d.split(":")[1]) if ":" in d else 0
                if idx >= n_gpu:
                    raise SystemExit(
                        f"Requested {d} but this machine has {n_gpu} CUDA device(s) "
                        f"(valid: {', '.join(f'cuda:{i}' for i in range(n_gpu)) or 'none'}). "
                        "Omit --devices to use the default."
                    )
    else:
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    print(f"{len(ckpts)} checkpoints (steps {ckpts[0][0]}..{ckpts[-1][0]}) "
          f"across {len(devices)} device(s): {', '.join(devices)}")
    print(f"text: {len(args.text.split())} words\n")

    # Round-robin so each device gets an interleaved share.
    shards = [ckpts[i::len(devices)] for i in range(len(devices))]

    results, lock, threads = [], threading.Lock(), []
    for device, shard in zip(devices, shards):
        if not shard:
            continue
        t = threading.Thread(
            target=run_on_device,
            args=(device, shard, meta, args.text, args.speaker,
                  args.duration_scale, args.out_dir, results, lock),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if not results:
        raise SystemExit("No audio was produced.")

    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(results):
            z.write(path, arcname=os.path.basename(path))
        z.writestr("script.txt", args.text + "\n")

    size_mb = os.path.getsize(args.zip_path) / 1e6
    print(f"\nwrote {len(results)} wav files -> {args.zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()