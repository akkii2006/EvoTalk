"""Synthesize with EvoTalk + Vocos, chunking text so the model never has to
generate a long frame span in one shot.

Why chunk
---------
The model is trained on single short utterances, so the decoder rarely sees
frame positions far beyond a few hundred. Long spans degrade (and partially
recover) mid-sentence. Rather than asking the decoder to hold coherence over
1000+ frames, this splits text into short chunks, synthesizes each, and joins
them with pauses sized to the boundary type so it still reads as connected
speech.

Splitting is hierarchical, cheapest boundary first:
    1. sentence boundaries   (. ! ?)      -> long pause
    2. clause boundaries     (, ; : --)   -> short pause
    3. hard wrap at MAX_WORDS on a word boundary -> very short pause

Install:
    pip install vocos

Usage:
    python test_voc.py --text "hello world"
    python test_voc.py                       # interactive
    python test_voc.py --max_words 12        # tighter chunks
    python test_voc.py --max_words 999       # effectively disable wrapping
"""

import os
import re
import json
import argparse

import torch
import torchaudio

from model import EvoTalk, EvoTalkConfig
from prepare import text_to_arpabet, phonemes_to_ids


CHECKPOINT = "checkpoints/evotalk_final.pt"
METADATA = "data/metadata.json"
AUDIO_DIR = "audio"

# Target chunk size. Chunks longer than this are wrapped at a word boundary.
# ~12 words is roughly 400 frames at this speaker's rate — well inside the
# range the decoder saw constantly during training.
MAX_WORDS = 12

# Pause after each chunk, by the boundary that ended it.
PAUSE = {"sentence": 0.40, "clause": 0.20, "wrap": 0.10}


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_clauses(sentence):
    """Split on clause punctuation, keeping the punctuation with the left part."""
    parts = re.split(r"(?<=[,;:])\s+|\s+--\s+|\s+—\s+", sentence)
    return [p.strip() for p in parts if p and p.strip()]


def wrap_words(piece, max_words):
    words = piece.split()
    if len(words) <= max_words:
        return [piece]
    out = []
    for i in range(0, len(words), max_words):
        out.append(" ".join(words[i:i + max_words]))
    return out


def chunk_text(text, max_words=MAX_WORDS):
    """-> list of (chunk_text, boundary_kind) where boundary_kind describes the
    pause that should FOLLOW the chunk."""
    chunks = []
    for sentence in split_sentences(text):
        clauses = split_clauses(sentence)
        for ci, clause in enumerate(clauses):
            last_clause = ci == len(clauses) - 1
            pieces = wrap_words(clause, max_words)
            for pi, piece in enumerate(pieces):
                last_piece = pi == len(pieces) - 1
                if last_piece and last_clause:
                    kind = "sentence"
                elif last_piece:
                    kind = "clause"
                else:
                    kind = "wrap"
                chunks.append((piece, kind))
    return chunks


# ---------------------------------------------------------------------------
# Model / vocoder
# ---------------------------------------------------------------------------
def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", None) or EvoTalkConfig()
    model = EvoTalk(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_vocos(device):
    try:
        from vocos import Vocos
    except ImportError:
        raise SystemExit("Vocos is not installed. Run:  pip install vocos")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()
    return vocos


def synth_chunk(model, vocos, text, token2id, speaker_ids, duration_scale, device):
    phonemes = text_to_arpabet(text)
    if not phonemes:
        return None
    ids = torch.tensor([phonemes_to_ids(phonemes, token2id)],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        mel = model.inference(ids, speaker_ids, src_mask=None,
                              duration_scale=duration_scale)
        wav = vocos.decode(mel.float().transpose(1, 2).contiguous())
    return wav.squeeze(0).float().cpu()


def synthesize(model, vocos, meta, text, speaker_id, duration_scale, device,
               max_words=MAX_WORDS, verbose=False):
    sr = meta["sample_rate"]
    n_speakers = meta.get("n_speakers", 1)
    speaker_id = max(0, min(int(speaker_id), n_speakers - 1))
    spk = torch.tensor([speaker_id], dtype=torch.long, device=device)

    chunks = chunk_text(text, max_words)
    if not chunks:
        print("nothing to synthesize")
        return None

    pieces = []
    for piece, kind in chunks:
        wav = synth_chunk(model, vocos, piece, meta["token2id"], spk,
                          duration_scale, device)
        if wav is None:
            continue
        if verbose:
            print(f"  [{kind:>8}] {wav.numel()/sr:5.2f}s  {piece}")
        pieces.append(wav)
        pieces.append(torch.zeros(int(PAUSE[kind] * sr)))

    if not pieces:
        print("could not phonemize any chunk")
        return None
    return torch.cat(pieces[:-1])  # drop trailing pause


def save_waveform(waveform, sr, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.98
    torchaudio.save(out_path, wav, sr)
    print(f"saved -> {out_path}")


def run_repl(model, vocos, meta, device, speaker_id, duration_scale, max_words):
    sr = meta["sample_rate"]
    n_speakers = meta.get("n_speakers", 1)
    print("EvoTalk + Vocos (chunked). Commands: ':speaker N', ':scale X', "
          "':words N', ':quit'")
    idx = 0
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == ":quit":
            break
        if text.startswith(":speaker"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                speaker_id = max(0, min(int(parts[1]), n_speakers - 1))
                print(f"speaker set to {speaker_id}")
            else:
                print("usage: :speaker N")
            continue
        if text.startswith(":scale"):
            parts = text.split()
            try:
                duration_scale = float(parts[1])
                print(f"duration_scale set to {duration_scale}")
            except (IndexError, ValueError):
                print("usage: :scale X")
            continue
        if text.startswith(":words"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                max_words = int(parts[1])
                print(f"max_words set to {max_words}")
            else:
                print("usage: :words N")
            continue

        wav = synthesize(model, vocos, meta, text, speaker_id, duration_scale,
                         device, max_words, verbose=True)
        if wav is not None:
            out_path = os.path.join(AUDIO_DIR, f"voc_{idx:04d}.wav")
            save_waveform(wav, sr, out_path)
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--metadata", type=str, default=METADATA)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--out", type=str, default=os.path.join(AUDIO_DIR, "output_voc.wav"))
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--duration_scale", type=float, default=1.0)
    parser.add_argument("--max_words", type=int, default=MAX_WORDS,
                        help="hard wrap chunks longer than this many words")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_metadata(args.metadata)
    if int(meta["sample_rate"]) != 24000 or int(meta.get("n_mels", 0)) != 100:
        raise SystemExit(
            "test_voc.py expects a Vocos-matched front-end (24 kHz, 100 mels). "
            f"Got sample_rate={meta['sample_rate']}, n_mels={meta.get('n_mels')}."
        )

    model = load_model(args.checkpoint, device)
    vocos = load_vocos(device)

    if args.text is None:
        run_repl(model, vocos, meta, device, args.speaker, args.duration_scale,
                 args.max_words)
    else:
        wav = synthesize(model, vocos, meta, args.text, args.speaker,
                         args.duration_scale, device, args.max_words,
                         verbose=not args.quiet)
        if wav is not None:
            save_waveform(wav, meta["sample_rate"], args.out)


if __name__ == "__main__":
    main()