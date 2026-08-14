import os
import json
import argparse
import torch
import torchaudio
import torchaudio.transforms as T

from model import EvoTalk, EvoTalkConfig
from prepare import text_to_arpabet, phonemes_to_ids


CHECKPOINT = "checkpoints/evotalk_final.pt"
METADATA = "data/metadata.json"
AUDIO_DIR = "audio"
GRIFFIN_LIM_ITERS = 60


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", None)
    if config is None:
        config = EvoTalkConfig()
    model = EvoTalk(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def build_vocoder(meta):
    n_fft = meta["n_fft"]
    hop = meta["hop_length"]
    win = meta["win_length"]
    n_mels = meta["n_mels"]
    sr = meta["sample_rate"]
    f_min = meta.get("f_min", 0.0)
    f_max = meta.get("f_max", 8000.0)
    n_stft = n_fft // 2 + 1

    inverse_mel = T.InverseMelScale(
        n_stft=n_stft,
        n_mels=n_mels,
        sample_rate=sr,
        f_min=f_min,
        f_max=f_max,
    )
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        power=1.0,
        n_iter=GRIFFIN_LIM_ITERS,
    )
    return inverse_mel, griffin_lim


def mel_to_waveform(mel_log, inverse_mel, griffin_lim):
    # Training stored log-magnitude mel (power=1.0), so invert the log first.
    mel_lin = torch.exp(mel_log).squeeze(0).transpose(0, 1).contiguous()  # (n_mels, T)
    mel_lin = torch.clamp(mel_lin, min=1e-5)
    spec = inverse_mel(mel_lin)          # (n_stft, T)
    waveform = griffin_lim(spec)         # (samples,)
    return waveform


def text_to_ids(text, token2id, device):
    phonemes = text_to_arpabet(text)
    if not phonemes:
        return None
    ids = phonemes_to_ids(phonemes, token2id)
    return torch.tensor([ids], dtype=torch.long, device=device)


def synthesize(model, meta, inverse_mel, griffin_lim, text, speaker_id, duration_scale, device):
    token2id = meta["token2id"]
    phoneme_ids = text_to_ids(text, token2id, device)
    if phoneme_ids is None:
        print("could not phonemize the given text")
        return None

    n_speakers = meta.get("n_speakers", 1)
    speaker_id = max(0, min(int(speaker_id), n_speakers - 1))
    speaker_ids = torch.tensor([speaker_id], dtype=torch.long, device=device)

    with torch.no_grad():
        mel_out = model.inference(
            phoneme_ids, speaker_ids, src_mask=None, duration_scale=duration_scale
        )

    waveform = mel_to_waveform(mel_out.float().cpu(), inverse_mel, griffin_lim)
    return waveform


def save_waveform(waveform, sr, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.98
    torchaudio.save(out_path, wav, sr)
    print(f"saved -> {out_path}")


def run_repl(model, meta, inverse_mel, griffin_lim, device, speaker_id, duration_scale):
    sr = meta["sample_rate"]
    n_speakers = meta.get("n_speakers", 1)
    print("EvoTalk interactive synthesis. Commands: ':speaker N', ':scale X', ':quit'")
    print(f"speaker: {speaker_id} (available 0..{n_speakers - 1}) | duration_scale: {duration_scale}")
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

        waveform = synthesize(
            model, meta, inverse_mel, griffin_lim, text, speaker_id, duration_scale, device
        )
        if waveform is not None:
            out_path = os.path.join(AUDIO_DIR, f"utt_{idx:04d}.wav")
            save_waveform(waveform, sr, out_path)
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--metadata", type=str, default=METADATA)
    parser.add_argument("--text", type=str, default=None, help="text to synthesize (omit for interactive mode)")
    parser.add_argument("--out", type=str, default=os.path.join(AUDIO_DIR, "output.wav"))
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--duration_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_metadata(args.metadata)
    model, config = load_model(args.checkpoint, device)
    inverse_mel, griffin_lim = build_vocoder(meta)

    if args.text is None:
        run_repl(model, meta, inverse_mel, griffin_lim, device, args.speaker, args.duration_scale)
    else:
        waveform = synthesize(
            model, meta, inverse_mel, griffin_lim,
            args.text, args.speaker, args.duration_scale, device,
        )
        if waveform is not None:
            save_waveform(waveform, meta["sample_rate"], args.out)


if __name__ == "__main__":
    main()
