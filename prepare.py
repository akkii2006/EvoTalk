import os
import io
import json
import random
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset, Audio
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset: Hi-Fi TTS (MikhailT/hifi-tts), single male speaker.
#
#   speaker 9017  John Van Stan  (M)  ~53 h   <- default, largest single voice
#   speaker 6097  Phil Benson    (M)  ~30 h
#   speaker 92    Cori Samuel    (F)  ~27 h
#   ... (see --list_speakers)
#
# 44.1 kHz FLAC, resampled to 24 kHz here. Columns: speaker, file, duration,
# text (already lowercased / de-punctuated), text_normalized, audio.
# ---------------------------------------------------------------------------
HF_REPO = "MikhailT/hifi-tts"
HF_CONFIG = "clean"
# With the "clean" config the splits are named train/dev/test.
# (The "train.clean" style names only exist under the "all" config.)
HF_SPLIT = "train"
DEFAULT_SPEAKER = "9017"

SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
# Mel front-end is matched EXACTLY to the pretrained Vocos vocoder
# (charactr/vocos-mel-24khz): 100 bands, f_max = Nyquist (24000/2 = 12000),
# power=1 magnitude, log floor 1e-7. This makes the model's mel a native Vocos
# feature, so inference vocodes with Vocos directly (no lossy adapter).
N_MELS = 100
F_MIN = 0.0
F_MAX = 12000.0
LOG_CLIP = 1e-7
MIN_DURATION = 0.5
MAX_DURATION = 20.0

# Minimum frames per phoneme. Must match MIN_DUR in model.py.
MIN_DUR = 1

# Fraction of the speaker's utterances held out for validation.
VAL_FRACTION = 0.02
VAL_MIN = 20
VAL_MAX = 200
SPLIT_SEED = 1234

PHONEME_VOCAB = [
    "<pad>", "<unk>", "<sil>",
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]

# ---- module-level singletons (built once, reused, kept on device) ----------
_DEVICE = None
_MEL_TRANSFORM = None
_RESAMPLERS = {}
_ALIGNER_MODEL = None
_ALIGNER_DICT = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def get_mel_transform():
    global _MEL_TRANSFORM
    if _MEL_TRANSFORM is None:
        _MEL_TRANSFORM = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=1.0,
        ).to(get_device())
    return _MEL_TRANSFORM


def get_resampler(orig_sr, new_sr):
    key = (int(orig_sr), int(new_sr))
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = T.Resample(orig_freq=orig_sr, new_freq=new_sr).to(get_device())
    return _RESAMPLERS[key]


def build_vocab():
    return {tok: i for i, tok in enumerate(PHONEME_VOCAB)}


def phonemes_to_ids(phonemes, token2id):
    unk = token2id["<unk>"]
    return [token2id.get(p, unk) for p in phonemes]


def get_aligner():
    global _ALIGNER_MODEL, _ALIGNER_DICT
    if _ALIGNER_MODEL is not None:
        return _ALIGNER_MODEL, _ALIGNER_DICT, get_device()
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(get_device())
    dictionary = bundle.get_dict(star=None)
    _ALIGNER_MODEL = model
    _ALIGNER_DICT = dictionary
    return model, dictionary, get_device()


# ---------------------------------------------------------------------------
# Text -> ARPABET (espeak -> IPA -> ARPABET)
# ---------------------------------------------------------------------------
IPA_TO_ARPABET = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "g": "G",
    "f": "F", "v": "V", "s": "S", "z": "Z", "h": "HH",
    "m": "M", "n": "N", "l": "L", "r": "R", "w": "W", "j": "Y",
    "tʃ": "CH", "dʒ": "JH", "ŋ": "NG", "ʃ": "SH", "ʒ": "ZH",
    "θ": "TH", "ð": "DH",
    "i": "IY", "ɪ": "IH", "e": "EY", "ɛ": "EH", "æ": "AE",
    "ɑ": "AA", "ɔ": "AO", "o": "OW", "ʊ": "UH", "u": "UW",
    "ʌ": "AH", "ə": "AH", "ɚ": "ER", "ɝ": "ER",
    "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY",
    "eɪ": "EY", "oʊ": "OW",
}

MULTI_CHAR_IPA = sorted([k for k in IPA_TO_ARPABET if len(k) > 1], key=len, reverse=True)

_ESPEAK_BACKEND = None


def get_espeak_backend():
    global _ESPEAK_BACKEND
    if _ESPEAK_BACKEND is None:
        from phonemizer.backend import EspeakBackend
        _ESPEAK_BACKEND = EspeakBackend(
            "en-us",
            preserve_punctuation=False,
            with_stress=False,
            language_switch="remove-flags",
        )
    return _ESPEAK_BACKEND


def ipa_to_arpabet(ipa_tokens):
    out = []
    for token in ipa_tokens:
        if token in IPA_TO_ARPABET:
            out.append(IPA_TO_ARPABET[token])
            continue
        i = 0
        while i < len(token):
            found = False
            for multi in MULTI_CHAR_IPA:
                if token[i:i + len(multi)] == multi:
                    out.append(IPA_TO_ARPABET[multi])
                    i += len(multi)
                    found = True
                    break
            if not found:
                ch = token[i]
                if ch in IPA_TO_ARPABET:
                    out.append(IPA_TO_ARPABET[ch])
                i += 1
    return out


def text_to_arpabet(text):
    try:
        backend = get_espeak_backend()
        result = backend.phonemize([text.lower().strip()], strip="all", njobs=1)
        ipa_tokens = result[0].split()
        arpabet = ipa_to_arpabet(ipa_tokens)
        return arpabet if arpabet else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Acoustic features (GPU)
# ---------------------------------------------------------------------------
def extract_mel(waveform):
    """waveform: (1, T) on device -> (n_frames, n_mels) log-magnitude mel on device.
    Matches Vocos safe_log: log(clip(mel, 1e-7))."""
    mel = get_mel_transform()(waveform)
    return torch.log(torch.clamp(mel, min=LOG_CLIP)).squeeze(0).T


def extract_pitch_fast(waveform, n_frames):
    """Vectorized frame-wise autocorrelation pitch (Hz), 0 for unvoiced frames.

    Equivalent to a per-frame autocorrelation search, but computed as a single
    batched FFT over all frames on the GPU (instead of a Python loop)."""
    device = waveform.device
    sig = waveform.squeeze(0)
    need = (n_frames - 1) * HOP_LENGTH + WIN_LENGTH
    if sig.numel() < need:
        sig = torch.cat([sig, torch.zeros(need - sig.numel(), device=device)])

    idx = (torch.arange(n_frames, device=device).unsqueeze(1) * HOP_LENGTH
           + torch.arange(WIN_LENGTH, device=device).unsqueeze(0))
    frames = sig[idx]  # (n_frames, WIN_LENGTH)

    nfft = 1
    while nfft < 2 * WIN_LENGTH:
        nfft *= 2

    S = torch.fft.rfft(frames, n=nfft, dim=1)
    ac = torch.fft.irfft(S * torch.conj(S), n=nfft, dim=1)[:, :WIN_LENGTH]  # (n_frames, WIN)

    min_lag = int(SAMPLE_RATE / 500)
    max_lag = int(SAMPLE_RATE / 50)
    max_lag = min(max_lag, WIN_LENGTH - 1)

    seg = ac[:, min_lag:max_lag]
    peak = torch.argmax(seg, dim=1) + min_lag              # (n_frames,)
    peak_val = ac.gather(1, peak.unsqueeze(1)).squeeze(1)  # (n_frames,)
    energy0 = ac[:, 0]

    voiced = peak_val > 0.3 * energy0
    f0 = torch.where(voiced, SAMPLE_RATE / peak.clamp(min=1).float(),
                     torch.zeros_like(peak_val))
    return _continuous_pitch(f0.cpu())


def _median1d(x, k):
    """1D median filter (numpy), edge-padded."""
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    stacked = np.stack([xp[i:i + len(x)] for i in range(k)], axis=0)
    return np.median(stacked, axis=0)


def _continuous_pitch(f0):
    """Turn a raw f0 contour (0 = unvoiced) into a smooth CONTINUOUS contour.

    Octave-error spikes are removed with a median filter on the voiced frames,
    then unvoiced gaps are linearly interpolated (edges held). This is the
    standard FastSpeech2 continuous log-F0 representation: it avoids the huge
    normalization outliers that unvoiced zeros would otherwise create, which is
    what kept the pitch loss stuck."""
    f = f0.numpy().astype(np.float64)
    voiced = f > 0
    n_voiced = int(voiced.sum())
    if n_voiced == 0:
        return torch.zeros(len(f), dtype=torch.float32)

    idx = np.arange(len(f))
    vv = _median1d(f[voiced], 5) if n_voiced >= 5 else f[voiced]
    f_cont = np.interp(idx, idx[voiced], vv)   # fill unvoiced gaps, hold edges
    f_cont = _median1d(f_cont, 3)              # light final smoothing
    return torch.tensor(f_cont, dtype=torch.float32)


def extract_energy(mel):
    return mel.norm(dim=1)


# ---------------------------------------------------------------------------
# Forced alignment -> positional per-phoneme durations (fix #1)
# ---------------------------------------------------------------------------
PHONEME_TO_CHARS = {
    "AA": "a", "AE": "ae", "AH": "ah", "AO": "ao", "AW": "aw", "AY": "ay",
    "B": "b", "CH": "ch", "D": "d", "DH": "dh", "EH": "eh", "ER": "er",
    "EY": "ey", "F": "f", "G": "g", "HH": "h", "IH": "ih", "IY": "iy",
    "JH": "jh", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "ow", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh",
    "T": "t", "TH": "th", "UH": "uh", "UW": "uw", "V": "v", "W": "w",
    "Y": "y", "Z": "z", "ZH": "zh",
}


def align_phonemes_to_audio(waveform_1ch, phoneme_list, sample_rate):
    from torchaudio.functional import forced_align, merge_tokens

    model, dictionary, device = get_aligner()

    if sample_rate != 16000:
        wav16 = get_resampler(sample_rate, 16000)(waveform_1ch)
    else:
        wav16 = waveform_1ch

    emission_frame_sec = 320 / 16000.0
    mel_frame_sec = HOP_LENGTH / SAMPLE_RATE
    emit_to_mel = emission_frame_sec / mel_frame_sec  # ~1.875 (cancels in rescale)

    token_ids = []
    char_phoneme_index = []
    for ph_idx, ph in enumerate(phoneme_list):
        if ph == "<sil>":
            continue
        chars = PHONEME_TO_CHARS.get(ph, ph.lower())
        for ch in chars:
            tok = dictionary.get(ch, 0)
            if tok != 0:
                token_ids.append(tok)
                char_phoneme_index.append(ph_idx)

    if not token_ids:
        return None

    tokens_tensor = torch.tensor([token_ids], dtype=torch.int32, device=device)

    with torch.inference_mode():
        emission, _ = model(wav16.to(device))

    if tokens_tensor.size(1) > emission.size(1):
        return None

    aligned, scores = forced_align(emission, tokens_tensor, blank=0)
    token_spans = merge_tokens(aligned[0].cpu(), scores[0].cpu())

    if len(token_spans) != len(token_ids):
        return None

    n_ph = len(phoneme_list)
    ph_emit = [0.0] * n_ph
    for span, ph_idx in zip(token_spans, char_phoneme_index):
        ph_emit[ph_idx] += float(span.end - span.start)

    ph_durations = []
    for ph_idx in range(n_ph):
        mel_frames = ph_emit[ph_idx] * emit_to_mel
        ph_durations.append(mel_frames if mel_frames > 0 else float(MIN_DUR))

    return list(phoneme_list), torch.tensor(ph_durations, dtype=torch.float32)


def integerize_durations(durations_float, n_frames):
    """Integer per-phoneme durations, each >= MIN_DUR, summing EXACTLY to n_frames."""
    n = durations_float.numel()
    d = durations_float.clamp(min=1e-3)
    d = d * (float(n_frames) / float(d.sum().item()))

    floor = torch.floor(d).long().clamp(min=MIN_DUR)
    frac = d - torch.floor(d)

    diff = int(n_frames - int(floor.sum().item()))
    if diff > 0:
        order = torch.argsort(frac, descending=True).tolist()
        for k in range(diff):
            floor[order[k % n]] += 1
    elif diff < 0:
        order = torch.argsort(frac, descending=False).tolist()
        need = -diff
        k = 0
        guard = 0
        max_guard = 100 * n + 10 * need + 100
        while need > 0 and guard < max_guard:
            i = order[k % n]
            if floor[i] > MIN_DUR:
                floor[i] -= 1
                need -= 1
            k += 1
            guard += 1

    return floor


# ---------------------------------------------------------------------------
# Per-utterance processing
# ---------------------------------------------------------------------------
def process_one(sample, token2id):
    try:
        import soundfile as sf_io

        audio_bytes = sample["audio"]["bytes"]
        audio_array, sr = sf_io.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]

        device = get_device()
        wav = torch.tensor(audio_array, dtype=torch.float32, device=device).unsqueeze(0)
        if sr != SAMPLE_RATE:
            wav = get_resampler(sr, SAMPLE_RATE)(wav)

        duration_sec = wav.size(1) / SAMPLE_RATE
        if not (MIN_DURATION <= duration_sec <= MAX_DURATION):
            return None

        text = sample.get("text", sample.get("text_normalized", "")) or ""
        text = text.strip()
        if not text:
            return None

        phoneme_list = text_to_arpabet(text)
        if not phoneme_list:
            return None

        mel = extract_mel(wav)              # (n_frames, n_mels) on device
        n_frames = mel.size(0)
        if n_frames < len(phoneme_list):
            return None

        aligned = align_phonemes_to_audio(wav, phoneme_list, SAMPLE_RATE)
        if aligned is None:
            return None
        ph_labels, durations_float = aligned
        if durations_float.numel() != len(ph_labels):
            return None

        durations = integerize_durations(durations_float, n_frames)
        if int(durations.sum().item()) != n_frames:
            return None

        pitch = extract_pitch_fast(wav, n_frames)   # (n_frames,) on cpu
        mel_cpu = mel.cpu()
        energy = extract_energy(mel_cpu)            # (n_frames,)

        return {
            "phonemes": torch.tensor(phonemes_to_ids(ph_labels, token2id), dtype=torch.long),
            "speaker_id": torch.tensor(0, dtype=torch.long),
            "durations": durations.float(),
            "pitch": pitch,
            "energy": energy,
            "mel": mel_cpu,
        }
    except Exception:
        return None


class RunningStats:
    def __init__(self):
        self.s = 0.0
        self.ss = 0.0
        self.n = 0

    def update(self, values):
        v = values.double()
        self.s += float(v.sum().item())
        self.ss += float((v * v).sum().item())
        self.n += int(v.numel())

    def mean_std(self):
        if self.n == 0:
            return 0.0, 1.0
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        std = var ** 0.5
        return mean, (std if std > 1e-5 else 1.0)


# ---------------------------------------------------------------------------
# Speaker selection / split
# ---------------------------------------------------------------------------
def list_speakers(dataset):
    counts = {}
    hours = {}
    n = len(dataset)
    for i in tqdm(range(n), desc="scanning speakers"):
        row = dataset[i]
        spk = str(row["speaker"])
        counts[spk] = counts.get(spk, 0) + 1
        d = row.get("duration", None)
        if d is not None:
            hours[spk] = hours.get(spk, 0.0) + float(d)
    print("speakers in this split:")
    for spk in sorted(counts, key=lambda s: hours.get(s, 0.0), reverse=True):
        h = hours.get(spk, 0.0) / 3600.0
        print(f"  speaker {spk:>6}: {counts[spk]:>6} utts | {h:6.2f} h")


def split_indices(n):
    idx = list(range(n))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(idx)
    n_val = min(VAL_MAX, max(VAL_MIN, int(round(VAL_FRACTION * n))))
    n_val = min(n_val, max(1, n - 1))
    return set(idx[:n_val])


def process_speaker(dataset, out_dir, token2id, target_speaker, max_utterances):
    target = str(target_speaker)
    print(f"filtering to speaker {target}...")
    spk_dataset = dataset.filter(lambda ex: str(ex["speaker"]) == target)
    total = len(spk_dataset)
    if total == 0:
        raise RuntimeError(
            f"Speaker {target} not found in {HF_SPLIT}. Run with --list_speakers to see options."
        )

    if max_utterances is not None and total > max_utterances:
        rng = random.Random(SPLIT_SEED)
        order = list(range(total))
        rng.shuffle(order)
        order = sorted(order[:max_utterances])
        spk_dataset = spk_dataset.select(order)
        total = len(spk_dataset)
        print(f"capped to {total} utterances (--max_utterances)")

    try:
        sample_n = min(total, 50)
        approx_h = (sum(float(spk_dataset[i]["duration"]) for i in range(sample_n))
                    / sample_n * total / 3600.0)
        print(f"speaker {target}: {total} utterances (~{approx_h:.1f} h estimated)")
    except Exception:
        print(f"speaker {target}: {total} utterances")

    val_idx = split_indices(total)

    train_out = os.path.join(out_dir, "train")
    val_out = os.path.join(out_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    print("loading aligner + mel/pitch on:", get_device())
    get_aligner()
    get_mel_transform()

    pitch_stats = RunningStats()
    energy_stats = RunningStats()

    train_manifest = []
    val_manifest = []
    skipped = 0

    for j in tqdm(range(total), desc="processing"):
        sample = spk_dataset[j]
        result = process_one(sample, token2id)
        if result is None:
            skipped += 1
            continue

        is_val = j in val_idx
        split_dir = val_out if is_val else train_out
        manifest = val_manifest if is_val else train_manifest

        fpath = os.path.join(split_dir, f"{j:07d}.pt")
        torch.save(result, fpath)
        manifest.append(fpath)

        if not is_val:
            # pitch is now a continuous contour (no unvoiced zeros), so stats
            # are computed over all frames.
            pitch = result["pitch"]
            pitch_stats.update(torch.log1p(pitch.clamp(min=0.0)))
            energy_stats.update(result["energy"])

    with open(os.path.join(train_out, "manifest.txt"), "w") as f:
        f.write("\n".join(train_manifest))
    with open(os.path.join(val_out, "manifest.txt"), "w") as f:
        f.write("\n".join(val_manifest))

    pitch_mean, pitch_std = pitch_stats.mean_std()
    energy_mean, energy_std = energy_stats.mean_std()

    print(f"train: {len(train_manifest)} | val: {len(val_manifest)} | skipped: {skipped}")
    print(f"pitch  (log1p) mean/std: {pitch_mean:.4f} / {pitch_std:.4f}")
    print(f"energy         mean/std: {energy_mean:.4f} / {energy_std:.4f}")

    return {
        "pitch_mean": pitch_mean, "pitch_std": pitch_std,
        "energy_mean": energy_mean, "energy_std": energy_std,
        "n_train": len(train_manifest), "n_val": len(val_manifest),
    }


def save_metadata(out_dir, token2id, target_speaker, stats):
    meta = {
        "token2id": token2id,
        "speaker_map": {str(target_speaker): 0},
        "target_speaker": str(target_speaker),
        "dataset": HF_REPO,
        "n_speakers": 1,
        "phoneme_vocab_size": len(token2id),
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "n_fft": N_FFT,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "min_dur": MIN_DUR,
        "pitch_mean": stats["pitch_mean"],
        "pitch_std": stats["pitch_std"],
        "energy_mean": stats["energy_mean"],
        "energy_std": stats["energy_std"],
        "n_train": stats["n_train"],
        "n_val": stats["n_val"],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"metadata saved to {out_dir}/metadata.json")


def main(out_dir, target_speaker, max_utterances, do_list):
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading {HF_REPO} ({HF_CONFIG} / {HF_SPLIT}) from HuggingFace...")
    ds = load_dataset(HF_REPO, HF_CONFIG, split=HF_SPLIT)
    ds = ds.cast_column("audio", Audio(decode=False))

    if do_list:
        list_speakers(ds)
        return

    token2id = build_vocab()
    stats = process_speaker(ds, out_dir, token2id, target_speaker, max_utterances)
    save_metadata(out_dir, token2id, target_speaker, stats)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--target_speaker", type=str, default=DEFAULT_SPEAKER,
        help="Hi-Fi TTS speaker id (default 9017, John Van Stan, male, ~53h).",
    )
    parser.add_argument(
        "--max_utterances", type=int, default=None,
        help="Optional cap on number of utterances (for a quick run).",
    )
    parser.add_argument(
        "--list_speakers", action="store_true",
        help="List speakers/hours in the split and exit.",
    )
    args = parser.parse_args()
    main(args.out_dir, args.target_speaker, args.max_utterances, args.list_speakers)