import os
import re
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import torchaudio

torch.set_warn_always(False)

from model import EvoTalk, EvoTalkConfig


VALID_ARPABET = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
}

_G2P_BACKEND = None


def get_g2p_backend():
    global _G2P_BACKEND
    if _G2P_BACKEND is None:
        from g2p_en import G2p
        _G2P_BACKEND = G2p()
    return _G2P_BACKEND


def text_to_arpabet(text):
    try:
        g2p = get_g2p_backend()
        raw_tokens = g2p(text.strip())
        arpabet = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            phon = re.sub(r"[0-9]", "", tok).upper()
            if phon in VALID_ARPABET:
                arpabet.append(phon)
        return arpabet if arpabet else None
    except Exception:
        return None


def phonemes_to_ids(phonemes, token2id):
    unk = token2id["<unk>"]
    return [token2id.get(p, unk) for p in phonemes]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(SCRIPT_DIR, "ckpt.pt")
METADATA = os.path.join(SCRIPT_DIR, "metadata.json")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")

MAX_WORDS = 12

PAUSE = {"sentence": 0.40, "clause": 0.20, "wrap": 0.10}

SAMPLE_RATE = 24000
N_FFT = 1024
N_MELS = 100
F_MIN = 0.0
F_MAX = 12000.0

VOCOS_REPO = "charactr/vocos-mel-24khz"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_clauses(sentence):
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


class WaveRNNVocoder:
    name = "wavernn"

    def __init__(self, device):
        self.device = device
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        self.model = bundle.get_vocoder().to(device).eval()
        self.native_sr = getattr(bundle, "sample_rate", 22050)

        n_stft = N_FFT // 2 + 1
        ours = torchaudio.functional.melscale_fbanks(
            n_freqs=n_stft, f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS,
            sample_rate=SAMPLE_RATE, norm=None, mel_scale="htk",
        ).to(device)
        self.ours_pinv = torch.linalg.pinv(ours.T).to(device)

        theirs = torchaudio.functional.melscale_fbanks(
            n_freqs=n_stft, f_min=0.0, f_max=8000.0, n_mels=80,
            sample_rate=self.native_sr, norm="slaney", mel_scale="slaney",
        ).to(device)
        self.theirs = theirs.T.contiguous()

    @torch.no_grad()
    def __call__(self, mel):
        m = mel.squeeze(0).transpose(0, 1).contiguous().to(self.device)
        linear = torch.matmul(self.ours_pinv, torch.exp(m)).clamp_min(0.0)
        feats = torch.log(torch.matmul(self.theirs, linear).clamp_min(1e-5))
        feats = feats.unsqueeze(0)
        lengths = torch.tensor([feats.size(-1)], dtype=torch.long, device=self.device)
        wav, wav_lengths = self.model(feats, lengths)
        wav = wav.squeeze(0)
        if wav.dim() > 1:
            wav = wav[0]
        return wav[:int(wav_lengths[0])].float().cpu()


class VocosVocoder:
    name = "vocos"

    def __init__(self, device):
        from vocos import Vocos
        self.device = device
        self.model = Vocos.from_pretrained(VOCOS_REPO).to(device).eval()

    @torch.no_grad()
    def __call__(self, mel):
        feats = mel.float().transpose(1, 2).contiguous()
        return self.model.decode(feats).squeeze(0).float().cpu()


VOCODERS = {"wavernn": WaveRNNVocoder, "vocos": VocosVocoder}


def diagnose_phonemizer(text):
    print(f"warning: text_to_arpabet produced no phonemes for: {text!r}")
    try:
        g2p = get_g2p_backend()
        raw = g2p(text.strip())
        print(f"  g2p_en loaded fine, raw output: {raw!r}")
    except Exception as e:
        print(f"  underlying g2p_en error: {type(e).__name__}: {e}")
        print("  make sure g2p_en is installed: pip install g2p_en")


def synth_chunk(model, vocoder, text, token2id, speaker_ids, duration_scale, device):
    phonemes = text_to_arpabet(text)
    if not phonemes:
        diagnose_phonemizer(text)
        return None

    try:
        ids_list = phonemes_to_ids(phonemes, token2id)
    except Exception as e:
        print(f"warning: phonemes_to_ids failed for {phonemes!r}: {e}")
        return None

    if not ids_list:
        print(f"warning: phonemes_to_ids returned empty ids for {phonemes!r}")
        return None

    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    with torch.no_grad():
        mel = model.inference(ids, speaker_ids, src_mask=None,
                              duration_scale=duration_scale)
    return vocoder(mel)


def synthesize(model, vocoder, meta, text, speaker_id, duration_scale, device,
               max_words=MAX_WORDS):
    sr = meta["sample_rate"]
    n_speakers = meta.get("n_speakers", 1)
    speaker_id = max(0, min(int(speaker_id), n_speakers - 1))
    spk = torch.tensor([speaker_id], dtype=torch.long, device=device)

    chunks = chunk_text(text, max_words)
    if not chunks:
        return None

    pieces = []
    for piece, kind in chunks:
        wav = synth_chunk(model, vocoder, piece, meta["token2id"], spk,
                          duration_scale, device)
        if wav is None:
            continue
        pieces.append(wav)
        pieces.append(torch.zeros(int(PAUSE[kind] * sr)))

    if not pieces:
        return None
    return torch.cat(pieces[:-1])


def save_waveform(waveform, sr, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    torchaudio.save(out_path, wav, sr)
    return out_path


def play_audio(path):
    try:
        if sys.platform == "darwin":
            os.system(f"afplay '{path}'")
        elif sys.platform.startswith("linux"):
            os.system(f"aplay '{path}' 2>/dev/null || paplay '{path}' 2>/dev/null")
        elif sys.platform.startswith("win"):
            os.system(f'start "" "{path}"')
    except Exception:
        pass


BANNER = r"""
  _____          _______     _ _
 | ____|_   _____|_   _|_ _ | | | __
 |  _| \ \ / / _ \| |/ _` || | |/ /
 | |___ \ V / (_) | | (_| || |   <
 |_____| \_/ \___/|_|\__,_||_|_|\_\

"""


def run(speaker_id, duration_scale, max_words, vocoder_name):
    print(BANNER)
    print(f"loading model from {CHECKPOINT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_metadata(METADATA)
    model = load_model(CHECKPOINT, device)

    cache = {}

    def get_vocoder(name):
        if name not in cache:
            print(f"loading vocoder: {name}")
            cache[name] = VOCODERS[name](device)
        return cache[name]

    vocoder = get_vocoder(vocoder_name)

    print(f"device: {device}")
    print(f"vocoder: {vocoder.name}  speaker: {speaker_id}  "
          f"duration_scale: {duration_scale}  max_words: {max_words}")
    print("type text and press enter to hear it")
    print("commands: :speaker N   :scale X   :words N   :voc NAME   :ab   :quit")
    print("-" * 50)

    n_speakers = meta.get("n_speakers", 1)
    idx = 0

    while True:
        try:
            text = input("\n> ").strip()
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
        if text.startswith(":voc"):
            parts = text.split()
            if len(parts) == 2 and parts[1] in VOCODERS:
                vocoder = get_vocoder(parts[1])
                print(f"vocoder set to {vocoder.name}")
            else:
                print(f"usage: :voc {'|'.join(VOCODERS)}")
            continue
        if text == ":ab":
            print("usage: :ab then type text on the next line")
            try:
                ab_text = input("ab> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not ab_text:
                continue
            for name in VOCODERS:
                v = get_vocoder(name)
                wav = synthesize(model, v, meta, ab_text, speaker_id,
                                 duration_scale, device, max_words)
                if wav is None:
                    print(f"{name}: could not synthesize")
                    continue
                out_path = os.path.join(AUDIO_DIR, f"ab_{idx:04d}_{name}.wav")
                save_waveform(wav, meta["sample_rate"], out_path)
                print(f"{name}: {wav.numel()/meta['sample_rate']:.2f}s  ->  {out_path}")
                play_audio(out_path)
            idx += 1
            continue

        wav = synthesize(model, vocoder, meta, text, speaker_id, duration_scale,
                         device, max_words)
        if wav is None:
            print("could not synthesize that text")
            continue

        out_path = os.path.join(AUDIO_DIR, f"line_{idx:04d}_{vocoder.name}.wav")
        save_waveform(wav, meta["sample_rate"], out_path)
        duration = wav.numel() / meta["sample_rate"]
        print(f"{duration:.2f}s  ->  {out_path}")
        play_audio(out_path)
        idx += 1

    print("goodbye")


if __name__ == "__main__":
    run(speaker_id=0, duration_scale=1, max_words=MAX_WORDS,
        vocoder_name="wavernn")
