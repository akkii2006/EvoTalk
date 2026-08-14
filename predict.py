"""Run EvoTalk inference on some text and save the predicted mel spectrogram as
an image.

Usage:
    python predict.py --text "hello world"
    python predict.py --text "hello world" --out predicted_mel.png
"""

import os
import re
import json
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(SCRIPT_DIR, "ckpt.pt")
METADATA = os.path.join(SCRIPT_DIR, "metadata.json")


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


def plot_mel(mel, out_path, title="Predicted Mel Spectrogram"):
    # mel: (1, T, n_mels) -> plot with time on x, mel bins on y.
    spec = mel.squeeze(0).T.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel Bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved predicted mel -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="text to synthesize")
    parser.add_argument("--out", type=str, default="predicted_mel.png")
    parser.add_argument("--duration_scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--metadata", type=str, default=METADATA)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_metadata(args.metadata)
    model = load_model(args.checkpoint, device)

    phonemes = text_to_arpabet(args.text)
    if not phonemes:
        print("could not phonemize the given text")
        return

    token2id = meta["token2id"]
    unk = token2id["<unk>"]
    ids_list = [token2id.get(p, unk) for p in phonemes]
    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    spk = torch.tensor([0], dtype=torch.long, device=device)

    with torch.no_grad():
        mel = model.inference(ids, spk, src_mask=None,
                              duration_scale=args.duration_scale)

    plot_mel(mel, args.out)


if __name__ == "__main__":
    main()
