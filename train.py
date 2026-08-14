import os
import time
import math
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import EvoTalk, EvoTalkConfig


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
GRAD_CLIP = 1.0
WARMUP_STEPS = 4000

# The ~80M model uses more memory than the old ~20M one. If you hit CUDA OOM,
# drop BATCH_SIZE (e.g. 8) and/or raise GRAD_ACCUM-equivalent by training longer.
BATCH_SIZE = 32
NUM_WORKERS = 4

MAX_STEPS = 150000
SAVE_EVERY = 5000
LOG_EVERY = 50
EVAL_EVERY = 500
EVAL_STEPS = 50

# When True, the decoder is fed the predictor's own (detached) pitch/energy
# during training instead of ground-truth. This removes the teacher-forcing leak
# that otherwise makes inference collapse. Set via --predicted_variance (used for
# fine-tuning from a checkpoint trained with teacher forcing).
EMBED_PREDICTED = False

GRAPH_DIR = "graphs"
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM = None


def smooth(values, window=50):
    if len(values) < 2:
        return values
    w = min(window, len(values))
    kernel = torch.ones(w) / w
    t = torch.tensor(values, dtype=torch.float32)
    pad = w // 2
    t_padded = torch.cat([t[:pad].flip(0), t, t[-pad:].flip(0)])
    smoothed = torch.nn.functional.conv1d(
        t_padded.view(1, 1, -1), kernel.view(1, 1, -1), padding=0
    ).squeeze()
    return smoothed[:len(values)].tolist()


def save_graphs(history, graph_dir):
    os.makedirs(graph_dir, exist_ok=True)
    steps = history["steps"]
    if len(steps) < 2:
        return

    style = dict(linewidth=1.2, alpha=0.9)
    raw_style = dict(linewidth=0.5, alpha=0.25)

    fig, ax = plt.subplots(figsize=(10, 4))
    raw = history["loss"]
    s = smooth(raw)
    ax.plot(steps, raw, color="#4C8BE8", **raw_style)
    ax.plot(steps, s, color="#4C8BE8", label="train", **style)
    if history["val_steps"]:
        ax.plot(history["val_steps"], history["val_loss"], "o--",
                color="#E8744C", label="val", linewidth=1.2, markersize=3)
    ax.set_title("Total Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "loss_total.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    components = [
        ("mel", "#4C8BE8", "Mel Loss"),
        ("dur", "#50C87A", "Duration Loss"),
        ("energy", "#C850A8", "Energy Loss"),
    ]
    for ax, (key, color, title) in zip(axes, components):
        raw = history[key]
        s = smooth(raw)
        ax.plot(steps, raw, color=color, **raw_style)
        ax.plot(steps, s, color=color, **style)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "loss_components.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    raw = history["pitch"]
    s = smooth(raw)
    ax.plot(steps, raw, color="#E8C84C", **raw_style)
    ax.plot(steps, s, color="#E8C84C", **style)
    ax.set_title("Pitch Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "loss_pitch.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["lr"], color="#888888", **style)
    ax.set_title("Learning Rate")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "learning_rate.png"), dpi=120)
    plt.close(fig)


class LibriTTSDataset(Dataset):
    """Single-speaker dataset. Pitch and energy targets are normalized here with
    the GLOBAL training statistics from metadata.json, so the exact same normalized
    representation is used for (a) the variance-predictor loss target and (b) the
    pitch/energy embedding at both training and inference time."""

    def __init__(self, data_dir, split, pitch_mean, pitch_std, energy_mean, energy_std):
        self.data_dir = data_dir
        self.split = split
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.samples = self._load_samples()

    def _load_samples(self):
        manifest = os.path.join(self.data_dir, self.split, "manifest.txt")
        samples = []
        with open(manifest, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
        return samples

    def __len__(self):
        return len(self.samples)

    def _norm_pitch(self, pitch):
        return (torch.log1p(pitch.clamp(min=0.0)) - self.pitch_mean) / self.pitch_std

    def _norm_energy(self, energy):
        return (energy - self.energy_mean) / self.energy_std

    @staticmethod
    def _to_phoneme_level(frame_values, durations):
        """Average a per-frame contour into one value per phoneme using the
        (integer) durations. sum(durations) == n_frames by construction, so the
        segmentation is exact. This is what makes pitch/energy phoneme-level."""
        d = durations.long()
        T = int(d.sum().item())
        fv = frame_values[:T]
        out = torch.zeros(d.size(0), dtype=torch.float32)
        idx = 0
        for i, di in enumerate(d.tolist()):
            if di > 0:
                seg = fv[idx:idx + di]
                if seg.numel() > 0:
                    out[i] = seg.mean()
            idx += di
        return out

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = torch.load(path, weights_only=True)
        durations = data["durations"]
        # Per-frame pitch/energy (stored by prepare.py) -> phoneme-level averages.
        phon_pitch = self._to_phoneme_level(data["pitch"], durations)
        phon_energy = self._to_phoneme_level(data["energy"], durations)
        return (
            data["phonemes"],
            data["speaker_id"],
            durations,
            self._norm_pitch(phon_pitch),
            self._norm_energy(phon_energy),
            data["mel"],
        )


def collate_fn(batch):
    phonemes, speaker_ids, durations, pitches, energies, mels = zip(*batch)

    phoneme_lengths = [p.size(0) for p in phonemes]
    mel_lengths = [m.size(0) for m in mels]

    max_phoneme_len = max(phoneme_lengths)
    max_mel_len = max(mel_lengths)
    n_mels = mels[0].size(1)

    phonemes_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
    durations_padded = torch.zeros(len(batch), max_phoneme_len)
    mels_padded = torch.zeros(len(batch), max_mel_len, n_mels)
    # pitch/energy are now PHONEME-LEVEL, so they pad to phoneme length.
    pitches_padded = torch.zeros(len(batch), max_phoneme_len)
    energies_padded = torch.zeros(len(batch), max_phoneme_len)
    src_mask = torch.zeros(len(batch), max_phoneme_len, dtype=torch.bool)

    for i, (p, d, pi, e, m) in enumerate(zip(phonemes, durations, pitches, energies, mels)):
        phonemes_padded[i, :p.size(0)] = p
        durations_padded[i, :d.size(0)] = d
        mels_padded[i, :m.size(0)] = m
        pitches_padded[i, :pi.size(0)] = pi
        energies_padded[i, :e.size(0)] = e
        src_mask[i, p.size(0):] = True

    speaker_ids = torch.stack(speaker_ids)
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)

    return {
        "phonemes": phonemes_padded,
        "speaker_ids": speaker_ids,
        "durations": durations_padded,
        "pitch": pitches_padded,
        "energy": energies_padded,
        "mels": mels_padded,
        "src_mask": src_mask,
        "mel_lengths": mel_lengths,
        "max_mel_len": max_mel_len,
    }


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def frame_mask_from_lengths(lengths, max_len, device):
    """(B, max_len) bool: True where the frame is valid (t < length)."""
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.to(device).unsqueeze(1)


def masked_mse(pred, target, valid_mask):
    """Mean squared error over valid entries only. valid_mask is (B, T) with True
    marking valid frames; it is broadcast over any trailing feature dimension."""
    if valid_mask.dim() == pred.dim() - 1:
        valid_mask = valid_mask.unsqueeze(-1)
    m = valid_mask.to(pred.dtype)
    diff = (pred - target) ** 2 * m
    denom = m.expand_as(pred).sum().clamp(min=1.0)
    return diff.sum() / denom


def masked_l1(pred, target, valid_mask):
    """Mean absolute error over valid entries only. L1 on the mel produces sharper,
    less over-smoothed spectrograms than MSE (standard in FastSpeech2-style models)."""
    if valid_mask.dim() == pred.dim() - 1:
        valid_mask = valid_mask.unsqueeze(-1)
    m = valid_mask.to(pred.dtype)
    diff = (pred - target).abs() * m
    denom = m.expand_as(pred).sum().clamp(min=1.0)
    return diff.sum() / denom


def compute_loss(mel_out, duration_preds, pitch_preds, energy_preds, batch):
    mels = batch["mels"]
    durations = batch["durations"]
    pitch = batch["pitch"]        # already normalized in the dataset
    energy = batch["energy"]      # already normalized in the dataset
    src_mask = batch["src_mask"]  # True = padded phoneme
    mel_lengths = batch["mel_lengths"]
    device = mel_out.device

    # --- Mel loss (L1, masked over valid frames) ---
    T_mel = min(mel_out.size(1), mels.size(1))
    valid_frames = frame_mask_from_lengths(mel_lengths, T_mel, device)
    mel_loss = masked_l1(mel_out[:, :T_mel], mels[:, :T_mel], valid_frames)

    # --- Duration loss (masked over valid phonemes) ---
    phon_valid = ~src_mask
    dur_targets = torch.clamp(durations.float(), min=1.0)
    log_dur_targets = torch.log(dur_targets + 1.0)
    dur_loss = masked_mse(duration_preds, log_dur_targets, phon_valid)

    # --- Pitch / energy loss (now PHONEME-LEVEL, masked over valid phonemes) ---
    T_var = min(pitch_preds.size(1), pitch.size(1))
    valid_var = phon_valid[:, :T_var]
    pitch_loss = masked_mse(pitch_preds[:, :T_var], pitch[:, :T_var], valid_var)
    energy_loss = masked_mse(energy_preds[:, :T_var], energy[:, :T_var], valid_var)

    total = mel_loss + dur_loss + pitch_loss + energy_loss
    return total, mel_loss, dur_loss, pitch_loss, energy_loss


def save_checkpoint(model, optimizer, scaler, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": loss,
        "config": model.config,
    }, path)
    print(f"saved checkpoint at step {step} -> {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"resumed from step {ckpt['step']} (loss {ckpt['loss']:.4f})")
    return ckpt["step"]


@torch.no_grad()
def evaluate(model, val_loader, device, max_steps):
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in val_loader:
        if steps >= max_steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mel_out, duration_preds, pitch_preds, energy_preds, _ = model(
            phonemes=batch["phonemes"],
            speaker_ids=batch["speaker_ids"],
            durations=batch["durations"],
            pitch_targets=batch["pitch"],
            energy_targets=batch["energy"],
            mel_targets=batch["mels"],
            src_mask=batch["src_mask"],
            max_mel_len=batch["max_mel_len"],
            embed_predicted=EMBED_PREDICTED,
        )
        loss, *_ = compute_loss(mel_out, duration_preds, pitch_preds, energy_preds, batch)
        total_loss += loss.item()
        steps += 1
    model.train()
    return total_loss / max(steps, 1)


def train(data_dir, init_from=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    config = EvoTalkConfig(
        phoneme_vocab_size=meta["phoneme_vocab_size"],
        n_speakers=meta["n_speakers"],
        n_mels=meta["n_mels"],
    )
    model = EvoTalk(config).to(device)

    if init_from is not None:
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started weights from {init_from}")

    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, device)
    scaler = GradScaler("cuda", enabled=use_amp)

    stat_kwargs = dict(
        pitch_mean=meta["pitch_mean"], pitch_std=meta["pitch_std"],
        energy_mean=meta["energy_mean"], energy_std=meta["energy_std"],
    )
    train_dataset = LibriTTSDataset(data_dir, "train", **stat_kwargs)
    val_dataset = LibriTTSDataset(data_dir, "val", **stat_kwargs)
    print(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    step = 0
    if RESUME_FROM is not None:
        step = load_checkpoint(RESUME_FROM, model, optimizer, scaler, device)

    history = {
        "steps": [], "loss": [], "mel": [], "dur": [],
        "pitch": [], "energy": [], "lr": [],
        "val_steps": [], "val_loss": [],
    }

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    while step < MAX_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast("cuda", enabled=use_amp):
            mel_out, duration_preds, pitch_preds, energy_preds, _ = model(
                phonemes=batch["phonemes"],
                speaker_ids=batch["speaker_ids"],
                durations=batch["durations"],
                pitch_targets=batch["pitch"],
                energy_targets=batch["energy"],
                mel_targets=batch["mels"],
                src_mask=batch["src_mask"],
                max_mel_len=batch["max_mel_len"],
                embed_predicted=EMBED_PREDICTED,
            )
            loss, mel_loss, dur_loss, pitch_loss, energy_loss = compute_loss(
                mel_out, duration_preds, pitch_preds, energy_preds, batch
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        history["steps"].append(step)
        history["loss"].append(loss.item())
        history["mel"].append(mel_loss.item())
        history["dur"].append(dur_loss.item())
        history["pitch"].append(pitch_loss.item())
        history["energy"].append(energy_loss.item())
        history["lr"].append(lr)

        if step % LOG_EVERY == 0:
            t1 = time.time()
            dt = (t1 - t0) / LOG_EVERY
            t0 = t1
            print(
                f"step {step:>6} | loss {loss.item():.4f} | mel {mel_loss.item():.4f} "
                f"| dur {dur_loss.item():.4f} | pitch {pitch_loss.item():.4f} "
                f"| energy {energy_loss.item():.4f} | lr {lr:.2e} | {dt*1000:.0f}ms/step"
            )
            save_graphs(history, GRAPH_DIR)

        if step % EVAL_EVERY == 0 and len(val_dataset) > 0:
            val_loss = evaluate(model, val_loader, device, EVAL_STEPS)
            print(f"step {step:>6} | val loss {val_loss:.4f}")
            history["val_steps"].append(step)
            history["val_loss"].append(val_loss)
            save_graphs(history, GRAPH_DIR)

        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"evotalk_{step:06d}.pt")
            save_checkpoint(model, optimizer, scaler, step, loss.item(), ckpt_path)

    save_checkpoint(model, optimizer, scaler, step, loss.item(),
                    os.path.join(CHECKPOINT_DIR, "evotalk_final.pt"))
    print("training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--init_from", type=str, default=None,
                        help="checkpoint to warm-start weights from (for fine-tuning)")
    parser.add_argument("--predicted_variance", action="store_true",
                        help="feed the decoder predicted (detached) pitch/energy instead of "
                             "ground-truth; removes the teacher-forcing leak")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="override max steps")
    parser.add_argument("--warmup", type=int, default=None, help="override warmup steps")
    args = parser.parse_args()

    if args.predicted_variance:
        EMBED_PREDICTED = True
    if args.lr is not None:
        LEARNING_RATE = args.lr
    if args.max_steps is not None:
        MAX_STEPS = args.max_steps
    if args.warmup is not None:
        WARMUP_STEPS = args.warmup

    train(args.data_dir, init_from=args.init_from)