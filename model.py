import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Minimum frames per phoneme. Kept identical in prepare.py (target integerization)
# and here (inference rounding + length regulator) so training and inference agree.
MIN_DUR = 1


class EvoTalkConfig:

    def __init__(self,
                 phoneme_vocab_size=42,
                 n_mels=100,
                 max_seq_len=1024,
                 max_mel_len=2048,
                 n_encoder_layers=6,
                 n_decoder_layers=6,
                 n_head=10,
                 n_embd=640,
                 n_query_groups=2,
                 bias=True,
                 dropout=0.1,
                 eps=1e-6,
                 use_rotary=True,
                 use_swiglu=True,
                 use_qk_norm=False,
                 use_gqa=True,
                 n_speakers=1,
                 speaker_emb_dim=256):
        self.phoneme_vocab_size = phoneme_vocab_size
        self.n_mels = n_mels
        self.max_seq_len = max_seq_len
        self.max_mel_len = max_mel_len
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_query_groups = n_query_groups if use_gqa else n_head
        self.bias = bias
        self.dropout = dropout
        self.eps = eps
        self.use_rotary = use_rotary
        self.use_swiglu = use_swiglu
        self.use_qk_norm = use_qk_norm
        self.use_gqa = use_gqa
        self.n_speakers = n_speakers
        self.speaker_emb_dim = speaker_emb_dim
        assert n_head % self.n_query_groups == 0, "n_head must be divisible by n_query_groups"


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    seq_len = xq_.size(2)
    freqs_cis_seq = freqs_cis[:seq_len]
    xq_out = torch.view_as_real(xq_ * freqs_cis_seq.unsqueeze(0)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_seq.unsqueeze(0)).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GroupedQueryAttention(nn.Module):

    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_query_groups = config.n_query_groups
        self.causal = causal

        self.kv_heads = config.n_head // config.n_query_groups if config.use_gqa else config.n_head
        qkv_proj_size = (config.n_head + 2 * self.kv_heads) * self.head_dim

        self.c_attn = nn.Linear(config.n_embd, qkv_proj_size, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.qk_norm = getattr(config, "use_qk_norm", False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.eps)

    def forward(self, x, freqs_cis=None, key_padding_mask=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q_size = self.n_head * self.head_dim
        k_size = self.kv_heads * self.head_dim
        v_size = self.kv_heads * self.head_dim

        q, k, v = qkv.split([q_size, k_size, v_size], dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_heads < self.n_head:
            repeats = self.n_head // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # key_padding_mask: (B, T) with True marking padded key positions.
        # Broadcast as (B, 1, 1, T) so it masks those keys for every query head/position,
        # which avoids materialising a full (B, 1, T, T) mask.
        if self.flash:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = torch.zeros(B, 1, 1, T, dtype=q.dtype, device=x.device)
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask[:, None, None, :], float("-inf")
                )
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if key_padding_mask is not None:
                att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):

    def __init__(self, config, causal=False):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = GroupedQueryAttention(config, causal=causal)

        if config.use_swiglu:
            self.mlp = nn.ModuleDict(dict(
                gate=nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                up=nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                down=nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                act=nn.SiLU(),
                dropout=nn.Dropout(config.dropout),
            ))
            m = self.mlp
            self.mlpf = lambda x: m.dropout(m.down(m.act(m.gate(x)) * m.up(x)))
        else:
            self.mlp = nn.ModuleDict(dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                act=nn.GELU(),
                dropout=nn.Dropout(config.dropout),
            ))
            m = self.mlp
            self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, freqs_cis=None, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, key_padding_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class DurationPredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1)
        self.norm1 = RMSNorm(config.n_embd, eps=config.eps)
        self.conv2 = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1)
        self.norm2 = RMSNorm(config.n_embd, eps=config.eps)
        self.linear = nn.Linear(config.n_embd, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2)))
        x = self.dropout(F.relu(self.norm2(self.conv2(x).transpose(1, 2)).transpose(1, 2)))
        x = x.transpose(1, 2)
        return self.linear(x).squeeze(-1)


class VariancePredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1)
        self.norm1 = RMSNorm(config.n_embd, eps=config.eps)
        self.conv2 = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1)
        self.norm2 = RMSNorm(config.n_embd, eps=config.eps)
        self.linear = nn.Linear(config.n_embd, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2)))
        x = self.dropout(F.relu(self.norm2(self.conv2(x).transpose(1, 2)).transpose(1, 2)))
        x = x.transpose(1, 2)
        return self.linear(x).squeeze(-1)


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, durations, max_len=None):
        outputs = []
        for i in range(x.size(0)):
            output = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            outputs.append(output)

        if max_len is None:
            max_len = max(o.size(0) for o in outputs)

        padded = torch.zeros(x.size(0), max_len, x.size(2), device=x.device, dtype=x.dtype)
        for i, o in enumerate(outputs):
            length = min(o.size(0), max_len)
            padded[i, :length] = o[:length]

        return padded

    def inference(self, x, durations, min_duration=MIN_DUR):
        durations = torch.clamp(durations.long(), min=min_duration)
        outputs = []
        for i in range(x.size(0)):
            output = torch.repeat_interleave(x[i], durations[i], dim=0)
            outputs.append(output)
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(x.size(0), max_len, x.size(2), device=x.device, dtype=x.dtype)
        for i, o in enumerate(outputs):
            padded[i, :o.size(0)] = o
        return padded


class VarianceAdaptor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.duration_predictor = DurationPredictor(config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)
        self.pitch_embedding = nn.Linear(1, config.n_embd)
        self.energy_embedding = nn.Linear(1, config.n_embd)

    def forward(self, x, durations=None, pitch_targets=None, energy_targets=None,
                max_mel_len=None, duration_scale=1.0, embed_predicted=False):
        # x: (B, T_phon, D) encoder output, one vector per phoneme.
        #
        # PHONEME-LEVEL variance (canonical FastSpeech2). Duration, pitch, and
        # energy are all predicted from the unexpanded encoder output — one value
        # per phoneme — and the pitch/energy embeddings are added BEFORE length
        # regulation. This is deliberate: a per-phoneme average of energy/pitch
        # carries far less information about the exact target mel than a per-frame
        # value (per-frame energy is essentially mel magnitude), so the decoder
        # can't lean on a leaked copy of its own target and then collapse when the
        # predictor's inference outputs differ.
        duration_preds = self.duration_predictor(x)   # (B, T_phon) log-domain
        pitch_preds = self.pitch_predictor(x)          # (B, T_phon)
        energy_preds = self.energy_predictor(x)        # (B, T_phon)

        # Choose the prosody actually fed to the decoder. Teacher forcing uses the
        # ground-truth (phoneme-level) targets; otherwise the predictor's own
        # (detached) outputs — detach keeps mel-loss gradients out of the
        # predictors so they stay honestly supervised by their own loss.
        tf_pitch = (pitch_targets is not None) and (not embed_predicted)
        tf_energy = (energy_targets is not None) and (not embed_predicted)
        pitch_used = pitch_targets if tf_pitch else pitch_preds.detach()
        energy_used = energy_targets if tf_energy else energy_preds.detach()

        # Targets are padded to the batch's max phoneme length; align to x.
        pitch_used = pitch_used[:, :x.size(1)]
        energy_used = energy_used[:, :x.size(1)]

        x = x + self.pitch_embedding(pitch_used.unsqueeze(-1)) \
              + self.energy_embedding(energy_used.unsqueeze(-1))

        # Expand the enriched phoneme representation to frame level.
        if durations is not None:
            dur_clamped = torch.clamp(durations.long(), min=MIN_DUR)
            x = self.length_regulator(x, dur_clamped, max_len=max_mel_len)
        else:
            dur_rounded = torch.clamp(
                torch.round((torch.exp(duration_preds) - 1) * duration_scale), min=MIN_DUR
            ).long()
            x = self.length_regulator.inference(x, dur_rounded)

        return x, duration_preds, pitch_preds, energy_preds


class EvoTalk(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.phoneme_emb = nn.Embedding(config.phoneme_vocab_size, config.n_embd)
        self.speaker_emb = nn.Embedding(config.n_speakers, config.speaker_emb_dim)
        self.speaker_proj = nn.Linear(config.speaker_emb_dim, config.n_embd)

        self.encoder_drop = nn.Dropout(config.dropout)
        self.decoder_drop = nn.Dropout(config.dropout)

        if config.use_rotary:
            head_dim = config.n_embd // config.n_head
            enc_len = max(config.max_seq_len, config.max_mel_len)
            self.freqs_cis = precompute_freqs_cis(head_dim, enc_len)
        else:
            self.freqs_cis = None
            self.encoder_pos = nn.Embedding(config.max_seq_len, config.n_embd)
            self.decoder_pos = nn.Embedding(config.max_mel_len, config.n_embd)

        self.encoder = nn.ModuleList([Block(config, causal=False) for _ in range(config.n_encoder_layers)])
        self.encoder_norm = RMSNorm(config.n_embd, eps=config.eps)

        self.variance_adaptor = VarianceAdaptor(config)

        self.decoder = nn.ModuleList([Block(config, causal=False) for _ in range(config.n_decoder_layers)])
        self.decoder_norm = RMSNorm(config.n_embd, eps=config.eps)

        self.mel_head = nn.Linear(config.n_embd, config.n_mels)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * (config.n_encoder_layers + config.n_decoder_layers)))

        print(f"EvoTalk parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _make_mel_padding_mask(self, durations, T_mel, device):
        # True marks padded frames (frames past the sum of the sample's durations).
        dur_clamped = torch.clamp(durations.long(), min=MIN_DUR)
        valid_len = dur_clamped.sum(dim=1).clamp(max=T_mel)  # (B,)
        positions = torch.arange(T_mel, device=device).unsqueeze(0)  # (1, T)
        return positions >= valid_len.unsqueeze(1)  # (B, T) bool

    def encode(self, phonemes, speaker_ids, src_mask=None):
        device = phonemes.device
        B, T = phonemes.size()

        x = self.phoneme_emb(phonemes)
        spk = self.speaker_proj(self.speaker_emb(speaker_ids))
        x = x + spk.unsqueeze(1)

        if self.config.use_rotary:
            freqs_cis = self.freqs_cis.to(device)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            x = x + self.encoder_pos(pos)
            freqs_cis = None

        x = self.encoder_drop(x)

        for block in self.encoder:
            x = block(x, freqs_cis, key_padding_mask=src_mask)

        x = self.encoder_norm(x)
        return x

    def forward(self, phonemes, speaker_ids, durations=None, pitch_targets=None, energy_targets=None,
                mel_targets=None, src_mask=None, max_mel_len=None, embed_predicted=False):
        device = phonemes.device

        x = self.encode(phonemes, speaker_ids, src_mask)

        x, duration_preds, pitch_preds, energy_preds = self.variance_adaptor(
            x,
            durations=durations,
            pitch_targets=pitch_targets,
            energy_targets=energy_targets,
            max_mel_len=max_mel_len,
            embed_predicted=embed_predicted,
        )

        B, T_mel, _ = x.size()

        # Build a decoder padding mask so padded (silence) frames do not leak into
        # real frames through the non-causal decoder self-attention.
        mel_mask = None
        if durations is not None:
            mel_mask = self._make_mel_padding_mask(durations, T_mel, device)

        if self.config.use_rotary:
            freqs_cis = self.freqs_cis.to(device)
        else:
            pos = torch.arange(0, T_mel, dtype=torch.long, device=device).unsqueeze(0)
            x = x + self.decoder_pos(pos)
            freqs_cis = None

        x = self.decoder_drop(x)

        for block in self.decoder:
            x = block(x, freqs_cis, key_padding_mask=mel_mask)

        x = self.decoder_norm(x)
        mel_out = self.mel_head(x)

        # The training loss (masked) is computed in train.py. We return mel_mask so
        # callers can reuse the exact valid-frame region if needed.
        return mel_out, duration_preds, pitch_preds, energy_preds, mel_mask

    @torch.no_grad()
    def inference(self, phonemes, speaker_ids, src_mask=None, duration_scale=1.0):
        x = self.encode(phonemes, speaker_ids, src_mask)
        x, _, _, _ = self.variance_adaptor(x, duration_scale=duration_scale)

        device = phonemes.device
        B, T_mel, _ = x.size()

        if self.config.use_rotary:
            freqs_cis = self.freqs_cis.to(device)
        else:
            pos = torch.arange(0, T_mel, dtype=torch.long, device=device).unsqueeze(0)
            x = x + self.decoder_pos(pos)
            freqs_cis = None

        # Inference is single-sequence, so there is no padding to mask.
        for block in self.decoder:
            x = block(x, freqs_cis, key_padding_mask=None)

        x = self.decoder_norm(x)
        mel_out = self.mel_head(x)
        return mel_out

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, RMSNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay and no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters not assigned: {param_dict.keys() - union_params}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=tuple(betas))
        return optimizer