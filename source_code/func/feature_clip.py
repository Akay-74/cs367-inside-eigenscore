"""Test time feature clipping (INSIDE paper, Section 3.2).

The published repository ships EigenScore (Section 3.1) but not the feature
clipping half of the method, so this module implements it from the paper.

The paper truncates extreme activations in the penultimate layer:

    FC(h) = h_min           h < h_min
            h                h_min <= h <= h_max
            h_max            h > h_max                              (Eq. 8)

h_min / h_max are per-neuron thresholds, set to the bottom and top p-th
percentile of the activations held in a memory bank of N token embeddings
collected at test time. The paper uses p = 0.2 (percent) and N = 3000.

The clip is applied by forward-hooking the penultimate decoder layer, so it
takes effect during autoregressive generation and therefore actually changes
what the model emits -- which is the point: it suppresses the overconfident,
self-consistent generations that consistency-based detectors otherwise miss.
"""
import numpy as np
import torch


class FeatureClipper:
    """Percentile clipper over a rolling memory bank of token activations.

    Usage:
        clipper = FeatureClipper(num_features=4096, bank_size=3000, percentile=0.2)
        handle  = clipper.attach(model)     # hooks the penultimate layer
        ...generate...
        handle.remove()

    The bank warms up before any clipping happens: while fewer than
    `min_bank` tokens have been seen the hook is a pass-through and only
    records. This avoids clipping against thresholds estimated from a
    handful of tokens.
    """

    def __init__(self, num_features, bank_size=3000, percentile=0.2,
                 min_bank=256, device='cuda', dtype=torch.float32):
        self.num_features = num_features
        self.bank_size = bank_size
        self.percentile = percentile          # p, in percent (paper: 0.2)
        self.min_bank = min(min_bank, bank_size)
        self.device = device
        self.dtype = dtype

        # Ring buffer of token activations: (bank_size, num_features)
        self._bank = torch.zeros(bank_size, num_features, device=device, dtype=dtype)
        self._pos = 0          # next write index
        self._filled = 0       # how many valid rows the bank holds

        self._h_min = None     # (num_features,) per-neuron thresholds
        self._h_max = None
        self._stale = True     # thresholds need recomputing
        self.enabled = True

        self.n_clipped = 0     # diagnostics
        self.n_seen = 0

    # ---------------- memory bank ----------------

    @torch.no_grad()
    def update(self, feats):
        """Push token activations into the bank. feats: (..., num_features)."""
        feats = feats.reshape(-1, feats.shape[-1]).to(self.device, self.dtype)
        n = feats.shape[0]
        if n == 0:
            return
        if n >= self.bank_size:
            self._bank.copy_(feats[-self.bank_size:])
            self._pos = 0
            self._filled = self.bank_size
        else:
            end = self._pos + n
            if end <= self.bank_size:
                self._bank[self._pos:end] = feats
            else:                                   # wrap around
                split = self.bank_size - self._pos
                self._bank[self._pos:] = feats[:split]
                self._bank[:end - self.bank_size] = feats[split:]
            self._pos = end % self.bank_size
            self._filled = min(self.bank_size, self._filled + n)
        self._stale = True

    @torch.no_grad()
    def _recompute_thresholds(self):
        """Per-neuron bottom/top p-th percentile over the bank."""
        bank = self._bank[:self._filled]
        q_lo = self.percentile / 100.0
        q_hi = 1.0 - q_lo
        # torch.quantile caps input size, so chunk over the feature axis.
        lo, hi = [], []
        for start in range(0, self.num_features, 512):
            chunk = bank[:, start:start + 512]
            qs = torch.quantile(chunk.float(),
                                torch.tensor([q_lo, q_hi], device=chunk.device),
                                dim=0)
            lo.append(qs[0])
            hi.append(qs[1])
        self._h_min = torch.cat(lo).to(self.dtype)
        self._h_max = torch.cat(hi).to(self.dtype)
        self._stale = False

    @property
    def ready(self):
        return self._filled >= self.min_bank

    # ---------------- the clip itself ----------------

    @torch.no_grad()
    def clip(self, h):
        """Apply Eq. 8 elementwise. h: (..., num_features)."""
        self.update(h)
        self.n_seen += h.reshape(-1, h.shape[-1]).shape[0]
        if not (self.enabled and self.ready):
            return h
        if self._stale:
            self._recompute_thresholds()
        h_min = self._h_min.to(h.device, h.dtype)
        h_max = self._h_max.to(h.device, h.dtype)
        clipped = torch.clamp(h, min=h_min, max=h_max)
        self.n_clipped += int((clipped != h).sum().item())
        return clipped

    # ---------------- wiring into a HF model ----------------

    def attach(self, model):
        """Hook the penultimate decoder layer. Returns a removable handle."""
        layers = _get_decoder_layers(model)
        penultimate = layers[-2]

        def hook(module, args, output):
            # Decoder layers return a tuple (hidden_states, ...) or a bare tensor.
            if isinstance(output, tuple):
                return (self.clip(output[0]),) + tuple(output[1:])
            return self.clip(output)

        return penultimate.register_forward_hook(hook)

    def stats(self):
        return dict(bank_filled=self._filled,
                    tokens_seen=self.n_seen,
                    activations_clipped=self.n_clipped,
                    h_min_mean=None if self._h_min is None else float(self._h_min.mean()),
                    h_max_mean=None if self._h_max is None else float(self._h_max.mean()))


def _get_decoder_layers(model):
    """Locate the list of decoder blocks across LLaMA / OPT / Falcon layouts."""
    candidates = [
        lambda m: m.model.layers,                # LLaMA, Mistral, Llama-3
        lambda m: m.model.decoder.layers,        # OPT
        lambda m: m.transformer.h,               # Falcon, GPT-2
        lambda m: m.model.transformer.h,
    ]
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) >= 2:
            return layers
    raise ValueError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        "add its layout to _get_decoder_layers().")


def get_hidden_size(model):
    cfg = model.config
    for attr in ('hidden_size', 'd_model', 'n_embd'):
        if getattr(cfg, attr, None):
            return getattr(cfg, attr)
    raise ValueError('Could not determine hidden size from model config.')


class MiddleLayerCapture:
    """Capture only the middle layer's per-step activations during generation.

    `output_hidden_states=True` returns every layer for every token, including the
    whole prompt -- for a 1000-token CoQA story at 33 layers x 5 sequences that is
    several GB, when EigenScore only ever reads one layer's last token.

    This hooks that single layer and keeps one (num_seq, d) row per decode step,
    reproducing exactly what getSentenceEmbeddings() would have indexed out of the
    full tuple:

      * step 0 is the prompt pass; upstream indexes position 0 there, so we do too
        (it only matters for degenerate 1-token generations)
      * later steps have length 1, so the last position is the only position

    Layer indexing matches upstream: hidden_states[i] is the embedding output for
    i = 0 and the output of decoder block i-1 for i >= 1, and upstream selects
    int(len(hidden_states[0])/2) = int((L+1)/2), i.e. block index int((L+1)/2)-1.
    """

    def __init__(self, model):
        layers = _get_decoder_layers(model)
        L = len(layers)
        self.selected_layer = int((L + 1) / 2)          # index into the hidden_states tuple
        self.block_index = self.selected_layer - 1      # index into the module list
        self._layer = layers[self.block_index]
        self.steps = []
        self._handle = None

    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        # first call is the prompt pass -> position 0, matching upstream indexing
        pos = 0 if not self.steps else -1
        self.steps.append(h[:, pos, :].detach().float().cpu())
        return output

    def __enter__(self):
        self.steps = []
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def embeddings(self, num_tokens):
        """(num_seq, d) last-token sentence embeddings, as getSentenceEmbeddings()."""
        import torch as _torch
        if len(self.steps) < 2:
            return None
        num_seq = self.steps[1].shape[0]
        out = _torch.zeros(num_seq, self.steps[1].shape[1])
        for ind in range(num_seq):
            step = min(max(num_tokens[ind] - 2, 0), len(self.steps) - 1)
            row = self.steps[step]
            # the prompt pass runs before beam/sample expansion on some models
            out[ind, :] = row[ind % row.shape[0], :]
        return out
