"""Setup smoke tests -- no large model download required.

    python -m tests.test_setup
"""
import sys

import numpy as np
import torch


def test_imports():
    import _settings, models, utils                      # noqa: F401
    import dataeval.coqa, dataeval.SQuAD                 # noqa: F401
    import dataeval.nq_open, dataeval.triviaqa           # noqa: F401
    from func.metric import (getEigenIndicator_v0, getSentenceEmbeddings,   # noqa: F401
                             getEigenScoreFromEmbeddings, get_perplexity_score,
                             get_energy_score, getLexicalSim)
    from func.feature_clip import FeatureClipper         # noqa: F401
    print("[ok] imports")


def test_thresholds_match_numpy():
    from func.feature_clip import FeatureClipper
    D, N, P = 64, 2000, 5.0
    fc = FeatureClipper(D, bank_size=N, percentile=P, min_bank=100, device='cpu')
    data = torch.randn(N, D)
    fc.update(data)
    fc._recompute_thresholds()
    ref = data.numpy()
    assert np.allclose(np.percentile(ref, P, axis=0), fc._h_min.numpy(), atol=1e-4)
    assert np.allclose(np.percentile(ref, 100-P, axis=0), fc._h_max.numpy(), atol=1e-4)
    print("[ok] feature-clip thresholds match numpy.percentile")


def test_clip_is_eq8():
    from func.feature_clip import FeatureClipper
    D = 32
    fc = FeatureClipper(D, bank_size=500, percentile=5.0, min_bank=50, device='cpu')
    fc.update(torch.randn(500, D))
    probe = torch.randn(40, D) * 4
    out = fc.clip(probe)
    inside = (probe >= fc._h_min) & (probe <= fc._h_max)
    assert torch.allclose(out[inside], probe[inside]), "in-band values must pass through"
    assert (out <= fc._h_max + 1e-4).all() and (out >= fc._h_min - 1e-4).all()
    print("[ok] clip implements Eq. 8")


def test_ring_buffer_and_warmup():
    from func.feature_clip import FeatureClipper
    fc = FeatureClipper(4, bank_size=10, percentile=5.0, min_bank=1, device='cpu')
    for i in range(7):
        fc.update(torch.full((3, 4), float(i)))
    assert fc._filled == 10, fc._filled

    fc2 = FeatureClipper(4, bank_size=100, percentile=5.0, min_bank=50, device='cpu')
    x = torch.randn(10, 4) * 100
    assert torch.allclose(fc2.clip(x), x), "must pass through during warm-up"
    print("[ok] ring buffer wraps; warm-up is a pass-through")


def test_eigenscore_pooling_equivalence():
    """The pooled path must reproduce getEigenIndicator_v0 exactly for one batch."""
    from func.metric import (getEigenIndicator_v0, getSentenceEmbeddings,
                             getEigenScoreFromEmbeddings)
    K, L, D, T = 6, 9, 48, 12
    hs = tuple(tuple(torch.randn(K, 1, D) for _ in range(L)) for _ in range(T))
    nt = [T] * K
    a, _ = getEigenIndicator_v0(hs, nt)
    b, _ = getEigenScoreFromEmbeddings(getSentenceEmbeddings(hs, nt))
    assert abs(a - b) < 1e-9, (a, b)
    print("[ok] pooled EigenScore == getEigenIndicator_v0")


def test_eigenscore_separates_consistent_from_diverse():
    """Tight answers -> low score; diverse answers -> high score (paper Section 3.1)."""
    from func.metric import getEigenScoreFromEmbeddings
    torch.manual_seed(0)
    base = torch.randn(1, 128)
    consistent = base.repeat(10, 1) + 0.01 * torch.randn(10, 128)
    diverse = torch.randn(10, 128)
    s_con, _ = getEigenScoreFromEmbeddings(consistent)
    s_div, _ = getEigenScoreFromEmbeddings(diverse)
    assert s_con < s_div, (s_con, s_div)
    print(f"[ok] EigenScore consistent={s_con:.3f} < diverse={s_div:.3f}")


def test_low_memory_capture_equivalence():
    """--low_memory must reproduce the full-hidden-states EigenScore exactly."""
    import torch as t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from func.metric import getSentenceEmbeddings, getEigenScoreFromEmbeddings
    from func.feature_clip import MiddleLayerCapture
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2"); tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    ids = tok("The capital of France is a city that", return_tensors="pt")
    nt = [10, 7, 10, 4]
    t.manual_seed(7)
    o1 = m.generate(**ids, max_new_tokens=10, num_return_sequences=4, do_sample=True,
                    output_hidden_states=True, return_dict_in_generate=True,
                    pad_token_id=tok.eos_token_id)
    full, _ = getEigenScoreFromEmbeddings(getSentenceEmbeddings(o1.hidden_states, nt))
    cap = MiddleLayerCapture(m)
    assert cap.selected_layer == int(len(o1.hidden_states[0]) / 2)
    t.manual_seed(7)
    with cap:
        m.generate(**ids, max_new_tokens=10, num_return_sequences=4, do_sample=True,
                   output_hidden_states=False, return_dict_in_generate=True,
                   pad_token_id=tok.eos_token_id)
    low, _ = getEigenScoreFromEmbeddings(cap.embeddings(nt))
    assert abs(full - low) < 1e-6, (full, low)
    print("[ok] --low_memory EigenScore == full-hidden-states EigenScore")


def test_layer_discovery():
    from transformers import AutoModelForCausalLM
    from func.feature_clip import _get_decoder_layers, get_hidden_size
    m = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
    assert len(_get_decoder_layers(m)) >= 2
    assert get_hidden_size(m) > 0
    print("[ok] decoder-layer discovery")


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
