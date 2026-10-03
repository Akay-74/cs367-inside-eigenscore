"""Model / tokenizer loading.

Two changes over the original:

  * A model name containing '/' is treated as a HuggingFace hub id and loaded
    straight from the hub cache. Bare names ('llama-7b-hf', 'opt-6.7b', ...)
    still resolve to a local directory under _settings.MODEL_PATH, so the
    original run.sh keeps working if you have the weights on disk.
  * Optional 4-bit / 8-bit quantization (bitsandbytes), which is what makes a
    7B model fit alongside its hidden states on an 8GB GPU.
"""
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import MODEL_PATH

# Convenience aliases so short names still work with hub-hosted weights.
_HUB_ALIASES = {
    'llama-7b-hf':   'huggyllama/llama-7b',
    'llama-13b-hf':  'huggyllama/llama-13b',
    'llama2-7b-hf':  'NousResearch/Llama-2-7b-hf',   # ungated mirror of meta-llama/Llama-2-7b-hf
    'llama2-13b-hf': 'meta-llama/Llama-2-13b-hf',
    'opt-1.3b':      'facebook/opt-1.3b',
    'opt-2.7b':      'facebook/opt-2.7b',
    'opt-6.7b':      'facebook/opt-6.7b',
    'opt-13b':       'facebook/opt-13b',
    'falcon-7b':     'tiiuae/falcon-7b',
}

_SEQ_CLS_MODELS = {'microsoft/deberta-large-mnli', 'roberta-large-mnli'}


def resolve_model_path(model_name):
    """Map a model name to something from_pretrained() accepts.

    Priority: an existing local directory under MODEL_PATH, then a hub id
    (either given directly with a '/', or via _HUB_ALIASES).
    """
    local = os.path.join(MODEL_PATH, model_name)
    if os.path.isdir(local):
        return local
    if '/' in model_name:
        return model_name
    if model_name in _HUB_ALIASES:
        return _HUB_ALIASES[model_name]
    return model_name


def _quantization_config(load_in_4bit, load_in_8bit, compute_dtype):
    if not (load_in_4bit or load_in_8bit):
        return None
    from transformers import BitsAndBytesConfig
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16,
                           load_in_4bit=False, load_in_8bit=False):
    path = resolve_model_path(model_name)
    quant_config = _quantization_config(load_in_4bit, load_in_8bit, torch_dtype)

    kwargs = dict(torch_dtype=torch_dtype)
    if quant_config is not None:
        # bitsandbytes places the weights itself; passing a device_map is required
        # and .to(device) afterwards is not allowed.
        kwargs['quantization_config'] = quant_config
        kwargs['device_map'] = {'': device}

    if model_name in _SEQ_CLS_MODELS:
        return AutoModelForSequenceClassification.from_pretrained(path).to(device)

    if model_name.startswith('facebook/opt') or model_name.startswith('opt-'):
        model = OPTForCausalLM.from_pretrained(path, **kwargs)
    elif 'falcon' in model_name:
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

    if quant_config is None:
        model.to(device)
    model.eval()
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    path = resolve_model_path(model_name)

    if model_name in _SEQ_CLS_MODELS:
        return AutoTokenizer.from_pretrained(path)

    trust = 'falcon' in model_name
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast,
                                              trust_remote_code=trust)

    # The dataset prompt builders and _generate_config() assume a LLaMA
    # tokenizer exposes bos/eos as ids 1/2 and has a pad token.
    if 'llama' in model_name.lower():
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        # Upstream forces pad == eos: generate() pads with eos, and
        # get_num_tokens() counts ids > 2, so the two must agree.
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
