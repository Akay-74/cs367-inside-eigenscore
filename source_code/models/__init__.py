from ._load_model import (_load_pretrained_model, _load_pretrained_tokenizer,
                          resolve_model_path)
# from .openai_models import openai_query  # requires the `openai` package


def load_model_and_tokenizer(model_name='opt-13b', device='cuda:0', **kwargs):
    if model_name in {'gpt-3.5-turbo'}:
        return None, None
    return (_load_pretrained_model(model_name, device, **kwargs),
            _load_pretrained_tokenizer(model_name))


def load_tokenizer(model_name='opt-13b', use_fast=False):
    if model_name in {'gpt-3.5-turbo'}:
        return None
    return _load_pretrained_tokenizer(model_name, use_fast=use_fast)
