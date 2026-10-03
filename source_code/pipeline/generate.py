import argparse
import contextlib
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers
from sentence_transformers import SentenceTransformer

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import models
import utils
from func.metric import *
from func.feature_clip import FeatureClipper, get_hidden_size, MiddleLayerCapture

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
# --- quantization: lets a 7B model + its hidden states fit on a small GPU ---
parser.add_argument('--load_in_4bit', action='store_true', help='NF4 4-bit weights (bitsandbytes)')
parser.add_argument('--load_in_8bit', action='store_true', help='int8 weights (bitsandbytes)')
# --- test time feature clipping, paper Section 3.2 ---
parser.add_argument('--feature_clip', action='store_true',
                    help='clip extreme activations in the penultimate layer (Eq. 8)')
parser.add_argument('--clip_percentile', type=float, default=0.2,
                    help='p, in percent: bottom/top p%% of the memory bank are truncated')
parser.add_argument('--memory_bank_size', type=int, default=3000,
                    help='N: number of token embeddings kept in the memory bank')
parser.add_argument('--max_new_tokens', type=int, default=256)
parser.add_argument('--low_memory', action='store_true',
                    help='capture only the middle layer via a hook instead of returning '
                         'every layer for every token. Mathematically identical for '
                         'EigenScore, but avoids multi-GB hidden-state tensors on long '
                         'prompts. Disables the LN-Entropy/perplexity-free path? No -- '
                         'scores are unaffected; only hidden-state retention changes.')
parser.add_argument('--no_sentsim', action='store_true',
                    help='skip the nli-roberta sentence model entirely (disables the '
                         'EigenScore-Output baseline; EigenScore itself is unaffected)')
parser.add_argument('--sentsim_device', type=str, default='cpu',
                    help="device for the nli-roberta sentence model; 'cpu' keeps ~1.4GB "
                         "of VRAM free for the LLM (it only encodes a few short strings)")
parser.add_argument('--num_return_sequences', type=int, default=None,
                    help='generations per forward pass; lower it if you run out of VRAM '
                         '(default: all of --num_generations_per_prompt at once)')


args = parser.parse_args()
os.makedirs(_settings.GENERATION_FOLDER, exist_ok=True)
logInfo = open(os.path.join(_settings.GENERATION_FOLDER,
                            "logInfo_{}_{}.txt".format(args.model.replace('/', '_'), args.dataset)),
               mode="w", encoding="utf-8")


# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name == "SQuAD":
        return SQuAD.get_dataset


def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = args.max_new_tokens
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    if data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=None):
    if max_num_gen_once is None:
        max_num_gen_once = args.num_return_sequences or args.num_generations_per_prompt
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(
        model_name, args.device,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit)
    # Only needed for the EigenScore-Output baseline (semantic divergence measured on
    # decoded text rather than internal states). If it is unavailable, generation still
    # produces EigenScore and every Table-1 baseline, so degrade instead of dying.
    SenSimModel = None
    if not args.no_sentsim:
        try:
            SenSimModel = SentenceTransformer(_settings.SENTENCE_SIM_MODEL,
                                              device=args.sentsim_device)
        except Exception as e:
            print(f"[warn] sentence model unavailable ({type(e).__name__}); "
                  f"EigenScore-Output will be recorded as 0. Everything else is unaffected.")

    # Test time feature clipping (paper Section 3.2). Attached as a forward hook on
    # the penultimate decoder layer so it shapes generation itself, not just the
    # embeddings EigenScore is read from.
    clipper, clip_handle = None, None
    if args.feature_clip:
        clipper = FeatureClipper(num_features=get_hidden_size(model),
                                 bank_size=args.memory_bank_size,
                                 percentile=args.clip_percentile,
                                 device=device)
        clip_handle = clipper.attach(model)
        print(f"[FC] feature clipping on: p={args.clip_percentile}%, N={args.memory_bank_size}")

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                        num_beams=1,
                                        do_sample=False,
                                        generation_config=generation_config,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=True)

            scores = dict_outputs.scores    #([logits],[logits],[logits])
            perplexity = get_perplexity_score(scores)
            energy_score = get_energy_score(scores)
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]

        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        # When K generations need more than one forward pass, the per-batch
        # sentence embeddings are pooled so the covariance matrix is still built
        # over all K responses (Eq. 4), not just the last batch. Entropy is
        # likewise averaged over batches weighted by how many sequences each held.
        embeddings_all, entropy_sum, entropy_n = [], 0.0, 0
        while num_gens > 0:
            # The KV cache scales with prompt length x sequences, and CoQA stories vary
            # from ~200 to ~1100 tokens, so a batch size that is fine for most questions
            # can OOM on a long one. Back off and retry rather than losing the whole run.
            n_this = min(max_num_gen_once, num_gens)
            while True:
                capture = MiddleLayerCapture(model) if args.low_memory else None
                ctx = capture if capture is not None else contextlib.nullcontext()
                try:
                    with ctx:
                        dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                        num_beams=1, num_return_sequences=n_this,
                                        do_sample=True, top_p=args.top_p, top_k=args.top_k,
                                        temperature=args.temperature, generation_config=generation_config,
                                        output_hidden_states=not args.low_memory,
                                        return_dict_in_generate=True, output_scores=True
                                        )
                    break
                except torch.cuda.OutOfMemoryError:
                    del capture
                    torch.cuda.empty_cache()
                    if n_this == 1:
                        raise
                    n_this = max(1, n_this // 2)
                    print(f"[oom] backing off to num_return_sequences={n_this} "
                          f"(prompt is {input_length} tokens)")

            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            scores = dict_outputs.scores
            entropy_sum += get_lenghthNormalized_entropy(scores, num_tokens) * len(generation)
            entropy_n += len(generation)
            if capture is not None:
                emb = capture.embeddings(num_tokens)
                capture.steps = []
            else:
                emb = getSentenceEmbeddings(dict_outputs.hidden_states, num_tokens)
            if emb is not None:
                embeddings_all.append(emb.float().cpu())
            del dict_outputs, capture
            torch.cuda.empty_cache()
            num_gens -= len(generation)

        predictive_entropy = entropy_sum / max(entropy_n, 1)
        if embeddings_all:
            eigenIndicator, eigenValue = getEigenScoreFromEmbeddings(torch.cat(embeddings_all, dim=0))
        else:
            eigenIndicator, eigenValue = 0, "None"

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        lexical_similarity = getLexicalSim(generated_texts)
        sent_bertscore = getAvgBertScore(None, best_generated_text, generated_texts)
        if SenSimModel is not None:
            eigenIndicatorOutput, eigenValue_O = getEigenIndicatorOutput(generated_texts, SenSimModel)
        else:
            eigenIndicatorOutput, eigenValue_O = 0.0, "None"


        # remember the data
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        curr_seq.update(
            dict(
                energy=energy_score
            )
        )
        curr_seq.update(
            dict(
                lexical_similarity=lexical_similarity
            )
        )
        curr_seq.update(
            dict(
                sent_bertscore=sent_bertscore
            )
        )
        curr_seq.update(
            dict(
                entropy=predictive_entropy
            )
        )
        curr_seq.update(
            dict(
                eigenIndicator=eigenIndicator
            )
        )
        curr_seq.update(
            dict(
                eigenIndicatorOutput=eigenIndicatorOutput
            )
        )
        if args.dataset == 'coqa' or args.dataset == "TruthfulQA":
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
        ########## 信息打印 #########
        # print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        print("Question:", batch['question'][0])
        print("AnswerGT:", batch['answer'][0])
        print("MostLikelyAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True))
        print("Batch_Generations:", generated_texts)
        print("Perplexity:", perplexity)
        print("Energy:", energy_score)
        print("NormalizedEntropy: ", predictive_entropy)
        print("LexicalSimilarity: ", lexical_similarity)
        print("EigenScore: ", eigenIndicator)
        print("EigenValue:", eigenValue)
        print("EigenScore-Output: ", eigenIndicatorOutput)

        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Question:", batch['question'][0], file=logInfo)
        print("GTAns:", batch['answer'][0], file=logInfo)
        print("BestAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=logInfo)
        print("BatchGenerations:", generated_texts, file=logInfo)
        print("Perplexity:", perplexity, file=logInfo)
        print("Energy:", energy_score, file=logInfo)
        print("NormalizedEntropy: ", predictive_entropy, file=logInfo)
        print("LexicalSimilarity: ", lexical_similarity, file=logInfo)
        print("SentBERTScore: ", sent_bertscore, file=logInfo)
        print("EigenScore: ", eigenIndicator, file=logInfo)
        print("EigenValue:", eigenValue, file=logInfo)
        print("EigenScore-Output: ", eigenIndicatorOutput, file=logInfo)
        print("\n","\n","\n", file=logInfo)

    if clip_handle is not None:
        clip_handle.remove()
        print("[FC]", clipper.stats())
        print("[FC]", clipper.stats(), file=logInfo)
    return sequences


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens


def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
