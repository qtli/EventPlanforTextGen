# this script used to generate knowledge path for knowledge extracted from downstream dataset using knowledge generator

from __future__ import absolute_import, division, print_function
import json
import argparse
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
import logging
import os
import pickle
import random
import re
import pdb
import shutil
import subprocess
from typing import List, Dict
import csv
import logging
import sys
import collections
import math
import spacy
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from tqdm import tqdm, trange
from dictionary import Dictionary
from seq_generator import SequenceGenerator
from data import DatasetHelper

from collections import Counter

from optimization import AdamW, WarmupLinearSchedule, WarmupCosineSchedule, WarmupConstantSchedule
from tokenization_gpt2 import GPT2Tokenizer
from modeling_gpt2 import KnowledgeGen, GPT2Config

logger = logging.getLogger()

MODEL_CLASSES = {
    'gpt2': (GPT2Config, KnowledgeGen, GPT2Tokenizer)
}


def list2str(list):
    return " ".join([str(x) for x in list])


def str2list(str):
    return [int(x) for x in str.split(" ")]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_log(log_file=None):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                            '%m/%d/%Y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file != None:
        logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def evaluate(args, model, tokenizer, evaluate_metrics="ppl", prefix='0'):
    eval_output_dir = args.output_dir
    if prefix == 'test':
        eval_data_file = os.path.join(args.data_file, 'test_simple.json')
        eval_knowledge_data_file = os.path.join(args.knowledge_data_file, 'test.txt')
    elif prefix == 'train':
        eval_data_file = os.path.join(args.data_file, 'train_simple.json')
        eval_knowledge_data_file = os.path.join(args.knowledge_data_file, 'train.txt')
    else:
        prefix = 'dev'
        eval_data_file = os.path.join(args.data_file, 'dev_simple.json')
        eval_knowledge_data_file = os.path.join(args.knowledge_data_file, 'dev.txt')

    eval_dataset = DatasetHelper(args,
                                 tokenizer,
                                 data_path=eval_data_file,
                                 knowledge_path=eval_knowledge_data_file,
                                 src_max_length=args.source_length,
                                 tgt_max_length=args.target_length,
                                 do_generate=evaluate_metrics == 'bleu')

    eval_dataset.load()
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=False,
                                 num_workers=args.workers, pin_memory=True, shuffle=False)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    gen_seqs = []
    gen_multi_seqs = []
    src = []  # kg
    tgt = []
    nb_eval_steps = 0
    model.eval()

    if evaluate_metrics == 'bleu':
        generator = build_generator(args, eval_dataset)
    else:
        generator = None
    top_ids = []
    Hit_num = 0

    for bidx, batch in enumerate(eval_dataloader):
        src.extend(batch[3])
        tgt.extend(batch[4])
        batch = tuple(t.to(args.device) for t in batch[0])

        with torch.no_grad():
            if evaluate_metrics == 'bleu':
                batch = {"src_input_ids": batch[6],
                         "attention_mask": batch[7],
                         "src_position_ids": batch[8],
                         "seq_generator": generator,
                         }
                if isinstance(model, torch.nn.DataParallel):
                    hypos, multi_hypos = model.module.aotoreg_generate(**batch)
                else:
                    hypos, multi_hypos = model.aotoreg_generate(**batch)
                gen_seqs.extend(hypos)
                gen_multi_seqs.extend(multi_hypos)
            if evaluate_metrics == 'ppl':
                eval_batch = {"src_input_ids": batch[0],
                              "attention_mask": batch[1],
                              "src_position_ids": batch[2],
                              "target_input_ids": batch[3],
                              "target_position_ids": batch[4],
                              "labels": batch[5], }
                eval_loss += torch.mean(model(**eval_batch)).item()
        nb_eval_steps += 1

    if evaluate_metrics == 'ppl':
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {
            "ppl": perplexity,
        }

    elif evaluate_metrics == 'bleu':
        references = [[x for x in y.strip().split(' ')] for y in eval_dataset.target_kg]
        predictions = [x.strip().split() for x in gen_seqs]
        assert len(gen_seqs) == len(eval_dataset.target), 'prediction is not finished !'
        bleu1 = _compute_bleu(references, predictions, max_order=1)
        bleu2 = _compute_bleu(references, predictions, max_order=2)
        bleu4 = _compute_bleu(references, predictions, max_order=4)
        result = {
            "blue1": bleu1[0],
            "blue2": bleu2[0],
            "bleu4": bleu4[0],
        }
        save_generation(args, eval_data_file, [gen_multi_seqs, src, tgt], prefix=prefix)

    elif evaluate_metrics == 'acc':
        save_generation(args, top_ids, prefix=prefix)
        result = {
            "acc": Hit_num / len(eval_dataset),
        }
    else:
        result = {}

    return result


def generate(args, generator, model, dataset):
    model.eval()
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.per_gpu_eval_batch_size)
    total_hypos = []

    for i, batch in enumerate(tqdm(loader)):
        batch = tuple(t.to(args.device) for t in batch)
        sample = {
            "input_ids": batch[0],
            "attention_mask": batch[2],
            "position_ids": batch[3]
        }

        with torch.no_grad():
            hypos = generator.generate(model, sample, dataset)

        total_hypos.extend(hypos)
    return total_hypos


def build_generator(args, dataset):
    generator = SequenceGenerator(
        args,
        Dictionary(dataset.tokenizer.encoder),
        dataset.tokenizer,
        beam_size=getattr(args, 'beam', 3),
        max_len_a=getattr(args, 'max_len_a', 0),
        max_len_b=getattr(args, 'max_len_b', dataset.tgt_max_length),
        min_len=getattr(args, 'min_len', 3),
        normalize_scores=(not getattr(args, 'unnormalized', False)),
        len_penalty=getattr(args, 'lenpen', 1),
        unk_penalty=getattr(args, 'unkpen', 0),
        sampling=getattr(args, 'sampling', False),
        sampling_topk=getattr(args, 'sampling_topk', -1),
        sampling_topp=getattr(args, 'sampling_topp', -1.0),
        temperature=getattr(args, 'temperature', 1.),
        diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
        diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len=getattr(args, 'match_source_len', False),
        no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
    )
    return generator


def save_generation(args, eval_data_file, results, prefix='0'):
    kg_results, kg_src, kg_tgt = results
    save_result_dir = os.path.join(args.output_dir, "{}_cand.json".format(prefix))
    with open(eval_data_file, 'r') as f:
        data = json.load(f)
        new_data = []
        for i, item in enumerate(data):
            item['kg_candidates'] = kg_results[i]
            new_data.append(item)
    json.dump(new_data, open(save_result_dir, 'w'), indent=4)
    logger.info("Save generation result in {}".format(save_result_dir))


class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def main():
    parser = argparse.ArgumentParser()

    ## File parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--knowledge_data_file", default=None, type=str, help="knowledge path of data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--knowledge_type", default="atomic", type=str, help="knowledge type (atomic, conceptnet)")
    parser.add_argument("--task_type", default="roc_last_sentence", type=str, help="downstream task type ...")

    ## My parameters
    parser.add_argument("--source_length", default=100, type=int, help="max source len")
    parser.add_argument("--target_length", default=100, type=int, help="max target_len")
    parser.add_argument("--tb_log_dir", default=None, type=str, help="log")
    parser.add_argument("--evaluate_metrics", default='ppl', type=str, help='choose between ppl and bleu')

    parser.add_argument("--model_type", default="gpt2", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--kg_model_name_or_path", default=None, type=str, help="knowledge path of pretrained kg generator")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_generate", action='store_true', help="generate data using trained model")
    parser.add_argument("--continue_train", action='store_true',
                        help="Whether to run training based on a trained checkpoint.")
    parser.add_argument("--workers", default=0, type=int, help="workers")
    parser.add_argument("--save_last", action='store_true', help="whether save the last epoch")

    parser.add_argument("--kg_source_length", default=100, type=int, help="max kg source len")
    parser.add_argument("--kg_target_length", default=100, type=int, help="max kg target_len")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio.")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--validate_steps', type=int, default=2000, help="evaluate model every x updates steps")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--beam', type=int, default=3, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    load_from_path = args.kg_model_name_or_path
    args.output_dir = os.path.join(args.output_dir, 'kg_path_gen')

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    set_log(os.path.join(args.output_dir, "log.txt"))
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=False)
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(load_from_path,
                                        source_length=args.source_length,
                                        tokenizer=tokenizer, )

    model.resize_token_embeddings(len(tokenizer))  # include <|bos|>, <|pad|>, <|endoftext|>
    # if args.do_eval:
    #     state_dict = torch.load(load_from_path + '/training_args.bin', map_location='cpu')
    #     model.load_state_dict(state_dict)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    result = evaluate(args, model, tokenizer, args.evaluate_metrics, 'dev')
    logger.info("Test evaluate {}".format(args.evaluate_metrics))
    for k in result:
        logger.info("{}: {:.4f}".format(k, result[k]))


if __name__ == '__main__':
    main()
    # references = [[x for x in y.strip().split(' ')] for y in eval_dataset.target_kg]
    # predictions = [x.strip().split() for x in gen_seqs]
    # from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
    #
    # references = [['_xNeed PersonX finish tournament in place'.split(' ')], ['_xAttr PersonX run inside wet'.split(' ')]]
    # predictions = ['_xIntent PersonX moves to colorado xAttr PersonX is adventurous'.split(' '), 'xReact PersonX feels wet _oReact PersonY throws PersonX overboard'.split()]
    # references = ['_xAttr PersonX run inside wet'.split(' ')]
    # predictions = 'xReact PersonX feels wet _oReact PersonY throws PersonX overboard'.split()

    # references = [['_xNeed PersonX finish tournament in place'.split(' ')]]
    # predictions = ['_xIntent PersonX moves to colorado xAttr PersonX is adventurous'.split(' ')]
    #
    # print(references)
    # print(predictions)
    # bleu2 = _compute_bleu(references, predictions, max_order=1)
    # bleu4 = _compute_bleu(references, predictions, max_order=4)
    # print(bleu2)
    # print(bleu4)
