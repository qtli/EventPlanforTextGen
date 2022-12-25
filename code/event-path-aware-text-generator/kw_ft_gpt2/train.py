import json
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import os
import random
import subprocess
import logging
import sys
import collections
import math
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from dictionary import Dictionary
from seq_generator import SequenceGenerator
from data import roc_DatasetHelper, ed_DatasetHelper, kg_DatasetHelper

from optimization import AdamW, WarmupLinearSchedule, WarmupCosineSchedule, WarmupConstantSchedule, BalancedDataParallel
from tokenization_gpt2 import GPT2Tokenizer
from modeling_gpt2 import TextGen, GPT2Config

logger = logging.getLogger()

MODEL_CLASSES = {
    'gpt2': (GPT2Config, TextGen, GPT2Tokenizer)
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


def _get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in enumerate(res):
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if getattr(train_dataset, "print_features", False):
        train_dataset.print_features()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  drop_last=True, num_workers=args.workers, pin_memory=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    gpt2_params = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(gn in n for gn in gpt2_params)]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=int(args.warmup_ratio * t_total), t_total=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = BalancedDataParallel(4, model, dim=0)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_valid = {'bleu': 0.0, 'ppl': 1e6, 'acc': 0.0}

    global_step = 0
    patient = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    if args.validate_steps == -1:
        args.validate_steps = len(train_dataloader)
    for epoch in train_iterator:
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)

            model.train()
            loss = model(*batch)
            loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info(
                        "Step: {} | Loss: {:.4f}".format(global_step, (tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

            if global_step % 50 == 0 and global_step < (args.validate_steps + 50):
                logger.info('global_step/validate_steps: {}/{}'.format(global_step, args.validate_steps))
            if global_step % args.validate_steps == 0:
                logger.info('step: {}'.format(step))
                sign_list = {'ppl': 1.0, 'bleu': -1.0, 'acc': -1.0}
                result = evaluate(args, model, tokenizer, args.evaluate_metrics, prefix=epoch)
                logger.info(
                    "Epoch {} evaluate {}: {:.4f}".format(epoch, args.evaluate_metrics, result[args.evaluate_metrics]))
                if (result[args.evaluate_metrics] - best_valid[args.evaluate_metrics]) * sign_list[
                    args.evaluate_metrics] < 0:
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                    torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.bin"))
                    subprocess.call(["cp", os.path.join(args.model_name_or_path, "vocab.json"), args.output_dir])
                    subprocess.call(["cp", os.path.join(args.model_name_or_path, "merges.txt"), args.output_dir])
                    logger.info("Saving model checkpoint to %s", args.output_dir)
                    best_valid[args.evaluate_metrics] = result[args.evaluate_metrics]
                    patient = 0
                    logger.info('Patient is {}.'.format(patient))
                else:
                    patient += 1
                    logger.info('Patient is {}.'.format(patient))
                    if patient > 2:
                        logger.info('Patient is {} and Stop training.'.format(patient))
                        break
            if patient > 2:
                break
        if patient > 2:
            break

    if args.save_last:
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'last_training_args.bin'))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "last_scheduler.bin"))
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "last_optimizer.bin"))
        subprocess.call(["cp", os.path.join(args.model_name_or_path, "vocab.json"), args.output_dir])
        subprocess.call(["cp", os.path.join(args.model_name_or_path, "merges.txt"), args.output_dir])
        logger.info("Saving model checkpoint to %s", args.output_dir)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, evaluate_metrics="ppl", prefix='0'):
    eval_output_dir = args.output_dir

    if prefix == 'test':
        eval_data_file = os.path.join(args.data_file, args.test_data_file)
    elif prefix == 'train':
        eval_data_file = os.path.join(args.data_file, args.train_data_file)
    else:
        prefix = 'dev'
        eval_data_file = os.path.join(args.data_file, args.dev_data_file)

    if 'rocstory' in args.task_type:
        dh = roc_DatasetHelper
    elif 'kg' in args.task_type:
        dh = kg_DatasetHelper
    elif 'kemp' in args.task_type:
        dh = kemp_DatasetHelper
    else:
        dh = ed_DatasetHelper
    eval_dataset = dh(args,
                      tokenizer,
                      data_path=eval_data_file,
                      src_max_length=args.source_length,
                      tgt_max_length=args.target_length,
                      do_generate=evaluate_metrics == 'bleu')
    eval_dataset.load()
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True,
                                 num_workers=args.workers, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    gen_seqs = []
    multi_gen_seqs = []
    src = []
    tgt = []
    nb_eval_steps = 0
    model.eval()

    if 'bleu' in evaluate_metrics:
        generator = build_generator(args, eval_dataset)
    else:
        generator = None
    top_ids = []
    Hit_num = 0
    # for bi, batch in enumerate(eval_dataloader):
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if args.do_eval:
            src.extend(batch[1])
            tgt.extend(batch[2])
            batch = tuple(t.to(args.device) for t in batch[0])
        else:
            batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if 'bleu' in evaluate_metrics:
                bleu_batch = {"src_input_ids": batch[0],
                              "src_input_type": batch[1],
                              "attention_mask": batch[2],
                              "src_position_ids": batch[3],
                              "seq_generator": generator,
                              "greedy_or_beam": "greedy" if args.beam == 1 else "beam"}
                if isinstance(model, torch.nn.DataParallel):
                    hypos, multi_hypos = model.module.aotoreg_generate(**bleu_batch)
                else:
                    hypos, multi_hypos = model.aotoreg_generate(**bleu_batch)
                gen_seqs.extend(hypos)
                multi_gen_seqs.extend(multi_hypos)
            if 'ppl' in evaluate_metrics:
                eval_batch = [batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]]
                eval_loss += torch.mean(model(*eval_batch)).item()
        nb_eval_steps += 1
    result = {}
    if 'ppl' in evaluate_metrics:
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result["ppl"] = perplexity

    if 'bleu' in evaluate_metrics:
        print_data = {'src': src, 'tgt': tgt, 'prediction': gen_seqs, 'prefix': prefix, 'multi_prediction': multi_gen_seqs if args.beam>1 else None, }
        save_generation(args, **print_data)

        references = [[[x for x in y.strip().split(' ')]] for y in tgt]  # each sample can have multiple references
        if args.beam > 1:
            predictions = [x.strip().split() for x in gen_seqs]
        else:
            predictions = [x['prediction'].strip().split() for x in gen_seqs]
        assert len(gen_seqs) == len(tgt), 'prediction is not finished !'
        bleu1 = _compute_bleu(references, predictions, max_order=1)
        bleu2 = _compute_bleu(references, predictions, max_order=2)
        bleu4 = _compute_bleu(references, predictions, max_order=4)
        if prefix == 'test':
            result["blue1"] = bleu1[0]
            result["blue2"] = bleu2[0]
            result["bleu4"] = bleu4[0]
        else:
            result["bleu"] = bleu4[0]
    if 'dist' in evaluate_metrics:
        predictions = [x.split() for x in gen_seqs]
        ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = _get_dist(predictions)
        result['dist-1'] = mi_dist1 * 100
        result['dist-2'] = mi_dist2 * 100
    if 'acc' in evaluate_metrics:
        save_generation(args, top_ids, prefix=prefix)
        result = {
            "acc": Hit_num / len(eval_dataset),
        }

    return result


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


def save_generation(args, src, tgt, prediction, prefix='0', multi_prediction=None,):
    save_result_dir = os.path.join(args.output_dir, "result_ep:{}.txt".format(prefix))
    with open(save_result_dir, 'w') as f:
        for i, line in enumerate(src):
            f.write('-----------' + str(i) + '-----------' + '\n')
            f.write('input: ' + line + '\n')
            f.write('target: ' + tgt[i].strip() + '\n')
            if multi_prediction is not None or args.beam > 1:  # beam size > 1
                f.write('prediction: ' + prediction[i] + '\n')
                f.write('multi_prediction: ' + str(multi_prediction[i]) + '\n')
            else:
                f.write('prediction: ' + prediction[i]['prediction'] + '\n')

    logger.info("Save generation result in {}".format(save_result_dir))


class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def add_custom_tokens(tokenizer):
    num_added_toks = tokenizer.add_tokens(['<|bos|>',
                                           '<|pad|>',
                                           '<|endoftext|>',
                                           '[name]'])
    return tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The path to the input training data file (a text file).")
    parser.add_argument("--train_data_file", default="ed_train_prop.json", type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--dev_data_file", default="ed_dev_prop.json", type=str, required=True,
                        help="The input dev data file (a text file).")
    parser.add_argument("--test_data_file", default="ed_test_prop.json", type=str, required=True,
                        help="The input test data file (a text file).")
    parser.add_argument("--train_rel_data_file", default="ed_train_prop.json", type=str,
                        help="The input training relation ids data file (a text file).")
    parser.add_argument("--dev_rel_data_file", default="ed_dev_prop.json", type=str,
                        help="The input dev relation ids data file (a text file).")
    parser.add_argument("--test_rel_data_file", default="ed_test_prop.json", type=str,
                        help="The input test relation ids data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_type", default="roc_last_sentence", type=str, help="downstream task type ...")
    parser.add_argument("--ending_or_complement", default="ending", type=str, help="for rocstories dataset ...")

    parser.add_argument("--source_length", default=100, type=int, help="max source len")
    parser.add_argument("--target_length", default=100, type=int, help="max target_len")
    parser.add_argument("--tb_log_dir", default=None, type=str, help="log")
    parser.add_argument("--evaluate_metrics", default='ppl', type=str, help='choose between ppl and bleu')

    parser.add_argument("--model_type", default="gpt2", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_generate", action='store_true', help="generate data using trained model")
    parser.add_argument("--continue_train", action='store_true',
                        help="Whether to run training based on a trained checkpoint.")
    parser.add_argument("--workers", default=0, type=int, help="workers")
    parser.add_argument("--save_last", action='store_true', help="whether save the last epoch")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
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
    parser.add_argument("--beam", default=1, type=int,
                        help="beam_size")  # 取出概率最大的k个词构成一个集合，然后将这个子集词的概率再归一化，最后重新的概率分布中采样词汇
    parser.add_argument("--sampling", action='store_true', help="whether topk sampling or topp sampling")
    parser.add_argument("--sampling_topk", default=-1, type=int,
                        help="topk sampling")  # 取出概率最大的k个词构成一个集合，然后将这个子集词的概率再归一化，最后重新的概率分布中采样词汇
    parser.add_argument("--sampling_topp", default=-1, type=float,
                        help="topp sampling")  # 固定候选集合的概率密度和在整个概率分布中的比例。也就是构造一个最小候选集，使得集合概率和大于P
    parser.add_argument("--temperature", default=1, type=float,
                        help="topp sampling")  # 固定候选集合的概率密度和在整个概率分布中的比例。也就是构造一个最小候选集，使得集合概率和大于P

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--prediction_dir", default='test', type=str, help='for prediction')

    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    if args.do_eval:
        load_from_path = args.output_dir
        if args.beam > 1:
            args.prediction_dir = '_beam' + str(args.beam) + \
                                  '_topk' + str(args.sampling_topk) + \
                                  '_temp' + str(args.temperature) + \
                                  '_' + args.prediction_dir
        args.output_dir = os.path.join(args.output_dir, args.prediction_dir)
    elif args.continue_train:
        load_from_path = args.output_dir
        logger.info('Continue training from {}'.format(args.output_dir))
    else:
        load_from_path = args.model_name_or_path

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
    tokenizer = tokenizer_class.from_pretrained(load_from_path, do_lower_case=False)
    config = config_class.from_pretrained(load_from_path)
    model = model_class.from_pretrained(load_from_path,
                                        source_length=args.source_length,
                                        tokenizer=tokenizer, )
    # for n,p in model.named_parameters():
    #     print(n)
    tokenizer = add_custom_tokens(tokenizer)
    if args.do_train or args.continue_train:
        model.resize_token_embeddings(len(tokenizer))  # include <|bos|>, <|pad|>, <|endoftext|>
    model.to(args.device)

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if 'roc' in args.task_type:
            dh = roc_DatasetHelper
        elif 'kg' in args.task_type:
            dh = kg_DatasetHelper
        else:
            dh = ed_DatasetHelper

        train_dataset = dh(args,
                           tokenizer,
                           data_path=os.path.join(args.data_file, args.train_data_file),
                           relation_path=os.path.join(args.data_file, args.train_rel_data_file),
                           src_max_length=args.source_length,
                           tgt_max_length=args.target_length, )
        train_dataset.load()
        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        print('len tok: ', len(tokenizer))
        result = evaluate(args, model, tokenizer, args.evaluate_metrics, 'test')
        logger.info("Test evaluate {}".format(args.evaluate_metrics))
        for k in result:
            logger.info("{}: {:.4f}".format(k, result[k]))


if __name__ == '__main__':
    main()
