# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import pdb
import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
import logging 

logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



class SequenceGenerator(object):
    def __init__(
        self,
        args,
        tgt_dict,
        tokenizer, 
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1.0,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            id2tok_fn (HuggingFace tokenizer.decode()): Used to transfer bpe indices to words
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_topp (float, optional): only sample among the smallest set
                of words whose cumulative probability mass exceeds p
                at each step (default: -1.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.args = args
        self.tgt_dict = tgt_dict
        self.tokenizer = tokenizer
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.bos = tgt_dict.bos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.gpt2_max_length = 1024

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)
        # "xAttr": 50262, "xEffect": 50263, "xIntent": 50264, "xNeed": 50265, "xReact": 50266, "xWant": 50267, "oEffect": 50268, "oReact": 50269, "oWant": 50270, "_xAttr": 50271,"_xEffect": 50272, "_xIntent": 50273, "_xNeed": 50274, "_xReact": 50275, "_xWant": 50276, "_oEffect": 50277, "_oReact": 50278, "_oWant": 50279
        # self.relation_ids = [50262, 50263, 50264, 50265, 50266, 50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279]
        # self.relation_begin_ids = [87,78,62]

        self.relation_to_alpha = {'oEffect':'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact':'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q', '_xWant': 'R', 'PRP': 'S',}
        self.alpha_to_relation = {'A': 'oEffect', 'B': 'oReact', 'C': 'oWant', 'D': 'xAttr', 'E': 'xEffect',
                                  'F': 'xIntent', 'G': 'xNeed', 'H': 'xReact', 'I': 'xWant', 'J': '_oEffect', 'K': '_oReact', 'L': '_oWant',
                                  'M': '_xAttr', 'N': '_xEffect', 'O': '_xIntent', 'P': '_xNeed', 'Q': '_xReact', 'R': '_xWant', 'S': 'PRP'}
        self.alpha_to_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18}
        self.id_to_alpha = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S'}

    @torch.no_grad()
    def ensemble_generate(self, models, sample, **kwargs):
        
        model = EnsembleModel(models)
        return self.generate(model, sample, **kwargs)

    @torch.no_grad()
    def greedy_generate(self,
                        model_forward,
                        sample,
                        relation_embed=False,
                        output_text=True,
                        ):
        bos_token = self.tgt_dict.bos()

        encoder_outs = {}
        k_order = ["input_ids", "input_rel_ids", "attention_mask", "position_ids", "tgt_bos_rel_ids", "is_end_at_relation", "past",]
        for i, (k,v) in enumerate(sample.items()):
            assert(k_order[i] == k), "generator input sample should be Dict with keys ordered in {}!".format(k_order)
            if k == 'past' or "attention_mask":
                encoder_outs[k] = v
            else:
                encoder_outs[k] = v[:,:self.args.source_length]  # pop out bos

        src_tokens = encoder_outs["input_ids"]
        src_lengths = encoder_outs["input_ids"].long().sum(dim=1)
        input_size = src_tokens.size()
        bsz = input_size[0]
        src_len = input_size[1]

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.gpt2_max_length - 1,
            )

        # prediction memory
        tokens = src_tokens.new(bsz, max_len + 2).to(src_tokens.device).long().fill_(self.pad)
        tokens[:, 0] = bos_token

        tokens_rels = src_tokens.new(bsz, max_len+2).to(src_tokens.device).long().fill_(self.pad)  # new() 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容
        tokens_rels[:, 0] = encoder_outs["tgt_bos_rel_ids"].squeeze(1)  # (bsz, 1) -> (bsz, )


        for step in range(max_len + 1):  # one extra step for EOS marker
            encoder_outs["input_ids"] = torch.cat((encoder_outs["input_ids"][:, :src_len], tokens[:, :step + 1]), dim=1)
            encoder_outs["input_rel_ids"] = torch.cat((encoder_outs["input_rel_ids"][:, :src_len], tokens_rels[:, :step+1]), dim=1)
            encoder_outs["attention_mask"] = torch.cat((encoder_outs["attention_mask"],
                                                        encoder_outs["attention_mask"].new_ones(encoder_outs["attention_mask"].size(0), 1)), dim=1)
            encoder_outs["position_ids"] = torch.cat((encoder_outs["position_ids"],
                                                      encoder_outs["position_ids"].new_ones(
                                                          encoder_outs["position_ids"].size(0), 1) * step), dim=1)
            assert encoder_outs["input_ids"].size() == encoder_outs["input_rel_ids"].size()
            probs = model_forward(input_ids=encoder_outs["input_ids"],
                                  input_rel_ids=encoder_outs["input_rel_ids"],
                                  attention_mask=encoder_outs["attention_mask"],
                                  position_ids=encoder_outs["position_ids"],
                                  past=encoder_outs["past"],
                                  do_generate=True,
                                  relation_embed=relation_embed,)

            probs = probs[:, -1, :]
            if not self.args.sampling:
                lprobs = probs.log()
            else:
                lprobs = probs

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            if step >= max_len:  # 强行选择eos结束掉
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
            elif step < self.min_len:  # 不要选择eos
                lprobs[:, self.eos] = -math.inf

            end_at_relations = encoder_outs["is_end_at_relation"]  # todo: mask metric 在数据预处理的时候准备好
            for bidx in range(bsz):
                lprobs_bsz = lprobs[bidx]


                next_logits, next_token = lprobs_bsz.topk(k=1, dim=0)
                tokens[bidx, step+1] = next_token  # 持续update

                # if prediction belongs to relation_id, then change its corresponding relation id to the relation_id
                last_tokens_rels = tokens_rels[bidx, step]
                if next_token.item() in self.id_to_alpha:  # 一个词对应一个id，可以这么判断
                    cur_tokens_rels = next_token.item()
                else:
                    cur_tokens_rels = last_tokens_rels.clone()
                tokens_rels[bidx, step+1] = cur_tokens_rels  # 持续update

        result_strs = []
        result_rels = []

        for j in range(bsz):
            hypo = tokens[j]
            hypo_rels = tokens_rels[j]
            hypo_rels = hypo_rels.tolist()[1:hypo.tolist().index(self.eos)]
            hypo = hypo.tolist()[1:hypo.tolist().index(self.eos)]
            if self.pad in encoder_outs["input_ids"][j, :src_len].tolist():
                ipt_words = encoder_outs["input_ids"][j, :src_len].tolist()[
                        :encoder_outs["input_ids"][j, :src_len].tolist().index(self.pad)]
            else:
                ipt_words = encoder_outs["input_ids"][j, :src_len].tolist()
            if output_text:
                pred_words = self.tokenizer.decode(hypo)
                ipt_words = self.tokenizer.decode(ipt_words)
                for alpha in self.alpha_to_relation:  # 把A这些替代的relation word换成xWant这些
                    pred_words = pred_words.replace(' ' + alpha + ' ', ' ' + self.alpha_to_relation[alpha] + ' ')
                    ipt_words = ipt_words.replace(' ' + alpha + ' ', ' ' + self.alpha_to_relation[alpha] + ' ')

                result_strs.append({'input': ipt_words.strip(), 'prediction': pred_words.strip()})
                result_rels.append(hypo_rels)
            else:
                result_strs.append({'input': ipt_words.strip(), 'prediction': hypo})
                result_rels.append(hypo_rels)
        return result_strs, result_rels

    @torch.no_grad()
    def beam_generate(
            self,
            model_forward,
            sample,
            prefix_tokens=None,
            output_text=True,
            **kwargs
    ):
        # if not self.retain_dropout:
        #    model.eval()
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
                - input_ids
                - attention_mask
                - position_ids
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
        """
        bos_token = self.tgt_dict.bos()

        encoder_outs = {}
        k_order = ["input_ids", "input_rel_ids", "attention_mask", "position_ids", "tgt_bos_rel_ids", "is_end_at_relation", "past",]
        for i, (k, v) in enumerate(sample.items()):
            assert (k_order[i] == k), "generator input sample should be Dict with keys ordered in {}!".format(k_order)
            if k == 'past' or k == 'attention_mask':
                encoder_outs[k] = v
            else:
                encoder_outs[k] = v[:,:self.args.source_length]  # pop out bos

        src_tokens = encoder_outs["input_ids"]
        src_lengths = encoder_outs["input_ids"].long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.gpt2_max_length - 1,
            )

        # compute the encoder output for each beam
        # encoder_outs = model.forward_encoder(encoder_input)

        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).to(src_tokens.device).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).to(src_tokens.device).long().fill_(self.pad)
        tokens_buf = tokens.clone()

        tokens[:, 0] = bos_token

        tokens_rels = src_tokens.new(bsz * beam_size, max_len + 2).to(src_tokens.device).long().fill_(self.pad)  # new() 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容
        tokens_rels_buf = tokens_rels.clone()
        tokens_rels[:, 0] = encoder_outs["tgt_bos_rel_ids"].squeeze(1)  # (bsz, 1) -> (bsz, )

        attn, attn_buf = None, None
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens, device=src_tokens.device):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new().to(device)
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs.to(src_tokens.device) - torch.arange(batch_idxs.numel()).type_as(batch_idxs).to(
                        src_tokens.device)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                encoder_outs = reorder_encoder_out(encoder_outs, reorder_state)

            encoder_outs["input_ids"] = torch.cat((encoder_outs["input_ids"][:, :src_len], tokens[:, :step + 1]), dim=1)
            encoder_outs["input_rel_ids"] = torch.cat((encoder_outs["input_rel_ids"][:, :src_len], tokens_rels[:, :step + 1]), dim=1)
            encoder_outs["attention_mask"] = torch.cat((encoder_outs["attention_mask"],encoder_outs["attention_mask"].new_ones(encoder_outs["attention_mask"].size(0), 1)), dim=1)
            encoder_outs["position_ids"] = torch.cat((encoder_outs["position_ids"],encoder_outs["position_ids"].new_ones(encoder_outs["position_ids"].size(0), 1) * step), dim=1)

            probs = model_forward(input_ids=encoder_outs["input_ids"],
                                  input_rel_ids=encoder_outs["input_rel_ids"],
                                  attention_mask=encoder_outs["attention_mask"],
                                  position_ids=encoder_outs["position_ids"],
                                  do_generate=True,
                                  past=encoder_outs["past"],
                                  temperature=self.temperature,)

            probs = probs[:, -1, :]
            if not self.args.sampling:
                lprobs = probs.log()
            else:
                lprobs = probs

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            assert torch.isnan(lprobs).sum() == 0, print('prob errors')

            # handle min and max length constraints
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
            elif step < self.min_len:
                lprobs[:, self.eos] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1):
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            avg_attn_scores = None
            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            bbsz_offsets = bbsz_offsets.to(cand_beams.device)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos (except for blacklisted ones)
            eos_mask = cand_indices.eq(self.eos)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents) # 总batch大小 - 完成的样本数

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)

                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)

                # todo: for relation embedding
                tokens_rels = tokens_rels.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_rels_buf.resize_as_(tokens_rels)

                # todo: for prefix
                new_past_key_values_prompt = []
                for key_values_prompt in encoder_outs["past"]:
                    # (2, batch_size, num_head, sql_len + 1, head_features)
                    new_key_values_prompt = key_values_prompt.view(
                        key_values_prompt.size(0),  # 2
                        bsz,
                        -1,
                        key_values_prompt.size(2),
                        key_values_prompt.size(3),
                        key_values_prompt.size(4),
                    ).index_select(1, batch_idxs).view(
                        key_values_prompt.size(0),
                        new_bsz * beam_size,
                        key_values_prompt.size(2),
                        key_values_prompt.size(3),
                        key_values_prompt.size(4),
                    )
                    new_past_key_values_prompt.append(new_key_values_prompt)
                encoder_outs["past"] = new_past_key_values_prompt

                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                (eos_mask.type_as(cand_offsets) * cand_size).to(src_tokens.device),
                cand_offsets[:eos_mask.size(1)].to(src_tokens.device),
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(  # previous tokens of active bbsz
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(  # new prediction given the previous tokens of active bbsz
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )

            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores


            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

            # todo: for relation embedding
            end_at_relations = encoder_outs["is_end_at_relation"]
            for bb_idx in range(tokens.size(0)):  # bsz * beam_size
                next_token = tokens[bb_idx, step + 1]
                last_tokens_rels = tokens_rels_buf[bb_idx, step]
                if next_token.item() in self.id_to_alpha:
                    cur_tokens_rels = next_token.item()
                else:
                    cur_tokens_rels = last_tokens_rels.item()
                tokens_rels[bb_idx, step + 1] = cur_tokens_rels  # 持续update

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        result_strs = []
        multi_result_strs = []
        result_scores = []
        for j, hypo in enumerate(finalized):
            hypo = hypo[0]
            hypo_tokens = hypo['tokens']
            result_scores.append(hypo['score'])
            if output_text:
                bsz_text = self.tokenizer.decode(hypo_tokens.tolist()[:-1])
                for alpha in self.alpha_to_relation:  # 把A这些替代的relation word换成xWant这些
                    bsz_text = bsz_text.replace(' ' + alpha + ' ', ' ' + self.alpha_to_relation[alpha] + ' ')
                result_strs.append(bsz_text.strip())
            else:
                result_strs.append(hypo_tokens.tolist()[:-1])

        for j, hypos in enumerate(finalized):
            sample_strs = []
            for hypo in hypos:
                hypo_tokens = hypo['tokens']
                if output_text:
                    this_hypo = self.tokenizer.decode(hypo_tokens.tolist()[:-1])
                    for alpha in self.alpha_to_relation:  # 把A这些替代的relation word换成xWant这些
                        this_hypo = this_hypo.replace(' ' + alpha + ' ', ' ' + self.alpha_to_relation[alpha] + ' ')
                    sample_strs.append(this_hypo.strip())
                else:
                    sample_strs.append(hypo_tokens.tolist()[:-1])
            multi_result_strs.append(sample_strs)

        return result_strs, multi_result_strs  # , result_scores

def reorder_encoder_out(encoder_outs, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_outs = {}
        for key, encoder_out in encoder_outs.items():
            if key == 'past':
                new_past = []
                for key_values_prompt in encoder_outs['past']:
                    new_key_values_prompt = key_values_prompt.index_select(1, new_order) # batch size dim=1
                    new_past.append(new_key_values_prompt)
                new_encoder_outs[key] = new_past
            else:
                new_encoder_outs[key] = encoder_out.index_select(0, new_order)
        return new_encoder_outs



class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


class SequenceGeneratorWithAlignment(SequenceGenerator):

    def __init__(self, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = EnsembleModelWithAlignment(models)
        finalized = super()._generate(model, sample, **kwargs)

        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = \
            self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in model.models):
            attn = model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn']
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn