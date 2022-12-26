import torch
import os
import json
import logging
import csv
import itertools
from torch.utils.data import Dataset
import random
import pdb

from transformers import BertTokenizer

logger = logging.getLogger()

def normalize_case(text):
    if len(text) > 1:
        try:
            normalized = text[0].upper() + text[1:].lower()
            if normalized[-1] != '.':
                normalized = normalized + '.'
        except:
            raise RuntimeError("Cannot normalize text {}".format(text))
        return normalized
    return text

import pdb
def get_relation_dist(data_path, o_path):
    o = open(o_path, 'w')
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            items = line.split('\t')
            ids = []
            for j, ec in enumerate(items):
                if j == len(items)-1:
                    ids.extend(len(ec.split())*[items[j-1]])
                else:
                    if (j+1) % 2 != 0:
                        ids.extend(len(ec.split())*[items[j+1]])
                    else:
                        ids.append(ec)
            o.write(' '.join(ids)+'\n')


class DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100, do_generate=False):
        self.do_generate = do_generate 
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.unk = self.tokenizer.encoder["<|unk|>"]
        self.data_path = data_path
        self.relation_path = relation_path
        self.relation_list = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant', 'PRP',]

        self.relation_to_alpha = {'oEffect':'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact':'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q', '_xWant': 'R', 'PRP': 'S',}

        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                                  'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                                  'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                                  '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                                  '_xWant': 17, 'PRP': 18,}

    def load(self):
        self.source = []   # i am happy xWant i want to sing and dance
        self.target = []   # oReact he is surprised by my action
        self.source_rel = []  # PRP xWant xWant
        self.target_rel = []  # oReact oReact

        rels = json.load(open(self.relation_path, 'r'))
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().strip('\n').split('<SEP>')
                rel = rels[i]
                src_ipt = line[0].split('\t')
                src_ipt.remove('')
                src_rel = rel[:len(src_ipt)]

                tgt_opt = line[1].split('\t')
                tgt_opt.remove('')
                tgt_rel = rel[len(src_ipt):]
                assert len(tgt_opt) == len(tgt_rel), pdb.set_trace()

                self.source.append(src_ipt)
                self.target.append(tgt_opt)
                self.source_rel.append(src_rel)
                self.target_rel.append(tgt_rel)

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-"*50 + "Features" + "-"*50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id,min(sample_id+3, len(self.source)))]
        for ex in exs:
            if self.args.do_eval:
                ex = ex[0]
            logger.info("Input: {}".format(self.tokenizer.decode(ex[0].tolist())))
            logger.info("Relation id: {}".format(ex[1].tolist()))
            logger.info("Attention mask: {}".format(ex[2].tolist()))
            logger.info("Position: {}".format(ex[3].tolist()))
            logger.info("Target: {}".format(self.tokenizer.decode(ex[4].tolist())))
            logger.info("Target Relation id: {}".format(ex[5].tolist()))
            logger.info("Position: {}".format(ex[6].tolist()))
            logger.info("Labels: {}".format(self.tokenizer.decode(ex[7].masked_select(ex[7]>=0).tolist())))


    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        src_rel = self.source_rel[idx]
        tgt_rel = self.target_rel[idx]

        src_input_ids = []
        src_input_rel_ids = []
        r_type = 1  # todo
        for si, s in enumerate(src):
            if (si+1) % 2 == 0:  # relation word because the kg path of input must start from event word:
                sid = self.tokenizer.encode(' '+self.relation_to_alpha[s.strip()])
                r_type += 1
                rid = [r_type] * len(sid)
            else:
                sid = self.tokenizer.encode(' '+s)
                rid = [r_type] * len(sid)

            src_input_ids.extend(sid)
            src_input_rel_ids.extend(rid)

        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]
            src_input_rel_ids = src_input_rel_ids[:self.src_max_length]

        src_position_ids = list(range(0, len(src_input_ids)))
        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_input_rel_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
        assert len(src_input_ids) == len(src_input_rel_ids)
        
        target_input_ids = [self.bos]
        target_input_rel_ids = [r_type]
        target_position_ids = []
        labels = []
        target_bos_rel_ids = []
        target_bos_rel_ids.append(self.relation_to_id[src_rel[-1]])

        is_end_at_relation = []
        if src[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        if not self.do_generate:
            for ti, t in enumerate(tgt):
                if t.strip() in self.relation_list:
                    tid = self.tokenizer.encode(' ' + self.relation_to_alpha[t.strip()])
                    r_type += 1
                    rid = [r_type] * len(tid)
                else:
                    tid = self.tokenizer.encode(' ' + t)
                    rid = [r_type] * len(tid)
                target_input_ids.extend(tid)
                target_input_rel_ids.extend(rid)

            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
                target_input_rel_ids = target_input_rel_ids[:self.tgt_max_length]

            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_input_rel_ids += [self.pad]
                target_position_ids += [0]
                labels += [-1]
            assert len(target_input_ids) == len(target_input_rel_ids) == len(labels), pdb.set_trace()
        labels = [-1] * self.src_max_length + labels

        if self.args.do_eval:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_input_rel_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_input_rel_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation)
                    ), '  '.join(src), '  '.join(tgt),
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_input_rel_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_input_rel_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    )


class roc_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False):
        self.do_generate = do_generate
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.unk = self.tokenizer.encoder["<|unk|>"]
        self.data_path = data_path
        self.relation_path = relation_path  # for downstream task.

        self.relation_list = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant', 'PRP',]

        self.relation_to_alpha = {'oEffect':'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact':'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q', '_xWant': 'R', 'PRP': 'S',}

        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                                  'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                                  'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                                  '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                                  '_xWant': 17, 'PRP': 18,}

    def load(self):
        self.source = []  # i am happy xWant i want to sing and dance
        self.target = []  # oReact he is surprised by my action
        self.source_rel = []  # PRP xWant xWant
        self.target_rel = []  # oReact oReact

        case_ids = [2, 5, 15, 14, 28, 34, 40, 42, 97, 123, 190, 251, 272, 292, 358, 3210, 2611, 2504, 2466, 1941, 1912, 1860, 1836, 1690, 1679, 1662, 1647, 1636, 1633, 1628, 1616, 1614, 1605, 1557, 1554, 1539, 1525, 1503, 1500, 1486, 1484, 1480, 1469, 1450, 1432, 1417, 1400, 1398, 1337, 1323]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            # for id in case_ids:
            #     item = data[id]
            #     whole_knowledge_path = item['context_ending_path']
            #     whole_relation_ids = item['context_ending_path_relation_ids']
            #
            #     context_knowledge_path = whole_knowledge_path[0]
            #     response_knowledge_path = whole_knowledge_path[1]
            #
            #     context_relation_ids = whole_relation_ids[0]
            #     response_relation_ids = whole_relation_ids[1]
            #
            #     self.source.append(context_knowledge_path)
            #     self.target.append(response_knowledge_path)
            #     self.source_rel.append(context_relation_ids)
            #     self.target_rel.append(response_relation_ids)
            for i, item in enumerate(data):
                whole_knowledge_path = item['context_ending_path']
                whole_relation_ids = item['context_ending_path_relation_ids']

                context_knowledge_path = whole_knowledge_path[0]
                response_knowledge_path = whole_knowledge_path[1]

                context_relation_ids = whole_relation_ids[0]
                response_relation_ids = whole_relation_ids[1]

                self.source.append(context_knowledge_path)
                self.target.append(response_knowledge_path)
                self.source_rel.append(context_relation_ids)
                self.target_rel.append(response_relation_ids)

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id,min(sample_id+3, len(self.source)))]
        for ex in exs:
            if self.args.do_eval:
                ex = ex[0]
            logger.info("Input: {}".format(self.tokenizer.decode(ex[0].tolist())))
            logger.info("Relation id: {}".format(ex[1].tolist()))
            logger.info("Attention mask: {}".format(ex[2].tolist()))
            logger.info("Position: {}".format(ex[3].tolist()))
            logger.info("Target: {}".format(self.tokenizer.decode(ex[4].tolist())))
            logger.info("Target Relation id: {}".format(ex[5].tolist()))
            logger.info("Position: {}".format(ex[6].tolist()))
            logger.info("Labels: {}".format(self.tokenizer.decode(ex[7].masked_select(ex[7]>=0).tolist())))


    # def __getitem__(self, idx):
    #     src = self.source[idx]
    #     tgt = self.target[idx]
    #     src_rel = self.source_rel[idx]
    #     tgt_rel = self.target_rel[idx]
    #
    #     src_input_ids = []
    #     src_input_rel_ids = []
    #     r_type = 0  # todo
    #     for si, s in enumerate(src):
    #         if (si + 1) % 2 == 0:  # relation word because the kg path of input must start from event word:
    #             sid = self.tokenizer.encode(' ' + self.relation_to_alpha[s.strip()])
    #             r_type += 1
    #             rid = [r_type] * len(sid)
    #         else:
    #             sid = self.tokenizer.encode(' ' + s)
    #             rid = [r_type] * len(sid)
    #
    #         src_input_ids.extend(sid)
    #         src_input_rel_ids.extend(rid)
    #         # src_input_rel_ids.extend([self.relation_to_id[src_rel[si]]] * len(sid))
    #
    #     if len(src_input_ids) > self.src_max_length:
    #         src_input_ids = src_input_ids[:self.src_max_length]
    #         src_input_rel_ids = src_input_rel_ids[:self.src_max_length]
    #
    #     src_position_ids = list(range(0, len(src_input_ids)))
    #     attention_mask = [1] * len(src_input_ids)
    #
    #     while len(src_input_ids) < self.src_max_length:
    #         src_input_ids += [self.pad]
    #         src_input_rel_ids += [self.pad]
    #         src_position_ids += [0]
    #         attention_mask += [0]
    #     assert len(src_input_ids) == len(src_input_rel_ids)
    #
    #     target_input_ids = [self.bos]
    #     target_input_rel_ids = [r_type]
    #     target_position_ids = []
    #     labels = []
    #     target_bos_rel_ids = [r_type]  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias
    #
    #     is_end_at_relation = []
    #     if src[-1].strip() in self.relation_list:
    #         is_end_at_relation.append(1)
    #     else:
    #         is_end_at_relation.append(0)
    #
    #     if not self.do_generate:
    #         for ti, t in enumerate(tgt):
    #             if t.strip() in self.relation_list:
    #                 tid = self.tokenizer.encode(' ' + self.relation_to_alpha[t.strip()])
    #                 r_type += 1
    #                 rid = [r_type] * len(tid)
    #             else:
    #                 tid = self.tokenizer.encode(' ' + t)
    #                 rid = [r_type] * len(tid)
    #
    #             target_input_ids.extend(tid)
    #             target_input_rel_ids.extend(rid)
    #
    #         if len(target_input_ids) > self.tgt_max_length:
    #             target_input_ids = target_input_ids[:self.tgt_max_length]
    #             target_input_rel_ids = target_input_rel_ids[:self.tgt_max_length]
    #
    #         target_position_ids = list(range(0, len(target_input_ids)))
    #         labels = target_input_ids[1:] + [self.eos]
    #
    #         while len(target_input_ids) < self.tgt_max_length:
    #             target_input_ids += [self.pad]
    #             target_input_rel_ids += [self.pad]
    #             target_position_ids += [0]
    #             labels += [-1]
    #         assert len(target_input_ids) == len(target_input_rel_ids) == len(labels), pdb.set_trace()
    #     labels = [-1] * self.src_max_length + labels
    #
    #     if self.args.do_eval:
    #         return (torch.tensor(src_input_ids),
    #                 torch.tensor(src_input_rel_ids),
    #                 torch.tensor(attention_mask),
    #                 torch.tensor(src_position_ids),
    #                 torch.tensor(target_input_ids),
    #                 torch.tensor(target_input_rel_ids),
    #                 torch.tensor(target_position_ids),
    #                 torch.tensor(labels),
    #                 torch.tensor(target_bos_rel_ids),
    #                 torch.tensor(is_end_at_relation)
    #                 ), '  '.join(src), '  '.join(tgt),
    #     else:
    #         return (torch.tensor(src_input_ids),
    #                 torch.tensor(src_input_rel_ids),
    #                 torch.tensor(attention_mask),
    #                 torch.tensor(src_position_ids),
    #                 torch.tensor(target_input_ids),
    #                 torch.tensor(target_input_rel_ids),
    #                 torch.tensor(target_position_ids),
    #                 torch.tensor(labels),
    #                 torch.tensor(target_bos_rel_ids),
    #                 torch.tensor(is_end_at_relation)
    #                 )


    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        src_rel = self.source_rel[idx]
        tgt_rel = self.target_rel[idx]

        src_input_ids = []
        src_input_rel_ids = []
        r_type = 0  # todo
        for si, s in enumerate(src):
            if (si + 1) % 2 == 0:  # relation word because the kg path of input must start from event word:
                sid = self.tokenizer.encode(' ' + self.relation_to_alpha[s.strip()])
                r_type += 1
                rid = [r_type] * len(sid)
            else:
                sid = self.tokenizer.encode(' ' + s)
                rid = [r_type] * len(sid)

            src_input_ids.extend(sid)
            src_input_rel_ids.extend(rid)

        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]
            src_input_rel_ids = src_input_rel_ids[:self.src_max_length]

        src_position_ids = list(range(0, len(src_input_ids)))
        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_input_rel_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
        assert len(src_input_ids) == len(src_input_rel_ids)


        target_input_ids = [self.bos]
        target_input_rel_ids = [r_type]
        target_position_ids = []
        labels = []
        target_bos_rel_ids = [r_type]  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias

        is_end_at_relation = []
        if src[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        if not self.do_generate:
            for ti, t in enumerate(tgt):
                if t.strip() in self.relation_list:
                    tid = self.tokenizer.encode(' ' + self.relation_to_alpha[t.strip()])
                    r_type += 1
                    rid = [r_type] * len(tid)
                else:
                    tid = self.tokenizer.encode(' ' + t)
                    rid = [r_type] * len(tid)

                target_input_ids.extend(tid)
                target_input_rel_ids.extend(rid)

            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
                target_input_rel_ids = target_input_rel_ids[:self.tgt_max_length]

            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_input_rel_ids += [self.pad]
                target_position_ids += [0]
                labels += [-1]
            assert len(target_input_ids) == len(target_input_rel_ids) == len(labels), pdb.set_trace()
        labels = [-1] * self.src_max_length + labels

        if self.args.do_eval:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_input_rel_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_input_rel_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation)
                    ), '  '.join(src), '  '.join(tgt),
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_input_rel_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_input_rel_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation)
                    )


class ed_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False):
        self.do_generate = do_generate
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.unk = self.tokenizer.encoder["<|unk|>"]
        self.data_path = data_path
        self.relation_path = relation_path  # for downstream task.

        self.relation_list = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant', 'PRP',]

        self.relation_to_alpha = {'oEffect':'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact':'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q', '_xWant': 'R', 'PRP': 'S',}

        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                                  'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                                  'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                                  '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                                  '_xWant': 17, 'PRP': 18,}

    def load(self):
        self.source = []  # i am happy xWant i want to sing and dance
        self.target = []  # oReact he is surprised by my action
        self.source_rel = []  # PRP xWant xWant
        self.target_rel = []  # oReact oReact

        case_ids = [4, 9, 45, 25, 28, 31, 32, 33, 37, 39, 40, 68, 95, 128, 140, 165, 178, 206, 214, 255, 399, 400, 416, 449, 459, 489, 493, 510, 590, 648, 764, 765, 765, 820, 885, 907, 937, 945, 962, 1000, 1010, 1094, 1119, 1123, 1147, 1188, 1235, 1354, 1374, 1441]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for id in case_ids:
                item = data[id]
                whole_knowledge_path = item['context_response_path']
                whole_relation_ids = item['context_response_path_relation_ids']

                context_knowledge_path = whole_knowledge_path[0]
                response_knowledge_path = whole_knowledge_path[1]

                context_relation_ids = whole_relation_ids[0]
                response_relation_ids = whole_relation_ids[1]

                self.source.append(context_knowledge_path)
                self.target.append(response_knowledge_path)
                self.source_rel.append(context_relation_ids)
                self.target_rel.append(response_relation_ids)

            # for i, item in enumerate(data):
            #     whole_knowledge_path = item['context_response_path']
            #     whole_relation_ids = item['context_response_path_relation_ids']
            #
            #     context_knowledge_path = whole_knowledge_path[0]
            #     response_knowledge_path = whole_knowledge_path[1]
            #
            #     context_relation_ids = whole_relation_ids[0]
            #     response_relation_ids = whole_relation_ids[1]
            #
            #     self.source.append(context_knowledge_path)
            #     self.target.append(response_knowledge_path)
            #     self.source_rel.append(context_relation_ids)
            #     self.target_rel.append(response_relation_ids)

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id,min(sample_id+3, len(self.source)))]
        for ex in exs:
            if self.args.do_eval:
                ex = ex[0]
            # logger.info("Input: {}".format([self.tokenizer.decoder[w] for w in ex[0].tolist()]))
            logger.info("Input: {}".format(self.tokenizer.decode(ex[0].tolist())))
            logger.info("Relation id: {}".format(ex[1].tolist()))
            logger.info("Attention mask: {}".format(ex[2].tolist()))
            logger.info("Position: {}".format(ex[3].tolist()))
            # logger.info("Target: {}".format([self.tokenizer.decoder[w] for w in ex[4].tolist()]))
            logger.info("Target: {}".format(self.tokenizer.decode(ex[4].tolist())))
            logger.info("Target Relation id: {}".format(ex[5].tolist()))
            logger.info("Position: {}".format(ex[6].tolist()))
            # logger.info("Labels: {}".format([self.tokenizer.decoder[w] for w in ex[7].masked_select(ex[7]>=0).tolist()]))
            logger.info("Labels: {}".format(self.tokenizer.decode(ex[7].masked_select(ex[7]>=0).tolist())))


    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        src_rel = self.source_rel[idx]
        tgt_rel = self.target_rel[idx]

        src_input_ids = []
        src_input_rel_ids = []
        r_type = 0  # todo
        for si, s in enumerate(src):
            if (si + 1) % 2 == 0:  # relation word because the kg path of input must start from event word:
                sid = self.tokenizer.encode(' ' + self.relation_to_alpha[s.strip()])
                r_type += 1
                rid = [r_type] * len(sid)
            else:
                sid = self.tokenizer.encode(' ' + s)
                rid = [r_type] * len(sid)

            src_input_ids.extend(sid)
            src_input_rel_ids.extend(rid)

        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]
            src_input_rel_ids = src_input_rel_ids[:self.src_max_length]

        src_position_ids = list(range(0, len(src_input_ids)))
        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_input_rel_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
        assert len(src_input_ids) == len(src_input_rel_ids)


        target_input_ids = [self.bos]
        target_input_rel_ids = [r_type]
        target_position_ids = []
        labels = []
        target_bos_rel_ids = [r_type]  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias

        is_end_at_relation = []
        if src[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        if not self.do_generate:
            for ti, t in enumerate(tgt):
                if t.strip() in self.relation_list:
                    tid = self.tokenizer.encode(' ' + self.relation_to_alpha[t.strip()])
                    r_type += 1
                    rid = [r_type] * len(tid)
                else:
                    tid = self.tokenizer.encode(' ' + t)
                    rid = [r_type] * len(tid)

                target_input_ids.extend(tid)
                target_input_rel_ids.extend(rid)

            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
                target_input_rel_ids = target_input_rel_ids[:self.tgt_max_length]

            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_input_rel_ids += [self.pad]
                target_position_ids += [0]
                labels += [-1]
            assert len(target_input_ids) == len(target_input_rel_ids) == len(labels), pdb.set_trace()
        labels = [-1] * self.src_max_length + labels

        if self.args.do_eval:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_input_rel_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_input_rel_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation)
                    ), '  '.join(src), '  '.join(tgt),
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_input_rel_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_input_rel_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation)
                    )


