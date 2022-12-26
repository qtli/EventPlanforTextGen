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
    


class DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100, do_generate=False, ending_or_complement='ending'):
        self.do_generate = do_generate 
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.data_path = data_path
        self.relation_path = relation_path  # for downstream task.
        self.ending_or_complement = ending_or_complement

    def load(self):
        self.source = []
        self.target = []
        self.source_kg = []
        self.target_kg = []
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if self.ending_or_complement == 'ending':
                    self.source.append(item['story'][:-1])
                    self.target.append(item['story'][-1])

                    self.source_kg.append(' '.join(item['story_ending_gen'][0]))  # string
                    self.target_kg.append(' '.join(item['story_ending_gen'][1]))  # string
                else:
                    self.source.append([item['story'][0]])
                    self.target.append(item['story'][1:])

                    self.source_kg.append(' '.join(item['story_complement'][0]))  # string
                    self.target_kg.append(' '.join(item['story_complement'][1]))  # string

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-"*50 + "Features" + "-"*50)
        exs = [self.__getitem__(i) for i in range(0,min(3, len(self.source)))]
        for ex in exs:
            ex = ex[0]
            logger.info("Input: {}".format(self.tokenizer.decode(ex[0].tolist())))
            logger.info("Attention mask: {}".format(ex[1].tolist()))
            logger.info("Position: {}".format(ex[2].tolist()))

            logger.info("KGInput: {}".format(self.tokenizer.decode(ex[6].tolist())))  # todo
            logger.info("KGAttention mask: {}".format(ex[7].tolist()))
            logger.info("KGPosition: {}".format(ex[8].tolist()))

            logger.info("TopKGCandidate: {}".format(ex[9].tolist()))

            logger.info("Target: {}".format(self.tokenizer.decode(ex[3].tolist())))
            logger.info("Position: {}".format(ex[4].tolist()))
            logger.info("Labels: {}".format(self.tokenizer.decode(ex[5].masked_select(ex[5]>=0).tolist())))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        src_kg = self.source_kg[idx]
        tgt_kg = self.target_kg[idx]

        src_input_ids = []
        for s in src:
            src_input_ids.extend(self.tokenizer.encode(' ' + s))
            src_input_ids.append(self.eos)
        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]

        src_position_ids = list(range(0, len(src_input_ids)))
        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
        
        target_input_ids = []
        target_position_ids = []
        labels = []
        if not self.do_generate:
            target_input_ids = [self.bos] + self.tokenizer.encode(tgt+' ')
            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]

            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_position_ids += [0] 
                labels += [-1]
        labels = [-1] * self.src_max_length + labels


        # ==========================================================================================
        src_kg_input_ids = self.tokenizer.encode(src_kg+' ')  # todo why add " " ?
        if len(src_kg_input_ids) > self.src_max_length:
            src_kg_input_ids = src_kg_input_ids[:self.src_max_length]

        src_kg_position_ids = list(range(0, len(src_kg_input_ids)))
        kg_attention_mask = [1] * len(src_kg_input_ids)

        while len(src_kg_input_ids) < self.src_max_length:
            src_kg_input_ids += [self.pad]
            src_kg_position_ids += [0]
            kg_attention_mask += [0]

        target_kg_input_ids = []
        target_kg_position_ids = []
        labels_kg = []
        if not self.do_generate:
            target_kg_input_ids = [self.bos] + self.tokenizer.encode(tgt_kg+' ')
            if len(target_kg_input_ids) > self.tgt_max_length:
                target_kg_input_ids = target_kg_input_ids[:self.tgt_max_length]
            target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
            labels_kg = target_kg_input_ids[1:] + [self.eos]
            while len(target_kg_input_ids) < self.tgt_max_length:
                target_kg_input_ids += [self.pad]
                target_kg_position_ids += [0]
                labels_kg += [-1]
        labels_kg = [-1] * self.src_max_length + labels_kg

        return (torch.tensor(src_input_ids),
                torch.tensor(attention_mask),
                torch.tensor(src_position_ids),
                torch.tensor(target_input_ids),
                torch.tensor(target_position_ids),
                torch.tensor(labels),
                torch.tensor(src_kg_input_ids),
                torch.tensor(kg_attention_mask),
                torch.tensor(src_kg_position_ids),
                torch.tensor(target_kg_input_ids),
                torch.tensor(target_kg_position_ids),
                torch.tensor(labels_kg),
                ), ' '.join(src), tgt, src_kg, tgt_kg


class ed_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False, ending_or_complement=""):
        self.do_generate = do_generate
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.data_path = data_path

        self.relation_path = relation_path  # for downstream task.

        self.relation_list = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant', 'PRP',]

        self.relation_to_alpha = {'oEffect':'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact':'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q', '_xWant': 'R', 'PRP': 'S',}

        # for relation embedding id
        # {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18}
        # {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S'}
        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                                  'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                                  'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                                  '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                                  '_xWant': 17, 'PRP': 18,}

    def load(self):
        rels = json.load(open(self.relation_path, 'r'))

        self.source = []
        self.target = []

        self.source_kg = []

        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                self.source.append(item['context'])
                self.target.append([item['response']])

                whole_knowledge_path = item['retrieve_kg']
                assert len(whole_knowledge_path)==3
                self.source_kg.append(whole_knowledge_path[2:])

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
            logger.info("Input type: {}".format(ex[1].tolist()))
            logger.info("Attention mask: {}".format(ex[2].tolist()))
            logger.info("Position: {}".format(ex[3].tolist()))

            logger.info("Target: {}".format(self.tokenizer.decode(ex[4].tolist())))
            logger.info("Target type: {}".format(ex[5].tolist()))
            logger.info("Position: {}".format(ex[6].tolist()))
            logger.info("Labels: {}".format(self.tokenizer.decode(ex[7].masked_select(ex[7] >= 0).tolist())))

            ################## KG PART ########################
            logger.info("KGInput: {}".format(self.tokenizer.decode(ex[8].tolist())))
            logger.info("KG Attention mask: {}".format(ex[9].tolist()))
            logger.info("KG Position: {}".format(ex[10].tolist()))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        src_kg = self.source_kg[idx]

        src_input_ids = []
        src_type_ids = []  # 0 / 1 交替出现
        for si, s in enumerate(src):
            s = ' ' + s
            s += ' <|endoftext|>'  # 每个utterance后面跟一个结束符
            s_ids = self.tokenizer.encode(s)
            src_input_ids.extend(s_ids)
            spk = 0 if si % 2 == 0 else 1
            src_type_ids += [spk for _ in range(len(s_ids))]

        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]
            src_type_ids = src_type_ids[:self.src_max_length]

        src_position_ids = list(range(0, len(src_input_ids)))
        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
            src_type_ids += [self.pad]

        target_input_ids = []
        target_type_ids = []
        target_position_ids = []
        labels = []
        if not self.do_generate:
            assert len(tgt) == 1
            for ti, t in enumerate(tgt):
                if ti == 0:
                    target_input_ids.extend([self.bos])
                    target_type_ids.extend([1])
                tid = self.tokenizer.encode(' '+t)
                target_input_ids.extend(tid)
                lsn = 1 if ti % 2 == 0 else 0
                target_type_ids += [lsn for _ in range(len(tid))]
            assert len(target_type_ids) == len(target_type_ids)

            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
                target_type_ids = target_type_ids[:self.tgt_max_length]
            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_position_ids += [0]
                labels += [-1]
                target_type_ids += [self.pad]
        labels = [-1] * self.src_max_length + labels

        ######################### KG PART #########################
        src_kg_input_ids = []
        for si, sk in enumerate(src_kg):
            sid = self.tokenizer.encode(' ' + sk)
            src_kg_input_ids.extend(sid)  # todo why add " " ??

        if len(src_kg_input_ids) > self.src_max_length:
            src_kg_input_ids = src_kg_input_ids[:self.src_max_length]

        src_kg_position_ids = list(range(0, len(src_kg_input_ids)))
        kg_attention_mask = [1] * len(src_kg_input_ids)

        while len(src_kg_input_ids) < self.src_max_length:
            src_kg_input_ids += [self.pad]
            src_kg_position_ids += [0]
            kg_attention_mask += [0]

        if self.args.do_eval:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(src_kg_input_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    ), '  '.join(src), '  '.join(tgt), '  '.join(src_kg),
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),  # 7
                    torch.tensor(src_kg_input_ids),  # 8
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    )





