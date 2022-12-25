import torch
import os
import json
import logging
import csv
import itertools
from torch.utils.data import Dataset
import random
import pdb
import copy
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



class roc_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False, ending_or_complement="", exp_memory=False,):
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
                              '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact',
                              '_xWant', 'PRP', ]

        self.relation_to_alpha = {'oEffect': 'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact': 'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q',
                                  '_xWant': 'R', 'PRP': 'S', }

        # for relation embedding id
        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                               'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                               'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                               '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                               '_xWant': 17, 'PRP': 18, }

        self.exp_memory = args.exp_memory

    def load(self):
        self.source = []
        self.target = []

        self.source_kg = []
        self.target_kg = []
        self.source_kg_rel = []  # PRP xWant xWant
        self.target_kg_rel = []  # oReact oReact

        self.pred_kg = []
        self.pred_kg_rel = []

        case_ids = [2, 5, 15, 14, 28, 34, 40, 42, 97, 123, 190, 251, 272, 292, 358, 3210, 2611, 2504, 2466, 1941, 1912, 1860, 1836, 1690, 1679, 1662, 1647, 1636, 1633, 1628, 1616, 1614, 1605, 1557, 1554, 1539, 1525, 1503, 1500, 1486, 1484, 1480, 1469, 1450, 1432, 1417, 1400, 1398, 1337, 1323]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):

                if i not in case_ids:
                    continue

                # 生成故事结尾
                self.source.append(item['story'][:4])
                self.target.append([item['story'][-1]])

                whole_knowledge_path = item['context_ending_path']
                whole_relation_ids = item['context_ending_path_relation_ids']

                context_knowledge_path = whole_knowledge_path[0]
                ending_knowledge_path = whole_knowledge_path[1]

                context_relation_ids = whole_relation_ids[0]
                ending_relation_ids = whole_relation_ids[1]

                new_context_knowledge_path = []  # remove relation
                new_context_relation_ids = []
                for wi, w in enumerate(context_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_context_knowledge_path.append(w)
                        new_context_relation_ids.append(context_relation_ids[wi])
                new_ending_knowledge_path = []
                new_ending_relation_ids = []
                for wi, w in enumerate(ending_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_ending_knowledge_path.append(w)
                        new_ending_relation_ids.append(ending_relation_ids[wi])
                self.source_kg.append(new_context_knowledge_path)
                self.target_kg.append(new_ending_knowledge_path)
                self.source_kg_rel.append(new_context_relation_ids)
                self.target_kg_rel.append(new_ending_relation_ids)

                pred_kg_path = []
                pred_kg_path_rels = []
                for wi, w in enumerate(item['pred_kg_path'][0]):
                    if w in self.relation_to_id:
                        continue
                    else:
                        pred_kg_path.append(w)
                        pred_kg_path_rels.append(item['pred_kg_relation_ids'][0][wi])
                self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
                self.pred_kg_rel.append(pred_kg_path_rels)

                # self.pred_kg.append(item['pred_kg_path'][0])  # 因为有多预测的路径
                # self.pred_kg_rel.append(item['pred_kg_relation_ids'][0])

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 3, len(self.source)))]
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
            logger.info("KG Relation id: {}".format(ex[9].tolist()))
            logger.info("KG Attention mask: {}".format(ex[10].tolist()))
            logger.info("KG Position: {}".format(ex[11].tolist()))
            logger.info("KG Target: {}".format(self.tokenizer.decode(ex[12].tolist())))
            logger.info("KG Target Relation id: {}".format(ex[13].tolist()))
            logger.info("KG Position: {}".format(ex[14].tolist()))
            logger.info("KG Labels: {}".format(self.tokenizer.decode(ex[15].masked_select(ex[15] >= 0).tolist())))

            ################## PRED KG PART ########################
            logger.info("Pred KG: {}".format(self.tokenizer.decode(ex[18].tolist())))
            logger.info("Pred KG Relation id: {}".format(ex[19].tolist()))
            logger.info("Pred KG Attention mask: {}".format(ex[20].tolist()))
            logger.info("Pred KG Position: {}".format(ex[21].tolist()))


            ################## PRED KG PART ########################
            logger.info("KG Memo: {}".format(self.tokenizer.decode(ex[22].tolist())))
            logger.info("KG Memo Relation id: {}".format(ex[23].tolist()))
            logger.info("KG Memo Attention mask: {}".format(ex[24].tolist()))
            logger.info("KG Memo Position: {}".format(ex[25].tolist()))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

        src_kg = self.source_kg[idx]
        tgt_kg = self.target_kg[idx]
        src_kg_rel = self.source_kg_rel[idx]
        tgt_kg_rel = self.target_kg_rel[idx]

        pred_kg = self.pred_kg[idx]
        pred_kg_rel = self.pred_kg_rel[idx]

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
                tid = self.tokenizer.encode(' ' + t)
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

        ##################### KG PART #########################
        if isinstance(src_kg, list):
            src_kg = ' '.join(src_kg)
        src_kg_input_ids = self.tokenizer.encode('<|bos|> ' + src_kg)

        if len(src_kg_input_ids) > self.src_max_length:
            src_kg_input_ids = src_kg_input_ids[:self.src_max_length]

        exp_kg_ipt = copy.deepcopy(src_kg_input_ids)

        src_kg_position_ids = list(range(0, len(src_kg_input_ids)))
        kg_attention_mask = [1] * len(src_kg_input_ids)

        while len(src_kg_input_ids) < self.src_max_length:
            src_kg_input_ids += [self.pad]
            src_kg_position_ids += [0]
            kg_attention_mask += [0]

        target_kg_input_ids = []
        target_kg_position_ids = []
        labels_kg = []
        target_bos_rel_ids = []  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias
        target_bos_rel_ids.append(self.tokenizer.encoder[src_kg_rel[-1]])

        is_end_at_relation = []
        if src_kg[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        if not self.do_generate:  # training
            if isinstance(tgt_kg, list):
                tgt_kg = ' '.join(tgt_kg)
            target_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + tgt_kg)
            exp_kg_tgt = copy.deepcopy(target_kg_input_ids)

            target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
            labels_kg = target_kg_input_ids[1:] + [self.eos]

            while len(target_kg_input_ids) < self.tgt_max_length:
                target_kg_input_ids += [self.pad]
                target_kg_position_ids += [0]
                labels_kg += [-1]
        else:
            exp_kg_tgt = []

        labels_kg = [-1] * self.src_max_length + labels_kg

        ##################### PRED KG PART #########################
        pred_kg_rel_ids = []
        if isinstance(pred_kg, list):
            pred_kg = ' '.join(pred_kg)
        pred_kg_input_ids = self.tokenizer.encode('  <|bos|> '+ pred_kg)

        if len(pred_kg_input_ids) > self.tgt_max_length:
            pred_kg_input_ids = pred_kg_input_ids[:self.tgt_max_length]

        exp_kg_pred = copy.deepcopy(pred_kg_input_ids)
        pred_kg_position_ids = list(range(0, len(pred_kg_input_ids)))
        pred_kg_attention_mask = [1] * len(pred_kg_input_ids)

        while len(pred_kg_input_ids) < self.src_max_length:
            pred_kg_input_ids += [self.pad]
            pred_kg_position_ids += [0]
            pred_kg_attention_mask += [0]

        ##################### EXP PRED KG PART #########################
        if not self.do_generate:  # training
            if not self.exp_memory:
                exp_kg_ids = exp_kg_ipt + exp_kg_tgt  # 没有 predictions
            else:
                exp_kg_ids = exp_kg_ipt + exp_kg_tgt + exp_kg_pred  # 有 predictions
        else:
            exp_kg_ids = exp_kg_ipt + exp_kg_pred  # 测试的时候 只有 prediction


        if len(exp_kg_ids) > self.src_max_length:
            exp_kg_ids = exp_kg_ids[:self.src_max_length]

        exp_kg_type_ids = [3] * len(exp_kg_ids)
        exp_kg_position_ids = list(range(0, len(exp_kg_ids)))
        exp_kg_attention_mask = [1] * len(exp_kg_ids)

        while len(exp_kg_ids) < self.src_max_length:
            exp_kg_ids += [self.pad]
            exp_kg_type_ids += [self.pad]
            exp_kg_position_ids += [0]
            exp_kg_attention_mask += [0]

        src_kg_rel_ids = []
        target_kg_rel_ids = []
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
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    ), '  '.join(src), '  '.join(tgt), \
                   '  '.join(src_kg) if isinstance(src_kg, list) else src_kg, \
                   '  '.join(tgt_kg) if isinstance(tgt_kg, list) else tgt_kg,\
                   '  '.join(pred_kg) if isinstance(pred_kg, list) else pred_kg
        else:
            return (torch.tensor(src_input_ids),  # 0-7 for text generator
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),  # 7
                    torch.tensor(src_kg_input_ids),  # 8   8-15  for knowledge generator 但是已经fine-tune过了
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),  # 12
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),  # 15
                    torch.tensor(target_bos_rel_ids),  # 16-17 与relation ids有关
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),  # 18 测试时用的 knowledge path 只有预测的结果  18-25与prediction&knowledge memory有关
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),  # 22 训练时要用的 knowledge path 包含 ground truth
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    )


class ed_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False, ending_or_complement="", exp_memory=False,):
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
                              '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact',
                              '_xWant', 'PRP', ]
        num_added_toks = tokenizer.add_tokens(self.relation_list)
        print('add relation tokens into the tokenizer: ', num_added_toks)

        self.relation_to_alpha = {'oEffect': 'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact': 'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q',
                                  '_xWant': 'R', 'PRP': 'S',}

        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                               'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                               'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                               '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                               '_xWant': 17, 'PRP': 18,}

        self.exp_memory = args.exp_memory

    def load(self):
        self.source = []
        self.target = []

        self.source_kg = []
        self.target_kg = []
        self.source_kg_rel = []  # PRP xWant xWant
        self.target_kg_rel = []  # oReact oReact

        self.pred_kg = []
        self.pred_kg_rel = []

        case_ids = [4, 9, 45, 25, 28, 31, 32, 33, 37, 39, 40, 68, 95, 128, 140, 165, 178, 206, 214, 255, 399, 400, 416, 449, 459, 489, 493, 510, 590, 648, 764, 765, 765, 820, 885, 907, 937, 945, 962, 1000, 1010, 1094, 1119, 1123, 1147, 1188, 1235, 1354, 1374, 1441]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for id in case_ids:
                item = data[id]
                self.source.append(item['context'])
                self.target.append([item['response']])

                whole_knowledge_path = item['context_response_path']
                whole_relation_ids = item['context_response_path_relation_ids']

                context_knowledge_path = whole_knowledge_path[0]
                response_knowledge_path = whole_knowledge_path[1]

                context_relation_ids = whole_relation_ids[0]
                response_relation_ids = whole_relation_ids[1]

                # self.source_kg.append(context_knowledge_path)
                # self.target_kg.append(response_knowledge_path)
                # self.source_kg_rel.append(context_relation_ids)
                # self.target_kg_rel.append(response_relation_ids)

                new_context_knowledge_path = []  # remove relation
                new_context_relation_ids = []
                for wi, w in enumerate(context_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_context_knowledge_path.append(w)
                        new_context_relation_ids.append(context_relation_ids[wi])
                new_ending_knowledge_path = []
                new_ending_relation_ids = []
                for wi, w in enumerate(response_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_ending_knowledge_path.append(w)
                        new_ending_relation_ids.append(response_relation_ids[wi])
                self.source_kg.append(new_context_knowledge_path)
                self.target_kg.append(new_ending_knowledge_path)
                self.source_kg_rel.append(new_context_relation_ids)
                self.target_kg_rel.append(new_ending_relation_ids)

                pred_kg_path = []
                pred_kg_path_rels = []
                for wi, w in enumerate(item['beam_pred_kg_path'][0]):
                    if w in self.relation_to_id:
                        continue
                    else:
                        pred_kg_path.append(w)
                        pred_kg_path_rels.append(item['beam_pred_kg_path'][0][wi])
                self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
                self.pred_kg_rel.append(pred_kg_path_rels)

                # for wi, w in enumerate(item['pred_kg_path'][0]):
                #     if w in self.relation_to_id:  # todo: 要去掉relation吗？
                #         continue
                #     else:
                #         pred_kg_path.append(w)
                #         pred_kg_path_rels.append(item['pred_kg_relation_ids'][0][wi])
                # self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
                # self.pred_kg_rel.append(pred_kg_path_rels)


            # for i, item in enumerate(data):
            #     self.source.append(item['context'])
            #     self.target.append([item['response']])
            #
            #     whole_knowledge_path = item['context_response_path']
            #     whole_relation_ids = item['context_response_path_relation_ids']
            #
            #     context_knowledge_path = whole_knowledge_path[0]
            #     response_knowledge_path = whole_knowledge_path[1]
            #
            #     context_relation_ids = whole_relation_ids[0]
            #     response_relation_ids = whole_relation_ids[1]
            #
            #     # self.source_kg.append(context_knowledge_path)
            #     # self.target_kg.append(response_knowledge_path)
            #     # self.source_kg_rel.append(context_relation_ids)
            #     # self.target_kg_rel.append(response_relation_ids)
            #
            #     new_context_knowledge_path = []  # remove relation
            #     new_context_relation_ids = []
            #     for wi, w in enumerate(context_knowledge_path):
            #         if w.strip() in self.relation_to_id:
            #             continue
            #         else:
            #             new_context_knowledge_path.append(w)
            #             new_context_relation_ids.append(context_relation_ids[wi])
            #     new_ending_knowledge_path = []
            #     new_ending_relation_ids = []
            #     for wi, w in enumerate(response_knowledge_path):
            #         if w.strip() in self.relation_to_id:
            #             continue
            #         else:
            #             new_ending_knowledge_path.append(w)
            #             new_ending_relation_ids.append(response_relation_ids[wi])
            #     self.source_kg.append(new_context_knowledge_path)
            #     self.target_kg.append(new_ending_knowledge_path)
            #     self.source_kg_rel.append(new_context_relation_ids)
            #     self.target_kg_rel.append(new_ending_relation_ids)
            #
            #     pred_kg_path = []
            #     pred_kg_path_rels = []
            #     for wi, w in enumerate(item['pred_kg_path'][0]):
            #         if w in self.relation_to_id:  # todo: 要去掉relation吗？
            #             continue
            #         else:
            #             pred_kg_path.append(w)
            #             pred_kg_path_rels.append(item['pred_kg_relation_ids'][0][wi])
            #     self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
            #     self.pred_kg_rel.append(pred_kg_path_rels)

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 3, len(self.source)))]
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
            logger.info("KG Relation id: {}".format(ex[9].tolist()))
            logger.info("KG Attention mask: {}".format(ex[10].tolist()))
            logger.info("KG Position: {}".format(ex[11].tolist()))
            logger.info("KG Target: {}".format(self.tokenizer.decode(ex[12].tolist())))
            logger.info("KG Target Relation id: {}".format(ex[13].tolist()))
            logger.info("KG Position: {}".format(ex[14].tolist()))
            logger.info("KG Labels: {}".format(self.tokenizer.decode(ex[15].masked_select(ex[15] >= 0).tolist())))

            ################## PRED KG PART ########################
            logger.info("Pred KG: {}".format(self.tokenizer.decode(ex[18].tolist())))
            logger.info("Pred KG Relation id: {}".format(ex[19].tolist()))
            logger.info("Pred KG Attention mask: {}".format(ex[20].tolist()))
            logger.info("Pred KG Position: {}".format(ex[21].tolist()))


            ################## PRED KG PART ########################
            logger.info("KG Memo: {}".format(self.tokenizer.decode(ex[22].tolist())))
            logger.info("KG Memo Relation id: {}".format(ex[23].tolist()))
            logger.info("KG Memo Attention mask: {}".format(ex[24].tolist()))
            logger.info("KG Memo Position: {}".format(ex[25].tolist()))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

        src_kg = self.source_kg[idx]
        tgt_kg = self.target_kg[idx]
        src_kg_rel = self.source_kg_rel[idx]
        tgt_kg_rel = self.target_kg_rel[idx]

        pred_kg = self.pred_kg[idx]
        pred_kg_rel = self.pred_kg_rel[idx]

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
                tid = self.tokenizer.encode(' ' + t)
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

        ##################### KG PART #########################
        if isinstance(src_kg, list):
            src_kg = ' '.join(src_kg)
        src_kg_input_ids = self.tokenizer.encode('<|bos|> ' + src_kg)

        if len(src_kg_input_ids) > self.src_max_length:
            src_kg_input_ids = src_kg_input_ids[:self.src_max_length]

        exp_kg_ipt = copy.deepcopy(src_kg_input_ids)
        src_kg_position_ids = list(range(0, len(src_kg_input_ids)))
        kg_attention_mask = [1] * len(src_kg_input_ids)

        while len(src_kg_input_ids) < self.src_max_length:
            src_kg_input_ids += [self.pad]
            src_kg_position_ids += [0]
            kg_attention_mask += [0]

        target_kg_input_ids = []
        target_kg_position_ids = []
        labels_kg = []
        target_bos_rel_ids = []  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias
        target_bos_rel_ids.append(self.tokenizer.encoder[src_kg_rel[-1]])

        is_end_at_relation = []
        if src_kg[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        # if not self.do_generate:  # training
        #     if isinstance(tgt_kg, list):
        #         tgt_kg = ' '.join(tgt_kg)
        #     target_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + tgt_kg)
        #     exp_kg_tgt = copy.deepcopy(target_kg_input_ids)
        #
        #     target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
        #     labels_kg = target_kg_input_ids[1:] + [self.eos]
        #
        #     while len(target_kg_input_ids) < self.tgt_max_length:
        #         target_kg_input_ids += [self.pad]
        #         target_kg_position_ids += [0]
        #         labels_kg += [-1]
        # else:
        #     exp_kg_tgt = []
        #     exp_kg_tgt_rel = []

        if isinstance(tgt_kg, list):
            tgt_kg = ' '.join(tgt_kg)
        target_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + tgt_kg)
        exp_kg_tgt = copy.deepcopy(target_kg_input_ids)

        target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
        labels_kg = target_kg_input_ids[1:] + [self.eos]

        while len(target_kg_input_ids) < self.tgt_max_length:
            target_kg_input_ids += [self.pad]
            target_kg_position_ids += [0]
            labels_kg += [-1]


        labels_kg = [-1] * self.src_max_length + labels_kg

        ##################### PRED KG PART #########################
        pred_kg_rel_ids = []
        if isinstance(pred_kg, list):
            pred_kg = ' '.join(pred_kg)
        pred_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + pred_kg)

        if len(pred_kg_input_ids) > self.tgt_max_length:
            pred_kg_input_ids = pred_kg_input_ids[:self.tgt_max_length]

        exp_kg_pred = copy.deepcopy(pred_kg_input_ids)
        pred_kg_position_ids = list(range(0, len(pred_kg_input_ids)))
        pred_kg_attention_mask = [1] * len(pred_kg_input_ids)

        while len(pred_kg_input_ids) < self.src_max_length:
            pred_kg_input_ids += [self.pad]
            pred_kg_position_ids += [0]
            pred_kg_attention_mask += [0]

        ##################### EXP PRED KG PART #########################
        if not self.do_generate:  # training
            # exp_kg_ids = exp_kg_tgt
            if not self.exp_memory:
                exp_kg_ids = exp_kg_ipt + exp_kg_tgt  # 没有 predictions
            else:
                exp_kg_ids = exp_kg_ipt + exp_kg_tgt + exp_kg_pred  # 有 predictions
        else:
            # exp_kg_ids = exp_kg_tgt  # 测试的时候 只有 prediction
            exp_kg_ids = exp_kg_ipt + exp_kg_pred  # 测试的时候 只有 prediction

        if len(exp_kg_ids) > self.src_max_length:
            exp_kg_ids = exp_kg_ids[:self.src_max_length]
        exp_kg_ids_print = copy.deepcopy(exp_kg_ids)

        exp_kg_type_ids = [3] * len(exp_kg_ids)
        exp_kg_position_ids = list(range(0, len(exp_kg_ids)))
        exp_kg_attention_mask = [1] * len(exp_kg_ids)

        while len(exp_kg_ids) < self.src_max_length:
            exp_kg_ids += [self.pad]
            exp_kg_type_ids += [self.pad]
            exp_kg_position_ids += [0]
            exp_kg_attention_mask += [0]

        src_kg_rel_ids = []
        target_kg_rel_ids = []

        # if self.args.do_eval:
        if self.do_generate:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(src_kg_input_ids),
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    ), '  '.join(src), '  '.join(tgt),\
                   '  '.join(src_kg) if isinstance(src_kg, list) else src_kg, \
                   '  '.join(tgt_kg) if isinstance(tgt_kg, list) else tgt_kg,\
                   '  '.join(pred_kg) if isinstance(pred_kg, list) else pred_kg, \
                   self.tokenizer.decode(exp_kg_ids_print),


        else:
            return (torch.tensor(src_input_ids),  # 0-7 for text generator
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),  # 7
                    torch.tensor(src_kg_input_ids),  # 8   8-15  for knowledge generator 但是已经fine-tune过了
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),  # 12
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),  # 15
                    torch.tensor(target_bos_rel_ids),  # 16-17 与relation ids有关
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),  # 18 测试时用的 knowledge path 只有预测的结果  18-25与prediction&knowledge memory有关
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),  # 22 训练时要用的 knowledge path 包含 ground truth
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    )



class ed_retrieve_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False, ending_or_complement="", exp_memory=False,):
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
                              '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact',
                              '_xWant', 'PRP', ]
        num_added_toks = tokenizer.add_tokens(self.relation_list)
        print('add relation tokens into the tokenizer: ', num_added_toks)

        self.relation_to_alpha = {'oEffect': 'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact': 'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q',
                                  '_xWant': 'R', 'PRP': 'S',}

        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                               'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                               'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                               '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                               '_xWant': 17, 'PRP': 18,}

        self.exp_memory = args.exp_memory

    def load(self):
        self.source = []
        self.target = []

        self.source_kg = []
        self.target_kg = []
        self.source_kg_rel = []  # PRP xWant xWant
        self.target_kg_rel = []  # oReact oReact

        self.pred_kg = []
        self.pred_kg_rel = []

        case_ids = [4, 9, 45, 25, 28, 31, 32, 33, 37, 39, 40, 68, 95, 128, 140, 165, 178, 206, 214, 255, 399, 400, 416, 449, 459, 489, 493, 510, 590, 648, 764, 765, 765, 820, 885, 907, 937, 945, 962, 1000, 1010, 1094, 1119, 1123, 1147, 1188, 1235, 1354, 1374, 1441]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for id in case_ids:
                item = data[id]
                self.source.append(item['context'])
                self.target.append([item['response']])

                whole_knowledge_path = item['context_response_path']
                whole_relation_ids = item['context_response_path_relation_ids']

                context_knowledge_path = whole_knowledge_path[0]
                response_knowledge_path = whole_knowledge_path[1]

                context_relation_ids = whole_relation_ids[0]
                response_relation_ids = whole_relation_ids[1]

                # self.source_kg.append(context_knowledge_path)
                # self.target_kg.append(response_knowledge_path)
                # self.source_kg_rel.append(context_relation_ids)
                # self.target_kg_rel.append(response_relation_ids)

                new_context_knowledge_path = []  # remove relation
                new_context_relation_ids = []
                for wi, w in enumerate(context_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_context_knowledge_path.append(w)
                        new_context_relation_ids.append(context_relation_ids[wi])
                new_ending_knowledge_path = []
                new_ending_relation_ids = []
                for wi, w in enumerate(response_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_ending_knowledge_path.append(w)
                        new_ending_relation_ids.append(response_relation_ids[wi])
                self.source_kg.append(new_context_knowledge_path)
                self.target_kg.append(new_ending_knowledge_path)
                self.source_kg_rel.append(new_context_relation_ids)
                self.target_kg_rel.append(new_ending_relation_ids)

                pred_kg_path = item['retrieve_kg'][-1]
                pred_kg_path_rels = item['retrieve_kg'][1]
                self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
                self.pred_kg_rel.append(pred_kg_path_rels)
            # for i, item in enumerate(data):
            #     self.source.append(item['context'])
            #     self.target.append([item['response']])
            #
            #     whole_knowledge_path = item['context_response_path']
            #     whole_relation_ids = item['context_response_path_relation_ids']
            #
            #     context_knowledge_path = whole_knowledge_path[0]
            #     response_knowledge_path = whole_knowledge_path[1]
            #
            #     context_relation_ids = whole_relation_ids[0]
            #     response_relation_ids = whole_relation_ids[1]
            #
            #     # self.source_kg.append(context_knowledge_path)
            #     # self.target_kg.append(response_knowledge_path)
            #     # self.source_kg_rel.append(context_relation_ids)
            #     # self.target_kg_rel.append(response_relation_ids)
            #
            #     new_context_knowledge_path = []  # remove relation
            #     new_context_relation_ids = []
            #     for wi, w in enumerate(context_knowledge_path):
            #         if w.strip() in self.relation_to_id:
            #             continue
            #         else:
            #             new_context_knowledge_path.append(w)
            #             new_context_relation_ids.append(context_relation_ids[wi])
            #     new_ending_knowledge_path = []
            #     new_ending_relation_ids = []
            #     for wi, w in enumerate(response_knowledge_path):
            #         if w.strip() in self.relation_to_id:
            #             continue
            #         else:
            #             new_ending_knowledge_path.append(w)
            #             new_ending_relation_ids.append(response_relation_ids[wi])
            #     self.source_kg.append(new_context_knowledge_path)
            #     self.target_kg.append(new_ending_knowledge_path)
            #     self.source_kg_rel.append(new_context_relation_ids)
            #     self.target_kg_rel.append(new_ending_relation_ids)
            #
            #     pred_kg_path = item['retrieve_kg'][-1]
            #     pred_kg_path_rels = item['retrieve_kg'][1]
            #     self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
            #     self.pred_kg_rel.append(pred_kg_path_rels)

    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 3, len(self.source)))]
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
            logger.info("KG Relation id: {}".format(ex[9].tolist()))
            logger.info("KG Attention mask: {}".format(ex[10].tolist()))
            logger.info("KG Position: {}".format(ex[11].tolist()))
            logger.info("KG Target: {}".format(self.tokenizer.decode(ex[12].tolist())))
            logger.info("KG Target Relation id: {}".format(ex[13].tolist()))
            logger.info("KG Position: {}".format(ex[14].tolist()))
            logger.info("KG Labels: {}".format(self.tokenizer.decode(ex[15].masked_select(ex[15] >= 0).tolist())))

            ################## PRED KG PART ########################
            logger.info("Pred KG: {}".format(self.tokenizer.decode(ex[18].tolist())))
            logger.info("Pred KG Relation id: {}".format(ex[19].tolist()))
            logger.info("Pred KG Attention mask: {}".format(ex[20].tolist()))
            logger.info("Pred KG Position: {}".format(ex[21].tolist()))


            ################## PRED KG PART ########################
            logger.info("KG Memo: {}".format(self.tokenizer.decode(ex[22].tolist())))
            logger.info("KG Memo Relation id: {}".format(ex[23].tolist()))
            logger.info("KG Memo Attention mask: {}".format(ex[24].tolist()))
            logger.info("KG Memo Position: {}".format(ex[25].tolist()))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

        src_kg = self.source_kg[idx]
        tgt_kg = self.target_kg[idx]
        src_kg_rel = self.source_kg_rel[idx]
        tgt_kg_rel = self.target_kg_rel[idx]

        pred_kg = self.pred_kg[idx]
        pred_kg_rel = self.pred_kg_rel[idx]

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
                tid = self.tokenizer.encode(' ' + t)
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

        ##################### KG PART #########################
        if isinstance(src_kg, list):
            src_kg = ' '.join(src_kg)
        src_kg_input_ids = self.tokenizer.encode('<|bos|> ' + src_kg)

        if len(src_kg_input_ids) > self.src_max_length:
            src_kg_input_ids = src_kg_input_ids[:self.src_max_length]

        exp_kg_ipt = copy.deepcopy(src_kg_input_ids)
        src_kg_position_ids = list(range(0, len(src_kg_input_ids)))
        kg_attention_mask = [1] * len(src_kg_input_ids)

        while len(src_kg_input_ids) < self.src_max_length:
            src_kg_input_ids += [self.pad]
            src_kg_position_ids += [0]
            kg_attention_mask += [0]

        target_kg_input_ids = []
        target_kg_position_ids = []
        labels_kg = []
        target_bos_rel_ids = []  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias
        target_bos_rel_ids.append(self.tokenizer.encoder[src_kg_rel[-1]])

        is_end_at_relation = []
        if src_kg[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        # if not self.do_generate:  # training
        #     if isinstance(tgt_kg, list):
        #         tgt_kg = ' '.join(tgt_kg)
        #     target_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + tgt_kg)
        #     exp_kg_tgt = copy.deepcopy(target_kg_input_ids)
        #
        #     target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
        #     labels_kg = target_kg_input_ids[1:] + [self.eos]
        #
        #     while len(target_kg_input_ids) < self.tgt_max_length:
        #         target_kg_input_ids += [self.pad]
        #         target_kg_position_ids += [0]
        #         labels_kg += [-1]
        # else:
        #     exp_kg_tgt = []
        #     exp_kg_tgt_rel = []

        if isinstance(tgt_kg, list):
            tgt_kg = ' '.join(tgt_kg)
        target_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + tgt_kg)
        exp_kg_tgt = copy.deepcopy(target_kg_input_ids)

        target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
        labels_kg = target_kg_input_ids[1:] + [self.eos]

        while len(target_kg_input_ids) < self.tgt_max_length:
            target_kg_input_ids += [self.pad]
            target_kg_position_ids += [0]
            labels_kg += [-1]


        labels_kg = [-1] * self.src_max_length + labels_kg

        ##################### PRED KG PART #########################
        pred_kg_rel_ids = []
        if isinstance(pred_kg, list):
            pred_kg = ' '.join(pred_kg)
        pred_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + pred_kg)

        if len(pred_kg_input_ids) > self.tgt_max_length:
            pred_kg_input_ids = pred_kg_input_ids[:self.tgt_max_length]

        exp_kg_pred = copy.deepcopy(pred_kg_input_ids)
        pred_kg_position_ids = list(range(0, len(pred_kg_input_ids)))
        pred_kg_attention_mask = [1] * len(pred_kg_input_ids)

        while len(pred_kg_input_ids) < self.src_max_length:
            pred_kg_input_ids += [self.pad]
            pred_kg_position_ids += [0]
            pred_kg_attention_mask += [0]

        ##################### EXP PRED KG PART #########################

        exp_kg_ids = exp_kg_pred  # 测试的时候 只有 prediction

        if len(exp_kg_ids) > self.src_max_length:
            exp_kg_ids = exp_kg_ids[:self.src_max_length]
        exp_kg_ids_print = copy.deepcopy(exp_kg_ids)

        exp_kg_type_ids = [3] * len(exp_kg_ids)
        exp_kg_position_ids = list(range(0, len(exp_kg_ids)))
        exp_kg_attention_mask = [1] * len(exp_kg_ids)

        while len(exp_kg_ids) < self.src_max_length:
            exp_kg_ids += [self.pad]
            exp_kg_type_ids += [self.pad]
            exp_kg_position_ids += [0]
            exp_kg_attention_mask += [0]

        src_kg_rel_ids = []
        target_kg_rel_ids = []

        # if self.args.do_eval:
        if self.do_generate:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(src_kg_input_ids),
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    ), '  '.join(src), '  '.join(tgt),\
                   '  '.join(src_kg) if isinstance(src_kg, list) else src_kg, \
                   '  '.join(tgt_kg) if isinstance(tgt_kg, list) else tgt_kg,\
                   '  '.join(pred_kg) if isinstance(pred_kg, list) else pred_kg, \
                   self.tokenizer.decode(exp_kg_ids_print),
        else:
            return (torch.tensor(src_input_ids),  # 0-7 for text generator
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),  # 7
                    torch.tensor(src_kg_input_ids),  # 8   8-15  for knowledge generator 但是已经fine-tune过了
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),  # 12
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),  # 15
                    torch.tensor(target_bos_rel_ids),  # 16-17 与relation ids有关
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),  # 18 测试时用的 knowledge path 只有预测的结果  18-25与prediction&knowledge memory有关
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),  # 22 训练时要用的 knowledge path 包含 ground truth
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    )


class roc_retrieve_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False, ending_or_complement="", exp_memory=False,):
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
                              '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact',
                              '_xWant', 'PRP', ]

        self.relation_to_alpha = {'oEffect': 'A', 'oReact': 'B', 'oWant': 'C', 'xAttr': 'D',
                                  'xEffect': 'E', 'xIntent': 'F', 'xNeed': 'G', 'xReact': 'H',
                                  'xWant': 'I', '_oEffect': 'J', '_oReact': 'K', '_oWant': 'L',
                                  '_xAttr': 'M', '_xEffect': 'N', '_xIntent': 'O', '_xNeed': 'P', '_xReact': 'Q',
                                  '_xWant': 'R', 'PRP': 'S', }

        # for relation embedding id
        self.relation_to_id = {'oEffect': 0, 'oReact': 1, 'oWant': 2, 'xAttr': 3,
                               'xEffect': 4, 'xIntent': 5, 'xNeed': 6, 'xReact': 7,
                               'xWant': 8, '_oEffect': 9, '_oReact': 10, '_oWant': 11,
                               '_xAttr': 12, '_xEffect': 13, '_xIntent': 14, '_xNeed': 15, '_xReact': 16,
                               '_xWant': 17, 'PRP': 18, }

        self.exp_memory = args.exp_memory

    def load(self):
        self.source = []
        self.target = []

        self.source_kg = []
        self.target_kg = []
        self.source_kg_rel = []  # PRP xWant xWant
        self.target_kg_rel = []  # oReact oReact

        self.pred_kg = []
        self.pred_kg_rel = []
        case_ids = [2, 5, 15, 14, 28, 34, 40, 42, 97, 123, 190, 251, 272, 292, 358, 3210, 2611, 2504, 2466, 1941, 1912, 1860, 1836, 1690, 1679, 1662, 1647, 1636, 1633, 1628, 1616, 1614, 1605, 1557, 1554, 1539, 1525, 1503, 1500, 1486, 1484, 1480, 1469, 1450, 1432, 1417, 1400, 1398, 1337, 1323]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for id in case_ids:
                item = data[id]
                # 生成故事结尾
                self.source.append(item['story'][:4])
                self.target.append([item['story'][-1]])

                whole_knowledge_path = item['context_ending_path']
                whole_relation_ids = item['context_ending_path_relation_ids']

                context_knowledge_path = whole_knowledge_path[0]
                ending_knowledge_path = whole_knowledge_path[1]

                context_relation_ids = whole_relation_ids[0]
                ending_relation_ids = whole_relation_ids[1]

                new_context_knowledge_path = []  # remove relation
                new_context_relation_ids = []
                for wi, w in enumerate(context_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_context_knowledge_path.append(w)
                        new_context_relation_ids.append(context_relation_ids[wi])
                new_ending_knowledge_path = []
                new_ending_relation_ids = []
                for wi, w in enumerate(ending_knowledge_path):
                    if w.strip() in self.relation_to_id:
                        continue
                    else:
                        new_ending_knowledge_path.append(w)
                        new_ending_relation_ids.append(ending_relation_ids[wi])
                self.source_kg.append(new_context_knowledge_path)
                self.target_kg.append(new_ending_knowledge_path)
                self.source_kg_rel.append(new_context_relation_ids)
                self.target_kg_rel.append(new_ending_relation_ids)

                pred_kg_path = item['retrieve_kg'][-1]
                pred_kg_path_rels = item['retrieve_kg'][1]
                self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
                self.pred_kg_rel.append(pred_kg_path_rels)

            # for i, item in enumerate(data):
            #     # 生成故事结尾
            #     self.source.append(item['story'][:4])
            #     self.target.append([item['story'][-1]])
            #
            #     whole_knowledge_path = item['context_ending_path']
            #     whole_relation_ids = item['context_ending_path_relation_ids']
            #
            #     context_knowledge_path = whole_knowledge_path[0]
            #     ending_knowledge_path = whole_knowledge_path[1]
            #
            #     context_relation_ids = whole_relation_ids[0]
            #     ending_relation_ids = whole_relation_ids[1]
            #
            #     new_context_knowledge_path = []  # remove relation
            #     new_context_relation_ids = []
            #     for wi, w in enumerate(context_knowledge_path):
            #         if w.strip() in self.relation_to_id:
            #             continue
            #         else:
            #             new_context_knowledge_path.append(w)
            #             new_context_relation_ids.append(context_relation_ids[wi])
            #     new_ending_knowledge_path = []
            #     new_ending_relation_ids = []
            #     for wi, w in enumerate(ending_knowledge_path):
            #         if w.strip() in self.relation_to_id:
            #             continue
            #         else:
            #             new_ending_knowledge_path.append(w)
            #             new_ending_relation_ids.append(ending_relation_ids[wi])
            #     self.source_kg.append(new_context_knowledge_path)
            #     self.target_kg.append(new_ending_knowledge_path)
            #     self.source_kg_rel.append(new_context_relation_ids)
            #     self.target_kg_rel.append(new_ending_relation_ids)
            #
            #     pred_kg_path = item['retrieve_kg'][-1]
            #     pred_kg_path_rels = item['retrieve_kg'][1]
            #     self.pred_kg.append(pred_kg_path)  # 因为有多预测的路径
            #     self.pred_kg_rel.append(pred_kg_path_rels)


    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 10000)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 3, len(self.source)))]
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
            logger.info("KG Relation id: {}".format(ex[9].tolist()))
            logger.info("KG Attention mask: {}".format(ex[10].tolist()))
            logger.info("KG Position: {}".format(ex[11].tolist()))
            logger.info("KG Target: {}".format(self.tokenizer.decode(ex[12].tolist())))
            logger.info("KG Target Relation id: {}".format(ex[13].tolist()))
            logger.info("KG Position: {}".format(ex[14].tolist()))
            logger.info("KG Labels: {}".format(self.tokenizer.decode(ex[15].masked_select(ex[15] >= 0).tolist())))

            ################## PRED KG PART ########################
            logger.info("Pred KG: {}".format(self.tokenizer.decode(ex[18].tolist())))
            logger.info("Pred KG Relation id: {}".format(ex[19].tolist()))
            logger.info("Pred KG Attention mask: {}".format(ex[20].tolist()))
            logger.info("Pred KG Position: {}".format(ex[21].tolist()))


            ################## PRED KG PART ########################
            logger.info("KG Memo: {}".format(self.tokenizer.decode(ex[22].tolist())))
            logger.info("KG Memo Relation id: {}".format(ex[23].tolist()))
            logger.info("KG Memo Attention mask: {}".format(ex[24].tolist()))
            logger.info("KG Memo Position: {}".format(ex[25].tolist()))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

        src_kg = self.source_kg[idx]
        tgt_kg = self.target_kg[idx]
        src_kg_rel = self.source_kg_rel[idx]
        tgt_kg_rel = self.target_kg_rel[idx]

        pred_kg = self.pred_kg[idx]
        pred_kg_rel = self.pred_kg_rel[idx]

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
                tid = self.tokenizer.encode(' ' + t)
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

        ##################### KG PART #########################
        if isinstance(src_kg, list):
            src_kg = ' '.join(src_kg)
        src_kg_input_ids = self.tokenizer.encode('<|bos|> ' + src_kg)

        if len(src_kg_input_ids) > self.src_max_length:
            src_kg_input_ids = src_kg_input_ids[:self.src_max_length]

        exp_kg_ipt = copy.deepcopy(src_kg_input_ids)

        src_kg_position_ids = list(range(0, len(src_kg_input_ids)))
        kg_attention_mask = [1] * len(src_kg_input_ids)

        while len(src_kg_input_ids) < self.src_max_length:
            src_kg_input_ids += [self.pad]
            src_kg_position_ids += [0]
            kg_attention_mask += [0]

        target_kg_input_ids = []
        target_kg_position_ids = []
        labels_kg = []
        target_bos_rel_ids = []  # bos token的relation id与source的最后一个token的relation id保持一致，目的是减少training和inference的bias
        target_bos_rel_ids.append(self.tokenizer.encoder[src_kg_rel[-1]])

        is_end_at_relation = []
        if src_kg[-1].strip() in self.relation_list:
            is_end_at_relation.append(1)
        else:
            is_end_at_relation.append(0)

        if not self.do_generate:  # training
            if isinstance(tgt_kg, list):
                tgt_kg = ' '.join(tgt_kg)
            target_kg_input_ids = self.tokenizer.encode(' <|bos|> ' + tgt_kg)
            exp_kg_tgt = copy.deepcopy(target_kg_input_ids)

            target_kg_position_ids = list(range(0, len(target_kg_input_ids)))
            labels_kg = target_kg_input_ids[1:] + [self.eos]

            while len(target_kg_input_ids) < self.tgt_max_length:
                target_kg_input_ids += [self.pad]
                target_kg_position_ids += [0]
                labels_kg += [-1]
        else:
            exp_kg_tgt = []

        labels_kg = [-1] * self.src_max_length + labels_kg

        ##################### PRED KG PART #########################
        pred_kg_rel_ids = []
        if isinstance(pred_kg, list):
            pred_kg = ' '.join(pred_kg)
        pred_kg_input_ids = self.tokenizer.encode('  <|bos|> '+ pred_kg)

        if len(pred_kg_input_ids) > self.tgt_max_length:
            pred_kg_input_ids = pred_kg_input_ids[:self.tgt_max_length]

        exp_kg_pred = copy.deepcopy(pred_kg_input_ids)
        pred_kg_position_ids = list(range(0, len(pred_kg_input_ids)))
        pred_kg_attention_mask = [1] * len(pred_kg_input_ids)

        while len(pred_kg_input_ids) < self.src_max_length:
            pred_kg_input_ids += [self.pad]
            pred_kg_position_ids += [0]
            pred_kg_attention_mask += [0]

        ##################### EXP PRED KG PART #########################

        exp_kg_ids = exp_kg_pred  # 测试的时候 只有 prediction

        if len(exp_kg_ids) > self.src_max_length:
            exp_kg_ids = exp_kg_ids[:self.src_max_length]
        exp_kg_ids_print = copy.deepcopy(exp_kg_ids)

        exp_kg_type_ids = [3] * len(exp_kg_ids)
        exp_kg_position_ids = list(range(0, len(exp_kg_ids)))
        exp_kg_attention_mask = [1] * len(exp_kg_ids)

        while len(exp_kg_ids) < self.src_max_length:
            exp_kg_ids += [self.pad]
            exp_kg_type_ids += [self.pad]
            exp_kg_position_ids += [0]
            exp_kg_attention_mask += [0]

        src_kg_rel_ids = []
        target_kg_rel_ids = []
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
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),
                    torch.tensor(target_bos_rel_ids),
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    ), '  '.join(src), '  '.join(tgt), \
                   '  '.join(src_kg) if isinstance(src_kg, list) else src_kg, \
                   '  '.join(tgt_kg) if isinstance(tgt_kg, list) else tgt_kg,\
                   '  '.join(pred_kg) if isinstance(pred_kg, list) else pred_kg,\
                   self.tokenizer.decode(exp_kg_ids_print),
        else:
            return (torch.tensor(src_input_ids),  # 0-7 for text generator
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),  # 7
                    torch.tensor(src_kg_input_ids),  # 8   8-15  for knowledge generator 但是已经fine-tune过了
                    torch.tensor(src_kg_rel_ids),
                    torch.tensor(kg_attention_mask),
                    torch.tensor(src_kg_position_ids),
                    torch.tensor(target_kg_input_ids),  # 12
                    torch.tensor(target_kg_rel_ids),
                    torch.tensor(target_kg_position_ids),
                    torch.tensor(labels_kg),  # 15
                    torch.tensor(target_bos_rel_ids),  # 16-17 与relation ids有关
                    torch.tensor(is_end_at_relation),
                    torch.tensor(pred_kg_input_ids),  # 18 测试时用的 knowledge path 只有预测的结果  18-25与prediction&knowledge memory有关
                    torch.tensor(pred_kg_rel_ids),
                    torch.tensor(pred_kg_position_ids),
                    torch.tensor(pred_kg_attention_mask),
                    torch.tensor(exp_kg_ids),  # 22 训练时要用的 knowledge path 包含 ground truth
                    torch.tensor(exp_kg_type_ids),  # change to simple type id
                    torch.tensor(exp_kg_position_ids),
                    torch.tensor(exp_kg_attention_mask),
                    )
