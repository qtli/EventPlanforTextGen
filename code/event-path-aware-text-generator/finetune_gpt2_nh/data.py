import pdb

import torch
import json
import logging
from torch.utils.data import Dataset
import random

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


class kg_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=80, tgt_max_length=30,
                 do_generate=False):
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

        self.relation_list = ['[name]', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                              '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact',
                              '_xWant', 'PRP',]
        num_added_toks = tokenizer.add_tokens(self.relation_list)




    def load(self):
        self.source = []
        self.target = []
        with open(self.data_path, 'r') as f:
            # data = json.load(f)
            for line in f.readlines():
                entities = line.split('\t')
                self.source.append(' '.join(entities[:2]))
                self.target.append([entities[2]])

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
            logger.info("Attention mask: {}".format(ex[1].tolist()))
            logger.info("Position: {}".format(ex[2].tolist()))
            logger.info("Target: {}".format(self.tokenizer.decode(ex[3].tolist())))
            logger.info("Position: {}".format(ex[4].tolist()))
            logger.info("Labels: {}".format(self.tokenizer.decode(ex[5].masked_select(ex[5] >= 0).tolist())))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        src_input_ids = []
        # src_type_ids = []  # 0 / 1 交替出现
        for si, s in enumerate(src):
            s = ' ' + s
            s += ' <|endoftext|>'  # 每个utterance后面跟一个结束符
            s_ids = self.tokenizer.encode(s)
            src_input_ids.extend(s_ids)
            # spk = 0 if si % 2 == 0 else 1
            # src_type_ids += [spk for _ in range(len(s_ids))]

        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]
            # src_type_ids = src_type_ids[:self.src_max_length]

        src_position_ids = list(range(0, len(src_input_ids)))
        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
            # src_type_ids += [self.pad]

        target_input_ids = []
        # target_type_ids = []
        target_position_ids = []
        labels = []

        if not self.do_generate:
            assert len(tgt) == 1
            for ti, t in enumerate(tgt):
                if ti == 0:
                    target_input_ids.extend([self.bos])
                    # target_type_ids.extend([1])
                tid = self.tokenizer.encode(' '+t)
                target_input_ids.extend(tid)
                # lsn = 1 if ti % 2 == 0 else 0
                # target_type_ids += [lsn for _ in range(len(tid))]
            # assert len(target_type_ids) == len(target_type_ids)

            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
                # target_type_ids = target_type_ids[:self.tgt_max_length]
            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_position_ids += [0]
                labels += [-1]
                # target_type_ids += [self.pad]
        labels = [-1] * self.src_max_length + labels

        if self.args.do_eval:
            return (torch.tensor(src_input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    ), '  '.join(src), '  '.join(tgt),
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    )


class roc_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=80, tgt_max_length=30,
                 do_generate=False):
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


    def load(self):
        self.source = []
        self.target = []
        # case_ids = [2, 5, 15, 14, 28, 34, 40, 42, 97, 123, 190, 251, 272, 292, 358, 3210, 2611, 2504, 2466, 1941, 1912, 1860, 1836, 1690, 1679, 1662, 1647, 1636, 1633, 1628, 1616, 1614, 1605, 1557, 1554, 1539, 1525, 1503, 1500, 1486, 1484, 1480, 1469, 1450, 1432, 1417, 1400, 1398, 1337, 1323]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            # for i, item in enumerate(data):
            #     if i not in case_ids:
            #         continue
            for item in data:
                # 生成故事结尾
                self.source.append(item['story'][:4])
                self.target.append(item['story'][4:])

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

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

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

        if self.args.do_eval:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    ), '  '.join(src), '  '.join(tgt),
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(src_type_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    )


class ed_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=80, tgt_max_length=30,
                 do_generate=False):
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

        num_added_toks = tokenizer.add_tokens(['<|EVENT|>',
                                               '<|CONCEPT|>'])


    def load(self):
        self.source = []
        self.target = []

        self.source_event, self.target_event = [], []
        self.source_kw, self.target_kw = [],[]

        with open(self.data_path, 'r') as f:
            data = json.load(f)
            for item in data:
                self.source.append(item['context'])
                self.target.append([item['response']])

                context_event_path = ' '.join(item['dialogue_events'][0])
                response_event_path = ' '.join(item['event_prediction'][1:4])

                context_concept_path = ' '.join(item['context_keywords'])
                try:
                    response_concept_path = ' '.join(item['prediction_keywords'][0][:3])
                except KeyError:
                    print(item)
                    pdb.set_trace()


                self.source_event.append(context_event_path)
                self.target_event.append(response_event_path)
                self.source_kw.append(context_concept_path)
                self.target_kw.append(response_concept_path)

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

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

        src_event = self.source_event[idx]
        tgt_event = self.target_kg[idx]
        src_kw = self.source_kw[idx]
        tgt_kw = self.target_kw[idx]


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

        if isinstance(src_event, list):
            src_event = ' '.join(src_event)
        src_event_input_ids = self.tokenizer.encode('<|bos|> ' + src_event)
        if isinstance(tgt_event, list):
            tgt_event = ' '.join(tgt_event)
        tgt_event_input_ids = self.tokenizer.encode(tgt_event)
        event_input_ids = src_event_input_ids + tgt_event_input_ids
        if len(event_input_ids) > self.src_max_length:
            event_input_ids = event_input_ids[:self.src_max_length]

        event_position_ids = list(range(0, len(event_input_ids)))
        event_attention_mask = [1] * len(event_input_ids)
        event_type_ids = [2] * len(event_input_ids)

        while len(event_input_ids) < self.src_max_length:
            event_input_ids += [self.pad]
            event_position_ids += [0]
            event_attention_mask += [0]
            event_type_ids += [self.pad]

        if isinstance(src_kw, list):
            src_kw = ' '.join(src_kw)
        src_kw_input_ids = self.tokenizer.encode('<|bos|> ' + src_kw)
        if isinstance(tgt_kw, list):
            tgt_kw = ' '.join(tgt_kw)
        tgt_kw_input_ids = self.tokenizer.encode(tgt_kw)
        kw_input_ids = src_kw_input_ids + tgt_kw_input_ids
        if len(kw_input_ids) > self.src_max_length:
            kw_input_ids = kw_input_ids[:self.src_max_length]

        kw_position_ids = list(range(0, len(kw_input_ids)))
        kw_attention_mask = [1] * len(kw_input_ids)
        kw_type_ids = [3] * len(kw_input_ids)
        while len(kw_input_ids) < self.src_max_length:
            kw_input_ids += [self.pad]
            kw_position_ids += [0]
            kw_attention_mask += [0]
            kw_type_ids += [self.pad]


        all_src_input_ids = kw_input_ids + event_input_ids + src_input_ids
        all_src_attention_mask = kw_attention_mask + event_attention_mask + attention_mask
        all_src_position_ids = kw_position_ids + event_position_ids + src_position_ids
        all_src_type_ids = kw_type_ids + event_type_ids + src_type_ids


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

        if self.args.do_eval:
            return (torch.tensor(all_src_input_ids),
                    torch.tensor(all_src_type_ids),
                    torch.tensor(all_src_attention_mask),
                    torch.tensor(all_src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    ), \
                   '  '.join(src), \
                   '  '.join(tgt),  \
                   '  '.join(src_event) if isinstance(src_event, list) else src_event, \
                   '  '.join(tgt_event) if isinstance(tgt_event, list) else tgt_event,\
                   '  '.join(src_kw) if isinstance(src_kw, list) else src_kw, \
                   '  '.join(tgt_kw) if isinstance(tgt_kw, list) else tgt_kw


        else:
            return (torch.tensor(all_src_input_ids),
                    torch.tensor(all_src_type_ids),
                    torch.tensor(all_src_attention_mask),
                    torch.tensor(all_src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_type_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    )


