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


class ed_DatasetHelper(Dataset):
    def __init__(self, args, tokenizer, data_path, relation_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False,):
        self.do_generate = do_generate
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.data_path = data_path


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
                response_concept_path = ' '.join(item['prediction_keywords'][:3])

                self.source_event.append(context_event_path)
                self.target_event.append(response_event_path)
                self.source_kw.append(context_concept_path)
                self.target_kw.append(response_concept_path)


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]

        src_event = self.source_event[idx]
        tgt_event = self.target_kg[idx]
        src_kw = self.source_kw[idx]
        tgt_kw = self.target_kw[idx]

        src_input_ids = []
        for si, s in enumerate(src):
            s = ' ' + s
            s += ' <|endoftext|>'  # 每个utterance后面跟一个结束符
            s_ids = self.tokenizer.encode(s)
            src_input_ids.extend(s_ids)

        ##################### DIALOGUE PART #########################
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
            assert len(tgt) == 1
            for ti, t in enumerate(tgt):
                if ti == 0:
                    target_input_ids.extend([self.bos])
                tid = self.tokenizer.encode(' ' + t)
                target_input_ids.extend(tid)

            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
            target_position_ids = list(range(0, len(target_input_ids)))
            labels = target_input_ids[1:] + [self.eos]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_position_ids += [0]
                labels += [-1]
        labels = [-1] * self.src_max_length + labels

        ##################### KG PART #########################
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
        while len(event_input_ids) < self.src_max_length:
            event_input_ids += [self.pad]
            event_position_ids += [0]
            event_attention_mask += [0]

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
        while len(kw_input_ids) < self.src_max_length:
            kw_input_ids += [self.pad]
            kw_position_ids += [0]
            kw_attention_mask += [0]

        # if self.args.do_eval:
        if self.do_generate:
            return (torch.tensor(src_input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(event_input_ids),
                    torch.tensor(event_attention_mask),
                    torch.tensor(event_position_ids),
                    torch.tensor(kw_input_ids),
                    torch.tensor(kw_position_ids),
                    torch.tensor(kw_attention_mask),
                    ), '  '.join(src), '  '.join(tgt),\
                   '  '.join(src_event) if isinstance(src_event, list) else src_event, \
                   '  '.join(tgt_event) if isinstance(tgt_event, list) else tgt_event,\
                   '  '.join(src_kw) if isinstance(src_kw, list) else src_kw, \
                   '  '.join(tgt_kw) if isinstance(tgt_kw, list) else tgt_kw
        else:
            return (torch.tensor(src_input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(src_position_ids),
                    torch.tensor(target_input_ids),
                    torch.tensor(target_position_ids),
                    torch.tensor(labels),
                    torch.tensor(event_input_ids),
                    torch.tensor(event_attention_mask),
                    torch.tensor(event_position_ids),
                    torch.tensor(kw_input_ids),
                    torch.tensor(kw_position_ids),
                    torch.tensor(kw_attention_mask),
                    )


