import json
import configparser
import os.path

from tqdm import tqdm
import random
import math
import linecache

config = configparser.ConfigParser()
config.read("paths.cfg")

def remove_empty_str(str):
    while '  ' in str:
        str = str.replace('  ', ' ')
    return str

rel_tevent_translate = {
    'xAttr': 'PersonX is',
    'xEffect': 'PersonX',
    'xIntent': 'PersonX wants',
    'xNeed': 'PersonX needs',
    'xReact': 'PersonX feels',
    'xWant': 'PersonX wants ',
    'oEffect': 'PersonY',
    'oReact': 'PersonY feels',
    'oWant': 'PersonY wants',
}

def get_all_event_triples():
    all_triples = set()
    with open(config["paths"]["atomic"], "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="saving to graph"):
            ls = line.strip('\n').split('\t')
            hevent = remove_empty_str(ls[0].strip())
            rel = ls[1]
            tevent = ls[2].strip()

            if tevent.startswith('PersonX'):
                tevent = tevent.lstrip('PersonX')
            if tevent.startswith('PersonY'):
                tevent = tevent.lstrip('PersonY')
            tevent = rel_tevent_translate[rel] + ' ' + tevent
            tevent = remove_empty_str(tevent)
            hevent_clean = hevent.replace('PersonX', '[NAME]')
            hevent_clean = hevent_clean.replace('PersonY', '[NAME]')
            tevent_clean = tevent.replace('PersonX', '[NAME]')
            tevent_clean = tevent_clean.replace('PersonY', '[NAME]')

            if tevent_clean.replace('[NAME]', '').strip() == '' \
                    or hevent_clean.replace('[NAME]', '').strip() == '' \
                    or hevent_clean == tevent_clean:  # delete loops
                continue
            prefix = ls[3]

            hns = hevent.split()
            h_name_list = []  # string list
            for s in hns:
                if 'PersonX' in s:
                    h_name_list.append('PersonX')
                elif 'PersonY' in s:
                    h_name_list.append('PersonY')
                else:
                    continue
            tns = tevent.split()
            t_name_list = []  # int list
            for s in tns:
                if 'PersonX' in s:
                    if 'PersonX' in h_name_list:
                        t_name_list.append(h_name_list.index('PersonX'))
                    else:
                        t_name_list.append(999)
                elif 'PersonY' in s:
                    if 'PersonY' not in h_name_list:
                        t_name_list.append(999)
                    else:
                        t_name_list.append(h_name_list.index('PersonY'))

            new_triple = '\t'.join([hevent_clean.lower().strip().strip('\n'), rel, tevent_clean.lower().strip().strip('\n')])
            all_triples.add(new_triple)
    all_triples = list(all_triples)
    if os.path.exists('./event_pairs') is False:
        os.makedirs('./event_pairs')
    json.dump(all_triples, open(config["paths"]["all_event_triples"], 'w'), indent=4)


def split_event_triples():
    data = json.load(open(config["paths"]["all_event_triples"], 'r'))
    numb = len(data)
    idx = [i for i in range(numb)]
    tst_dev_idx = random.sample(idx, math.ceil(numb * 0.1))
    tst_dev_num = len(tst_dev_idx)
    tst_idx = random.sample(tst_dev_idx, math.ceil(tst_dev_num * 0.5))

    test = open(config["paths"]["train_event_triples"], 'w', encoding='utf-8')
    dev = open(config["paths"]["dev_event_triples"], 'w', encoding='utf-8')
    train = open(config["paths"]["test_event_triples"], 'w', encoding='utf-8')

    tc = 0
    for tst_i in tst_idx:
        test.write(data[tst_i] + '\n')
        tc += 1
    print('finish test: {}'.format(tc))
    test.close()

    dc = 0
    for dev_i in tst_dev_idx:
        if dev_i not in tst_idx:
            dev.write(data[dev_i]+'\n')
            dc+=1
    print('finish dev: {}'.format(dc))
    dev.close()

    trc = 0
    for tra_i in idx:
        if tra_i not in tst_dev_idx:
            train.write(data[tra_i] + '\n')
            trc += 1
    print('finish atomic train: {}'.format(trc))
    train.close()


if __name__ == '__main__':
    # Step 1: collect event triples  "head_event \t relation \t tail_event" from ATOMIC graph
    get_all_event_triples()

    # Step 2: split event triples into training/development/test sets.
    split_event_triples()
