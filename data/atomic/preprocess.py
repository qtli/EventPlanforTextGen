import configparser
import csv
import ast
import json
import pdb

from tqdm import tqdm
import networkx as nx
import os
from pattern.en import lemma, tag
import random
import numpy as np
import math
import linecache
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

relation_mapping = dict()

config = configparser.ConfigParser()
config.read("paths.cfg")


def pattern_stopiteration_workaround():
    try:
        print(lemma('gave'))
    except:
        pass


atomic = None
relations = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
self_rel = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
other_rel = ['oEffect', 'oReact', 'oWant']

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


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract():
    """
    Reads original atomic csv file and extracts all relations into a new file,
    with the following format for each line: <head> <relation> <tail> <prefix>
    :return:
    """
    total_data = set()
    titles = []
    with open(config["paths"]["atomic_raw"], encoding="utf8") as f:
        for i, line in enumerate(csv.reader(f)):
            if i == 0:
                titles = line
                continue
            info = line
            event = info[0]
            event = event.replace('___', 'none')
            event = event.lower()
            event = event.replace('personx', 'PersonX')
            event = event.replace('persony', 'PersonY')
            # prefix = ast.literal_eval(info[10])
            prefix = info[10]
            for rel in range(1,10):
                subevent = ast.literal_eval(info[rel])
                for se in subevent:
                    if se == 'none':
                        continue
                    se = se.lower()
                    se = se.replace('personx', 'PersonX')
                    se = se.replace('persony', 'PersonY')
                    total_data.add("\t".join([event.strip(), titles[rel].strip(), se.strip(), prefix]))

    total_data = list(total_data)
    with open(config["paths"]["atomic"], "w", encoding="utf8") as f:
        f.write("\n".join(total_data))
    print(len(total_data))  # 712466


def remove_empty_str(str):
    while '  ' in str:
        str = str.replace('  ', ' ')
    return str


def create_dic():
    event_file = open(config["paths"]["atomic_event"], "w", encoding="utf8")
    rel_file = open(config["paths"]["atomic_rel"], "w", encoding="utf8")

    with open(config["paths"]["atomic"], "r", encoding="utf8") as f:
        e2i = {}
        r2i = {}
        i2e = {}
        i2r = {}
        for line in f.readlines():
            tri = line.strip('\n').split('\t')
            hevent = remove_empty_str(tri[0].strip())
            rel = tri[1].strip()
            tevent = tri[2].strip()

            if tevent.startswith('PersonX'):
                tevent = tevent.lstrip('PersonX')
            if tevent.startswith('PersonY'):
                tevent = tevent.lstrip('PersonY')
            if tevent.strip()=='':
                continue

            # prefix = tri[3]
            # e.g., to help PersonX --> PersonX wants to help PersonY. Then the tail event also could serve as a head event.
            tevent = rel_tevent_translate[rel] + ' ' + tevent
            tevent = remove_empty_str(tevent)

            hevent = hevent.replace('PersonX','[NAME]')
            hevent = hevent.replace('PersonY','[NAME]')
            tevent = tevent.replace('PersonX','[NAME]')
            tevent = tevent.replace('PersonY','[NAME]')
            hevent = hevent.lower()
            tevent = tevent.lower()

            if hevent.strip().strip('\n') not in e2i:
                e2i[hevent.strip().strip('\n')] = len(e2i)
                i2e[len(i2e)] = hevent.strip().strip('\n')
            if tevent.strip().strip('\n') not in e2i:
                e2i[tevent.strip().strip('\n')] = len(e2i)
                i2e[len(i2e)] = tevent.strip().strip('\n')
            if rel.strip().strip('\n') not in r2i:
                r2i[rel.strip().strip('\n')] = len(r2i)
                i2r[len(i2r)] = rel.strip().strip('\n')

        r2i_original = r2i.copy()
        for r in r2i_original:
            r2i['_' + r] = len(r2i)
            i2r[len(i2r)] = '_' + r

    json.dump([e2i, i2e], event_file, indent=4)
    json.dump([r2i, i2r], rel_file, indent=4)


def load_event_and_relation_dicts():
    with open(config["paths"]["atomic_event"], "r", encoding="utf8") as f:
        [e2i, i2e] = json.load(f)
    print("event2id done")

    with open(config["paths"]["atomic_rel"], "r", encoding="utf8") as f:
        [r2i, i2r] = json.load(f)
    print("relation2id done")
    return e2i, i2e, r2i, i2r




def save_atomic():
    graph = nx.MultiDiGraph()
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
            rel = r2i[rel]
            hevent_clean = hevent.replace('PersonX', '[NAME]')
            hevent_clean = hevent_clean.replace('PersonY', '[NAME]')
            tevent_clean = tevent.replace('PersonX', '[NAME]')
            tevent_clean = tevent_clean.replace('PersonY', '[NAME]')

            if tevent_clean.replace('[NAME]','').strip() == '' \
                    or hevent_clean.replace('[NAME]','').strip() == '' \
                    or hevent_clean == tevent_clean:  # delete loops
                continue
            prefix = ls[3]

            # whether the order of NAMES in head event & tail event is consistent，
            # transfer the [NAME] in multi-hop path into PersonY or PersonX
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
            graph.add_edge(e2i[hevent_clean.lower().strip().strip('\n')], e2i[tevent_clean.lower().strip().strip('\n')], rel=rel, prefix=prefix, head_names=h_name_list, tail_name_pos=t_name_list)

            # change the event order
            hns = tevent.split()
            h_name_list = []  # string list
            for s in hns:
                if 'PersonX' in s:
                    h_name_list.append('PersonX')
                elif 'PersonY' in s:
                    h_name_list.append('PersonY')
                else:
                    continue
            tns = hevent.split()
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
            graph.add_edge(e2i[tevent_clean.lower().strip().strip('\n')], e2i[hevent_clean.lower().strip().strip('\n')], rel=rel+int(len(r2i)/2), prefix=prefix, head_names=h_name_list, tail_name_pos=t_name_list)  # reverse link

    nx.write_gpickle(graph, config["paths"]["atomic_graph"])
    print(len(graph.nodes()))  # 317476



def load_atomic_graph():
    print("loading atomic....")
    kg_full = nx.read_gpickle(config["paths"]["atomic_graph"])

    kg_simple = nx.DiGraph()  # 有向图
    for u, v, data in kg_full.edges(data=True):
        kg_simple.add_edge(u, v)

    return kg_full, kg_simple


def path2str(path, path_names):
    str2w = []
    _idx = 0
    _idx_rel = 0
    ent = i2e[str(path[_idx])]  # first event in the path
    ent_w_xy = []
    hent_names = path_names[_idx_rel][0]
    name_idx = 0
    for i,s in enumerate(ent.split()):
        if '[name]' in s:
            ent_w_xy.append(s.replace('[name]',hent_names[name_idx]))
            name_idx +=1
        else:
            ent_w_xy.append(s)
    str2w.append(' '.join(ent_w_xy))

    _idx += 1
    while _idx < len(path):
        rel = i2r[str(path[_idx])]
        _idx += 1

        ent = i2e[str(path[_idx])]
        ent_w_xy = []
        name_idx = 0
        hent_names_tmp = []
        for i,s in enumerate(ent.split()):
            if '[name]' in s:
                if path_names[_idx_rel][1][name_idx] != 999:
                    name = hent_names[path_names[_idx_rel][1][name_idx]]
                else:
                    name = 'PersonY' if hent_names[-1] == 'PersonX' else 'PersonX'
                hent_names_tmp.append(name)
                ent_w_xy.append(s.replace('[name]',name))
                name_idx+=1
            else:
                ent_w_xy.append(s)
        _idx += 1
        str2w.append(rel)
        str2w.append(' '.join(ent_w_xy))

        hent_names = hent_names_tmp
        _idx_rel += 1
    str2w = '\t'.join(str2w) + '\n'
    return str2w


def random_walk(start_node, kg_full, kg_simple, max_len=3):
    """
    For informativeness, you can require all relation types in a path to be distinct using 'flag_valid' variable.
    :param start_node:
    :param kg_full:
    :param kg_simple:
    :param max_len:
    :return:
    """
    curr_node = start_node
    num_sampled_nodes = 1
    path = [curr_node]
    path_names = []
    node_visited = set()
    relation_visited = [None]
    node_visited.add(curr_node)
    while num_sampled_nodes != max_len:  # construct a path
        if curr_node not in kg_simple:  # leaf node
            break
        edges = [n for n in kg_simple[curr_node]]
        iteration_node = 0
        chosen_node = None
        chosen_rel = None
        while True:
            if len(edges) == 0 :
                continue
            index_of_edge = random.randint(0, len(edges) - 1)  # select an edge(relation)
            chosen_node = edges[index_of_edge]  # select a node(event)
            if not chosen_node in node_visited:  # select this neighbour node
                rel_list = kg_full[curr_node][chosen_node]
                rel_list = list(set([rel_list[item]['rel'] for item in rel_list]))
                iteration_rel = 0
                while True:  # select an arbitrary relation between two nodes
                    index_of_rel = np.random.choice(len(rel_list), 1)[0]
                    chosen_rel = kg_full[curr_node][chosen_node][index_of_rel]['rel']

                    if relation_visited[-1] is not None and chosen_rel % 9 == relation_visited[-1] % 9 and chosen_rel != relation_visited[-1]:
                        # avoid walking back
                        flag_valid = False
                        iteration_rel += 1
                        if iteration_rel >= 2:
                            break
                    else:
                        path_names.append([kg_full[curr_node][chosen_node][index_of_rel]['head_names'],
                                           kg_full[curr_node][chosen_node][index_of_rel]['tail_name_pos']])
                        flag_valid = True  # we do not set any restriction towards relation type.
                        break
                if flag_valid:  # this path is valid
                    break
                else:  # this path is invalid
                    iteration_node += 1
                    if iteration_node >= 2:  # try at most two times from this neighbour node
                        return [],[]
            else:
                iteration_node += 1
                if iteration_node >= 2:
                    return [],[]

        node_visited.add(chosen_node)
        relation_visited.append(chosen_rel)

        path.append(chosen_rel)
        path.append(chosen_node)

        curr_node = chosen_node
        num_sampled_nodes += 1

    return path, path_names


def sample_path():
    # num_paths = [10, 8, 6]
    # path_lens = [2, 3, 4]  # 1,2,3 hop

    path_lens = [2, 3, 4, 5, 6]  # 1,2,3,4,5 hop
    num_paths = [8, 8, 6, 6, 4]

    print('loading relations and events')
    print('num of events: {}'.format(len(i2e)))  # 388809 --> 317476
    print('num of relations: {}'.format(len(i2r)))  # 18

    print('loading kg')
    kg_full, kg_simple = load_atomic_graph()

    nr_nodes = len(kg_full.nodes())
    visited_event = dict()
    not_visited_idx = []
    for evt_idx in range(nr_nodes):
        if not i2e[str(evt_idx)] in visited_event:
            not_visited_idx.append(evt_idx)
    print('not visited: {}'.format(len(not_visited_idx)))

    with open(config["paths"]["atomic_sample_path"], 'w') as fw:
        for curr_node in tqdm(not_visited_idx, desc='generating paths'):
            if not curr_node in kg_simple:
                continue
            for _id, _len in enumerate(path_lens):
                visited_path = set()
                for pid in range(num_paths[_id]):
                    num_try = 0
                    while True:
                        path, path_names = random_walk(curr_node, kg_full, kg_simple, _len)
                        if len(path) > 0:
                            str2w = path2str(path, path_names)
                            if str2w not in visited_path:
                                fw.write(str(len(path))+' | '+str(path)+' | ')
                                fw.write(str2w)
                                visited_path.add(str2w)
                                break
                        num_try += 1
                        if num_try >10:
                            break

    fw.close()
    print('finish!')


def split_sample_path():
    numb = len(open(config["paths"]["atomic_sample_path"], 'r').readlines())
    with open(config["paths"]["atomic_sample_path"], 'r') as f:
        idx = [i for i in range(numb)]
        # tst_dev_idx = random.sample(idx, math.ceil(numb * 0.1))  # without replacement sampling
        # tst_dev_num = len(tst_dev_idx)
        # tst_idx = random.sample(tst_dev_idx, math.ceil(tst_dev_num * 0.5))

        test = open(config["paths"]["test_path"], 'w', encoding='utf-8')
        dev = open(config["paths"]["dev_path"], 'w', encoding='utf-8')
        train = open(config["paths"]["train_path"], 'w', encoding='utf-8')
        tc = math.ceil(numb * 0.05) # 200735 test size  332018
        dc = math.ceil(numb * 0.05)  # 200735 dev size
        trc = numb-tc-dc  # 3,614,981 train size
        random.shuffle(idx)

        test_c = 0
        dev_c = 0
        train_c = 0

        for i, id in tqdm(enumerate(idx)):
            line = linecache.getline(config["paths"]["atomic_sample_path"], id)
            if i < tc:
                test.write(line)
                test_c += 1
            elif i < (tc + dc):
                dev.write(line)
                dev_c += 1
            elif i < tc+dc+trc:
                train.write(line)
                train_c += 1
            else:
                break

        print('finish train: {}'.format(train_c))
        print('finish dev: {}'.format(dev_c))
        print('finish test: {}'.format(test_c))
        test.close()
        train.close()
        dev.close()


def add_special_token():
    test = open(config["paths"]["test_path"].replace('test', 'atomic_test'), 'w')
    with open(config["paths"]["test_path"], 'r') as fr:
        for line in fr:
            line_split = line.strip().split('\t')
            if '' in line_split:
                line_split.remove('')
            if line_split[-1].strip() in special_tokens:
                continue
            text = []
            insert_num = len(line_split)-1
            sep_pos = np.random.choice(insert_num, 1)
            # randomly add a [SEP] token between two items (an event and a relation)
            for _idx, element in enumerate(line_split):
                if sep_pos == _idx:
                    text.append(element)
                    text.append('<SEP>')
                else:
                    text.append(element)
            assert len(text) == len(line_split)+1
            test.write('\t'.join(text) + '\n')
    test.close()

    dev = open(config["paths"]["dev_path"].replace('dev', 'atomic_dev'), 'w')
    with open(config["paths"]["dev_path"], 'r') as fr:
        for line in fr:
            line_split = line.strip().split('\t')
            if '' in line_split:
                line_split.remove('')
            if line_split[-1].strip() in special_tokens:
                continue
            text = []
            insert_num = len(line_split)-1
            sep_pos = np.random.choice(insert_num, 1)
            for _idx, element in enumerate(line_split):
                if sep_pos == _idx:
                    text.append(element)
                    text.append('<SEP>')
                else:
                    text.append(element)
            assert len(text) == len(line_split)+1
            dev.write('\t'.join(text) + '\n')
    dev.close()

    train = open(config["paths"]["train_path"].replace('train', 'atomic_train'), 'w')
    with open(config["paths"]["train_path"], 'r') as fr:
        for line in fr:
            line_split = line.strip().split('\t')
            if '' in line_split:
                line_split.remove('')
            if line_split[-1].strip() in special_tokens:
                continue
            text = []
            insert_num = len(line_split) - 1
            sep_pos = np.random.choice(insert_num, 1)
            for _idx, element in enumerate(line_split):
                if sep_pos == _idx:
                    text.append(element)
                    text.append('<SEP>')
                else:
                    text.append(element)
            assert len(text) == len(line_split)+1
            train.write('\t'.join(text) + '\n')
        train.close()


special_tokens = ['<SEP>', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant']
unconsider_tokens = {'PersonX': 'he', 'PersonY': 'she', 'PersonZ': ''}

# just remove stopwords and unconsider_tokens
def simplify_each_path():
    pattern_stopiteration_workaround()

    test = open(config["paths"]["test_path"].replace('test', 'sim_atomic_test'),'w')
    with open(config["paths"]["test_path"].replace('test', 'atomic_test'), 'r') as fr:
        for line in fr:
            line = line.split('|')[-1].strip('')
            items = line.split('\t')
            new_line = []
            for term in items:
                new_term = []
                words = term.split()
                for word in words:
                    if_add = True
                    if word.lower() in stopwords:  # not consider stopword
                        continue
                    for ut in unconsider_tokens:  # replace unconsidered_tokens with specified tokens
                        if ut in word:
                            word = word.replace(ut,unconsider_tokens[ut])
                    if word not in special_tokens:  # special tokens need to be reserved
                        if '\'s' not in word:
                            word = lemma(word)  # Morphological reduction
                    if if_add:
                        new_term.append(word)
                if new_term == []:
                    new_term = words
                new_term = ' '.join(new_term)
                new_line.append(new_term)
            new_line = '\t'.join(new_line)
            test.write(new_line + '\n')
    test.close()

    dev = open(config["paths"]["dev_path"].replace('dev', 'sim_atomic_dev'),'w')
    with open(config["paths"]["dev_path"].replace('dev', 'atomic_dev'), 'r') as fr:
        for line in fr:
            line = line.split('|')[-1].strip('')
            items = line.split('\t')
            new_line = []
            for term in items:
                new_term = []
                words = term.split()
                for word in words:
                    if_add = True
                    if word.lower() in stopwords:
                        continue
                    for ut in unconsider_tokens:
                        if ut in word:
                            word = word.replace(ut,unconsider_tokens[ut])
                    if word not in special_tokens:  # special tokens need to be reserved
                        if '\'s' not in word:
                            word = lemma(word)  # Morphological reduction
                    if if_add:
                        new_term.append(word)
                if new_term == []:
                    new_term = words
                new_term = ' '.join(new_term)
                new_line.append(new_term)
            new_line = '\t'.join(new_line)
            dev.write(new_line + '\n')
    dev.close()

    train = open(config["paths"]["train_path"].replace('train', 'sim_atomic_train'),'w')
    with open(config["paths"]["test_path"].replace('train', 'atomic_train'), 'r') as fr:
        for line in fr:
            line = line.split('|')[-1].strip('')
            items = line.split('\t')
            new_line = []
            for term in items:
                new_term = []
                words = term.split()
                for word in words:
                    if_add = True
                    if word.lower() in stopwords:
                        continue
                    for ut in unconsider_tokens:
                        if ut in word:
                            word = word.replace(ut, unconsider_tokens[ut])
                    if word not in special_tokens:  # special tokens need to be reserved
                        if '\'s' not in word:
                            word = lemma(word)  # Morphological reduction
                    if if_add:
                        new_term.append(word)
                if new_term == []:
                    new_term = words
                new_term = ' '.join(new_term)
                new_line.append(new_term)
            new_line = '\t'.join(new_line)
            train.write(new_line + '\n')
    train.close()


def get_relation_embeddding_id():
    test_relation_emb_id = open(config["paths"]["test_relation_embedding_id"],'w')
    total_ids = []
    with open(config["paths"]["test_path"].replace('test', 'sim_atomic_test'),'r') as f:
        for line in f.readlines():
            ids = []
            line = line.replace('<SEP>','')
            items = line.strip('\n').split('\t')
            if '' in items: items.remove('')
            last_rel = ''
            for i, item in enumerate(items):
                if i == 0:
                    ids.append('PRP')
                elif (i+1) % 2 ==0:
                    ids.append(item.strip())
                    last_rel = item.strip()
                else:
                    ids.append(last_rel)
            total_ids.append(ids)
    json.dump(total_ids, test_relation_emb_id, indent=4)
    test_relation_emb_id.close()

    dev_relation_emb_id = open(config["paths"]["dev_relation_embedding_id"],'w')
    total_ids = []
    with open(config["paths"]["dev_path"].replace('dev', 'sim_atomic_dev'),'r') as f:
        for line in f.readlines():
            ids = []
            line = line.replace('<SEP>','')
            items = line.strip('\n').split('\t')
            if '' in items: items.remove('')
            last_rel = ''
            for i, item in enumerate(items):
                if i == 0:
                    ids.append('PRP')
                elif (i+1) % 2 ==0:
                    ids.append(item.strip())
                    last_rel = item.strip()
                else:
                    ids.append(last_rel)
            total_ids.append(ids)
    json.dump(total_ids, dev_relation_emb_id, indent=4)
    dev_relation_emb_id.close()

    train_relation_emb_id = open(config["paths"]["train_relation_embedding_id"],'w')
    total_ids = []
    with open(config["paths"]["train_path"].replace('train', 'sim_atomic_train'),'r') as f:
        for line in f.readlines():
            ids = []
            line = line.replace('<SEP>','')
            items = line.strip('\n').split('\t')
            if '' in items: items.remove('')
            last_rel = ''
            for i, item in enumerate(items):
                if i == 0:
                    ids.append('PRP')
                elif (i+1) % 2 ==0:
                    ids.append(item.strip())
                    last_rel = item.strip()
                else:
                    ids.append(last_rel)
            total_ids.append(ids)
    json.dump(total_ids, train_relation_emb_id, indent=4)
    train_relation_emb_id.close()


if __name__ == "__main__":
    # Step 1: extract event transition graph from raw ATOMIC file. Produce the atomic.all file.
    extract()
    print('finish extracting eventual graph')


    # Step 2: creat dictionary for each event entity and each relation.
    create_dic()
    print('finish creating event dict')


    e2i, i2e, r2i, i2r = load_event_and_relation_dicts()

    # Step 3: save the processed atomic data in a graph file.
    save_atomic()
    print('finish saving processed atomic')


    # Step 4: sample event transition paths from atomic graph via random walk.
    sample_path()
    print('finish path sampling')


    # Step 5: split sampled paths into training/development/test set with a ratio of 9.0/0.5/0.5.
    split_sample_path()
    print('finish train/dev/test splits')


    # Step 6: process each sampled path.
    # 6.1. add a <SEP> special token into each path
    add_special_token()
    print('finish add special token')


    # 6.2. simplify each path.
    simplify_each_path()
    print('finish simplying')

    # 6.3. get relation embedding id for each path
    get_relation_embeddding_id()
    print('finish relation embedding')
