import time
import json
import csv
from multiprocessing import Pool, Lock, Value
import argparse
from aser.client import ASERClient
import pdb
import string
punctuation_string = string.punctuation  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
'''
aser-server -n_workers 1 -n_concurrent_back_socks 10 -port 8000 -port_out 8001 -corenlp_path "/root/ASER/stanford-corenlp-3.9.2" -base_corenlp_port 9000
'''
client = ASERClient(port=8000, port_out=8001)


def parse_one(text):
    text = text.strip()
    parsed_e, parsed_r = client.extract_eventualities_and_relations(text)

    res = []
    if parsed_e is None:
        text_s = text.split(' . ')
        for t in text_s:
            parsed_e, parsed_r = client.extract_eventualities_and_relations(t)
            if parsed_e is not None:
                res += parsed_e
        if res != []:
            parsed_e = res
    events = []
    if parsed_e is None:
        events = [text]
        # print('parsed_e is None: ', events)
    else:
        for s in parsed_e:
            for e in s:
                events.append(' '.join(e.words))
        if events == []:
            events = [text]
    return events


def parse_texts(texts):
    parsed = []
    count = 0
    start = time.time()
    for text in texts:
        result = parse_one(text)
        parsed.append(result)

        count += 1
        if count % 10 == 0:
            print(count, (time.time() - start) / count)
    return parsed


lock = Lock()
counter = Value('i', 0)
def parse_one_wrap(text, start):
    result = parse_one(text)
    global lock, counter
    with lock:
        counter.value += 1
    if counter.value % 1000 == 0:
        print(counter.value, (time.time() - start) / counter.value)
    return result


def multi_parse_texts(texts):
    parsed = []
    start = time.time()
    pool = Pool(processes=32)
    for text in texts:
        parsed.append(pool.apply_async(parse_one_wrap, (text, start, )))
    pool.close()
    pool.join()
    parsed = [p.get() for p in parsed]
    return parsed

def get_dialogue_event(input_file, output_file):
    data = json.load(open(input_file, 'r'))
    new_data = []
    for i, item in enumerate(data):
        print(i)
        context = item['context']
        dialogue_events = []  # each item contains the events of each utterance
        for utterance in context:
            # remove punctuation
            for i in punctuation_string:
                utterance = utterance.replace(i, '')
            u_e = parse_one(utterance)
            dialogue_events.append(u_e)
        response = item['response']
        u_r = parse_one(response)
        dialogue_events.append(u_r)

        item['dialogue_events'] = dialogue_events
        new_data.append(item)

    json.dump(new_data, open(output_file, 'w'), indent=4)


def simplify_dialogue_event(input_file, output_file):
    data = json.load(open(input_file, 'r'))
    new_data = []
    for i, item in enumerate(data):
        print(i)
        context = item['context']
        dialogue_events = item['dialogue_events']  # each item contains the events of each utterance
        # for events in dialogue_events:
        dialogue_sim_events = []
        for u_events in dialogue_events:
            new_u_events = []
            for event in u_events:
                new_event = []
                for w in event.split():
                    if w not in stopwords:
                        new_event.append(w)
                new_u_events.append(' '.join(new_event))
            dialogue_sim_events.append(new_u_events)
        item['dialogue_sim_events'] = dialogue_sim_events
        new_data.append(item)
    json.dump(new_data, open(output_file, 'w'), indent=4)

def extract_event_pairs(input_file, output_file):
    data = json.load(open(input_file, 'r'))
    f = open(output_file, 'w')
    for item in data:
        last_event = ""
        dialogue_events = item["dialogue_events"]
        for utt_events in dialogue_events:
            for event in utt_events:
                if last_event == "":
                    last_event = event
                    continue
                else:
                    f.write(last_event + '\t' + event + '\n')
                    last_event = event

    f.close()


def feed_relation_embedding_id(input_file, output_file):
    data = json.load(open(input_file, 'r'))
    new_data = []
    for item in data:
        knowledge_path = item["dialogue_events_relations"]

        context_path = []
        for u_path in knowledge_path[:-1]:
            context_path.extend(u_path)
        response_path = knowledge_path[-1]
        item["context_response_path"] = [context_path, response_path]

        context_relation_ids = []
        last_rel = ""
        for i, cp in enumerate(context_path):
            if i == 0:
                context_relation_ids.append('PRP')
            elif (i + 1) % 2 == 0:
                context_relation_ids.append(cp.strip())
                last_rel = cp.strip()
            else:
                context_relation_ids.append(last_rel)

        response_relation_ids = []
        last_rel = ""
        for j, rp in enumerate(response_path):
            if (j+1) % 2 == 1:
                response_relation_ids.append(rp.strip())
                last_rel = rp.strip()
            else:
                response_relation_ids.append(last_rel)
        item["context_response_path_relation_ids"] = [context_relation_ids, response_relation_ids]
        new_data.append(item)
        pdb.set_trace()

    json.dump(new_data, open(output_file, 'r'), indent=4)

import math
class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores

def retrieve_kg_from_context(input_file, knowledge_file, output_file):
    doc = []
    doc_te = []
    doc_r = []

    dev_knowledge = open('/apdcephfs/share_916081/qtli/kggen/data/knowledge/atomic/event_pairs/sim_event_pairs_dev.txt', 'r').readlines()
    for line in dev_knowledge:
        he, r, te = line.split('\t')
        he_words = he.strip().split(' ')
        if '' in he_words:
            he_words.remove('')
        doc.append(he_words)
        doc_te.append(te)
        doc_r.append(r)

    train_knowledge = open('/apdcephfs/share_916081/qtli/kggen/data/knowledge/atomic/event_pairs/sim_event_pairs_train.txt', 'r').readlines()
    for line in train_knowledge:
        he, r, te = line.split('\t')
        he_words = he.strip().split(' ')
        if '' in he_words:
            he_words.remove('')
        doc.append(he_words)
        doc_te.append(te)
        doc_r.append(r)

    test_knowledge = open('/apdcephfs/share_916081/qtli/kggen/data/knowledge/atomic/event_pairs/sim_event_pairs_test.txt', 'r').readlines()
    for line in test_knowledge:
        he, r, te = line.split('\t')
        he_words = he.strip().split(' ')
        if '' in he_words:
            he_words.remove('')
        doc.append(he_words)
        doc_te.append(te)
        doc_r.append(r)
    s = BM25(doc)
    print('finish bm25')

    # data = json.load(open(input_file, 'r'))
    # new_data = []
    # for i, item in enumerate(data):
    #     if i%50 == 0:
    #         print('train finish {}/{}'.format(i,len(data)))
    #     context = item["context"]
    #     context_words = []
    #     for c in context:
    #         for w in c.split(' '):
    #             w = w.strip()
    #             if w in punctuation_string or w in stopwords:
    #                 continue
    #             else:
    #                 context_words.append(w)
    #     scores = s.simall(context_words)
    #     retrieve_he = ' '.join(doc[scores.index(max(scores))])
    #     retrieve_te = doc_te[scores.index(max(scores))].strip()
    #     retrieve_r = doc_r[scores.index(max(scores))].strip()
    #     retrieve_kg = [retrieve_he, retrieve_r, retrieve_te]
    #     item['retrieve_kg'] = retrieve_kg
    #     new_data.append(item)
    #     if (i+1) % 50 == 0:
    #         json.dump(new_data, open('prop/ed_dev_propppp.json', 'w'), indent=4)
    # json.dump(new_data, open(output_file,'w'), indent=4)


    # data = json.load(open('prop/ed_train_prop.json', 'r'))
    # new_data = []
    # for i, item in enumerate(data):
    #     if i%50 == 0:
    #         print('train finish {}/{}'.format(i,len(data)))
    #     context = item["context"]
    #     context_words = []
    #     for c in context:
    #         for w in c.split(' '):
    #             w = w.strip()
    #             if w in punctuation_string or w in stopwords:
    #                 continue
    #             else:
    #                 context_words.append(w)
    #     scores = s.simall(context_words)
    #     retrieve_he = ' '.join(doc[scores.index(max(scores))])
    #     retrieve_te = doc_te[scores.index(max(scores))].strip()
    #     retrieve_r = doc_r[scores.index(max(scores))].strip()
    #     retrieve_kg = [retrieve_he, retrieve_r, retrieve_te]
    #     item['retrieve_kg'] = retrieve_kg
    #     new_data.append(item)
    #     if (i+1) % 50 == 0:
    #         json.dump(new_data, open('prop/ed_train_propppp.json', 'w'), indent=4)
    # json.dump(new_data, open('prop/ed_train_propppp.json','w'), indent=4)

    data_ori = json.load(open('prop/ed_test_prop.json', 'r'))
    data_prop = json.load(open('prop/ed_test_propppp.json', 'r'))

    data = data_ori[len(data_prop):]
    # new_data = []
    for i, item in enumerate(data):
        # if i%50 == 0:
        #     print('finish {}/{}'.format(i,len(data)))
        context = item["context"]
        context_words = []
        for c in context:
            for w in c.split(' '):
                w = w.strip()
                if w in punctuation_string or w in stopwords:
                    continue
                else:
                    context_words.append(w)
        scores = s.simall(context_words)
        retrieve_he = ' '.join(doc[scores.index(max(scores))])
        retrieve_te = doc_te[scores.index(max(scores))].strip()
        retrieve_r = doc_r[scores.index(max(scores))].strip()
        retrieve_kg = [retrieve_he, retrieve_r, retrieve_te]
        item['retrieve_kg'] = retrieve_kg
        data_prop.append(item)
        # new_data.append(item)
        # if (i+1) % 50 == 0:
        #     json.dump(new_data, open('prop/ed_test_propppp.json', 'w'), indent=4)
    assert len(data_ori)==len(data_prop),pdb.set_trace()
    json.dump(data_prop, open('prop/ed_test_propppp.json','w'), indent=4)
    '''
    final:
    prop/ed_test_prop.json
    prop/ed_dev_prop.json
    prop/ed_train_prop.json
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str, required=True, help='raw file')
    parser.add_argument('--output_file', default=None, type=str, required=True, help='processed file')
    parser.add_argument('--knowledge_file', default=None, type=str, required=True, help='processed file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    knowledge_file = args.knowledge_file

    # get_dialogue_event(input_file, output_file)
    simplify_dialogue_event(input_file, output_file)

    # extract event pairs from dataset for feed predicted relation
    extract_event_pairs(input_file, output_file)

    # prepare relation embedding ids for knowledge path
    # python prop_kg_path.py --input_file prop/ed_dev_prop.json --output_file prop/ed_dev_prop.json
    # feed_relation_embedding_id(input_file, output_file)

    # bm25的方式检索kg
    retrieve_kg_from_context(input_file, knowledge_file, output_file)
