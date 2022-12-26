import json
import csv
import os.path
import sys
import pdb
import re
import configparser
import string
import math
from tqdm import tqdm
from aser.client import ASERClient
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
punctuation_string = string.punctuation  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

config = configparser.ConfigParser()
config.read("paths.cfg")
from aser.extract.aser_extractor import SeedRuleASERExtractor, DiscourseASERExtractor

'''
Please refer to: https://hkust-knowcomp.github.io/ASER/html/tutorial/get-started.html
run:
aser-server -n_workers 1 -n_concurrent_back_socks 10 -port 8000 -port_out 8001 -corenlp_path "YOUR_DIRECTORY/stanford-corenlp-3.9.2" -base_corenlp_port 9000
'''
client = ASERClient(port=8000, port_out=8001)


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


def punc(s):
    s = s.replace('_comma_', ',')
    tmp = re.sub(r'([a-zA-Z])([,.!:;?])', r'\1 \2', s)
    s = re.sub(r'([,.!])([a-zA-Z])', r'\1 \2', tmp)
    s = s.strip()
    return s


def read_from_raw():
    train_data = []
    dev_data = []
    test_data = []
    dir = os.path.dirname(config["paths"]["prop_tst"])
    if os.path.exists(dir) is False:
        os.makedirs(dir)

    tf = csv.reader(open(os.path.join(config["paths"]["raw_tst"], "target.csv")))
    tf_list = list(tf)
    sf = csv.reader(open(os.path.join(config["paths"]["raw_tst"], "source.csv")))
    for i, line in enumerate(sf):
        source = line[1].split('\t')
        target = tf_list[i][1]
        story = source + [target]
        test_data.append({'source': source, 'target': target, 'story': story})

    tf = csv.reader(open(os.path.join(config["paths"]["raw_trn"], "target.csv")))
    tf_list = list(tf)
    sf = csv.reader(open(os.path.join(config["paths"]["raw_trn"], "source.csv")))
    for i, line in enumerate(sf):
        source = line[1].split('\t')
        target = tf_list[i][1]
        story = source + [target]
        train_data.append({'source': source, 'target': target, 'story': story})

    tf = csv.reader(open(os.path.join(config["paths"]["raw_dev"], "target.csv")))
    tf_list = list(tf)
    sf = csv.reader(open(os.path.join(config["paths"]["raw_dev"], "source.csv")))
    for i, line in enumerate(sf):
        source = line[1].split('\t')
        target = tf_list[i][1]
        story = source + [target]
        dev_data.append({'source': source, 'target': target, 'story': story})

    print('train: ', len(train_data))  # 90000
    print('dev: ', len(dev_data))  # 4081
    print('test: ', len(test_data))  # 4081

    json.dump(train_data, open(config["paths"]["prop_trn"], "w"), indent=4)
    json.dump(dev_data, open(config["paths"]["prop_dev"], "w"), indent=4)
    json.dump(test_data, open(config["paths"]["prop_tst"], "w"), indent=4)


def parse_one(text, client, extractor):
    '''
    transfer a sentence into events
    :param text:
    :return:
    '''
    text = text.strip()
    # parsed_e = aser_extractor.extract_from_text(text, in_order=True, use_lemma=True)
    parsed_e = extractor.extract_eventualities_from_text(text, in_order=True, use_lemma=True)

    events = []
    if parsed_e is [[]]:
        events = [text]
    else:
        for s in parsed_e:
            for e in s:
                events.append(' '.join(e.words))
        if events == []:
            events = [text]
    return events


def get_story_event():
    client = ASERClient(port=8000, port_out=8001)

    aser_extractor = DiscourseASERExtractor(
        corenlp_path=sys.argv[1], corenlp_port=9000
    )

    paths = [config["paths"]["prop_tst"], config["paths"]["prop_dev"], config["paths"]["prop_trn"]]
    for path in paths:
        data = json.load(open(path, 'r'))
        new_data = []
        for idx, item in tqdm(enumerate(data), total=len(data)):
            story = item['story']
            story_events = []  # each item contains the events of each utterance
            for utterance in story:
                # for i in punctuation_string:
                #     utterance = utterance.replace(i, '')
                u_e = parse_one(utterance, client=client, extractor=aser_extractor)
                story_events.append(u_e)
            item['events'] = story_events
            new_data.append(item)
        json.dump(new_data, open(path, 'w'), indent=4)


def simplify_dialogue_event():
    paths = [config["paths"]["prop_tst"], config["paths"]["prop_dev"], config["paths"]["prop_trn"]]
    for path in paths:
        data = json.load(open(path, 'r'))
        new_data = []
        for i, item in enumerate(data):
            story_events = item['events']  # each item contains the events of each utterance
            story_sim_events = []
            for u_events in story_events:
                new_u_events = []
                for event in u_events:
                    new_event = []
                    for w in event.split():
                        if w not in stopwords:
                            new_event.append(w)
                    new_u_events.append(' '.join(new_event))
                story_sim_events.append(new_u_events)
            item['sim_events'] = story_sim_events
            new_data.append(item)
        json.dump(new_data, open(path, 'w'), indent=4)



def extract_event_pairs():
    test_data = json.load(open(config["paths"]["prop_tst"], 'r'))
    with open(config["paths"]["test_event_pairs"], 'w') as f:
        for item in test_data:
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

    dev_data = json.load(open(config["paths"]["prop_dev"], 'r'))
    with open(config["paths"]["dev_event_pairs"], 'w') as f:
        for item in dev_data:
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

    train_data = json.load(open(config["paths"]["prop_trn"], 'r'))
    with open(config["paths"]["train_event_pairs"], 'w') as f:
        for item in train_data:
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




def retrieve_event_from_context():
    doc = []
    doc_te = []
    doc_r = []
    atomic_dev = open('../../../data/atomic/event_triples/dev_event_triples.txt', 'r').readlines()
    atomic_train = open('../../../data/atomic/event_triples/train_event_triples.txt', 'r').readlines()
    atomic_test = open('../../../data/atomic/event_triples/test_event_triples.txt', 'r').readlines()
    for split in [atomic_train, atomic_dev, atomic_test]:
        for line in split:
            he, r, te = line.split('\t')
            he_words = he.strip().split(' ')
            if '' in he_words:
                he_words.remove('')
            doc.append(he_words)
            doc_te.append(te)
            doc_r.append(r)
    s = BM25(doc)
    print('finish bm25')

    paths = [config["paths"]["prop_tst"], config["paths"]["prop_dev"], config["paths"]["prop_trn"]]
    for path in paths:
        data = json.load(open(path, 'r'))
        new_data = []
        for i, item in tqdm(enumerate(data), total=len(data)):
            context = item["story"][:4]
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
            item['retrieve_event'] = retrieve_kg
            new_data.append(item)
        json.dump(new_data, open(path, 'w'), indent=4)


if __name__ == '__main__':
    # Step 1: read raw csv files.
    read_from_raw()

    # Step 2: parse each utterance into events.
    get_story_event()

    # Step 3: remove stopwords from events.
    simplify_dialogue_event()

    # Step 4: extract event pairs from dataset for to obtain predicted relation from BERT-based relation classifier.
    extract_event_pairs()

    # Step 5: retrieve event from atomic using bm25 (one baseline)
    retrieve_event_from_context()


