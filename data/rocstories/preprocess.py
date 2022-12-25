import json
import csv
import re
import configparser
import string
from aser.client import ASERClient
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
punctuation_string = string.punctuation  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

config = configparser.ConfigParser()
config.read("paths.cfg")

'''
Please refer to: https://hkust-knowcomp.github.io/ASER/html/tutorial/get-started.html
run:
aser-server -n_workers 1 -n_concurrent_back_socks 10 -port 8000 -port_out 8001 -corenlp_path "YOUR_DIRECTORY/stanford-corenlp-3.9.2" -base_corenlp_port 9000
'''
client = ASERClient(port=8000, port_out=8001)


def punc(s):
    s = s.replace('_comma_', ',')
    tmp = re.sub(r'([a-zA-Z])([,.!:;?])', r'\1 \2', s)
    s = re.sub(r'([,.!])([a-zA-Z])', r'\1 \2', tmp)
    s = s.strip()
    return s

def read_from_raw():
    tf = csv.reader(open(config["paths"]["raw_tst"]))
    pcid = 'hit:0_conv:0'
    conv = {}
    conv['emotion'] = 'guilty'
    conv['situation'] = punc("I felt guilty when I was driving home one night and a person tried to fly into my lane_comma_ and didn't see me. I honked and they swerved back into their lane_comma_ slammed on their brakes_comma_ and hit the water cones.")
    conv['context'] = []
    test_data = []

    for i, line in enumerate(tf):
        if i == 0:
            continue
        cid = line[0]
        if cid == pcid:
            pcid = cid
            conv['context'].append(punc(line[5]))
        else:
            conv['context'] = conv['context']
            test_data.append(conv)
            conv = {}
            conv['emotion'] = line[2]
            conv['situation'] = punc(line[3])
            conv['context'] = [punc(line[5])]
            pcid = cid



    vf = csv.reader(open(config["paths"]["raw_dev"]))
    pcid = 'hit:3_conv:6'
    conv = {}
    conv['emotion'] = 'terrified'
    conv['situation'] = punc(
        "Today_comma_as i was leaving for work in the morning_comma_i had a tire burst in the middle of a busy road. That scared the hell out of me!")
    conv['context'] = []
    valid_data = []
    for i, line in enumerate(vf):
        if i == 0:
            continue
        cid = line[0]
        if cid == pcid:
            pcid = cid
            conv['context'].append(punc(line[5]))
        else:
            conv['context'] = conv['context']
            valid_data.append(conv)
            conv = {}
            conv['emotion'] = line[2]
            conv['situation'] = punc(line[3])
            conv['context'] = [punc(line[5])]
            pcid = cid

    trf = csv.reader(open(config["paths"]["raw_trn"]))
    pcid = 'hit:0_conv:1'
    conv = {}
    conv['emotion'] = 'sentimental'
    conv['situation'] = punc(
        "I remember going to the fireworks with my best friend. There was a lot of people_comma_ but it only felt like us in the world.")
    conv['context'] = []
    train_data = []
    for i, line in enumerate(trf):
        if i == 0:
            continue
        cid = line[0]
        if cid == pcid:
            pcid = cid
            conv['context'].append(punc(line[5]))
        else:
            conv['context'] = conv['context']
            train_data.append(conv)
            conv = {}
            conv['emotion'] = line[2]
            conv['situation'] = punc(line[3])
            conv['context'] = [punc(line[5])]
            pcid = cid

    print("====================================")
    max = 0
    t_data = []
    for t in test_data:
        for i, u in enumerate(t['context']):
            if (i+1)%2==0:
                new_t = t.copy()
                new_t['context'] = t['context'][:i]
                if len(new_t['context'])>max:
                    max = len(new_t['context'])
                new_t['response'] = t['context'][i]
                t_data.append(new_t)
    test_data = list(set(test_data))
    print('test: ', len(test_data))  # 2541
    json.dump(test_data, open(config["paths"]["prop_tst"],'w'), indent=4)

    v_data = []
    for t in valid_data:
        for i, u in enumerate(t['context']):
            if (i+1)%2==0:
                new_t = t.copy()
                new_t['context'] = t['context'][:i]
                if len(new_t['context'])>max:
                    max = len(new_t['context'])
                new_t['response'] = t['context'][i]
                v_data.append(new_t)
    valid_data = list(set(valid_data))
    print('valid: ', len(valid_data))  # 2762
    json.dump(valid_data, open(config["paths"]["prop_dev"], 'w'), indent=4)

    tr_data = []
    for t in train_data:
        for i, u in enumerate(t['context']):
            if (i+1)%2==0:
                new_t = t.copy()
                new_t['context'] = t['context'][:i]
                if len(new_t['context'])>max:
                    max = len(new_t['context'])
                new_t['response'] = t['context'][i]
                tr_data.append(new_t)
    train_data = list(set(train_data))
    print('train:', len(train_data))
    json.dump(train_data, open(config["paths"]["prop_trn"], 'w'), indent=4)

    print('test: ', len(t_data))  # 8398
    print('valid: ', len(v_data))
    print('train: ', len(tr_data))
    print(max)  # the maximize total utterance number of the context and response is 6


def parse_one(text):
    '''
    transfer a sentence into events
    :param text:
    :return:
    '''
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
    else:
        for s in parsed_e:
            for e in s:
                events.append(' '.join(e.words))
        if events == []:
            events = [text]
    return events


def get_dialogue_event():
    test_data = json.load(open(config["paths"]["prop_tst"], 'r'))
    new_data = []
    for i, item in enumerate(test_data):
        context = item['context']
        dialogue_events = []  # each item contains the events of each utterance
        for utterance in context:
            for i in punctuation_string:   # remove punctuation
                utterance = utterance.replace(i, '')
            u_e = parse_one(utterance)
            dialogue_events.append(u_e)
        response = item['response']
        u_r = parse_one(response)
        dialogue_events.append(u_r)
        item['events'] = dialogue_events
        new_data.append(item)
    json.dump(new_data, open(config["paths"]["prop_tst"], 'w'), indent=4)

    dev_data = json.load(open(config["paths"]["prop_dev"], 'r'))
    new_data = []
    for i, item in enumerate(dev_data):
        context = item['context']
        dialogue_events = []  # each item contains the events of each utterance
        for utterance in context:
            for i in punctuation_string:   # remove punctuation
                utterance = utterance.replace(i, '')
            u_e = parse_one(utterance)
            dialogue_events.append(u_e)
        response = item['response']
        u_r = parse_one(response)
        dialogue_events.append(u_r)
        item['events'] = dialogue_events
        new_data.append(item)
    json.dump(new_data, open(config["paths"]["prop_dev"], 'w'), indent=4)

    train_data = json.load(open(config["paths"]["prop_trn"], 'r'))
    new_data = []
    for i, item in enumerate(train_data):
        context = item['context']
        dialogue_events = []  # each item contains the events of each utterance
        for utterance in context:
            for i in punctuation_string:   # remove punctuation
                utterance = utterance.replace(i, '')
            u_e = parse_one(utterance)
            dialogue_events.append(u_e)
        response = item['response']
        u_r = parse_one(response)
        dialogue_events.append(u_r)
        item['events'] = dialogue_events
        new_data.append(item)
    json.dump(new_data, open(config["paths"]["prop_trn"], 'w'), indent=4)


def simplify_dialogue_event():
    test_data = json.load(open(config["paths"]["prop_tst"], 'r'))
    new_data = []
    for i, item in enumerate(test_data):
        dialogue_events = item['events']  # each item contains the events of each utterance
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
        item['sim_events'] = dialogue_sim_events
        new_data.append(item)
    json.dump(new_data, open(config["paths"]["prop_tst"], 'w'), indent=4)

    dev_data = json.load(open(config["paths"]["prop_dev"], 'r'))
    new_data = []
    for i, item in enumerate(dev_data):
        dialogue_events = item['events']  # each item contains the events of each utterance
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
        item['sim_events'] = dialogue_sim_events
        new_data.append(item)
    json.dump(new_data, open(config["paths"]["prop_dev"], 'w'), indent=4)

    train_data = json.load(open(config["paths"]["prop_trn"], 'r'))
    new_data = []
    for i, item in enumerate(train_data):
        dialogue_events = item['events']  # each item contains the events of each utterance
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
        item['sim_events'] = dialogue_sim_events
        new_data.append(item)
    json.dump(new_data, open(config["paths"]["prop_trn"], 'w'), indent=4)


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



if __name__ == '__main__':
    # Step 1: read raw csv files.
    read_from_raw()

    # Step 2: parse each utterance into events.
    get_story_event()

    # Step 3: remove stopwords from events.
    simplify_dialogue_event()

    # Step 4: extract event pairs from dataset for to obtain predicted relation from BERT-based relation classifier.
    extract_event_pairs()


