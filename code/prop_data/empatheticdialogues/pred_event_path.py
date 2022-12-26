import json
import configparser
import ast
import pickle
config = configparser.ConfigParser()
config.read("paths.cfg")

special_tokens = ['<SEP>', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant']

id2r = {0: 'xAttr', 1: 'xEffect', 2: 'xIntent', 3: 'xNeed', 4: 'xReact', 5: 'xWant',
         6: 'oEffect', 7: 'oReact', 8: 'oWant', 9: '_xAttr', 10: '_xEffect', 11: '_xIntent',
         12: '_xNeed', 13: '_xReact', 14: '_xWant', 15: '_oEffect', 16: '_oReact', 17: '_oWant'}

def add_pred_relation(split):
    if split == 'tst':
        split_data = json.load(open(config["paths"]["prop_tst"], 'r'))
        pred = open(config["paths"]["dev_pred_relations"], 'rb')
    elif split == 'dev':
        split_data = json.load(open(config["paths"]["prop_dev"], 'r'))
        pred = open(config["paths"]["dev_pred_relations"], 'rb')
    else:
        split_data = json.load(open(config["paths"]["prop_trn"], 'r'))
        pred = open(config["paths"]["trn_pred_relations"], 'rb')

    new_data = []
    idx = 0
    '''
    "event_transition_path": [
            [
                "my friend leave",
                relation,
                "she be housesitt for i",
                relation,
                "my cat snuck out",
                relation,
                "my cat be kill"
            ],
            [
                relation,
                "that isnt good be mad at friend"
            ],
            [   relation,
                "be you go to get another cat"
            ]
        ]
    '''
    for item in split_data:
        event = item['events']
        tmp = []
        for i, es in enumerate(event):  # iterate each utterance's events
            tmp_es = []  # include this utterance's events and relations
            for j, e in enumerate(es):  # iterate this utterance's events
                if i==0 and j==0:  # if it is the first utterance's first event
                    tmp_es.append(e)
                else:
                    tmp_es.append(id2r[int(pred[idx])])
                    tmp_es.append(e)
                    idx +=1
            tmp.append(tmp_es)
        item['event_path'] = tmp
        new_data.append(item)

    if split == 'tst':
        json.dump(new_data, open(config["paths"]["prop_tst"], 'w'), indent=4)
    elif split == 'dev':
        json.dump(new_data, open(config["paths"]["prop_dev"], 'w'), indent=4)
    else:
        json.dump(new_data, open(config["paths"]["prop_trn"], 'w'), indent=4)


def add_path_relation_ids():
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


def add_pred_path(split='tst'):
    if split == 'tst':
        split_data = json.load(open(config["paths"]["prop_tst"], 'r'))
        pred = open(config["paths"]["test_pred_paths"], 'r')
    elif split == 'dev':
        split_data = json.load(open(config["paths"]["prop_dev"], 'r'))
        pred = open(config["paths"]["dev_pred_paths"], 'r')
    else:
        split_data = json.load(open(config["paths"]["prop_trn"], 'r'))
        pred = open(config["paths"]["trn_pred_paths"], 'r')

    count = 0
    new_data = []
    for line in pred.readlines():
        if line.startswith('beam text prediction: '):
            pred_path_list = []
            beams_pred_path = line.strip('beam text prediction: ').strip('\n').strip()
            pred_list = ast.literal_eval(beams_pred_path)
            for beam_pred in pred_list:
                beam_pred_path_list = []
                beam_item = ''
                for w in beam_pred.strip().split():
                    if w in special_tokens:
                        if beam_item != '':
                            beam_pred_path_list.append(beam_item.strip())
                            beam_item = ''
                        beam_pred_path_list.append(w)
                    else:
                        beam_item += w
                        beam_item += ' '
                if beam_item != '':
                    beam_pred_path_list.append(beam_item.strip())
                pred_path_list.append(beam_pred_path_list)
            data = split_data[count]
            data["pred_event_path"] = pred_path_list
            new_data.append(data)
            count += 1

    assert len(split_data) == len(new_data)
    if split == 'tst':
        json.dump(new_data, open(config["paths"]["prop_tst"], 'w'), indent=4)
    elif split == 'dev':
        json.dump(new_data, open(config["paths"]["prop_dev"], 'w'), indent=4)
    else:
        json.dump(new_data, open(config["paths"]["prop_trn"], 'w'), indent=4)


def add_pred_path_relation_ids(split):
    if split == 'tst':
        split_data = json.load(open(config["paths"]["prop_tst"], 'r'))
    elif split == 'dev':
        split_data = json.load(open(config["paths"]["prop_dev"], 'r'))
    else:
        split_data = json.load(open(config["paths"]["prop_trn"], 'r'))

    new_data = []
    for item in split_data:
        paths = item['pred_event_path']
        response_rel_ids = item['context_ending_path_relation_ids'][1]
        beam_pred_rel_ids = []
        for beam_idx, path in enumerate(paths):
            pred_rel_ids = []
            last_rel = ""

            for idx, er in enumerate(path):
                if er in special_tokens:
                    pred_rel_ids.append(er)
                    last_rel = er
                elif last_rel != '':
                    pred_rel_ids.append(last_rel)
                else:
                    pred_rel_ids.append(response_rel_ids[0])

            assert len(path)==len(pred_rel_ids)
            beam_pred_rel_ids.append(pred_rel_ids)

        item['pred_kg_relation_ids']=beam_pred_rel_ids
        new_data.append(item)

    if split == 'tst':
        json.dump(new_data, open(config["paths"]["prop_tst"], 'w'), indent=4)
    elif split == 'dev':
        json.dump(new_data, open(config["paths"]["prop_dev"], 'w'), indent=4)
    else:
        json.dump(new_data, open(config["paths"]["prop_trn"], 'w'), indent=4)


if __name__ == '__main__':
    # Step 1: predict relations between event pairs using BERT-based relation classifier.

    # Step 1: add predicted relations to complete event transition paths.
    add_pred_relation(split='tst')
    add_pred_relation(split='dev')
    add_pred_relation(split='trn')

    # Step 2:
    add_path_relation_ids(split='tst')


    # Step 3: predict following event transiton paths given path corresponding to input context.
    add_pred_path(split='tst')
    add_pred_path(split='dev')
    add_pred_path(split='trn')


    # Step 4: prepare relation embedding ids for each complete path
    add_pred_path_relation_ids(split='tst')