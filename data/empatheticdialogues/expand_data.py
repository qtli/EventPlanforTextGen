import json
import pdb


special_tokens = ['<SEP>', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant']


def add_pred_path():
    dev = json.load(open('ed_train_prop.json', 'r'))
    pred = open('result_train.txt', 'r')

    count = 0
    new_dev = []
    for line in pred.readlines():
        if line.startswith('text prediction: '):
            pred_path_list = []
            item = ''
            pred_path = line.strip('text prediction: ').strip('\n').split(' ')
            for w in pred_path:
                if w in special_tokens:
                    if item != '':
                        pred_path_list.append(item.strip())
                        item = ''
                    pred_path_list.append(w)
                else:
                    item += w
                    item += ' '
                # print('after')
                # pdb.set_trace()
            if item != '':
                pred_path_list.append(item.strip())

            data = dev[count]
            data["pred_kg_path"] = pred_path_list
            new_dev.append(data)
            count += 1

    print('old: ', len(dev))
    print('new: ', len(new_dev))
    # print(data["pred_kg_path"])
    # pdb.set_trace()
    assert len(dev) == len(new_dev)
    json.dump(new_dev, open('ed_train_prop_exp.json', 'w'), indent=4)


def add_pred_path_relation_ids():
    data = json.load(open('ed_train_prop_exp.json', 'r'))
    new_data = []
    for item in data:
        pred_rel_ids = []
        last_rel = ""
        path = item['pred_kg_path']
        response_rel_ids = item['context_response_path_relation_ids'][1]
        for idx, er in enumerate(path):
            if er in special_tokens:
                pred_rel_ids.append(er)
                last_rel = er
            elif last_rel != '':
                pred_rel_ids.append(last_rel)
            else:
                pred_rel_ids.append(response_rel_ids[0])

        assert len(path)==len(pred_rel_ids)
        item['pred_kg_relation_ids']=pred_rel_ids
        new_data.append(item)

        # pdb.set_trace()

    json.dump(new_data, open('ed_train_prop_exp.json', 'w'),  indent=4)


if __name__ == '__main__':
    # add_pred_path()
    add_pred_path_relation_ids()


