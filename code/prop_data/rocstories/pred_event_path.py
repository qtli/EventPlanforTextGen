import json
import pdb
import ast

special_tokens = ['<SEP>', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant']


def add_pred_path(split):
    dev = json.load(open('prop/' + split + '_prop.json', 'r'))
    pred = open('prop/result_' + split + '.txt', 'r')

    count = 0
    new_dev = []
    for line in pred.readlines():
        if line.startswith('beam text prediction: '):
            pred_path_list = []
            item = ''
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

            data = dev[count]
            data["pred_kg_path"] = pred_path_list
            new_dev.append(data)
            count += 1

    print('old: ', len(dev))
    print('new: ', len(new_dev))
    assert len(dev) == len(new_dev)
    json.dump(new_dev, open('prop/' + split + '_prop_expppp.json', 'w'), indent=4)


def add_pred_path_relation_ids(split):
    data = json.load(open('prop/' + split + '_prop_expppp.json', 'r'))
    new_data = []
    for item in data:
        paths = item['pred_kg_path']
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

    json.dump(new_data, open('prop/' + split + '_prop_expppp.json', 'w'),  indent=4)


if __name__ == '__main__':
    add_pred_path(split='test')
    add_pred_path(split='train')
    add_pred_path(split='dev')

    add_pred_path_relation_ids(split='test')
    add_pred_path_relation_ids(split='train')
    add_pred_path_relation_ids(split='dev')



