import json
import ast

special_tokens = ['<SEP>', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant']


def for_ed():
    a=json.load(open('ed_test_prop_exp.json', 'r'))
    case_ids = [4, 9, 45, 25, 28, 31, 32, 33, 37, 39, 40, 68, 95, 128, 140, 165, 178, 206, 214, 255, 399, 400, 416, 449,
                459, 489, 493, 510, 590, 648, 764, 765, 765, 820, 885, 907, 937, 945, 962, 1000, 1010, 1094, 1119, 1123,
                1147, 1188, 1235, 1354, 1374, 1441]

    prediction = open('p_for_human.txt','r')
    all_pred = []
    all_pred_relations = []
    count = 0
    for line in prediction.readlines():
        if line.startswith('beam text prediction:'):
            line = line.lstrip('beam text prediction:').strip('\n')
            beams = ast.literal_eval(line)
            # all_pred.append(beams)

            beam_pred_path_list = []
            beam_pred_path_rel_list = []
            for b in beams:
                pred_path_list = []
                item = ''
                for w in b.split(' '):
                    if w in special_tokens:
                        if item != '':
                            pred_path_list.append(item.strip())
                            item = ''
                        pred_path_list.append(w)
                    else:
                        item += w
                        item += ' '
                if item != '':
                    pred_path_list.append(item.strip())
                beam_pred_path_list.append(pred_path_list)

                pred_rel_ids = []
                last_rel = ""
                item = a[case_ids[count]]
                response_rel_ids = item['context_response_path_relation_ids'][1]
                for idx, er in enumerate(pred_path_list):
                    if er in special_tokens:
                        pred_rel_ids.append(er)
                        last_rel = er
                    elif last_rel != '':
                        pred_rel_ids.append(last_rel)
                    else:
                        pred_rel_ids.append(response_rel_ids[0])
                beam_pred_path_rel_list.append(pred_rel_ids)

            all_pred.append(beam_pred_path_list)
            all_pred_relations.append(beam_pred_path_rel_list)
            count += 1


    newa = []
    for idx, item in enumerate(a):
        if idx not in case_ids:
            newa.append(item)
        else:
            pred_paths = all_pred[case_ids.index(idx)]
            pred_paths_relation = all_pred_relations[case_ids.index(idx)]
            item['beam_pred_kg_path'] = pred_paths
            item['beam_pred_kg_relation_ids'] = pred_paths_relation
            newa.append(item)
    json.dump(newa, open('ed_test_prop_exp_human.json','w'), indent=4)




def for_roc():
    a = json.load(open('test_prop_exp.json', 'r'))
    case_ids = [2, 5, 15, 14, 28, 34, 40, 42, 97, 123, 190, 251, 272, 292, 358, 3210, 2611, 2504, 2466, 1941, 1912,
                1860, 1836, 1690, 1679, 1662, 1647, 1636, 1633, 1628, 1616, 1614, 1605, 1557, 1554, 1539, 1525, 1503,
                1500, 1486, 1484, 1480, 1469, 1450, 1432, 1417, 1400, 1398, 1337, 1323]

    prediction = open('p_for_human.txt', 'r')
    all_pred = []
    all_pred_relations = []
    count = 0
    for line in prediction.readlines():
        if line.startswith('beam text prediction:'):
            line = line.lstrip('beam text prediction:').strip('\n')
            beams = ast.literal_eval(line)
            # all_pred.append(beams)

            beam_pred_path_list = []
            beam_pred_path_rel_list = []
            for b in beams:
                pred_path_list = []
                item = ''
                for w in b.split(' '):
                    if w in special_tokens:
                        if item != '':
                            pred_path_list.append(item.strip())
                            item = ''
                        pred_path_list.append(w)
                    else:
                        item += w
                        item += ' '
                if item != '':
                    pred_path_list.append(item.strip())
                beam_pred_path_list.append(pred_path_list)

                pred_rel_ids = []
                last_rel = ""
                item = a[case_ids[count]]
                response_rel_ids = item['context_response_path_relation_ids'][1]
                for idx, er in enumerate(pred_path_list):
                    if er in special_tokens:
                        pred_rel_ids.append(er)
                        last_rel = er
                    elif last_rel != '':
                        pred_rel_ids.append(last_rel)
                    else:
                        pred_rel_ids.append(response_rel_ids[0])
                beam_pred_path_rel_list.append(pred_rel_ids)

            all_pred.append(beam_pred_path_list)
            all_pred_relations.append(beam_pred_path_rel_list)
            count += 1

    newa = []
    for idx, item in enumerate(a):
        if idx not in case_ids:
            newa.append(item)
        else:
            pred_paths = all_pred[case_ids.index(idx)]
            pred_paths_relation = all_pred_relations[case_ids.index(idx)]
            item['beam_pred_kg_path'] = pred_paths
            item['beam_pred_kg_relation_ids'] = pred_paths_relation
            newa.append(item)
    json.dump(newa, open('test_prop_exp_human.json', 'w'), indent=4)


if __name__ == '__main__':
    for_ed()