import pickle
import json
import pdb

with open('my_dataset_preproc.p', 'rb') as f:
    [data_tra, data_val, data_tst, vocab] = pickle.load(f)


train_data = []
for i in range(len(data_tra['context'])):
    item = {}
    context = []
    for c in data_tra['context'][i]:
        context.append(' '.join(c).strip())
    item['context'] = context
    item['response'] = ' '.join(data_tra['target'][i]).strip()
    kg_words = []
    kg_words_emotion = []
    for si, skg in enumerate(data_tra['concepts'][i]):
        for ti, tkg in enumerate(skg[0]):
            for wi, w in enumerate(tkg):
                if w not in kg_words:
                    kg_words.append(w)
                    if w in item['response']:
                        kg_words_emotion.append(data_tra['concepts'][i][si][2][ti][wi] + 1)
                    else:
                        kg_words_emotion.append(data_tra['concepts'][i][si][2][ti][wi])
    assert len(kg_words) == len(kg_words_emotion)

    if kg_words_emotion != []:
        new_kg_words = list(zip(*sorted(zip(kg_words_emotion, kg_words), reverse=True)))[1]
    else:
        new_kg_words = []

    item['ekg'] = new_kg_words

    train_data.append(item)
json.dump(train_data, open('kemp_data/kemp_train.json', 'w'), indent=4)


train_data = []
for i in range(len(data_val['context'])):
    item = {}
    context = []
    for c in data_val['context'][i]:
        context.append(' '.join(c).strip())
    item['context'] = context
    item['response'] = ' '.join(data_val['target'][i]).strip()
    kg_words = []
    kg_words_emotion = []
    for si, skg in enumerate(data_val['concepts'][i]):
        for ti, tkg in enumerate(skg[0]):
            for wi, w in enumerate(tkg):
                if w not in kg_words:
                    kg_words.append(w)
                    if w in item['response']:
                        kg_words_emotion.append(data_val['concepts'][i][si][2][ti][wi] + 1)
                    else:
                        kg_words_emotion.append(data_val['concepts'][i][si][2][ti][wi])
    assert len(kg_words) == len(kg_words_emotion)
    if kg_words_emotion != []:
        new_kg_words = list(zip(*sorted(zip(kg_words_emotion, kg_words), reverse=True)))[1]
    else:
        new_kg_words = []
    item['ekg'] = new_kg_words

    train_data.append(item)
json.dump(train_data, open('kemp_data/kemp_dev.json', 'w'), indent=4)


train_data = []
for i in range(len(data_tst['context'])):
    item = {}
    context = []
    for c in data_tst['context'][i]:
        context.append(' '.join(c).strip())
    item['context'] = context
    item['response'] = ' '.join(data_tst['target'][i]).strip()
    kg_words = []
    kg_words_emotion = []
    for si, skg in enumerate(data_tst['concepts'][i]):
        for ti, tkg in enumerate(skg[0]):
            for wi, w in enumerate(tkg):
                if w not in kg_words:
                    kg_words.append(w)
                    if w in item['response']:
                        kg_words_emotion.append(data_tst['concepts'][i][si][2][ti][wi] + 1)
                    else:
                        kg_words_emotion.append(data_tst['concepts'][i][si][2][ti][wi])
    assert len(kg_words) == len(kg_words_emotion)

    if kg_words_emotion != []:
        new_kg_words = list(zip(*sorted(zip(kg_words_emotion, kg_words), reverse=True)))[1]
    else:
        new_kg_words = []

    item['ekg'] = new_kg_words

    train_data.append(item)
json.dump(train_data, open('kemp_data/kemp_test.json', 'w'), indent=4)



