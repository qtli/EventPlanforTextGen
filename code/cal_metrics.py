import collections
import math
import ast
import pdb
import os

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


relation_list = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
                  '_oEffect', '_oReact', '_oWant', '_xAttr', '_xEffect', '_xIntent', '_xNeed', '_xReact', '_xWant', 'PRP',]

def cal_one_model(file):
    print(file)
    with open(file, 'r') as f:
        target = []
        pred = []

        res = {}
        itr = 0
        # for line in f.readlines():
        #     if line.startswith('target:'):
        #         t = line.strip('target: ').strip()
        #     # if line.startswith('beam text prediction: '):
        #     #     ps = line.strip('beam text prediction: ').strip()
        #     #     p_list = ast.literal_eval(ps)
        #     #     p = p_list[0].strip()
        #     if line.startswith('text prediction:'):
        #         p = line.strip('text prediction:').strip()
        #         target.append([t.split()])  # [['_xEffect', 'you', 'suffer', 'any', 'injury']]
        #         pred.append(p.split())
        #
        #         res[itr] = p.split()
        #         itr += 1
        for line in f.readlines():
            if line.startswith('target:'):
                t = line.strip('target: ').strip()
            # if line.startswith('beam text prediction: '):
            #     ps = line.strip('beam text prediction: ').strip()
            #     p_list = ast.literal_eval(ps)
            #     p = p_list[0].strip()
            if line.startswith('text prediction:'):
                p = line.strip('text prediction:').strip()
                tls = t.split()
                new_tls = []
                for t in tls:
                    if t not in relation_list:
                        new_tls.append(t)
                target.append([new_tls])
                pls = p.split()
                new_pls = []
                for p in pls:
                    if p not in relation_list:
                        new_pls.append(p)
                pred.append(new_pls)
                # pdb.set_trace()
                res[itr] = pls
                itr += 1
        bleu1 = _compute_bleu(target, pred, max_order=1)
        bleu2 = _compute_bleu(target, pred, max_order=2)
        bleu4 = _compute_bleu(target, pred, max_order=4)
        result = {
            "blue1": round(bleu1[0]*100,2),
            "blue2": round(bleu2[0]*100,2),
            "bleu4": round(bleu4[0]*100,2),
        }

    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)
    print("Dist-1\tDist-2")
    print(
        "{:.2f}\t{:.2f}".format(mi_dist1 * 100, mi_dist2 * 100))
    print(result)
    print('\n\n')


def cal_one_model_path(file):
    print(file)
    with open(file, 'r') as f:
        target = []
        pred = []

        res = {}
        itr = 0
        for line in f.readlines():
            if line.startswith('target:'):
                t = line.strip('target: ').strip()
            if line.startswith('text prediction:'):
                p = line.strip('text prediction:').strip()
                tls = t.split()
                target.append([tls])
                pls = p.split()
                pred.append(pls)
                res[itr] = pls
                itr += 1
        bleu1 = _compute_bleu(target, pred, max_order=1)
        bleu2 = _compute_bleu(target, pred, max_order=2)
        bleu4 = _compute_bleu(target, pred, max_order=4)
        result = {
            "blue1": round(bleu1[0]*100,2),
            "blue2": round(bleu2[0]*100,2),
            "bleu4": round(bleu4[0]*100,2),
        }

    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)
    print("Dist-1\tDist-2")
    print(
        "{:.2f}\t{:.2f}".format(mi_dist1 * 100, mi_dist2 * 100))
    print(result)
    print('\n\n')


def cal_one_model_moel(file):
    print(file)
    with open(file, 'r') as f:
        target = []
        pred = []

        res = {}
        itr = 0
        # for line in f.readlines():
        #     if line.startswith('target:'):
        #         t = line.strip('target: ').strip()
        #     # if line.startswith('beam text prediction: '):
        #     #     ps = line.strip('beam text prediction: ').strip()
        #     #     p_list = ast.literal_eval(ps)
        #     #     p = p_list[0].strip()
        #     if line.startswith('text prediction:'):
        #         p = line.strip('text prediction:').strip()
        #         target.append([t.split()])  # [['_xEffect', 'you', 'suffer', 'any', 'injury']]
        #         pred.append(p.split())
        #
        #         res[itr] = p.split()
        #         itr += 1
        for line in f.readlines():
            if line.startswith('Pred:'):
                p = line.strip('Pred:').strip()
            # if line.startswith('beam text prediction: '):
            #     ps = line.strip('beam text prediction: ').strip()
            #     p_list = ast.literal_eval(ps)
            #     p = p_list[0].strip()
            if line.startswith('Ref:'):
                t = line.strip('Ref:').strip()
                tls = t.split()
                new_tls = []
                for t in tls:
                    if t not in relation_list:
                        new_tls.append(t)
                target.append([new_tls])
                pls = p.split()
                new_pls = []
                for p in pls:
                    if p not in relation_list:
                        new_pls.append(p)
                pred.append(new_pls)
                # pdb.set_trace()
                res[itr] = pls
                itr += 1
        bleu1 = _compute_bleu(target, pred, max_order=1)
        bleu2 = _compute_bleu(target, pred, max_order=2)
        bleu4 = _compute_bleu(target, pred, max_order=4)
        result = {
            "blue1": round(bleu1[0]*100,2),
            "blue2": round(bleu2[0]*100,2),
            "bleu4": round(bleu4[0]*100,2),
        }

    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)
    print("Dist-1\tDist-2")
    print(
        "{:.2f}\t{:.2f}".format(mi_dist1 * 100, mi_dist2 * 100))
    print(result)
    print('\n\n')


def cal_each_sample_bleu(dir, file, model_name=''):
    file = os.path.join(dir,file)
    print(file)
    with open(file, 'r') as f:
        # for line in f.readlines():
        #     if line.startswith('target:'):
        #         t = line.strip('target: ').strip()
        #     # if line.startswith('beam text prediction: '):
        #     #     ps = line.strip('beam text prediction: ').strip()
        #     #     p_list = ast.literal_eval(ps)
        #     #     p = p_list[0].strip()
        #     if line.startswith('text prediction:'):
        #         p = line.strip('text prediction:').strip()
        #         target.append([t.split()])  # [['_xEffect', 'you', 'suffer', 'any', 'injury']]
        #         pred.append(p.split())
        #
        #         res[itr] = p.split()
        #         itr += 1
        target = []
        pred = []
        bleu1_file = open(os.path.join(dir, model_name+'_'+'bleu1.txt'), 'w')
        print('write to : ', os.path.join(dir,'bleu1.txt'))
        for line in f.readlines():
            if line.startswith('target:'):
                t = line.strip('target: ').strip()
            # if line.startswith('beam text prediction: '):
            #     ps = line.strip('beam text prediction: ').strip()
            #     p_list = ast.literal_eval(ps)
            #     p = p_list[0].strip()
            if line.startswith('prediction:'):
                p = line.strip('prediction:').strip()
                tls = t.split()
                new_tls = []
                for t in tls:
                    if t not in relation_list:
                        new_tls.append(t)
                target = [[new_tls]]
                pls = p.split()
                new_pls = []
                for p in pls:
                    if p not in relation_list:
                        new_pls.append(p)
                pred = [new_pls]
                if target != [] and pred != []:
                    bleu1 = _compute_bleu(target, pred, max_order=1)
                    assert len(target)==1 and len(pred)==1 and len(bleu1) == 6
                    bleu1 = round(bleu1[0]*100,2)
                    bleu1_file.write(str(bleu1)+'\n')
                    target = []
                    pred = []



if __name__ == '__main__':
    # knowledge generator
    pt_then_ft = 'knowledge_generator/result/rocstory/pt_then_ft/seq_rel_emb/result_ep:test.txt'
    direct_ft = 'knowledge_generator/result/rocstory/direct_ft/seq_rel_emb/result_ep:test.txt'
    prefix_pt_ft = 'knowledge_generator/result/rocstory/prefix_pt_ft/result_ep:test.txt'

    # ed knowledge generator
    ed_direct_ft = 'prefix_knowledge_generator/result/empatheticdialogue/direct_ft/result_ep:test.txt'
    ed_prefix_ft = 'prefix_knowledge_generator/result/empatheticdialogue/prefix_pt_ft/result_ep:test.txt'
    ed_pt_then_ft = 'prefix_knowledge_generator/result/empatheticdialogue/pt_then_ft/result_ep:test.txt'

    # cal_one_model_path('prefix_knowledge_generator/result/rocstory/pt_then_ft/seq_rel_emb/result_ep:test.txt')
    # cal_one_model_path('prefix_knowledge_generator/result/rocstory/pt_then_ft/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/rocstory/retrieve/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/rocstory/direct_ft/seq_rel_emb/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/rocstory/only_pt/result_ep:test.txt')

    # cal_one_model('prefix_knowledge_generator/result/empatheticdialogue/direct_ft/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/empatheticdialogue/prefix_pt_ft/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/empatheticdialogue/only_pt/result_ep:test.txt')
    cal_one_model_path('prefix_knowledge_generator/result/empatheticdialogue/only_pt/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/empatheticdialogue/retrieve/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/empatheticdialogue/prefix_ft1/result_ep:test.txt')
    # cal_one_model('prefix_knowledge_generator/result/rocstory/prefix_ft/result_ep:test.txt')

    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/gpt2_ft/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/gpt2_ft/result_ep:test_1.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/ed_query_update_wo_rid_embed_ep50/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/ed_query_update_wo_rid_embed_ep50_lre-4/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/ed_query_update_wo_rid_embed_ep50_lre-4/result_ep:173.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/ed_query_update_wo_rid_embed_ep50_lre-4/result_ep:346.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/ed_last_layer_update_wo_rid_embed/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/pt_ft/result_ep:test.txt')


    # cal_one_model('mymodel/TK_GEN/empatheticdialogue/ret/result_ep:test.txt')
    # cal_one_model_moel('mymodel/TK_GEN/empatheticdialogue/empdg/EmpDG.txt')


    # cal_one_model('mymodel/TK_GEN/rocstory/gpt2_ft/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/rocstory/part_update30/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/rocstory/ret/result_ep:test.txt')
    # cal_one_model('mymodel/TK_GEN/rocstory/ret/result_ep:test0.txt')
    # cal_one_model('mymodel/TK_GEN/rocstory/pt_ft/result_ep:test.txt')



    # cal_each_sample_bleu(dir='mymodel/TK_GEN/rocstory/gpt2_ft/',file='result_ep:test.txt', model_name='gpt2')
    # cal_each_sample_bleu(dir='text_knowledge_generators_fast/TK_GEN/rocstory/part_update30', file='result_ep:test.txt', model_name='mymodel')
    # cal_each_sample_bleu(dir='mymodel/TK_GEN/rocstory/ret', file='result_ep:test.txt', model_name='ret')
    # cal_each_sample_bleu(dir='mymodel/TK_GEN/empatheticdialogue/gpt2_ft', file='result_ep:test.txt', model_name='edgpt2')
    # cal_each_sample_bleu(dir='mymodel/TK_GEN/empatheticdialogue/ed_query_update_wo_rid_embed_ep50_lre-4', file='result_ep:346.txt', model_name='edmymodel')
    # cal_each_sample_bleu(dir='mymodel/TK_GEN/empatheticdialogue/ret/',file='result_ep:test.txt', model_name='edret')
    # cal_each_sample_bleu(dir='mymodel/TK_GEN/empatheticdialogue/pt_ft/', file='result_ep:test.txt', model_name='pt_ft')



    # a = _compute_bleu([[['o','love','you']]], [['you']], max_order=1)
    # print(round(a[0]*100,2))
'''
roc - knowledge generator

>>direct_ft
knowledge_generator/result/rocstory/direct_ft/seq_rel_emb/result_ep:test_beam.txt
Dist-1	Dist-2
6.16	20.54
{'blue1': 0.1525317644113413, 'blue2': 0.06872166726995245, 'bleu4': 0.01449537909344123}


finetune_knowledge_generator/result/rocstory/direct_ft/seq_rel_emb/result_ep:test.txt
Dist-1	Dist-2
6.16	20.37
{'blue1': 0.16861738749452337, 'blue2': 0.08022877012236009, 'bleu4': 0.017127621234371504}


>>pt_then_ft
finetune_knowledge_generator/result/rocstory/pt_then_ft/seq_rel_emb/result_ep:test.txt (是否用2ep atomic 重新跑一下？)
Dist-1	Dist-2
7.25	21.81
{'blue1': 0.1602349163941425, 'blue2': 0.07719865823323754, 'bleu4': 0.012856747368973494}


>>prefix_pt_ft
knowledge_generator/result/rocstory/prefix_pt_ft/result_ep:test.txt
Dist-1	Dist-2
5.83	17.48
{'blue1': 0.19515265974189486, 'blue2': 0.09017222751508834, 'bleu4': 0.013525633238000755}

'''

'''
roc - mymodel

gpt2_ft
text_knowledge_generators_fast/TK_GEN/rocstory/gpt2_ft/result_ep:test.txt
Dist-1	Dist-2
12.12	33.42
{'blue1': 0.14590188743244553, 'blue2': 0.06413219858221902, 'bleu4': 0.01886832204323029}


overall model
text_knowledge_generators_fast/TK_GEN/rocstory/part_update30/result_ep:test.txt  (THIS)
Dist-1	Dist-2
12.69	36.22
{'blue1': 0.15185733907693616, 'blue2': 0.06537334739681468, 'bleu4': 0.018677050063298038}

'''


'''
ed - knowledge generator：

direct_ft
Dist-1	Dist-2
1.57	4.18
{'blue1': 0.23437633703186062, 'blue2': 0.11500288338286502, 'bleu4': 0.03314755025290194}

pt_then_ft
Dist-1	Dist-2
1.80	5.13
{'blue1': 0.2358106273196343, 'blue2': 0.11858024383827194, 'bleu4': 0.0358957054237657}

ed_prefix_ft
Dist-1	Dist-2
1.18	2.55
{'blue1': 0.26526128715246744, 'blue2': 0.12382571957058863, 'bleu4': 0.032969524843951346} 
'''


'''
ed - mymodel

gpt-ft:
Dist-1	Dist-2
4.05	11.44
{'blue1': 0.08480851688014925, 'blue2': 0.033520089203510656, 'bleu4': 0.009111872981211482}


mymodel/TK_GEN/empatheticdialogue/ed_query_update_wo_rid_embed_ep50/result_ep:test.txt (THIS)
Dist-1	Dist-2
3.57	9.75
{'blue1': 0.0850793183381757, 'blue2': 0.03382143286031333, 'bleu4': 0.009365835764097682}

'''


