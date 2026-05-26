import os
import json
from PROBE import PROBE
from typing import List
from collections import defaultdict

def get_e2id(data_path):
    with open(os.path.join(data_path, 'entities.dict'), encoding='utf-8') as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity
    global nentity
    nentity = len(entity2id)
    return entity2id, id2entity

def get_r2id(data_path):
    with open(os.path.join(data_path, 'relations.dict'), encoding='utf-8') as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation
    global nrelation
    nrelation = len(relation2id)
    return relation2id, id2relation
            
def get_triples(data_path, entity2id, relation2id):
    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    return train_triples, valid_triples, test_triples
    
def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def count_entities(train_triples):
    eid2count = {}
    for i in range(nentity):
        eid2count[i] = 0

    for _triple in train_triples:
        h, r, t = _triple
        eid2count[h] += 1
        eid2count[t] += 1
    count_info_dict = dict(sorted(eid2count.items(), key=lambda item : -item[1]))
    return count_info_dict

def count_relations(train_triples):
    rid2count = {}
    for i in range(nrelation):
        rid2count[i] = 0

    for _triple in train_triples:
        h, r, t = _triple
        rid2count[r] += 1
    count_info_dict = dict(sorted(rid2count.items(), key=lambda item: -item[1]))
    return count_info_dict

def get_ranks(rank_path):
    def convert_to_float(ls):
        return list(float(el) for el in ls)
    save_ranks = list()
    files = os.listdir(rank_path)
    
    for file in files:
        with open(rank_path+file, encoding='utf-8') as f:
            load_ranks = json.load(f)
            load_ranks = dict(sorted(load_ranks.items()))
        save_ranks.append({int(k):convert_to_float(v) for k, v in load_ranks.items()})
        
    return save_ranks

def aggregate_performance(_mets : List[PROBE], alpha):
    A = 1 / (1 - (1 / nentity) ** alpha)
    result = defaultdict(list)
    for met in _mets:
        for k, ls in met.raw_ranks.items():
            result[k] += ls
    return {k:sum([A * ((1 / r) ** alpha - 1) + 1 for r in v_ls])/len(v_ls) for k,v_ls in result.items()}

def get_filters(model_name, dataset):
    folder = '../filterings'
    file = f'{dataset}_filt.json'
    if model_name == 'TuckER': file = f'{model_name}_{file}'
    with open(os.path.join(folder, file), encoding='utf-8') as f:
        filters = json.load(f)
    return filters

def get_infos(model_name, dataset):
    path = f'../infos/{model_name}/{dataset}'
    info_ls = []
    files = os.listdir(path)
    for file in files:
        if 'track' == file: continue
        with open(os.path.join(path, file), encoding='utf-8') as f:
            ls = json.load(f)
        info_ls.append(ls)
    return info_ls

def get_rel_distribution(triples, len_r):
    e2Rdistri = {}
    for triple in triples:
        h, r, t = triple
        if h not in e2Rdistri:
            e2Rdistri[h] = [0] * len_r
        if t not in e2Rdistri:
            e2Rdistri[t] = [0] * len_r
        e2Rdistri[h][r] += 1
        e2Rdistri[t][r] += 1
    for e, cnt_distri in e2Rdistri.items():
        agg = sum(cnt_distri)
        if agg:
            e2Rdistri[e] = [r_p / agg for r_p in cnt_distri]
    return e2Rdistri

def get_ship(path):
    result = list()
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            result.append(json.loads(line.strip()))
    return result

def get_decoded(path):
    decode_map = None
    with open(path, 'r') as f:
        decode_map = json.load(f)
    return decode_map