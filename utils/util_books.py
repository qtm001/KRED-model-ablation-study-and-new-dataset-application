import pandas as pd
import torch
import random
import numpy as np
import os
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from IPython.core.display import clear_output
from sentence_transformers import SentenceTransformer
import requests
import math
import zipfile
from tqdm import tqdm
import pickle
import shutil
import csv
import json
import gzip

# Create file and save data using pickle
def save_to_pickle(data, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp)



# Load data from file using pickle
def load_from_pickle(filename):
    with gzip.open(filename, "rb") as fp:
        data = pickle.load(fp)
    return data

def entities_books(config):
    entities = set()
    # Read entities from train books config["data"]["train_books"]=LinkedIn-books.csv
    with open(config['books']['wikidata_ids'], 'r', encoding='utf-8') as file:
        entities = set(line.strip() for line in file)
    return entities

# Return a dictionary from entity names to ids
def entity_to_id_books(config, entities):
    entity2id_dict = {}
    # Get the association entity-id from the file
    with open(config["books"]["entity_index"], encoding='utf-8') as fp:
        #entity_num = int(fp.readline().split('\n')[0])
        for line in fp:
            entity, entityid = line.strip().split('\t')
            if entity in entities:
                # Entity id is increased by one in order to be compatible with all the following operations
                entity2id_dict[entity] = int(entityid) + 1
    return entity2id_dict


# Return a dictionary from entity ids to names
def id_to_entity_books(config, ids):
    entity2id_dict = {}
    # Get the association entity-id from the file
    with open(config["books"]["entity_index"], encoding='utf-8') as fp:
        #entity_num = int(fp.readline().split('\n')[0])
        for line in fp:
            entity, entityid = line.strip().split('\t')
            # Since the entity ids are increased by one when reading from the file,
            # then it is also done here before the comparison
            if int(entityid) + 1 in ids:
                entity2id_dict[entity] = int(entityid) + 1
    return entity2id_dict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def construct_adj(graph_file, entity2id_file, args):  # graph is triple
    print('constructing adjacency matrix books ...')
    graph_file_fp = open(graph_file, 'r', encoding='utf-8')
    graph = []
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        if len(linesplit) > 1:
            graph.append([linesplit[0], linesplit[1], linesplit[2]])

    kg = {}
    for triple in graph:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    fp_entity2id = open(entity2id_file, 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])
    print(entity_num)
    entity_adj = []
    relation_adj = []
    for i in range(entity_num):
        entity_adj.append([])
        relation_adj.append([])
    for key in kg.keys():
        for index in range(args.entity_neighbor_num):
            i = random.randint(0, len(kg[key]) - 1)
            entity_adj[int(key)].append(int(kg[key][i][0]))
            relation_adj[int(key)].append(int(kg[key][i][1]))
    entity_adj = np.array(entity_adj)
    relation_adj = np.array(relation_adj)
    return entity_adj, relation_adj


def construct_embedding(entity_embedding_file, relation_embedding_file):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(entity_embedding_file, 'r', encoding='utf-8')
    fp_relation_embedding = open(relation_embedding_file, 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    for line in fp_relation_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        relation_embedding.append(linesplit)
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)


def my_collate_fn(batch):
    return batch


def construct_entity_dict(entity_file):
    fp_entity2id = open(entity_file, 'r', encoding='utf-8')
    entity_dict = {}
    entity_num_all = int(fp_entity2id.readline().split('\n')[0])
    lines = fp_entity2id.readlines()
    for line in lines:
        entity, entityid = line.strip().split('\t')
        entity_dict[entity] = entityid
    return entity_dict


def real_batch(batch):
    data = {}
    data['item1'] = []
    data['item2'] = []
    data['label'] = []
    for item in batch:
        data['item1'].append(item['item1'])
        data['item2'].append(item['item2'])
        data['label'].append(item['label'])
    return data


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)


def get_user2item_data_books(config):
    negative_num = config['trainer']['train_neg_num']
    train_data = {}
    user_id = []
    news_id = []
    label = []
    fp_train = open(config['books']['train_behavior'], 'r', encoding='utf-8')
    next(fp_train)
    for line in fp_train:
        if len(line.strip().split('\t')) == 3:
            userid, history, behavior = line.strip().split('\t')
            behavior = behavior.split(' ')
            positive_list = []
            negative_list = []
            for news in behavior:
                if len(news.split('-')) == 2:
                    newsid, news_label = news.split('-')
                    if news_label == "1":
                        positive_list.append(newsid)
                    else:
                        negative_list.append(newsid)
            for pos_news in positive_list:
                user_id.append(userid+ "_train")
                if len(negative_list) >= negative_num:
                    neg_news = random.sample(negative_list, negative_num)
                else:
                    neg_news = negative_list
                    for i in range(negative_num - len(negative_list)):
                        neg_news.append("B0")
                all_news = neg_news
                all_news.append(pos_news)
                news_id.append(all_news)
                label.append([])
                for i in range(negative_num):
                    label[-1].append(0)
                label[-1].append(1)

    train_data['item1'] = user_id
    train_data['item2'] = news_id
    train_data['label'] = label

    dev_data = {}
    session_id = []
    user_id = []
    news_id = []
    label = []
    fp_dev = open(config['books']['valid_behavior'], 'r', encoding='utf-8')
    next(fp_dev)
    for index, line in enumerate(fp_dev):
        if len(line.strip().split('\t')) == 3:
            userid, history, behavior = line.strip().split('\t')
            behavior = behavior.split(' ')
            for news in behavior:
                if len(news.split('-')) == 1:
                    positive_list.append(news)
                else:
                    newsid, news_label = news.split('-')
                    session_id.append(index)
                    user_id.append(userid+ "_dev")
                    if news_label == "1":
                        news_id.append(newsid)
                        label.append(1.0)
                    else:
                        news_id.append(newsid)
                        label.append(0.0)

    dev_data['item1'] = user_id
    dev_data['session_id'] = session_id
    dev_data['item2'] = news_id
    dev_data['label'] = label

    return train_data, dev_data


def build_user_history_books(config):
    user_history_dict = {}
    fp_train_behavior = open(config['books']['train_behavior'], 'r', encoding='utf-8')
    next(fp_train_behavior)
    for line in fp_train_behavior:
        if len(line.strip().split('\t')) == 3:
            user_id, history, behavior = line.strip().split('\t')
            if len(history.split(' ')) >= config['model']['user_his_num']:
                user_history_dict[user_id+"_train"] = history.split(' ')[:config['model']['user_his_num']]
            else:
                user_history_dict[user_id + "_train"] = history.split(' ')
                for i in range(config['model']['user_his_num']-len(history.split(' '))):
                    user_history_dict[user_id + "_train"].append("B0")
                if user_history_dict[user_id + "_train"][0] == '':
                    user_history_dict[user_id + "_train"][0] = 'B0'

    fp_dev_behavior = open(config['books']['valid_behavior'], 'r', encoding='utf-8')
    next(fp_dev_behavior)
    for line in fp_dev_behavior:
        if len(line.strip().split('\t')) == 3:
            user_id, history, behavior = line.strip().split('\t')
            if len(history.split(' ')) >= config['model']['user_his_num']:
                user_history_dict[user_id+"_dev"] = history.split(' ')[:config['model']['user_his_num']]
            else:
                user_history_dict[user_id + "_dev"] = history.split(' ')
                for i in range(config['model']['user_his_num']-len(history.split(' '))):
                    user_history_dict[user_id + "_dev"].append("B0")
                if user_history_dict[user_id + "_dev"][0] == '':
                    user_history_dict[user_id + "_dev"][0] = 'B0'
    return user_history_dict

def build_news_features_mind_books(config, embedding_folder = None):
    if embedding_folder != None:
        a,b,c,d = load_from_pickle(embedding_folder)
        return a,b,c,d
    else:
        print('building books features')
        entity2id_dict = {}
        fp_entity2id_entity = open(config['books']['entity_index'], 'r', encoding='utf-8')
        entity_num = len(fp_entity2id_entity.readlines())+1
        fp_entity2id = open(config['books']['entity_index'], 'r', encoding='utf-8')
        for line in fp_entity2id.readlines():
            entity, entityid = line.strip().split('\t')
            entity2id_dict[entity] = int(entityid) + 1

        news_features = {}
        news_feature_dict = {}
        fp_train_books = open(config['books']['train_books'], 'r', encoding='utf-8')
        df = pd.read_csv(fp_train_books, sep=',')
        df.replace(np.nan,0)
        for index, row in df.iterrows():
            booksid = row['post_id']
            title = row['title']
            abstract = str(row['description'])
            entity_info_title = eval(row['entity_info_title'])
            entity_info_abstract = []
            news_feature_dict[booksid] = (title + " " + abstract, entity_info_title, entity_info_abstract)
        # for line in fp_train_news:
        #     newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        #     news_feature_dict[newsid] = (title+" "+abstract, entity_info_title, entity_info_abstract)

        # entityid, entity_freq, entity_position, entity_type
        fp_dev_books = open(config['books']['valid_books'], 'r', encoding='utf-8')
        df = pd.read_csv(fp_dev_books, sep=',')
        df.replace(np.nan, 0)
        for index, row in df.iterrows():
            booksid = row['post_id']
            title = row['title']
            abstract = str(row['description'])
            entity_info_title = eval(row['entity_info_title'])
            entity_info_abstract = []
            news_feature_dict[booksid] = (title + " " + abstract, entity_info_title, entity_info_abstract)

        # fp_dev_books = open(config['books']['valid_books'], 'r', encoding='utf-8')
        # next(fp_dev_books)
        # for line in fp_dev_books:
        #     print(line)
        #     newsid, title, abstract, _ ,_ ,_ , entity_info_title, entity_info_abstract = line.strip().split(',')
        #     news_feature_dict[newsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)


        #deal with doc feature
        entity_type_dict = {}
        entity_type_index = 1
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        for i, news in enumerate(news_feature_dict):
            if i >= 5000:
                sentence_embedding = model.encode(news_feature_dict[news][0])
                news_entity_feature_list = []
                title_entity_json = json.dumps(news_feature_dict[news][1])

                news_entity_feature = {}
                for item in json.loads(title_entity_json):
                    item = json.loads(item)
                    if item['Type'] not in entity_type_dict:
                        entity_type_dict[item['Type']] = entity_type_index
                        entity_type_index = entity_type_index + 1

                    news_entity_feature[item['WikidataId']] = (len([item['OccurrenceOffsets']]), 1, entity_type_dict[item['Type']]) #entity_freq, entity_position, entity_type

                for entity in news_entity_feature:
                    if entity in entity2id_dict:
                        news_entity_feature_list.append([entity2id_dict[entity], news_entity_feature[entity][0], news_entity_feature[entity][1], news_entity_feature[entity][2]])
                news_entity_feature_list.append([0, 0, 0, 0])
                if len(news_entity_feature_list) > config['model']['news_entity_num']:
                    news_entity_feature_list = news_entity_feature_list[:config['model']['news_entity_num']]
                else:
                    for i in range(len(news_entity_feature_list), config['model']['news_entity_num']):
                        news_entity_feature_list.append([0, 0, 0, 0])
                news_feature_list_ins = [[],[],[],[],[]]
                for i in range(len(news_entity_feature_list)):
                    for j in range(4):
                        news_feature_list_ins[j].append(news_entity_feature_list[i][j])
                news_feature_list_ins[4] = sentence_embedding
                news_features[news] = news_feature_list_ins
        news_features["B0"] = [[],[],[],[],[]]
        for i in range(config['model']['news_entity_num']):
            for j in range(4):
                news_features["B0"][j].append(0)
        news_features["B0"][4] = np.zeros(config['model']['document_embedding_dim'])
        return news_features, 100, 10, 100


def construct_adj_mind_books(config):  # graph is triple
    print('constructing adjacency matrix ...')
    graph_file_fp = open(config['books']['knowledge_graph'], 'r', encoding='utf-8')
    graph = []
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        graph.append([int(linesplit[0])+1, int(linesplit[1])+1, int(linesplit[2])+1])
    kg = {}
    for triple in graph:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    fp_entity2id = open(config['books']['entity_index'], 'r', encoding='utf-8')
    entity_num = len(fp_entity2id.readlines())+1
    print(entity_num)
    entity_adj = []
    relation_adj = []
    for i in range(entity_num+1):
        entity_adj.append([])
        relation_adj.append([])
    for i in range(config['model']['entity_neighbor_num']):
        entity_adj[0].append(0)
        relation_adj[0].append(0)
    for key in kg.keys():
        for index in range(config['model']['entity_neighbor_num']):
            i = random.randint(0,len(kg[key])-1)
            entity_adj[int(key)].append(int(kg[key][i][0]))
            relation_adj[int(key)].append(int(kg[key][i][1]))

    return entity_adj, relation_adj


# Load the emdedding of the entities in entity2id_dict, append them to entity_embedding and
# update entity2embedding_dict
def construct_embedding_mind_books(config):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(config['books']['entity_embedding'], 'r', encoding='utf-8')
    fp_relation_embedding = open(config['books']['relation_embedding'], 'r', encoding='utf-8')
    zero_array = np.zeros(config['model']['entity_embedding_dim'])
    entity_embedding.append(zero_array)
    relation_embedding.append(zero_array)
    for line in fp_entity_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    for line in fp_relation_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        relation_embedding.append(linesplit)
    return torch.FloatTensor(np.array(entity_embedding)), torch.FloatTensor(np.array(relation_embedding))

def build_vert_data_books(config):
    random.seed(2023)
    vert_label_dict = {}
    label_index = 0
    all_books_data = []
    vert_train = {}
    vert_dev = {}
    item1_list_train = []
    item2_list_train = []
    label_list_train = []
    item1_list_dev = []
    item2_list_dev = []
    label_list_dev = []
    fp_train_books = open(config['books']['train_books'], 'r', encoding='utf-8')
    df = pd.read_csv(fp_train_books, sep='\t')
    df.replace(np.nan, 0)
    for index, row in df.iterrows():
        booksid = row['post_id']
        vert = row['industries']
        if vert not in vert_label_dict:
            vert_label_dict[vert] = label_index
            label_index = label_index + 1
        all_books_data.append((booksid, vert_label_dict[vert]))
    for i in range(len(all_books_data)):
        if random.random() < 0.8:
            item1_list_train.append("U0")
            item2_list_train.append(all_books_data[i][0])
            label_list_train.append(all_books_data[i][1])
        else:
            item1_list_dev.append("U0")
            item2_list_dev.append(all_books_data[i][0])
            label_list_dev.append(all_books_data[i][1])
    vert_train['item1'] = item1_list_train
    vert_train['item2'] = item2_list_train
    vert_train['label'] = label_list_train
    vert_dev['item1'] = item1_list_dev
    vert_dev['item2'] = item2_list_dev
    vert_dev['label'] = label_list_dev

    return vert_train, vert_dev


def build_pop_data_books(config):
    fp_train = open(config['books']['train_behavior'], 'r', encoding='utf-8')
    next(fp_train)
    news_imp_dict = {}
    pop_train = {}
    pop_test = {}
    next(fp_train)
    for line in fp_train:
        userid, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [1, 1]
                else:
                    news_imp_dict[newsid][0] = news_imp_dict[newsid][0] + 1
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
            else:
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [0, 1]
                else:
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
    return pop_train, pop_test


def build_item2item_data_books(config):
    fp_train = open(config['books']['train_behavior'], 'r', encoding='utf-8')
    next(fp_train)
    item2item_train = {}
    item2item_test = {}
    item1_train = []
    item2_train = []
    label_train = []
    item1_dev = []
    item2_dev = []
    label_dev = []
    user_history_dict = {}
    books_click_dict = {}
    doc_doc_dict = {}
    all_books_set = set()
    for line in fp_train:
        userid, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        if userid not in user_history_dict:
            user_history_dict[userid] = set()
        for news in behavior:
            newsid, news_label = news.split('-')
            all_books_set.add(newsid)
            if news_label == "1":
                user_history_dict[userid].add(newsid)
                if newsid not in books_click_dict:
                    books_click_dict[newsid] = 1
                else:
                    books_click_dict[newsid] = books_click_dict[newsid] + 1
        news = history.split(' ')
        for newsid in news:
            user_history_dict[userid].add(newsid)
            if newsid not in books_click_dict:
                books_click_dict[newsid] = 1
            else:
                books_click_dict[newsid] = books_click_dict[newsid] + 1
    for user in user_history_dict:
        list_user_his = list(user_history_dict[user])
        for i in range(len(list_user_his) - 1):
            for j in range(i + 1, len(list_user_his)):
                doc1 = list_user_his[i]
                doc2 = list_user_his[j]
                if doc1 != doc2:
                    if (doc1, doc2) not in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = 1
                    elif (doc1, doc2) in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = doc_doc_dict[(doc1, doc2)] + 1
                    elif (doc2, doc1) in doc_doc_dict and (doc1, doc2) not in doc_doc_dict:
                        doc_doc_dict[(doc2, doc1)] = doc_doc_dict[(doc2, doc1)] + 1
    weight_doc_doc_dict = {}
    for item in doc_doc_dict:
        if item[0] in books_click_dict and item[1] in books_click_dict:
            weight_doc_doc_dict[item] = doc_doc_dict[item] / math.sqrt(
                books_click_dict[item[0]] * books_click_dict[item[1]])

    THRED_CLICK_TIME = 10
    freq_news_set = set()
    for news in books_click_dict:
        if books_click_dict[news] > THRED_CLICK_TIME:
            freq_news_set.add(news)
    news_pair_thred_w_dict = {}  # {(new1, news2): click_weight}
    for item in weight_doc_doc_dict:
        if item[0] in freq_news_set and item[1] in freq_news_set:
            news_pair_thred_w_dict[item] = weight_doc_doc_dict[item]

    news_positive_pairs = []
    for item in news_pair_thred_w_dict:
        if news_pair_thred_w_dict[item] > 0.05:
            news_positive_pairs.append(item)

    for item in news_positive_pairs:
        random_num = random.random()
        if random_num < 0.8:
            item1_train.append(item[0])
            item2_train.append(item[1])
            label_train.append(1)
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                item1_train.append(item[0])
                item2_train.append(negative)
                label_train.append(0)
        else:
            item1_dev.append(item[0])
            item2_dev.append(item[1])
            label_dev.append(1)
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                item1_train.append(item[0])
                item2_train.append(negative)
                label_dev.append(0)
    item2item_train["item1"] = item1_train
    item2item_train["item2"] = item2_train
    item2item_train["label"] = label_train
    item2item_test["item1"] = item1_dev
    item2item_test["item2"] = item2_dev
    item2item_test["label"] = label_dev
    return item2item_train, item2item_test


def load_data_mind_books(config, embedding_folder=None):


    entity_adj, relation_adj = construct_adj_mind_books(config)
    
    
    if os.path.isfile(embedding_folder):
        news_feature, max_entity_freq, max_entity_pos, max_entity_type = load_from_pickle(embedding_folder)
    else:
        news_feature, max_entity_freq, max_entity_pos, max_entity_type = build_news_features_mind_books(config)
        save_compressed_pickle(embedding_folder, [news_feature, max_entity_freq, max_entity_pos, max_entity_type]) 

        
    
    user_history = build_user_history_books(config)

    entity_embedding, relation_embedding = construct_embedding_mind_books(config)

    if config['trainer']['training_type'] == "multi-task":
        train_data, dev_data = get_user2item_data_books(config)
        vert_train, vert_test = build_vert_data_books(config)
        pop_train, pop_test = build_pop_data_books(config)
        item2item_train, item2item_test = build_item2item_data_books(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data, vert_train, vert_test, pop_train, pop_test, item2item_train, item2item_test
    elif config['trainer']['task'] == "user2item":
        train_data, dev_data = get_user2item_data_books(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data
    elif config['trainer']['task'] == "item2item":
        item2item_train, item2item_test = build_item2item_data_books(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, item2item_train, item2item_test
    elif config['trainer']['task'] == "vert_classify":
        vert_train, vert_test = build_vert_data_books(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, vert_train, vert_test
    elif config['trainer']['task'] == "pop_predict":
        pop_train, pop_test = build_pop_data_books(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, pop_train, pop_test
    else:
        print("task error, please check config")

def load_compressed_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_compressed_pickle(filename, obj):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)

