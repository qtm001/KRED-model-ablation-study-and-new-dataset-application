import requests
import nltk
import spacy
import re
import ast
import json
# from nltk.tokenize import sent_tokenize, word_tokenize
from heapq import nlargest
import pandas as pd

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')




def get_ner(text):
    sent = nltk.word_tokenize(text)
    entities = nltk.ne_chunk(nltk.pos_tag(sent))
    return entities

def get_wikidata_id(entity):
    query = entity.split()
    query = "+".join(query)
    # print(query)
    response = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&limit=1&search={query}")
    if response.status_code == 200:
        # print(response.json())
        response_json = response.json()
        if "search" in response_json and response_json["search"]:
            return response_json["search"][0]["id"]
        else:
            return None
    return None


def long_wikiID_extract(text, wikidata_ids):
    try:

        t = nlp(text)
        list_dict = []
        keys = ["Label", "WikidataId", "OccurrenceOffsets", "Type"]
        wiki_h = []
        entities = get_ner(text)
        for entity in entities.subtrees():
            wikidata_dict = {}
            # print(entity.label())
            if entity.label() != 'S':
                # print(entity.label())
                entity_name = " ".join([i[0] for i in entity.leaves()])
                # print(entity_name)
                wikidata_id = get_wikidata_id(entity_name)
                # print(entity_name)
                # print(wikidata_id)
                if wikidata_id and wikidata_id not in wiki_h:
                    if wikidata_id not in wikidata_ids:
                        wikidata_ids.append(wikidata_id)
                    wiki_h.append(wikidata_id)
                    wikidata_dict["Label"] = entity_name
                    wikidata_dict["WikidataId"] = wikidata_id
                    wikidata_dict["Type"] = entity.label()[0]
                    for token in t:
                        if str(token) in entity_name:
                            # print(token)
                            wikidata_dict["OccurrenceOffsets"] = token.idx
                            break
                    # for ent in entity:
                    #     if ent in wikidata_dict["Label"]:
                    #         wikidata_dict["Type"] = ent.label_[0]
                    if wikidata_dict and all(key in wikidata_dict for key in keys):
                        list_dict.append(json.dumps(wikidata_dict, sort_keys=True))
        # print(wikidata_ids)
        del wiki_h
        return list_dict
    except ValueError:
        return {}

def short_wikiID_extract(text, wikidata_ids, type):
    try:
        # print(text)
        list_dict = []
        keys = ["Label", "WikidataId", "OccurrenceOffsets", "Type"]
        wiki_h = []
        punt = ['[', ']', "'", "'"]
        for p in punt:
            text = text.replace(p, '')
            text_list = re.split(',|&', text)
        for i, t in enumerate(text_list):
            # print(t)
            wikidata_dict = {}
            wikidata_id = get_wikidata_id(t)
            if wikidata_id and wikidata_id not in wiki_h:
                    if wikidata_id not in wikidata_ids:
                        wikidata_ids.append(wikidata_id)
                    wiki_h.append(wikidata_id)
                    wikidata_dict["Label"] = t
                    wikidata_dict["WikidataId"] = wikidata_id
                    wikidata_dict["OccurrenceOffsets"] = i
                    if type == 'author':
                      wikidata_dict['Type'] = 'P'
                    elif type == 'cate':
                      wikidata_dict['Type'] = 'W'
                    # print(wikidata_dict)
                    if wikidata_dict and all(key in wikidata_dict for key in keys):
                      list_dict.append(json.dumps(wikidata_dict))

        del wiki_h
        return list_dict
    except ValueError:
        return {}
        
def get_book_wikid(book_info, wikidata_ids):            # get_extract
    

    title = book_info['Title']
    description = book_info['description']
    author = book_info['authors']
    categories = book_info['categories']

    title_id = long_wikiID_extract(title, wikidata_ids)
    description_id = long_wikiID_extract(description, wikidata_ids)
    author_id = short_wikiID_extract(author, wikidata_ids, 'author')
    categories_id = short_wikiID_extract(categories, wikidata_ids, 'cate')
    # print(author_id)
    info_final = categories_id + author_id + title_id + description_id
    # print(info_final)
    return info_final


