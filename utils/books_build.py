import ast
from sklearn.decomposition import PCA
import requests
from operator import itemgetter
from extract_wikidata import get_book_wikid 
import csv
import os
from tqdm import tqdm

'''
This file is for buliding books dataset, with structure:
    post_id: String, the id of books, start with 'B'
    title: String, the titile of books.
    description: String, general description of books.
    authors: String, the author of book
    image: String, the link to an images of books.
    categories: String, the categories of books.
    entity_info_title: List, includes a list of entities of the books, for each entity, with structure:
        Label: The name of entity.
        WikidataId: The wikidata id of entity.
        OccurrenceOffsets: The position of the entity in sentence.
        Type: the type of entity, based on NER model.
    entity_info_abstract: A empty list, we will not use it.

'''


def build_books_dataset(records, books_info, datapath_src, debug_mode):
    dbook = {}
    wikidata_ids = []
    if debug_mode == True:
        book_file_name = 'small_amazon_books.csv'
        wiki_data_name = 'small_wikidata_ids.txt'
        records = records[:120]
    else:
        wiki_data_name = 'wikidata_ids.txt'
        book_file_name = 'amazon_books.csv'
        
    with open(os.path.join(datapath_src, book_file_name), 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = books_info)
        # this is just for the header
        dbook["post_id"] = 'post_id'
        dbook["title"] = 'Title'
        dbook["description"] = 'description'
        dbook["authors"] = 'authors'
        dbook["image"] = 'image'
        dbook["categories"] = 'categories'
        dbook["entity_info_title"] = "entity_info_title"
        dbook["entity_info_abstract"] = "entity_info_abstract" 
        writer.writerow(dbook)

        for book in tqdm(records):
            dbook["post_id"] = book['post_id']
            dbook["title"] = book['Title']
            dbook["description"] = book['description']
            dbook["authors"] = book['authors']
            dbook["image"] = book['image']
            dbook["categories"] = book['categories']
            dbook["entity_info_title"] = get_book_wikid(book, wikidata_ids)
            dbook["entity_info_abstract"] = []
            
            writer.writerow(dbook)

    # Creation file with all uniques wikidata_ids
    print("Start building wiki_ids file...")
    file = open(os.path.join(datapath_src, wiki_data_name), 'w', encoding='utf-8')
    for wikidata in tqdm(wikidata_ids):
        file.write(wikidata + "\n")
    file.close()
    return wikidata_ids

# Function to get entity description
def get_description(id):
    query = f"""
    SELECT ?Label ?Description
    WHERE 
    {{
      wd:{id} rdfs:label ?Label .
      FILTER (LANG(?Label) = "en").
      OPTIONAL {{ wd:{id} schema:description ?Description . FILTER (LANG(?Description) = "en") }}
    }}
    """
    max_tries = 100
    for i in range(max_tries):
      try:
        response = requests.get("https://query.wikidata.org/sparql", params={'query': query, 'format': 'json'})
        response_json = response.json()
        label = response_json['results']['bindings'][0]['Label']['value']
        description = response_json['results']['bindings'][0].get('Description', {}).get('value', '')
        description = label + ' ' + description
        return description
      except:
        pass
    return None

# Function to extract the embeddings for a given entity
def extract_embeddings_entities(entity_id, model):
    entity_description = get_description(entity_id)
    if entity_description == None:
        return None, None
    sentence_embeddings = model.encode(entity_description)
    return entity_description, sentence_embeddings

# Extract entity embeddings and descriptions
def extract_entity_embeddings_descriptions(entity_ids,model):
    entities_embeddings = []
    entities_descriptions = {}
    print('obtaining description and embedding from wiki_ids...')
    for entity_id in tqdm(entity_ids):
        entity_description, sentence_embeddings = extract_embeddings_entities(entity_id, model)
        if not entity_description is None:
            entities_embeddings.append(sentence_embeddings)
            entities_descriptions[entity_id] = entity_description
    return entities_embeddings, entities_descriptions
    
# Function to reduce the size of the embeddings
def reduce_embeddings_size(embeddings):
    pca = PCA(n_components=100)
    return pca.fit_transform(embeddings)

def update_books_dataset(df_books,descripriptions):
    new_col = []
    entities_good = list(descripriptions.keys())
    for r in df_books['entity_info_title']:
        entities = ast.literal_eval(r)
        new_entities = []
        for e in entities:
            entity = ast.literal_eval(e)
            if entity['WikidataId'] in entities_good:
                entity['OccurrenceOffsets'] = [entity['OccurrenceOffsets']]
                new_entities.append(entity)
        new_col.append(new_entities)
    
    df_books['entity_info_title'] = new_col
    return df_books

# Function to create entities/relation to id variables
def create_x2id_dict(elements):
    result = {}
    for e in elements:
        result[e] = len(result)

    return result 
  
# Extract relationships between entities
def extract_relationships(entity_ids):
    # Define the API endpoint for retrieving information about entities
    endpoint = "https://www.wikidata.org/w/api.php"
    # Split entities in chuncks to follow wikidata api constraints
    chunks = [entity_ids[x:x+50] for x in range(0, len(entity_ids), 50)]

    entities = {}

    for c in tqdm(chunks):
    # Define the parameters for the API request
        params = {
            "action": "wbgetentities",
            "ids": "|".join(c),
            "format": "json"
        }

        # Send the API request and retrieve the response
        response = requests.get(endpoint, params=params)

        # Extract the JSON data from the response
        data = response.json()
        #print(data)
        

        # Extract the entity information from the data
        for entity_id, entity in data["entities"].items():
            #print(entity_id)
            try:
                entities[entity_id] = {
                    "label": entity["labels"]["en"]["value"],
                    "description": entity["descriptions"]["en"]["value"],
                    "claims": entity.get("claims", {})
                }
            except:
                entities[entity_id] = {
                    "label": entity["labels"]["en"]["value"],
                    "description": entity["labels"]["en"]["value"],
                    "claims": entity.get("claims", {})
                }
    # print(entities)
    # Define a list to store the relationships between entities
    relationships = []

    # Extract the relationships between entities from the entity information
    print(f'Obtaining relationships between entities...')
    for entity_id, entity in tqdm(entities.items()):
        for property_id, property_values in entity["claims"].items():
            for property_value in property_values:
                if "mainsnak" in property_value and "datavalue" in property_value["mainsnak"]:
                    datavalue = property_value["mainsnak"]["datavalue"]
                    if "value" in datavalue and "id" in datavalue["value"]:
                        try:
                            target_entity_id = datavalue["value"]["id"]
                            if target_entity_id in entity_ids:
                                relationships.append((entity_id, property_id, target_entity_id))
                        except:
                            pass
    
    return relationships

# Function to extract the embedding for a given description
def extract_embeddings_relation(relation_id, model):
    relation_description = get_description(relation_id)
    if relation_description == None:
        return None, None
    sentence_embeddings = model.encode(relation_description)
    return relation_description, sentence_embeddings

# Extract relation embeddings and descriptions
def extract_relation_embeddings_descriptions(relation_ids,model):
    relation_embeddings = []
    relation_descs = {}
    print('extracting relationships from relation wiki_ids')
    for relation_id in tqdm(relation_ids):
        rela_d, emb_ = extract_embeddings_relation(relation_id, model)
        if not emb_ is None:
            relation_embeddings.append(emb_)
            relation_descs[relation_id] =rela_d
    return relation_embeddings, relation_descs

# Function to create entity to id
def get_triple2id(relationships,entity2id_dict,relation2id_dict):
    relationships = list(set(relationships))
    triple2id = [(entity2id_dict[relation[0]], relation2id_dict[relation[1]], entity2id_dict[relation[2]]) for relation in relationships]
    triple2id = sorted(triple2id, key=itemgetter(0, 1, 2))
    return triple2id