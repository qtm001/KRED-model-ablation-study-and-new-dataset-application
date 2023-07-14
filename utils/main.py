from books_build import *
from user_build import *
import pandas as pd
import argparse 
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--save_dir_name', type=str, default='Amazon_Books', help='Directory name in which preprocessed data will be stored')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode. Will only save a small subset of the data')
    # Generate args
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    model = SentenceTransformer('all-mpnet-base-v2')
    args = parse_args()
    data_path = args.data_dir
    save_path = os.path.join(args.data_dir, args.save_dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(os.path.join(data_path, 'Amazon_Books/book_data_clean.csv')): 
        print("Don't find processed dataset, building...")
        # remove useless columns, shuffle the rows, remove the rows with nan value and take first 10k rows.
        df = pd.read_csv(os.path.join(data_path, 'books_data.csv'))
        post_id = []
        for i ,b in enumerate(df.iterrows()):
          post_id.append('B' + str(i+1))
        df.insert(0, 'post_id', post_id)
        df = df.drop(columns=['previewLink', 'publisher', 'publishedDate', 'infoLink', 'ratingsCount']).sample(frac = 1)
        df_clean = df[~pd.isna(df['description']) & ~pd.isna(df['authors']) & ~pd.isna(df['categories'])].iloc[:10001]
        df_clean.to_csv(os.path.join(data_path, 'Amazon_Books/book_data_clean.csv'), index=None)
        print(df_clean.head())
    
    if args.debug_mode == True:
        print("Debug mode activated")
        book_file_name = 'small_amazon_books.csv'
        entity2vecd100_path = 'small_entity2vecd100_books.vec'
        entity2id_path = 'small_entity2id_books.txt'
    else:
        book_file_name = 'amazon_books.csv'
        entity2vecd100_path = 'entity2vecd100_books.vec'
        entity2id_path = 'entity2id_books.txt'
    
    df_books = pd.read_csv(os.path.join(data_path, 'Amazon_Books/book_data_clean.csv'))
    records = df_books.to_dict(orient='records')
    books_info = ['post_id', 'title', 'description', 'authors', 'image', 'categories', 'entity_info_title', 'entity_info_abstract']
    # Construct books dataset for KRED
    if args.debug_mode == True or not os.path.exists(os.path.join(data_path, f'Amazon_Books/{book_file_name}')):
        print("Start building entity to ids dataset...")
        wikidata_ids = build_books_dataset(records, books_info, save_path, debug_mode = args.debug_mode)
    else:
      with open(os.path.join(data_path, "Amazon_Books/wikidata_ids.txt"), 'r') as txt:
        wikidata_ids = txt.read().splitlines()
        
    df_books = pd.read_csv(os.path.join(data_path, f'Amazon_Books/{book_file_name}'))
    
    


    # Extract sentence embedding of entities
    if args.debug_mode == True or not os.path.exists(os.path.join(data_path, f'Amazon_Books/{entity2vecd100_path}')):
        
        print('Cannot find entity2vec file, building...')
        
        
        if torch.cuda.is_available():
            model.to('cuda:0')
        else:
            model.to('cpu')
        
        # Extract entity embeddings and descriptions
        entity_embeddings, entity_descriptions = extract_entity_embeddings_descriptions(wikidata_ids,model)
        
        # Reduce the embeddings size
        reduced_entity_embeddings = reduce_embeddings_size(entity_embeddings)
        #print(reduced_entity_embeddings.shape)
    
        # save entity_description_embeddings to file 
        np.savetxt(os.path.join(save_path, entity2vecd100_path), reduced_entity_embeddings, fmt='%.6f', delimiter='\t')
        
        # Update books dataframe removing not embedded entities
        df_books = update_books_dataset(df_books,entity_descriptions)
    
        # List of the embedded entities (entities who have descriptions)
        entities = list(entity_descriptions.keys())
    
        # Create entity-to-id dictionary
        entity2id_dict = create_x2id_dict(entities)

        # Save entity2id file
            
        with open(os.path.join(save_path, entity2id_path), 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            print('Writing entity2id file...')
            for key, value in tqdm(entity2id_dict.items()):
                writer.writerow([key, value])
    else:    
        # read from entity2id file
        with open(os.path.join(data_path, f'Amazon_Books/{entity2id_path}')) as file:
            wikiid_list = [line.rstrip().split('\t') for line in file]
        entity2id_dict = dict(wikiid_list)

        df_books = update_books_dataset(df_books,entity2id_dict)
        entities = list(entity2id_dict.keys())
    

    #Extract sentence embeddings of relationship
    if args.debug_mode:
        relationship2vecd100_path = 'small_relationship2vecd100_books.vec'
        relationship2id_path = 'small_relationship2id_books.txt'
        triple2id_path = 'small_triple2id_books.txt'
    else:
        relationship2vecd100_path = 'relationship2vecd100_books.vec'
        relationship2id_path = 'relationship2id_books.txt'
        triple2id_path = 'triple2id_books.txt'

    if args.debug_mode == True or not os.path.exists(os.path.join(data_path, f'Amazon_Books/{relationship2vecd100_path}')):

        print('Extract relationships...')

        # Extract relationships between entities
        relationships = extract_relationships(entities)

        # Extract unique relations
        relations = list(set([rel[1] for rel in relationships]))

        # Extract relation embeddings and relations
        relations_embeddings, relations_descriptions = extract_relation_embeddings_descriptions(relations,model)
        print(np.asarray(relations_embeddings).shape)

        # Reduce the embeddings size
        reduced_embeddings_relations = reduce_embeddings_size(relations_embeddings)
        

        # Save relation embedding file

        np.savetxt(os.path.join(save_path, relationship2vecd100_path), reduced_embeddings_relations, fmt='%.6f', delimiter='\t')

        # Create relation to id
        relation2id_dict = create_x2id_dict(list(relations_descriptions.keys()))

        # Save relation2id file
        with open(os.path.join(save_path, relationship2id_path), 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for key, value in relation2id_dict.items():
                writer.writerow([key, value])

        # Build head entity - relation - tail entity triples.

        triple2id = get_triple2id(relationships,entity2id_dict,relation2id_dict)

        # Save triple to id file
        with open(os.path.join(save_path, triple2id_path), 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for tuple_ in triple2id:
                writer.writerow(tuple_)

    # Update books extracting entities per book and removing unuseful columns
    df_books = modify_df_books(df_books) # (post_id, entities) post_id is the book id, entities are the wiki_id
    df_books = df_books[df_books["entities"].map(len) > 0] # Remove the books without entities

    # From book ratings, extract wikidata_ids
    if not os.path.exists(os.path.join(data_path, 'Amazon_Books/user_data_clean.csv')): 
        user_info = ["User", "Entities"]
        df = pd.read_csv(os.path.join(data_path, 'Books_rating.csv'))
        df = df[(df['review/score']>= 4)].sample(frac = 1)
        df = df[~pd.isna(df['review/text']) & ~pd.isna(df['Title'])]
        df = df[['User_id','Title', 'review/text']]
        cleaned_df = df.iloc[:10000]
        cleaned_df.to_csv(os.path.join(data_path, 'user_data_clean.csv'))

    df =  pd.read_csv(os.path.join(data_path, 'Amazon_Books/user_data_clean.csv'))

    # Extract user data
    if args.debug_mode:
        user_entities_filename = 'small_user_entities_bookreview.csv'
        behavior_filename = 'small_behavior_books.csv'
    else:
        user_entities_filename = 'user_entities_bookreview.csv'
        behavior_filename = 'behavior_books.csv'
        
    if args.debug_mode == True or not os.path.exists(os.path.join(data_path, f'Amazon_Books/{user_entities_filename}')):
        user_info = ["User", "Entities"]
        fix_survey(df, user_info, save_path ,args.debug_mode)
        
    df_users = pd.read_csv(os.path.join(data_path, f'Amazon_Books/{user_entities_filename}'))
    df_users = df_users[df_users["Entities"].map(len) > 2]  # Remove the books without entities

    if args.debug_mode == True or not os.path.exists(os.path.join(data_path, f'Amazon_Books/{behavior_filename}')):
        
        # Generate history and behaviors gpr
        df_behavior = generate_history_behaviors(df_users, df_books)
        
        print('Create behavior files.')

        df_behavior.to_csv(os.path.join(save_path, behavior_filename), sep='\t')

        split_behaviors(save_path, os.path.join(data_path, f'Amazon_Books/{behavior_filename}'), args.debug_mode)
        
    print('Extraction completed! Proceeding with creating pkl files')
    data = load_data_mind_books(config, 'books_embeddings_new.pkl')
    task = config['trainer']['task']
    save_compressed_pickle(f"./data_mind_small_book_{task}_new.pkl", data)