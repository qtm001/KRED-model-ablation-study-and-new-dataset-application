import os
import numpy as np
import ast
import csv
import pandas as pd
from extract_wikidata import *
from tqdm import tqdm

'''
This file is for user data extraction, with the structure:
    User: String, the id of users, start with 'U'.
    Title: String, the title of book which the user has reviewed.
    review/text: String, the review from this user.

'''


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

def modify_df_books(df_books):
    new_col = []
    print('creating (post_id, entities) pairs')
    for r in tqdm(df_books['entity_info_title']):
        present = []
        # entities = ast.literal_eval(r)
        # for entity in entities:
        for entity in r:
            present.append(entity['WikidataId'])
        new_col.append(present)
    df_books['entities'] = new_col
    df_books = df_books[['post_id','entities']]
    return df_books

def fix_survey(df, user_info, data_path, debug_mode):
    if debug_mode:
        user_entities_filename = 'small_user_entities_bookreview.csv'
    else:
        user_entities_filename = 'user_entities_bookreview.csv'
    with open(os.path.join(data_path, user_entities_filename), 'w', encoding='utf-8',
             newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=user_info)
        writer.writeheader()
        dbook = {}

        if debug_mode:
            df = df.iloc[:101]
        
        print('creating (user, entities) pairs')
        for i in tqdm(range(len(df.index))):  # truncated to 10000
            wikidata_ids = []
            dbook["User"] = "U" + str(i)
            title = long_wikiID_extract(df.Title.iloc[i], wikidata_ids) # we get the wiki_ids of the language
            review = long_wikiID_extract(df['review/text'].iloc[i], wikidata_ids)
            # Save the couple U0, ['Q123', 'Q456', ... ]
            #print(title, review)
            dbook["Entities"] = wikidata_ids # save all the wiki_ids in book['Entities']
            # Write the file in order to save them once for all
            writer.writerow(dbook)

def generate_history_behaviors(df_users,df_books): 
    p = 0.75
    clicks_col = []
    behaviors_col = []
    print('creating behavior for each user')
    for user_entities in tqdm(df_users['Entities']):
        feasible_books = [] 
        clicks = []
        behaviors = []
        for index, row in df_books.iterrows():
            n_occurrences = 0
            for entity in ast.literal_eval(user_entities): #split the string into a list
                if entity in row['entities']:
                    n_occurrences += 1
            if n_occurrences > 0:
                feasible_books.append((row['post_id'],n_occurrences)) # possible connection between book and user

        # for each user, we list all the possible books with their corresponding occurrence
        probabilities = softmax([i[1] for i in feasible_books])

        if len(feasible_books) > 0:
            if set([i[0] for i in feasible_books]) == set(clicks+[j.split('-')[0] for j in behaviors]):
                break

            while True: 
                new_click = np.random.choice([i[0] for i in feasible_books], p=probabilities)
                if new_click not in clicks:
                    clicks.append(new_click)
                                
                if len(clicks) >= 3:
                    if np.random.choice([0,1], p=[p,1-p]):
                        break

                if set([i[0] for i in feasible_books]) == set(clicks+[j.split('-')[0] for j in behaviors]):
                    break

            while len(behaviors) < 20:
                new_beh = np.random.choice(list(df_books['post_id'].values))
                if new_beh not in clicks:
                    if new_beh in [i[0] for i in feasible_books]:
                        behaviors.append(new_beh+'-1')
                    else:
                        behaviors.append(new_beh+'-0')
        # behavior:[clicks on book based on the entities, random choice of the all possible books]
        #                                                 '-1' if the book is in the feasible books, '-0' otherwise

        clicks_col.append(clicks) 
        behaviors_col.append(behaviors) 
    
    clicks_str = [' '.join(r) for r in clicks_col]
    behaviors_str = [' '.join(r) for r in behaviors_col]
    df_users['Histories'] = clicks_str
    df_users['Behaviors'] = behaviors_str

    df_users = df_users[['User','Histories','Behaviors']]
    return df_users

def split_behaviors(save_path, filename, debug_mode):
    if debug_mode:
        train_name = 'small_behaviors_train_books.tsv'
        val_name = 'small_behaviors_valid_books.tsv'
    else:
        train_name = 'behaviors_train_books.tsv'
        val_name = 'behaviors_valid_books.tsv'
    with open(filename, 'r', encoding='utf-8') as csv_file:
        # Load the data from the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file, sep='\t')
        df = df.drop(df.columns[:1], axis=1)

        # Calculate the number of rows in the DataFrame
        n = df.shape[0]

        # Generate a random permutation of the indices
        idx = np.random.permutation(n)

        # Split the indices into training and validation sets
        split = int(0.8 * n)
        train_idx = idx[:split]
        val_idx = idx[split:]

        # Split the DataFrame into training and validation sets based on the indices
        train_df = df.iloc[train_idx, :]
        val_df = df.iloc[val_idx, :]

        # Save the training and validation sets as CSV files
        train_df.to_csv(os.path.join(save_path, train_name), sep='\t', index=False)
        val_df.to_csv(os.path.join(save_path, val_name), sep='\t', index=False)
