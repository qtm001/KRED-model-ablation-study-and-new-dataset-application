import argparse
from train_test import *
from utils.util_books import *
import pickle
from parse_config import ConfigParser
import gzip


import os

def main(config):
    # data = load_data_mind_books(config, 'books_embeddings_new.pkl')
    task = config['trainer']['task']
    # save_compressed_pickle(f"./data_mind_small_books_{task}_new.pkl", data)
    data_type = config['trainer']['data_type']
    with gzip.open(f"./data/data_mind_small_{data_type}_{task}_new.pkl", 'rb') as f:
        data = pickle.load(f)
    if config['trainer']['training_type'] == "single_task":
        single_task_training(config, data)
    else:
        multi_task_training(config, data)

    test_data = data[-1]
    testing(test_data, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KRED')

    parser.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
                      
    config = ConfigParser.from_args(parser)
    main(config)