import argparse
from train_test import *
from utils.util import *
import pickle
from parse_config import ConfigParser
import gzip


import os

def main(config):
    #data = load_data_mind(config)
    task = config['trainer']['task']
    data_type = config['trainer']['data_type']
    with open(f"./data/data_mind_small_{data_type}_{task}.pkl", 'rb') as f:
        data = pickle.load(f)
        for i,k in enumerate(data[-2]['item2']):
            if k == '':
                del data[-2]['item2'][i]
                del data[-2]['item1'][i]
                del data[-2]['label'][i]
                # print(f'deleted item {i}')
        for i,k in enumerate(data[-1]['item2']):
            if k == '':
                del data[-1]['item2'][i]
                del data[-1]['item1'][i]
                del data[-1]['label'][i]
                # print(f'deleted item {i}')
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